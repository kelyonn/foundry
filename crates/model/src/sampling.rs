//! Sampling: greedy, temperature, top-k, top-p. Seeded and reproducible.
//!
//! # Order of operations, and why it is this one
//!
//! ```text
//! logits → [mask] → temperature → softmax → top-k → top-p → renormalise → draw
//! ```
//!
//! Masking (from `crates/grammar`) happens *before* everything, by setting illegal
//! logits to `-inf`. That matters for more than tidiness: `exp(-inf) == 0`, so a
//! masked token contributes nothing to the softmax denominator, and top-k/top-p
//! then select from a distribution that already excludes it. Filtering after the
//! fact would let an illegal token occupy a slot in the top-k, silently shrinking
//! the candidate set.
//!
//! Temperature is applied to logits, not probabilities, because dividing logits by
//! `T` and then softmaxing is the actual definition — softmaxing first and then
//! reweighting is a different function.

/// xorshift64*. Small, seedable, and adequate for sampling — this is not
/// cryptography, and a reproducible seed matters more than statistical purity.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // A zero seed is a fixed point of xorshift and produces all zeros forever.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`. 24 bits of mantissa is all an `f32` holds anyway.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// `0.0` means greedy — take the argmax, ignore every other setting.
    pub temperature: f32,
    /// Keep only the `k` highest-probability tokens. `None` disables.
    pub top_k: Option<usize>,
    /// Keep the smallest set of tokens whose cumulative probability reaches `p`.
    /// `None` disables.
    pub top_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        SamplingConfig::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

pub struct Sampler {
    pub config: SamplingConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(config: SamplingConfig, seed: u64) -> Sampler {
        Sampler {
            config,
            rng: Rng::new(seed),
        }
    }

    /// Choose a token. `logits` may contain `-inf` entries; they are never chosen.
    ///
    /// Returns `None` only if *every* logit is `-inf`, which means the caller
    /// masked everything — a real error, not a token to invent.
    pub fn sample(&mut self, logits: &[f32]) -> Option<u32> {
        if self.config.is_greedy() {
            return argmax(logits);
        }

        // Temperature, then a numerically-stable softmax. The max subtraction is
        // what makes `-inf` entries land on exactly 0 rather than NaN.
        let inv_t = 1.0 / self.config.temperature;
        let mut max = f32::NEG_INFINITY;
        for &l in logits {
            let scaled = l * inv_t;
            if scaled > max {
                max = scaled;
            }
        }
        if !max.is_finite() {
            return None; // everything masked
        }

        let mut probs: Vec<f32> = logits.iter().map(|&l| (l * inv_t - max).exp()).collect();
        let total: f32 = probs.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        for p in probs.iter_mut() {
            *p /= total;
        }

        // Rank once; both filters are prefix operations on the same ordering.
        let mut order: Vec<u32> = (0..probs.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b as usize]
                .partial_cmp(&probs[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = order.len();
        if let Some(k) = self.config.top_k {
            keep = keep.min(k.max(1));
        }
        if let Some(p) = self.config.top_p {
            // Nucleus: the smallest prefix whose cumulative mass reaches `p`.
            // The token that crosses the threshold is *included* — otherwise a
            // p of 0.9 on a distribution whose top token has 0.95 would keep
            // nothing at all.
            let mut cum = 0.0f32;
            let mut n = 0usize;
            for &id in order.iter().take(keep) {
                cum += probs[id as usize];
                n += 1;
                if cum >= p {
                    break;
                }
            }
            keep = keep.min(n.max(1));
        }

        let kept = &order[..keep];
        let mass: f32 = kept.iter().map(|&i| probs[i as usize]).sum();
        if mass <= 0.0 {
            return argmax(logits);
        }

        let mut target = self.rng.next_f32() * mass;
        for &id in kept {
            target -= probs[id as usize];
            if target <= 0.0 {
                return Some(id);
            }
        }
        // Floating-point shortfall; the last kept token is the right answer.
        kept.last().copied()
    }
}

/// Highest logit, skipping `-inf`. Ties go to the lowest index, matching
/// `torch.argmax`, so greedy decoding is comparable against the reference.
pub fn argmax(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        match best {
            Some((_, b)) if l <= b => {}
            _ => best = Some((i as u32, l)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_largest() {
        let mut s = Sampler::new(SamplingConfig::greedy(), 1);
        assert_eq!(s.sample(&[0.1, 0.9, 0.3]), Some(1));
    }

    #[test]
    fn greedy_breaks_ties_toward_the_lowest_index() {
        let mut s = Sampler::new(SamplingConfig::greedy(), 1);
        assert_eq!(s.sample(&[0.5, 0.5, 0.1]), Some(0));
    }

    #[test]
    fn masked_tokens_are_never_chosen() {
        let logits = vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY];
        let mut greedy = Sampler::new(SamplingConfig::greedy(), 1);
        assert_eq!(greedy.sample(&logits), Some(1));

        let mut hot = Sampler::new(
            SamplingConfig {
                temperature: 2.0,
                top_k: None,
                top_p: None,
            },
            7,
        );
        for _ in 0..200 {
            assert_eq!(hot.sample(&logits), Some(1));
        }
    }

    #[test]
    fn everything_masked_returns_none() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut greedy = Sampler::new(SamplingConfig::greedy(), 1);
        assert_eq!(greedy.sample(&logits), None);
        let mut hot = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: None,
            },
            1,
        );
        assert_eq!(hot.sample(&logits), None);
    }

    #[test]
    fn same_seed_gives_the_same_sequence() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let a: Vec<u32> = {
            let mut s = Sampler::new(cfg.clone(), 42);
            (0..50).filter_map(|_| s.sample(&logits)).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(cfg, 42);
            (0..50).filter_map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        let logits = vec![5.0, 4.0, 0.0, -5.0];
        let mut s = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: Some(2),
                top_p: None,
            },
            3,
        );
        for _ in 0..300 {
            let id = s.sample(&logits).unwrap();
            assert!(id == 0 || id == 1, "top-k let through {id}");
        }
    }

    #[test]
    fn top_p_keeps_at_least_one_token() {
        // The top token already holds more than p; the nucleus must still be
        // non-empty rather than collapsing to nothing.
        let logits = vec![20.0, 0.0, 0.0];
        let mut s = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: Some(0.5),
            },
            9,
        );
        for _ in 0..100 {
            assert_eq!(s.sample(&logits), Some(0));
        }
    }

    /// Low temperature should concentrate mass on the top token; high temperature
    /// should spread it. This is the property, not the exact numbers.
    #[test]
    fn temperature_controls_concentration() {
        let logits = vec![3.0, 2.0, 1.0];
        let count_top = |t: f32| {
            let mut s = Sampler::new(
                SamplingConfig {
                    temperature: t,
                    top_k: None,
                    top_p: None,
                },
                11,
            );
            (0..2000).filter(|_| s.sample(&logits) == Some(0)).count()
        };
        let cold = count_top(0.2);
        let hot = count_top(3.0);
        assert!(cold > hot, "cold {cold} should exceed hot {hot}");
        assert!(
            cold > 1900,
            "temperature 0.2 should be nearly deterministic"
        );
        assert!(hot < 1200, "temperature 3.0 should spread mass");
    }

    #[test]
    fn a_zero_seed_still_produces_varied_output() {
        let mut r = Rng::new(0);
        let first = r.next_u64();
        assert_ne!(first, 0);
        assert_ne!(first, r.next_u64());
    }

    #[test]
    fn sampling_roughly_follows_the_distribution() {
        // Two tokens, logits differing by ln(3) → expect about 3:1.
        let logits = vec![0.0, 3.0f32.ln()];
        let mut s = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: None,
            },
            2024,
        );
        let n = 20_000;
        let ones = (0..n).filter(|_| s.sample(&logits) == Some(1)).count();
        let ratio = ones as f64 / n as f64;
        assert!(
            (ratio - 0.75).abs() < 0.02,
            "expected ~0.75, got {ratio:.4}"
        );
    }
}
