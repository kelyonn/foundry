//! The Qwen2 forward pass.
//!
//! Prefill only, no cache — one call processes a whole prompt. Correctness first
//! (hard rule 2); the KV cache and any optimization come after parity passes.
//!
//! Tensor convention per docs/ARCHITECTURE.md: batch is elided, attention tensors
//! are `[heads, seq, head_dim]` ("hsd"), everything else is `[seq, hidden]`.

use crate::config::Config;
use crate::kv_cache::KvCache;
use crate::weights::{LayerWeights, Weights};
use std::collections::HashMap;
use tensor::ops::{add_, mul_, rms_norm, silu_, softmax_rows};
use tensor::Tensor;

/// Optional capture of intermediate activations, for the parity harness.
///
/// Names match the fixture names produced by `scripts/dump_reference.py`, so the
/// comparison is a straight lookup rather than a translation table — a translation
/// table is somewhere else to introduce an off-by-one.
#[derive(Default)]
pub struct Trace {
    pub tensors: HashMap<String, Tensor>,
}

impl Trace {
    fn put(&mut self, name: impl Into<String>, t: &Tensor) {
        self.tensors.insert(name.into(), t.clone());
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}

pub struct Model {
    pub config: Config,
    pub weights: Weights,
}

impl Model {
    pub fn new(config: Config, weights: Weights) -> Model {
        Model { config, weights }
    }

    /// Run tokens through the model with no history. Returns `[seq, vocab_size]`.
    ///
    /// Implemented as a cached call against a throwaway cache, deliberately: if
    /// the cached and uncached paths were separate implementations they would
    /// drift, and the drift would show up as a model that is correct on the first
    /// token and subtly wrong afterwards. `cached_decode_matches_full_forward` in
    /// `tests/` is the test that would catch it; sharing the code means there is
    /// nothing to catch.
    pub fn forward(&self, tokens: &[u32], trace: Option<&mut Trace>) -> Tensor {
        let cfg = &self.config;
        let mut cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            tokens.len().max(1),
        );
        self.forward_cached(tokens, &mut cache, trace)
    }

    /// Run tokens, continuing from whatever is already in `cache`.
    ///
    /// Prefill is this with an empty cache and the whole prompt; decode is this
    /// with one token, repeatedly. They are the same code — the only thing that
    /// changes is `seq` and how much history attention scans.
    pub fn forward_cached(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        trace: Option<&mut Trace>,
    ) -> Tensor {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "cannot run an empty prompt");
        assert!(
            cache.has_room_for(seq),
            "KV cache holds {} positions, needed {}",
            cache.max_seq(),
            cache.len() + seq
        );

        // Absolute position of the first new token. Every position-dependent thing
        // below is offset by this, and forgetting to do so is *the* KV cache bug:
        // generation stays fluent because RoPE still applies *some* rotation, and
        // the model just quietly behaves as if every token were at position 0.
        let offset = cache.len();

        let mut trace = trace;
        macro_rules! capture {
            ($name:expr, $t:expr) => {
                if let Some(tr) = trace.as_deref_mut() {
                    tr.put($name, $t);
                }
            };
        }

        // Embedding lookup: gather rows of [vocab, hidden] → [seq, hidden].
        // A "matmul with a one-hot matrix" conceptually, but nobody does that.
        let vocab_rows = self.weights.embed_tokens.shape()[0];
        let hidden = cfg.hidden_size;
        let mut x_data = Vec::with_capacity(seq * hidden);
        for &tok in tokens {
            // Checked explicitly rather than left to the slice index inside
            // `row_f32` to panic on: forward_cached is a public entry point, and
            // "token id out of range" is a far more useful message than whatever
            // an out-of-bounds slice panic says about a `Vec` it knows nothing
            // about the meaning of.
            assert!(
                (tok as usize) < vocab_rows,
                "token id {tok} is out of range for a {vocab_rows}-row embedding table"
            );
            let row = self.weights.embed_tokens.row_f32(tok as usize);
            x_data.extend_from_slice(&row);
        }
        let mut x = Tensor::new(x_data, vec![seq, hidden]);
        capture!("embed_out", &x);

        // Precompute RoPE tables once — they depend on position and head_dim only,
        // not on the layer, so recomputing per layer would be 24× the work.
        let (cos, sin) = rope_tables(offset, seq, cfg.head_dim(), cfg.rope_theta);

        for (i, layer) in self.weights.layers.iter().enumerate() {
            x = self.layer(
                i,
                layer,
                &x,
                &cos,
                &sin,
                cache,
                offset,
                trace.as_deref_mut(),
            );
        }

        // Commit the new positions once, after every layer has written its keys
        // and values. Advancing per layer would make layer 1 see layer 0's tokens
        // as history.
        cache.advance(seq).expect("checked room above");

        let x = rms_norm(&x, &self.weights.final_norm, cfg.rms_norm_eps);
        capture!("final_norm", &x);

        // Output projection. With tied embeddings this is the embedding matrix
        // reused as [vocab, hidden] — exactly the [out, in] layout `linear` wants,
        // so no transpose is needed anywhere.
        let logits = self.weights.output_projection().matmul(&x, None);
        capture!("logits", &logits);
        logits
    }

    /// One decoder layer. Two residual connections, pre-norm on both.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        idx: usize,
        w: &LayerWeights,
        x: &Tensor,
        cos: &[f32],
        sin: &[f32],
        cache: &mut KvCache,
        offset: usize,
        trace: Option<&mut Trace>,
    ) -> Tensor {
        let cfg = &self.config;
        let mut trace = trace;
        macro_rules! capture {
            ($suffix:expr, $t:expr) => {
                if let Some(tr) = trace.as_deref_mut() {
                    tr.put(format!("layer{:02}.{}", idx, $suffix), $t);
                }
            };
        }

        // ---- attention block -------------------------------------------------
        // Pre-norm: normalise going *into* the sublayer, and the residual carries
        // the unnormalised stream. Post-norm (the original transformer) normalises
        // the sum instead, and is markedly harder to train deep.
        let normed = rms_norm(x, &w.input_layernorm, cfg.rms_norm_eps);
        capture!("input_layernorm", &normed);

        let q = w.q_proj.matmul(&normed, Some(&w.q_bias)); // [seq, 896]
        let k = w.k_proj.matmul(&normed, Some(&w.k_bias)); // [seq, 128]
        let v = w.v_proj.matmul(&normed, Some(&w.v_bias)); // [seq, 128]
        capture!("q_proj", &q);
        capture!("k_proj", &k);
        capture!("v_proj", &v);

        let seq = x.shape()[0];
        let head_dim = cfg.head_dim();
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;

        // [seq, heads*head_dim] → [heads, seq, head_dim].
        let mut q_hsd = split_heads(&q, seq, n_q, head_dim);
        let mut k_hsd = split_heads(&k, seq, n_kv, head_dim);
        let v_hsd = split_heads(&v, seq, n_kv, head_dim);

        // RoPE goes on q and k, never on v. v carries content, not position — and
        // rotating it would be a bug that still produces plausible-looking output.
        apply_rope(&mut q_hsd, n_q, seq, head_dim, cos, sin);
        apply_rope(&mut k_hsd, n_kv, seq, head_dim, cos, sin);
        capture!("rope_q", &hsd_tensor(&q_hsd, n_q, seq, head_dim));
        capture!("rope_k", &hsd_tensor(&k_hsd, n_kv, seq, head_dim));

        // Keys and values go into the cache *before* attention reads it, so the
        // new tokens can attend to themselves as well as to the history.
        cache.append_layer(idx, &k_hsd, &v_hsd);

        let attn = self.attention(cache, idx, &q_hsd, seq, offset, n_q, n_kv, head_dim);
        let attn_out = w.o_proj.matmul(&attn, None);
        capture!("o_proj", &attn_out);
        capture!("attn_out", &attn_out);

        let mut x = x.clone();
        add_(x.data_mut(), attn_out.data());

        // ---- MLP block -------------------------------------------------------
        let normed = rms_norm(&x, &w.post_attention_layernorm, cfg.rms_norm_eps);
        capture!("post_attention_layernorm", &normed);

        // SwiGLU: down(silu(gate(x)) * up(x)). Two parallel projections up, one
        // down — which is why intermediate_size (4864) is not simply 4× hidden.
        let mut gate = w.gate_proj.matmul(&normed, None);
        let up = w.up_proj.matmul(&normed, None);
        capture!("gate_proj", &gate);
        capture!("up_proj", &up);

        silu_(gate.data_mut());
        mul_(gate.data_mut(), up.data());
        let down = w.down_proj.matmul(&gate, None);
        capture!("down_proj", &down);
        capture!("mlp_out", &down);

        add_(x.data_mut(), down.data());
        capture!("layer_out", &x);
        x
    }

    /// Causal grouped-query attention over the cache.
    ///
    /// `q` is the `seq` new queries; keys and values come from the cache, which
    /// already contains the history *and* the new tokens. Returns
    /// `[seq, n_q * head_dim]`, ready for `o_proj`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        cache: &KvCache,
        layer: usize,
        q: &[Vec<f32>],
        seq: usize,
        offset: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
    ) -> Tensor {
        let group = n_q / n_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = offset + seq;
        let mut out = vec![0.0f32; seq * n_q * head_dim];

        for (h, q_head) in q.iter().enumerate().take(n_q) {
            // The GQA mapping. `h / group`, not `h % n_kv` — with 14 heads and 2
            // KV heads, `/` gives heads 0–6 → KV 0 and 7–13 → KV 1, while `%`
            // would interleave them. Both "work" dimensionally and produce fluent
            // output; only one is right. docs/ARCHITECTURE.md flags this as the
            // worst available failure mode for exactly that reason.
            let kv = h / group;
            let keys = cache.keys(layer, kv, total);
            let values = cache.values(layer, kv, total);

            for i in 0..seq {
                // Absolute position of this query. During decode `offset` is the
                // history length and `i` is 0, so `p` is the real position — this
                // is what makes one-token-at-a-time attend over everything.
                let p = offset + i;

                // Causal masking by *not computing* positions after `p`, rather
                // than computing them and adding -inf. Cheaper, and there is no
                // masked value left around to leak through as a NaN.
                let mut scores = vec![0.0f32; p + 1];
                let qi = &q_head[i * head_dim..(i + 1) * head_dim];
                for (j, score) in scores.iter_mut().enumerate() {
                    let kj = &keys[j * head_dim..(j + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qi[d] * kj[d];
                    }
                    *score = dot * scale;
                }
                softmax_rows(&mut scores, p + 1);

                for (j, &w) in scores.iter().enumerate() {
                    let vj = &values[j * head_dim..(j + 1) * head_dim];
                    let base = i * (n_q * head_dim) + h * head_dim;
                    for d in 0..head_dim {
                        out[base + d] += w * vj[d];
                    }
                }
            }
        }

        Tensor::new(out, vec![seq, n_q * head_dim])
    }
}

/// `[seq, heads*head_dim]` → one `Vec<f32>` per head, each `[seq, head_dim]`.
///
/// The projection output is head-major within a row: element `(s, h*head_dim + d)`.
fn split_heads(t: &Tensor, seq: usize, heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
    let d = t.data();
    let width = heads * head_dim;
    (0..heads)
        .map(|h| {
            let mut out = Vec::with_capacity(seq * head_dim);
            for s in 0..seq {
                let start = s * width + h * head_dim;
                out.extend_from_slice(&d[start..start + head_dim]);
            }
            out
        })
        .collect()
}

/// Re-pack per-head buffers into a `[heads, seq, head_dim]` tensor, for tracing.
fn hsd_tensor(heads_data: &[Vec<f32>], heads: usize, seq: usize, head_dim: usize) -> Tensor {
    let mut out = Vec::with_capacity(heads * seq * head_dim);
    for h in heads_data {
        out.extend_from_slice(h);
    }
    Tensor::new(out, vec![heads, seq, head_dim])
}

/// Precompute `cos` and `sin` tables, each `[seq, head_dim]`.
///
/// `inv_freq[i] = theta^(-2i/head_dim)` for `i` in `0..head_dim/2`, then the table
/// is that vector **concatenated with itself** — not interleaved. This is the
/// GPT-NeoX layout that HuggingFace uses, and it pairs dimension `i` with
/// `i + head_dim/2`.
///
/// The alternative convention pairs `2i` with `2i+1`. Both are rotations by the
/// same angles and both "work" in the sense of training a model; they are simply
/// incompatible, so using the wrong one against these weights gives garbage that
/// still has plausible magnitudes.
fn rope_tables(offset: usize, seq: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];

    // Row `i` of the table is for *absolute* position `offset + i`.
    for i in 0..seq {
        let pos = offset + i;
        for f in 0..half {
            // f64 for the exponent: theta is 1e6 here and f32 loses meaningful
            // precision in the high-frequency terms.
            let inv_freq = 1.0f64 / (theta as f64).powf(2.0 * f as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            let (sn, cs) = angle.sin_cos();
            cos[i * head_dim + f] = cs as f32;
            cos[i * head_dim + f + half] = cs as f32;
            sin[i * head_dim + f] = sn as f32;
            sin[i * head_dim + f + half] = sn as f32;
        }
    }
    (cos, sin)
}

/// Rotary position embedding, in place.
///
/// `out = x * cos + rotate_half(x) * sin`, where
/// `rotate_half([a, b]) = [-b, a]` splitting at `head_dim/2`.
fn apply_rope(
    heads_data: &mut [Vec<f32>],
    heads: usize,
    seq: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for h in heads_data.iter_mut().take(heads) {
        for pos in 0..seq {
            let base = pos * head_dim;
            let c = &cos[base..base + head_dim];
            let s = &sin[base..base + head_dim];
            let row = &mut h[base..base + head_dim];

            // Snapshot first: the rotation reads the other half of the row, so
            // updating in place without a copy would feed rotated values back in.
            let orig: Vec<f32> = row.to_vec();
            for d in 0..head_dim {
                let rotated = if d < half {
                    -orig[d + half]
                } else {
                    orig[d - half]
                };
                row[d] = orig[d] * c[d] + rotated * s[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        // angle = 0 → cos = 1, sin = 0.
        let head_dim = 8;
        let (cos, sin) = rope_tables(0, 1, head_dim, 10000.0);
        let mut heads = vec![(0..head_dim).map(|i| i as f32).collect::<Vec<f32>>()];
        let before = heads[0].clone();
        apply_rope(&mut heads, 1, 1, head_dim, &cos, &sin);
        for (a, b) in before.iter().zip(&heads[0]) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn rope_preserves_the_norm_of_each_rotated_pair() {
        // A rotation changes direction, never length.
        let head_dim = 8;
        let seq = 4;
        let (cos, sin) = rope_tables(0, seq, head_dim, 10000.0);
        let mut heads: Vec<Vec<f32>> =
            vec![(0..seq * head_dim).map(|i| (i % 7) as f32 - 3.0).collect()];
        let before: Vec<f32> = heads[0].clone();
        apply_rope(&mut heads, 1, seq, head_dim, &cos, &sin);

        let half = head_dim / 2;
        for pos in 0..seq {
            for i in 0..half {
                let b0 = before[pos * head_dim + i];
                let b1 = before[pos * head_dim + i + half];
                let a0 = heads[0][pos * head_dim + i];
                let a1 = heads[0][pos * head_dim + i + half];
                let nb = (b0 * b0 + b1 * b1).sqrt();
                let na = (a0 * a0 + a1 * a1).sqrt();
                assert!((nb - na).abs() < 1e-5, "pair norm changed: {nb} -> {na}");
            }
        }
    }

    #[test]
    fn split_heads_is_head_major_within_a_row() {
        // seq=2, heads=2, head_dim=3 → row s is [h0d0 h0d1 h0d2 h1d0 h1d1 h1d2]
        let t = Tensor::new(
            vec![0., 1., 2., 10., 11., 12., 3., 4., 5., 13., 14., 15.],
            vec![2, 6],
        );
        let heads = split_heads(&t, 2, 2, 3);
        assert_eq!(heads[0], vec![0., 1., 2., 3., 4., 5.]);
        assert_eq!(heads[1], vec![10., 11., 12., 13., 14., 15.]);
    }
}
