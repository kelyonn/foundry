//! Scalar kernels. Every one of these has its shape-before/shape-after written
//! down, per docs/RULES.md's code-style rule.

use crate::Tensor;

/// Linear layer: `y = x @ w^T (+ bias)`.
///
/// Shapes: `x [m, k]`, `w [n, k]`, `bias [n]` → `y [m, n]`.
///
/// **Note `w` is `[out, in]`, not `[in, out]`.** That is how PyTorch's `nn.Linear`
/// stores its weight and therefore how it appears in the safetensors file, so the
/// natural thing is to consume it as stored and transpose *in the indexing* rather
/// than materialising a transposed copy. Each output element is a dot product of a
/// row of `x` with a row of `w`, which means both operands are walked contiguously —
/// the cache-friendliest arrangement available without blocking.
///
/// Getting this backwards is a classic: the shapes still line up when `in == out`,
/// which is true for `o_proj` and `q_proj` in this model (both `[896, 896]`), so the
/// bug would pass silently there and explode on `k_proj` `[128, 896]`.
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    assert_eq!(x.shape().len(), 2, "x must be 2-D");
    assert_eq!(w.shape().len(), 2, "w must be 2-D");
    let (m, k) = (x.shape()[0], x.shape()[1]);
    let (n, k_w) = (w.shape()[0], w.shape()[1]);
    assert_eq!(
        k, k_w,
        "inner dimensions disagree: x is [{m}, {k}], w is [{n}, {k_w}]"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), n, "bias length must equal output width");
    }

    let mut out = vec![0.0f32; m * n];
    par_linear(x.data(), w.data(), m, n, k, &mut out);

    if let Some(b) = bias {
        let bd = b.data();
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] += bd[j];
            }
        }
    }

    Tensor::new(out, vec![m, n])
}

/// Split the output columns across threads.
///
/// **Why columns (`n`) and not rows (`m`).** During decode `m == 1` — one token —
/// so splitting by rows would leave every thread but one idle in exactly the phase
/// that dominates generation time. `n` is 896, 4864 or 151936, so it always
/// divides well. Prefill splits fine either way.
///
/// Each thread writes into a private scratch buffer that is then copied into
/// place. The copy is `m × n` against `m × n × k` of arithmetic — for `k = 896`
/// that is under a tenth of a percent, and it buys disjoint contiguous writes
/// rather than a strided borrow dance.
///
/// Every output element is still computed by exactly the same sequence of
/// operations as the single-threaded version, so this cannot change a single bit
/// of the result. That is deliberate: it lets parity re-run as a pure regression
/// check, with any difference meaning a bug rather than an expected reordering.
fn par_linear(xd: &[f32], wd: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let threads = crate::pool::threads().min(n).max(1);

    // Not worth the spawn cost on small problems.
    if threads == 1 || m * n * k < 1 << 16 {
        dot_block(xd, wd, m, n, k, 0, n, out, n);
        return;
    }

    let chunk = n.div_ceil(threads);
    // One private scratch per task; each is locked exactly once, by its own task,
    // so the mutex is uncontended — it is only there to hand `&mut` to a `Fn`.
    let scratches: Vec<std::sync::Mutex<Vec<f32>>> = (0..threads)
        .map(|t| {
            let lo = (t * chunk).min(n);
            let hi = ((t + 1) * chunk).min(n);
            std::sync::Mutex::new(vec![0.0f32; m * hi.saturating_sub(lo)])
        })
        .collect();

    // Persistent pool, not `thread::scope` (D22): no per-call thread spawn.
    crate::pool::parallel_for(threads, &|t| {
        let lo = (t * chunk).min(n);
        let hi = ((t + 1) * chunk).min(n);
        if lo >= hi {
            return;
        }
        let mut scratch = scratches[t].lock().unwrap();
        dot_block(xd, wd, m, n, k, lo, hi, &mut scratch, hi - lo);
    });

    for (t, scratch) in scratches.iter().enumerate() {
        let scratch = scratch.lock().unwrap();
        let lo = (t * chunk).min(n);
        let hi = ((t + 1) * chunk).min(n);
        if lo >= hi {
            continue;
        }
        let width = hi - lo;
        for i in 0..m {
            out[i * n + lo..i * n + hi].copy_from_slice(&scratch[i * width..(i + 1) * width]);
        }
    }
}

/// Rows of `x` processed per tile. A tile of `TILE_M` rows of `x` (64 × 896 × 4 B ≈
/// 229 KB) stays resident in L2 while each weight row is streamed past it, so a
/// weight row is read from memory once per *tile* instead of once per *token*.
/// For a 512-token prefill that is 8 passes over the weights instead of 512.
const TILE_M: usize = 64;

/// Compute output columns `[j_lo, j_hi)` into `out`, whose row stride is `out_n`.
///
/// Loop order is `i-tile → j → i`: the weight row `w_j` (≈3.5 KB) is hot in L1
/// while the tile's rows of `x` stream through it. The previous `i → j` order
/// re-read the entire weight chunk from memory for every token.
#[allow(clippy::too_many_arguments)]
fn dot_block(
    xd: &[f32],
    wd: &[f32],
    m: usize,
    _n: usize,
    k: usize,
    j_lo: usize,
    j_hi: usize,
    out: &mut [f32],
    out_n: usize,
) {
    let dot = select_dot();
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + TILE_M).min(m);
        for j in j_lo..j_hi {
            let w_row = &wd[j * k..(j + 1) * k];
            for i in i0..i1 {
                out[i * out_n + (j - j_lo)] = dot(&xd[i * k..(i + 1) * k], w_row);
            }
        }
        i0 = i1;
    }
}

type DotFn = fn(&[f32], &[f32]) -> f32;

/// Pick the dot-product kernel once. `FOUNDRY_SCALAR=1` forces the scalar f64
/// kernel so a before/after comparison can be run from one binary (hard rule 3:
/// a speedup claim needs both numbers from the same harness).
fn select_dot() -> DotFn {
    static CHOICE: std::sync::OnceLock<DotFn> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        if std::env::var_os("FOUNDRY_SCALAR").is_none() {
            return dot_neon_safe;
        }
        dot_scalar
    })
}

/// The reference kernel: f64 accumulation, four independent accumulators.
///
/// Why f64: a sequential f32 sum over k = 896 terms measured 1.3e-4 error on
/// `q_proj` (D14) — the whole parity budget before layer 2. The reference (BLAS)
/// does not sum sequentially either, so imitating its error is pointless; being
/// more accurate than both is how you agree with it. Kept as the oracle that the
/// SIMD kernel is tested against.
pub fn dot_scalar(x_row: &[f32], w_row: &[f32]) -> f32 {
    let k = x_row.len();
    debug_assert_eq!(k, w_row.len());
    let mut a0 = 0.0f64;
    let mut a1 = 0.0f64;
    let mut a2 = 0.0f64;
    let mut a3 = 0.0f64;
    let chunks = k / 4;
    for c in 0..chunks {
        let b = c * 4;
        a0 += x_row[b] as f64 * w_row[b] as f64;
        a1 += x_row[b + 1] as f64 * w_row[b + 1] as f64;
        a2 += x_row[b + 2] as f64 * w_row[b + 2] as f64;
        a3 += x_row[b + 3] as f64 * w_row[b + 3] as f64;
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    for t in chunks * 4..k {
        acc += x_row[t] as f64 * w_row[t] as f64;
    }
    acc as f32
}

#[cfg(target_arch = "aarch64")]
fn dot_neon_safe(x_row: &[f32], w_row: &[f32]) -> f32 {
    assert_eq!(x_row.len(), w_row.len());
    // SAFETY: NEON is mandatory on aarch64; lengths checked equal above.
    unsafe { dot_neon(x_row, w_row) }
}

/// NEON kernel: 4 vector accumulators × 4 lanes = 16 independent partial sums,
/// with f32 FMA inside a block and a flush to an f64 total every `BLOCK` elements.
///
/// The flush is the accuracy trick: rounding error of an f32 accumulator grows
/// with the number of terms *it* has absorbed, so bounding each accumulator to
/// `BLOCK / 16` terms keeps the error near the f64 kernel's while the hot loop
/// stays pure f32 FMA at 16 elements per iteration.
///
/// # Safety
/// `x_row.len() == w_row.len()`; all loads are within `[0, k16)` with `k16 <= k`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(x_row: &[f32], w_row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK: usize = 256;
    let k = x_row.len();
    let k16 = k / 16 * 16;
    let xp = x_row.as_ptr();
    let wp = w_row.as_ptr();
    let mut total = 0.0f64;
    let mut i = 0;
    while i < k16 {
        let end = (i + BLOCK).min(k16);
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        while i < end {
            // SAFETY: i + 16 <= end <= k16 <= k.
            a0 = vfmaq_f32(a0, vld1q_f32(xp.add(i)), vld1q_f32(wp.add(i)));
            a1 = vfmaq_f32(a1, vld1q_f32(xp.add(i + 4)), vld1q_f32(wp.add(i + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(xp.add(i + 8)), vld1q_f32(wp.add(i + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(xp.add(i + 12)), vld1q_f32(wp.add(i + 12)));
            i += 16;
        }
        let s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        total += vaddvq_f32(s) as f64;
    }
    for t in k16..k {
        total += x_row[t] as f64 * w_row[t] as f64;
    }
    total as f32
}

/// Root-mean-square layer norm, applied per row.
///
/// Shapes: `x [m, k]`, `weight [k]` → `[m, k]`.
///
/// `y = x / sqrt(mean(x²) + eps) * weight`
///
/// Two differences from LayerNorm worth being able to state: there is **no mean
/// subtraction** (so it is not centred) and there is **no bias term**. That makes it
/// cheaper, and empirically the centring was doing nothing useful in transformers.
///
/// The `eps` goes *inside* the square root, which matters: HuggingFace computes
/// `x * rsqrt(mean + eps)`, and putting eps outside would produce a small but
/// parity-breaking difference.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.shape().len(), 2, "rms_norm expects [m, k]");
    let (m, k) = (x.shape()[0], x.shape()[1]);
    assert_eq!(weight.len(), k, "norm weight must match the row width");

    let xd = x.data();
    let wd = weight.data();
    let mut out = vec![0.0f32; m * k];

    for i in 0..m {
        let row = &xd[i * k..(i + 1) * k];
        // f64 for the same reason as `linear`: 896 squared terms summed in f32
        // loses more precision than the parity budget allows.
        let mut sum_sq = 0.0f64;
        for &v in row {
            sum_sq += v as f64 * v as f64;
        }
        let scale = (1.0 / ((sum_sq / k as f64) + eps as f64).sqrt()) as f32;
        for t in 0..k {
            out[i * k + t] = row[t] * scale * wd[t];
        }
    }

    Tensor::new(out, vec![m, k])
}

/// Numerically-stable softmax over the last dimension, in place.
///
/// Shapes: `[.., n]`, unchanged.
///
/// The max is subtracted before exponentiating. Without it, `exp` of a large logit
/// overflows to `inf` and the result is `NaN` — and note this is exactly what makes
/// `-inf` masking work: `exp(-inf - max) == 0` exactly, no special-casing needed.
pub fn softmax_rows(x: &mut [f32], n: usize) {
    assert!(
        n > 0 && x.len().is_multiple_of(n),
        "length must be a multiple of n"
    );
    for row in x.chunks_mut(n) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            // Every entry masked. Leave it to the caller to notice; producing NaN
            // here would silently poison everything downstream.
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// SiLU, a.k.a. swish: `x * sigmoid(x)`. Applied elementwise, in place.
///
/// Qwen2's `hidden_act`. Written as `x / (1 + exp(-x))` to avoid computing the
/// sigmoid separately.
pub fn silu_(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// Elementwise multiply into the first slice.
pub fn mul_(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "elementwise multiply needs equal lengths");
    for (x, y) in a.iter_mut().zip(b) {
        *x *= y;
    }
}

/// Elementwise add into the first slice. Used for residual connections.
pub fn add_(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "elementwise add needs equal lengths");
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `k` not a multiple of 4 exercises the remainder tail after the 4-wide
    /// accumulator loop in `dot_block` — the case most likely to have an
    /// off-by-one, since `chunks * 4..k` is empty exactly when it divides evenly.
    #[test]
    fn linear_handles_k_not_a_multiple_of_four() {
        for k in [1usize, 2, 3, 5, 6, 7, 9] {
            let x = Tensor::new((0..k).map(|i| (i + 1) as f32).collect(), vec![1, k]);
            let w = Tensor::new(vec![1.0; k], vec![1, k]);
            let y = linear(&x, &w, None);
            let expected: f32 = (1..=k as u32).map(|i| i as f32).sum();
            assert!(
                (y.data()[0] - expected).abs() < 1e-3,
                "k={k}: got {}, expected {expected}",
                y.data()[0]
            );
        }
    }

    /// An empty batch (`m = 0`) must produce an empty tensor, not panic — this is
    /// what a zero-length prompt or an already-exhausted generation would hand in.
    #[test]
    fn linear_handles_an_empty_batch() {
        let x = Tensor::new(vec![], vec![0, 4]);
        let w = Tensor::new(vec![1.0; 8], vec![2, 4]);
        let y = linear(&x, &w, None);
        assert_eq!(y.shape(), &[0, 2]);
        assert!(y.data().is_empty());
    }

    /// `n` not evenly divisible by the thread count exercises the boundary between
    /// the last full chunk and the ragged remainder in `par_linear`'s column
    /// split. Real runs already cover this incidentally (896 / 10 cores does not
    /// divide evenly, and parity is bit-identical) — this pins it as a small,
    /// fast, deterministic unit test independent of the checkpoint.
    #[test]
    fn linear_output_is_correct_at_odd_column_counts() {
        for n in [1usize, 2, 7, 13, 97] {
            let k = 6;
            let x = Tensor::new((0..k).map(|i| i as f32 + 1.0).collect(), vec![1, k]);
            let w = Tensor::new(
                (0..n * k).map(|i| ((i % 5) as f32) - 2.0).collect(),
                vec![n, k],
            );
            let got = linear(&x, &w, None);

            // Hand-compute the same thing with a trivial sequential loop, as an
            // oracle independent of `par_linear`'s threading and accumulator
            // structure.
            let xd = x.data();
            let wd = w.data();
            let mut want = vec![0.0f64; n];
            for (j, slot) in want.iter_mut().enumerate() {
                for t in 0..k {
                    *slot += xd[t] as f64 * wd[j * k + t] as f64;
                }
            }
            for (j, &expected) in want.iter().enumerate() {
                assert!(
                    (got.data()[j] as f64 - expected).abs() < 1e-3,
                    "n={n} j={j}: got {}, want {}",
                    got.data()[j],
                    expected
                );
            }
        }
    }

    /// The kernel actually used (NEON on aarch64) must agree with the f64 scalar
    /// oracle to well inside the parity budget, across ragged lengths: below one
    /// 16-wide step, exact multiples, multi-block (> 256), and with a scalar tail.
    #[test]
    fn selected_kernel_matches_the_f64_oracle() {
        let dot = select_dot();
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for k in [
            1usize, 3, 15, 16, 17, 31, 64, 255, 256, 257, 896, 4864, 4871,
        ] {
            let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..k).map(|_| rnd()).collect();
            let want = dot_scalar(&x, &w);
            let got = dot(&x, &w);
            let scale: f32 = x
                .iter()
                .zip(&w)
                .map(|(a, b)| (a * b).abs())
                .sum::<f32>()
                .max(1.0);
            assert!(
                (got - want).abs() <= 1e-6 * scale,
                "k={k}: got {got}, want {want}, scale {scale}"
            );
        }
    }

    /// A batch larger than `TILE_M` exercises the tile boundary in `dot_block`.
    #[test]
    fn linear_is_correct_across_tile_boundaries() {
        let (m, n, k) = (TILE_M * 2 + 5, 9, 20);
        let x = Tensor::new(
            (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect(),
            vec![m, k],
        );
        let w = Tensor::new(
            (0..n * k).map(|i| ((i % 5) as f32) - 2.0).collect(),
            vec![n, k],
        );
        let y = linear(&x, &w, None);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k)
                    .map(|t| x.data()[i * k + t] as f64 * w.data()[j * k + t] as f64)
                    .sum();
                assert!(
                    (y.data()[i * n + j] as f64 - want).abs() < 1e-3,
                    "i={i} j={j}"
                );
            }
        }
    }

    #[test]
    fn linear_matches_hand_computed() {
        // x [2, 3]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // w [2, 3] — stored as [out, in], so row j is the j-th output's weights.
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0], vec![2, 3]);
        let y = linear(&x, &w, None);
        assert_eq!(y.shape(), &[2, 2]);
        // row 0: [1,2,3]·[1,0,0] = 1 ; [1,2,3]·[0,1,1] = 5
        // row 1: [4,5,6]·[1,0,0] = 4 ; [4,5,6]·[0,1,1] = 11
        assert_eq!(y.data(), &[1.0, 5.0, 4.0, 11.0]);
    }

    #[test]
    fn linear_applies_bias() {
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);
        let w = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0], vec![2]);
        let y = linear(&x, &w, Some(&b));
        assert_eq!(y.data(), &[12.0, 24.0]);
    }

    /// The non-square case that catches a transposed indexing bug. `k_proj` in this
    /// model is `[128, 896]`, so a square-only test would not have caught it.
    #[test]
    fn linear_handles_non_square_weights() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let w = Tensor::new((0..8).map(|i| i as f32).collect(), vec![2, 4]);
        let y = linear(&x, &w, None);
        assert_eq!(y.shape(), &[1, 2]);
        // [1,2,3,4]·[0,1,2,3] = 0+2+6+12 = 20
        // [1,2,3,4]·[4,5,6,7] = 4+10+18+28 = 60
        assert_eq!(y.data(), &[20.0, 60.0]);
    }

    #[test]
    fn rms_norm_normalises_and_scales() {
        let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
        let w = Tensor::new(vec![1.0, 1.0], vec![2]);
        let y = rms_norm(&x, &w, 0.0);
        // rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
        let rms = 12.5f32.sqrt();
        assert!((y.data()[0] - 3.0 / rms).abs() < 1e-6);
        assert!((y.data()[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_does_not_centre() {
        // A constant row has zero variance. LayerNorm would map it to zeros;
        // RMSNorm must not, because it never subtracts the mean.
        let x = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![1, 4]);
        let w = Tensor::new(vec![1.0; 4], vec![4]);
        let y = rms_norm(&x, &w, 0.0);
        for &v in y.data() {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut x, 3);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);

        // Without max-subtraction this overflows to NaN.
        let mut big = vec![1000.0, 1000.5];
        softmax_rows(&mut big, 2);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// `-inf` entries must come out as exactly zero probability. This is the
    /// property the whole grammar crate depends on.
    #[test]
    fn softmax_gives_masked_entries_exactly_zero() {
        let mut x = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax_rows(&mut x, 3);
        assert_eq!(x[1], 0.0);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silu_matches_definition() {
        let mut x = vec![0.0, 1.0, -1.0];
        silu_(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-7);
        assert!((x[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        // SiLU is not monotone: it dips below zero for negative inputs.
        assert!(x[2] < 0.0);
    }
}
