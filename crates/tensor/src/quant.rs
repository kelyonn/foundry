//! int8 symmetric per-channel weight quantization.
//!
//! **Symmetric** — no zero point. The quantized value is `round(x / scale)` and the
//! dequantized value is `q * scale`, so zero maps to zero exactly. Asymmetric
//! quantization (with a zero point) fits skewed distributions better, but weight
//! distributions in a trained transformer are roughly zero-centred, so the extra
//! parameter buys little and costs an add in the inner loop.
//!
//! **Per-channel** — one scale per *output* channel, i.e. per row of the `[out, in]`
//! weight matrix, rather than one scale for the whole tensor. This matters: a single
//! outlier row would otherwise force a huge scale on every other row and crush them
//! toward zero. Per-channel is the standard choice for weight-only quantization and
//! it costs one `f32` per row — for a `[896, 896]` weight that is 896 extra floats
//! against 802,816 saved bytes.
//!
//! Activations stay `f32`. This is *weight-only* quantization: the memory win is
//! real (4× on weights, which dominate) and the accuracy cost is small, but it does
//! not speed up arithmetic by itself, because we dequantize on the fly to multiply.

use crate::Tensor;

/// A weight matrix stored as `i8` with one `f32` scale per row.
///
/// Deliberately a separate type rather than a `Tensor` variant, per
/// docs/ARCHITECTURE.md: quantized weights cannot be fed to the `f32` kernels, and
/// the type system should be the thing that says so rather than a runtime check.
pub struct QTensor {
    data: Vec<i8>,
    scales: Vec<f32>,
    shape: Vec<usize>,
}

impl QTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Bytes of weight storage, for the memory comparison in the benchmarks.
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Quantize a `[out, in]` weight matrix, one scale per row.
    ///
    /// `scale = max|row| / 127`. Note 127, not 128: the `i8` range is `-128..=127`,
    /// and using 127 keeps the mapping symmetric so that `-max` and `+max` are
    /// equally representable. Giving up the single extra negative value costs
    /// nothing and avoids an asymmetry that is easy to get subtly wrong.
    pub fn quantize(w: &Tensor) -> QTensor {
        assert_eq!(w.shape().len(), 2, "quantize expects a 2-D weight");
        let (rows, cols) = (w.shape()[0], w.shape()[1]);
        let wd = w.data();

        let mut data = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row = &wd[r * cols..(r + 1) * cols];
            let max_abs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // An all-zero row would give scale 0 and then a division by zero. Use
            // 1.0 so the row round-trips to zeros, which is the correct answer.
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales[r] = scale;
            for c in 0..cols {
                // round-half-away-from-zero, then clamp. The clamp is defensive:
                // max_abs/127 * 127 can round to 128 for the extreme element.
                let q = (row[c] / scale).round().clamp(-127.0, 127.0);
                data[r * cols + c] = q as i8;
            }
        }

        QTensor {
            data,
            scales,
            shape: vec![rows, cols],
        }
    }

    /// One dequantized row. Used for embedding lookup, which is a gather rather
    /// than a matmul, so there is no integer dot product to fold the scale into.
    pub fn row_f32(&self, i: usize) -> Vec<f32> {
        let cols = self.shape[1];
        let s = self.scales[i];
        self.data[i * cols..(i + 1) * cols]
            .iter()
            .map(|&q| q as f32 * s)
            .collect()
    }

    /// Reconstruct the `f32` weight. Used only by tests — the whole point is to
    /// avoid materialising this at run time.
    pub fn dequantize(&self) -> Tensor {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let s = self.scales[r];
            for c in 0..cols {
                out[r * cols + c] = self.data[r * cols + c] as f32 * s;
            }
        }
        Tensor::new(out, vec![rows, cols])
    }
}

/// `y = x @ qw^T (+ bias)`, dequantizing on the fly.
///
/// Shapes: `x [m, k]`, `qw [n, k]`, `bias [n]` → `y [m, n]`.
///
/// The scale is applied **once per output element**, after the integer dot product,
/// not once per multiply. That is the entire trick: `sum_t x[t] * (q[t] * s)` equals
/// `s * sum_t x[t] * q[t]` because `s` is constant across the row. Doing it the
/// naive way — dequantizing each weight inside the loop — would be `k` multiplies
/// instead of one and would defeat the purpose.
pub fn linear_q(x: &Tensor, qw: &QTensor, bias: Option<&Tensor>) -> Tensor {
    assert_eq!(x.shape().len(), 2, "x must be 2-D");
    let (m, k) = (x.shape()[0], x.shape()[1]);
    let (n, k_w) = (qw.shape[0], qw.shape[1]);
    assert_eq!(k, k_w, "inner dimensions disagree");

    let mut out = vec![0.0f32; m * n];
    par_linear_q(x.data(), &qw.data, &qw.scales, m, n, k, &mut out);

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

/// Threaded over output columns, mirroring `ops::par_linear`. Same reasoning:
/// `m == 1` during decode, so splitting by rows would leave every thread idle.
fn par_linear_q(
    xd: &[f32],
    wd: &[i8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    let threads = crate::pool::threads().min(n).max(1);

    if threads == 1 || m * n * k < 1 << 16 {
        dot_block_q(xd, wd, scales, m, k, 0, n, out, n);
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
        dot_block_q(xd, wd, scales, m, k, lo, hi, &mut scratch, hi - lo);
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

/// Same `i-tile → j → i` order as `ops::dot_block`: the int8 weight row (896 B) stays
/// in L1 while a tile of activation rows streams through it.
#[allow(clippy::too_many_arguments)]
fn dot_block_q(
    xd: &[f32],
    wd: &[i8],
    scales: &[f32],
    m: usize,
    k: usize,
    j_lo: usize,
    j_hi: usize,
    out: &mut [f32],
    out_n: usize,
) {
    const TILE_M: usize = 64;
    let dot = select_dot_q();
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + TILE_M).min(m);
        for j in j_lo..j_hi {
            let w_row = &wd[j * k..(j + 1) * k];
            for i in i0..i1 {
                // Scale once per output element, after the dot product.
                out[i * out_n + (j - j_lo)] = dot(&xd[i * k..(i + 1) * k], w_row) * scales[j];
            }
        }
        i0 = i1;
    }
}

type DotQFn = fn(&[f32], &[i8]) -> f32;

fn select_dot_q() -> DotQFn {
    static CHOICE: std::sync::OnceLock<DotQFn> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        if std::env::var_os("FOUNDRY_SCALAR").is_none() {
            return dot_q_neon_safe;
        }
        dot_q_scalar
    })
}

/// Scalar reference. f32 accumulation is fine here, unlike the f32 path: the weights
/// are already quantized to 8 bits, so the dominant error is the quantization itself
/// (~1e-3 relative) and it swamps accumulation rounding by three orders of magnitude.
pub fn dot_q_scalar(x_row: &[f32], w_row: &[i8]) -> f32 {
    let k = x_row.len();
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let chunks = k / 4;
    for c in 0..chunks {
        let b = c * 4;
        a0 += x_row[b] * w_row[b] as f32;
        a1 += x_row[b + 1] * w_row[b + 1] as f32;
        a2 += x_row[b + 2] * w_row[b + 2] as f32;
        a3 += x_row[b + 3] * w_row[b + 3] as f32;
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    for t in chunks * 4..k {
        acc += x_row[t] * w_row[t] as f32;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
fn dot_q_neon_safe(x_row: &[f32], w_row: &[i8]) -> f32 {
    assert_eq!(x_row.len(), w_row.len());
    // SAFETY: NEON is mandatory on aarch64; lengths checked equal above.
    unsafe { dot_q_neon(x_row, w_row) }
}

/// NEON int8 kernel, 16 weights per step: load 16 × i8, sign-extend i8→i16→i32,
/// convert to f32, FMA against 4 loads of `x`. The conversions are the cost of
/// weight-only quantization — they are why int8 saves memory and bandwidth but
/// not arithmetic (the module doc says so; this is where it is visible).
///
/// # Safety
/// `x_row.len() == w_row.len()`; loads are within `[0, k16)` with `k16 <= k`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q_neon(x_row: &[f32], w_row: &[i8]) -> f32 {
    use std::arch::aarch64::*;
    let k = x_row.len();
    let k16 = k / 16 * 16;
    let xp = x_row.as_ptr();
    let wp = w_row.as_ptr();
    let mut a0 = vdupq_n_f32(0.0);
    let mut a1 = vdupq_n_f32(0.0);
    let mut a2 = vdupq_n_f32(0.0);
    let mut a3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i < k16 {
        // SAFETY: i + 16 <= k16 <= k.
        let w16 = vld1q_s8(wp.add(i));
        let lo = vmovl_s8(vget_low_s8(w16));
        let hi = vmovl_s8(vget_high_s8(w16));
        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
        a0 = vfmaq_f32(a0, vld1q_f32(xp.add(i)), f0);
        a1 = vfmaq_f32(a1, vld1q_f32(xp.add(i + 4)), f1);
        a2 = vfmaq_f32(a2, vld1q_f32(xp.add(i + 8)), f2);
        a3 = vfmaq_f32(a3, vld1q_f32(xp.add(i + 12)), f3);
        i += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for t in k16..k {
        acc += x_row[t] * w_row[t] as f32;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::linear;

    #[test]
    fn selected_q_kernel_matches_the_scalar_one() {
        let dot = select_dot_q();
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for k in [1usize, 5, 15, 16, 17, 33, 255, 896, 4871] {
            let x: Vec<f32> = (0..k)
                .map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let w: Vec<i8> = (0..k).map(|_| (rnd() % 256) as u8 as i8).collect();
            let want = dot_q_scalar(&x, &w);
            let got = dot(&x, &w);
            let scale: f32 = x
                .iter()
                .zip(&w)
                .map(|(a, b)| (a * *b as f32).abs())
                .sum::<f32>()
                .max(1.0);
            assert!(
                (got - want).abs() <= 1e-5 * scale,
                "k={k}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn round_trip_error_is_bounded_by_half_a_step() {
        // Every element must land within half a quantization step of the original.
        let rows = 8;
        let cols = 64;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) / 100.0)
            .collect();
        let w = Tensor::new(data, vec![rows, cols]);

        let q = QTensor::quantize(&w);
        let back = q.dequantize();

        for r in 0..rows {
            let step = q.scales()[r];
            for c in 0..cols {
                let before = w.data()[r * cols + c];
                let after = back.data()[r * cols + c];
                assert!(
                    (before - after).abs() <= step * 0.5 + 1e-6,
                    "row {r} col {c}: {before} -> {after}, step {step}"
                );
            }
        }
    }

    #[test]
    fn scales_are_per_row_so_one_huge_row_cannot_crush_the_others() {
        // Row 0 is tiny, row 1 is enormous. With a single tensor-wide scale, row 0
        // would quantize to all zeros. Per-channel must preserve it.
        let mut data = vec![0.0f32; 8];
        for (c, slot) in data[..4].iter_mut().enumerate() {
            *slot = 0.001 * (c as f32 + 1.0);
        }
        for (c, slot) in data[4..].iter_mut().enumerate() {
            *slot = 1000.0 * (c as f32 + 1.0);
        }
        let w = Tensor::new(data, vec![2, 4]);
        let q = QTensor::quantize(&w);
        let back = q.dequantize();

        assert!(q.scales()[0] < q.scales()[1]);
        // The small row survives with meaningful precision.
        for c in 0..4 {
            let before = w.data()[c];
            let after = back.data()[c];
            let rel = (before - after).abs() / before.abs();
            assert!(rel < 0.01, "small row lost precision: {before} -> {after}");
        }
    }

    #[test]
    fn quantized_linear_approximates_the_f32_one() {
        let m = 3;
        let k = 32;
        let n = 5;
        let x = Tensor::new(
            (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect(),
            vec![m, k],
        );
        let w = Tensor::new(
            (0..n * k)
                .map(|i| ((i % 23) as f32 - 11.0) / 11.0)
                .collect(),
            vec![n, k],
        );

        let exact = linear(&x, &w, None);
        let q = QTensor::quantize(&w);
        let approx = linear_q(&x, &q, None);

        let err = exact.max_abs_diff(&approx);
        let magnitude = exact.data().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            err / magnitude < 0.02,
            "relative error {} too large (abs {err}, magnitude {magnitude})",
            err / magnitude
        );
    }

    #[test]
    fn all_zero_rows_do_not_divide_by_zero() {
        let w = Tensor::new(vec![0.0; 8], vec![2, 4]);
        let q = QTensor::quantize(&w);
        let back = q.dequantize();
        assert!(back.data().iter().all(|&v| v == 0.0));
        assert!(q.scales().iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn size_is_a_quarter_plus_scales() {
        let w = Tensor::new(vec![0.5; 896 * 896], vec![896, 896]);
        let q = QTensor::quantize(&w);
        let f32_bytes = 896 * 896 * 4;
        assert_eq!(q.size_bytes(), 896 * 896 + 896 * 4);
        // Just under 4× smaller — the scales are the overhead.
        let ratio = f32_bytes as f64 / q.size_bytes() as f64;
        assert!(ratio > 3.9 && ratio < 4.0, "ratio was {ratio}");
    }
}
