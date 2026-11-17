# 04 — int8 weight quantization

Code: [`crates/tensor/src/quant.rs`](../../crates/tensor/src/quant.rs),
`LinearWeight` in [`weights.rs`](../../crates/model/src/weights.rs).

## What it does

Each `[out, in]` weight matrix is stored as `i8` plus **one f32 scale per output row**:

```
scale_r = max|row_r| / 127          q = round(w / scale_r)  ∈ [-127, 127]
dequant:  w ≈ q · scale_r
```

- **Symmetric** (no zero-point): zero maps to zero exactly; trained transformer weights are
  roughly zero-centred so an asymmetric zero-point buys little and costs an add per element.
- **127, not 128:** keeps the grid symmetric (−max and +max equally representable).
- **Per-channel (per row), not per-tensor:** one outlier row would otherwise force a huge
  scale on everything and crush the other rows toward 0. A test pins this
  (`scales_are_per_row_so_one_huge_row_cannot_crush_the_others`).
- **Weight-only.** Activations stay f32. Memory shrinks ~4×; arithmetic does *not* get
  cheaper, because we widen i8→f32 to multiply.

## The trick that makes dequant cheap

`Σ_t x[t]·(q[t]·s) = s · Σ_t x[t]·q[t]` — `s` is constant along the row, so it is applied
**once per output element**, after the dot product, not once per weight.

## Size, computed

Per `[896, 896]` matrix: f32 = 3,211,264 B; int8 = 802,816 B + 896 × 4 B scales =
806,400 B → **3.98×** (not 4×, the scales cost 0.4%). Whole model measured: 1,884.6 MiB
→ 473.1 MiB.

## What the NEON kernel does (and why int8 doesn't give 4× speed)

Per 16 weights: load 16×i8 → sign-extend i8→i16 → i32 → convert to f32 (4 vectors), then
4 FMAs against 16 f32 activations. The widening/convert instructions are the price of
weight-only quantization. The win is **bandwidth**: a decode step streams ~¼ of the bytes.
That is why int8 decode is much faster than f32 decode (memory-bound) while int8
*prefill* is not faster (compute-bound, with extra conversions) — see the README's
results and `docs/BENCHMARKS.md`. Honest statement: "int8 is faster" is only true if you say
which phase.

A real next step (rejected for now, D23): quantize *activations* too and use the integer
dot-product instructions (`sdot`). That changes numerics beyond weight-only int8, so it
would need its own perplexity gate.

## Accuracy

Measured with `cargo run --release -p cli --bin perplexity` on a committed wikitext-2
slice. The sign of the delta flipped with sample size (D20: −0.17% at 256 tokens, +0.1% at
768), which is why the default is 768 tokens — below that the number is noise. The
layer-by-layer divergence from an f64 oracle is analysed in D18.

## Questions

1. *Why per-channel?* Outlier rows. *Why per output channel and not per input?* The scale must factor out of the dot product, so it must be constant along the reduction axis `k`.
2. *What if I'd used 128 as the divisor?* −128 would exist but +128 wouldn't be representable (i8 max 127) → asymmetric clipping error on positive max.
3. *Why isn't int8 4× faster?* Weight-only: still f32 math, plus conversions; only memory traffic shrinks.
4. *Compute:* int8 bytes for `gate_proj` (4864×896)? 4,358,144 + 4864×4 = 4,377,600 B.
5. *Where does f32 accumulation suffice here but not in the f32 path?* int8 error (~1e-3 relative) swamps f32 rounding, so f64 accumulation would buy precision the input doesn't have.
