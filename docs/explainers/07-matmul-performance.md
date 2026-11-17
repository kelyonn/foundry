# 07 — Making the matmul fast (and what that did and didn't buy)

Code: [`crates/tensor/src/pool.rs`](../../crates/tensor/src/pool.rs),
[`ops.rs`](../../crates/tensor/src/ops.rs) (`par_linear`, `dot_block`, `dot_neon`),
[`quant.rs`](../../crates/tensor/src/quant.rs). Decisions: D16, D22, D23.

## Where the time goes

Per token the model does ≈ 2 × 494 M = **0.99 GFLOP**, almost all of it matrix×vector
(decode) or matrix×matrix (prefill) with the weights as the left operand. Everything else
(norms, RoPE, softmax) is rounding error next to that, except attention over a long history.

Two regimes:
- **Decode** (seq = 1): each weight is read once and used once → ~0.5 FLOP/byte. Speed is set
  by *bytes moved*. f32 streams ≈ 1.98 GB/token, int8 ≈ 0.5 GB/token.
- **Prefill** (seq = N): each weight is reused across N tokens → compute-bound *if* the
  loop order lets the weight stay in cache.

## Three changes, in the order they were made

**1. Persistent thread pool (D22).** The first version split output columns across cores with
`thread::scope`, i.e. it spawned OS threads on every matmul — ~170 calls per token. For a
decode step with little work per call that spawn tax is a fixed cost. The pool starts workers
once; a call is a condvar wake-up. The caller also works, calls are serialised, and a task
panic is re-raised after every task finishes (otherwise the lifetime-erased closure pointer
could dangle — that pointer is the single `unsafe` and its invariant is written at `Job`).
Splitting by **columns** (`n`), not rows, because decode has m = 1.

**2. Tiled loop order (D23).** Old: `for token { for weight_row { dot } }` — re-streams the
whole weight chunk from memory for every token. New: `for 64-token tile { for weight_row {
for token in tile { dot } } }`. A weight row (≈3.5 KB f32) stays in L1; a 64-token tile of
activations (64 × 896 × 4 B ≈ 229 KB) stays in L2. For a 512-token prefill the weights are
read 8 times instead of 512.

**3. NEON dot kernels (D23).** 4 vector accumulators × 4 lanes = 16 partial sums, 16 elements
per loop iteration (`vfmaq_f32`). The f32 kernel accumulates in f32 inside 256-element
blocks and flushes each block into an f64 total — rounding error grows with the number of
terms an accumulator has absorbed, so bounding that number keeps it near the f64 kernel's
(parity 1.78e-6 vs 1.48e-6). The int8 kernel widens i8→i16→i32→f32 before the FMA.
`FOUNDRY_SCALAR=1` selects the old f64 scalar kernel (kept as the test oracle).

## Measured (same binary; median of 10; 512 → 128 tokens)

| | scalar | NEON |
|---|---|---|
| f32 prefill / decode | 32.0 / 20.3 | 95.8 / 21.0 |
| int8 prefill / decode | 25.5 / 13.4 | 96.2 / 40.8 |

**Read the table as a physics lesson:**
- f32 **decode did not speed up**. Implied weight traffic is 1.98 GB × 21 tok/s ≈ 41 GB/s;
  the kernel was never the bottleneck, memory was. A faster dot product on a starved core is
  a no-op. (The 1.04× is a negative result and it is *why* int8 exists.)
- int8 decode **tripled**: a quarter of the bytes, and the scalar kernel's int→float
  conversion work is now vectorised.
- Prefill is now flat at ~96 tok/s regardless of precision. When two very different matmul
  kernels give the same number, the matmul isn't the limit any more. Attention is still
  scalar, per-head, O(seq²) with per-position allocation — the next place to look, but that
  is a hypothesis; **it was not profiled**.

## Why it is still 4–8× behind llama.cpp on prefill

It uses Accelerate BLAS (AMX) for prompt processing; we may not (hard rule 1), and our kernel
is a single-instruction-stream NEON dot, not a register-blocked matrix-multiply microkernel
(no outer-product tiling, no `i8mm`/`sdot`). Decode — where BLAS doesn't help — is within
1.3× (f32) and 2.1× (int8).

## Questions

1. *Why split output columns across threads?* m = 1 at decode; n is thousands.
2. *Why is f32 decode unchanged after SIMD?* Bandwidth-bound; compute wasn't the limit.
3. *What does the tile size trade?* Too small → weights re-streamed; too big → activation tile falls out of L2.
4. *Why flush f32 blocks into f64?* Bounds accumulation error without paying f64 FMA throughput.
5. *Why is the SIMD result not bit-identical, and how do you know it's right?* Different summation order; parity gate (relative 5e-5) + kernel-vs-oracle unit tests on ragged lengths.
6. *Compute:* bytes streamed per f32 decode token ≈ 494,032,768 × 4 = 1.976 GB; at 21 tok/s ≈ 41.5 GB/s.
