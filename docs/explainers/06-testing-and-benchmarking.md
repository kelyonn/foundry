# 06 — How correctness and speed are established here

Code: [`crates/model/src/bin/parity.rs`](../../crates/model/src/bin/parity.rs),
[`cli/src/bin/bench.rs`](../../crates/cli/src/bin/bench.rs), `perplexity.rs`,
`json_compare.rs`, [`scripts/bench.sh`](../../scripts/bench.sh),
[TESTING.md](../TESTING.md), [BENCHMARKS.md](../BENCHMARKS.md).

## Correctness is a gate, not a benchmark

1. **Reference.** `scripts/dump_reference.py` runs the HuggingFace model in **float64**
   with forward hooks and saves every layer's activations and the final logits as `.npy`.
   f64 on purpose: a bf16/f32 reference has its own rounding noise, so you could not tell
   *our* error from *its* error.
2. **Parity bin.** Runs our forward pass on the same tokens and compares all 24 layers'
   intermediates plus logits. Metric is **max-abs for diagnosis, relative for the pass/fail
   gate** (5e-5): the first, absolute `1e-4` tolerance was unsatisfiable because logits reach
   magnitude ~19, where one f32 ulp is 2^-19 ≈ 1.9e-6 (the parity bin reports its error in
   ulps: 2.86e-5 absolute = 15 ulps) (D14). A mean metric is never used —
   it hides one catastrophically wrong element among thousands.
3. **Localisation.** Because every layer is compared, a divergence is pinned to a layer and
   an op ("first divergence at layer 3 `rope_q`") instead of "logits are wrong". Five real
   bugs were found this way (D14–D18).
4. **Every optimisation re-runs it.** Thread pool: still 1.482e-6 (bit-identical).
   NEON: 1.78e-6, within the gate, plus unit tests of the kernel against the scalar f64
   oracle on ragged lengths (1, 15, 16, 17, 255, 256, 257, 4871…).

## Speed: the rules in `bench.rs`

- discard **3 warm-up** runs; take **n ≥ 10** (the binary asserts it);
- report **median and p95** (nearest-rank; for n = 10 the "p95" is the worst run — said out
  loud in the output) — never the mean or the best run;
- **prefill and decode separately** — different bottleneck (compute vs memory bandwidth);
- peak RSS from `getrusage`, weights-resident from the actual allocation;
- `bench.sh` prints rustc version and git state and warns **DIRTY WORKING TREE** — numbers
  from a dirty tree are not reproducible.

**Hard rule 3:** no number in the README that `scripts/bench.sh` did not print.

## What a speedup claim needs

Same binary, same harness, one switch. `FOUNDRY_SCALAR=1` forces the scalar f64 kernel so the
NEON number and the scalar number come from one build (D23). Without that you are comparing
two different builds on a machine whose clock and background load drift.

## Comparing with llama.cpp honestly

`llama-bench` is a *reference point*, not a like-for-like race:
- different quantization (Q8_0 = blocks of 32 with a scale each, vs our one scale per row);
- the Homebrew build links Apple's Accelerate BLAS for prompt processing, which our
  hand-written kernels are not allowed to use (hard rule 1);
- GGUF files are the official Qwen conversions, not produced by this repo.
Report the gap and the reasons. "Within Nx of llama.cpp with a hand-written NEON kernel and
no BLAS" is a defensible result; a claim of parity would not be.

## Questions

1. *Why compare against an f64 reference?* To separate our rounding from the reference's.
2. *Why relative tolerance?* Logit magnitudes up to ~19: absolute error scales with magnitude, relative doesn't.
3. *Why median, not mean?* Outlier runs (thermal, OS) skew a mean; median is robust. p95 shows the tail.
4. *Why measure prefill and decode separately?* Compute-bound vs bandwidth-bound; one blended number hides both.
5. *A teammate says "NEON gave 3×". What do you ask?* Which phase, which precision, same harness, how many runs, clean tree?
