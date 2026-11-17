# Foundry — Rules

## Hard rules

1. **No ML libraries in `crates/`.** Not `candle`, not `burn`, not `ndarray`, not
   `tch`, not `ort`. If a dependency implements a tensor operation, it is out. File
   I/O, JSON parsing, memory mapping, CLI parsing, and error handling are fine.
2. **Correctness before speed, always.** Do not write a SIMD kernel for an operation
   whose scalar version has not passed parity. Every optimization must be preceded by
   a passing test and followed by the same test.
3. **Never quote a benchmark number that wasn't produced by `scripts/bench.sh`.** No
   numbers from memory, no numbers from a single run, no numbers in the README that
   aren't reproducible. See [BENCHMARKS.md](BENCHMARKS.md).
4. **Verify model config against `config.json`, not against this repo's docs.**
   [MODEL.md](MODEL.md) records expected values, but the file on disk wins.
   If they disagree, the doc is wrong — fix the doc.
5. **Constrained decoding correctness is binary.** 99.8% valid JSON is a failed
   implementation, not a good result. If output can be invalid, the DFA or the vocab
   index has a bug. Find it.
6. **Record decisions in [DECISIONS.md](DECISIONS.md)** when a non-obvious
   fork is taken. One short entry, including what was rejected.

## Code style

- Rust 2021, `rustfmt` defaults, `clippy` clean before commit.
- Prefer explicit shapes in names: `q_bhsd`, `k_bhsd` beats `q`, `k`.
- Every non-obvious tensor operation gets a comment giving the **shape before and
  after**. This is the single highest-value comment type in this codebase.
- Panics are acceptable in week 1–2 for shape mismatches. Convert to typed errors in
  week 4, not before — error plumbing early is a time sink.
- No `unsafe` without a comment explaining the invariant being upheld.
