# Benchmarks

**Read this before producing a single number.** Methodology decided after the fact is
how you end up with a results table you can't defend.

## Rules

1. **Every number in the README comes from `scripts/bench.sh`.** No numbers typed from
   memory, no numbers from a single run, no numbers from a different machine.
2. **Report median and p95 over n ≥ 10 runs.** A mean over three runs is noise.
3. **Warm up.** Discard the first 3 runs. Page cache and CPU frequency scaling make
   cold runs meaningless.
4. **State the machine.** Model, chip, core count, RAM, OS version. Every time.
5. **Prefill and decode are separate numbers.** They are bound by different resources
   (compute vs. memory bandwidth). A single "tok/s" hides this and an interviewer will
   ask.
6. **The baseline must be fair.** Same model, same quantization, same context length,
   same machine, same thread count. If you can't make it fair, say what's unfair.

## Environment template

Record with every result set:

```
Machine:     <model, chip, cores, RAM>
OS:          <version>
Rust:        <rustc --version>, --release, target-cpu=native
Model:       Qwen2.5-0.5B-Instruct
Precision:   <f32 | int8>
Threads:     <n>
Context:     <prompt tokens> → <generated tokens>
Baseline:    llama.cpp <commit>, <quant>, <threads>
Date:        <date>
```

## What gets measured

### Correctness (week 1, gate for everything else)

| Metric | Target |
|---|---|
| Max abs error vs. HF logits | `< 1e-4` |
| Max abs error, per layer | `< 1e-4`, no step changes |
| Tokenizer ID parity | exact |

Correctness is not a benchmark, it's a gate. If it fails, no other number on this page
means anything.

### Throughput (week 2)

| Metric | Foundry f32 | Foundry int8 | llama.cpp |
|---|---|---|---|
| Prefill (tok/s), 512-token prompt | — | — | — |
| Decode (tok/s), 128 tokens | — | — | — |
| Time to first token (ms) | — | — | — |

### Memory (week 2)

| Metric | f32 | int8 |
|---|---|---|
| Weights on disk | — | — |
| Peak RSS, 512 ctx | — | — |
| Peak RSS, 4096 ctx | — | — |
| KV cache @ 4096 | — | — |

Report the KV cache separately from the weights. At long context it exceeds the int8
weights, which is the whole reason grouped-query attention exists — and it's a good
number to have at your fingertips.

### Quantization quality (week 2, cuttable)

| Metric | f32 | int8 |
|---|---|---|
| Perplexity, wikitext-2 subset | — | — |

If cut for time, say "I measured the memory saving but not the quality cost" rather
than implying the cost is zero.

### Constrained decoding (week 3–4)

| Metric | Constrained | Prompted only |
|---|---|---|
| Valid JSON rate (n=500) | — (must be 100%) | — |
| Schema-valid rate | — | — |
| Decode throughput (tok/s) | — | — |
| Index build, first visit (ms) | — | n/a |
| Index build, cached (ms) | — | n/a |
| States visited / total | — | n/a |

**The prompted-only column is not optional.** Without it, "100% valid JSON" is an
unfalsifiable claim.

## Honesty rules

These exist because the temptation to shade numbers is real and getting caught is fatal.

- **If llama.cpp is faster, report by how much.** It almost certainly will be — it has
  years of hand-tuned kernels and you have four weeks. "Within 3× of llama.cpp with a
  hand-written scalar matmul" is a genuinely good result and an honest one. A fudged
  win is a career-limiting move.
- **Never compare your int8 against their f32**, or your 8 threads against their 1.
- **Report negative results.** If SIMD gave 1.2× where you expected 4×, that's a real
  finding about memory-bandwidth limits and it's more interesting than the win.
- **If a number is from one run, label it.** Better: don't publish it.

## `scripts/bench.sh`

Written in week 4. Must:

- print the environment block above, gathered automatically
- run each configuration n ≥ 13 times, discard the first 3
- emit median and p95
- write a markdown table straight into `docs/BENCHMARKS.md`, so the published numbers
  are literally generated, never typed

## Results

Generated from a clean-tree run of `scripts/bench.sh` on 2026-11-17 (git `b2b3a82`),
Apple M5, 10 cores, 16 GB, Darwin 27.2.0, rustc 1.97.1 `--release`, 10 threads,
512-token prompt → 128 generated, 10 timed runs after 3 warm-ups. Raw output is not
committed (`bench-results/` is gitignored); rerun the script to regenerate it. The
README table is the same data. Numbers vary a few percent run to run, and llama-bench's
own spread is large (see ± below), so differences under ~10% are not claims.

### Throughput (medians)

| Metric | Foundry f32 | Foundry int8 | llama.cpp F16 | llama.cpp Q8_0 |
|---|---|---|---|---|
| Prefill (tok/s), 512-token prompt | 95.8 (p95 97.4) | 96.2 (p95 96.9) | 362 ± 45 | 771 ± 45 |
| Decode (tok/s), 128 tokens | 21.0 (p95 21.6) | 40.8 (p95 43.0) | 26.8 ± 2.3 | 87.2 ± 2.0 |
| Time to first token (ms) | 5,346 | 5,367 | — | — |

llama.cpp: Homebrew build (backend string `BLAS,MTL`), official Qwen2.5-0.5B-Instruct
GGUFs, `-ngl 0 -t 10 -r 10`, mean ± sd as `llama-bench` reports. **Not like-for-like:**
Accelerate BLAS handles its prompt processing; Q8_0 is per-32-block scales vs our
per-row; F16 vs our f32 weights (half the bytes). The honest summary: ~3.8× behind
(f32/F16) and ~8× behind (int8/Q8_0) on prefill, ~1.3× and ~2.1× behind on decode.

### Kernel A/B (same binary, `FOUNDRY_SCALAR=1`)

| | scalar | NEON |
|---|---|---|
| f32 prefill / decode | 32.0 / 20.3 | 95.8 / 21.0 |
| int8 prefill / decode | 25.5 / 13.4 | 96.2 / 40.8 |

Negative result worth keeping: f32 decode did not improve (1.04×) — it streams ~2 GB of
weights per token and is bound by that, not by the dot product. int8 decode improved 3×
because it moves a quarter of the bytes. Prefill is now flat at ~96 tok/s for both,
suggesting the remaining cost is elsewhere (attention is scalar, O(seq²); unprofiled).

### Memory

| Metric | f32 | int8 |
|---|---|---|
| Weights resident | 1,884.6 MB | 473.1 MB (3.98×) |
| Peak RSS (512 prompt + 128 gen) | 3,555 MB | 2,387 MB |
| KV cache @ 4096 (arithmetic) | 96.0 MB | 96.0 MB |

Peak RSS includes the mmapped 988 MB checkpoint and, for int8, the transient f32 tensors
created while quantizing.

### Quantization quality

wikitext-2 test slice, 767 scored tokens: f32 perplexity 9.2710, int8 9.2798, **+0.10%**
(`cargo run --release -p cli --bin perplexity`).

### Constrained decoding (`json_compare`, 100 generations per mode, temperature 0.7)

| Metric | Constrained | Prompted only |
|---|---|---|
| Parses as JSON (strict), f32 | 100 / 100 | 55 / 100 |
| Parses after stripping a ``` fence | 100 / 100 | 98 / 100 |
| Conforms to schema (strict) | 100 / 100 | 37 / 100 |
| Conforms after stripping a fence | 100 / 100 | 76 / 100 |
| DFA mask + advance (ms/token, median) | 0.19 | 0 (padding mask only) |
| Decode tok/s (median per generation) | 26.0 | 23.8 |
| Vocab-index mask builds / cache hits | 153 / 1,626 | n/a |

int8 run: strict parse 100 vs 53, strict conformance 100 vs 38. The tok/s gap is a
length confound (constrained outputs are shorter), not a speedup. The grammar crate's own
sampler-free test (500 generations over rotating schemas) is separate and also 500 / 500.

**Not measured:** Metal/GPU; longer prompts; multiple machines; the grammar index cold-build
time per state (only the aggregate hit/build counts are reported).
