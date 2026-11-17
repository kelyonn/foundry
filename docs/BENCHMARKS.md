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

*(empty until week 2 — do not populate by hand)*
