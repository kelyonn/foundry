# Explainers

One file per part of the system. Each says what the module does, *why it is shaped that way*,
the shapes and sizes worked out by hand, how it was verified, and the questions an
interviewer is likely to ask. They are study material — they do not replace being able to
re-derive the answers unaided (see D21 in [../DECISIONS.md](../DECISIONS.md)).

| # | File | Covers |
|---|---|---|
| 01 | [loading-and-tokenizer](01-loading-and-tokenizer.md) | safetensors, bf16→f32, parameter count, byte-level BPE |
| 02 | [forward-pass](02-forward-pass.md) | RMSNorm, RoPE, GQA, SwiGLU, shapes, FLOPs |
| 03 | [kv-cache-and-sampling](03-kv-cache-and-sampling.md) | cache layout/size, prefill vs decode, sampler order |
| 04 | [int8-quantization](04-int8-quantization.md) | per-channel symmetric int8, why not 4× faster |
| 05 | [constrained-decoding](05-constrained-decoding.md) | schema → byte DFA → token mask |
| 06 | [testing-and-benchmarking](06-testing-and-benchmarking.md) | parity gate, benchmark rules, llama.cpp caveats |
| 07 | [matmul-performance](07-matmul-performance.md) | thread pool, tiling, NEON, measured effect |

Also: [WRITEUP](../WRITEUP.md) (what was hard, what went wrong), [GLOSSARY](../GLOSSARY.md),
[GRAMMAR](../GRAMMAR.md), [DECISIONS](../DECISIONS.md).
