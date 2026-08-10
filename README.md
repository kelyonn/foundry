# Foundry

**An LLM inference engine written from scratch, with grammar-constrained decoding.**

No PyTorch. No `transformers`. No `llama.cpp`. No ONNX. Weights are read off disk,
the transformer forward pass is implemented by hand, and the output is constrained
by a state machine compiled from a grammar — so malformed output is not unlikely,
it is *impossible*.

> **Status:** pre-week-1. Nothing is built yet. See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## The two halves

**1. The engine.** Load `safetensors` weights directly, implement byte-level BPE
tokenization, the Qwen2 forward pass (RMSNorm, RoPE, grouped-query attention with a
KV cache, SwiGLU MLP), sampling, and int8 weight quantization. The only dependencies
are for file I/O and JSON parsing — every tensor operation is ours.

**2. Constrained decoding.** Compile a JSON Schema or EBNF grammar into a DFA over
*bytes*, intersect that DFA with the tokenizer vocabulary to precompute the legal
token set at each state, and mask the logits before sampling. The model can only emit
tokens that keep the output on a path to a valid parse.

The second half is the interesting one. "Ask the model for JSON and retry on failure"
is a prompt. This is a proof.

---

## Why it's built this way

Every stage of this project has an **objective oracle**:

| Stage | The thing that says you're wrong |
|---|---|
| Forward pass | Logits must match HuggingFace reference within `1e-4` |
| Each layer | Per-layer activations must match reference dumps |
| Tokenizer | Round-trip and ID-parity against `tokenizer.json` |
| Quantization | Perplexity delta on a held-out corpus |
| Constrained decoding | 100% of generations must parse. Not 99%. |

There is no stage where "it looks about right" is an acceptable answer. That is a
deliberate design property of the project, not a coincidence — see
[docs/TESTING.md](docs/TESTING.md).

---

## How it's actually built

Claude writes the code, module by module. Before each module is written, Kalyan
explains its logic out loud from having learned it. After it's written and its tests
pass, he's quizzed closed-book — recall, "why not the alternative," and at least one
question requiring him to compute a shape or a size. A module isn't done until the
quiz passes, regardless of whether the code already does. Every quiz result is logged
in [docs/QUIZ_LOG.md](docs/QUIZ_LOG.md). Full rationale in
[docs/DECISIONS.md](docs/DECISIONS.md) D10.

---

## Results

Filled in during week 4. Numbers here must be reproducible via `scripts/bench.sh`.

| Metric | Foundry | Baseline | Notes |
|---|---|---|---|
| Logit parity vs. HF (max abs err) | — | — | target `< 1e-4` |
| Prefill throughput (tok/s) | — | llama.cpp | same model, same machine |
| Decode throughput (tok/s) | — | llama.cpp | |
| Peak RSS, fp32 | — | — | |
| Peak RSS, int8 | — | — | |
| Perplexity delta, fp32 → int8 | — | — | wikitext-2 subset |
| Valid JSON, constrained | — | — | target 100% |
| Valid JSON, prompted only | — | — | the comparison that matters |
| Masking overhead (tok/s) | — | — | cost of the guarantee |

---

## Repo layout

```
foundry/
├── crates/
│   ├── tensor/       hand-written tensor ops, SIMD kernels, quantization
│   ├── tokenizer/    byte-level BPE encode/decode
│   ├── model/        Qwen2 forward pass, KV cache, sampling
│   ├── grammar/      schema/EBNF → DFA → vocabulary index → logit mask
│   └── cli/          the `foundry` binary
├── docs/
├── scripts/          reference dumping, benchmarks (Python allowed here only)
└── tests/            parity fixtures and integration tests
```

Python is permitted in `scripts/` — that is where reference activations are dumped
*from* HuggingFace so we can check ourselves *against* it. Python never appears in
the inference path.

---

## Docs

| Doc | What's in it |
|---|---|
| [DAY_ONE.md](docs/DAY_ONE.md) | **Start here.** The first session, step by step |
| [ROADMAP.md](docs/ROADMAP.md) | Four weeks, with a hard checkpoint each Sunday |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crate boundaries, tensor layout, KV cache design |
| [MODEL.md](docs/MODEL.md) | Qwen2.5-0.5B spec, weight names, forward pass |
| [GRAMMAR.md](docs/GRAMMAR.md) | Constrained decoding: the DFA, the vocab index, the hard parts |
| [TESTING.md](docs/TESTING.md) | Oracles, parity harness, how to debug a wrong layer |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measurement methodology — read before quoting a number |
| [DECISIONS.md](docs/DECISIONS.md) | Why Rust, why this model, why byte-level DFAs |
| [GLOSSARY.md](docs/GLOSSARY.md) | Every term in this repo, defined |
| [QUIZ_LOG.md](docs/QUIZ_LOG.md) | Per-module understanding checks — the real interview-readiness record |

---

## Quickstart

Not yet. This section gets written when `foundry run` produces a token.

---

## License

MIT
