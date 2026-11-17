# Foundry

**An LLM inference engine written from scratch, with grammar-constrained decoding.**

No PyTorch. No `transformers`. No `llama.cpp`. No ONNX. Weights are read off disk,
the transformer forward pass is implemented by hand, and the output is constrained
by a state machine compiled from a grammar — so malformed output is not unlikely,
it is *impossible*.

> **Status:** weeks 1–3 done, week 4 (writeup, llama.cpp comparison) is not.
>
> **Working, end to end:** `cargo run --release -p cli -- --chat --prompt "..." \
> --schema '...'` loads the real Qwen2.5-0.5B-Instruct checkpoint, runs the
> from-scratch forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU, a
> KV cache), and masks every logit through `crates/grammar` before sampling —
> so the output is provably valid JSON, from the real model, not a stand-in.
> `--int8` quantizes weights to a quarter of the memory. 95 tests, clippy clean.
>
> **Verified, not just built:** logit parity vs. HF at 1.48e-6 relative error;
> tokenizer ID parity exact, 42/42; incremental KV-cache decode bit-identical to
> a fresh forward pass; 500/500 constrained generations parse against the real
> 151,665-token vocabulary; int8 measured at 3.98× smaller weights for +0.10%
> perplexity on a real wikitext-2 slice. A systematic bug hunt across all five
> crates found and fixed six real defects, including a reachable panic and a
> one-line JSON Schema that could stall the process for minutes — see
> [docs/DECISIONS.md](docs/DECISIONS.md) D14–D20.
>
> **Not done:** the llama.cpp comparison (not installed here) and
> `docs/WRITEUP.md`. See [docs/ROADMAP.md](docs/ROADMAP.md).

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

## Results

Produced by `./scripts/bench.sh` (throughput, RSS) and
`cargo run --release -p cli --bin perplexity` (quality delta). Machine: Apple M5,
10 cores, 16 GB, macOS. n=10 timed runs, 3 warmup discarded, 512-token prompt →
128 generated, per docs/BENCHMARKS.md. Re-run these yourself — don't take them on
faith, and don't quote them from a dirty working tree, which this run was (see
the script's own output).

| Metric | Foundry f32 | Foundry int8 | Baseline | Notes |
|---|---|---|---|---|
| Logit parity vs. HF (relative) | 1.48e-6 | — | — | target `< 5e-5`; see D14–D18 for why relative, not the unsatisfiable absolute `1e-4` first written down |
| Prefill throughput (tok/s, median) | 20.61 | 17.12 | — (no llama.cpp here) | int8 is *slower* to prefill — see note below |
| Decode throughput (tok/s, median) | 10.36 | 11.78 | — | |
| Peak RSS | 2779.0 MB | 2320.9 MB | — | |
| Weights resident | 1884.6 MB | 473.1 MB | — | 3.98× |
| KV cache @ 4096 ctx | 96.0 MB | 96.0 MB | — | unaffected by weight precision |
| Perplexity delta, fp32 → int8 | — | +0.10% | — | real wikitext-2 test split, 767 tokens — see note below |
| Valid JSON, constrained | 100% (500/500) | — | — | real 151,665-token Qwen vocab, hostile random logits |
| Valid JSON, prompted only | — | — | — | not measured — needs a real-model unconstrained run, which hasn't been done |
| Masking overhead (tok/s) | — | — | — | not measured with `bench.sh` discipline; single ad hoc runs suggested a real but unquantified cost |
| llama.cpp comparison | — | — | not installed | the empty column above; do not guess it |

**int8 prefill is slower than f32, decode is faster — measured, not a typo.**
Dequantizing on the fly (`w_row[t] as f32`) adds a per-element cast the f32 path
doesn't pay, which shows up most during prefill's large batched matmuls. During
decode (`m=1`), the smaller weight footprint (473 MB vs 1885 MB) wins on cache
locality instead. Both effects are real; neither dominates uniformly, which is
why quoting "int8 is faster" without saying *which phase* would be wrong.

**The perplexity sample needs to be at least ~768 tokens or the delta's *sign*
isn't trustworthy.** Measured directly: at 256 tokens the delta read −0.17%
(int8 looked *better* than f32); at 512, +0.11%; at 768, +0.10%, matching the
trend from a longer partial run. Below ~768 tokens the number is sampling noise
around a true effect close to zero, not a reliable read of direction. Regenerate
the corpus with `python scripts/fetch_wikitext2.py` (needs `pip install
datasets`); `tests/fixtures/wikitext2_test.txt` is committed so this reproduces
with no network access.

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

---

## Quickstart

Needs `models/qwen2.5-0.5b/{config.json,model.safetensors,tokenizer.json}` — see
[MODEL.md](docs/MODEL.md) for where to get them. Not committed (`.gitignore`);
~1 GB.

```
cargo test --release                # 95 tests across the workspace
cargo run --release -p cli --bin parity   # logit parity vs. HF, layer by layer

# plain generation
cargo run --release -p cli -- --prompt "The capital of France is" --max-tokens 20

# constrained: output is provably valid JSON, from the real model
cargo run --release -p cli -- --chat --prompt "Give me a person: name, age, city." \
  --schema '{"type":"object","properties":{"name":{"type":"string","maxLength":16},"age":{"type":"integer"},"city":{"type":"string","maxLength":16}}}'

cargo run --release -p cli --bin perplexity   # fp32 -> int8 quality delta
./scripts/bench.sh                            # the numbers in the table above
```

No checkpoint on hand? The constrained decoder needs none — it runs against a
built-in synthetic vocabulary or a downloaded `tokenizer.json` alone:

```
cargo test -p grammar
cargo run -p grammar --bin demo -- --tokenizer models/qwen2.5-0.5b/tokenizer.json
```

---

## License

MIT
