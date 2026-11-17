# Writeup: building Foundry

What was built, what was hard, what went wrong, and what I'd do differently.
Numbers are from `scripts/bench.sh` on one Apple M5 (see [BENCHMARKS.md](BENCHMARKS.md));
decision numbers (D#) point into [DECISIONS.md](DECISIONS.md).

## What it is

An LLM inference engine in Rust with no ML libraries, running Qwen2.5-0.5B-Instruct on CPU,
plus a grammar-constrained decoder that compiles a JSON Schema into a byte-level DFA and
masks logits so output is valid by construction.

The result in one paragraph: logits match an f64 HuggingFace reference to 1.78e-6 relative;
int8 weights are 3.98× smaller for +0.10% perplexity; 100 of 100 constrained generations on
the real model conform to their schema versus 37 of 100 for the same model prompted only,
at 0.19 ms/token of masking. It is 4–8× slower than llama.cpp on prefill and 1.3–2.1× on decode.

## The decision that shaped the project: the roadmap was inverted (D11)

The plan was engine first, grammar third. An interview arrived while the repo was still at
week 0, so the grammar crate was built first and made standalone: its whole interface to a
model is `&mut [f32]` plus a token-id→bytes table. It turned out to be a better design than
the deadline compromise it started as — the grammar can be tested with no weights and no
numerics, which is why it is trustworthy.

## What was hard

**Parity is a debugging discipline, not a test.** An LLM that is 1% wrong still writes fluent
English, so "the output looks right" is worthless. Dumping every layer from an f64 reference
and comparing each activation turns "the logits are wrong" into "layer 3 `rope_q` diverges".
RoPE pairing (rotate-half vs adjacent), the GQA head mapping (`h / 7`, not `h % 2`) and the
KV-cache position offset are all mistakes that stay fluent — only a numeric oracle sees them.

**Tokens are bytes.** Byte-level BPE means a token can be half a UTF-8 character. A
character-level DFA is wrong in a way that only appears on non-ASCII input, so the automaton
is over bytes (D4) with states for partial sequences. Finding that `"\uDafe"` — a lone
surrogate escape — passed the grammar but failed `serde_json` (1 in ~55 generations) is the
kind of bug only a real parser in the loop finds (D15).

**What a DFA cannot say.** `minimum`/`maximum` need counting; recursion is context-free;
"these keys, each once, in any order" is n! alternations. Each was a decision about what to
promise instead (D12–D14): the grammar guarantees *shape*, not value bounds, and says so.

## What went wrong, including negative results

1. **The f32 accumulation trap (D17).** First matmul summed in f32, "to match the reference".
   Measured error on `q_proj` was 1.3e-4 — most of the budget in the first layer. The reference
   (BLAS) doesn't sum sequentially either; the way to agree with it is to be *more* accurate.
   f64 accumulation fixed it, at a throughput cost that SIMD later had to win back.
2. **The perplexity sign flip (D20).** At 256 tokens, int8 appeared to *improve* the model
   (−0.17%). At 768 it was +0.10%. A negative delta has no mechanism, so the measurement was
   noise; the default sample size was raised to the smallest value where the sign stopped moving.
3. **A one-line schema DoS (D19).** `maxLength: 4096` compiled to 192,515 states in 34.5 s and
   ~197 MB. Subset construction is superlinear; user-supplied bounds now have a ceiling.
4. **A panic reachable from the CLI** at high temperature (padding-tail logits with no token),
   a tokenizer loop that could hang on an empty special token, and silent flag defaults
   (`--max-tokens abc` ran with 32) — found by an adversarial pass, fixed with regression tests
   (D19, D24).
5. **The thread-spawn tax (D16 → D22).** The first threaded matmul spawned OS threads per call.
   Fine for prefill, a fixed cost for decode. Replaced by a persistent pool; output unchanged.
6. **SIMD helped less than hoped, and in a specific way (D23).** Same binary, scalar vs NEON:
   f32 prefill 32→96 tok/s (3.0×), int8 decode 13.4→40.8 (3.0×), but **f32 decode 20.3→21.0
   (1.04×)**. Decode streams ~2 GB of weights per token; at 21 tok/s that is ~41 GB/s and a
   faster dot product has nothing to speed up. And prefill now plateaus near 96 tok/s for
   both precisions, which means the matmul is no longer the limit — attention is the suspect
   (scalar, O(seq²)), but I did not profile it, so that stays a hypothesis.
7. **"Constrained is faster" is a confound.** `json_compare` shows constrained decode tok/s
   slightly above prompted-only (26.0 vs 23.8). That is output length and context size, not a
   speedup from masking. The honest overhead is 0.19 ms/token against a ~47 ms step.
8. **llama.cpp is not a clean comparison.** The Homebrew build links Accelerate BLAS for
   prompt processing, which this project forbids itself, and its Q8_0 format differs from mine.
   The prefill gap (3.8×/8×) is partly that. Decode (1.3×/2.1× behind) is the fairer number.

## What I'd do differently

- Profile before guessing. The prefill plateau is the obvious next target and I'd start with
  a profiler, not with the attention code.
- Write the prompted-only baseline *first*. It makes "100% valid" meaningful, and it showed
  that even stripping markdown fences leaves a quarter of prompted outputs off-schema.
- Make a register-blocked matmul microkernel (outer-product tiling, `sdot`/`i8mm`) rather
  than a dot-product kernel, and quantize activations — a separately-gated numerics change.
- Batch the attention heads and avoid the per-position allocation in decode.
- Add a second model (Llama-3.2-1B) only after the above; one model done properly beat two.

## What is not claimed

Value-bounded schemas (`minimum`, `maximum`); recursive schemas; EBNF input; GPU anything;
numbers from more than one machine; any comparison beyond the stated llama.cpp caveats.

## Reproduce

```bash
cargo test --release            # 103 tests
./scripts/bench.sh              # every number above except perplexity
cargo run --release -p cli --bin perplexity
```
