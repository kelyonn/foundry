# Roadmap

Four weeks of *scope*, in parallel with Cerberus. Each week ends in a **binary
checkpoint** — something that either passes or doesn't. No "mostly working." Scope is
unchanged from the original plan: full v1, including quantization, SIMD, and grammar-
constrained decoding. Nothing below has been cut.

If a checkpoint slips past Tuesday of the following week, cut from the **Cut list** at
the bottom rather than extending the week. Shipping four weeks of scope in five weeks
is fine; shipping three weeks of scope in four is not.

**Build mechanics changed (see [DECISIONS.md](DECISIONS.md) D10) — the checklist items
below did not.** Claude writes each item's code; before it's written, Kalyan explains
its logic out loud, and after it's written and tested, he's quiz-gated on it (log in
[QUIZ_LOG.md](QUIZ_LOG.md)). Treat each `[ ]` below as "explained → coded → tested →
quizzed," not just "coded." A checkpoint isn't real until every item under it has
cleared its quiz, not just its test.

---

## Week 1 — It runs and it's correct

**Checkpoint: `foundry run --prompt "The capital of France is"` produces logits that
match HuggingFace within `1e-4` max absolute error, on all 24 layers.**

- [ ] Download `Qwen2.5-0.5B-Instruct`, dump `config.json`, confirm every value in
      [MODEL.md](MODEL.md). Fix the doc where it's wrong.
- [ ] `scripts/dump_reference.py` — run the HF model with forward hooks, save
      per-layer activations and final logits as `.npy` for a fixed prompt set.
- [ ] Parse the safetensors container (JSON header + raw tensor blob, memory-mapped).
- [ ] Byte-level BPE tokenizer: load merges and vocab from `tokenizer.json`, encode,
      decode, round-trip. Parity-check IDs against the reference dump.
- [ ] Tensor primitives, scalar only: matmul, RMSNorm, softmax, SiLU, elementwise.
- [ ] Qwen2 forward pass, single token, no cache. RoPE, GQA, SwiGLU, tied embeddings.
- [ ] Layer-by-layer parity harness (see [TESTING.md](TESTING.md)).

**Do not optimize anything this week.** A scalar `f32` matmul that takes 90 seconds per
token is a *pass* if the logits are right. Speed is week 2's problem.

**The trap:** starting with SIMD or Metal before parity passes. You will then be
debugging two things at once — a wrong algorithm and a wrong kernel — and you will
lose three days. Every experienced person who has built this has lost those three days.

---

## Week 2 — It's fast and it's small

**Checkpoint: coherent multi-sentence generation, throughput measured against
llama.cpp, int8 weights loading with a recorded perplexity delta.**

- [ ] KV cache — allocation, layout, incremental decode. This is where prefill and
      decode become different code paths.
- [ ] Sampling: greedy, temperature, top-k, top-p. Seeded and reproducible.
- [ ] Optimize the matmul. Blocking/tiling first, then SIMD. Re-run parity after
      *every* change.
- [ ] int8 symmetric per-channel weight quantization: scale computation, packing,
      dequant-on-the-fly matmul.
- [ ] Measure: prefill tok/s, decode tok/s, peak RSS, fp32 vs int8.
- [ ] Perplexity harness on a wikitext-2 subset to quantify the quantization cost.

**The trap:** claiming a speedup you didn't measure properly. Read
[BENCHMARKS.md](BENCHMARKS.md) *before* you produce the first number, not after.

---

## Week 3 — The part nobody else builds

**Checkpoint: 500 consecutive generations against a JSON Schema, 100% parse
successfully. Not 99.6%.**

- [ ] Grammar frontend: JSON Schema → intermediate grammar → regex/EBNF.
- [ ] Compile to a DFA **over bytes**, not characters. See [GRAMMAR.md](GRAMMAR.md)
      for why this is not a detail.
- [ ] Vocabulary index: for each DFA state, the set of token IDs that can be emitted
      without leaving the language. Built lazily and cached.
- [ ] Logit masking in the sampling loop; DFA state advanced by the chosen token.
- [ ] Handle the hard cases: tokens spanning multiple states, the EOS-only-in-accepting
      -states rule, and unreachable-state detection.
- [ ] Adversarial validation — schemas with nested objects, arrays, enums, numbers
      with constraints. Every generation goes through a real JSON parser.

**The trap:** treating the DFA as character-level because the schema is text. The
tokenizer emits byte sequences, some of which are not valid UTF-8 on their own. If the
DFA thinks in `char`, multi-byte content will break it in ways that only show up on
non-ASCII input — i.e. in the demo, in front of someone.

---

## Week 4 — It's defensible

**Checkpoint: a filled-in results table in the README, every number reproducible by
running one script, and a writeup you could hand to an interviewer.**

- [ ] `scripts/bench.sh` — the single source of every number, with warmup, n≥10 runs,
      median and p95.
- [ ] Head-to-head vs. llama.cpp: same model, same quantization, same machine, stated
      methodology.
- [ ] The comparison that matters: valid-JSON rate constrained vs. prompted-only, plus
      the tok/s cost of masking.
- [ ] Error types replacing panics. `clippy` clean.
- [ ] README results table. `docs/WRITEUP.md` — what was hard, what you got wrong,
      what you'd do differently. **Include the negative results.**
- [ ] A 60-second screen recording. Prompt in, constrained JSON out, live.

**The trap:** running out of time and skipping the writeup. The writeup is the part
that converts the work into a job. Protect week 4 by cutting week 2 and 3 scope, not
by cutting week 4.

---

## Cut list

When you're behind, cut in this order. Everything here is genuinely optional — the
project is still complete and defensible without any of it.

1. top-p sampling (top-k and temperature are enough)
2. Perplexity measurement (report memory savings only, and say why you didn't measure quality)
3. SIMD kernels (blocked scalar matmul is a legitimate result — report the tok/s honestly)
4. EBNF frontend (JSON Schema alone is enough to demonstrate the technique)
5. Metal / GPU backend — **this is already out of scope for v1, do not add it**
6. Llama-3.2-1B support (one model, done properly, beats two done partially)

## Explicitly not in v1

Batching, continuous batching, paged attention, speculative decoding, LoRA, training,
fine-tuning, a server mode, a web UI. Several of these are excellent v2 material and
one of them is a good answer to "what would you do next" — but none of them are
allowed to consume week 1–4 hours.
