# Decisions

One entry per non-obvious fork. Each records what was chosen, what was rejected, and
why — because "why did you do it that way?" is the most common interview question and
"I don't remember" is the worst answer.

Format: **Status** is `accepted`, `superseded`, or `open`. Add new entries at the
bottom. Never delete one — supersede it.

---

### D1 — Rust for the engine
**accepted** · pre-week-1

Rust for all of `crates/`. Python only in `scripts/`, and never in the inference path.

*Rejected:* **Go** — Kalyan knows it best and Cerberus uses it, but Go's SIMD story
requires assembly or cgo, and the GC is a liability in a decode loop. Also, using the
same language for both flagships makes them read as one skill rather than two.
**C++** — what llama.cpp uses and the most natural fit, but the tooling and build
friction cost days that a four-week budget doesn't have. **Zig** — appealing, too
small an ecosystem for safetensors and JSON.

*Why it's the right risk:* Rust's borrow checker fights hardest on graph- and
pointer-heavy code. An inference engine is flat arrays and loops — ownership is nearly
trivial here. This is one of the friendlier Rust projects to learn on, and `portable-simd`
plus a real ecosystem (`memmap2`, `serde_json`, `criterion`) makes weeks 2 and 4 easier.

*Reconsider if:* week 1 is not passing parity by day 5. At that point the language is
costing more than it's returning — switch to Go and take the SIMD hit.

---

### D2 — Qwen2.5-0.5B-Instruct as the v1 target
**accepted** · pre-week-1

*Rejected:* **Llama-3.2-1B** — better name recognition, but 2× the parameters means
slower week-1 iteration, and week-1 iteration speed determines whether weeks 2–4
happen. **TinyLlama / GPT-2** — smaller still, but GPT-2 lacks RoPE, GQA, RMSNorm and
SwiGLU, so implementing it teaches an architecture nobody ships anymore.

Llama-3.2-1B is the stretch goal once v1 is complete. One model done properly beats two
done partially.

---

### D3 — CPU only, no GPU backend in v1
**accepted** · pre-week-1

*Rejected:* **Metal** — genuinely tempting given the Hue experience, and it would make
the throughput numbers look far better. Cut anyway: it doubles the surface area, and a
GPU kernel bug is much harder to localise than a CPU one. Also it makes the llama.cpp
comparison murkier.

The story "hand-written scalar and SIMD CPU kernels, honestly benchmarked" is complete
and defensible. "Half-working Metal shaders" is not. Metal is the strongest v2 answer
to "what would you do next" — which is worth more unbuilt than built badly.

---

### D4 — DFA alphabet is bytes, not characters
**accepted** · pre-week-1

The constrained-decoding DFA transitions on `u8`. 256-wide transition table.

*Rejected:* **`char`-level** — reads more naturally against a text grammar, but the
tokenizer is byte-level BPE and individual tokens are frequently UTF-8 *fragments*.
A char DFA simply cannot answer "is this token legal" for those, so non-ASCII output
breaks. That failure only surfaces on multi-byte input — i.e. during a demo.

This is the single most consequential decision in the grammar crate. See
[GRAMMAR.md](GRAMMAR.md).

---

### D5 — int8, symmetric, per-output-channel, weights only
**accepted** · pre-week-1

*Rejected:* **Asymmetric** (zero-point) — better range utilisation, but adds an add to
every dequant in the hot loop for marginal gain on weight distributions that are
roughly zero-centred anyway. **Per-tensor scale** — one scale is dominated by outlier
channels and costs real accuracy. **int4** — bigger memory win, but packing two values
per byte adds bit-twiddling that isn't worth week-2 hours; good v2 material.
**Activation quantization** — needs a calibration pipeline for a fraction of the
benefit at 0.5B parameters.

---

### D6 — Contiguous row-major tensors, no strides or views
**accepted** · pre-week-1

`Tensor` is `Vec<f32>` + shape. Transposes copy.

*Rejected:* a **strided/view-based** design like NumPy or PyTorch. It's the "proper"
answer and avoids copies — and it is also the number-one source of subtle bugs in
hand-written tensor libraries. In a four-week budget, paying for some copies to
eliminate an entire bug class is correct.

*Reconsider if:* profiling in week 2 shows transposes are a real bottleneck. Measure
before changing this.

---

### D7 — Fixed batch size of 1
**accepted** · pre-week-1

No batch dimension anywhere. Shapes are `[heads, seq, head_dim]`.

*Rejected:* carrying a batch dim "for later." It complicates every shape and every
test in exchange for a feature that is explicitly out of scope. Batching and continuous
batching are v2.

---

### D8 — Working agreement: Kalyan writes the math-bearing code
**superseded by D10** · pre-week-1

Attention, RoPE, GQA mapping, KV cache path, RMSNorm, SwiGLU, quantization, DFA
construction, vocabulary intersection, logit masking — his. Scaffolding, harnesses,
benchmarks, reference dumpers, plumbing, review — Claude's.

*Why (at the time):* the project's purpose is surviving a technical interview, not
existing. The listed components are precisely the ones that get asked about. Code he
didn't write is code he can't defend, and an indefensible flagship is worse than a
modest one.

*Rejected then:* AI-writing the whole thing for speed. It would finish sooner and be
worth less than nothing in the room.

*Superseded because:* real time budget turned out to be ~1–2 hrs/day shared with
Cerberus, and this split needed 20–25 hrs of Kalyan's own typing just for Foundry's
core modules — didn't fit. See D10 for the resolution: same non-negotiable outcome
(he must be able to defend every module unaided), different mechanism for getting
there.

---

### D10 — Hybrid build: Claude codes, Kalyan explains-first and is quiz-gated
**accepted** · pre-week-1, supersedes D8

Claude writes every module, including the previously-reserved math-bearing ones.
Before Claude writes a module, Kalyan explains its logic out loud from having learned
it (not from the code, which doesn't exist yet). After it's written, Kalyan runs the
correctness test himself, then is quizzed closed-book — recall, "why not the obvious
alternative," and at least one question requiring him to compute something (a shape,
a size). A module is only "done" when the quiz passes, regardless of whether the code
already passes its tests. Full procedure and quiz format in
[../CLAUDE.md](../CLAUDE.md); results logged per-module in
[QUIZ_LOG.md](QUIZ_LOG.md).

*Rejected:* **Kalyan types everything (original D8)** — correct in principle, didn't
fit the real time budget once Cerberus was sharing it; risked neither project
finishing. **Claude writes everything, no quiz** — fastest, but reintroduces exactly
the failure D8 existed to prevent: code he can't defend. **Kalyan types after seeing
Claude's version once** — considered and rejected, because typing a solution you've
already seen tests memory of the solution, not understanding of the problem; the
explain-first step in D10 is what actually forces the latter.

*Why this preserves the goal:* the interview doesn't test who pressed the keys, it
tests whether Kalyan can explain, extend, and debug the thing. Explaining *before*
seeing the code forces him to have a real model of the logic rather than a
recognition of Claude's implementation. The quiz gate is what makes this checkable
rather than aspirational — D8 had no verification step at all, it just assumed typing
implies understanding, which is not actually a safe assumption either.

*Risk this accepts:* understanding-by-explaining-and-quiz is probably somewhat
shallower than understanding-by-debugging-your-own-bugs for the specific class of
subtle bugs in [TESTING.md](TESTING.md)'s failure-signature table (e.g. GQA mapping
backwards producing fluent-but-wrong output). Mitigation: debugging stays fully
collaborative and Socratic per the CLAUDE.md rule regardless of who wrote the
original code — when a real failure occurs, Claude still localises-not-fixes. Real
failures during the build are not skipped just because Claude wrote the module.

*Reconsider if:* the quiz log shows repeated failures on re-quiz (same module, same
question type, more than once) — that's a signal the loop itself isn't producing
retention and the mechanism needs to change, not just the individual module redone.

---

### D11 — *(next decision goes here)*
**open**

Candidates that will need entries during the build: transpose-at-load vs. at-matmul
(D6 related); whitespace policy in the JSON grammar; nesting-depth bound; thread count
and parallelism strategy; stop-token handling.
