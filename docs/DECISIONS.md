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

*Rejected:* **Go** — the language I know best and Cerberus uses it, but Go's SIMD story
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

---

---

### D11 — Build the grammar crate before the engine, and make it standalone
**decided** — 2026-08-15

The roadmap puts constrained decoding in week 3, after a working forward pass. It was
built first instead, and it depends on no part of the engine: its entire interface to
a model is `&mut [f32]` of logits, plus a token-id→bytes map.

*Why:* an interview came up while the repo was still at week 0. Constrained decoding
needs no weights, no numerics, and no parity chase, so it is the only half that can be
made real and demonstrable under a hard deadline. It is also the differentiating half.

*Rejected:* rushing week 1 (safetensors + tokenizer + forward pass). Logit parity
debugging is not timeboxable, which is exactly what ROADMAP.md's week-1 "trap" note
says, and a half-debugged forward pass demonstrates nothing.

*Consequence worth keeping:* the crate has no dependency on the engine at all, so the
decoupling turned out to be a design improvement rather than a deadline compromise.
The engine will plug in by handing it a logit slice.

---

### D12 — The JSON grammar emits no whitespace
**decided** — 2026-08-15

GRAMMAR.md asked for this choice to be made explicitly. Output is canonical, compact
JSON — no spaces or newlines anywhere in the language.

*Why:* permitting arbitrary whitespace multiplies the legal token set at every state,
which both enlarges the automaton and lets the model spend its token budget on
indentation carrying no information.

*Rejected:* a bounded whitespace allowance. It has all the cost and no real benefit —
pretty-printing is a downstream concern for a human reader, not a generation-time one.

---

### D13 — Legality means an accepting state is still reachable, not that one byte survived
**decided** — 2026-08-15

Subset construction produces states that are alive (they have outgoing transitions)
but doomed (every path from them eventually dies). `Dfa::prune` removes them by
reverse BFS from the accepting states before the automaton is ever used.

*Why:* without it, the masker allows a token leading into a doomed state, because that
token "didn't hit the dead state" — and generation then wanders into a region where
nothing can complete the parse. The advertised guarantee is that every prefix is
extendable to a valid document, which is a reachability claim.

*Rejected:* detecting the dead end at decode time only. The `DeadEnd` error still
exists as a loud backstop, but relying on it would turn a compile-time property into a
runtime failure.

---

### D14 — Value bounds are enforced by restricting the digit budget, or not at all
**decided** — 2026-08-15

Two bugs found by the test oracle, both fixed in the grammar rather than by loosening
the assertion:

1. `{"score":2760420040100242100E670}` — well-formed per RFC 8259, rejected by
   `serde_json` because it exceeds `f64`. Fixed by capping the exponent at two digits;
   with an 18-digit mantissa the maximum reachable value is ~1e117.
2. `-0` for an `integer` — legal JSON, mathematically an integer, but `serde_json`
   parses it as a float because `i64` has no negative zero. Fixed by excluding `-0`
   from the integer literal (but not from a `number`, since `-0.5` needs it).

*Why this shape of fix:* a DFA cannot compare magnitudes, so "must fit in an f64" is
not directly expressible. Restricting the digit budget makes the constraint structural,
which is the only kind this layer can enforce. `minimum`/`maximum` remain unsupported
and are documented as such rather than approximated.

*Rejected:* post-hoc validation and resampling. That reintroduces exactly the
retry-loop failure mode the project exists to eliminate.

---

### D15 — `\uXXXX` escapes enforce UTF-16 surrogate pairing in the automaton
**decided** — 2026-08-16

`\u` followed by any four hex digits is wrong. `\uXXXX` encodes a UTF-16 *code
unit*, and `D800..DFFF` is the surrogate block — legal only as a high surrogate
immediately followed by a low one. The escape is now split three ways: non-surrogate
code units, an indivisible high+low pair, and nothing else. A lone low surrogate has
no production at all.

*How it was found — this is the part worth remembering.* With the 291-token synthetic
vocabulary, 500/500 generations passed and the bug was invisible: no token in that
vocabulary can spell a lone surrogate. Against the real 151,936-token Qwen vocabulary,
the same schema failed 9 times in 500, on strings like `"\uDafe"`. The test suite was
green throughout. **The oracle was right and the fixture was too weak** — which is the
general lesson, not a JSON-specific one.

*Why it stays regular:* pairing is a fixed-length adjacency requirement, so the
automaton simply has no path from a high surrogate to anything but the backslash of
its partner. Contrast `minimum`/`maximum` on a number (D14), which is genuinely not
expressible. "Looks semantic" and "is beyond regular" are different questions, and it
is worth checking which one you are facing before giving up on the automaton.

*Cost:* DFA grew from 1,128 to 1,518 states for the demo schema.

---

### D16 — Threaded matmul splits output columns, and must not change a single bit
**decided** — 2026-08-16

`linear` and `linear_q` split work across `available_parallelism()` threads by
output column (`n`), not by input row (`m`).

*Why columns:* during decode `m == 1` — one token — so splitting by rows would
leave every thread but one idle in exactly the phase that dominates generation
time. `n` is 896, 4864, or 151936, so it always divides usefully.

*The non-negotiable constraint:* every element is still computed by the exact
same sequence of operations as the single-threaded version — same accumulator
structure, same order, just executed by a different thread. This makes the
change a pure regression check against the parity harness rather than a new
thing to re-verify: parity was bit-identical before and after
(1.138e-5 / 2.861e-5 relative error, unchanged to the last digit).

*Measured effect:* 6.3× on decode (2.22 → 13.95 tok/s), 4.9× on prefill, on a
10-core M5, from threading alone — before any SIMD.

---

### D17 — int8 accumulates in f32, not f64; the f32 matmul accumulates in f64
**decided** — 2026-08-16

The two matmuls in this codebase use different accumulator precision, and it is
not an oversight.

`linear` (f32 weights) accumulates in f64. The first version accumulated in f32,
reasoning that parity should imitate the reference's own arithmetic — that was
backwards (see D-adjacent note in `crates/tensor/src/ops.rs`): a naive f32
sum over k=896 terms hit 1.3e-4 error on its own, before 23 more layers
contributed anything, while accumulating in f64 and narrowing once at the end
made `q_proj` agree with a float64 oracle to 3.8e-6 — roughly 34× better.

`linear_q` (int8 weights) accumulates in f32. Here the weights are already
quantized to 8 bits, so the input to the dot product carries at most ~0.4%
relative error before a single multiply happens — three orders of magnitude
larger than anything f32 accumulation rounding could add. Paying for f64 would
buy precision the input doesn't have.

*The general principle:* accumulator precision should exceed the precision of
what's being accumulated, not match some fixed policy. Check what the actual
error floor is before choosing.

---

### D18 — int8's relative error stays small until the residual stream's own scale collapses, and that collapse is real
**decided** — 2026-08-16

Running `parity --int8` against the float64 oracle (added specifically to check
the int8 path, which had only been checked indirectly via perplexity) showed
relative error flat at 0.15–0.28% from layer 0 through layer 20 — healthy,
expected weight-only quantization noise — then jumping to 13–28% at layers
21–23 and 21% at the logits.

*What it turned out to be, and how it was checked without touching Rust:* a
direct `numpy` read of the float64 reference dump (independent of any code in
this repo) shows the reference model's own residual-stream magnitude collapses
about 30× between layer 20 and layer 21 — max|x| 1574 → 53, RMS 24 → 2.2, the
same ratio in both, meaning a genuine signal collapse rather than noise. The
int8 path's *absolute* error stays roughly flat (~2.4) straight through this
point. The relative-error spike is the same near-constant absolute noise
dividing into a suddenly much smaller signal — a real architectural feature of
Qwen2.5-0.5B's own weights, not a defect in the quantizer.

*Why this isn't "fixed":* there is nothing to fix. Naive per-channel weight-only
int8 with no calibration genuinely degrades more where a model's own dynamic
range is naturally low — that is a property of the technique, not a bug in this
implementation. Confirmed the mechanism itself is sound two other ways: argmax
still agrees with the reference on this prompt, and the measured perplexity
delta is +0.69% (`cargo run --release -p cli --bin perplexity`), which is what you'd
expect if the divergence, while locally large in a few layers, doesn't
meaningfully change the output distribution's shape.

*Second-order lesson, same shape as D18's older sibling D15/the ulp discovery:*
a fixed global-max-based relative-error metric can itself mislead exactly where
a network's own scale is non-stationary. The number was real and the alarm was
worth investigating; the fix was understanding the metric, not the code.

**Correction — 2026-08-16, same day, re-verified during the bug hunt.** "Flat
0.15–0.28% from layer 0 through layer 20" was not quite right: re-running
`parity --int8` shows two isolated failures before layer 20 —
`layer07.mlp_out` at 7.2% and `layer08.mlp_out` at 5.4%. Checked rather than
waved away: `mlp_out`'s own magnitude dips to 1.08 and 1.74 at those two layers
(typical elsewhere is 3–7), while the near-constant absolute int8 noise stays
put — the identical mechanism as the headline finding, just a local dip in an
intermediate quantity rather than the layer 20→21 collapse. It is invisible in
the residual stream: `layer_out` at the same two layers has magnitude ~1571 and
relative error ~1.6e-3, because the residual add swamps a component that is
locally small. The corrected claim: relative error is flat and small in the
*residual stream* throughout, with two harmless local spikes in an
intermediate, non-residual quantity — not "flat everywhere through layer 20."

---

### D19 — Bug hunt: six real defects found across the four non-grammar crates
**decided** — 2026-08-16

A systematic adversarial pass across `tensor`, `tokenizer`, `model`, `cli`, and a
second look at `grammar`, after week 2 closed. All six below are real — found by
either reasoning about an edge case and then confirming it, or by measuring
first and finding the code disagreed with its own comments. None were found by
inspection alone without running something to check.

**1. Unconstrained CLI decode could panic on a padding-tail token.** The forward
pass emits 151,936 logits over a 151,665-token vocabulary (D-note in
docs/GRAMMAR.md). `crates/grammar`'s `apply_mask` already masks the 271-wide
padding tail unconditionally. The CLI's *unconstrained* decode path (no
`--schema`) never did — it sampled from the raw row, so a hot enough
temperature could select a padding id and `tokenizer::decode` would index out
of bounds. Greedy never surfaced it because the padding logits measure
strongly negative on this checkpoint (confirmed: min −5.03, max −1.95, versus
13–19 for real tokens) — a property of this specific model, not a guarantee.
Fixed with `cli::mask_padding_tail`, called on every decode step when no
grammar is active, with a unit test that puts `999.0` in the padding slots and
confirms neither `argmax` nor the sampler can select them.

**2. Empty `--prompt` panicked instead of erroring.** `tk.encode("")` is zero
tokens; `Model::forward` asserts `seq > 0`. Fixed with a clean check-and-exit
in the CLI before the model ever sees it.

**3. `string`'s `minLength > maxLength` was silently clamped; `array`'s
`minItems > maxItems` was correctly rejected.** The identical contradiction,
two different behaviors, found by auditing the two length-bound code paths
against each other. `array` already had a test and a comment explaining why
silent clamping is wrong ("would look like compiled fine right up until every
generation hits a DeadEnd"); `string` just hadn't gotten the same treatment.
Now both reject.

**4. `Tokenizer::encode_with_specials` could hang forever on a malformed
`tokenizer.json`.** `str::find("")` always returns `Some(0)` — an
`added_tokens` entry with empty `content` would match at position 0 with
length 0 on every loop iteration, `rest` would never shrink, and the loop
would spin forever. Not reachable with the real Qwen file (checked: zero
empty-content entries), but nothing upstream validates that guarantee, and a
hang is a worse failure than a crash for the same reason the grammar crate's
`DeadEnd` is a loud error rather than a silent stall. Fixed by skipping
zero-length needles; regression test spawns the call on a thread and asserts
it finishes within 2 seconds.

**5. No ceiling on user-supplied `maxLength` / `maxItems` — a one-line schema
DoS.** The *default* values (64, 16) were bounded and documented; an explicit
`maxLength: 100000` was not. Measured before fixing: `maxLength: 4096` alone
compiles to 192,515 states in 34.5s and a ~197 MB transition table, and the
growth is superlinear in the bound — subset construction's cost does not
grow linearly with state count. One short line of attacker-controlled JSON
Schema could stall the process for minutes and allocate gigabytes. Added a
512 ceiling (`MAX_BOUNDED_REPEAT`, well above every value used anywhere in
the test suite), enforced at parse time on both fields, with a test that
oversized values are rejected and the ceiling value itself still compiles.

**6. Out-of-range token id produced a bare, unhelpful slice-index panic.**
`forward_cached` is a public entry point — a caller (a test, a future API
consumer, a corrupted token stream) can hand in any `u32`. The embedding
lookup indexed straight into `embed_tokens` with no bounds check, so an
out-of-range id failed with Rust's generic "index out of bounds: the len is
X but the index is Y," which says nothing about *what* was out of range or
why. Added an explicit check before the lookup: "token id {tok} is out of
range for a {vocab_rows}-row embedding table" — same fix shape as #1 and #2
in spirit, a public boundary given a real error message instead of an
implementation detail's panic leaking through it. Regression test in
`crates/model/tests/kv_cache.rs` asserts the panic message, not just that it
panics, so a future refactor that turns this back into an opaque slice panic
fails the test even though "something panics" would still technically hold.

*Method note, since it's the actual point of this entry:* every one of these
was caught by asking "what happens at the edge, and does the code's own
behavior match what its comments already claim about analogous cases" —
not by writing more happy-path tests. #3 and #1 were both found by noticing an
existing comment or an existing safeguard elsewhere in the codebase that
implied a guarantee, then checking whether the guarantee actually held
everywhere it needed to.

---

### D20 — Perplexity moved to real wikitext-2, and the sample size turned out to matter
**decided** — 2026-08-16

Swapped the perplexity harness's default corpus from a built-in prose paragraph
to a real slice of wikitext-2's test split (Merity et al., 2016) — the corpus
docs/ROADMAP.md names for this measurement. `scripts/fetch_wikitext2.py` pulls
it via the `datasets` library from `Salesforce/wikitext` (the original
`wikitext` repo's loading script is deprecated in current `datasets` and fails
outright — the Salesforce mirror is the same corpus republished as plain
Parquet). Output is committed at `tests/fixtures/wikitext2_test.txt` (~8 KB, CC
BY-SA, redistribution-safe) so the measurement reproduces with no network
access. `perplexity.rs` prefers it automatically and falls back to the old
paragraph — explicitly labeled "not wikitext-2" — only if the fixture is
missing.

*What was found while picking a sample size, and why it's worth recording:*
the fp32→int8 perplexity delta's **sign** moved as the token count grew —
−0.17% at 256 tokens, +0.11% at 512, +0.10% at 768, consistent with a longer
partial run at ~1888 tokens. A negative delta would mean quantization
*improved* the model, which has no mechanism behind it here (weight-only int8
adds noise, it doesn't add information) — the honest read is that 256 tokens
is too small a sample for this effect size, and the measurement was reporting
noise, not quantization quality. 768 is now the default specifically because
it's the smallest size where the sign stopped moving.

*Why not just run the full ~1888-token test-split slice and be done with it:*
tried it. Attention cost is O(seq²) per layer in this project's scalar,
unbatched-attention implementation, so 1888 tokens is not "3.7× the cost of
512" (linear scaling) — it's closer to 3.7× on the linear-projection term and
~14× on the attention term. The full-length run outlived a 120s foreground
window, was moved to background, and never completed the int8 phase (exited
with no error and no result, most likely the harness's own runaway-process
handling rather than anything in the code — no panic, no OOM message, nothing
Rust-side to debug). 768 tokens finishes in ~78 seconds and gives a number that
already agrees in sign and rough magnitude with what the partial long run was
trending toward, which is the actual goal — a stable, trustworthy delta — not
"use every byte of the fixture."

*Rejected:* reporting the original 256-token, wrong-sign number, or quietly
switching to whichever sample size gave the "nicer" (more flattering) result
without saying why. Both would be exactly the kind of unfalsifiable
number-picking this project's testing philosophy exists to rule out.

---

### D21 — The quiz-gated loop is waived for the final push
**decided** — 2026-11-17

Kalyan asked for the remaining work (SIMD, llama.cpp comparison, typed errors,
writeup, demo) to be finished end to end without being kept in the loop, and said
to skip the "I write, you review" step. D10's per-module quiz gate and
`QUIZ_LOG.md` entries are therefore **not** run for the modules touched here
(thread pool, NEON kernels, error types). The substitute is `docs/explainers/`:
one interview-oriented explainer per module with worked shape/size calculations
and likely questions. That is study material, not proof of understanding — the
non-negotiable line in CLAUDE.md (he must be able to answer unaided) still stands
and the spot-check before any interview is still owed.

*Rejected:* quietly marking modules as quizzed, or leaving the QUIZ_LOG implying
these were gated.

---

### D22 — Persistent worker pool instead of per-call `thread::scope`
**decided** — 2026-11-17

`crates/tensor/src/pool.rs`: std-only pool (mutex + condvar, caller participates,
submissions serialised, task panics propagated after all tasks finish). Replaces
the per-matmul thread spawns that CLAUDE.md flagged as the known decode
inefficiency. Same column split as D16, so output is unchanged: parity stayed at
exactly 1.482e-6 after this change alone (before the SIMD work).
The lifetime-erased closure pointer is the one `unsafe` here; the invariant is
documented at `Job` (caller blocks until `finished == total`).

*Rejected:* `rayon` (fine under rule 1 but breaks the zero-dependency tensor
crate and hides the very thing worth explaining); a lock-free queue (more code,
no measurable win at ~170 jobs/token).

---

### D23 — Tiled loop order + NEON kernels; scalar f64 kept as the oracle
**decided** — 2026-11-17

Two separate changes, measured together with `FOUNDRY_SCALAR=1` as the A/B switch
(same binary, same harness):
1. **Loop order** `i → j` became `i-tile(64) → j → i`. The old order re-streamed
   the whole weight chunk from memory for every token; the new one reads weights
   once per 64-token tile (8 passes instead of 512 at prefill).
2. **NEON dot kernels** (`std::arch::aarch64`, 4 vector accumulators). f32 path
   does f32 FMA inside 256-element blocks and flushes each block into an f64 total
   — keeping error near the f64 kernel's. int8 widens i8→i16→i32→f32 then FMAs.

Parity after: relative 1.78e-6 (was 1.48e-6), argmax agrees, inside the 5e-5
gate. It is no longer bit-identical to the scalar kernel and is not meant to be:
unit tests compare the selected kernel to `dot_scalar` / `dot_q_scalar` across
ragged lengths. Non-aarch64 builds fall back to the scalar kernels.

*Rejected:* Accelerate/BLAS (an ML-adjacent dependency, banned by rule 1 in
spirit and it would make the comparison against llama.cpp meaningless);
activation quantization + `sdot` (changes numerics beyond weight-only int8 —
a real next step, not a quiet change here).

---

### D24 — Typed errors at the boundary; invariant asserts stay asserts
**decided** — 2026-11-17

The library crates already returned typed errors for everything input-driven
(`ConfigError`, `LoadError`, `TokenizerError`, `SchemaError`, `DecodeError`). The
remaining gaps were in the binaries: `.expect()` on file loading in `bench`,
`perplexity`, `json_compare`, and `main`, and — worse — flag parsing that used
`.parse().unwrap_or(default)`, so `--max-tokens abc` silently ran with 32.

Now: `cli::CliError` gathers the library errors; `foundry`'s `main` is
`run() -> Result<(), CliError>`; exit code 2 for usage errors, 1 for failures;
flag values are parsed strictly. The measurement bins use `cli::or_exit`.

Kept as panics on purpose: shape/length `assert!`s in `tensor` and `model` (a
mismatched matmul is a bug in this code, not bad input) and the empty-prompt /
KV-capacity asserts in `forward_cached`, whose preconditions the CLI checks first.
`model`'s `parity` bin still uses `expect` — it is a test harness and `model`
cannot depend on `cli`.

*Rejected:* `thiserror`/`anyhow` (a `Display` impl is eight lines and the
dependency budget is part of the pitch); turning the forward-pass asserts into
`Result` (touches every call site and test to report conditions that cannot occur
through the CLI).

---

### D25 — *(next decision goes here)*
**open**
