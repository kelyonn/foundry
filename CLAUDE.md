# CLAUDE.md — working context for Foundry

Read this before touching anything in this repo.

## What this project is

Foundry is an LLM inference engine written from scratch in Rust, plus a
grammar-constrained decoder that compiles a JSON Schema or EBNF into a DFA over bytes
and masks logits so output is valid by construction.

Target model for v1: **Qwen2.5-0.5B-Instruct**. See [docs/MODEL.md](docs/MODEL.md).

## Why it exists (this changes how you should help)

This is a portfolio flagship built to be **defended in a technical interview**, not
merely to work. Kalyan will sit across from someone who asks "why does grouped-query
attention reduce KV cache size but not FLOPs?" or "walk me through what happens to a
token that spans two DFA states." If the code works but he can't answer, the project
has failed at its actual purpose.

So: **working code is necessary but not sufficient here.** Optimize for him
understanding it, not for it existing.

## Working agreement — hybrid, quiz-gated (superseded D8, see D10)

This is a proposal — Kalyan should edit it if he disagrees, but it should be explicit
either way.

Claude writes the code for every math-bearing module (attention/RoPE/GQA, KV cache,
RMSNorm/SwiGLU, quantization, DFA construction, vocab intersection, logit masking).
This is faster than Kalyan typing it, and the tradeoff is deliberate: the time saved
is spent on understanding depth instead, via the loop below. **The point is not to
save Kalyan effort — it's to spend the hours on understanding rather than on typing.**

### The per-module loop — do not skip steps or reorder them

1. **Kalyan explains the logic first, out loud, before any code exists.** What the
   module does, why it's shaped this way. If he can't, stop and go back to the
   relevant doc/resource — don't start coding on a shaky explanation.
2. **Claude writes the module**, narrating non-obvious choices while writing: shape
   before/after each op, why this convention and not the other one, which line
   corresponds to which line of the paper/spec.
3. **Kalyan runs the parity/correctness test himself** and reads the actual diff —
   Claude does not report "it passed," Kalyan sees the number.
4. **Immediate quiz, closed-book (no re-opening the file), 4–6 questions:**
   - what the module does
   - why this approach and not an obvious alternative
   - one deliberately-wrong "what if I'd done X instead" — checks he isn't
     pattern-matching the happy path
   - at least one question that requires computing something (a shape, a size, a
     count) rather than recalling a fact
5. **Gate: if the quiz reveals a shaky answer, the module is not done.** Re-explain,
   have Kalyan re-derive the shaky part out loud (or type a corrected snippet
   himself if that's what it takes), then re-quiz. Passing tests is necessary, not
   sufficient — the module is done when Kalyan passes, not when `cargo test` does.
6. Log a one-line note in the module's section of `docs/QUIZ_LOG.md` — date, module,
   any question he got wrong, and whether the re-quiz passed. This is the thing that
   catches "I understood it three weeks ago and it's since evaporated."

**Claude always writes**, no quiz gate on these (not asked about in interviews):
crate scaffolding, `Cargo.toml`s, error types, `scripts/dump_reference.py`, test
harnesses, fixture generation, the benchmark runner, CLI parsing, file I/O,
safetensors parsing, logging.

**Both, always:** debugging. When a layer's activations diverge, help Kalyan
*localise* it — don't hand him the corrected line. Narrow to the layer, then the op,
then ask what he thinks is wrong before saying. Debugging a real failure is itself
one of the best teaching moments available and shouldn't be shortcut even under this
agreement.

**Non-negotiable regardless of who typed a module:** Kalyan must be able to, unaided,
answer every question in the quiz log for it, plus the standing questions in
[GRAMMAR.md](docs/GRAMMAR.md) and terms in [GLOSSARY.md](docs/GLOSSARY.md). If a
pre-interview spot check (see below) fails on a module, that module gets re-derived
and re-quizzed before moving on to new work, even if it costs a day.

## Hard rules

1. **No ML libraries in `crates/`.** Not `candle`, not `burn`, not `ndarray`, not
   `tch`, not `ort`. If a dependency implements a tensor operation, it is out. File
   I/O, JSON parsing, memory mapping, CLI parsing, and error handling are fine.
2. **Correctness before speed, always.** Do not write a SIMD kernel for an operation
   whose scalar version has not passed parity. Every optimization must be preceded by
   a passing test and followed by the same test.
3. **Never quote a benchmark number that wasn't produced by `scripts/bench.sh`.** No
   numbers from memory, no numbers from a single run, no numbers in the README that
   aren't reproducible. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
4. **Verify model config against `config.json`, not against this repo's docs.**
   [docs/MODEL.md](docs/MODEL.md) records expected values, but the file on disk wins.
   If they disagree, the doc is wrong — fix the doc.
5. **Constrained decoding correctness is binary.** 99.8% valid JSON is a failed
   implementation, not a good result. If output can be invalid, the DFA or the vocab
   index has a bug. Find it.
6. **Record decisions in [docs/DECISIONS.md](docs/DECISIONS.md)** when a non-obvious
   fork is taken. One short entry, including what was rejected.

## Commands

Nothing is built yet. Fill these in as they start working.

```
cargo test                          # all tests
cargo test -p model parity          # logit parity vs reference dumps
python scripts/dump_reference.py    # regenerate reference activations (needs HF)
./scripts/bench.sh                  # the only source of benchmark numbers
```

## Code style

- Rust 2021, `rustfmt` defaults, `clippy` clean before commit.
- Prefer explicit shapes in names: `q_bhsd`, `k_bhsd` beats `q`, `k`.
- Every non-obvious tensor operation gets a comment giving the **shape before and
  after**. This is the single highest-value comment type in this codebase.
- Panics are acceptable in week 1–2 for shape mismatches. Convert to typed errors in
  week 4, not before — error plumbing early is a time sink.
- No `unsafe` without a comment explaining the invariant being upheld.

## Current phase

**Week 0 — nothing built.** Next action is the week 1 checkpoint in
[docs/ROADMAP.md](docs/ROADMAP.md): weights load, tokenizer round-trips, forward pass
matches reference logits within `1e-4`.

Update this section as phases complete. It is the first thing to read after this
file's header.

## Before ending any session where a module was completed

Update [docs/QUIZ_LOG.md](docs/QUIZ_LOG.md) with what was quizzed and how it went —
this is the actual record of whether Kalyan is interview-ready, not just whether the
code exists. Don't let a session end without it if a module closed in that session.

## Related work by the same author

Cerberus (`../cerberus`) is a from-scratch Raft KV store being built in parallel on
the same timeline. Same principle applies there: he writes the consensus core. Don't
let the two projects cannibalise each other's hours — if one slips, it should slip
deliberately, not by accident.
