# 05 — Grammar-constrained decoding

Code: [`crates/grammar/`](../../crates/grammar/src) — `schema.rs` → `pattern.rs` → `dfa.rs`
→ `vocab.rs` → `index.rs`. Standing questions live in [GRAMMAR.md](../GRAMMAR.md).

## The idea in one sentence

Compile the schema into a finite automaton over **bytes**; at every step, set the logit of
every token that would leave the language to `-inf`; so the output is valid *by
construction*, not by hope or retry.

## Pipeline

```
JSON Schema ──schema.rs──▶ Pattern (regex-like AST over bytes)
Pattern ──Thompson──▶ NFA ──subset construction──▶ DFA ──prune dead states──▶ DFA
DFA + Vocab ──VocabIndex (lazy, per state)──▶ TokenMask (bitset, 151,665 bits ≈ 19 KB)
each step: logits[illegal] = -inf → sample → DFA advances by the token's *bytes*
```

### Why bytes, not characters (the interview question)

The tokenizer emits byte sequences. A token can end in the middle of a UTF-8 character, and
the next token finishes it. A `char`-level DFA cannot represent "I'm halfway through `é`"
— so it either rejects legal tokens or accepts illegal ones, and only on non-ASCII input
(i.e. in the live demo). The DFA therefore has states for partial UTF-8 sequences, and
tests cover every length of well-formed UTF-8, ill-formed input, and a character split
across tokens.

### A token that spans several DFA states

Token `":"` of bytes `3a 22` drives the DFA through two transitions. A token is **legal in
state S iff `step_token(S, bytes)` never hits the dead state**; the new state is wherever the
last byte lands. The mask for S is therefore "all tokens whose full byte string walks
cleanly from S" — computed by scanning the vocabulary once per *reached* state.

### Lazy index

Precomputing masks for all states × 151,665 tokens is wasteful: a generation visits few
states. `VocabIndex::mask_for` builds on first touch and caches (`builds` vs `hits` are
reported). The second generation against the same schema pays nothing.

### EOS

End-of-sequence is legal **only in accepting states**, and that rule is applied at mask
time, not baked into cached masks (it depends on the state, not the vocabulary — baking it
in would double the cache).

### Dead-state pruning

States from which no accepting state is reachable must be removed. Without this the mask can
allow a token that leads into a dead end, and the model then has *no legal continuation* —
the generation fails halfway. Pruning is what turns "prefix-valid" into "completable".

### Special tokens and the padding tail

Control tokens (`<|im_end|>`…) are never legal as content: their *surface text* is not what
they mean. Logits beyond the real vocab (151,665 → 151,936) are masked unconditionally.

## What the grammar does *not* promise (say this before being asked)

- **Compact JSON only** — no whitespace; every declared property is required and emitted in
  declaration order (permuting n keys needs n! alternations; a regular language can't say
  "each once in any order" cheaply).
- **Shape, not value bounds** — `maxLength`/`maxItems` yes (bounded repetition is regular);
  `minimum`/`maximum` no (comparing magnitudes needs counting).
- **No `$ref`/recursion** — recursive schemas are context-free, not regular.
- Length ceilings exist because subset construction blows up (D19: a one-line schema with a
  huge `maxLength` could DoS the process for minutes).

## Verified how

500/500 constrained generations over a rotating set of adversarial schemas parse with
`serde_json` against the real 151,665-token vocab (`crates/grammar/tests/correctness.rs`);
that test found the lone-surrogate bug (`\uD800`–`\uDFFF` escapes). On the real model,
`json_compare` additionally checks *schema conformance* with an independent checker and
reports the prompted-only baseline next to it (see README results).

## Questions

1. *Walk through a token spanning two DFA states.* (above)
2. *Why must EOS be state-dependent?* Otherwise the model can stop mid-object.
3. *Why can't a DFA enforce `"minimum": 10`?* Needs unbounded counting/comparison.
4. *What is the cost per token?* One mask lookup (cached) + one `-inf` fill + DFA walk of the chosen token's bytes. Measured: see README "masking cost".
5. *Why is 99.8% a failed implementation?* If a mask is correct, the failure rate is 0 by construction; any non-zero rate means a DFA or index bug.
