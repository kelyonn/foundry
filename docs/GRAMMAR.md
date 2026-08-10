# Grammar-constrained decoding

This is the half of the project that almost nobody implements from scratch. Understand
it deeply; it is the thing that will get asked about.

## The claim

Given a JSON Schema, Foundry produces output that parses against that schema **every
time**. Not "usually." The guarantee is structural: at every decoding step, tokens
that would make a valid completion impossible have their logits set to `-inf`, so the
sampler cannot choose them regardless of what the model wants.

The model is not persuaded to behave. It is prevented from misbehaving.

## The pipeline

```
JSON Schema
    │  frontend
    ▼
regular grammar (regex / restricted EBNF)
    │  Thompson construction + subset construction
    ▼
DFA over BYTES                       states × 256 transition table
    │  intersect with tokenizer vocabulary
    ▼
VocabIndex:  state → set of legal token IDs
    │  at each decode step
    ▼
logits[illegal] = -inf  →  sample  →  advance DFA by chosen token's bytes
```

## Why bytes, not characters

This is the design decision that matters most, and getting it wrong is the most likely
way this project fails in front of an audience.

The tokenizer is **byte-level BPE**. Tokens map to sequences of *bytes*, and an
individual token's bytes are frequently not valid UTF-8 on their own — a multi-byte
character like `é` or `₹` or an emoji is routinely split across two or three tokens.

If your DFA transitions on `char`, you cannot ask "does this token keep us in the
language?" for any token that is a UTF-8 fragment. You'll be forced to either skip
those tokens (silently breaking non-ASCII output) or guess (silently breaking
correctness).

So: **the DFA alphabet is `u8`, all 256 of them.** A transition table is
`[num_states][256]`. Every token is a byte string, and checking a token means walking
its bytes through the DFA one at a time.

Everything downstream is simpler because of this decision. Make it on day one of week 3.

## The vocabulary index

The core question, asked once per decode step:

> Given DFA state `s`, which of the 151,936 tokens can I emit without leaving the
> language?

A token is legal from state `s` if walking its bytes from `s` never hits the dead
state. Note the token may traverse *many* states — a token like `":\"` is three bytes
and three transitions. The result is the state you land in, which becomes the new
current state.

```rust
/// Walk a token's bytes through the DFA. None = illegal from this state.
fn step_token(dfa: &Dfa, from: StateId, token_bytes: &[u8]) -> Option<StateId> {
    let mut s = from;
    for &b in token_bytes {
        s = dfa.transition(s, b)?;   // None on dead state
    }
    Some(s)
}
```

### Cost, and why you must build it lazily

The full index is `num_states × vocab_size` membership tests. With a few hundred states
and ~152k tokens that's tens of millions of token-walks — seconds to minutes of
precompute, and most of it wasted, because a typical generation visits a small fraction
of states.

**Build lazily and cache.** On first arrival at a state, scan the vocabulary once and
store the resulting bitmask (`151936 bits` = 19 KB per state — cheap). Every subsequent
visit is a table lookup.

```rust
struct VocabIndex {
    cache: HashMap<StateId, BitVec>,   // 19 KB per cached state
}
```

Report the cache hit rate in your benchmarks. "The first generation pays 40 ms of index
construction, subsequent ones pay 0.1 ms" is a much better story than a single averaged
number, and it demonstrates you understand where the cost actually lives.

## Hard cases

**Tokens spanning multiple states.** Already handled by `step_token` — but it means the
DFA cannot be advanced byte-by-byte from outside. Advancing is always "walk the whole
token."

**EOS.** The end-of-sequence token is legal **only in an accepting state**. Otherwise
the model can terminate mid-object and produce `{"name": "Ka` — which parses as
nothing. This is a one-line special case and it is the single most common bug in naive
implementations.

**Dead ends.** A state from which no token is legal means generation is stuck: the
grammar admits a continuation but no *token* spells it. This is rare but real, and it's
caused by tokenizer/grammar mismatch. Detect it explicitly and error loudly rather than
looping — a silent hang in a demo is worse than a crash.

**Whitespace.** JSON permits arbitrary whitespace between tokens, which massively
expands the legal set at every state and lets the model waste its budget on spaces.
Decide deliberately: either forbid whitespace entirely (canonical output) or allow a
bounded amount. Write the choice down here when you make it.

**Numbers and unbounded repetition.** `[0-9]+` is fine for a DFA, but a schema with
`maxLength` or `maximum` constraints pushes toward counting, which regular languages
can't do. For v1, enforce structure only and document the limitation honestly — "my
DFA guarantees the shape, not the value bounds" is a *strong* answer because it shows
you know where the expressiveness boundary is.

**Beyond regular.** Arbitrarily nested JSON is context-free, not regular. Bounding
nesting depth (say, 8) keeps it regular and finite. Say so out loud. A pushdown
automaton is the correct general answer and a great "what would you do next."

## What to measure (week 3–4)

| Metric | Why it matters |
|---|---|
| Valid JSON rate, constrained | Must be exactly 100% |
| Valid JSON rate, prompted only | The baseline that makes the number mean something |
| Masking overhead (tok/s delta) | The price of the guarantee |
| Index build time, first vs. cached | Shows you know where the cost is |
| States visited / states total | Justifies the lazy design |

The unconstrained baseline is essential. "100% valid JSON" alone is unfalsifiable — a
model might hit 97% unaided, in which case your contribution is 3 points and a
guarantee. If it's 60%, your contribution is enormous. **Measure it and report whatever
it is**, including if it makes your work look less impressive. That honesty is worth
more in an interview than the number.

## Prior art

`outlines` (Willard & Louf, 2023) introduced the FSM-vocabulary-index formulation.
`llguidance` and `xgrammar` are the current fast implementations. Read the outlines
paper — but implement from the idea, not from their source. Knowing the lineage and
being able to say "here's what they do differently and why" is part of the value.

## Interview questions to be ready for

- Why must the DFA be over bytes rather than characters?
- Why is the vocabulary index built lazily?
- What happens when a single token crosses several DFA states?
- Why can't a DFA handle arbitrarily nested JSON, and what would you use instead?
- Why `-inf` before softmax rather than zeroing after?
- Does constraining decoding hurt output quality? *(Yes — it can force the model off
  its preferred distribution. This is a real trade-off. Have a view on it.)*
