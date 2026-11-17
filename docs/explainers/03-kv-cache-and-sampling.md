# 03 — KV cache and sampling

Code: [`kv_cache.rs`](../../crates/model/src/kv_cache.rs),
[`sampling.rs`](../../crates/model/src/sampling.rs),
decode loop in [`crates/cli/src/main.rs`](../../crates/cli/src/main.rs).

## KV cache

**Problem.** Attention at position `p` needs keys and values of positions `0..=p`. They
are a pure function of tokens already fixed, so recomputing them per step is wasted work.

**Why K and V but not Q.** The query for position `p` is used exactly once, at step `p`.
Keys and values are read by every later position. So the cache holds two of the three
projections.

**Layout.** `[layer][kv_head][position][head_dim]`, preallocated at `max_seq`; position is
second-innermost so one head's history is contiguous (the order attention scans it).
Preallocated because a growing `Vec` reallocates and copies mid-generation.

**Size** = `layers × 2 × kv_heads × max_seq × head_dim × 4 B`:
24 × 2 × 2 × 4096 × 64 × 4 = **96 MiB** at 4096 context. With full multi-head (14 KV heads)
it would be 672 MiB. Bench output prints this number; it is not measured, it is arithmetic.

**Prefill vs decode** are the same function (`forward_cached`): prefill = empty cache + the
whole prompt (`seq = N`, matmuls are matrix×matrix, compute-bound); decode = one token
(`seq = 1`, every matmul is matrix×vector, **memory-bound**: the whole weight set streams
past for one token). The cache offset (`cache.len()`) feeds RoPE and the causal bound —
forgetting it is *the* classic KV-cache bug (fluent but subtly wrong output).

**Commit order.** `cache.advance(seq)` runs once after all layers. Advancing per layer
would make layer 1 see layer 0's *new* tokens as history.

**Verified:** incremental decode is bit-for-bit identical to a fresh full forward pass at
every step (`crates/model/tests/kv_cache.rs`).

## Sampling

Order: `logits → [mask] → temperature → softmax → top-k → top-p → renormalise → draw`.

- **Mask first.** Illegal tokens get `-inf`; `exp(-inf) = 0`, so they contribute nothing to
  the softmax denominator and cannot occupy a top-k slot. Filtering *after* top-k would
  silently shrink the candidate set.
- **Temperature divides logits**, then softmax. (Softmax-then-reweight is a different function.)
  `T → 0` approaches argmax; `T = 0` is special-cased to argmax.
- **Greedy ties** go to the lowest index, matching `torch.argmax`, so greedy output is
  comparable to the reference.
- **RNG** is xorshift64* — seedable, reproducible, not cryptographic. Same seed ⇒ same text.
- **Nucleus (top-p):** smallest set whose cumulative probability ≥ p.

## Questions

1. *Why does decode have lower arithmetic intensity than prefill?* One token ⇒ each weight is used for one multiply; prefill reuses each weight across N tokens.
2. *What if I cache Q too?* Never read again — wasted memory.
3. *Why does masking before temperature matter?* It doesn't change the result of temperature (−inf/T = −inf) but it does for top-k/top-p ordering — mask must precede them.
4. *Compute:* KV bytes per generated token (all layers)? 24 × 2 × 2 × 64 × 4 = 24,576 B = 24 KiB.
5. *What breaks sampling if the whole row is masked?* No legal token → `sample` returns `None` (handled, reported), never a NaN draw.
