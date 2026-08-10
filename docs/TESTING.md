# Testing strategy

The whole project is designed around having an **objective oracle at every stage**.
This document is how you use them.

## The core technique: layer-by-layer parity

The naive approach is to compare final logits against HuggingFace. When they don't
match — and they won't, for days — you know only that *something* among 24 layers ×
~10 operations is wrong. That's ~240 candidates and no way to bisect.

Instead: dump **every intermediate activation** from the reference model, and compare
at each point. Then a failure localises to one operation in one layer immediately.

```
scripts/dump_reference.py    # PyTorch + forward hooks → .npy per tensor
        │
        ▼
tests/fixtures/qwen05b/
    tokens.npy
    embed_out.npy
    layer00.input_layernorm.npy
    layer00.q_proj.npy
    layer00.rope_q.npy
    layer00.attn_out.npy
    layer00.residual1.npy
    layer00.post_attention_layernorm.npy
    layer00.mlp_out.npy
    layer00.residual2.npy
    ...
    final_norm.npy
    logits.npy
```

The Rust test walks the same forward pass and compares at every checkpoint, **failing
at the first divergence** rather than at the end:

```rust
#[test]
fn parity_layer_by_layer() {
    let fx = Fixtures::load("qwen05b");
    // ... forward pass, asserting after each named step
    assert_close(&got, &fx.get("layer00.rope_q"), 1e-4);  // first failure wins
}
```

### Tolerance

`1e-4` max absolute error is the target for `f32`. Some drift accumulates across 24
layers — that's expected and fine. What is *not* fine is a step change: if layer 7 is
at `1e-6` and layer 8 is at `0.3`, layer 8 has a bug. Watch the *trend*, not just the
final value.

Do not chase exact bit equality. Different summation orders in a matmul produce
different rounding, legitimately.

## Failure signatures

A short table you will actually use. When a specific layer diverges, the cause is
usually one of these:

| Symptom | Almost certainly |
|---|---|
| Divergence starts at `q_proj`/`k_proj` | Weight transpose. PyTorch stores `[out, in]` |
| `q` matches, `rope_q` doesn't | Wrong RoPE pairing (rotate-half vs. interleaved) or wrong position index |
| RoPE right, attention wrong | GQA head mapping. Check `kv = i / 7` and not `i % 2` |
| Attention right, residual wrong | Dropped or doubled residual add |
| Everything right, logits wrong | Tied embedding transpose, or forgot the final norm |
| Layer 0 perfect, layer 1 broken | KV cache write path, or state leaking between layers |
| Prefill right, decode wrong | Position index during decode, or the causal mask applied when it shouldn't be |
| Small constant offset everywhere | RMSNorm eps, or subtracting the mean (that's LayerNorm) |
| Output *nearly* coherent | GQA mapping backwards. The worst one — it looks like it works |

That last row is why parity testing exists. A model with the KV head mapping reversed
produces fluent, grammatical, subtly-wrong text. You cannot catch it by reading output.

## Per-crate oracles

**`tensor`** — property tests against NumPy references for matmul, RMSNorm, softmax,
SiLU. Plus invariants: `softmax` sums to 1, `rms_norm` output has unit RMS before the
weight is applied, `quantize → dequantize` error is bounded by `scale/2`.

**`tokenizer`** — encode a corpus, compare IDs against `tokenizer.json`'s own output
via the reference dump. Round-trip: `decode(encode(s)) == s` for ASCII, for multi-byte
UTF-8, for emoji, for the special tokens, and for a string with a lone `\xff` byte.
That last one is the case byte-level BPE exists to handle.

**`model`** — layer-by-layer parity, as above.

**`grammar`** — the strongest oracle in the project, because it's binary:

```rust
#[test]
fn constrained_output_always_parses() {
    for schema in SCHEMAS {           // nested objects, arrays, enums, unicode strings
        for seed in 0..500 {
            let out = generate_constrained(schema, seed);
            serde_json::from_str::<Value>(&out).expect("must parse");
            validate_against_schema(&out, schema).expect("must validate");
        }
    }
}
```

500 generations, zero failures. If one fails, the implementation is wrong — not
"mostly right." Do not relax this test to make it pass.

**Cross-check with an independent parser.** Validate with `serde_json` and, at least
once, with Python's `json` module. A bug shared between your DFA and your validator
would otherwise be invisible.

## Regression discipline

Parity tests run on **every commit**, and specifically after every optimization. The
workflow for making something faster is:

1. Parity passes (baseline)
2. Make the change
3. Parity passes again — **same tolerance, not a relaxed one**
4. Benchmark

Skipping step 3 is how you end up with a fast engine that produces subtly wrong output
and no idea which of your last twelve commits did it.

## What isn't tested, and say so

Be explicit in the writeup about coverage gaps rather than letting someone find them:

- Batch size > 1 — unimplemented in v1
- Contexts beyond the tested length
- Non-Qwen2 architectures
- Numerical behaviour under `f16`/`bf16` — v1 is `f32` throughout
- Value-level schema constraints (`maximum`, `maxLength`) — structure only

"Here's what I didn't test and why" is a strong answer. Being caught not knowing is not.
