# Architecture

## Crate boundaries

Five crates. The boundaries exist so that each one can be tested against an oracle
independently — if the tokenizer and the model are one crate, a tokenization bug
presents as a wrong-logits bug and costs you a day.

```
cli ──────► model ──────► tensor
             │  ▲
             │  └──────── tokenizer
             ▼
          grammar ──────► tokenizer
```

| Crate | Owns | Oracle |
|---|---|---|
| `tensor` | Storage, shapes, matmul, RMSNorm, softmax, SiLU, quantization | Property tests + NumPy reference |
| `tokenizer` | Byte-level BPE encode/decode, vocab, merges | ID parity vs. `tokenizer.json` |
| `model` | Weight loading, Qwen2 forward pass, KV cache, sampling | Per-layer activation parity vs. HF |
| `grammar` | Schema → DFA → vocabulary index → logit mask | 100% parse rate |
| `cli` | Arg parsing, the run loop, output | Manual |

`grammar` depends on `tokenizer` because the vocabulary index is a function of the
tokenizer's byte sequences. It must **not** depend on `model` — the mask is computed
from DFA state alone and knows nothing about logits' provenance. Keeping that boundary
clean is what lets you test constrained decoding without running the model.

---

## Tensor layout

**Everything is row-major, contiguous, `f32` in v1.** No strides, no views, no
broadcasting engine. A `Tensor` is a `Vec<f32>` plus a shape.

This is a deliberate simplification and it is the right one for a four-week build. The
cost is extra copies on transpose; the benefit is that you never debug a stride bug,
which is the single most common time sink in a hand-written tensor library.

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,   // row-major, contiguous, no strides
}
```

Quantized weights are a separate type, not a variant of `Tensor` — they cannot be fed
to the same kernels, and the type system should say so:

```rust
pub struct QTensor {
    data: Vec<i8>,
    scales: Vec<f32>,    // one per output channel (row)
    shape: Vec<usize>,
}
```

### Attention tensor convention

Batch size is fixed at 1 in v1, so the batch dimension is elided everywhere. The
convention is **`[heads, seq, head_dim]`**, written `hsd` in variable names.

- `q_hsd` — `[14, seq, 64]`
- `k_hsd`, `v_hsd` — `[2, seq, 64]` (grouped-query: only 2 KV heads)
- Query head `i` reads KV head `i / 7`. Write this mapping down and test it in
  isolation; getting it backwards produces output that is *almost* coherent, which is
  the worst possible failure mode because you won't notice for hours.

---

## KV cache

Preallocated at `max_seq_len`, never reallocated during a generation. Growth mid-
generation means a realloc and a copy at exactly the moment you're measuring decode
throughput.

```
kv_cache: [layer][2][kv_head][max_seq][head_dim]
                  ↑
                  0 = keys, 1 = values
```

One `Vec<f32>` per layer, indexed manually. `seq_len` is a cursor, not a length —
the buffer is full-size from the start.

**Memory:** `2 × layers × kv_heads × max_seq × head_dim × 4 bytes`. With the expected
Qwen2.5-0.5B config that is `2 × 24 × 2 × max_seq × 64 × 4` = **24 KB per token**.
At 4096 context that's ~100 MB — larger than the int8 weights. Know this number; it is
exactly the kind of thing an interviewer will ask you to compute on the spot, and it's
the reason grouped-query attention exists.

### Prefill vs. decode

Two different code paths with two different shapes, and conflating them is a classic
bug:

- **Prefill** — the whole prompt at once. `q` has `seq > 1`. Attention is
  `[seq, seq]` and needs a **causal mask**. Compute-bound.
- **Decode** — one token. `q` has `seq == 1`. Attention is `[1, cached_len]` and needs
  **no mask at all**, because everything in the cache is by definition in the past.
  Memory-bandwidth-bound.

Benchmark them separately. They have different bottlenecks and reporting a single
"tok/s" number conflates them — which is itself a good thing to point out in an
interview.

---

## Forward pass data flow

```
token_ids
   │
   ▼
embed_tokens                                    [seq, 896]
   │
   ├──────────────────────────────┐  ← residual stream
   ▼                              │
 for each of 24 layers:           │
   input_layernorm (RMSNorm)      │
   │                              │
   q_proj, k_proj, v_proj (+bias) │            q:[seq,896] k,v:[seq,128]
   RoPE(q, k, position)           │
   write k,v → KV cache           │
   GQA attention (+causal mask)   │
   o_proj                         │
   └──────────────► add ──────────┘
   │
   ├──────────────────────────────┐
   ▼                              │
   post_attention_layernorm       │
   gate_proj, up_proj             │            [seq, 4864]
   SiLU(gate) * up                │
   down_proj                      │            [seq, 896]
   └──────────────► add ──────────┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head  (tied to embed_tokens — transpose, don't copy)   [seq, 151936]
   │
   ▼
logits ──► grammar mask ──► sampler ──► token
```

The residual stream is the thing to keep clear in your head: attention and MLP write
*into* it, they don't replace it. If your parity check fails at layer 0 but the
individual ops are right, you've almost certainly dropped a residual add.

---

## Quantization

int8, symmetric, per-output-channel. Weights only — activations stay `f32`.

```
scale[i]  = max(|W[i, :]|) / 127
Wq[i, j]  = round(W[i, j] / scale[i])
```

Dequantize on the fly inside the matmul inner loop rather than materializing the `f32`
weight matrix — the point of quantization is the memory bandwidth saving, and
materializing throws it away.

**Why symmetric:** no zero-point to track, so the dequant is one multiply instead of a
multiply and an add. **Why per-channel:** a single scale for the whole matrix is
dominated by outlier channels and costs real accuracy. **Why weights only:** activation
quantization needs calibration data and a whole extra pipeline, for a fraction of the
benefit at this model size. Each of those is a legitimate interview answer — make sure
you can give the *why*, not just the *what*.

---

## Grammar integration point

Exactly one function, called between logits and sampling:

```rust
// mask[i] = 0.0 if token i is legal in the current DFA state, else f32::NEG_INFINITY
fn apply_mask(logits: &mut [f32], state: DfaState, index: &VocabIndex);
```

Additive `-inf` masking, applied *before* softmax. Not multiplicative, not post-
softmax, not "set to zero" — a zeroed logit is still a large probability after
softmax. Get this wrong and constrained decoding will *mostly* work, which means you
won't catch it until the demo.

After a token is sampled, advance the DFA by that token's byte sequence. Details in
[GRAMMAR.md](GRAMMAR.md).
