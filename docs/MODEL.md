# Target model — Qwen2.5-0.5B-Instruct

> **Every number on this page is an expectation, not a fact.** They are written from
> memory of the Qwen2 architecture and have **not** been verified against a downloaded
> checkpoint. Task one of week 1 is to dump `config.json` and correct this file.
> When the file on disk disagrees with this page, **the file is right**.

## Why this model

- Small enough that a scalar `f32` forward pass completes in seconds, so week 1's
  parity loop is fast. Iteration speed in week 1 determines whether weeks 2–4 happen.
- Qwen2 is a clean modern architecture: RMSNorm, RoPE, grouped-query attention, SwiGLU,
  tied embeddings. Everything you'd need for Llama-3 is here.
- Single safetensors shard — no multi-file weight assembly.
- Instruct-tuned, so it will actually follow a "produce JSON" prompt, which makes the
  week 3 unconstrained-vs-constrained comparison meaningful.

Llama-3.2-1B is the stretch target. It differs mainly in vocab size, RoPE scaling, and
the absence of QKV bias. Do not attempt it until v1 is complete.

## Config — VERIFY ALL OF THESE

```jsonc
{
  "hidden_size":             896,       // d_model
  "intermediate_size":       4864,      // MLP inner dim
  "num_hidden_layers":       24,
  "num_attention_heads":     14,        // query heads
  "num_key_value_heads":     2,         // GQA — 7 query heads share each KV head
  "vocab_size":              151936,
  "max_position_embeddings": 32768,
  "rope_theta":              1000000.0,
  "rms_norm_eps":            1e-6,
  "tie_word_embeddings":     true,
  "hidden_act":              "silu"
}
```

Derived: `head_dim = hidden_size / num_attention_heads = 896 / 14 = 64`.

Note `num_attention_heads = 14` is not a power of two. If any code assumes it is,
that assumption breaks here — which is arguably a feature, since it surfaces the bug
now rather than on a different model.

**Qwen2 has bias terms on `q_proj`, `k_proj`, `v_proj` but not on `o_proj` or the MLP
projections.** Verify this by listing the tensor names in the safetensors header —
this is exactly the kind of detail that silently costs a day of parity debugging.

## Weight names

Expected keys in the safetensors header:

```
model.embed_tokens.weight                              [151936, 896]

model.layers.{0..23}.input_layernorm.weight            [896]
model.layers.{0..23}.self_attn.q_proj.weight           [896, 896]
model.layers.{0..23}.self_attn.q_proj.bias             [896]
model.layers.{0..23}.self_attn.k_proj.weight           [128, 896]     ← 2 heads × 64
model.layers.{0..23}.self_attn.k_proj.bias             [128]
model.layers.{0..23}.self_attn.v_proj.weight           [128, 896]
model.layers.{0..23}.self_attn.v_proj.bias             [128]
model.layers.{0..23}.self_attn.o_proj.weight           [896, 896]
model.layers.{0..23}.post_attention_layernorm.weight   [896]
model.layers.{0..23}.mlp.gate_proj.weight              [4864, 896]
model.layers.{0..23}.mlp.up_proj.weight                [4864, 896]
model.layers.{0..23}.mlp.down_proj.weight              [896, 4864]

model.norm.weight                                      [896]
```

`lm_head.weight` is **absent** when `tie_word_embeddings` is true — the output
projection reuses `embed_tokens`, transposed. Don't copy it; that's 545 MB of `f32`
for nothing.

**PyTorch linear layers store weights as `[out_features, in_features]`.** So
`y = x @ W.T`, not `x @ W`. Deciding once whether you transpose at load time or at
matmul time — and writing it down here — will save you a full debugging session.

## Forward pass

```
x = embed_tokens[token_ids]                                 [seq, 896]

for layer in 0..24:
    # ---- attention block ----
    h = rms_norm(x, layer.input_layernorm, eps)
    q = h @ Wq.T + bq                                       [seq, 896]
    k = h @ Wk.T + bk                                       [seq, 128]
    v = h @ Wv.T + bv                                       [seq, 128]

    reshape:  q -> [14, seq, 64]    k, v -> [2, seq, 64]
    q, k = rope(q, k, positions, theta=1e6)
    append k, v to KV cache for this layer

    for each query head i in 0..14:
        kv = i / 7                                          # integer division
        scores = q[i] @ K[kv].T / sqrt(64)                  [seq, cached_len]
        scores += causal_mask                                # prefill only
        attn[i] = softmax(scores) @ V[kv]                   [seq, 64]

    a = concat(attn) @ Wo.T                                 [seq, 896]
    x = x + a                                                # residual

    # ---- MLP block ----
    h = rms_norm(x, layer.post_attention_layernorm, eps)
    g = h @ Wgate.T                                         [seq, 4864]
    u = h @ Wup.T                                           [seq, 4864]
    m = (silu(g) * u) @ Wdown.T                             [seq, 896]
    x = x + m                                                # residual

x = rms_norm(x, model.norm, eps)
logits = x @ embed_tokens.T                                 [seq, 151936]
```

## The operations, precisely

**RMSNorm** — no mean subtraction, no bias. That is the whole difference from LayerNorm.

```
rms_norm(x)[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]
```

**SiLU** (a.k.a. Swish): `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`

**SwiGLU MLP:** `down(silu(gate(x)) * up(x))`. The elementwise multiply is the "gating"
— `up` is *not* an activation, it's a second projection multiplied in. Three matrices
where a vanilla MLP has two.

**RoPE** — rotate pairs of dimensions by a position-dependent angle:

```
for pair index j in 0..head_dim/2:
    freq  = 1 / (theta ^ (2j / head_dim))
    angle = position * freq
    (x[2j], x[2j+1]) → (x[2j]·cos − x[2j+1]·sin,
                        x[2j]·sin + x[2j+1]·cos)
```

Applied to **q and k only** — never to v. Two things to check against the reference,
because both are easy to get wrong and both produce plausible-but-wrong output:

1. **Pairing scheme.** HuggingFace's implementation pairs dimension `j` with
   `j + head_dim/2` (the "rotate-half" formulation), not `2j` with `2j+1` (the
   interleaved formulation from the paper). They are not equivalent. Match whatever
   the reference dump shows.
2. **Position indexing during decode.** The single token being decoded is at absolute
   position `cached_len`, not `0`. Off-by-one here degrades quality gradually rather
   than obviously.

**Causal mask** — prefill only. `mask[i][j] = -inf where j > i`. Additive, applied
before softmax. During decode there is nothing to mask.

**Attention scale** — `1/sqrt(head_dim)` = `1/8` here, applied to the scores before
softmax.

## Chat template

The instruct model expects Qwen's ChatML-style format. Read the exact template from
`tokenizer_config.json` rather than hardcoding it from this page:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

Special token IDs (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) come from
`tokenizer.json`. `<|im_end|>` is the practical stop token for chat.
