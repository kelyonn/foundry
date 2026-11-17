# 02 — The Qwen2 forward pass

Code: [`crates/model/src/forward.rs`](../../crates/model/src/forward.rs),
[`crates/tensor/src/ops.rs`](../../crates/tensor/src/ops.rs).

## One layer, with shapes (seq = S)

```
x            [S, 896]
n  = RMSNorm(x)                         [S, 896]
q  = n @ Wq^T + bq                      [S, 896]   = 14 heads × 64
k  = n @ Wk^T + bk                      [S, 128]   =  2 heads × 64
v  = n @ Wv^T + bv                      [S, 128]
split heads → q [14,S,64]  k,v [2,S,64]
RoPE on q and k (never v)
append k,v to the cache
per query head h: kv = h / 7;  scores = q·k / √64  (causal) → softmax → Σ w·v
concat heads                            [S, 896]
x  = x + attn @ Wo^T                    [S, 896]   (residual 1)
n2 = RMSNorm(x)
x  = x + Wdown( silu(n2 @ Wgate^T) * (n2 @ Wup^T) )   (residual 2)
```

Then after 24 layers: final RMSNorm → `logits = x @ E^T` with `E` the tied embedding,
`[S, 151936]`. Weights are stored `[out, in]` (PyTorch `nn.Linear`), so `linear` computes
dot products of rows with rows — no transpose copy anywhere.

## The five things people get wrong (and where each is handled)

**RMSNorm.** `x / sqrt(mean(x²) + eps) · w`. No mean subtraction, no bias; eps *inside* the
sqrt. Sum of squares accumulated in f64 because 896 f32 terms lose too much precision.

**RoPE.** Rotates pairs of dims in q and k by an angle ∝ position, with frequency
`θ^(-2i/64)`, θ = 1,000,000. Consequence: the dot product `q_p · k_j` depends only on the
*relative* offset `p − j`. HuggingFace pairs dim `i` with `i + 32` ("rotate half"), **not**
`2i` with `2i+1`. Both are valid rotations and both train fine; they are *incompatible with
each other*, so the wrong one gives plausible-magnitude garbage. Not applied to `v`:
v carries content, not position.

**GQA.** 14 query heads share 2 KV heads: head `h` uses KV head `h / 7` (not `h % 2`).
- **Cache size:** 7× smaller than MHA (2 KV heads vs 14).
- **FLOPs:** *unchanged* — all 14 query heads still do a full dot product against the keys.
GQA is a memory-bandwidth optimisation, not a compute one. That sentence is the interview
answer to "why does GQA shrink the cache but not the FLOPs".

**Causal mask.** Done by *not computing* positions after `p` (scores has length `p+1`),
not by adding `-inf`. Cheaper and nothing can leak as NaN.

**SwiGLU.** Two up-projections (gate, up) and one down: `down(silu(gate) ⊙ up)`.
That is three matrices of 896×4864, not the two a classic 4×-hidden MLP has, so the
inner width is a free design choice rather than "4× hidden" (4864 ≈ 5.43× here).
MLP params per layer: 3 × 4864 × 896 = 13.07 M of 14.9 M — **87.7% of each layer**.

## FLOPs per token (decode, ignoring attention-over-history)

Every weight is used once per token as a multiply-add: ≈ 2 × 494M ≈ **0.99 GFLOP**.
Attention over history adds `4 · layers · heads · head_dim · ctx` ≈ 4×24×14×64×ctx
= 86 K·ctx FLOPs (e.g. 44 MFLOP at ctx=512) — small next to the weights until contexts are
very long. This is why decode is **memory-bound**: ~2 GB of f32 weights must stream past for
about 1 GFLOP of work (arithmetic intensity ≈ 0.5 FLOP/byte).

## How it was verified

`cargo run --release --bin parity` compares every activation of all 24 layers against an
f64 HuggingFace reference (`scripts/dump_reference.py --dtype float64`): final logits
relative error **1.78e-6** after the NEON work (1.48e-6 with the scalar kernel); argmax
agrees. Five real bugs were found by this harness (DECISIONS D14–D18) — read those before
touching the forward pass.

## Questions

1. *Why does the shape still line up if I transpose a weight?* For q_proj/o_proj (896×896) it does; k_proj (128×896) is where it explodes. That is why there's a non-square test.
2. *What happens if RoPE offset is forgotten in decode?* Every new token thinks it is at position 0 — output stays fluent but wrong.
3. *Compute:* KV cache bytes at ctx 4096? 24 × 2 × 2 × 4096 × 64 × 4 = 100,663,296 = **96 MiB** (MHA would be 672 MiB).
4. *Why prenorm?* Residual stream stays un-normalised; trains stably at depth.
5. *Why f64 accumulation in the reference kernel?* D14: f32 sequential sum cost 1.3e-4 on q_proj — the whole budget.
