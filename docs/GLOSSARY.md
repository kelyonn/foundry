# Glossary

Every term used in this repo. If you can't explain one of these in two sentences
without looking, you're not ready to be asked about it.

## Transformer

**Residual stream** — the `[seq, hidden]` tensor threaded through all 24 layers. Each
block *adds* to it rather than replacing it. Losing a residual add is a top-three
parity bug.

**RMSNorm** — `x / sqrt(mean(x²) + eps) * weight`. Like LayerNorm but with no mean
subtraction and no bias. Cheaper, and empirically just as good.

**RoPE (Rotary Position Embedding)** — encodes position by *rotating* pairs of
dimensions in q and k by a position-dependent angle. Relative position falls out of the
dot product for free. Applied to q and k only, never v. No learned position parameters.

**GQA (Grouped-Query Attention)** — fewer KV heads than query heads; several query
heads share one KV head. Here 14 query heads, 2 KV heads, so 7 queries per KV head.
**It shrinks the KV cache by 7×, not the FLOPs** — the attention math still runs per
query head. Know this distinction; it's a favourite interview question.

**MHA / MQA / GQA** — the family. Multi-head = one KV head per query head. Multi-query
= exactly one KV head total. Grouped-query = in between. GQA is the compromise that won.

**SwiGLU** — `down(silu(gate(x)) * up(x))`. Three matrices where a vanilla MLP has two;
`up` is a second projection multiplied in elementwise, not an activation. The multiply
is the "gating."

**SiLU / Swish** — `x * sigmoid(x)`. Smooth, non-monotonic near zero.

**Tied embeddings** — the output projection reuses the input embedding matrix,
transposed. Saves `vocab × hidden` parameters (~136M floats here). Present in Qwen2.5-0.5B.

**Causal mask** — additive `-inf` above the diagonal, so position `i` can't attend to
`j > i`. Needed during prefill; unnecessary during decode, where everything cached is
already in the past.

**Attention scale** — divide scores by `sqrt(head_dim)` before softmax, to keep the
softmax out of its saturated regime as `head_dim` grows.

**Logits** — pre-softmax scores over the vocabulary, one per token. `[151936]` here.

## Inference

**Prefill** — processing the whole prompt at once. Parallel over sequence,
**compute-bound**.

**Decode** — generating one token at a time, each depending on the last. Inherently
serial, **memory-bandwidth-bound** because you re-read the entire weight matrix per token.

**KV cache** — the stored keys and values for every previous position, so each new
token doesn't recompute them. Turns generation from O(n²) into O(n) per token. Its size
grows linearly with context and it will exceed the weights at long context.

**TTFT (time to first token)** — prefill latency. What a user perceives as "did it
hang?"

**Throughput vs. latency** — tok/s once running vs. time to the first one. Optimizing
one can hurt the other.

**Quantization** — storing weights in fewer bits (`f32` → `int8`). Mostly a memory-
bandwidth win, which is why it speeds up decode more than prefill.

**Scale (quantization)** — the `f32` multiplier converting an `int8` back to
approximately its original value. Per-output-channel here.

**Symmetric quantization** — zero maps to zero; no zero-point offset needed. One
multiply to dequantize instead of a multiply and an add.

**Perplexity** — `exp(mean negative log-likelihood)`. Roughly, how surprised the model
is by real text. Lower is better. Used here to price the accuracy cost of int8.

## Tokenization

**BPE (Byte Pair Encoding)** — start from bytes, repeatedly merge the most frequent
adjacent pair, following a learned merge list.

**Byte-level BPE** — the alphabet is the 256 byte values, not Unicode characters. Means
*any* input encodes, including invalid UTF-8. Also means a single token can be a
fragment of a multi-byte character — which is exactly why the grammar DFA must be
byte-level.

**Vocabulary** — token string ↔ ID map. 151,936 entries here.

**Special tokens** — `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`. Not produced by BPE
merges; inserted structurally.

**Chat template** — the formatting an instruct model was tuned on. Get it wrong and
quality silently degrades. Read it from `tokenizer_config.json`.

## Grammars and automata

**DFA (Deterministic Finite Automaton)** — states, an alphabet, one transition per
(state, symbol). Here the alphabet is `u8`.

**NFA** — nondeterministic: multiple transitions per symbol. Regexes compile to NFAs
(Thompson construction), which convert to DFAs (subset construction) at exponential
worst-case cost but fine in practice for these grammars.

**Accepting state** — a state where input so far is a complete valid string. **EOS is
legal only here** — the most common bug in naive constrained decoding.

**Dead state** — no path to any accepting state. Any token leading here is masked out.

**Regular language** — what a DFA can recognise. Notably *cannot* count or match
arbitrarily nested brackets.

**Context-free language** — needs a stack (pushdown automaton). Arbitrarily nested JSON
is context-free, which is why v1 bounds nesting depth to stay regular.

**Vocabulary index** — the precomputed map from DFA state → set of legal token IDs.
The central data structure of the grammar crate.

**Logit masking** — setting illegal tokens' logits to `-inf` *before* softmax, so they
receive exactly zero probability. Additive and pre-softmax; zeroing post-softmax does
not work.

**Constrained decoding** — the whole technique. Output is valid **by construction**,
not by persuasion.

## Systems

**safetensors** — weight format: a JSON header (names, dtypes, shapes, byte offsets)
followed by a raw tensor blob. Memory-mappable, no code execution on load, unlike
pickle.

**Memory-mapped I/O (mmap)** — map a file into the address space so the OS pages it in
on demand. Weights load instantly and pages are shared and evictable.

**SIMD** — one instruction, several values (NEON on Apple silicon, AVX on x86). The
main CPU lever for matmul.

**Memory-bandwidth-bound** — limited by how fast bytes move from RAM, not by arithmetic.
Decode is this. It's why quantization helps decode more than prefill.

**Blocking / tiling** — restructuring a matmul loop so working sets fit in cache. Often
a bigger win than SIMD, and simpler. Do it first.

**RSS (Resident Set Size)** — physical memory actually held. The honest memory number.
