# 01 — Loading weights and tokenizing

Code: [`crates/model/src/safetensors.rs`](../../crates/model/src/safetensors.rs),
[`config.rs`](../../crates/model/src/config.rs), [`weights.rs`](../../crates/model/src/weights.rs),
[`crates/tokenizer/`](../../crates/tokenizer/src/lib.rs).

## What it does

Turn two files on disk into (a) a bag of `f32` tensors and (b) a function
`&str → Vec<u32>` and back.

## safetensors, in one paragraph

```
bytes 0..8        u64 little-endian: N = header length
bytes 8..8+N      JSON: { "tensor.name": {"dtype":"BF16","shape":[..],"data_offsets":[a,b]}, ... }
bytes 8+N..end    raw tensor data; tensor i lives at [8+N+a, 8+N+b)
```

The file is `mmap`ed, so "loading" is parsing a ~25 KB header; bytes are only paged in
when a tensor is read. **Why a format like this and not pickle:** no code execution on
load, and no parsing pass over the data. The `unsafe` in `open` is the mmap; the
invariant is "nobody truncates the file while it is mapped".

**bf16 → f32** is a 16-bit left shift: bf16 is literally the top half of an f32 (same
8 exponent bits, 7 mantissa bits instead of 23). That is why the conversion is free and
why it is exact — there is no rounding going *up*. The checkpoint is 988 MB on disk
(494,032,768 params × 2 B) and **1,884.6 MiB** in RAM as f32 (× 4 B).

## The parameter count, derived (you should be able to do this on a whiteboard)

Per layer, with hidden 896, MLP 4864, 14 query heads, 2 KV heads, head_dim 64:

| tensor | shape | params |
|---|---|---|
| q_proj (+bias) | 896×896 (+896) | 803,712 |
| k_proj, v_proj (+bias) | 128×896 (+128) each | 229,632 |
| o_proj (no bias) | 896×896 | 802,816 |
| gate, up, down | 4864×896 each | 13,074,432 |
| 2 RMSNorm | 896 each | 1,792 |
| **layer** | | **14,912,384** |

24 layers = 357,897,216; embedding 151,936×896 = 136,134,656; final norm 896.
Sum = **494,032,768**, which is exactly what the loader reports. The embedding is
**tied**: the same matrix is the LM head, so it is counted once (and is 27.6% of all
parameters — that is why the LM-head matmul is a big slice of every decode step).

## Why `config.json` wins over the docs

Hard rule 4. Every `Config` field is read from the file and a missing field is an error,
never a default. A silent default is how you debug "the forward pass is wrong" for a day
when it was actually configured wrong.

## The tokenizer

Byte-level BPE (GPT-2 family).

1. **Pre-tokenize** with a regex into chunks (words, numbers, punctuation, whitespace).
   Qwen's pattern uses bare `\p{N}` — **every digit is its own chunk**, so
   `2025` → `2`,`0`,`2`,`5`. Using `\p{N}+` would merge them and break ID parity.
2. **Map bytes to printable unicode** (the GPT-2 `bytes_to_unicode` table: printable
   bytes map to themselves, the rest to `U+0100…`). This is why vocab entries look like
   `Ġthe` — `Ġ` is the space byte.
3. **Merge** greedily by rank: repeatedly merge the adjacent pair with the lowest merge
   rank until none applies.
4. **Decode** is the inverse and is trivial: id → bytes, concatenate, UTF-8 decode.

**Why byte-level:** no unknown token exists — any input is some byte sequence.
**The consequence that matters for the grammar crate:** a token is *bytes*, not
characters, and a single multi-byte character (an emoji, `é`) can be split across tokens.
Decoding token by token can therefore produce invalid UTF-8 mid-stream.

**Special tokens** (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) are matched *before*
BPE by `encode_with_specials`; plain `encode` treats the text literally. Mixing those up is
a prompt-injection bug waiting to happen (user text containing `<|im_end|>`).

**Vocab-size trap:** `config.json` says `vocab_size = 151936` (padded for alignment) but
the tokenizer defines 151,665 ids. The 271 padding logits have no token; sampling one would
panic in `decode`. Fixed in `cli::mask_padding_tail` and in the grammar's `apply_mask`.

## Verified how

`scripts/dump_tokenizer_ids.py` → `tests/fixtures/tokenizer_ids.json`; 42/42 exact ID
matches vs HuggingFace, including digits, unicode, and specials.

## Questions you should be able to answer

1. *Why is the checkpoint 988 MB but the process 1.9 GB?* bf16→f32 doubles it. (D-log: we chose f32 working precision for parity.)
2. *Why does `2025` become four tokens?* Per-digit pre-tokenizer regex.
3. *What breaks if you decode one token at a time?* Partial UTF-8; you must buffer bytes.
4. *Why mmap?* No copy, lazy paging, header-only parse.
5. *Compute:* params in `k_proj` across all layers? 24 × (128×896 + 128) = 2,755,584.
