# Day one

The first session. Nothing here is Rust — it's all setup and reconnaissance, and it
exists because week 1 dies more often from a wrong assumption on day one than from
anything hard later.

Target: end the session knowing the model is real, the config is confirmed, and the
oracle produces files.

## 1. Get the model

```bash
pip install torch transformers numpy safetensors huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b
```

~1 GB. It goes in `models/`, which is gitignored.

## 2. Confirm the config

```bash
cat models/qwen2.5-0.5b/config.json
```

Open [MODEL.md](MODEL.md) beside it and check **every value**. That page was written
from memory and is explicitly untrusted. Where they differ, edit the doc.

Pay attention to: `num_key_value_heads` (the GQA ratio), `rope_theta`,
`tie_word_embeddings`, and `intermediate_size`.

## 3. List the tensors

```bash
python -c "
from safetensors import safe_open
import json
f = safe_open('models/qwen2.5-0.5b/model.safetensors', framework='pt')
keys = sorted(f.keys())
print(len(keys), 'tensors')
for k in keys[:20] + keys[-5:]:
    print(f'{k:60s} {list(f.get_slice(k).get_shape())}')
print('lm_head present:', any('lm_head' in k for k in keys))
print('qkv bias present:', any('q_proj.bias' in k for k in keys))
"
```

Two things you are specifically checking:

- **Is `lm_head.weight` absent?** If so, embeddings are tied and the output projection
  reuses `embed_tokens` transposed.
- **Do `q_proj.bias` / `k_proj.bias` / `v_proj.bias` exist, and does `o_proj.bias`
  not?** Qwen2 is expected to have QKV bias only. Confirm rather than assume.

Update [MODEL.md](MODEL.md) with the real shapes.

## 4. Run the oracle

```bash
python scripts/dump_reference.py
```

You should get ~290 `.npy` files in `tests/fixtures/qwen05b/` plus a `meta.json`. The
script prints the config at the end — reconcile it against the doc one more time.

Sanity check the output makes sense:

```bash
python -c "
import numpy as np, json
m = json.load(open('tests/fixtures/qwen05b/meta.json'))
lg = np.load('tests/fixtures/qwen05b/logits.npy')
print('logits', lg.shape)
print('argmax of last position:', lg[-1].argmax())
"
```

Decode that token ID. For the prompt `The capital of France is`, it should be
something like ` Paris`. If it isn't, something is wrong with the load or the
tokenization and you want to know **now**, not on day four.

## 5. Read the safetensors header by hand

Before writing a parser, look at what you're parsing:

```bash
python -c "
import struct, json
with open('models/qwen2.5-0.5b/model.safetensors','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
print('header bytes:', n)
print('data starts at:', 8 + n)
k = 'model.embed_tokens.weight'
print(k, hdr[k])
"
```

That's the entire format: a `u64` length, a JSON header of `{name: {dtype, shape,
data_offsets}}`, then the raw blob. Offsets are relative to the start of the data
section, not the file. Writing the Rust parser is now a 40-line job instead of a
guessing game.

## 6. Scaffold (Claude's job)

```bash
cargo new --lib crates/tensor
cargo new --lib crates/tokenizer
cargo new --lib crates/model
cargo new --lib crates/grammar
cargo new --bin  crates/cli
```

Workspace `Cargo.toml`, module skeletons, the `assert_close` / `Fixtures` test helper
that loads `.npy` files. Ask for it — this is explicitly not your column, and it's an
hour of your time you should not spend.

## 7. Then stop

Do not start the forward pass tonight. Day two starts with the tokenizer, because a
tokenizer bug presents as a logits bug and you want it eliminated first.

---

## Day one exit criteria

- [ ] Model downloaded, `config.json` read with your own eyes
- [ ] [MODEL.md](MODEL.md) corrected against reality
- [ ] Tensor names and shapes listed and confirmed
- [ ] `dump_reference.py` runs, fixtures on disk
- [ ] Reference logits argmax decodes to something sensible
- [ ] safetensors header structure understood
- [ ] Cargo workspace exists and `cargo test` runs (zero tests is fine)

If all seven are done, week 1 is on schedule. If the model won't download or
`transformers` won't install, that is today's entire problem — solve it and stop.
