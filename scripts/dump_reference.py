#!/usr/bin/env python3
"""
Dump per-layer reference activations from HuggingFace Qwen2.5-0.5B-Instruct.

This is the oracle the entire project is checked against. Everything in crates/ is
correct exactly insofar as it reproduces these arrays.

Python is allowed here and only here. Nothing in this file ever runs in the
inference path.

Usage:
    pip install torch transformers numpy safetensors
    python scripts/dump_reference.py
    python scripts/dump_reference.py --prompt "Hello" --out tests/fixtures/hello

Output:
    <out>/meta.json                        config, prompt, token ids, shapes
    <out>/tokens.npy                       int64 [seq]
    <out>/embed_out.npy                    f32   [seq, hidden]
    <out>/layer{NN}.<point>.npy            f32   per-layer activations
    <out>/final_norm.npy                   f32   [seq, hidden]
    <out>/logits.npy                       f32   [seq, vocab]

See docs/TESTING.md for how these are consumed.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Per-layer submodules to hook. Name on the left is the fixture name Rust looks up;
# the right side is the attribute path relative to model.model.layers[i].
HOOK_POINTS = {
    "input_layernorm": "input_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "attn_out": "self_attn",
    "post_attention_layernorm": "post_attention_layernorm",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
    "mlp_out": "mlp",
    "layer_out": "",  # the decoder layer itself — i.e. the residual stream after this layer
}


def resolve(module, path):
    """Walk a dotted attribute path. Empty path returns the module itself."""
    if not path:
        return module
    for part in path.split("."):
        module = getattr(module, part)
    return module


def to_numpy(x):
    """Hook outputs are sometimes tuples (decoder layers, attention). Take [0]."""
    if isinstance(x, tuple):
        x = x[0]
    return x.detach().to(torch.float32).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="tests/fixtures/qwen05b")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap the prompt in the model's chat template (use for instruct testing)",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # f32 everywhere — we are checking numerics, not speed
        attn_implementation="eager",  # sdpa/flash fuse ops and hide intermediates
    )
    model.eval()

    if args.chat_template:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = args.prompt

    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"prompt: {text!r}")
    print(f"tokens: {input_ids.shape[1]} -> {input_ids[0].tolist()}")

    captured = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, output):
            captured[name] = to_numpy(output)

        return hook

    # embeddings
    handles.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_out")))

    # per-layer
    for i, layer in enumerate(model.model.layers):
        for fixture_name, path in HOOK_POINTS.items():
            target = resolve(layer, path)
            handles.append(
                target.register_forward_hook(make_hook(f"layer{i:02d}.{fixture_name}"))
            )

    # final norm
    handles.append(model.model.norm.register_forward_hook(make_hook("final_norm")))

    with torch.no_grad():
        result = model(input_ids, use_cache=False)

    for h in handles:
        h.remove()

    captured["tokens"] = input_ids[0].to(torch.int64).cpu().numpy()
    captured["logits"] = result.logits[0].to(torch.float32).cpu().numpy()

    for name, arr in captured.items():
        np.save(out / f"{name}.npy", arr)

    cfg = model.config
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "templated_text": text,
        "used_chat_template": args.chat_template,
        "token_ids": input_ids[0].tolist(),
        "seq_len": int(input_ids.shape[1]),
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": getattr(cfg, "rope_theta", None),
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "hidden_act": cfg.hidden_act,
            "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        },
        "shapes": {k: list(v.shape) for k, v in captured.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {len(captured)} arrays to {out}/")
    print("\n--- CONFIG: reconcile these against docs/MODEL.md ---")
    print(json.dumps(meta["config"], indent=2))
    print("\nIf docs/MODEL.md disagrees with the above, the doc is wrong. Fix it.")

    # --------------------------------------------------------------------------
    # NOT CAPTURED, and you will want it:
    #
    # RoPE is applied inside the attention forward as a function call, not a
    # submodule, so no forward hook can see it. If q_proj matches but attention
    # doesn't, you need post-RoPE q and k to bisect further.
    #
    # To get them, monkeypatch the rotary function before the forward pass:
    #
    #     import transformers.models.qwen2.modeling_qwen2 as qwen2
    #     _orig = qwen2.apply_rotary_pos_emb
    #     def patched(q, k, cos, sin, *a, **kw):
    #         q2, k2 = _orig(q, k, cos, sin, *a, **kw)
    #         captured.setdefault("rope_q", []).append(to_numpy(q2))
    #         captured.setdefault("rope_k", []).append(to_numpy(k2))
    #         return q2, k2
    #     qwen2.apply_rotary_pos_emb = patched
    #
    # Layers fire in order, so index i of the list is layer i. The exact function
    # name and signature move between transformers versions — check the installed
    # source rather than trusting this comment.
    #
    # Add this when you need it (probably day 3 of week 1), not before.
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    main()
