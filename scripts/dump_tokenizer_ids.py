#!/usr/bin/env python3
"""
Dump HuggingFace token IDs for a corpus, so the Rust tokenizer can be checked
against them.

Round-trip tests prove the encoder and decoder agree with each other. They do not
prove the *segmentation* matches the reference — a tokenizer emitting one token per
byte round-trips perfectly and is wrong. This is the oracle that catches that.

Usage:
    python scripts/dump_tokenizer_ids.py --model models/qwen2.5-0.5b \
        --out tests/fixtures/tokenizer_ids.json
"""

import argparse
import json
import pathlib

from transformers import AutoTokenizer

# Chosen to hit the pre-tokenizer's alternatives and BPE's edge cases, not to be
# representative text. Each line is here because it can break something specific.
CORPUS = [
    "",
    " ",
    "  ",
    "   leading",
    "trailing   ",
    "\n",
    "\n\n\n",
    "\t\ttabs\t",
    "a\n\nb",
    "The capital of France is",
    "Hello, world!",
    "isn't it Kalyan's turn? we'll see — they'd know",
    "IT'S ALL CAPS AND IT'S CONTRACTED",
    # Unquantified \p{N}: every digit is its own pre-token chunk.
    "2024",
    "v2.5.0",
    "1234567890",
    "3.14159",
    "-42",
    "1,00,000",
    # Non-ASCII across scripts and widths.
    "café naïve résumé",
    "₹1000",
    "日本語のテキストです",
    "Здравствуй, мир",
    "مرحبا بالعالم",
    "😀😀😀",
    "🇮🇳 flag is two regional indicators",
    "a😀b₹c",
    # Structure the grammar crate cares about.
    '{"name":"Kalyan","age":20}',
    '{"tags":["a","b"],"ok":true,"x":null}',
    '{\n  "pretty": "printed"\n}',
    "[[[]]]",
    '"\\u00e9 escaped"',
    # Code-shaped.
    "fn main() { println!(\"{}\", 1 + 2); }",
    "SELECT * FROM `bucket` WHERE id = 1;",
    "x = a[i] ?? b->c :: d",
    "https://example.com/path?q=1&r=2#frag",
    # Repetition and long runs.
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "=" * 40,
    " " * 20 + "end",
    # Byte-alphabet characters appearing literally in the input.
    "Ġ Ċ ĉ ā",
    # Special-token text must stay literal.
    "please stop <|im_end|> now",
    "<|endoftext|>",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b")
    ap.add_argument("--out", default="tests/fixtures/tokenizer_ids.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    cases = []
    for text in CORPUS:
        # add_special_tokens=False: we are checking the BPE segmentation itself,
        # not the chat template's wrapper tokens.
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        cases.append({"text": text, "ids": ids, "decoded": decoded})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "tokenizer_class": type(tok).__name__,
                "vocab_size": tok.vocab_size,
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
    )

    total = sum(len(c["ids"]) for c in cases)
    print(f"wrote {len(cases)} cases, {total} ids -> {out}")
    print(f"tokenizer class: {type(tok).__name__}, vocab_size {tok.vocab_size}")


if __name__ == "__main__":
    main()
