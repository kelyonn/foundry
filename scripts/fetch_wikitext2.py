#!/usr/bin/env python3
"""
Pull a subset of the real wikitext-2 test split for the perplexity harness.

wikitext-2 (Merity et al., 2016) is the corpus docs/ROADMAP.md names for week 2's
perplexity measurement. The built-in fallback in
crates/cli/src/bin/perplexity.rs is honestly labelled as "not wikitext-2" — this
script produces the real thing so that label can be retired.

Usage:
    pip install datasets
    python scripts/fetch_wikitext2.py

Output:
    tests/fixtures/wikitext2_test.txt — a slice of the real test split, concatenated
"""

import pathlib

from datasets import load_dataset

# Enough non-blank text to comfortably exceed the 256-token score budget the
# perplexity binary uses by default, without dumping the entire test split (the
# corpus is small, but there's no reason to commit more of it than is used).
MIN_CHARS = 8000

out = pathlib.Path("tests/fixtures/wikitext2_test.txt")
out.parent.mkdir(parents=True, exist_ok=True)

print("loading wikitext-2-raw-v1 (test split) from the HF Hub ...")
# The original `wikitext` repo uses a legacy loading script the current
# `datasets` library refuses to run; `Salesforce/wikitext` is the same corpus
# republished in plain Parquet, which needs no script.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

# The raw split is mostly empty lines and bare section headers (" = Title = \n").
# Keep only real prose paragraphs, in document order, until there's enough text.
lines = []
total = 0
for row in ds:
    text = row["text"]
    stripped = text.strip()
    if not stripped or stripped.startswith("="):
        continue
    lines.append(stripped)
    total += len(stripped)
    if total >= MIN_CHARS:
        break

corpus = "\n\n".join(lines)
out.write_text(corpus, encoding="utf-8")

print(f"wrote {len(corpus)} chars ({len(lines)} paragraphs) -> {out}")
print(f"\nfirst 300 chars:\n{corpus[:300]}")
