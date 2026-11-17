#!/usr/bin/env bash
# The 60-second demo: a prompt goes in, schema-valid JSON comes out, live.
#
#   ./scripts/demo.sh            # run it
#   vhs docs/readme/demo.tape    # record it to docs/readme/demo.gif (needs `brew install vhs`)
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -q -p cli --bin foundry

SCHEMA='{"type":"object","properties":{"name":{"type":"string","maxLength":20},"age":{"type":"integer"},"city":{"enum":["London","Tokyo","Delhi"]},"active":{"type":"boolean"}}}'

say() { printf '\n\033[1m# %s\033[0m\n' "$*"; sleep 1; }

say "1. Unconstrained: a small model asked for JSON, with no guardrails"
./target/release/foundry --chat --int8 -q -t 0.9 --seed 7 -n 60 \
  -p "Describe a fictional person as JSON with name, age, city and active."

say "2. Same prompt, constrained by a JSON Schema compiled to a byte DFA"
printf '%s\n' "$SCHEMA"
./target/release/foundry --chat --int8 -t 0.9 --seed 7 -n 80 --schema "$SCHEMA" \
  -p "Describe a fictional person as JSON with name, age, city and active."

say "3. Every logit that would leave the language was set to -inf before sampling"
