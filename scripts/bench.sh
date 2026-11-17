#!/usr/bin/env bash
# The single source of every benchmark number in this repo (docs/RULES.md hard rule 3).
#
#   ./scripts/bench.sh                  # f32 and int8, 512 -> 128
#   ./scripts/bench.sh --quick          # smaller, for iterating
#
# Anything printed here is quotable. Anything not printed here is not.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-models/qwen2.5-0.5b}"
PROMPT_TOKENS=512
DECODE_TOKENS=128
RUNS=10

if [[ "${1:-}" == "--quick" ]]; then
  PROMPT_TOKENS=128
  DECODE_TOKENS=32
  RUNS=10
fi

if [[ ! -f "$MODEL/model.safetensors" ]]; then
  echo "no checkpoint at $MODEL — see docs/MODEL.md" >&2
  exit 1
fi

echo "building --release ..."
cargo build --release -q -p cli --bins

echo
echo "================================================================"
echo " rustc: $(rustc --version)"
echo " git:   $(git rev-parse --short HEAD 2>/dev/null || echo 'not a repo')"
echo "        $(git diff --quiet 2>/dev/null && echo 'clean' || echo 'DIRTY WORKING TREE — numbers are not reproducible')"
echo "================================================================"

for PRECISION in f32 int8; do
  echo
  echo "################ $PRECISION ################"
  FLAG=""
  [[ "$PRECISION" == "int8" ]] && FLAG="--int8"
  ./target/release/bench \
    --model "$MODEL" \
    --runs "$RUNS" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --decode-tokens "$DECODE_TOKENS" \
    $FLAG
done

echo
echo "################ constrained vs prompted-only JSON ################"
JSON_N=100
[[ "${1:-}" == "--quick" ]] && JSON_N=20
./target/release/json_compare --model "$MODEL" --n "$JSON_N"
./target/release/json_compare --model "$MODEL" --n "$JSON_N" --int8

echo
echo "################ correctness gate ################"
echo "Correctness is not a benchmark, it is a gate (docs/BENCHMARKS.md)."
if [[ -d tests/fixtures/qwen05b_f64 ]]; then
  ./target/release/parity "$MODEL" tests/fixtures/qwen05b_f64 2>/dev/null | tail -5
else
  echo "no reference fixtures — run scripts/dump_reference.py --dtype float64"
fi

echo
echo "################ NOT measured ################"
cat <<'EOF'
- llama.cpp head-to-head. Not installed here, so the comparison in
  docs/BENCHMARKS.md is still empty. Do not guess it.
- Perplexity is separate from this script — run it with:
    cargo run --release -p cli --bin perplexity
EOF
