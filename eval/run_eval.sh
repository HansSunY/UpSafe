#!/usr/bin/env bash
# Generate responses and judge each requested benchmark at every safety temperature.
set -euo pipefail
MODEL="${1:?Usage: run_eval.sh <model_path> <tag>}"
TAG="${2:?Usage: run_eval.sh <model_path> <tag>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TAUS="${TAUS:-0.5}"
BENCHES="${BENCHES:-jbb strong wildchat wj xs}"
for TAU in $TAUS; do
  for BENCH in $BENCHES; do
    OUT="results/${TAG}_tau${TAU}_${BENCH}.json"
    python "$HERE/QA_gen.py" --model_path "$MODEL" --bench "$BENCH" \
      --tau "$TAU" --output_file "$OUT"
    if [[ "$BENCH" == "xs" ]]; then
      python "$HERE/judge/refuse_judge.py" --QA_path "$OUT" --save "${OUT%.json}_labels.json"
    else
      python "$HERE/judge/QA_eval.py" --QA_path "$OUT" --save "${OUT%.json}_scores.json"
    fi
  done
done
