#!/usr/bin/env bash

set -euo pipefail

EPISODES=100
TEMP=0.2
HIST_CAPACITY=5
SEED=0

OPPONENTS=(
  uniform
  aggressive
  rank_aware_tight
  rank_aware_aggressive
)

AGENTS=(
  gpt
  claude
  grok
)

for agent in "${AGENTS[@]}"; do
  for opponent in "${OPPONENTS[@]}"; do
    echo
    echo "======================================================================"
    echo "Running LLM=${agent} vs opponent=${opponent}"
    echo "======================================================================"

    poetry run python scripts/leduc/play_llm.py \
      --agent "$agent" \
      --villain-policy "$opponent" \
      --episodes "$EPISODES" \
      --temperature "$TEMP" \
      --hist-capacity "$HIST_CAPACITY" \
      --seed "$SEED"
  done
done