#!/usr/bin/env bash

set -euo pipefail

EPISODES=100000
EVAL_EVERY=5000
EVAL_EPISODES=5000
ALPHA=0.1
GAMMA=1.0
EPSILON=1.0
EPSILON_MIN=0.05
EPSILON_DECAY=0.99995
SEED=0

OPPONENTS=(
  uniform
  check_call
  tight
  aggressive
  always_raise
  always_fold
  rank_aware_tight
  rank_aware_aggressive
)

for OPPONENT in "${OPPONENTS[@]}"; do
  for OBS_MODE in perfect imperfect; do
    echo
    echo "======================================================================"
    echo "Training tabular Q: obs_mode=${OBS_MODE}, opponent=${OPPONENT}"
    echo "======================================================================"

    poetry run python scripts/leduc/train_tab_q.py \
      --episodes "$EPISODES" \
      --eval-every "$EVAL_EVERY" \
      --eval-episodes "$EVAL_EPISODES" \
      --villain-policy "$OPPONENT" \
      --obs-mode "$OBS_MODE" \
      --alpha "$ALPHA" \
      --gamma "$GAMMA" \
      --epsilon "$EPSILON" \
      --epsilon-min "$EPSILON_MIN" \
      --epsilon-decay "$EPSILON_DECAY" \
      --seed "$SEED"
  done
done