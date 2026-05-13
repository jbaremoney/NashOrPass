#!/usr/bin/env bash

set -euo pipefail

EPISODES=30000
EVAL_EVERY=1000
EVAL_EPISODES=500
GAMMA=1.0
SEED=0
LR=1e-4
HIDDEN_DIM=256
LEARNING_STARTS=1000
TRAIN_STEPS_PER_TRANSITION=1

poetry run python scripts/limit_holdem/train_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent aggressive \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION"

poetry run python scripts/limit_holdem/train_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent tight \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION"

poetry run python scripts/limit_holdem/train_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent checkcall \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION"

poetry run python scripts/limit_holdem/train_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent random \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION"
