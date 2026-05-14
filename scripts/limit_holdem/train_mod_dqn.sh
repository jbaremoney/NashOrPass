#!/usr/bin/env bash

set -euo pipefail

EPISODES=100000
EVAL_EVERY=1000
EVAL_EPISODES=500

GAMMA=1.0
SEED=0
LR=1e-4
HIDDEN_DIM=256

LEARNING_STARTS=1000
TRAIN_STEPS_PER_TRANSITION=1

# ------------------------------------------------------------
# Modded DQN policy choices
# ------------------------------------------------------------
# Standard DQN:
#   BEHAVIOR_POL="eps_greedy"
#   TARGET_POL="greedy"
#
# Softmax-policy / Expected-SARSA-ish:
#   BEHAVIOR_POL="softmax"
#   TARGET_POL="softmax"
#   TEMPERATURE=1.0
# ------------------------------------------------------------

BEHAVIOR_POL="softmax"
TARGET_POL="greedy" # change this to greedy... currently on policy
TEMPERATURE=1.0

poetry run python scripts/limit_holdem/train_mod_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent aggressive \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION" \
  --behavior-pol "$BEHAVIOR_POL" \
  --target-pol "$TARGET_POL" \
  --temperature "$TEMPERATURE"

poetry run python scripts/limit_holdem/train_mod_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent tight \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION" \
  --behavior-pol "$BEHAVIOR_POL" \
  --target-pol "$TARGET_POL" \
  --temperature "$TEMPERATURE"

poetry run python scripts/limit_holdem/train_mod_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent checkcall \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION" \
  --behavior-pol "$BEHAVIOR_POL" \
  --target-pol "$TARGET_POL" \
  --temperature "$TEMPERATURE"

poetry run python scripts/limit_holdem/train_mod_dqn.py \
  --episodes "$EPISODES" \
  --gamma "$GAMMA" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --opponent random \
  --seed "$SEED" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --learning-starts "$LEARNING_STARTS" \
  --train-steps-per-transition "$TRAIN_STEPS_PER_TRANSITION" \
  --behavior-pol "$BEHAVIOR_POL" \
  --target-pol "$TARGET_POL" \
  --temperature "$TEMPERATURE"