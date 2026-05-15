#!/usr/bin/env bash

set -euo pipefail

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

GAMMA=1.0

for OPPONENT in "${OPPONENTS[@]}"; do
  echo
  echo "======================================================================"
  echo "Solving DP best response against opponent: ${OPPONENT}"
  echo "======================================================================"

  poetry run python -m NashOrPass.leduc.agents.DP \
    --villain-policy "${OPPONENT}" \
    --gamma "${GAMMA}" \
    --save-filename "dp_policy_vs_${OPPONENT}.pkl"
done

echo
echo "======================================================================"
echo "Done solving all DP best responses"
echo "======================================================================"