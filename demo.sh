#!/usr/bin/env bash
# Paused, step-by-step walkthrough for live teaching (like record.sh in the Simmons repo).
# Enter = run next step · s = skip · q = quit.   Extra flags pass through, e.g.  ./demo.sh --device cpu
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
declare -a TITLES=(
  "doctor     · check the toolchain"
  "1 load     · find the residual writers"
  "2 data     · split + filter the prompt sets"
  "3 capture  · record the residual stream"
  "4 directions · difference of means per layer"
  "5 select   · bypass / induce / KL for every candidate"
  "6 edit     · orthogonalize the weights"
  "7 verify   · hooks == weights?"
  "8 eval     · held-out before/after"
  "induce     · add the direction to a harmless prompt"
)
declare -a CMDS=(
  "doctor" "run --step 1" "run --step 2" "run --step 3" "run --step 4"
  "run --step 5" "run --step 6" "run --step 7" "run --step 8" "induce"
)
for i in "${!CMDS[@]}"; do
  printf "\n\033[1;38;5;111m▶ %s\033[0m   [Enter] run  [s] skip  [q] quit " "${TITLES[$i]}"
  read -r key
  case "$key" in q) exit 0 ;; s) continue ;; esac
  $PY abliterate.py ${CMDS[$i]} "$@"
done
printf "\n\033[1mDone.\033[0m Talk to both models:  ./chat before   ./chat after\n"
