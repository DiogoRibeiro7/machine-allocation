#!/usr/bin/env bash
# Reproduce every result. Experiment 1 must run first; the others only need its output.
# FROM=exp3 ./run_all.sh  skips the experiments listed before the first match.
cd "$(dirname "$0")/experiments"
started=${FROM:+0}
for e in exp1_reference_plan exp3_run_times exp2_frontier exp6_load exp4_breakdown exp5_scaling; do
  if [[ "${started:-1}" == 0 && "$e" != ${FROM}* ]]; then continue; fi
  started=1
  echo "== $e $(date +%T)"
  python3 "$e.py" > "../results/$e.log" 2>&1 || echo "   FAILED: see results/$e.log"
done
echo "== exp2_frontier at 110% volume $(date +%T)"
VOLUME=1.1 python3 exp2_frontier.py > ../results/exp2_frontier_v110.log 2>&1 || echo "   FAILED: see results/exp2_frontier_v110.log"
echo "== exp5b_large_long $(date +%T)"
python3 exp5b_large_long.py > ../results/exp5b_large_long.log 2>&1 || echo "   FAILED: see results/exp5b_large_long.log"
echo "== done $(date +%T)"
