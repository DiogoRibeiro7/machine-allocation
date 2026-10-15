#!/usr/bin/env bash
# Reproduce every result. Experiment 1 must run first; the others only need its output.
#   FROM=exp3 ./run_all.sh   skips the experiments listed before the first match.
#   QUICK=1 ./run_all.sh     smoke test with tiny budgets; writes to results_quick/.
#   PYTHON=python3.12 ...    interpreter to use (default: python, else python3).
# Each experiment logs to <results folder>/<name>.log. Exits 1 if any of them failed.
set -u
cd "$(dirname "$0")/experiments" || exit 1
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
out="../results${QUICK:+_quick}"
mkdir -p "$out"
failed=0

run() {  # run <log name> <script> [VAR=value ...]
  local name=$1 script=$2
  echo "== $name $(date +%T)"
  if ! env "${@:3}" "$PYTHON" "$script.py" > "$out/$name.log" 2>&1; then
    echo "   FAILED: see ${out#../}/$name.log"
    failed=1
  fi
}

started=${FROM:+0}
for e in exp1_reference_plan exp3_run_times exp2_frontier exp6_load exp4_breakdown exp5_scaling; do
  if [[ "${started:-1}" == 0 && "$e" != ${FROM:-}* ]]; then continue; fi
  started=1
  run "$e" "$e"
done
run exp2_frontier_v110 exp2_frontier VOLUME=1.1
run exp5b_large_long exp5b_large_long
echo "== done $(date +%T)"
exit $failed
