#!/usr/bin/env bash
#reproduce the project results from the processed data in data/.
#   bash scripts/reproduce.sh test        #unit tests no API key
#   bash scripts/reproduce.sh baselines   #chkpt 2 baselines, 5 circuit splits
#   bash scripts/reproduce.sh agent       #chkpt 3 ReAct agent vs grid vs Optuna (needs API KEY)
#   bash scripts/reproduce.sh demo        #batch demo from a saved checkpoint, no API key
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-test}" in
  test)
    python baseline.py selftest
    python agent.py selftest            
    ;;
  baselines)
    for s in 0 1 2 3 4; do
      python baseline.py run --data-dir data --out "results_seed${s}" --seed "${s}"
    done
    python baseline.py report --results "results_seed*" --data-dir data --out report
    ;;
  agent)
    : "${GEMINI_API_KEY:?Set GEMINI_API_KEY first (export GEMINI_API_KEY=...). Never commit it.}"
    python agent.py compare --data-dir data --out results_agent
    python agent.py report --out results_agent
    ;;
  demo)
    python agent.py demo --data-dir data --out results_agent --seed 0 --n 10
    ;;
  *)
    echo "usage: bash scripts/reproduce.sh {test|baselines|agent|demo}" >&2
    exit 1
    ;;
esac
