#!/usr/bin/env bash
# Reproduce every table and figure of the article.  Usage: bash reproduce.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--quick" ]]; then
  ARGS="--reps 2000 --boot 10 --boot-reps 300"; SIMREPS=1000
else
  ARGS="--reps 20000 --boot 200 --boot-reps 2000"; SIMREPS=10000
fi

echo "== Validation on the household populations"
python3 run_all.py solomon vanuatu belize kampala $ARGS
python3 make_tables.py
python3 make_figures.py

echo "== Simulation study"
cd simulation
python3 sim.py $SIMREPS
python3 tcov.py
python3 boot.py
python3 planning.py
python3 planning2.py
python3 figs.py
python3 tabs2.py
cd ..
echo "done: results/, tables/, figures/, simulation/"
