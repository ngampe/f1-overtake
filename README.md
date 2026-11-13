# Telemetry-Based Detection and Prediction of F1 Overtaking Events

IDAI.780.01 Predicts whether an overtake attempt succeeds
using only telemetry available at or before T0 using FastF1 data

## Setup
    python -m venv .venv
    source .venv/bin/activate      
    pip install -r requirements.txt

## Usage
    python baseline.py selftest                       # unit test
    python baseline.py build --years 2023 --rounds 1  # download + detect attempts
    python baseline.py run --data-dir data --out results
    python baseline.py run --synthetic                

`ff1cache/` and `data/` are git-ignored. `results/` is committed.

## Log of detection-threshold changes
Record each change to `DetectCfg` with the attempt count and success rate it produced.

## ReAct tuning agent (Ckpt 3)
    export GEMINI_API_KEY=...   # never commit your key
    python agent.py compare     # agent vs grid vs Optuna, seeds 0-4
    python agent.py report      # table + figure
    python agent.py demo --seed 0 --n 10
