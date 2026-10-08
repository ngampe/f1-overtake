Circuit-level held-out test, mean ± sd over 5 different circuit splits.

| model | ROC-AUC | PR-AUC | F1 | ECE | Brier | PR-AUC random split | PR-AUC gap (random - circuit) | Extra gap vs logistic regression |
|---|---|---|---|---|---|---|---|---|
| Gap heuristic | 0.831 ± 0.033 | 0.348 ± 0.068 | 0.360 ± 0.052 | n/a | n/a | 0.346 | -0.002 ± 0.084 | -0.047 ± 0.056 |
| Logistic regression | 0.859 ± 0.039 | 0.424 ± 0.124 | 0.446 ± 0.089 | 0.015 ± 0.013 | 0.053 ± 0.008 | 0.470 | +0.046 ± 0.127 | +0.000 ± 0.000 |
| XGBoost (grid) | 0.859 ± 0.039 | 0.431 ± 0.121 | 0.439 ± 0.082 | 0.024 ± 0.011 | 0.052 ± 0.007 | 0.537 | +0.106 ± 0.104 | +0.061 ± 0.043 |
| XGBoost (Optuna) | 0.864 ± 0.038 | 0.448 ± 0.105 | 0.451 ± 0.075 | 0.019 ± 0.006 | 0.051 ± 0.007 | 0.536 | +0.088 ± 0.090 | +0.042 ± 0.058 |
