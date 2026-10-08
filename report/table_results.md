Circuit-level held-out test, mean ± sd over 5 circuit splits.

| model | ROC-AUC | PR-AUC | F1 | ECE | Brier | PR-AUC random split | PR-AUC gap (random - circuit) | Extra gap vs logistic regression |
|---|---|---|---|---|---|---|---|---|
| Gap heuristic | 0.831 ± 0.033 | 0.348 ± 0.068 | 0.360 ± 0.052 | n/a | n/a | 0.337 | -0.011 ± 0.081 | -0.056 ± 0.049 |
| Logistic regression | 0.859 ± 0.039 | 0.425 ± 0.125 | 0.442 ± 0.088 | 0.018 ± 0.014 | 0.053 ± 0.008 | 0.469 | +0.044 ± 0.120 | +0.000 ± 0.000 |
| XGBoost (grid) | 0.862 ± 0.042 | 0.432 ± 0.128 | 0.437 ± 0.087 | 0.023 ± 0.009 | 0.053 ± 0.007 | 0.531 | +0.099 ± 0.105 | +0.054 ± 0.039 |
| XGBoost (Optuna) | 0.865 ± 0.041 | 0.444 ± 0.112 | 0.453 ± 0.073 | 0.020 ± 0.009 | 0.052 ± 0.008 | 0.525 | +0.081 ± 0.097 | +0.036 ± 0.055 |
