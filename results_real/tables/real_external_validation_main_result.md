| Model | Configuration | Countries | Rows | Events | PR-AUC | F1 | Recall | Precision | FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | Base | 6 | 147 | 31 | 0.388 | 0.243 | 0.297 | 0.288 | 0.282 |
| logistic_regression | Local features | 6 | 147 | 31 | 0.326 | 0.262 | 0.400 | 0.369 | 0.325 |
| logistic_regression | Country-class balanced | 6 | 147 | 31 | 0.306 | 0.242 | 0.369 | 0.186 | 0.415 |
| random_forest | Base | 6 | 147 | 31 | 0.371 | 0.298 | 0.331 | 0.347 | 0.199 |
| random_forest | Local features | 6 | 147 | 31 | 0.395 | 0.215 | 0.197 | 0.292 | 0.163 |
| random_forest | Country-class balanced | 6 | 147 | 31 | 0.317 | 0.286 | 0.522 | 0.210 | 0.483 |
| xgboost | Base | 6 | 147 | 31 | 0.269 | 0.000 | 0.000 | 0.000 | 0.000 |
| xgboost | Local features | 6 | 147 | 31 | 0.378 | 0.000 | 0.000 | 0.000 | 0.000 |
| xgboost | Country-class balanced | 6 | 147 | 31 | 0.346 | 0.216 | 0.261 | 0.185 | 0.192 |
