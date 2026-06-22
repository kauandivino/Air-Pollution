| Configuration | Threshold policy | Countries | Rows | Mean threshold | PR-AUC | F1 | Recall | Precision | False alerts / 100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | default_0.50 | 6 | 147 | 0.500 | 0.269 | 0.000 | 0.000 | 0.000 | 0.0 |
| Base | validation_f1 | 6 | 147 | 0.158 | 0.269 | 0.256 | 0.556 | 0.170 | 54.6 |
| Country-class balanced | default_0.50 | 6 | 147 | 0.500 | 0.346 | 0.216 | 0.261 | 0.185 | 19.2 |
| Country-class balanced | validation_f1 | 6 | 147 | 0.467 | 0.346 | 0.240 | 0.372 | 0.195 | 29.9 |
| Local features | default_0.50 | 6 | 147 | 0.500 | 0.378 | 0.000 | 0.000 | 0.000 | 0.0 |
| Local features | validation_f1 | 6 | 147 | 0.142 | 0.378 | 0.355 | 0.681 | 0.344 | 58.1 |
