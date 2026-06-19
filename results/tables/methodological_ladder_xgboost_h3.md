# Methodological ladder - XGBoost h=3

| step | stage | pr_auc | f1 | recall | precision | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | RF base LOCO | 0.329 | 0.276 | 0.328 | 0.261 | Baseline forte com features globais, antes de boosting. |
| 2 | XGBoost base LOCO | 0.330 | 0.230 | 0.215 | 0.298 | Boosting melhora o ranqueamento, mas ainda sofre sob LOCO. |
| 3 | XGBoost + local features | 0.379 | 0.254 | 0.244 | 0.333 | Representacao local/anomalias reduz parte do shift geografico. |
| 4 | XGBoost + similarity weighting | 0.385 | 0.251 | 0.238 | 0.346 | Paises ambientalmente similares melhoram ranking, mas sacrificam recall. |
| 5 | XGBoost + country-class weighting | 0.386 | 0.317 | 0.390 | 0.304 | Balancear pais e classe aumenta fortemente recall e F1. |

