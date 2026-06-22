| Finding | Synthetic | Real | Interpretation |
|---|---:|---:|---|
| Random/temporal split exceeds LOCO | yes | partial | Protocol optimism is strong in the synthetic benchmark; in the real filtered analysis it is clear for random split but weaker for temporal split. |
| Local features improve PR-AUC | yes | yes | Local normalization improves ranking in both settings under the selected XGBoost comparison. |
| Country-class weighting increases recall | yes | yes | The recall-oriented behavior of country-class weighting is reproduced in the filtered real-data validation. |
| Country-class weighting increases false alarms | yes | yes | The higher-recall configuration also increases the false-alarm rate, preserving the operational trade-off. |
