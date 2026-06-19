from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, TABLES_DIR, ensure_results_dirs


TARGET = "extreme_abs_151_h1"
MODEL = "xgboost"


def main() -> None:
    ensure_results_dirs()
    base = pd.read_csv(METRICS_DIR / f"loco_models_{TARGET}_test.csv")
    base = base[(base["target"] == TARGET) & (base["model"] == MODEL) & (base["part"] == "test")]
    base = base.copy()
    base["feature_set"] = "base_all_features"

    local = pd.read_csv(METRICS_DIR / "loco_local_features_metrics.csv")
    local = local[(local["target"] == TARGET) & (local["model"] == MODEL) & (local["part"] == "test")]

    combined = pd.concat([base, local], ignore_index=True, sort=False)
    combined.to_csv(TABLES_DIR / "local_features_base_vs_local_country_metrics.csv", index=False)

    summary = (
        combined.groupby(["target", "model", "feature_set"])
        .agg(
            countries=("held_out_country", "nunique"),
            mean_event_rate=("event_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            mean_missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / "local_features_base_vs_local_summary.csv", index=False)

    wide = combined.pivot_table(
        index=["held_out_country", "target", "model"],
        columns="feature_set",
        values=["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{feature_set}" for metric, feature_set in wide.columns]
    wide = wide.reset_index()
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]:
        wide[f"delta_local_minus_base_{metric}"] = (
            wide[f"{metric}_all_plus_local_normalization"] - wide[f"{metric}_base_all_features"]
        )
    wide.to_csv(TABLES_DIR / "local_features_base_vs_local_delta_by_country.csv", index=False)

    print("Local feature comparison completed.")
    print()
    print(summary.to_string(index=False))
    print()
    print("Mean local - base deltas:")
    print(
        wide[[column for column in wide.columns if column.startswith("delta_local_minus_base_")]]
        .mean()
        .to_string()
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
