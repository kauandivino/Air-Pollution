from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, TABLES_DIR, ensure_results_dirs


TARGET_H1 = "extreme_abs_151_h1"
TARGET_H3 = "extreme_abs_151_h3"


def load_loco_model_summary(target: str) -> pd.DataFrame:
    path = METRICS_DIR / f"loco_models_{target}_test_by_model.csv"
    df = pd.read_csv(path)
    suffix = target.rsplit("_", maxsplit=1)[-1]
    metric_columns = [column for column in df.columns if column.startswith("mean_")]
    return df[["model", "model_family", "countries"] + metric_columns].rename(
        columns={column: f"{suffix}_{column}" for column in metric_columns}
    )


def load_loco_country_metrics(target: str) -> pd.DataFrame:
    path = METRICS_DIR / f"loco_models_{target}_test.csv"
    df = pd.read_csv(path)
    suffix = target.rsplit("_", maxsplit=1)[-1]
    columns = [
        "held_out_country",
        "model",
        "event_rate",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    return df[columns].rename(
        columns={column: f"{suffix}_{column}" for column in columns if column not in ["held_out_country", "model"]}
    )


def main() -> None:
    ensure_results_dirs()

    h1 = load_loco_model_summary(TARGET_H1)
    h3 = load_loco_model_summary(TARGET_H3)
    model_gap = h1.merge(h3, on=["model", "model_family", "countries"], how="inner")

    for metric in [
        "mean_balanced_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_false_alarm_rate",
        "mean_missed_event_rate",
    ]:
        model_gap[f"delta_h1_minus_h3_{metric}"] = (
            model_gap[f"h1_{metric}"] - model_gap[f"h3_{metric}"]
        )

    country_h1 = load_loco_country_metrics(TARGET_H1)
    country_h3 = load_loco_country_metrics(TARGET_H3)
    country_gap = country_h1.merge(country_h3, on=["held_out_country", "model"], how="inner")
    for metric in [
        "event_rate",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        country_gap[f"delta_h1_minus_h3_{metric}"] = (
            country_gap[f"h1_{metric}"] - country_gap[f"h3_{metric}"]
        )

    model_gap = model_gap.sort_values("delta_h1_minus_h3_mean_pr_auc", ascending=False)
    country_gap = country_gap.sort_values(
        ["model", "delta_h1_minus_h3_pr_auc"],
        ascending=[True, False],
    )

    model_gap.to_csv(TABLES_DIR / "loco_abs151_h1_vs_h3_model_gap.csv", index=False)
    country_gap.to_csv(TABLES_DIR / "loco_abs151_h1_vs_h3_country_gap.csv", index=False)

    print("LOCO h1 vs h3 comparison completed.")
    print()
    print(
        model_gap[
            [
                "model",
                "h1_mean_pr_auc",
                "h3_mean_pr_auc",
                "delta_h1_minus_h3_mean_pr_auc",
                "h1_mean_recall",
                "h3_mean_recall",
                "delta_h1_minus_h3_mean_recall",
                "h1_mean_f1",
                "h3_mean_f1",
                "delta_h1_minus_h3_mean_f1",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
