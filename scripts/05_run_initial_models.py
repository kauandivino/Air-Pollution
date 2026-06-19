from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.evaluation import evaluate_model_on_parts
from src.models import get_model_specs
from src.splits import make_random_split, make_temporal_split, validate_split_integrity


TARGET_COLUMN = "extreme_abs_151_h1"
FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
MODEL_NAMES = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]


def load_feature_columns() -> list[str]:
    catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    return catalog["feature"].drop_duplicates().tolist()


def load_modeling_frame(feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date", TARGET_COLUMN] + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def run_split_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    if split_name == "random_split":
        split = make_random_split(df, TARGET_COLUMN)
    elif split_name == "temporal_split":
        split = make_temporal_split(df, TARGET_COLUMN)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    validate_split_integrity(split)

    x = df[feature_columns]
    y = df[TARGET_COLUMN]
    parts = {
        "train": split.train_idx,
        "validation": split.val_idx,
        "test": split.test_idx,
    }

    rows = []
    for spec in get_model_specs(MODEL_NAMES):
        print(f"Training {spec.name} on {split_name}...")
        estimator = spec.estimator
        estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))
        metrics = evaluate_model_on_parts(
            estimator,
            x=x,
            y=y,
            parts=parts,
            base_metadata={
                "split_name": split.name,
                "target": TARGET_COLUMN,
                "model": spec.name,
                "model_family": spec.family,
            },
        )
        rows.append(metrics)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_results_dirs()
    feature_columns = load_feature_columns()
    df = load_modeling_frame(feature_columns)
    df = df[df[TARGET_COLUMN].notna()].copy()

    all_metrics = []
    for split_name in ["random_split", "temporal_split"]:
        split_metrics = run_split_experiment(df, feature_columns, split_name)
        split_metrics.to_csv(METRICS_DIR / f"initial_models_{split_name}.csv", index=False)
        all_metrics.append(split_metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(METRICS_DIR / "initial_models_random_temporal.csv", index=False)

    test_metrics = metrics[metrics["part"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["split_name", "pr_auc"], ascending=[True, False])

    print()
    print("Initial model test metrics:")
    print(
        test_metrics[
            [
                "split_name",
                "model",
                "rows",
                "event_rate",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_alarm_rate",
                "missed_event_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
