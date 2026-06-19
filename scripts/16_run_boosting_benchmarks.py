from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.evaluation import evaluate_model_on_parts
from src.models import get_model_specs
from src.splits import make_random_split, make_temporal_split, validate_split_integrity


FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
MODELS = ["random_forest", "lightgbm", "xgboost"]
SPLITS = ["random_split", "temporal_split"]


def load_feature_columns() -> list[str]:
    catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    return catalog["feature"].drop_duplicates().tolist()


def load_modeling_frame(feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date"] + TARGETS + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def make_split(df: pd.DataFrame, target: str, split_name: str):
    if split_name == "random_split":
        return make_random_split(df, target)
    if split_name == "temporal_split":
        return make_temporal_split(df, target)
    raise ValueError(f"Unknown split: {split_name}")


def checkpoint_path(target: str, split_name: str, model: str) -> Path:
    return METRICS_DIR / "boosting_checkpoints" / f"{target}__{split_name}__{model}.csv"


def run_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
    valid_df = df[df[target].notna()].copy()
    split = make_split(valid_df, target, split_name)
    validate_split_integrity(split)

    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)
    x = valid_df[feature_columns]
    y = valid_df[target]

    estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))
    return evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={
            "train": split.train_idx,
            "validation": split.val_idx,
            "test": split.test_idx,
        },
        base_metadata={
            "target": target,
            "split_name": split.name,
            "model": spec.name,
            "model_family": spec.family,
        },
    )


def save_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["part"] == "test"].copy()
    summary = (
        test.groupby(["target", "split_name", "model", "model_family"])
        .agg(
            rows=("rows", "mean"),
            event_rate=("event_rate", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            false_alarm_rate=("false_alarm_rate", "mean"),
            missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
        .sort_values(["target", "split_name", "pr_auc"], ascending=[True, True, False])
    )
    summary.to_csv(TABLES_DIR / "boosting_benchmark_test_summary.csv", index=False)
    return summary


def main() -> None:
    ensure_results_dirs()
    checkpoint_dir = METRICS_DIR / "boosting_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns()
    df = load_modeling_frame(feature_columns)

    for target in TARGETS:
        for split_name in SPLITS:
            for model_name in MODELS:
                path = checkpoint_path(target, split_name, model_name)
                if path.exists():
                    print(f"Skipping {path.name}: checkpoint exists.")
                    continue
                print(f"Training {model_name} | {target} | {split_name}...")
                metrics = run_one(
                    df=df,
                    feature_columns=feature_columns,
                    target=target,
                    split_name=split_name,
                    model_name=model_name,
                )
                metrics.to_csv(path, index=False)

    checkpoint_files = sorted(checkpoint_dir.glob("*.csv"))
    if not checkpoint_files:
        raise RuntimeError("No boosting checkpoints found.")

    metrics = pd.concat((pd.read_csv(path) for path in checkpoint_files), ignore_index=True)
    metrics.to_csv(METRICS_DIR / "boosting_benchmark_metrics.csv", index=False)
    summary = save_summary(metrics)

    print()
    print("Boosting benchmark completed.")
    print()
    print(
        summary[
            [
                "target",
                "split_name",
                "model",
                "event_rate",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
