from __future__ import annotations

import argparse
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
DEFAULT_TARGETS = [
    "extreme_abs_151_h3",
    "extreme_abs_201_h1",
    "extreme_abs_201_h3",
    "extreme_city_p90_h1",
    "extreme_city_p90_h3",
    "extreme_city_p95_h1",
    "extreme_city_p95_h3",
]
DEFAULT_MODEL_NAMES = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random and temporal benchmarks for multiple targets."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target columns to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help="Model names from the model registry.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute target metrics even when output files already exist.",
    )
    return parser.parse_args()


def load_feature_columns() -> list[str]:
    catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    return catalog["feature"].drop_duplicates().tolist()


def load_modeling_frame(feature_columns: list[str], target_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date"] + target_columns + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def run_split_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_name: str,
    model_names: list[str],
) -> pd.DataFrame:
    if split_name == "random_split":
        split = make_random_split(df, target_column)
    elif split_name == "temporal_split":
        split = make_temporal_split(df, target_column)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    validate_split_integrity(split)

    x = df[feature_columns]
    y = df[target_column]
    parts = {
        "train": split.train_idx,
        "validation": split.val_idx,
        "test": split.test_idx,
    }

    rows = []
    for spec in get_model_specs(model_names):
        print(f"Training {spec.name} on {split_name} for {target_column}...")
        estimator = clone(spec.estimator)
        estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))
        metrics = evaluate_model_on_parts(
            estimator,
            x=x,
            y=y,
            parts=parts,
            base_metadata={
                "split_name": split.name,
                "target": target_column,
                "model": spec.name,
                "model_family": spec.family,
            },
        )
        rows.append(metrics)

    return pd.concat(rows, ignore_index=True)


def run_target(
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str],
    resume: bool,
) -> pd.DataFrame:
    combined_path = METRICS_DIR / f"initial_models_{target_column}_random_temporal.csv"
    if resume and combined_path.exists():
        print(f"Skipping {target_column}: {combined_path.name} already exists.")
        return pd.read_csv(combined_path)

    df = full_df[full_df[target_column].notna()].copy()
    all_metrics = []

    for split_name in ["random_split", "temporal_split"]:
        split_metrics = run_split_experiment(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            split_name=split_name,
            model_names=model_names,
        )
        split_metrics.to_csv(
            METRICS_DIR / f"initial_models_{target_column}_{split_name}.csv",
            index=False,
        )
        all_metrics.append(split_metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(combined_path, index=False)
    return metrics


def save_cross_target_summary(metrics: pd.DataFrame) -> None:
    test_metrics = metrics[metrics["part"] == "test"].copy()
    summary = (
        test_metrics.groupby(["target", "split_name", "model", "model_family"])
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
    summary.to_csv(METRICS_DIR / "initial_models_multi_target_test_summary.csv", index=False)

    best = (
        summary.sort_values("pr_auc", ascending=False)
        .groupby(["target", "split_name"])
        .head(1)
        .sort_values(["target", "split_name"])
    )
    best.to_csv(METRICS_DIR / "initial_models_multi_target_best_by_target.csv", index=False)

    print()
    print("Best model by target/protocol:")
    print(
        best[
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


def main() -> None:
    args = parse_args()
    ensure_results_dirs()

    feature_columns = load_feature_columns()
    df = load_modeling_frame(feature_columns, args.targets)
    resume = not args.no_resume

    all_metrics = []
    for target_column in args.targets:
        metrics = run_target(
            full_df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_names=args.models,
            resume=resume,
        )
        all_metrics.append(metrics)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(METRICS_DIR / "initial_models_multi_target_random_temporal.csv", index=False)
    save_cross_target_summary(combined)

    print()
    print(f"Multi-target random/temporal metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()

