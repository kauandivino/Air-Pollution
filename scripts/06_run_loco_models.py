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
from src.splits import (
    iter_leave_one_country_out_splits,
    validate_leave_one_country_integrity,
)


TARGET_COLUMN = "extreme_abs_151_h1"
FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DEFAULT_MODEL_NAMES = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]
CHECKPOINT_DIR = METRICS_DIR / "loco_checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run initial models under leave-one-country-out evaluation."
    )
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Target column to model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help="Model names from the model registry.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Optional subset of countries to hold out.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute checkpoints even when country/model metrics already exist.",
    )
    parser.add_argument(
        "--skip-train-metrics",
        action="store_true",
        help="Evaluate only validation and test parts to reduce runtime.",
    )
    return parser.parse_args()


def load_feature_columns() -> list[str]:
    catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    return catalog["feature"].drop_duplicates().tolist()


def load_modeling_frame(feature_columns: list[str], target_column: str) -> pd.DataFrame:
    usecols = ["Country", "Date", target_column] + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def checkpoint_path(target_column: str, country: str, model_name: str) -> Path:
    safe_country = country.replace(" ", "_").replace("/", "_")
    return CHECKPOINT_DIR / f"{target_column}__{safe_country}__{model_name}.csv"


def run_country_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split,
    model_spec,
    evaluate_train: bool,
) -> pd.DataFrame:
    x = df[feature_columns]
    y = df[target_column]
    estimator = clone(model_spec.estimator)

    estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))

    parts = {
        "validation": split.val_idx,
        "test": split.test_idx,
    }
    if evaluate_train:
        parts = {"train": split.train_idx, **parts}

    return evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts=parts,
        base_metadata={
            "split_name": split.name,
            "target": target_column,
            "held_out_country": split.metadata["held_out_country"],
            "model": model_spec.name,
            "model_family": model_spec.family,
        },
    )


def combine_checkpoints(target_column: str) -> pd.DataFrame:
    checkpoint_files = sorted(CHECKPOINT_DIR.glob(f"{target_column}__*.csv"))
    if not checkpoint_files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in checkpoint_files), ignore_index=True)


def save_loco_summaries(metrics: pd.DataFrame, target_column: str) -> None:
    all_path = METRICS_DIR / f"loco_models_{target_column}.csv"
    test_path = METRICS_DIR / f"loco_models_{target_column}_test.csv"
    country_path = METRICS_DIR / f"loco_models_{target_column}_test_by_country.csv"
    model_path = METRICS_DIR / f"loco_models_{target_column}_test_by_model.csv"

    metrics.to_csv(all_path, index=False)

    test_metrics = metrics[metrics["part"] == "test"].copy()
    test_metrics = test_metrics.sort_values(
        ["model", "held_out_country"],
        ascending=[True, True],
    )
    test_metrics.to_csv(test_path, index=False)

    test_metrics.sort_values(["held_out_country", "pr_auc"], ascending=[True, False]).to_csv(
        country_path,
        index=False,
    )

    model_summary = (
        test_metrics.groupby(["model", "model_family"])
        .agg(
            countries=("held_out_country", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            mean_missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
        .sort_values("mean_pr_auc", ascending=False)
    )
    model_summary.to_csv(model_path, index=False)


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns()
    df = load_modeling_frame(feature_columns, args.target)
    df = df[df[args.target].notna()].copy()

    model_specs = get_model_specs(args.models)
    evaluate_train = not args.skip_train_metrics
    resume = not args.no_resume

    for split in iter_leave_one_country_out_splits(
        df,
        args.target,
        countries=args.countries,
    ):
        validate_leave_one_country_integrity(df, split)
        country = split.metadata["held_out_country"]

        for model_spec in model_specs:
            path = checkpoint_path(args.target, country, model_spec.name)
            if resume and path.exists():
                print(f"Skipping {country} / {model_spec.name}: checkpoint exists.")
                continue

            print(f"Training {model_spec.name} with held-out country: {country}...")
            metrics = run_country_model(
                df=df,
                feature_columns=feature_columns,
                target_column=args.target,
                split=split,
                model_spec=model_spec,
                evaluate_train=evaluate_train,
            )
            metrics.to_csv(path, index=False)

    combined_metrics = combine_checkpoints(args.target)
    if combined_metrics.empty:
        raise RuntimeError("No LOCO checkpoint metrics were produced.")

    save_loco_summaries(combined_metrics, args.target)

    test_metrics = combined_metrics[combined_metrics["part"] == "test"].copy()
    model_summary = (
        test_metrics.groupby("model")
        .agg(
            countries=("held_out_country", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
        )
        .reset_index()
        .sort_values("mean_pr_auc", ascending=False)
    )

    print()
    print("Leave-one-country-out test summary by model:")
    print(model_summary.to_string(index=False))
    print()
    print("Best model per held-out country by PR-AUC:")
    best_by_country = (
        test_metrics.sort_values("pr_auc", ascending=False)
        .groupby("held_out_country")
        .head(1)
        .sort_values("held_out_country")
    )
    print(
        best_by_country[
            [
                "held_out_country",
                "model",
                "rows",
                "event_rate",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "pr_auc",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"LOCO metrics saved to: {METRICS_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
