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
from src.feature_sets import (
    FEATURE_SET_ORDER,
    build_feature_sets,
    load_feature_catalog,
    summarize_feature_sets,
)
from src.models import get_model_specs
from src.splits import make_random_split, make_temporal_split, validate_split_integrity


FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DEFAULT_TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
DEFAULT_MODELS = ["logistic_regression", "random_forest"]
DEFAULT_SPLITS = ["random_split", "temporal_split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-set ablation experiments.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_modeling_frame(targets: list[str], feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    features = sorted({feature for columns in feature_sets.values() for feature in columns})
    usecols = ["Country", "Date"] + targets + features
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def make_split(df: pd.DataFrame, target: str, split_name: str):
    if split_name == "random_split":
        return make_random_split(df, target)
    if split_name == "temporal_split":
        return make_temporal_split(df, target)
    raise ValueError(f"Unknown split: {split_name}")


def checkpoint_path(target: str, split_name: str, feature_set: str, model: str) -> Path:
    return METRICS_DIR / "ablation_checkpoints" / f"{target}__{split_name}__{feature_set}__{model}.csv"


def run_one(
    df: pd.DataFrame,
    target: str,
    split_name: str,
    feature_set: str,
    feature_columns: list[str],
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
            "feature_set": feature_set,
            "n_features": len(feature_columns),
            "model": spec.name,
            "model_family": spec.family,
        },
    )


def summarize_ablation(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = metrics[metrics["part"] == "test"].copy()
    summary = test.sort_values(
        ["target", "split_name", "model", "pr_auc"],
        ascending=[True, True, True, False],
    )

    all_features = test[test["feature_set"] == "all_features"][
        ["target", "split_name", "model", "pr_auc", "f1", "recall", "precision"]
    ].rename(
        columns={
            "pr_auc": "all_features_pr_auc",
            "f1": "all_features_f1",
            "recall": "all_features_recall",
            "precision": "all_features_precision",
        }
    )
    gap = test.merge(all_features, on=["target", "split_name", "model"], how="left")
    for metric in ["pr_auc", "f1", "recall", "precision"]:
        gap[f"delta_vs_all_{metric}"] = gap[metric] - gap[f"all_features_{metric}"]

    return summary, gap


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    checkpoint_dir = METRICS_DIR / "ablation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_feature_catalog(TABLES_DIR / "feature_catalog.csv")
    feature_sets = build_feature_sets(catalog)
    summarize_feature_sets(feature_sets).to_csv(TABLES_DIR / "feature_set_summary.csv", index=False)

    df = load_modeling_frame(args.targets, feature_sets)
    resume = not args.no_resume

    for target in args.targets:
        for split_name in args.splits:
            for feature_set in FEATURE_SET_ORDER:
                for model_name in args.models:
                    path = checkpoint_path(target, split_name, feature_set, model_name)
                    if resume and path.exists():
                        print(f"Skipping {path.name}: checkpoint exists.")
                        continue

                    print(
                        f"Training {model_name} | {target} | {split_name} | {feature_set} "
                        f"({len(feature_sets[feature_set])} features)..."
                    )
                    metrics = run_one(
                        df=df,
                        target=target,
                        split_name=split_name,
                        feature_set=feature_set,
                        feature_columns=feature_sets[feature_set],
                        model_name=model_name,
                    )
                    metrics.to_csv(path, index=False)

    checkpoints = sorted(checkpoint_dir.glob("*.csv"))
    if not checkpoints:
        raise RuntimeError("No ablation checkpoints were produced.")

    metrics = pd.concat((pd.read_csv(path) for path in checkpoints), ignore_index=True)
    metrics.to_csv(METRICS_DIR / "feature_ablation_metrics.csv", index=False)
    summary, gap = summarize_ablation(metrics)
    summary.to_csv(TABLES_DIR / "feature_ablation_test_summary.csv", index=False)
    gap.to_csv(TABLES_DIR / "feature_ablation_gap_vs_all.csv", index=False)

    print()
    print("Feature ablation completed.")
    print()
    print("Feature set sizes:")
    print(summarize_feature_sets(feature_sets).to_string(index=False))
    print()
    print("Test summary:")
    print(
        summary[
            [
                "target",
                "split_name",
                "model",
                "feature_set",
                "n_features",
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

