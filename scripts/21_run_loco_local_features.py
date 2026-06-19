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
from src.splits import iter_leave_one_country_out_splits, validate_leave_one_country_integrity


LOCAL_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global_local.csv"
DEFAULT_TARGETS = ["extreme_abs_151_h1"]
DEFAULT_MODELS = ["xgboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOCO with local normalized features.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_feature_columns() -> tuple[list[str], list[str], list[str]]:
    base = pd.read_csv(TABLES_DIR / "feature_catalog.csv")["feature"].drop_duplicates().tolist()
    local = pd.read_csv(TABLES_DIR / "local_feature_catalog.csv")["feature"].drop_duplicates().tolist()
    return base, local, base + local


def load_modeling_frame(targets: list[str], feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date"] + targets + feature_columns
    return pd.read_csv(LOCAL_FEATURE_MATRIX, usecols=usecols, parse_dates=["Date"])


def checkpoint_path(target: str, model: str, country: str) -> Path:
    safe_country = country.replace(" ", "_").replace("/", "_")
    return METRICS_DIR / "loco_local_feature_checkpoints" / f"{target}__{model}__{safe_country}.csv"


def run_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    split,
) -> pd.DataFrame:
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)
    x = df[feature_columns]
    y = df[target]

    estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))
    return evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={"validation": split.val_idx, "test": split.test_idx},
        base_metadata={
            "target": target,
            "model": model_name,
            "model_family": spec.family,
            "feature_set": "all_plus_local_normalization",
            "n_features": len(feature_columns),
            "held_out_country": split.metadata["held_out_country"],
        },
    )


def save_summary(metrics: pd.DataFrame) -> None:
    metrics.to_csv(METRICS_DIR / "loco_local_features_metrics.csv", index=False)
    test = metrics[metrics["part"] == "test"].copy()
    summary = (
        test.groupby(["target", "model", "model_family", "feature_set"])
        .agg(
            countries=("held_out_country", "nunique"),
            n_features=("n_features", "mean"),
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
        .sort_values(["target", "mean_pr_auc"], ascending=[True, False])
    )
    summary.to_csv(TABLES_DIR / "loco_local_features_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    checkpoint_dir = METRICS_DIR / "loco_local_feature_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _, _, feature_columns = load_feature_columns()
    df = load_modeling_frame(args.targets, feature_columns)
    resume = not args.no_resume

    for target in args.targets:
        target_df = df[df[target].notna()].copy()
        for split in iter_leave_one_country_out_splits(target_df, target):
            validate_leave_one_country_integrity(target_df, split)
            country = split.metadata["held_out_country"]
            for model_name in args.models:
                path = checkpoint_path(target, model_name, country)
                if resume and path.exists():
                    print(f"Skipping {path.name}: checkpoint exists.")
                    continue
                print(f"Training {model_name} with local features | {target} | held out: {country}")
                metrics = run_one(
                    df=target_df,
                    feature_columns=feature_columns,
                    target=target,
                    model_name=model_name,
                    split=split,
                )
                metrics.to_csv(path, index=False)

    checkpoint_files = sorted(checkpoint_dir.glob("*.csv"))
    if not checkpoint_files:
        raise RuntimeError("No local feature checkpoints found.")

    metrics = pd.concat((pd.read_csv(path) for path in checkpoint_files), ignore_index=True)
    metrics = metrics[metrics["target"].isin(args.targets) & metrics["model"].isin(args.models)]
    save_summary(metrics)

    summary = pd.read_csv(TABLES_DIR / "loco_local_features_summary.csv")
    print()
    print("LOCO local feature experiment completed.")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
