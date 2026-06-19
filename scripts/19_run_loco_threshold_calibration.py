from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.evaluation import evaluate_binary_classifier, prediction_scores
from src.models import get_model_specs
from src.splits import iter_leave_one_country_out_splits, validate_leave_one_country_integrity


FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DEFAULT_TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
DEFAULT_MODELS = ["random_forest", "xgboost"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOCO threshold calibration experiments.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--criterion",
        choices=["f1", "balanced_accuracy", "recall_at_precision_30"],
        default="f1",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_feature_columns() -> list[str]:
    catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    return catalog["feature"].drop_duplicates().tolist()


def load_modeling_frame(feature_columns: list[str], targets: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date"] + targets + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def checkpoint_path(target: str, model: str, country: str, criterion: str) -> Path:
    safe_country = country.replace(" ", "_").replace("/", "_")
    return (
        METRICS_DIR
        / "threshold_calibration_checkpoints"
        / f"{target}__{model}__{safe_country}__{criterion}.csv"
    )


def choose_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    criterion: str,
) -> tuple[float, dict[str, float | int]]:
    rows = []
    for threshold in THRESHOLD_GRID:
        y_pred = (y_score >= threshold).astype(int)
        metrics = evaluate_binary_classifier(y_true, y_pred, y_score)
        rows.append({"threshold": float(threshold), **metrics})

    table = pd.DataFrame(rows)
    if criterion == "f1":
        selected = table.sort_values(["f1", "pr_auc", "recall"], ascending=False).iloc[0]
    elif criterion == "balanced_accuracy":
        selected = table.sort_values(
            ["balanced_accuracy", "f1", "recall"],
            ascending=False,
        ).iloc[0]
    else:
        feasible = table[table["precision"] >= 0.30]
        if feasible.empty:
            selected = table.sort_values(["precision", "recall"], ascending=False).iloc[0]
        else:
            selected = feasible.sort_values(["recall", "f1"], ascending=False).iloc[0]

    return float(selected["threshold"]), selected.to_dict()


def evaluate_at_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    return evaluate_binary_classifier(y_true, y_pred, y_score)


def run_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    split,
    criterion: str,
) -> pd.DataFrame:
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)

    x = df[feature_columns]
    y = df[target]
    estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))

    val_score = prediction_scores(estimator, x.loc[split.val_idx])
    test_score = prediction_scores(estimator, x.loc[split.test_idx])
    if val_score is None or test_score is None:
        raise ValueError(f"Model does not expose scores for threshold calibration: {model_name}")

    threshold, validation_selected = choose_threshold(
        y.loc[split.val_idx].astype(int),
        val_score,
        criterion=criterion,
    )
    test_calibrated = evaluate_at_threshold(
        y.loc[split.test_idx].astype(int),
        test_score,
        threshold=threshold,
    )
    test_default = evaluate_at_threshold(
        y.loc[split.test_idx].astype(int),
        test_score,
        threshold=0.50,
    )

    base = {
        "target": target,
        "model": model_name,
        "model_family": spec.family,
        "held_out_country": split.metadata["held_out_country"],
        "criterion": criterion,
        "selected_threshold": threshold,
    }

    rows = [
        {
            **base,
            "part": "validation_selected",
            **{f"validation_{key}": value for key, value in validation_selected.items()},
        },
        {
            **base,
            "part": "test_calibrated",
            **test_calibrated,
        },
        {
            **base,
            "part": "test_default_threshold",
            **test_default,
        },
    ]
    return pd.DataFrame(rows)


def save_summary(
    metrics: pd.DataFrame,
    criterion: str,
    targets: list[str],
    models: list[str],
) -> None:
    metrics = metrics[
        metrics["target"].isin(targets) & metrics["model"].isin(models)
    ].copy()
    metrics.to_csv(METRICS_DIR / f"loco_threshold_calibration_{criterion}.csv", index=False)

    test = metrics[metrics["part"].isin(["test_calibrated", "test_default_threshold"])].copy()
    summary = (
        test.groupby(["target", "model", "model_family", "criterion", "part"])
        .agg(
            countries=("held_out_country", "nunique"),
            mean_threshold=("selected_threshold", "mean"),
            mean_event_rate=("event_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            mean_missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / f"loco_threshold_calibration_summary_{criterion}.csv", index=False)

    wide = summary.pivot_table(
        index=["target", "model", "criterion"],
        columns="part",
        values=[
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_balanced_accuracy",
            "mean_false_alarm_rate",
            "mean_missed_event_rate",
        ],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{part}" for metric, part in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_balanced_accuracy",
        "mean_false_alarm_rate",
        "mean_missed_event_rate",
    ]:
        wide[f"delta_calibrated_minus_default_{metric}"] = (
            wide[f"{metric}_test_calibrated"] - wide[f"{metric}_test_default_threshold"]
        )
    wide.to_csv(TABLES_DIR / f"loco_threshold_calibration_delta_{criterion}.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    checkpoint_dir = METRICS_DIR / "threshold_calibration_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns()
    df = load_modeling_frame(feature_columns, args.targets)
    resume = not args.no_resume

    for target in args.targets:
        target_df = df[df[target].notna()].copy()
        for split in iter_leave_one_country_out_splits(target_df, target):
            validate_leave_one_country_integrity(target_df, split)
            country = split.metadata["held_out_country"]
            for model_name in args.models:
                path = checkpoint_path(target, model_name, country, args.criterion)
                if resume and path.exists():
                    print(f"Skipping {path.name}: checkpoint exists.")
                    continue
                print(f"Calibrating {model_name} | {target} | held out: {country}")
                metrics = run_one(
                    df=target_df,
                    feature_columns=feature_columns,
                    target=target,
                    model_name=model_name,
                    split=split,
                    criterion=args.criterion,
                )
                metrics.to_csv(path, index=False)

    checkpoint_files = sorted(checkpoint_dir.glob(f"*__{args.criterion}.csv"))
    if not checkpoint_files:
        raise RuntimeError("No threshold calibration checkpoints found.")

    metrics = pd.concat((pd.read_csv(path) for path in checkpoint_files), ignore_index=True)
    save_summary(metrics, args.criterion, targets=args.targets, models=args.models)

    summary = pd.read_csv(TABLES_DIR / f"loco_threshold_calibration_delta_{args.criterion}.csv")
    print()
    print("LOCO threshold calibration completed.")
    print()
    print(
        summary[
            [
                "target",
                "model",
                "delta_calibrated_minus_default_mean_f1",
                "delta_calibrated_minus_default_mean_recall",
                "delta_calibrated_minus_default_mean_false_alarm_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
