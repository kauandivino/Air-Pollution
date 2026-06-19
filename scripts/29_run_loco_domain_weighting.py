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
from src.domain_weighting import domain_weights
from src.evaluation import evaluate_model_on_parts
from src.models import get_model_specs
from src.splits import iter_leave_one_country_out_splits, validate_leave_one_country_integrity


BASE_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
LOCAL_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global_local.csv"
DEFAULT_TARGETS = ["extreme_abs_151_h3"]
DEFAULT_MODELS = ["xgboost"]
DEFAULT_STRATEGIES = ["inverse_country_size", "country_class_balanced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOCO with robust domain sample weights.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--feature-regime", choices=["base", "local"], default="local")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_feature_columns(feature_regime: str) -> list[str]:
    base = pd.read_csv(TABLES_DIR / "feature_catalog.csv")["feature"].drop_duplicates().tolist()
    if feature_regime == "base":
        return base
    local = pd.read_csv(TABLES_DIR / "local_feature_catalog.csv")["feature"].drop_duplicates().tolist()
    return base + local


def load_modeling_frame(
    targets: list[str],
    feature_columns: list[str],
    feature_regime: str,
) -> pd.DataFrame:
    path = LOCAL_FEATURE_MATRIX if feature_regime == "local" else BASE_FEATURE_MATRIX
    usecols = ["Country", "Date", *targets, *feature_columns]
    usecols = list(dict.fromkeys(usecols))
    return pd.read_csv(path, usecols=usecols, parse_dates=["Date"])


def checkpoint_path(
    target: str,
    model: str,
    strategy: str,
    feature_regime: str,
    country: str,
) -> Path:
    safe_country = country.replace(" ", "_").replace("/", "_")
    filename = f"{target}__{model}__{feature_regime}__{strategy}__{safe_country}.csv"
    return METRICS_DIR / "loco_domain_weighting_checkpoints" / filename


def fit_with_weights(estimator, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series):
    return estimator.fit(
        x_train,
        y_train.astype(int),
        **{"model__sample_weight": sample_weight.to_numpy()},
    )


def run_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    strategy: str,
    feature_regime: str,
    split,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)
    x = df[feature_columns]
    y = df[target]

    sample_weight, diagnostics = domain_weights(
        df=df,
        train_index=split.train_idx,
        target=target,
        strategy=strategy,
    )
    fit_with_weights(
        estimator,
        x_train=x.loc[split.train_idx],
        y_train=y.loc[split.train_idx],
        sample_weight=sample_weight,
    )

    feature_set = (
        f"all_plus_local_normalization_domain_weighted_{strategy}"
        if feature_regime == "local"
        else f"all_features_domain_weighted_{strategy}"
    )
    metrics = evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={"validation": split.val_idx, "test": split.test_idx},
        base_metadata={
            "target": target,
            "model": model_name,
            "model_family": spec.family,
            "feature_set": feature_set,
            "feature_regime": feature_regime,
            "weighting_strategy": strategy,
            "n_features": len(feature_columns),
            "held_out_country": split.metadata["held_out_country"],
            "train_weight_min": sample_weight.min(),
            "train_weight_mean": sample_weight.mean(),
            "train_weight_max": sample_weight.max(),
            "train_weight_std": sample_weight.std(),
        },
    )
    diagnostics["target"] = target
    diagnostics["model"] = model_name
    diagnostics["feature_regime"] = feature_regime
    diagnostics["weighting_strategy"] = strategy
    diagnostics["held_out_country"] = split.metadata["held_out_country"]
    return metrics, diagnostics


def save_summary(metrics: pd.DataFrame) -> None:
    metrics.to_csv(METRICS_DIR / "loco_domain_weighting_metrics.csv", index=False)
    test = metrics[metrics["part"] == "test"].copy()
    summary = (
        test.groupby(
            [
                "target",
                "model",
                "model_family",
                "feature_set",
                "feature_regime",
                "weighting_strategy",
            ],
            observed=True,
        )
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
            mean_train_weight_std=("train_weight_std", "mean"),
        )
        .reset_index()
        .sort_values(["target", "mean_pr_auc"], ascending=[True, False])
    )
    summary.to_csv(TABLES_DIR / "loco_domain_weighting_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    checkpoint_dir = METRICS_DIR / "loco_domain_weighting_checkpoints"
    diagnostic_dir = METRICS_DIR / "loco_domain_weighting_diagnostics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(args.feature_regime)
    df = load_modeling_frame(args.targets, feature_columns, args.feature_regime)
    resume = not args.no_resume

    for target in args.targets:
        target_df = df[df[target].notna()].copy()
        for split in iter_leave_one_country_out_splits(target_df, target):
            validate_leave_one_country_integrity(target_df, split)
            country = split.metadata["held_out_country"]
            for model_name in args.models:
                for strategy in args.strategies:
                    path = checkpoint_path(
                        target,
                        model_name,
                        strategy,
                        args.feature_regime,
                        country,
                    )
                    diagnostic_path = diagnostic_dir / path.name
                    if resume and path.exists():
                        print(f"Skipping {path.name}: checkpoint exists.")
                        continue

                    print(
                        f"Training {model_name} | {target} | {args.feature_regime} features | "
                        f"domain-weighted={strategy} | held out: {country}"
                    )
                    metrics, diagnostics = run_one(
                        df=target_df,
                        feature_columns=feature_columns,
                        target=target,
                        model_name=model_name,
                        strategy=strategy,
                        feature_regime=args.feature_regime,
                        split=split,
                    )
                    metrics.to_csv(path, index=False)
                    diagnostics.to_csv(diagnostic_path, index=False)

    checkpoint_files = sorted(checkpoint_dir.glob("*.csv"))
    if not checkpoint_files:
        raise RuntimeError("No domain-weighting checkpoints found.")

    metrics = pd.concat((pd.read_csv(path) for path in checkpoint_files), ignore_index=True)
    metrics = metrics[
        metrics["target"].isin(args.targets)
        & metrics["model"].isin(args.models)
        & metrics["weighting_strategy"].isin(args.strategies)
        & (metrics["feature_regime"] == args.feature_regime)
    ]
    save_summary(metrics)

    diagnostics = pd.concat(
        (pd.read_csv(path) for path in sorted(diagnostic_dir.glob("*.csv"))),
        ignore_index=True,
    )
    diagnostics = diagnostics[
        diagnostics["target"].isin(args.targets)
        & diagnostics["model"].isin(args.models)
        & diagnostics["weighting_strategy"].isin(args.strategies)
        & (diagnostics["feature_regime"] == args.feature_regime)
    ]
    diagnostics.to_csv(TABLES_DIR / "loco_domain_weighting_diagnostics.csv", index=False)

    summary = pd.read_csv(TABLES_DIR / "loco_domain_weighting_summary.csv")
    print()
    print("LOCO domain-weighting experiment completed.")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
