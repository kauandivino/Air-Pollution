from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_weighting import domain_weights
from src.evaluation import evaluate_binary_classifier, prediction_scores
from src.models import get_model_specs


PROCESSED_REAL_DIR = PROJECT_ROOT / "data" / "real" / "processed"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
METRICS_DIR = RESULTS_REAL_DIR / "metrics"

DATASET = "waqd2024"
TARGET = "PM25_p95_rel_p90_h1"
MODELS = ["logistic_regression", "random_forest", "xgboost"]
RANDOM_SEED = 42
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass(frozen=True)
class Split:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    country: str


def load_frame() -> pd.DataFrame:
    path = PROCESSED_REAL_DIR / f"{DATASET}_real_feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def load_features(regime: str) -> list[str]:
    path = TABLES_DIR / f"{DATASET}_real_feature_catalog.csv"
    catalog = pd.read_csv(path)
    if regime == "base":
        catalog = catalog[catalog["group"] != "local_normalized"]
    elif regime != "local":
        raise ValueError(f"Unknown feature regime: {regime}")
    return catalog["feature"].drop_duplicates().tolist()


def iter_loco_splits(df: pd.DataFrame, min_test_rows: int = 8, min_train_rows: int = 30):
    for country in sorted(df["Country"].dropna().unique()):
        test_idx = df[df["Country"] == country].index
        train_val = df[df["Country"] != country].copy()
        if len(test_idx) < min_test_rows:
            continue
        if len(train_val) < min_train_rows or train_val[TARGET].nunique() < 2:
            continue

        stratify = train_val[TARGET] if train_val[TARGET].value_counts().min() >= 2 else None
        train_idx, val_idx = train_test_split(
            train_val.index,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
        yield Split(
            train_idx=pd.Index(train_idx),
            val_idx=pd.Index(val_idx),
            test_idx=pd.Index(test_idx),
            country=country,
        )


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true.astype(int), y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def fit_estimator(estimator, x_train: pd.DataFrame, y_train: pd.Series, sample_weight=None):
    if sample_weight is None:
        return estimator.fit(x_train, y_train.astype(int))
    return estimator.fit(
        x_train,
        y_train.astype(int),
        **{"model__sample_weight": sample_weight.to_numpy()},
    )


def run_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    feature_regime: str,
    weighting_strategy: str,
    split: Split,
) -> dict[str, object] | None:
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)
    x = df[feature_columns]
    y = df[TARGET]

    sample_weight = None
    if weighting_strategy == "country_class_balanced":
        sample_weight, _ = domain_weights(
            df=df,
            train_index=split.train_idx,
            target=TARGET,
            strategy="country_class_balanced",
        )
    elif weighting_strategy != "none":
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

    fit_estimator(
        estimator,
        x.loc[split.train_idx],
        y.loc[split.train_idx],
        sample_weight=sample_weight,
    )

    val_score = prediction_scores(estimator, x.loc[split.val_idx])
    test_score = prediction_scores(estimator, x.loc[split.test_idx])
    if val_score is None or test_score is None:
        return None

    threshold, validation_f1 = choose_threshold(y.loc[split.val_idx], val_score)
    test_pred = (test_score >= threshold).astype(int)
    metrics = evaluate_binary_classifier(y.loc[split.test_idx], test_pred, test_score)

    return {
        "dataset": DATASET,
        "target": TARGET,
        "protocol": "leave_one_country_out",
        "model": model_name,
        "model_family": spec.family,
        "feature_regime": feature_regime,
        "weighting_strategy": weighting_strategy,
        "held_out_country": split.country,
        "threshold": threshold,
        "validation_f1_at_threshold": validation_f1,
        "n_features": len(feature_columns),
        **metrics,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "target",
        "protocol",
        "model",
        "model_family",
        "feature_regime",
        "weighting_strategy",
    ]
    return (
        metrics.groupby(group_columns, dropna=False, observed=True)
        .agg(
            countries=("held_out_country", "nunique"),
            rows=("rows", "sum"),
            mean_threshold=("threshold", "mean"),
            median_threshold=("threshold", "median"),
            mean_event_rate=("event_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
        )
        .reset_index()
        .sort_values(["model", "mean_f1"], ascending=[True, False])
    )


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_frame()
    rows = []
    configs = [
        ("base", "none"),
        ("local", "none"),
        ("local", "country_class_balanced"),
    ]

    for split in iter_loco_splits(df):
        for feature_regime, weighting_strategy in configs:
            feature_columns = load_features(feature_regime)
            for model_name in MODELS:
                print(
                    f"{DATASET} | {TARGET} | threshold-calibrated LOCO | {model_name} | "
                    f"{feature_regime} | {weighting_strategy} | held_out={split.country}"
                )
                result = run_one(
                    df=df,
                    feature_columns=feature_columns,
                    model_name=model_name,
                    feature_regime=feature_regime,
                    weighting_strategy=weighting_strategy,
                    split=split,
                )
                if result is not None:
                    rows.append(result)

    if not rows:
        raise RuntimeError("No threshold-calibrated runs were completed.")

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)

    metrics.to_csv(METRICS_DIR / "real_threshold_calibration_metrics.csv", index=False)
    summary.to_csv(TABLES_DIR / "real_threshold_calibration_summary.csv", index=False)

    print()
    print("Real threshold calibration completed.")
    print(summary.to_string(index=False))
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Summary saved to: {TABLES_DIR / 'real_threshold_calibration_summary.csv'}")


if __name__ == "__main__":
    main()
