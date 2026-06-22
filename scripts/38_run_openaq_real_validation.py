from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_weighting import domain_weights
from src.evaluation import evaluate_model_on_parts
from src.models import get_model_specs


PROCESSED_REAL_DIR = PROJECT_ROOT / "data" / "real" / "processed"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
METRICS_DIR = RESULTS_REAL_DIR / "metrics"

DEFAULT_DATASETS = ["waqd2024", "openaq"]
DEFAULT_TARGETS = ["PM25_p95_rel_p90_h1"]
DEFAULT_MODELS = ["logistic_regression", "xgboost"]
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class RealSplit:
    name: str
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external validation on real OpenAQ-like data.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--min-test-rows", type=int, default=8)
    parser.add_argument("--min-train-rows", type=int, default=30)
    return parser.parse_args()


def load_real_matrix(dataset: str) -> pd.DataFrame:
    path = PROCESSED_REAL_DIR / f"{dataset}_real_feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/37_build_openaq_real_targets_features.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def load_feature_catalog(dataset: str, regime: str) -> list[str]:
    path = TABLES_DIR / f"{dataset}_real_feature_catalog.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/37_build_openaq_real_targets_features.py first."
        )

    catalog = pd.read_csv(path)
    if regime == "base":
        catalog = catalog[catalog["group"] != "local_normalized"]
    elif regime != "local":
        raise ValueError(f"Unknown feature regime: {regime}")

    return catalog["feature"].drop_duplicates().tolist()


def valid_target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target not found: {target}")
    valid = df[df[target].notna()].copy()
    valid[target] = valid[target].astype(int)
    return valid


def make_random_split(
    df: pd.DataFrame,
    target: str,
    random_seed: int,
) -> RealSplit | None:
    if df[target].nunique() < 2 or len(df) < 20:
        return None

    stratify = df[target] if df[target].value_counts().min() >= 2 else None
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=0.15,
        random_state=random_seed,
        stratify=stratify,
    )

    train_val = df.loc[train_val_idx]
    stratify_train_val = (
        train_val[target] if train_val[target].value_counts().min() >= 2 else None
    )
    train_idx, val_idx = train_test_split(
        train_val.index,
        test_size=0.15 / 0.85,
        random_state=random_seed,
        stratify=stratify_train_val,
    )

    return RealSplit(
        name="random_split",
        train_idx=pd.Index(train_idx),
        val_idx=pd.Index(val_idx),
        test_idx=pd.Index(test_idx),
        metadata={"random_seed": random_seed, "stratified": stratify is not None},
    )


def make_temporal_split(df: pd.DataFrame, target: str) -> RealSplit | None:
    train_idx = df[df["Date"] <= pd.Timestamp("2021-12-31")].index
    val_idx = df[(df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] <= pd.Timestamp("2022-12-31"))].index
    test_idx = df[df["Date"] >= pd.Timestamp("2023-01-01")].index

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        return None
    if df.loc[train_idx, target].nunique() < 2:
        return None

    return RealSplit(
        name="temporal_split",
        train_idx=pd.Index(train_idx),
        val_idx=pd.Index(val_idx),
        test_idx=pd.Index(test_idx),
        metadata={
            "train_end": "2021-12-31",
            "val_start": "2022-01-01",
            "val_end": "2022-12-31",
            "test_start": "2023-01-01",
        },
    )


def iter_loco_splits(
    df: pd.DataFrame,
    target: str,
    random_seed: int,
    min_test_rows: int,
    min_train_rows: int,
):
    for country in sorted(df["Country"].dropna().unique()):
        test_idx = df[df["Country"] == country].index
        train_val = df[df["Country"] != country].copy()

        if len(test_idx) < min_test_rows:
            continue
        if len(train_val) < min_train_rows or train_val[target].nunique() < 2:
            continue

        stratify = (
            train_val[target]
            if train_val[target].value_counts().min() >= 2
            else None
        )
        train_idx, val_idx = train_test_split(
            train_val.index,
            test_size=0.15,
            random_state=random_seed,
            stratify=stratify,
        )

        yield RealSplit(
            name="leave_one_country_out",
            train_idx=pd.Index(train_idx),
            val_idx=pd.Index(val_idx),
            test_idx=pd.Index(test_idx),
            metadata={
                "held_out_country": country,
                "random_seed": random_seed,
                "val_size": 0.15,
                "stratified": stratify is not None,
            },
        )


def split_has_train_classes(df: pd.DataFrame, split: RealSplit, target: str) -> bool:
    return df.loc[split.train_idx, target].nunique() >= 2


def fit_estimator(estimator, x_train: pd.DataFrame, y_train: pd.Series, sample_weight=None):
    if sample_weight is None:
        return estimator.fit(x_train, y_train.astype(int))
    return estimator.fit(
        x_train,
        y_train.astype(int),
        **{"model__sample_weight": sample_weight.to_numpy()},
    )


def run_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    dataset: str,
    model_name: str,
    feature_regime: str,
    weighting_strategy: str,
    split: RealSplit,
) -> pd.DataFrame:
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)

    x = df[feature_columns]
    y = df[target]

    sample_weight = None
    if weighting_strategy == "country_class_balanced":
        sample_weight, _ = domain_weights(
            df=df,
            train_index=split.train_idx,
            target=target,
            strategy="country_class_balanced",
        )
    elif weighting_strategy != "none":
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

    fit_estimator(
        estimator,
        x_train=x.loc[split.train_idx],
        y_train=y.loc[split.train_idx],
        sample_weight=sample_weight,
    )

    held_out_country = split.metadata.get("held_out_country", "")
    return evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={"validation": split.val_idx, "test": split.test_idx},
        base_metadata={
            "dataset": dataset,
            "target": target,
            "protocol": split.name,
            "model": model_name,
            "model_family": spec.family,
            "feature_regime": feature_regime,
            "weighting_strategy": weighting_strategy,
            "n_features": len(feature_columns),
            "held_out_country": held_out_country,
            **split.metadata,
        },
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["part"] == "test"].copy()
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
        test.groupby(group_columns, dropna=False, observed=True)
        .agg(
            evaluations=("part", "size"),
            countries=("held_out_country", lambda values: int(pd.Series(values).replace("", pd.NA).dropna().nunique())),
            rows=("rows", "sum"),
            mean_event_rate=("event_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "target", "protocol", "mean_pr_auc"], ascending=[True, True, True, False])
    )


def run_dataset_target(
    dataset: str,
    target: str,
    model_names: list[str],
    random_seed: int,
    min_test_rows: int,
    min_train_rows: int,
) -> list[pd.DataFrame]:
    df = valid_target_frame(load_real_matrix(dataset), target)
    results = []

    split_candidates: list[RealSplit] = []
    random_split = make_random_split(df, target, random_seed=random_seed)
    if random_split is not None:
        split_candidates.append(random_split)

    temporal_split = make_temporal_split(df, target)
    if temporal_split is not None:
        split_candidates.append(temporal_split)

    split_candidates.extend(
        iter_loco_splits(
            df,
            target=target,
            random_seed=random_seed,
            min_test_rows=min_test_rows,
            min_train_rows=min_train_rows,
        )
    )

    for split in split_candidates:
        if not split_has_train_classes(df, split, target):
            continue

        regimes_and_weights = [
            ("base", "none"),
            ("local", "none"),
            ("local", "country_class_balanced"),
        ]

        for feature_regime, weighting_strategy in regimes_and_weights:
            feature_columns = load_feature_catalog(dataset, feature_regime)
            for model_name in model_names:
                print(
                    f"{dataset} | {target} | {split.name} | {model_name} | "
                    f"{feature_regime} | {weighting_strategy} | "
                    f"held_out={split.metadata.get('held_out_country', '')}"
                )
                try:
                    result = run_split(
                        df=df,
                        feature_columns=feature_columns,
                        target=target,
                        dataset=dataset,
                        model_name=model_name,
                        feature_regime=feature_regime,
                        weighting_strategy=weighting_strategy,
                        split=split,
                    )
                except ValueError as exc:
                    print(f"Skipping failed run: {exc}")
                    continue
                results.append(result)

    return results


def main() -> None:
    args = parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset in args.datasets:
        for target in args.targets:
            all_results.extend(
                run_dataset_target(
                    dataset=dataset,
                    target=target,
                    model_names=args.models,
                    random_seed=args.random_seed,
                    min_test_rows=args.min_test_rows,
                    min_train_rows=args.min_train_rows,
                )
            )

    if not all_results:
        raise RuntimeError("No real-data validation runs were completed.")

    metrics = pd.concat(all_results, ignore_index=True)
    summary = summarize(metrics)

    metrics.to_csv(METRICS_DIR / "real_validation_metrics.csv", index=False)
    summary.to_csv(TABLES_DIR / "real_validation_summary.csv", index=False)

    print()
    print("Real-data validation completed.")
    print(summary.to_string(index=False))
    print()
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Summary saved to: {TABLES_DIR / 'real_validation_summary.csv'}")


if __name__ == "__main__":
    main()
