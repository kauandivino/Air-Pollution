from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_REAL_DIR = PROJECT_ROOT / "data" / "real" / "processed"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"

DEFAULT_DATASETS = ["waqd2024", "openaq"]
DEFAULT_TARGETS = [
    "PM25_p95_rel_p90_h1",
    "PM25_p95_rel_p90_h3",
    "PM10_p95_rel_p90_h1",
    "PM10_p95_rel_p90_h3",
]
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_TEST_ROWS = 8
DEFAULT_MIN_TRAIN_ROWS = 30
DEFAULT_VAL_SIZE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the statistical support of real-data LOCO folds."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--min-test-rows", type=int, default=DEFAULT_MIN_TEST_ROWS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    return parser.parse_args()


def load_real_matrix(dataset: str) -> pd.DataFrame:
    path = PROCESSED_REAL_DIR / f"{dataset}_real_feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/37_build_openaq_real_targets_features.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def valid_target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target not found: {target}")
    valid = df[df[target].notna()].copy()
    valid[target] = valid[target].astype(int)
    return valid


def reliability_status(test_rows: int, positives: int, non_events: int) -> tuple[str, str]:
    if positives == 0 or non_events == 0:
        return (
            "unstable_single_class",
            "test fold has a single observed class; threshold metrics and ranking metrics are fragile",
        )
    if test_rows < 10 or positives < 2 or non_events < 2:
        return (
            "unstable_low_support",
            "test fold has both classes but too few samples or too few events",
        )
    if test_rows < 20 or positives < 5:
        return (
            "limited_support",
            "test fold has both classes but limited sample/event support",
        )
    return (
        "usable",
        "test fold has both classes and enough support for cautious aggregation",
    )


def metric_reliability(status: str, metric: str) -> str:
    if status == "usable":
        return "usable"
    if status == "limited_support":
        return "limited"
    if status == "unstable_low_support":
        return "unstable"
    if status == "unstable_single_class":
        if metric == "pr_auc":
            return "unstable_single_class"
        return "unstable_single_class"
    return "unknown"


def split_train_validation(
    train_val: pd.DataFrame,
    target: str,
    random_seed: int,
    val_size: float,
) -> tuple[pd.Index, pd.Index, bool]:
    stratify = train_val[target] if train_val[target].value_counts().min() >= 2 else None
    train_idx, val_idx = train_test_split(
        train_val.index,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return pd.Index(train_idx), pd.Index(val_idx), stratify is not None


def diagnose_target(
    df: pd.DataFrame,
    dataset: str,
    target: str,
    random_seed: int,
    min_test_rows: int,
    min_train_rows: int,
    val_size: float,
) -> list[dict[str, object]]:
    valid = valid_target_frame(df, target)
    rows: list[dict[str, object]] = []

    for country in sorted(valid["Country"].dropna().unique()):
        test = valid[valid["Country"] == country].copy()
        train_val = valid[valid["Country"] != country].copy()

        test_rows = int(len(test))
        positives = int(test[target].sum())
        non_events = int(test_rows - positives)
        status, reason = reliability_status(test_rows, positives, non_events)

        eligible_for_training = (
            test_rows >= min_test_rows
            and len(train_val) >= min_train_rows
            and train_val[target].nunique() >= 2
        )

        train_idx: pd.Index = pd.Index([])
        val_idx: pd.Index = pd.Index([])
        validation_stratified = False
        if eligible_for_training:
            train_idx, val_idx, validation_stratified = split_train_validation(
                train_val=train_val,
                target=target,
                random_seed=random_seed,
                val_size=val_size,
            )

        train = valid.loc[train_idx] if len(train_idx) else valid.iloc[0:0]
        validation = valid.loc[val_idx] if len(val_idx) else valid.iloc[0:0]

        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "held_out_country": country,
                "eligible_for_training": bool(eligible_for_training),
                "test_rows": test_rows,
                "test_events": positives,
                "test_non_events": non_events,
                "test_event_rate": positives / test_rows if test_rows else pd.NA,
                "test_months": int(test["Date"].dt.to_period("M").nunique()),
                "test_start_month": (
                    test["Date"].min().strftime("%Y-%m") if test_rows else pd.NA
                ),
                "test_end_month": (
                    test["Date"].max().strftime("%Y-%m") if test_rows else pd.NA
                ),
                "test_class_count": int(test[target].nunique()),
                "has_two_test_classes": bool(test[target].nunique() == 2),
                "fold_status": status,
                "diagnostic_reason": reason,
                "pr_auc_reliability": metric_reliability(status, "pr_auc"),
                "f1_reliability": metric_reliability(status, "f1"),
                "train_rows": int(len(train)),
                "train_events": int(train[target].sum()) if len(train) else 0,
                "train_event_rate": (
                    float(train[target].mean()) if len(train) else pd.NA
                ),
                "validation_rows": int(len(validation)),
                "validation_events": int(validation[target].sum())
                if len(validation)
                else 0,
                "validation_event_rate": (
                    float(validation[target].mean()) if len(validation) else pd.NA
                ),
                "validation_stratified": bool(validation_stratified),
                "random_seed": random_seed,
                "validation_size": val_size,
                "min_test_rows": min_test_rows,
                "min_train_rows": min_train_rows,
            }
        )

    return rows


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        diagnostics.groupby(["dataset", "target", "fold_status"], dropna=False)
        .agg(
            countries=("held_out_country", "nunique"),
            eligible_countries=("eligible_for_training", "sum"),
            total_test_rows=("test_rows", "sum"),
            total_test_events=("test_events", "sum"),
            median_test_rows=("test_rows", "median"),
            median_test_events=("test_events", "median"),
        )
        .reset_index()
    )
    summary["total_test_event_rate"] = (
        summary["total_test_events"] / summary["total_test_rows"]
    )
    return summary


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        df = load_real_matrix(dataset)
        for target in args.targets:
            if target not in df.columns:
                continue
            all_rows.extend(
                diagnose_target(
                    df=df,
                    dataset=dataset,
                    target=target,
                    random_seed=args.random_seed,
                    min_test_rows=args.min_test_rows,
                    min_train_rows=args.min_train_rows,
                    val_size=args.val_size,
                )
            )

    diagnostics = pd.DataFrame(all_rows)
    diagnostics_path = TABLES_DIR / "real_loco_fold_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    summary = summarize_diagnostics(diagnostics)
    summary_path = TABLES_DIR / "real_loco_fold_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    primary = diagnostics[
        (diagnostics["dataset"] == "waqd2024")
        & (diagnostics["target"] == "PM25_p95_rel_p90_h1")
    ].copy()
    primary_path = TABLES_DIR / "real_loco_fold_diagnostics_primary.csv"
    primary.to_csv(primary_path, index=False)

    print(f"Saved fold diagnostics: {diagnostics_path}")
    print(f"Saved fold diagnostics summary: {summary_path}")
    print(f"Saved primary fold diagnostics: {primary_path}")
    if not primary.empty:
        print(
            primary["fold_status"]
            .value_counts(dropna=False)
            .rename_axis("fold_status")
            .reset_index(name="countries")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
