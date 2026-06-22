from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
METRICS_DIR = RESULTS_REAL_DIR / "metrics"

PRIMARY_DATASET = "waqd2024"
PRIMARY_TARGET = "PM25_p95_rel_p90_h1"
DEFAULT_MIN_TEST_ROWS = 10
DEFAULT_MIN_TEST_EVENTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize real-data LOCO results with all eligible folds and with a "
            "minimum-support country filter."
        )
    )
    parser.add_argument("--min-test-rows", type=int, default=DEFAULT_MIN_TEST_ROWS)
    parser.add_argument("--min-test-events", type=int, default=DEFAULT_MIN_TEST_EVENTS)
    return parser.parse_args()


def load_metrics() -> pd.DataFrame:
    path = METRICS_DIR / "real_validation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/38_run_openaq_real_validation.py first."
        )
    return pd.read_csv(path)


def load_diagnostics() -> pd.DataFrame:
    path = TABLES_DIR / "real_loco_fold_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/41_diagnose_real_loco_folds.py first."
        )
    return pd.read_csv(path)


def metric_columns(metrics: pd.DataFrame) -> list[str]:
    candidates = [
        "event_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "false_alarm_rate",
        "missed_event_rate",
        "roc_auc",
        "pr_auc",
    ]
    return [column for column in candidates if column in metrics.columns]


def summarize_loco(metrics: pd.DataFrame, scenario: str) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "target",
        "model",
        "model_family",
        "feature_regime",
        "weighting_strategy",
    ]
    metrics_to_average = metric_columns(metrics)

    grouped = (
        metrics.groupby(group_columns, dropna=False)
        .agg(
            evaluations=("held_out_country", "count"),
            countries=("held_out_country", "nunique"),
            rows=("rows", "sum"),
            events=("events", "sum"),
            **{f"mean_{column}": (column, "mean") for column in metrics_to_average},
        )
        .reset_index()
    )
    grouped.insert(0, "scenario", scenario)
    grouped["pooled_event_rate"] = grouped["events"] / grouped["rows"]
    return grouped


def build_country_mask(
    diagnostics: pd.DataFrame,
    min_test_rows: int,
    min_test_events: int,
) -> pd.DataFrame:
    mask = diagnostics.copy()
    mask["passes_min_support_filter"] = (
        mask["eligible_for_training"].astype(bool)
        & (mask["test_rows"] >= min_test_rows)
        & (mask["test_events"] >= min_test_events)
        & (mask["test_non_events"] >= 1)
        & mask["has_two_test_classes"].astype(bool)
    )
    mask["filter_min_test_rows"] = min_test_rows
    mask["filter_min_test_events"] = min_test_events
    return mask


def apply_mask(metrics: pd.DataFrame, mask: pd.DataFrame, filtered: bool) -> pd.DataFrame:
    keys = ["dataset", "target", "held_out_country"]
    allowed = mask[mask["eligible_for_training"].astype(bool)].copy()
    if filtered:
        allowed = allowed[allowed["passes_min_support_filter"].astype(bool)]

    return metrics.merge(allowed[keys], on=keys, how="inner")


def add_scenario_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "dataset",
        "target",
        "model",
        "model_family",
        "feature_regime",
        "weighting_strategy",
    ]
    delta_columns = [
        column
        for column in [
            "mean_pr_auc",
            "mean_f1",
            "mean_recall",
            "mean_precision",
            "mean_false_alarm_rate",
            "countries",
            "rows",
            "events",
        ]
        if column in summary.columns
    ]

    wide = summary.pivot_table(
        index=id_columns,
        columns="scenario",
        values=delta_columns,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    if "mean_pr_auc__minimum_support" in wide and "mean_pr_auc__all_eligible" in wide:
        wide["delta_pr_auc_filtered_minus_all"] = (
            wide["mean_pr_auc__minimum_support"] - wide["mean_pr_auc__all_eligible"]
        )
    if "mean_f1__minimum_support" in wide and "mean_f1__all_eligible" in wide:
        wide["delta_f1_filtered_minus_all"] = (
            wide["mean_f1__minimum_support"] - wide["mean_f1__all_eligible"]
        )
    if "mean_recall__minimum_support" in wide and "mean_recall__all_eligible" in wide:
        wide["delta_recall_filtered_minus_all"] = (
            wide["mean_recall__minimum_support"] - wide["mean_recall__all_eligible"]
        )

    return wide


def configuration_label(row: pd.Series) -> str:
    if row["feature_regime"] == "base" and row["weighting_strategy"] == "none":
        return "Base"
    if row["feature_regime"] == "local" and row["weighting_strategy"] == "none":
        return "Local features"
    if (
        row["feature_regime"] == "local"
        and row["weighting_strategy"] == "country_class_balanced"
    ):
        return "Country-class balanced"
    return f"{row['feature_regime']} + {row['weighting_strategy']}"


def primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[
        (summary["dataset"] == PRIMARY_DATASET)
        & (summary["target"] == PRIMARY_TARGET)
    ].copy()
    primary["configuration"] = primary.apply(configuration_label, axis=1)
    columns = [
        "scenario",
        "model",
        "configuration",
        "evaluations",
        "countries",
        "rows",
        "events",
        "pooled_event_rate",
        "mean_event_rate",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "mean_false_alarm_rate",
    ]
    return primary[columns].sort_values(["model", "configuration", "scenario"])


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    diagnostics = load_diagnostics()

    loco_test = metrics[
        (metrics["protocol"] == "leave_one_country_out")
        & (metrics["part"] == "test")
    ].copy()

    mask = build_country_mask(
        diagnostics=diagnostics,
        min_test_rows=args.min_test_rows,
        min_test_events=args.min_test_events,
    )

    all_eligible = apply_mask(loco_test, mask, filtered=False)
    minimum_support = apply_mask(loco_test, mask, filtered=True)

    all_summary = summarize_loco(all_eligible, scenario="all_eligible")
    filtered_summary = summarize_loco(minimum_support, scenario="minimum_support")
    combined_summary = pd.concat([all_summary, filtered_summary], ignore_index=True)

    mask_path = TABLES_DIR / "real_loco_min_support_country_mask.csv"
    summary_path = TABLES_DIR / "real_loco_all_vs_min_support_summary.csv"
    primary_path = TABLES_DIR / "real_loco_primary_all_vs_min_support.csv"
    deltas_path = TABLES_DIR / "real_loco_all_vs_min_support_deltas.csv"

    mask.to_csv(mask_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    primary_table(combined_summary).to_csv(primary_path, index=False)
    add_scenario_deltas(combined_summary).to_csv(deltas_path, index=False)

    primary = primary_table(combined_summary)
    print(f"Saved country mask: {mask_path}")
    print(f"Saved all vs filtered LOCO summary: {summary_path}")
    print(f"Saved primary all vs filtered LOCO summary: {primary_path}")
    print(f"Saved all vs filtered deltas: {deltas_path}")
    print(
        primary[
            [
                "scenario",
                "model",
                "configuration",
                "countries",
                "rows",
                "events",
                "mean_pr_auc",
                "mean_f1",
                "mean_recall",
                "mean_precision",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
