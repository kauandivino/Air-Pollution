from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
METRICS_DIR = RESULTS_REAL_DIR / "metrics"

PRIMARY_DATASET = "waqd2024"
PRIMARY_TARGET = "PM25_p95_rel_p90_h1"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


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


def summarize(metrics: pd.DataFrame, scenario: str, threshold_policy: str) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "target",
        "model",
        "model_family",
        "feature_regime",
        "weighting_strategy",
    ]
    threshold_agg = ("threshold", "mean") if "threshold" in metrics.columns else ("rows", lambda _: 0.50)
    summary = (
        metrics.groupby(group_columns, dropna=False)
        .agg(
            countries=("held_out_country", "nunique"),
            rows=("rows", "sum"),
            events=("events", "sum"),
            mean_threshold=threshold_agg,
            mean_event_rate=("event_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "threshold_policy", threshold_policy)
    summary.insert(0, "scenario", scenario)
    summary["configuration"] = summary.apply(configuration_label, axis=1)
    summary["pooled_event_rate"] = summary["events"] / summary["rows"]
    summary["false_alerts_per_100_city_months"] = summary["mean_false_alarm_rate"] * 100
    return summary


def allowed_countries(mask: pd.DataFrame, filtered: bool) -> pd.DataFrame:
    allowed = mask[
        (mask["dataset"] == PRIMARY_DATASET)
        & (mask["target"] == PRIMARY_TARGET)
        & (mask["eligible_for_training"].astype(bool))
    ].copy()
    if filtered:
        allowed = allowed[allowed["passes_min_support_filter"].astype(bool)]
    return allowed[["dataset", "target", "held_out_country"]]


def filter_primary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        (metrics["dataset"] == PRIMARY_DATASET)
        & (metrics["target"] == PRIMARY_TARGET)
        & (metrics["protocol"] == "leave_one_country_out")
    ].copy()


def add_policy_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "scenario",
        "dataset",
        "target",
        "model",
        "model_family",
        "feature_regime",
        "weighting_strategy",
        "configuration",
    ]
    value_columns = [
        "countries",
        "rows",
        "events",
        "mean_threshold",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "mean_false_alarm_rate",
        "false_alerts_per_100_city_months",
    ]
    wide = summary.pivot_table(
        index=id_columns,
        columns="threshold_policy",
        values=value_columns,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{policy}" for metric, policy in wide.columns]
    wide = wide.reset_index()

    pairs = [
        ("mean_f1", "delta_f1_calibrated_minus_default"),
        ("mean_recall", "delta_recall_calibrated_minus_default"),
        ("mean_precision", "delta_precision_calibrated_minus_default"),
        ("mean_false_alarm_rate", "delta_far_calibrated_minus_default"),
        (
            "false_alerts_per_100_city_months",
            "delta_false_alerts_per_100_calibrated_minus_default",
        ),
    ]
    for metric, delta_name in pairs:
        calibrated = f"{metric}__validation_f1"
        default = f"{metric}__default_0.50"
        if calibrated in wide.columns and default in wide.columns:
            wide[delta_name] = wide[calibrated] - wide[default]

    return wide


def make_article_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[
        (summary["scenario"] == "minimum_support")
        & (summary["model"] == "xgboost")
        & (summary["configuration"].isin(["Base", "Local features", "Country-class balanced"]))
    ].copy()
    columns = [
        "configuration",
        "threshold_policy",
        "countries",
        "rows",
        "mean_threshold",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "false_alerts_per_100_city_months",
    ]
    return table[columns].sort_values(["configuration", "threshold_policy"])


def make_markdown(table: pd.DataFrame) -> str:
    header = (
        "| Configuration | Threshold policy | Countries | Rows | Mean threshold | "
        "PR-AUC | F1 | Recall | Precision | False alerts / 100 |\n"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for _, row in table.iterrows():
        rows.append(
            "| "
            f"{row['configuration']} | {row['threshold_policy']} | "
            f"{int(row['countries'])} | {int(row['rows'])} | "
            f"{row['mean_threshold']:.3f} | {row['mean_pr_auc']:.3f} | "
            f"{row['mean_f1']:.3f} | {row['mean_recall']:.3f} | "
            f"{row['mean_precision']:.3f} | "
            f"{row['false_alerts_per_100_city_months']:.1f} |"
        )
    return header + separator + "\n".join(rows) + "\n"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    default_metrics = read_csv(METRICS_DIR / "real_validation_metrics.csv")
    calibrated_metrics = read_csv(METRICS_DIR / "real_threshold_calibration_metrics.csv")
    mask = read_csv(TABLES_DIR / "real_loco_min_support_country_mask.csv")

    default_metrics = filter_primary_metrics(default_metrics)
    default_metrics = default_metrics[default_metrics["part"] == "test"].copy()
    calibrated_metrics = filter_primary_metrics(calibrated_metrics)

    rows = []
    for scenario, filtered in [
        ("all_eligible", False),
        ("minimum_support", True),
    ]:
        countries = allowed_countries(mask, filtered=filtered)
        default_scenario = default_metrics.merge(
            countries, on=["dataset", "target", "held_out_country"], how="inner"
        )
        calibrated_scenario = calibrated_metrics.merge(
            countries, on=["dataset", "target", "held_out_country"], how="inner"
        )
        rows.append(
            summarize(
                default_scenario,
                scenario=scenario,
                threshold_policy="default_0.50",
            )
        )
        rows.append(
            summarize(
                calibrated_scenario,
                scenario=scenario,
                threshold_policy="validation_f1",
            )
        )

    summary = pd.concat(rows, ignore_index=True)
    deltas = add_policy_deltas(summary)
    article_table = make_article_table(summary)

    summary_path = TABLES_DIR / "real_operational_threshold_summary.csv"
    deltas_path = TABLES_DIR / "real_operational_threshold_deltas.csv"
    article_csv_path = TABLES_DIR / "real_operational_threshold_article_table.csv"
    article_md_path = TABLES_DIR / "real_operational_threshold_article_table.md"

    summary.to_csv(summary_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    article_table.to_csv(article_csv_path, index=False)
    article_md_path.write_text(make_markdown(article_table), encoding="utf-8")

    print(f"Saved operational threshold summary: {summary_path}")
    print(f"Saved operational threshold deltas: {deltas_path}")
    print(f"Saved article-ready threshold table: {article_csv_path}")
    print(f"Saved article-ready markdown table: {article_md_path}")
    print(make_markdown(article_table))


if __name__ == "__main__":
    main()
