from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
FIGURES_DIR = RESULTS_REAL_DIR / "figures"
METRICS_DIR = RESULTS_REAL_DIR / "metrics"

PRIMARY_DATASET = "waqd2024"
PRIMARY_TARGET = "PM25_p95_rel_p90_h1"


def load_summary() -> pd.DataFrame:
    path = TABLES_DIR / "real_validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/38_run_openaq_real_validation.py first."
        )
    return pd.read_csv(path)


def load_metrics() -> pd.DataFrame:
    path = METRICS_DIR / "real_validation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/38_run_openaq_real_validation.py first."
        )
    return pd.read_csv(path)


def load_threshold_summary() -> pd.DataFrame:
    path = TABLES_DIR / "real_threshold_calibration_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def primary_protocol_table(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[
        (summary["dataset"] == PRIMARY_DATASET)
        & (summary["target"] == PRIMARY_TARGET)
        & (
            (
                (summary["feature_regime"] == "base")
                & (summary["weighting_strategy"] == "none")
            )
            | (
                (summary["feature_regime"] == "local")
                & (summary["weighting_strategy"] == "none")
            )
            | (
                (summary["feature_regime"] == "local")
                & (summary["weighting_strategy"] == "country_class_balanced")
            )
        )
    ].copy()

    primary["configuration"] = primary.apply(configuration_label, axis=1)
    columns = [
        "protocol",
        "model",
        "configuration",
        "evaluations",
        "countries",
        "rows",
        "mean_event_rate",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "mean_false_alarm_rate",
    ]
    return primary[columns].sort_values(["protocol", "model", "configuration"])


def real_external_ladder(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[
        (summary["dataset"] == PRIMARY_DATASET)
        & (summary["target"] == PRIMARY_TARGET)
        & (summary["protocol"] == "leave_one_country_out")
    ].copy()

    rows = []
    for model in sorted(primary["model"].unique()):
        model_data = primary[primary["model"] == model]
        for label, regime, strategy in [
            ("Base", "base", "none"),
            ("Local features", "local", "none"),
            ("Country-class balanced", "local", "country_class_balanced"),
        ]:
            row = model_data[
                (model_data["feature_regime"] == regime)
                & (model_data["weighting_strategy"] == strategy)
            ]
            if row.empty:
                continue
            item = row.iloc[0].to_dict()
            rows.append(
                {
                    "dataset": PRIMARY_DATASET,
                    "target": PRIMARY_TARGET,
                    "protocol": "leave_one_country_out",
                    "model": model,
                    "stage": label,
                    "evaluations": int(item["evaluations"]),
                    "countries": int(item["countries"]),
                    "rows": int(item["rows"]),
                    "event_rate": item["mean_event_rate"],
                    "pr_auc": item["mean_pr_auc"],
                    "f1": item["mean_f1"],
                    "recall": item["mean_recall"],
                    "precision": item["mean_precision"],
                    "false_alarm_rate": item["mean_false_alarm_rate"],
                }
            )
    ladder = pd.DataFrame(rows)

    if not ladder.empty:
        ladder["delta_pr_auc_vs_base"] = ladder.groupby("model")["pr_auc"].transform(
            lambda values: values - values.iloc[0]
        )
        ladder["delta_f1_vs_base"] = ladder.groupby("model")["f1"].transform(
            lambda values: values - values.iloc[0]
        )
        ladder["delta_recall_vs_base"] = ladder.groupby("model")["recall"].transform(
            lambda values: values - values.iloc[0]
        )

    return ladder


def loco_country_table(metrics: pd.DataFrame) -> pd.DataFrame:
    country = metrics[
        (metrics["dataset"] == PRIMARY_DATASET)
        & (metrics["target"] == PRIMARY_TARGET)
        & (metrics["protocol"] == "leave_one_country_out")
        & (metrics["part"] == "test")
    ].copy()
    columns = [
        "held_out_country",
        "model",
        "feature_regime",
        "weighting_strategy",
        "rows",
        "events",
        "event_rate",
        "pr_auc",
        "f1",
        "recall",
        "precision",
        "false_alarm_rate",
    ]
    return country[columns].sort_values(["model", "feature_regime", "weighting_strategy", "pr_auc"])


def sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    loco = summary[summary["protocol"] == "leave_one_country_out"].copy()
    loco["configuration"] = loco.apply(configuration_label, axis=1)
    columns = [
        "dataset",
        "target",
        "model",
        "configuration",
        "evaluations",
        "countries",
        "rows",
        "mean_event_rate",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "mean_false_alarm_rate",
    ]
    return loco[columns].sort_values(["dataset", "target", "model", "configuration"])


def threshold_vs_default_table(summary: pd.DataFrame, threshold_summary: pd.DataFrame) -> pd.DataFrame:
    if threshold_summary.empty:
        return pd.DataFrame()

    default = summary[
        (summary["dataset"] == PRIMARY_DATASET)
        & (summary["target"] == PRIMARY_TARGET)
        & (summary["protocol"] == "leave_one_country_out")
    ].copy()
    default["threshold_policy"] = "default_0.50"
    default["mean_threshold"] = 0.50
    default["configuration"] = default.apply(configuration_label, axis=1)
    default = default.rename(
        columns={
            "mean_pr_auc": "pr_auc",
            "mean_f1": "f1",
            "mean_recall": "recall",
            "mean_precision": "precision",
            "mean_false_alarm_rate": "false_alarm_rate",
        }
    )

    calibrated = threshold_summary[
        (threshold_summary["dataset"] == PRIMARY_DATASET)
        & (threshold_summary["target"] == PRIMARY_TARGET)
        & (threshold_summary["protocol"] == "leave_one_country_out")
    ].copy()
    calibrated["threshold_policy"] = "validation_f1"
    calibrated["configuration"] = calibrated.apply(configuration_label, axis=1)
    calibrated = calibrated.rename(
        columns={
            "mean_pr_auc": "pr_auc",
            "mean_f1": "f1",
            "mean_recall": "recall",
            "mean_precision": "precision",
            "mean_false_alarm_rate": "false_alarm_rate",
        }
    )

    columns = [
        "dataset",
        "target",
        "model",
        "configuration",
        "threshold_policy",
        "countries",
        "rows",
        "mean_threshold",
        "pr_auc",
        "f1",
        "recall",
        "precision",
        "false_alarm_rate",
    ]
    return pd.concat([default[columns], calibrated[columns]], ignore_index=True).sort_values(
        ["model", "configuration", "threshold_policy"]
    )


def configuration_label(row: pd.Series) -> str:
    if row["feature_regime"] == "base" and row["weighting_strategy"] == "none":
        return "Base"
    if row["feature_regime"] == "local" and row["weighting_strategy"] == "none":
        return "Local features"
    if row["feature_regime"] == "local" and row["weighting_strategy"] == "country_class_balanced":
        return "Country-class balanced"
    return f"{row['feature_regime']} + {row['weighting_strategy']}"


def save_protocol_plot(protocol_table: pd.DataFrame) -> None:
    plot_data = protocol_table[
        (protocol_table["model"].isin(["logistic_regression", "random_forest", "xgboost"]))
        & (protocol_table["configuration"] == "Base")
    ].copy()
    if plot_data.empty:
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_data,
        x="protocol",
        y="mean_pr_auc",
        hue="model",
        palette="Set2",
    )
    plt.xlabel("Evaluation protocol")
    plt.ylabel("Mean PR-AUC")
    plt.title("Real-data validation: protocol comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_validation_protocol_pr_auc.png", dpi=160)
    plt.close()


def save_ladder_plot(ladder: pd.DataFrame) -> None:
    if ladder.empty:
        return

    sns.set_theme(style="whitegrid")
    plot_data = ladder.melt(
        id_vars=["model", "stage"],
        value_vars=["pr_auc", "f1", "recall", "precision"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_data,
        x="stage",
        y="value",
        hue="metric",
        palette="Set2",
    )
    plt.xlabel("LOCO configuration")
    plt.ylabel("Mean score")
    plt.title("Real-data validation: LOCO methodological ladder")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_validation_loco_ladder_metrics.png", dpi=160)
    plt.close()


def save_country_plot(country: pd.DataFrame) -> None:
    plot_data = country[
        (country["model"] == "logistic_regression")
        & (country["feature_regime"] == "local")
        & (country["weighting_strategy"] == "country_class_balanced")
    ].copy()
    if plot_data.empty:
        return

    plot_data = plot_data.sort_values("pr_auc", ascending=False)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        y="held_out_country",
        x="pr_auc",
        color="#4c78a8",
    )
    plt.xlabel("PR-AUC")
    plt.ylabel("Held-out country")
    plt.title("Real-data validation: LOCO PR-AUC by country")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_validation_loco_pr_auc_by_country.png", dpi=160)
    plt.close()


def save_threshold_plot(threshold_table: pd.DataFrame) -> None:
    if threshold_table.empty:
        return

    plot_data = threshold_table[
        (threshold_table["model"].isin(["logistic_regression", "random_forest", "xgboost"]))
        & (threshold_table["configuration"].isin(["Base", "Local features", "Country-class balanced"]))
    ].copy()
    if plot_data.empty:
        return

    plot_data = plot_data.melt(
        id_vars=["model", "configuration", "threshold_policy"],
        value_vars=["f1", "recall", "precision", "false_alarm_rate"],
        var_name="metric",
        value_name="value",
    )

    sns.set_theme(style="whitegrid")
    grid = sns.catplot(
        data=plot_data,
        x="configuration",
        y="value",
        hue="threshold_policy",
        col="metric",
        row="model",
        kind="bar",
        height=2.6,
        aspect=1.2,
        palette="Set2",
        sharey=False,
    )
    grid.set_axis_labels("Configuration", "Mean value")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_validation_threshold_default_vs_calibrated.png", dpi=160)
    plt.close()


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    metrics = load_metrics()
    threshold_summary = load_threshold_summary()

    protocol = primary_protocol_table(summary)
    ladder = real_external_ladder(summary)
    country = loco_country_table(metrics)
    sensitivity = sensitivity_table(summary)
    threshold_table = threshold_vs_default_table(summary, threshold_summary)

    protocol.to_csv(TABLES_DIR / "real_primary_protocol_comparison.csv", index=False)
    ladder.to_csv(TABLES_DIR / "real_external_loco_ladder.csv", index=False)
    country.to_csv(TABLES_DIR / "real_loco_country_metrics.csv", index=False)
    sensitivity.to_csv(TABLES_DIR / "real_sensitivity_loco_summary.csv", index=False)
    if not threshold_table.empty:
        threshold_table.to_csv(TABLES_DIR / "real_threshold_default_vs_calibrated.csv", index=False)

    save_protocol_plot(protocol)
    save_ladder_plot(ladder)
    save_country_plot(country)
    save_threshold_plot(threshold_table)

    print("Real-data validation summary completed.")
    print()
    print("Primary LOCO ladder:")
    print(ladder.to_string(index=False))
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
