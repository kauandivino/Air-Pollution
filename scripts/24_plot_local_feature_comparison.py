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

from src.config import FIGURES_DIR, TABLES_DIR, ensure_results_dirs


PRIMARY_METRICS = ["pr_auc", "f1"]
EXPERIMENT_LABELS = {
    ("extreme_abs_151_h1", "xgboost"): "XGBoost h=1",
    ("extreme_abs_151_h3", "xgboost"): "XGBoost h=3",
    ("extreme_abs_151_h1", "random_forest"): "Random Forest h=1",
}
EXPERIMENT_ORDER = list(EXPERIMENT_LABELS.values())


def add_experiment_label(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["experiment"] = [
        EXPERIMENT_LABELS.get((target, model), f"{model} | {target}")
        for target, model in zip(data["target"], data["model"])
    ]
    data["experiment"] = pd.Categorical(
        data["experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )
    return data


def plot_base_vs_local(summary: pd.DataFrame) -> None:
    plot_rows = []
    for _, row in summary[summary["metric"].isin(PRIMARY_METRICS)].iterrows():
        plot_rows.append(
            {
                "experiment": row["experiment"],
                "metric": row["metric"].upper().replace("_", "-"),
                "feature_regime": "Base",
                "value": row["base_mean"],
            }
        )
        plot_rows.append(
            {
                "experiment": row["experiment"],
                "metric": row["metric"].upper().replace("_", "-"),
                "feature_regime": "Local",
                "value": row["local_mean"],
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    sns.set_theme(style="whitegrid")

    grid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="experiment",
        y="value",
        hue="feature_regime",
        col="metric",
        order=EXPERIMENT_ORDER,
        palette=["#466b8f", "#c44e52"],
        height=5,
        aspect=1.15,
        sharey=False,
    )
    grid.set_axis_labels("", "Score medio LOCO")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=20)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    grid.figure.suptitle("Features locais vs base em generalizacao geografica", y=1.04)
    grid.tight_layout()
    grid.savefig(FIGURES_DIR / "local_features_base_vs_local_pr_auc_f1.png", dpi=180)
    plt.close(grid.figure)


def plot_delta_heatmap(deltas: pd.DataFrame, metric: str) -> None:
    metric_col = f"{metric}_delta"
    plot_df = deltas.copy()
    pivot = plot_df.pivot_table(
        index="held_out_country",
        columns="experiment",
        values=metric_col,
        aggfunc="mean",
        observed=True,
    )
    pivot["mean_delta"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_delta", ascending=False).drop(columns="mean_delta")

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    plt.figure(figsize=(9, 11))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": f"Delta {metric.upper().replace('_', '-')}"},
    )
    plt.title(f"Delta por pais retido: local - base ({metric.upper().replace('_', '-')})")
    plt.xlabel("")
    plt.ylabel("Pais retido no LOCO")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"local_features_country_delta_heatmap_{metric}.png", dpi=180)
    plt.close()


def save_top_country_deltas(deltas: pd.DataFrame) -> None:
    rows = []
    for metric in PRIMARY_METRICS:
        metric_col = f"{metric}_delta"
        ranked = deltas.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False)
        for _, row in ranked.head(5).iterrows():
            rows.append(
                {
                    "metric": metric,
                    "direction": "largest_gain",
                    "experiment": row["experiment"],
                    "held_out_country": row["held_out_country"],
                    "delta": row[metric_col],
                }
            )
        for _, row in ranked.tail(5).sort_values(metric_col).iterrows():
            rows.append(
                {
                    "metric": metric,
                    "direction": "smallest_delta",
                    "experiment": row["experiment"],
                    "held_out_country": row["held_out_country"],
                    "delta": row[metric_col],
                }
            )

    pd.DataFrame(rows).to_csv(
        TABLES_DIR / "local_features_top_country_deltas.csv",
        index=False,
    )


def plot_delta_summary(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    plot_df = summary[summary["metric"].isin(PRIMARY_METRICS)].merge(
        tests[["target", "model", "metric", "p_value"]],
        on=["target", "model", "metric"],
        how="left",
    )
    plot_df["metric_label"] = plot_df["metric"].str.upper().str.replace("_", "-", regex=False)
    plot_df["significant"] = plot_df["p_value"] < 0.05

    plt.figure(figsize=(10, 5.5))
    axis = sns.barplot(
        data=plot_df,
        x="experiment",
        y="mean_delta",
        hue="metric_label",
        order=EXPERIMENT_ORDER,
        palette=["#2a9d8f", "#8d5a97"],
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_title("Ganho medio das features locais no LOCO")
    axis.set_xlabel("")
    axis.set_ylabel("Delta medio (local - base)")
    axis.tick_params(axis="x", rotation=20)

    for container in axis.containers:
        labels = [f"{bar.get_height():+.3f}" for bar in container]
        axis.bar_label(container, labels=labels, padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "local_features_mean_delta_summary.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_results_dirs()

    summary = pd.read_csv(TABLES_DIR / "local_features_base_vs_local_summary.csv")
    deltas = pd.read_csv(TABLES_DIR / "local_features_base_vs_local_delta_by_country.csv")
    tests = pd.read_csv(TABLES_DIR / "local_features_base_vs_local_wilcoxon.csv")

    summary = add_experiment_label(summary)
    deltas = add_experiment_label(deltas)

    plot_base_vs_local(summary)
    plot_delta_summary(summary, tests)
    plot_delta_heatmap(deltas, "pr_auc")
    plot_delta_heatmap(deltas, "f1")
    save_top_country_deltas(deltas)

    print("Local feature comparison figures completed.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Top country deltas saved to: {TABLES_DIR / 'local_features_top_country_deltas.csv'}")


if __name__ == "__main__":
    main()
