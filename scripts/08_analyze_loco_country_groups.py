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

from src.config import FIGURES_DIR, METRICS_DIR, TABLES_DIR, ensure_results_dirs


TARGET_COLUMN = "extreme_abs_151_h1"
LOCO_TEST_METRICS = METRICS_DIR / f"loco_models_{TARGET_COLUMN}_test.csv"
MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]
GROUP_ORDER = [
    "near_zero",
    "low",
    "moderate",
    "high",
]


def prevalence_group(event_rate: float) -> str:
    if event_rate < 0.01:
        return "near_zero"
    if event_rate < 0.10:
        return "low"
    if event_rate < 0.30:
        return "moderate"
    return "high"


def load_loco_metrics() -> pd.DataFrame:
    df = pd.read_csv(LOCO_TEST_METRICS)
    df = df[df["part"] == "test"].copy()
    df["prevalence_group"] = df["event_rate"].apply(prevalence_group)
    df["prevalence_group"] = pd.Categorical(
        df["prevalence_group"],
        categories=GROUP_ORDER,
        ordered=True,
    )
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def build_country_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    country_table = (
        df[["held_out_country", "rows", "events", "event_rate", "prevalence_group"]]
        .drop_duplicates()
        .sort_values("event_rate", ascending=False)
        .reset_index(drop=True)
    )
    return country_table


def build_group_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["prevalence_group", "model", "model_family"], observed=True)
        .agg(
            countries=("held_out_country", "nunique"),
            mean_event_rate=("event_rate", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            median_pr_auc=("pr_auc", "median"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            mean_missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
        .sort_values(["prevalence_group", "mean_pr_auc"], ascending=[True, False])
    )
    return summary


def build_best_model_by_country(df: pd.DataFrame) -> pd.DataFrame:
    best = (
        df.sort_values(["held_out_country", "pr_auc"], ascending=[True, False])
        .groupby("held_out_country", observed=True)
        .head(1)
        .sort_values("event_rate", ascending=False)
        .reset_index(drop=True)
    )
    return best[
        [
            "held_out_country",
            "prevalence_group",
            "rows",
            "events",
            "event_rate",
            "model",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_alarm_rate",
            "missed_event_rate",
        ]
    ]


def build_problematic_country_notes(country_table: pd.DataFrame) -> pd.DataFrame:
    notes = country_table.copy()
    notes["methodological_note"] = notes["prevalence_group"].map(
        {
            "near_zero": "Near-zero event prevalence; PR-AUC and F1 are unstable or undefined.",
            "low": "Low event prevalence; precision is sensitive to small false-positive counts.",
            "moderate": "Moderate event prevalence; useful for balanced transfer analysis.",
            "high": "High event prevalence; recall is important but majority-risk bias must be checked.",
        }
    )
    return notes


def save_figures(df: pd.DataFrame, country_table: pd.DataFrame, summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    country_plot = country_table.sort_values("event_rate", ascending=True)
    plt.figure(figsize=(11, 9))
    sns.barplot(
        data=country_plot,
        y="held_out_country",
        x="event_rate",
        hue="prevalence_group",
        hue_order=GROUP_ORDER,
        dodge=False,
        palette=["#8d99ae", "#457b9d", "#2a9d8f", "#b23a48"],
    )
    plt.title("Prevalencia de eventos no teste LOCO por pais")
    plt.xlabel("Prevalencia de AQI >= 151")
    plt.ylabel("Pais excluido")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_country_event_prevalence_groups.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="prevalence_group",
        y="mean_pr_auc",
        hue="model",
        order=GROUP_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#999999", "#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("PR-AUC LOCO por grupo de prevalencia")
    plt.xlabel("Grupo de prevalencia")
    plt.ylabel("PR-AUC medio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_group_pr_auc_by_model.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="prevalence_group",
        y="mean_recall",
        hue="model",
        order=GROUP_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#999999", "#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("Recall LOCO por grupo de prevalencia")
    plt.xlabel("Grupo de prevalencia")
    plt.ylabel("Recall medio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_group_recall_by_model.png", dpi=170)
    plt.close()

    heatmap_data = (
        df.pivot_table(
            index="held_out_country",
            columns="model",
            values="pr_auc",
            aggfunc="mean",
            observed=True,
        )
        .reindex(columns=MODEL_ORDER)
        .reindex(country_table.sort_values("event_rate", ascending=False)["held_out_country"])
    )
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        heatmap_data,
        cmap="mako",
        cbar_kws={"label": "PR-AUC"},
    )
    plt.title("PR-AUC LOCO ordenado por prevalencia de eventos")
    plt.xlabel("Modelo")
    plt.ylabel("Pais excluido")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_pr_auc_by_country_prevalence_order.png", dpi=170)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    df = load_loco_metrics()
    country_table = build_country_prevalence_table(df)
    group_summary = build_group_model_summary(df)
    best_by_country = build_best_model_by_country(df)
    notes = build_problematic_country_notes(country_table)

    country_table.to_csv(TABLES_DIR / "loco_country_prevalence_groups.csv", index=False)
    group_summary.to_csv(TABLES_DIR / "loco_group_model_summary.csv", index=False)
    best_by_country.to_csv(TABLES_DIR / "loco_best_model_by_country.csv", index=False)
    notes.to_csv(TABLES_DIR / "loco_problematic_country_notes.csv", index=False)

    save_figures(df, country_table, group_summary)

    print("LOCO country-group analysis completed.")
    print()
    print("Countries by prevalence group:")
    print(
        country_table.groupby("prevalence_group", observed=True)
        .agg(countries=("held_out_country", "count"), mean_event_rate=("event_rate", "mean"))
        .reset_index()
        .to_string(index=False)
    )
    print()
    print("Group/model summary:")
    print(
        group_summary[
            [
                "prevalence_group",
                "model",
                "countries",
                "mean_event_rate",
                "mean_pr_auc",
                "mean_recall",
                "mean_f1",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
