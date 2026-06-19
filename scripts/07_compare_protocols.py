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
METRICS = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_alarm_rate",
    "missed_event_rate",
]
MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]
PROTOCOL_ORDER = [
    "random_split",
    "temporal_split",
    "leave_one_country_out",
]


def load_protocol_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    random_temporal = pd.read_csv(METRICS_DIR / "initial_models_random_temporal.csv")
    random_temporal = random_temporal[random_temporal["part"] == "test"].copy()

    loco = pd.read_csv(METRICS_DIR / f"loco_models_{TARGET_COLUMN}_test.csv")
    loco["split_name"] = "leave_one_country_out"
    loco["part"] = "test"

    combined = pd.concat([random_temporal, loco], ignore_index=True, sort=False)
    combined = combined[combined["target"] == TARGET_COLUMN].copy()
    return combined, loco


def build_protocol_model_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, model, model_family), group in combined.groupby(
        ["split_name", "model", "model_family"],
        dropna=False,
    ):
        row = {
            "split_name": split_name,
            "model": model,
            "model_family": model_family,
            "evaluations": len(group),
            "countries": group["held_out_country"].nunique()
            if "held_out_country" in group.columns
            else 0,
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = group[metric].mean(skipna=True)
            row[f"std_{metric}"] = group[metric].std(skipna=True)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["split_name"] = pd.Categorical(
        summary["split_name"],
        categories=PROTOCOL_ORDER,
        ordered=True,
    )
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["split_name", "model"]).reset_index(drop=True)


def build_protocol_drop_table(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [f"mean_{metric}" for metric in METRICS]
    wide = summary.pivot(index="model", columns="split_name", values=metric_columns)
    rows = []

    for model in summary["model"].dropna().unique():
        row = {"model": str(model)}
        for metric in METRICS:
            random_value = _wide_value(wide, model, f"mean_{metric}", "random_split")
            temporal_value = _wide_value(wide, model, f"mean_{metric}", "temporal_split")
            loco_value = _wide_value(wide, model, f"mean_{metric}", "leave_one_country_out")

            row[f"random_{metric}"] = random_value
            row[f"temporal_{metric}"] = temporal_value
            row[f"loco_{metric}"] = loco_value
            row[f"drop_random_to_loco_{metric}"] = random_value - loco_value
            row[f"drop_temporal_to_loco_{metric}"] = temporal_value - loco_value
            row[f"relative_drop_random_to_loco_{metric}"] = _relative_drop(
                random_value,
                loco_value,
            )
            row[f"relative_drop_temporal_to_loco_{metric}"] = _relative_drop(
                temporal_value,
                loco_value,
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("drop_random_to_loco_pr_auc", ascending=False)


def build_loco_country_gap_table(combined: pd.DataFrame) -> pd.DataFrame:
    random_baseline = (
        combined[(combined["split_name"] == "random_split")]
        .set_index("model")[["pr_auc", "f1", "recall", "precision"]]
        .add_prefix("random_")
    )
    loco = combined[combined["split_name"] == "leave_one_country_out"].copy()
    loco = loco.merge(random_baseline, left_on="model", right_index=True, how="left")

    for metric in ["pr_auc", "f1", "recall", "precision"]:
        loco[f"drop_random_to_loco_{metric}"] = loco[f"random_{metric}"] - loco[metric]
        loco[f"relative_drop_random_to_loco_{metric}"] = _relative_drop_series(
            loco[f"random_{metric}"],
            loco[metric],
        )

    return loco.sort_values(["model", "drop_random_to_loco_pr_auc"], ascending=[True, False])


def save_figures(summary: pd.DataFrame, drop_table: pd.DataFrame, country_gap: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plot_summary = summary.copy()
    plot_summary["split_name"] = plot_summary["split_name"].astype(str)
    plot_summary["model"] = plot_summary["model"].astype(str)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_summary,
        x="model",
        y="mean_pr_auc",
        hue="split_name",
        order=MODEL_ORDER,
        hue_order=PROTOCOL_ORDER,
        palette=["#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("PR-AUC por protocolo de avaliacao")
    plt.xlabel("Modelo")
    plt.ylabel("PR-AUC medio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_comparison_pr_auc.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_summary,
        x="model",
        y="mean_f1",
        hue="split_name",
        order=MODEL_ORDER,
        hue_order=PROTOCOL_ORDER,
        palette=["#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("F1 por protocolo de avaliacao")
    plt.xlabel("Modelo")
    plt.ylabel("F1 medio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_comparison_f1.png", dpi=170)
    plt.close()

    drop_plot = drop_table.sort_values("drop_random_to_loco_pr_auc", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=drop_plot,
        x="model",
        y="drop_random_to_loco_pr_auc",
        order=drop_plot["model"].tolist(),
        color="#b23a48",
    )
    plt.title("Queda de PR-AUC: random split para leave-one-country-out")
    plt.xlabel("Modelo")
    plt.ylabel("Queda absoluta de PR-AUC")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_drop_random_to_loco_pr_auc.png", dpi=170)
    plt.close()

    heatmap_data = (
        country_gap.pivot_table(
            index="held_out_country",
            columns="model",
            values="pr_auc",
            aggfunc="mean",
        )
        .reindex(columns=MODEL_ORDER)
        .sort_index()
    )
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "PR-AUC"},
    )
    plt.title("PR-AUC LOCO por pais excluido e modelo")
    plt.xlabel("Modelo")
    plt.ylabel("Pais excluido")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_country_model_pr_auc_heatmap.png", dpi=170)
    plt.close()


def _wide_value(
    wide: pd.DataFrame,
    model: str,
    metric_column: str,
    split_name: str,
) -> float:
    try:
        return float(wide.loc[model, (metric_column, split_name)])
    except KeyError:
        return float("nan")


def _relative_drop(reference: float, comparison: float) -> float:
    if pd.isna(reference) or reference == 0:
        return float("nan")
    return (reference - comparison) / reference


def _relative_drop_series(reference: pd.Series, comparison: pd.Series) -> pd.Series:
    return (reference - comparison) / reference.replace(0, pd.NA)


def main() -> None:
    ensure_results_dirs()
    combined, loco = load_protocol_metrics()
    summary = build_protocol_model_summary(combined)
    drop_table = build_protocol_drop_table(summary)
    country_gap = build_loco_country_gap_table(combined)

    combined.to_csv(TABLES_DIR / "protocol_comparison_all_test_metrics.csv", index=False)
    summary.to_csv(TABLES_DIR / "protocol_comparison_model_summary.csv", index=False)
    drop_table.to_csv(TABLES_DIR / "protocol_comparison_drop_summary.csv", index=False)
    country_gap.to_csv(TABLES_DIR / "protocol_comparison_loco_country_gap.csv", index=False)

    save_figures(summary, drop_table, country_gap)

    display_columns = [
        "model",
        "random_pr_auc",
        "temporal_pr_auc",
        "loco_pr_auc",
        "drop_random_to_loco_pr_auc",
        "relative_drop_random_to_loco_pr_auc",
        "random_f1",
        "temporal_f1",
        "loco_f1",
        "drop_random_to_loco_f1",
    ]

    print("Protocol comparison completed.")
    print()
    print("Performance drop summary:")
    print(drop_table[display_columns].to_string(index=False))
    print()
    print("LOCO countries evaluated:", loco["held_out_country"].nunique())
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
