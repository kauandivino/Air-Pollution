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


TARGET_ORDER = [
    "extreme_abs_151_h1",
    "extreme_abs_151_h3",
    "extreme_abs_201_h1",
    "extreme_abs_201_h3",
    "extreme_city_p90_h1",
    "extreme_city_p90_h3",
    "extreme_city_p95_h1",
    "extreme_city_p95_h3",
]
MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
]


def load_all_random_temporal_metrics() -> pd.DataFrame:
    frames = []

    legacy_path = METRICS_DIR / "initial_models_random_temporal.csv"
    if legacy_path.exists():
        frames.append(pd.read_csv(legacy_path))

    multi_path = METRICS_DIR / "initial_models_multi_target_random_temporal.csv"
    if multi_path.exists():
        frames.append(pd.read_csv(multi_path))

    if not frames:
        raise FileNotFoundError("No random/temporal metrics found.")

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.drop_duplicates(
        subset=["target", "split_name", "model", "part"],
        keep="last",
    )
    return metrics[metrics["part"] == "test"].copy()


def add_target_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["target_type"] = result["target"].map(
        lambda value: "absolute" if "_abs_" in value else "city_relative"
    )
    result["severity"] = result["target"].str.extract(r"(151|201|p90|p95)")[0]
    result["horizon"] = result["target"].str.extract(r"_h(\d+)")[0].astype(int)
    result["target"] = pd.Categorical(result["target"], categories=TARGET_ORDER, ordered=True)
    result["model"] = pd.Categorical(result["model"], categories=MODEL_ORDER, ordered=True)
    return result


def build_target_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "target",
                "target_type",
                "severity",
                "horizon",
                "split_name",
                "model",
                "model_family",
            ],
            observed=True,
        )
        .agg(
            rows=("rows", "mean"),
            event_rate=("event_rate", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            false_alarm_rate=("false_alarm_rate", "mean"),
            missed_event_rate=("missed_event_rate", "mean"),
        )
        .reset_index()
        .sort_values(["target", "split_name", "pr_auc"], ascending=[True, True, False])
    )


def build_question_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best = (
        summary.sort_values("pr_auc", ascending=False)
        .groupby(["target", "split_name"], observed=True)
        .head(1)
        .sort_values(["target", "split_name"])
    )

    horizon_gap = _paired_gap(
        summary,
        pair_key_columns=["target_type", "severity", "split_name", "model"],
        pair_column="horizon",
        left_value=1,
        right_value=3,
        prefix_left="h1",
        prefix_right="h3",
    )

    severity_gap = _paired_gap(
        summary[summary["target_type"] == "absolute"],
        pair_key_columns=["horizon", "split_name", "model"],
        pair_column="severity",
        left_value="151",
        right_value="201",
        prefix_left="abs151",
        prefix_right="abs201",
    )

    relative_gap = _paired_gap(
        summary,
        pair_key_columns=["horizon", "split_name", "model"],
        pair_column="target_type",
        left_value="absolute",
        right_value="city_relative",
        prefix_left="absolute",
        prefix_right="city_relative",
        metric_filter=lambda frame: frame[
            ((frame["target_type"] == "absolute") & (frame["severity"] == "151"))
            | ((frame["target_type"] == "city_relative") & (frame["severity"] == "p90"))
        ],
    )

    return best, horizon_gap, severity_gap, relative_gap


def save_figures(summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plot_df = summary.copy()
    plot_df["target"] = plot_df["target"].astype(str)
    plot_df["model"] = plot_df["model"].astype(str)

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_df[plot_df["split_name"] == "temporal_split"],
        x="target",
        y="pr_auc",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#999999", "#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("PR-AUC por alvo no protocolo temporal")
    plt.xlabel("Alvo")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multi_target_temporal_pr_auc.png", dpi=170)
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_df[plot_df["split_name"] == "random_split"],
        x="target",
        y="pr_auc",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#999999", "#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("PR-AUC por alvo no random split")
    plt.xlabel("Alvo")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multi_target_random_pr_auc.png", dpi=170)
    plt.close()

    heatmap_data = plot_df.pivot_table(
        index="target",
        columns=["split_name", "model"],
        values="pr_auc",
        observed=True,
    ).reindex(TARGET_ORDER)
    plt.figure(figsize=(13, 8))
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={"label": "PR-AUC"})
    plt.title("PR-AUC por alvo, protocolo e modelo")
    plt.xlabel("Protocolo / modelo")
    plt.ylabel("Alvo")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multi_target_pr_auc_heatmap.png", dpi=170)
    plt.close()


def _paired_gap(
    summary: pd.DataFrame,
    pair_key_columns: list[str],
    pair_column: str,
    left_value,
    right_value,
    prefix_left: str,
    prefix_right: str,
    metric_filter=None,
) -> pd.DataFrame:
    frame = metric_filter(summary.copy()) if metric_filter else summary.copy()
    left = frame[frame[pair_column] == left_value].copy()
    right = frame[frame[pair_column] == right_value].copy()

    metrics = ["event_rate", "pr_auc", "recall", "f1", "precision"]
    left = left[pair_key_columns + metrics].rename(
        columns={metric: f"{prefix_left}_{metric}" for metric in metrics}
    )
    right = right[pair_key_columns + metrics].rename(
        columns={metric: f"{prefix_right}_{metric}" for metric in metrics}
    )
    merged = left.merge(right, on=pair_key_columns, how="inner")

    for metric in metrics:
        merged[f"delta_{prefix_left}_minus_{prefix_right}_{metric}"] = (
            merged[f"{prefix_left}_{metric}"] - merged[f"{prefix_right}_{metric}"]
        )
    return merged


def main() -> None:
    ensure_results_dirs()
    metrics = add_target_metadata(load_all_random_temporal_metrics())
    summary = build_target_model_summary(metrics)
    best, horizon_gap, severity_gap, relative_gap = build_question_tables(summary)

    summary.to_csv(TABLES_DIR / "multi_target_random_temporal_summary.csv", index=False)
    best.to_csv(TABLES_DIR / "multi_target_best_model_by_target.csv", index=False)
    horizon_gap.to_csv(TABLES_DIR / "multi_target_h1_vs_h3_gap.csv", index=False)
    severity_gap.to_csv(TABLES_DIR / "multi_target_abs151_vs_abs201_gap.csv", index=False)
    relative_gap.to_csv(TABLES_DIR / "multi_target_absolute_vs_relative_gap.csv", index=False)

    save_figures(summary)

    print("Multi-target comparison completed.")
    print()
    print("Best model by target/protocol:")
    print(
        best[
            [
                "target",
                "split_name",
                "model",
                "event_rate",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )
    print()
    print("Average h1 - h3 PR-AUC delta:")
    print(
        horizon_gap.groupby(["target_type", "severity", "split_name"], observed=True)[
            "delta_h1_minus_h3_pr_auc"
        ]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

