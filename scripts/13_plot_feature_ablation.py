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
from src.feature_sets import FEATURE_SET_ORDER


MODEL_ORDER = ["logistic_regression", "random_forest"]


def main() -> None:
    ensure_results_dirs()
    summary = pd.read_csv(TABLES_DIR / "feature_ablation_test_summary.csv")
    summary["feature_set"] = pd.Categorical(
        summary["feature_set"],
        categories=FEATURE_SET_ORDER,
        ordered=True,
    )
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["target", "split_name", "model", "feature_set"])

    incremental = summary.copy()
    incremental["previous_pr_auc"] = incremental.groupby(
        ["target", "split_name", "model"],
        observed=True,
    )["pr_auc"].shift(1)
    incremental["incremental_pr_auc_gain"] = incremental["pr_auc"] - incremental["previous_pr_auc"]
    incremental.to_csv(TABLES_DIR / "feature_ablation_incremental_gains.csv", index=False)

    sns.set_theme(style="whitegrid")

    plot_df = summary.copy()
    plot_df["feature_set"] = plot_df["feature_set"].astype(str)
    plot_df["model"] = plot_df["model"].astype(str)

    for target in sorted(plot_df["target"].unique()):
        target_df = plot_df[plot_df["target"] == target]
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=target_df,
            x="feature_set",
            y="pr_auc",
            hue="model",
            style="split_name",
            markers=True,
            dashes=False,
            hue_order=MODEL_ORDER,
        )
        plt.title(f"Ablation de features - {target}")
        plt.xlabel("Conjunto de features")
        plt.ylabel("PR-AUC")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"feature_ablation_pr_auc_{target}.png", dpi=170)
        plt.close()

    plt.figure(figsize=(13, 7))
    sns.barplot(
        data=plot_df,
        x="feature_set",
        y="pr_auc",
        hue="model",
        order=FEATURE_SET_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#3d5a80", "#b23a48"],
    )
    plt.title("Ablation de features - PR-AUC agregado")
    plt.xlabel("Conjunto de features")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_ablation_pr_auc_overall.png", dpi=170)
    plt.close()

    print("Feature ablation figures completed.")
    print()
    print("Incremental PR-AUC gains:")
    print(
        incremental[
            [
                "target",
                "split_name",
                "model",
                "feature_set",
                "n_features",
                "pr_auc",
                "incremental_pr_auc_gain",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
