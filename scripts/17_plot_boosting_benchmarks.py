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


MODEL_ORDER = ["random_forest", "lightgbm", "xgboost"]
TARGET_ORDER = ["extreme_abs_151_h1", "extreme_abs_151_h3"]


def main() -> None:
    ensure_results_dirs()
    summary = pd.read_csv(TABLES_DIR / "boosting_benchmark_test_summary.csv")
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary["target"] = pd.Categorical(summary["target"], categories=TARGET_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="target",
        y="pr_auc",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("Boosting benchmark - PR-AUC")
    plt.xlabel("Alvo")
    plt.ylabel("PR-AUC")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boosting_benchmark_pr_auc.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="target",
        y="f1",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#3d5a80", "#2a9d8f", "#b23a48"],
    )
    plt.title("Boosting benchmark - F1")
    plt.xlabel("Alvo")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boosting_benchmark_f1.png", dpi=170)
    plt.close()

    best = (
        summary.sort_values("pr_auc", ascending=False)
        .groupby(["target", "split_name"], observed=True)
        .head(1)
        .sort_values(["target", "split_name"])
    )
    best.to_csv(TABLES_DIR / "boosting_benchmark_best_by_target.csv", index=False)

    print("Boosting benchmark figures completed.")
    print()
    print("Best boosting-family model by target/protocol:")
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
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
