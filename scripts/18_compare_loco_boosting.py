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


TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "lightgbm",
    "xgboost",
]


def load_by_model() -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        path = METRICS_DIR / f"loco_models_{target}_test_by_model.csv"
        frame = pd.read_csv(path)
        frame["target"] = target
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_by_country() -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        path = METRICS_DIR / f"loco_models_{target}_test.csv"
        frame = pd.read_csv(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ensure_results_dirs()
    by_model = load_by_model()
    by_country = load_by_country()

    by_model["model"] = pd.Categorical(by_model["model"], categories=MODEL_ORDER, ordered=True)
    by_model = by_model.sort_values(["target", "mean_pr_auc"], ascending=[True, False])
    by_model.to_csv(TABLES_DIR / "loco_boosting_comparison_by_model.csv", index=False)

    best_by_country = (
        by_country.sort_values("pr_auc", ascending=False)
        .groupby(["target", "held_out_country"], observed=True)
        .head(1)
        .sort_values(["target", "held_out_country"])
    )
    best_by_country.to_csv(TABLES_DIR / "loco_boosting_best_model_by_country.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_df = by_model[by_model["model"].isin(MODEL_ORDER)].copy()
    plot_df["target"] = plot_df["target"].astype(str)
    plot_df["model"] = plot_df["model"].astype(str)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="target",
        y="mean_pr_auc",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=["#adb5bd", "#6c757d", "#2a9d8f", "#3d5a80", "#f4a261", "#b23a48"],
    )
    plt.title("LOCO - PR-AUC medio por modelo")
    plt.xlabel("Alvo")
    plt.ylabel("PR-AUC medio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_boosting_pr_auc_by_model.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="target",
        y="mean_f1",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=["#adb5bd", "#6c757d", "#2a9d8f", "#3d5a80", "#f4a261", "#b23a48"],
    )
    plt.title("LOCO - F1 medio por modelo")
    plt.xlabel("Alvo")
    plt.ylabel("F1 medio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loco_boosting_f1_by_model.png", dpi=170)
    plt.close()

    print("LOCO boosting comparison completed.")
    print()
    print(
        by_model[
            [
                "target",
                "model",
                "countries",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "mean_roc_auc",
                "mean_pr_auc",
                "mean_false_alarm_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
