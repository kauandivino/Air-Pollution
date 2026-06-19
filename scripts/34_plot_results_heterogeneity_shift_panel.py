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


TARGET = "extreme_abs_151_h3"
MODEL = "xgboost"


def main() -> None:
    ensure_results_dirs()
    heterogeneity = pd.read_csv(TABLES_DIR / "local_gain_heterogeneity_by_country.csv")
    shift = pd.read_csv(TABLES_DIR / "distribution_shift_local_gain_analysis.csv")

    heterogeneity = heterogeneity[
        (heterogeneity["target"] == TARGET) & (heterogeneity["model"] == MODEL)
    ].copy()
    shift = shift[(shift["target"] == TARGET) & (shift["model"] == MODEL)].copy()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.regplot(
        data=heterogeneity,
        x="event_prevalence",
        y="f1_delta",
        ax=axes[0],
        scatter_kws={"s": 55, "alpha": 0.85, "color": "#2a9d8f"},
        line_kws={"color": "#333333", "linewidth": 1.8},
        ci=None,
    )
    axes[0].axhline(0, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_title("(a) Local-feature F1 gain vs event prevalence")
    axes[0].set_xlabel("Held-out country event prevalence")
    axes[0].set_ylabel("F1 delta (local - base)")

    sns.regplot(
        data=shift,
        x="aqi_wasserstein_norm",
        y="pr_auc_delta",
        ax=axes[1],
        scatter_kws={"s": 55, "alpha": 0.85, "color": "#b23a48"},
        line_kws={"color": "#333333", "linewidth": 1.8},
        ci=None,
    )
    axes[1].axhline(0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_title("(b) Local-feature PR-AUC gain vs AQI shift")
    axes[1].set_xlabel("AQI Wasserstein distance (normalized)")
    axes[1].set_ylabel("PR-AUC delta (local - base)")

    fig.suptitle("Country-level heterogeneity of local-feature gains", y=1.04)
    fig.tight_layout()
    output = FIGURES_DIR / "results_heterogeneity_shift_panel_xgboost_h3.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
