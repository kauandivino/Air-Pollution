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


TARGET = "extreme_abs_151_h3"
METRICS = ["pr_auc", "f1", "recall", "precision"]

LADDER_ROWS = [
    {
        "step": 1,
        "stage": "RF base LOCO",
        "source": "base_loco",
        "model": "random_forest",
        "interpretation": "Baseline forte com features globais, antes de boosting.",
    },
    {
        "step": 2,
        "stage": "XGBoost base LOCO",
        "source": "base_loco",
        "model": "xgboost",
        "interpretation": "Boosting melhora o ranqueamento, mas ainda sofre sob LOCO.",
    },
    {
        "step": 3,
        "stage": "XGBoost + local features",
        "source": "local_features",
        "model": "xgboost",
        "interpretation": "Representacao local/anomalias reduz parte do shift geografico.",
    },
    {
        "step": 4,
        "stage": "XGBoost + similarity weighting",
        "source": "similarity_weighting",
        "model": "xgboost",
        "temperature": 0.5,
        "interpretation": "Paises ambientalmente similares melhoram ranking, mas sacrificam recall.",
    },
    {
        "step": 5,
        "stage": "XGBoost + country-class weighting",
        "source": "domain_weighting",
        "model": "xgboost",
        "weighting_strategy": "country_class_balanced",
        "interpretation": "Balancear pais e classe aumenta fortemente recall e F1.",
    },
]


def mean_metrics(data: pd.DataFrame) -> dict[str, float]:
    test = data[data["part"] == "test"].copy()
    return {metric: test[metric].mean() for metric in METRICS}


def load_base_loco(model: str) -> dict[str, float]:
    path = METRICS_DIR / f"loco_models_{TARGET}_test.csv"
    data = pd.read_csv(path)
    data = data[(data["target"] == TARGET) & (data["model"] == model)]
    if data.empty:
        raise RuntimeError(f"No base LOCO rows found for {TARGET} | {model}")
    return mean_metrics(data)


def load_local_features(model: str) -> dict[str, float]:
    checkpoint_dir = METRICS_DIR / "loco_local_feature_checkpoints"
    paths = sorted(checkpoint_dir.glob(f"{TARGET}__{model}__*.csv"))
    if not paths:
        raise RuntimeError(f"No local-feature checkpoints found for {TARGET} | {model}")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    data = data[(data["target"] == TARGET) & (data["model"] == model)]
    return mean_metrics(data)


def load_similarity_weighting(model: str, temperature: float) -> dict[str, float]:
    checkpoint_dir = METRICS_DIR / "loco_similarity_weighting_checkpoints"
    paths = sorted(checkpoint_dir.glob(f"{TARGET}__{model}__local__temp_*.csv"))
    if not paths:
        raise RuntimeError(f"No similarity-weighting checkpoints found for {TARGET} | {model}")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    data = data[
        (data["target"] == TARGET)
        & (data["model"] == model)
        & (data["temperature"] == temperature)
    ]
    if data.empty:
        raise RuntimeError(
            f"No similarity-weighting rows found for {TARGET} | {model} | temp={temperature}"
        )
    return mean_metrics(data)


def load_domain_weighting(model: str, strategy: str) -> dict[str, float]:
    checkpoint_dir = METRICS_DIR / "loco_domain_weighting_checkpoints"
    paths = sorted(checkpoint_dir.glob(f"{TARGET}__{model}__local__{strategy}__*.csv"))
    if not paths:
        raise RuntimeError(f"No domain-weighting checkpoints found for {TARGET} | {model}")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    data = data[
        (data["target"] == TARGET)
        & (data["model"] == model)
        & (data["weighting_strategy"] == strategy)
    ]
    return mean_metrics(data)


def build_ladder() -> pd.DataFrame:
    rows = []
    for config in LADDER_ROWS:
        source = config["source"]
        model = config["model"]
        if source == "base_loco":
            metrics = load_base_loco(model)
        elif source == "local_features":
            metrics = load_local_features(model)
        elif source == "similarity_weighting":
            metrics = load_similarity_weighting(model, config["temperature"])
        elif source == "domain_weighting":
            metrics = load_domain_weighting(model, config["weighting_strategy"])
        else:
            raise ValueError(f"Unknown ladder source: {source}")

        rows.append(
            {
                "target": TARGET,
                "step": config["step"],
                "stage": config["stage"],
                "model": model,
                "source": source,
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "interpretation": config["interpretation"],
            }
        )

    ladder = pd.DataFrame(rows).sort_values("step")
    for metric in METRICS:
        ladder[f"{metric}_delta_vs_previous"] = ladder[metric].diff()
        ladder[f"{metric}_delta_vs_xgboost_base"] = (
            ladder[metric] - ladder.loc[ladder["stage"] == "XGBoost base LOCO", metric].iloc[0]
        )
    return ladder


def save_markdown_table(ladder: pd.DataFrame) -> None:
    display = ladder[
        ["step", "stage", "pr_auc", "f1", "recall", "precision", "interpretation"]
    ].copy()
    for metric in METRICS:
        display[metric] = display[metric].map(lambda value: f"{value:.3f}")

    path = TABLES_DIR / "methodological_ladder_xgboost_h3.md"
    with path.open("w", encoding="utf-8") as file:
        file.write("# Methodological ladder - XGBoost h=3\n\n")
        headers = display.columns.tolist()
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in display.iterrows():
            values = [str(row[column]) for column in headers]
            file.write("| " + " | ".join(values) + " |\n")
        file.write("\n")


def plot_ladder_metrics(ladder: pd.DataFrame) -> None:
    plot_df = ladder.melt(
        id_vars=["step", "stage"],
        value_vars=METRICS,
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].str.upper().str.replace("_", "-", regex=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6.5))
    axis = sns.lineplot(
        data=plot_df,
        x="step",
        y="score",
        hue="metric",
        marker="o",
        linewidth=2.2,
        markersize=8,
        palette=["#2a9d8f", "#8d5a97", "#b23a48", "#466b8f"],
    )
    axis.set_title("Escada metodologica no LOCO: extreme_abs_151_h3")
    axis.set_xlabel("")
    axis.set_ylabel("Score medio por pais retido")
    axis.set_xticks(ladder["step"])
    axis.set_xticklabels(ladder["stage"], rotation=20, ha="right")
    axis.set_ylim(0, max(0.45, plot_df["score"].max() + 0.06))
    axis.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "methodological_ladder_xgboost_h3_metrics.png", dpi=180)
    plt.close()


def plot_ladder_heatmap(ladder: pd.DataFrame) -> None:
    matrix = ladder.set_index("stage")[METRICS]
    matrix.columns = [column.upper().replace("_", "-") for column in matrix.columns]

    plt.figure(figsize=(8.5, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score medio"},
    )
    plt.title("Escada metodologica - metricas principais")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "methodological_ladder_xgboost_h3_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    ladder = build_ladder()

    csv_path = TABLES_DIR / "methodological_ladder_xgboost_h3.csv"
    ladder.to_csv(csv_path, index=False)
    save_markdown_table(ladder)
    plot_ladder_metrics(ladder)
    plot_ladder_heatmap(ladder)

    print("Methodological ladder completed.")
    print()
    print(
        ladder[
            ["step", "stage", "pr_auc", "f1", "recall", "precision", "interpretation"]
        ].to_string(index=False)
    )
    print()
    print(f"Table saved to: {csv_path}")
    print(f"Markdown table saved to: {TABLES_DIR / 'methodological_ladder_xgboost_h3.md'}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
