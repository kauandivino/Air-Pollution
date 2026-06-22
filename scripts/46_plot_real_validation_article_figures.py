from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"
FIGURES_DIR = RESULTS_REAL_DIR / "figures"

PALETTE = {
    "Base": "#4c78a8",
    "Local features": "#59a14f",
    "Country-class balanced": "#f28e2b",
    "default_0.50": "#4c78a8",
    "validation_f1": "#e15759",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def clean_model_label(model: str) -> str:
    labels = {
        "logistic_regression": "Logistic\nRegression",
        "random_forest": "Random\nForest",
        "xgboost": "XGBoost",
    }
    return labels.get(model, model)


def annotate_bars(ax: plt.Axes, decimals: int = 2) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=f"%.{decimals}f", fontsize=8, padding=2)


def save_main_pr_auc(main: pd.DataFrame) -> None:
    plot_data = main.copy()
    plot_data["model_label"] = plot_data["model"].map(clean_model_label)

    order = ["logistic_regression", "random_forest", "xgboost"]
    config_order = ["Base", "Local features", "Country-class balanced"]
    plot_data["model_label"] = pd.Categorical(
        plot_data["model_label"],
        categories=[clean_model_label(model) for model in order],
        ordered=True,
    )
    plot_data["configuration"] = pd.Categorical(
        plot_data["configuration"], categories=config_order, ordered=True
    )

    plt.figure(figsize=(8.4, 4.6))
    ax = sns.barplot(
        data=plot_data,
        x="model_label",
        y="mean_pr_auc",
        hue="configuration",
        palette=[PALETTE[label] for label in config_order],
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean PR-AUC")
    ax.set_ylim(0, max(0.48, plot_data["mean_pr_auc"].max() + 0.08))
    ax.set_title("Real-data LOCO validation: PM2.5 h=1")
    ax.legend(title="Configuration", frameon=False, loc="upper center", ncol=3)
    annotate_bars(ax)
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_external_validation_main_pr_auc.png", dpi=180)
    plt.close()


def save_xgboost_ladder(main: pd.DataFrame) -> None:
    plot_data = main[main["model"] == "xgboost"].copy()
    metrics = [
        ("mean_pr_auc", "PR-AUC"),
        ("mean_f1", "F1"),
        ("mean_recall", "Recall"),
        ("mean_false_alarm_rate", "FAR"),
    ]
    plot_data = plot_data.melt(
        id_vars=["configuration"],
        value_vars=[metric for metric, _ in metrics],
        var_name="metric",
        value_name="value",
    )
    metric_labels = dict(metrics)
    plot_data["metric"] = plot_data["metric"].map(metric_labels)
    config_order = ["Base", "Local features", "Country-class balanced"]
    plot_data["configuration"] = pd.Categorical(
        plot_data["configuration"], categories=config_order, ordered=True
    )

    plt.figure(figsize=(8.2, 4.6))
    ax = sns.barplot(
        data=plot_data,
        x="metric",
        y="value",
        hue="configuration",
        palette=[PALETTE[label] for label in config_order],
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean value")
    ax.set_ylim(0, max(0.48, plot_data["value"].max() + 0.08))
    ax.set_title("XGBoost real-data ladder under LOCO")
    ax.legend(title="Configuration", frameon=False, loc="upper center", ncol=3)
    annotate_bars(ax)
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_external_validation_xgboost_ladder.png", dpi=180)
    plt.close()


def save_threshold_tradeoff(threshold: pd.DataFrame) -> None:
    plot_data = threshold.copy()
    plot_data["false_alerts_per_100_city_months"] = plot_data[
        "false_alerts_per_100_city_months"
    ].astype(float)
    plot_data["label"] = plot_data["configuration"] + " | " + plot_data["threshold_policy"]

    plt.figure(figsize=(7.2, 5.0))
    ax = sns.scatterplot(
        data=plot_data,
        x="false_alerts_per_100_city_months",
        y="mean_recall",
        hue="threshold_policy",
        style="configuration",
        s=120,
        palette=[PALETTE["default_0.50"], PALETTE["validation_f1"]],
    )

    for _, row in plot_data.iterrows():
        if (
            row["threshold_policy"] == "default_0.50"
            and row["configuration"] != "Country-class balanced"
        ):
            continue
        label = row["configuration"].replace("Country-class balanced", "CC balanced")
        if row["threshold_policy"] == "validation_f1":
            label = f"{label} calibrated"
        ax.annotate(
            label,
            (
                row["false_alerts_per_100_city_months"],
                row["mean_recall"],
            ),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    ax.set_xlabel("False alerts per 100 country-months")
    ax.set_ylabel("Mean recall")
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=-0.04, top=max(0.8, plot_data["mean_recall"].max() + 0.08))
    ax.set_title("Operational threshold trade-off in real-data LOCO")
    legend = ax.legend(title="", frameon=False, loc="lower right", fontsize=9)
    replacements = {
        "threshold_policy": "",
        "default_0.50": "Threshold 0.50",
        "validation_f1": "Validation-calibrated",
        "configuration": "",
    }
    for text in legend.get_texts():
        text.set_text(replacements.get(text.get_text(), text.get_text()))
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_operational_threshold_tradeoff.png", dpi=180)
    plt.close()


def save_bridge_heatmap(bridge: pd.DataFrame) -> None:
    score_map = {"no": 0.0, "partial": 0.5, "yes": 1.0}
    label_map = {"no": "No", "partial": "Partial", "yes": "Yes"}
    plot_data = bridge[["finding", "synthetic", "real"]].copy()
    plot_data = plot_data.set_index("finding")
    numeric = plot_data.replace(score_map).astype(float)
    labels = plot_data.replace(label_map)

    plt.figure(figsize=(7.4, 3.4))
    ax = sns.heatmap(
        numeric,
        annot=labels,
        fmt="",
        cmap=sns.color_palette(["#f2f2f2", "#f6c85f", "#59a14f"], as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 10},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Directional consistency between synthetic and real validation")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_vs_real_direction_bridge.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    main = read_csv(TABLES_DIR / "real_external_validation_main_result.csv")
    threshold = read_csv(TABLES_DIR / "real_operational_threshold_article_table.csv")
    bridge = read_csv(TABLES_DIR / "synthetic_vs_real_finding_bridge.csv")

    save_main_pr_auc(main)
    save_xgboost_ladder(main)
    save_threshold_tradeoff(threshold)
    save_bridge_heatmap(bridge)

    print("Saved article figures:")
    for name in [
        "real_external_validation_main_pr_auc.png",
        "real_external_validation_xgboost_ladder.png",
        "real_operational_threshold_tradeoff.png",
        "synthetic_vs_real_direction_bridge.png",
    ]:
        print(FIGURES_DIR / name)


if __name__ == "__main__":
    main()
