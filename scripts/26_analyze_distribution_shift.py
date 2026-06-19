from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs


FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
LOCAL_DELTA_TABLE = TABLES_DIR / "local_features_base_vs_local_delta_by_country.csv"

SHIFT_COLUMNS = [
    "AQI",
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "CO (mg/m3)",
    "NO2 (ug/m3)",
    "O3 (ug/m3)",
    "Wind_Speed (km/h)",
    "Humidity (%)",
    "Deforestation_Rate_%",
]

SHIFT_SUMMARY_COLUMNS = [
    "smd_mean_abs",
    "smd_max_abs",
    "wasserstein_mean_norm",
    "wasserstein_max_norm",
    "ks_mean",
    "ks_max",
    "aqi_smd_abs",
    "aqi_wasserstein_norm",
    "aqi_ks",
]

OUTCOME_COLUMNS = [
    "pr_auc_base",
    "f1_base",
    "pr_auc_delta",
    "f1_delta",
]

EXPERIMENT_LABELS = {
    ("extreme_abs_151_h1", "xgboost"): "XGBoost h=1",
    ("extreme_abs_151_h3", "xgboost"): "XGBoost h=3",
    ("extreme_abs_151_h1", "random_forest"): "Random Forest h=1",
}
EXPERIMENT_ORDER = list(EXPERIMENT_LABELS.values())


def experiment_label(target: str, model: str) -> str:
    return EXPERIMENT_LABELS.get((target, model), f"{model} | {target}")


def load_feature_matrix(targets: list[str]) -> pd.DataFrame:
    usecols = ["Country", *SHIFT_COLUMNS, *targets]
    return pd.read_csv(FEATURE_MATRIX, usecols=usecols)


def normalized_wasserstein(test_values: pd.Series, train_values: pd.Series) -> float:
    test_values = test_values.dropna()
    train_values = train_values.dropna()
    if test_values.empty or train_values.empty:
        return np.nan
    scale = pd.concat([test_values, train_values]).std()
    if not scale or np.isnan(scale):
        return np.nan
    return wasserstein_distance(test_values, train_values) / scale


def standardized_mean_difference(test_values: pd.Series, train_values: pd.Series) -> float:
    test_values = test_values.dropna()
    train_values = train_values.dropna()
    if test_values.empty or train_values.empty:
        return np.nan
    pooled = pd.concat([test_values, train_values]).std()
    if not pooled or np.isnan(pooled):
        return np.nan
    return (test_values.mean() - train_values.mean()) / pooled


def ks_statistic(test_values: pd.Series, train_values: pd.Series) -> float:
    test_values = test_values.dropna()
    train_values = train_values.dropna()
    if test_values.empty or train_values.empty:
        return np.nan
    return ks_2samp(test_values, train_values).statistic


def build_shift_table(feature_matrix: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        target_df = feature_matrix.dropna(subset=[target]).copy()
        for country in sorted(target_df["Country"].unique()):
            test_df = target_df[target_df["Country"] == country]
            train_df = target_df[target_df["Country"] != country]
            row = {
                "target": target,
                "held_out_country": country,
                "test_observations": len(test_df),
                "train_observations": len(train_df),
                "test_event_rate": test_df[target].mean(),
                "train_event_rate": train_df[target].mean(),
                "event_rate_gap_abs": abs(test_df[target].mean() - train_df[target].mean()),
            }

            smds = []
            wassersteins = []
            ks_values = []
            for column in SHIFT_COLUMNS:
                test_values = test_df[column]
                train_values = train_df[column]
                smd = standardized_mean_difference(test_values, train_values)
                wasserstein = normalized_wasserstein(test_values, train_values)
                ks_value = ks_statistic(test_values, train_values)

                safe_name = (
                    column.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                    .replace("%", "pct")
                    .replace(".", "")
                )
                row[f"{safe_name}_smd"] = smd
                row[f"{safe_name}_wasserstein_norm"] = wasserstein
                row[f"{safe_name}_ks"] = ks_value

                smds.append(abs(smd))
                wassersteins.append(wasserstein)
                ks_values.append(ks_value)

            row["smd_mean_abs"] = np.nanmean(smds)
            row["smd_max_abs"] = np.nanmax(smds)
            row["wasserstein_mean_norm"] = np.nanmean(wassersteins)
            row["wasserstein_max_norm"] = np.nanmax(wassersteins)
            row["ks_mean"] = np.nanmean(ks_values)
            row["ks_max"] = np.nanmax(ks_values)
            row["aqi_smd_abs"] = abs(row["AQI_smd"])
            row["aqi_wasserstein_norm"] = row["AQI_wasserstein_norm"]
            row["aqi_ks"] = row["AQI_ks"]
            rows.append(row)

    return pd.DataFrame(rows)


def build_analysis_table(shift: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    table = deltas.merge(
        shift,
        on=["target", "held_out_country"],
        how="left",
        validate="many_to_one",
    )
    table["experiment"] = [
        experiment_label(target, model)
        for target, model in zip(table["target"], table["model"])
    ]
    table["experiment"] = pd.Categorical(
        table["experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )
    return table


def build_correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in table.groupby(["target", "model"], observed=True):
        for shift_metric in SHIFT_SUMMARY_COLUMNS:
            for outcome in OUTCOME_COLUMNS:
                pair = group[[shift_metric, outcome]].dropna()
                if len(pair) < 4 or pair[shift_metric].nunique() < 2 or pair[outcome].nunique() < 2:
                    statistic = p_value = np.nan
                else:
                    result = spearmanr(pair[shift_metric], pair[outcome])
                    statistic = result.statistic
                    p_value = result.pvalue
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "experiment": experiment_label(target, model),
                        "shift_metric": shift_metric,
                        "outcome": outcome,
                        "n_countries": len(pair),
                        "spearman_r": statistic,
                        "spearman_p": p_value,
                    }
                )
    return pd.DataFrame(rows)


def plot_shift_vs_outcome(table: pd.DataFrame, shift_metric: str, outcome: str) -> None:
    sns.set_theme(style="whitegrid")
    grid = sns.lmplot(
        data=table,
        x=shift_metric,
        y=outcome,
        hue="experiment",
        hue_order=EXPERIMENT_ORDER,
        palette=["#b23a48", "#2a9d8f", "#3d5a80"],
        col="experiment",
        col_order=EXPERIMENT_ORDER,
        height=4.2,
        aspect=0.95,
        scatter_kws={"s": 58, "alpha": 0.85},
        line_kws={"linewidth": 1.8},
        ci=None,
    )
    grid.set_axis_labels(shift_metric.replace("_", " "), outcome.replace("_", " ").upper())
    grid.set_titles("{col_name}")
    grid.figure.suptitle(
        f"Distribution shift vs {outcome.replace('_', ' ').upper()}",
        y=1.05,
    )
    grid.tight_layout()
    grid.savefig(FIGURES_DIR / f"distribution_shift_{shift_metric}_vs_{outcome}.png", dpi=180)
    plt.close(grid.figure)


def plot_correlation_heatmap(correlations: pd.DataFrame, outcome: str) -> None:
    corr = correlations[correlations["outcome"] == outcome].copy()
    pivot = corr.pivot_table(
        index="shift_metric",
        columns="experiment",
        values="spearman_r",
        observed=True,
    ).reindex(index=SHIFT_SUMMARY_COLUMNS, columns=EXPERIMENT_ORDER)

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Spearman r"},
    )
    plt.title(f"Distribution shift correlations: {outcome.replace('_', ' ').upper()}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"distribution_shift_correlation_{outcome}.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    deltas = pd.read_csv(LOCAL_DELTA_TABLE)
    targets = sorted(deltas["target"].unique())

    feature_matrix = load_feature_matrix(targets)
    shift = build_shift_table(feature_matrix, targets)
    analysis = build_analysis_table(shift, deltas)
    correlations = build_correlation_table(analysis)

    shift_path = TABLES_DIR / "distribution_shift_by_country.csv"
    analysis_path = TABLES_DIR / "distribution_shift_local_gain_analysis.csv"
    correlations_path = TABLES_DIR / "distribution_shift_correlations.csv"

    shift.to_csv(shift_path, index=False)
    analysis.to_csv(analysis_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    plot_shift_vs_outcome(analysis, "wasserstein_mean_norm", "pr_auc_base")
    plot_shift_vs_outcome(analysis, "wasserstein_mean_norm", "pr_auc_delta")
    plot_shift_vs_outcome(analysis, "smd_mean_abs", "f1_delta")
    for outcome in OUTCOME_COLUMNS:
        plot_correlation_heatmap(correlations, outcome)

    strongest = (
        correlations.assign(abs_spearman=lambda df: df["spearman_r"].abs())
        .sort_values("abs_spearman", ascending=False)
        .head(12)
    )

    print("Distribution-shift analysis completed.")
    print()
    print("Strongest shift/outcome associations:")
    print(
        strongest[
            [
                "experiment",
                "shift_metric",
                "outcome",
                "n_countries",
                "spearman_r",
                "spearman_p",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Shift table saved to: {shift_path}")
    print(f"Analysis table saved to: {analysis_path}")
    print(f"Correlations saved to: {correlations_path}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
