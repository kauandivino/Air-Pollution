from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs


FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DELTA_TABLE = TABLES_DIR / "local_features_base_vs_local_delta_by_country.csv"

KEY_PROFILE_COLUMNS = [
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

DESCRIPTOR_COLUMNS = [
    "event_prevalence",
    "aqi_mean",
    "aqi_std",
    "aqi_p90",
    "aqi_p95",
    "aqi_seasonal_amplitude",
    "city_count",
    "observations",
    "aqi_mean_abs_smd_vs_global",
    "profile_shift_smd_mean",
    "profile_shift_smd_max",
]

DELTA_COLUMNS = ["pr_auc_delta", "f1_delta"]

EXPERIMENT_LABELS = {
    ("extreme_abs_151_h1", "xgboost"): "XGBoost h=1",
    ("extreme_abs_151_h3", "xgboost"): "XGBoost h=3",
    ("extreme_abs_151_h1", "random_forest"): "Random Forest h=1",
}
EXPERIMENT_ORDER = list(EXPERIMENT_LABELS.values())


def experiment_label(target: str, model: str) -> str:
    return EXPERIMENT_LABELS.get((target, model), f"{model} | {target}")


def load_required_feature_matrix(targets: list[str]) -> pd.DataFrame:
    usecols = ["Country", "City", "Date", *KEY_PROFILE_COLUMNS, *targets]
    return pd.read_csv(FEATURE_MATRIX, usecols=usecols, parse_dates=["Date"])


def smd(country_values: pd.DataFrame, global_means: pd.Series, global_stds: pd.Series) -> pd.DataFrame:
    safe_stds = global_stds.replace(0, np.nan)
    return (country_values - global_means) / safe_stds


def build_country_profiles(feature_matrix: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        target_df = feature_matrix.dropna(subset=[target]).copy()
        target_df["month"] = target_df["Date"].dt.month

        global_mean = target_df["AQI"].mean()
        global_std = target_df["AQI"].std()
        profile_global_means = target_df[KEY_PROFILE_COLUMNS].mean()
        profile_global_stds = target_df[KEY_PROFILE_COLUMNS].std()

        monthly_country = (
            target_df.groupby(["Country", "month"], observed=True)["AQI"]
            .mean()
            .reset_index()
        )
        seasonal = (
            monthly_country.groupby("Country", observed=True)["AQI"]
            .agg(aqi_seasonal_amplitude=lambda values: values.max() - values.min())
            .reset_index()
        )

        grouped = target_df.groupby("Country", observed=True)
        base = grouped.agg(
            observations=(target, "count"),
            events=(target, "sum"),
            event_prevalence=(target, "mean"),
            city_count=("City", "nunique"),
            aqi_mean=("AQI", "mean"),
            aqi_std=("AQI", "std"),
            aqi_p90=("AQI", lambda values: values.quantile(0.90)),
            aqi_p95=("AQI", lambda values: values.quantile(0.95)),
        ).reset_index()

        country_profile_means = grouped[KEY_PROFILE_COLUMNS].mean()
        country_smd = smd(country_profile_means, profile_global_means, profile_global_stds).abs()
        country_smd = country_smd.assign(
            profile_shift_smd_mean=country_smd.mean(axis=1),
            profile_shift_smd_max=country_smd.max(axis=1),
        )[["profile_shift_smd_mean", "profile_shift_smd_max"]].reset_index()

        base["aqi_mean_abs_smd_vs_global"] = (
            (base["aqi_mean"] - global_mean).abs() / global_std
            if global_std and not np.isnan(global_std)
            else np.nan
        )
        base = base.merge(seasonal, on="Country", how="left")
        base = base.merge(country_smd, on="Country", how="left")
        base["target"] = target
        rows.append(base)

    return pd.concat(rows, ignore_index=True)


def build_heterogeneity_table(profiles: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    table = deltas.merge(
        profiles,
        left_on=["target", "held_out_country"],
        right_on=["target", "Country"],
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
    return table.drop(columns=["Country"])


def correlation_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in table.groupby(["target", "model"], observed=True):
        for descriptor in DESCRIPTOR_COLUMNS:
            for delta_col in DELTA_COLUMNS:
                pair = group[[descriptor, delta_col]].dropna()
                if len(pair) < 4 or pair[descriptor].nunique() < 2 or pair[delta_col].nunique() < 2:
                    spearman_r = spearman_p = pearson_r = pearson_p = np.nan
                else:
                    spearman = spearmanr(pair[descriptor], pair[delta_col])
                    pearson = pearsonr(pair[descriptor], pair[delta_col])
                    spearman_r = spearman.statistic
                    spearman_p = spearman.pvalue
                    pearson_r = pearson.statistic
                    pearson_p = pearson.pvalue

                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "experiment": experiment_label(target, model),
                        "descriptor": descriptor,
                        "delta_metric": delta_col,
                        "n_countries": len(pair),
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                    }
                )
    return pd.DataFrame(rows)


def plot_scatter_grid(table: pd.DataFrame, delta_col: str) -> None:
    descriptors = [
        "event_prevalence",
        "aqi_std",
        "aqi_seasonal_amplitude",
        "profile_shift_smd_mean",
    ]
    titles = [
        "Prevalencia de eventos",
        "Variabilidade do AQI",
        "Amplitude sazonal do AQI",
        "Shift medio vs perfil global",
    ]

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, descriptor, title in zip(axes.flat, descriptors, titles):
        sns.scatterplot(
            data=table,
            x=descriptor,
            y=delta_col,
            hue="experiment",
            hue_order=EXPERIMENT_ORDER,
            palette=["#b23a48", "#2a9d8f", "#3d5a80"],
            s=65,
            alpha=0.85,
            ax=axis,
        )
        sns.regplot(
            data=table,
            x=descriptor,
            y=delta_col,
            scatter=False,
            color="#333333",
            ci=None,
            ax=axis,
        )
        axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel(delta_col.replace("_", " ").upper())

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for axis in axes.flat:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        f"Heterogeneidade dos ganhos locais ({delta_col.replace('_', ' ').upper()})",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"local_gain_heterogeneity_scatter_{delta_col}.png", dpi=180)
    plt.close(figure)


def plot_correlation_heatmap(correlations: pd.DataFrame, delta_col: str) -> None:
    corr = correlations[correlations["delta_metric"] == delta_col].copy()
    pivot = corr.pivot_table(
        index="descriptor",
        columns="experiment",
        values="spearman_r",
        observed=True,
    ).reindex(index=DESCRIPTOR_COLUMNS, columns=EXPERIMENT_ORDER)

    plt.figure(figsize=(8.5, 8))
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
    plt.title(f"Correlacoes pais-descritor vs {delta_col.replace('_', ' ').upper()}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"local_gain_heterogeneity_correlation_{delta_col}.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    deltas = pd.read_csv(DELTA_TABLE)
    targets = sorted(deltas["target"].unique())

    feature_matrix = load_required_feature_matrix(targets)
    profiles = build_country_profiles(feature_matrix, targets)
    heterogeneity = build_heterogeneity_table(profiles, deltas)
    correlations = correlation_rows(heterogeneity)

    profile_path = TABLES_DIR / "local_gain_country_descriptors.csv"
    heterogeneity_path = TABLES_DIR / "local_gain_heterogeneity_by_country.csv"
    correlation_path = TABLES_DIR / "local_gain_heterogeneity_correlations.csv"

    profiles.to_csv(profile_path, index=False)
    heterogeneity.to_csv(heterogeneity_path, index=False)
    correlations.to_csv(correlation_path, index=False)

    for delta_col in DELTA_COLUMNS:
        plot_scatter_grid(heterogeneity, delta_col)
        plot_correlation_heatmap(correlations, delta_col)

    strongest = (
        correlations[correlations["delta_metric"].isin(DELTA_COLUMNS)]
        .assign(abs_spearman=lambda df: df["spearman_r"].abs())
        .sort_values("abs_spearman", ascending=False)
        .head(12)
    )

    print("Local-gain heterogeneity analysis completed.")
    print()
    print("Strongest descriptor/delta associations:")
    print(
        strongest[
            [
                "experiment",
                "descriptor",
                "delta_metric",
                "n_countries",
                "spearman_r",
                "spearman_p",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Country descriptors saved to: {profile_path}")
    print(f"Heterogeneity table saved to: {heterogeneity_path}")
    print(f"Correlations saved to: {correlation_path}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
