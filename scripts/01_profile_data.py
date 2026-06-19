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

from src.config import FIGURES_DIR, NUMERIC_COLUMNS, TABLES_DIR, ensure_results_dirs
from src.data_loading import load_global_air_quality_data


AQI_COLUMN = "AQI"
COUNTRY_COLUMN = "Country"
STATE_COLUMN = "State"
CITY_COLUMN = "City"
DATE_COLUMN = "Date"
BUCKET_COLUMN = "AQI_Bucket"


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_location_dates = df.duplicated(
        subset=[COUNTRY_COLUMN, STATE_COLUMN, CITY_COLUMN, DATE_COLUMN]
    ).sum()

    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "countries": df[COUNTRY_COLUMN].nunique(),
        "states": df[[COUNTRY_COLUMN, STATE_COLUMN]].drop_duplicates().shape[0],
        "cities": df[[COUNTRY_COLUMN, STATE_COLUMN, CITY_COLUMN]].drop_duplicates().shape[0],
        "dates": df[DATE_COLUMN].nunique(),
        "min_date": df[DATE_COLUMN].min().date().isoformat(),
        "max_date": df[DATE_COLUMN].max().date().isoformat(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_city_months": int(duplicated_location_dates),
    }
    return pd.DataFrame([summary])


def build_country_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby(COUNTRY_COLUMN)
        .agg(
            rows=(AQI_COLUMN, "size"),
            states=(STATE_COLUMN, "nunique"),
            cities=(CITY_COLUMN, "nunique"),
            mean_aqi=(AQI_COLUMN, "mean"),
            median_aqi=(AQI_COLUMN, "median"),
            max_aqi=(AQI_COLUMN, "max"),
            extreme_abs_151=(AQI_COLUMN, lambda values: (values >= 151).sum()),
            extreme_abs_201=(AQI_COLUMN, lambda values: (values >= 201).sum()),
        )
        .reset_index()
    )
    profile["extreme_abs_151_rate"] = profile["extreme_abs_151"] / profile["rows"]
    profile["extreme_abs_201_rate"] = profile["extreme_abs_201"] / profile["rows"]
    return profile.sort_values(["extreme_abs_151_rate", "mean_aqi"], ascending=False)


def build_bucket_distribution(df: pd.DataFrame) -> pd.DataFrame:
    bucket_counts = df[BUCKET_COLUMN].value_counts(dropna=False).rename_axis(BUCKET_COLUMN)
    result = bucket_counts.reset_index(name="rows")
    result["rate"] = result["rows"] / len(df)
    return result


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[NUMERIC_COLUMNS].describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "variable"})
    return summary


def build_monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.assign(month=df[DATE_COLUMN].dt.month)
    profile = (
        monthly.groupby("month")
        .agg(
            rows=(AQI_COLUMN, "size"),
            mean_aqi=(AQI_COLUMN, "mean"),
            extreme_abs_151_rate=(AQI_COLUMN, lambda values: (values >= 151).mean()),
            extreme_abs_201_rate=(AQI_COLUMN, lambda values: (values >= 201).mean()),
        )
        .reset_index()
    )
    return profile


def save_figures(
    df: pd.DataFrame,
    country_profile: pd.DataFrame,
    bucket_distribution: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[AQI_COLUMN], bins=60, kde=True, color="#28666e")
    plt.title("Distribuicao global do AQI")
    plt.xlabel("AQI")
    plt.ylabel("Registros")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribuicao_aqi.png", dpi=160)
    plt.close()

    top_extreme = country_profile.head(15).copy()
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=top_extreme,
        y=COUNTRY_COLUMN,
        x="extreme_abs_151_rate",
        color="#7c2d12",
    )
    plt.title("Top 15 paises por taxa de eventos AQI >= 151")
    plt.xlabel("Taxa de eventos")
    plt.ylabel("Pais")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eventos_extremos_151_por_pais.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7))
    bucket_plot = bucket_distribution.sort_values("rows", ascending=True)
    sns.barplot(data=bucket_plot, y=BUCKET_COLUMN, x="rows", color="#3d5a80")
    plt.title("Distribuicao de AQI_Bucket")
    plt.xlabel("Registros")
    plt.ylabel("Categoria")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribuicao_aqi_bucket.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    df = load_global_air_quality_data()

    dataset_summary = build_dataset_summary(df)
    country_profile = build_country_profile(df)
    bucket_distribution = build_bucket_distribution(df)
    numeric_summary = build_numeric_summary(df)
    monthly_profile = build_monthly_profile(df)

    dataset_summary.to_csv(TABLES_DIR / "data_profile_summary.csv", index=False)
    country_profile.to_csv(TABLES_DIR / "country_profile.csv", index=False)
    bucket_distribution.to_csv(TABLES_DIR / "aqi_bucket_distribution.csv", index=False)
    numeric_summary.to_csv(TABLES_DIR / "numeric_summary.csv", index=False)
    monthly_profile.to_csv(TABLES_DIR / "monthly_profile.csv", index=False)

    save_figures(df, country_profile, bucket_distribution)

    print("Dataset profile completed.")
    print(dataset_summary.to_string(index=False))
    print()
    print("Top countries by AQI >= 151 event rate:")
    print(
        country_profile[
            [
                COUNTRY_COLUMN,
                "rows",
                "cities",
                "mean_aqi",
                "extreme_abs_151_rate",
                "extreme_abs_201_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
