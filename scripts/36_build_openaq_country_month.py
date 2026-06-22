from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASETS = {
    "openaq": PROJECT_ROOT / "data" / "openaq" / "openaq.csv",
    "waqd2024": PROJECT_ROOT / "data" / "world_air_quality" / "world_air_quality.csv",
}

PROCESSED_REAL_DIR = PROJECT_ROOT / "data" / "real" / "processed"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"

PRIORITY_POLLUTANTS = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
POLLUTANT_NAMES = {
    "PM2.5": "PM25",
    "PM10": "PM10",
    "O3": "O3",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO": "CO",
}


def read_real_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce", utc=True)
    df["Date"] = df["Last Updated"].dt.to_period("M").dt.to_timestamp()
    return df


def clean_real_measurements(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = (
        df.groupby("Pollutant", dropna=False)
        .agg(
            raw_rows=("Value", "size"),
            numeric_rows=("Value", lambda values: int(values.notna().sum())),
            negative_rows=("Value", lambda values: int((values < 0).sum())),
            non_negative_rows=("Value", lambda values: int((values >= 0).sum())),
            missing_date_rows=("Date", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )

    clean = df[
        df["Pollutant"].isin(PRIORITY_POLLUTANTS)
        & df["Value"].notna()
        & df["Value"].ge(0)
        & df["Date"].notna()
        & df["Country Label"].notna()
        & (df["Country Label"].astype(str).str.len() > 0)
    ].copy()

    return clean, diagnostics


def aggregate_country_pollutant_month(clean: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean.groupby(["Country Code", "Country Label", "Date", "Pollutant"], dropna=False)
        .agg(
            value_count=("Value", "size"),
            value_mean=("Value", "mean"),
            value_median=("Value", "median"),
            value_p90=("Value", lambda values: values.quantile(0.90)),
            value_p95=("Value", lambda values: values.quantile(0.95)),
            value_max=("Value", "max"),
            value_std=("Value", "std"),
            locations_count=("Location", "nunique"),
            cities_count=("City", "nunique"),
            sources_count=("Source Name", "nunique"),
            units_count=("Unit", "nunique"),
        )
        .reset_index()
    )
    return grouped


def pivot_country_month(aggregated: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["Country Code", "Country Label", "Date"]
    metric_columns = [
        "value_count",
        "value_mean",
        "value_median",
        "value_p90",
        "value_p95",
        "value_max",
        "value_std",
        "locations_count",
        "cities_count",
        "sources_count",
        "units_count",
    ]

    pivot = aggregated.pivot_table(
        index=id_columns,
        columns="Pollutant",
        values=metric_columns,
        aggfunc="first",
    )
    pivot.columns = [
        f"{POLLUTANT_NAMES.get(pollutant, pollutant)}_{metric}"
        for metric, pollutant in pivot.columns
    ]
    pivot = pivot.reset_index()

    coverage = (
        aggregated.groupby(id_columns, dropna=False)
        .agg(
            total_records=("value_count", "sum"),
            observed_pollutants=("Pollutant", "nunique"),
            total_locations=("locations_count", "sum"),
            total_cities=("cities_count", "sum"),
            total_sources=("sources_count", "sum"),
        )
        .reset_index()
    )
    result = pivot.merge(coverage, on=id_columns, how="left")
    result = result.rename(
        columns={
            "Country Code": "Country_Code",
            "Country Label": "Country",
        }
    )
    result = result.sort_values(["Country", "Date"]).reset_index(drop=True)
    return result


def summarize_country_month(name: str, country_month: pd.DataFrame) -> pd.DataFrame:
    pm25_column = "PM25_value_p95"
    pm10_column = "PM10_value_p95"
    summary = (
        country_month.groupby("Country", dropna=False)
        .agg(
            country_months=("Date", "nunique"),
            min_month=("Date", "min"),
            max_month=("Date", "max"),
            total_records=("total_records", "sum"),
            mean_records_per_month=("total_records", "mean"),
            mean_observed_pollutants=("observed_pollutants", "mean"),
            months_with_pm25=(pm25_column, lambda values: int(values.notna().sum()))
            if pm25_column in country_month.columns
            else ("Date", lambda values: 0),
            months_with_pm10=(pm10_column, lambda values: int(values.notna().sum()))
            if pm10_column in country_month.columns
            else ("Date", lambda values: 0),
        )
        .reset_index()
    )
    summary.insert(0, "dataset", name)
    return summary.sort_values(["months_with_pm25", "country_months"], ascending=False)


def main() -> None:
    PROCESSED_REAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    diagnostic_frames = []

    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        raw = read_real_dataset(path)
        clean, diagnostics = clean_real_measurements(raw)
        aggregated = aggregate_country_pollutant_month(clean)
        country_month = pivot_country_month(aggregated)

        aggregated.to_csv(
            PROCESSED_REAL_DIR / f"{name}_country_pollutant_month.csv",
            index=False,
        )
        country_month.to_csv(PROCESSED_REAL_DIR / f"{name}_country_month.csv", index=False)

        diagnostics.insert(0, "dataset", name)
        diagnostic_frames.append(diagnostics)
        summary_frames.append(summarize_country_month(name, country_month))

        print(f"{name}: raw rows={len(raw)}, clean rows={len(clean)}")
        print(f"{name}: country-month rows={len(country_month)}")

    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)

    diagnostics.to_csv(TABLES_DIR / "real_cleaning_diagnostics_by_pollutant.csv", index=False)
    summaries.to_csv(TABLES_DIR / "real_country_month_coverage.csv", index=False)

    print()
    print("Country-month datasets built.")
    print(f"Processed data saved to: {PROCESSED_REAL_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
