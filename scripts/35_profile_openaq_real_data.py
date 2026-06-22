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

RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"

REQUIRED_COLUMNS = [
    "Country Code",
    "City",
    "Location",
    "Coordinates",
    "Pollutant",
    "Source Name",
    "Unit",
    "Value",
    "Last Updated",
    "Country Label",
]


def read_real_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce", utc=True)
    df["month"] = df["Last Updated"].dt.to_period("M").dt.to_timestamp()
    return df


def dataset_summary(name: str, path: Path, df: pd.DataFrame) -> dict[str, object]:
    valid_dates = df["Last Updated"].dropna()
    return {
        "dataset": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "rows": len(df),
        "columns": df.shape[1],
        "country_codes": df["Country Code"].nunique(dropna=True),
        "country_labels": df["Country Label"].nunique(dropna=True),
        "locations": df[["Country Label", "Location"]].drop_duplicates().shape[0],
        "cities": df[["Country Label", "City"]].dropna().drop_duplicates().shape[0],
        "pollutants": df["Pollutant"].nunique(dropna=True),
        "sources": df["Source Name"].nunique(dropna=True),
        "units": df["Unit"].nunique(dropna=True),
        "missing_city_rows": int(df["City"].isna().sum() + (df["City"] == "").sum()),
        "missing_coordinates_rows": int(
            df["Coordinates"].isna().sum() + (df["Coordinates"] == "").sum()
        ),
        "missing_country_label_rows": int(
            df["Country Label"].isna().sum() + (df["Country Label"] == "").sum()
        ),
        "valid_dates": int(valid_dates.shape[0]),
        "min_datetime_utc": valid_dates.min().isoformat() if not valid_dates.empty else "",
        "max_datetime_utc": valid_dates.max().isoformat() if not valid_dates.empty else "",
        "distinct_months": df["month"].nunique(dropna=True),
        "negative_value_rows": int((df["Value"] < 0).sum()),
        "non_negative_value_rows": int((df["Value"] >= 0).sum()),
        "missing_value_rows": int(df["Value"].isna().sum()),
    }


def country_coverage(name: str, df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby(["Country Code", "Country Label"], dropna=False)
        .agg(
            rows=("Value", "size"),
            non_negative_rows=("Value", lambda values: int((values >= 0).sum())),
            negative_rows=("Value", lambda values: int((values < 0).sum())),
            locations=("Location", "nunique"),
            cities=("City", "nunique"),
            pollutants=("Pollutant", "nunique"),
            sources=("Source Name", "nunique"),
            distinct_months=("month", "nunique"),
            min_month=("month", "min"),
            max_month=("month", "max"),
        )
        .reset_index()
    )
    coverage.insert(0, "dataset", name)
    return coverage.sort_values(["rows", "locations"], ascending=False)


def pollutant_coverage(name: str, df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby(["Pollutant", "Unit"], dropna=False)
        .agg(
            rows=("Value", "size"),
            non_negative_rows=("Value", lambda values: int((values >= 0).sum())),
            negative_rows=("Value", lambda values: int((values < 0).sum())),
            countries=("Country Label", "nunique"),
            locations=("Location", "nunique"),
            sources=("Source Name", "nunique"),
            distinct_months=("month", "nunique"),
            min_value=("Value", "min"),
            mean_value=("Value", "mean"),
            max_value=("Value", "max"),
        )
        .reset_index()
    )
    coverage.insert(0, "dataset", name)
    return coverage.sort_values("rows", ascending=False)


def country_pollutant_months(name: str, df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["Value"].ge(0) & df["month"].notna()].copy()
    coverage = (
        clean.groupby(["Country Code", "Country Label", "Pollutant"], dropna=False)
        .agg(
            records=("Value", "size"),
            distinct_months=("month", "nunique"),
            locations=("Location", "nunique"),
            sources=("Source Name", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .reset_index()
    )
    coverage.insert(0, "dataset", name)
    return coverage.sort_values(["distinct_months", "records"], ascending=False)


def location_series_depth(name: str, df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["month"].notna()].copy()
    depth = (
        clean.groupby(["Country Code", "Country Label", "Location", "Pollutant"], dropna=False)
        .agg(
            records=("Value", "size"),
            distinct_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .reset_index()
    )
    depth.insert(0, "dataset", name)
    return depth.sort_values(["distinct_months", "records"], ascending=False)


def value_quality_by_pollutant(name: str, df: pd.DataFrame) -> pd.DataFrame:
    quality = (
        df.groupby("Pollutant", dropna=False)
        .agg(
            rows=("Value", "size"),
            numeric_rows=("Value", lambda values: int(values.notna().sum())),
            negative_rows=("Value", lambda values: int((values < 0).sum())),
            non_negative_rows=("Value", lambda values: int((values >= 0).sum())),
            min_value=("Value", "min"),
            mean_value=("Value", "mean"),
            median_value=("Value", "median"),
            max_value=("Value", "max"),
        )
        .reset_index()
    )
    quality.insert(0, "dataset", name)
    return quality.sort_values("rows", ascending=False)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    country_frames = []
    pollutant_frames = []
    country_pollutant_frames = []
    location_depth_frames = []
    value_quality_frames = []

    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        df = read_real_dataset(path)
        summaries.append(dataset_summary(name, path, df))
        country_frames.append(country_coverage(name, df))
        pollutant_frames.append(pollutant_coverage(name, df))
        country_pollutant_frames.append(country_pollutant_months(name, df))
        location_depth_frames.append(location_series_depth(name, df))
        value_quality_frames.append(value_quality_by_pollutant(name, df))

    summary = pd.DataFrame(summaries)
    country = pd.concat(country_frames, ignore_index=True)
    pollutant = pd.concat(pollutant_frames, ignore_index=True)
    country_pollutant = pd.concat(country_pollutant_frames, ignore_index=True)
    location_depth = pd.concat(location_depth_frames, ignore_index=True)
    value_quality = pd.concat(value_quality_frames, ignore_index=True)

    summary.to_csv(TABLES_DIR / "real_dataset_summary.csv", index=False)
    country.to_csv(TABLES_DIR / "real_country_coverage.csv", index=False)
    pollutant.to_csv(TABLES_DIR / "real_pollutant_coverage.csv", index=False)
    country_pollutant.to_csv(TABLES_DIR / "real_country_pollutant_months.csv", index=False)
    location_depth.to_csv(TABLES_DIR / "real_location_series_depth.csv", index=False)
    value_quality.to_csv(TABLES_DIR / "real_value_quality_by_pollutant.csv", index=False)

    print("Real-data profiling completed.")
    print(summary.to_string(index=False))
    print()
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
