from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROCESSED_REAL_DIR = PROJECT_ROOT / "data" / "real" / "processed"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"

DATASET_NAMES = ["waqd2024", "openaq"]
DEFAULT_MIN_HISTORY_MONTHS = 6

ID_COLUMNS = ["Country_Code", "Country", "Date"]
TARGET_SPECS = [
    ("PM25_value_p95", "PM25_p95_rel_p90", 0.90),
    ("PM10_value_p95", "PM10_p95_rel_p90", 0.90),
]
HORIZONS = [1, 3]
LAG_COLUMNS = ["PM25_value_p95", "PM10_value_p95", "O3_value_p95", "NO2_value_p95"]
LAGS = [1, 2, 3]
ROLLING_COLUMNS = ["PM25_value_p95", "PM10_value_p95"]
ROLLING_WINDOWS = [3]
LOCAL_NORMALIZATION_COLUMNS = ["PM25_value_p95", "PM10_value_p95", "O3_value_p95", "NO2_value_p95"]


def read_country_month(dataset_name: str) -> pd.DataFrame:
    path = PROCESSED_REAL_DIR / f"{dataset_name}_country_month.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/36_build_openaq_country_month.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Country", "Date"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    result = df.copy()
    min_date = result["Date"].min()
    result["year"] = result["Date"].dt.year
    result["month"] = result["Date"].dt.month
    result["quarter"] = result["Date"].dt.quarter
    result["time_index_months"] = (
        (result["Date"].dt.year - min_date.year) * 12
        + (result["Date"].dt.month - min_date.month)
    )
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    records = [
        feature_record(feature, "calendar", "Calendar encoding for country-month rows")
        for feature in [
            "year",
            "month",
            "quarter",
            "time_index_months",
            "month_sin",
            "month_cos",
        ]
    ]
    return result, records


def add_exact_future_values(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    lookup = result[["Country", "Date"] + value_columns].copy()

    for horizon in HORIZONS:
        future_lookup = lookup.copy()
        future_lookup["Date"] = future_lookup["Date"] - pd.DateOffset(months=horizon)
        rename_map = {column: f"future_{column}_h{horizon}" for column in value_columns}
        future_lookup = future_lookup.rename(columns=rename_map)
        result = result.merge(future_lookup, on=["Country", "Date"], how="left")

    return result


def add_historical_thresholds_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["Country", "Date"]).copy()

    for value_column, target_prefix, percentile in TARGET_SPECS:
        if value_column not in result.columns:
            continue

        threshold_column = f"{target_prefix}_threshold"
        history_count_column = f"{target_prefix}_history_count"
        result[threshold_column] = np.nan
        result[history_count_column] = 0

        for _, country_index in result.groupby("Country", observed=True).groups.items():
            values = result.loc[country_index, value_column].astype(float)
            thresholds = []
            counts = []
            for position in range(len(values)):
                history = values.iloc[: position + 1].dropna()
                counts.append(int(history.shape[0]))
                thresholds.append(
                    float(history.quantile(percentile))
                    if history.shape[0] >= DEFAULT_MIN_HISTORY_MONTHS
                    else np.nan
                )
            result.loc[country_index, threshold_column] = thresholds
            result.loc[country_index, history_count_column] = counts

        for horizon in HORIZONS:
            future_column = f"future_{value_column}_h{horizon}"
            target_column = f"{target_prefix}_h{horizon}"
            valid = result[future_column].notna() & result[threshold_column].notna()
            result[target_column] = pd.NA
            result.loc[valid, target_column] = (
                result.loc[valid, future_column] > result.loc[valid, threshold_column]
            ).astype("int8")

    return result


def add_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    result = df.copy()
    records: list[dict[str, str]] = []
    base = result[["Country", "Date"] + [c for c in LAG_COLUMNS if c in result.columns]].copy()

    for lag in LAGS:
        lagged = base.copy()
        lagged["Date"] = lagged["Date"] + pd.DateOffset(months=lag)
        rename_map = {
            column: f"{safe_name(column)}_lag_{lag}"
            for column in base.columns
            if column not in ["Country", "Date"]
        }
        lagged = lagged.rename(columns=rename_map)
        result = result.merge(lagged, on=["Country", "Date"], how="left")
        records.extend(
            feature_record(feature, "lag", f"Exact {lag}-month country-level lag")
            for feature in rename_map.values()
        )

    return result, records


def add_rolling_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    result = df.sort_values(["Country", "Date"]).copy()
    records: list[dict[str, str]] = []

    for column in ROLLING_COLUMNS:
        if column not in result.columns:
            continue
        base_name = safe_name(column)
        grouped = result.groupby("Country", observed=True)[column]
        shifted = grouped.shift(1)

        for window in ROLLING_WINDOWS:
            roll = shifted.groupby(result["Country"], observed=True)
            mean_feature = f"{base_name}_roll_mean_{window}"
            max_feature = f"{base_name}_roll_max_{window}"
            std_feature = f"{base_name}_roll_std_{window}"
            result[mean_feature] = roll.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            result[max_feature] = roll.transform(
                lambda values: values.rolling(window, min_periods=1).max()
            )
            result[std_feature] = roll.transform(
                lambda values: values.rolling(window, min_periods=2).std()
            )
            records.extend(
                [
                    feature_record(mean_feature, "rolling", "Past rolling country mean"),
                    feature_record(max_feature, "rolling", "Past rolling country maximum"),
                    feature_record(std_feature, "rolling", "Past rolling country standard deviation"),
                ]
            )

    return result, records


def add_local_normalized_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    result = df.sort_values(["Country", "Date"]).copy()
    records: list[dict[str, str]] = []

    for column in LOCAL_NORMALIZATION_COLUMNS:
        if column not in result.columns:
            continue

        base_name = safe_name(column)
        grouped = result.groupby("Country", observed=True)[column]
        past_mean = grouped.transform(lambda values: values.shift(1).expanding(min_periods=3).mean())
        past_std = grouped.transform(lambda values: values.shift(1).expanding(min_periods=3).std())
        past_count = grouped.transform(lambda values: values.shift(1).expanding(min_periods=1).count())

        anomaly_feature = f"{base_name}_country_anomaly"
        zscore_feature = f"{base_name}_country_zscore"
        count_feature = f"{base_name}_country_history_count"

        result[anomaly_feature] = result[column] - past_mean
        result[zscore_feature] = (result[column] - past_mean) / past_std.replace(0, np.nan)
        result[count_feature] = past_count

        records.extend(
            [
                feature_record(anomaly_feature, "local_normalized", "Deviation from past country mean"),
                feature_record(zscore_feature, "local_normalized", "Z-score against past country history"),
                feature_record(count_feature, "local_normalized", "Past observed country months"),
            ]
        )

    return result, records


def build_feature_catalog(df: pd.DataFrame, records: list[dict[str, str]]) -> pd.DataFrame:
    identifier_and_target_prefixes = (
        "future_",
        "PM25_p95_rel_p90",
        "PM10_p95_rel_p90",
    )
    excluded = set(ID_COLUMNS)
    excluded.update(
        column
        for column in df.columns
        if column.startswith(identifier_and_target_prefixes)
        or column.endswith("_threshold")
        or column.endswith("_history_count") and column.startswith(("PM25_p95_rel", "PM10_p95_rel"))
    )

    existing_records = {record["feature"]: record for record in records if record["feature"] in df.columns}
    for column in df.columns:
        if column in excluded:
            continue
        if column not in existing_records and pd.api.types.is_numeric_dtype(df[column]):
            existing_records[column] = feature_record(column, infer_group(column), "Current real-data covariate")

    catalog = pd.DataFrame(existing_records.values()).drop_duplicates("feature")
    return catalog.sort_values(["group", "feature"]).reset_index(drop=True)


def summarize_targets(df: pd.DataFrame) -> pd.DataFrame:
    target_columns = [
        column
        for column in df.columns
        if column.endswith("_h1") or column.endswith("_h3")
        if column.startswith(("PM25_p95_rel_p90", "PM10_p95_rel_p90"))
    ]
    rows = []
    for column in target_columns:
        valid = df[column].dropna().astype(int)
        rows.append(
            {
                "target": column,
                "valid_rows": int(valid.shape[0]),
                "events": int(valid.sum()) if not valid.empty else 0,
                "non_events": int(valid.shape[0] - valid.sum()) if not valid.empty else 0,
                "prevalence": float(valid.mean()) if not valid.empty else 0.0,
                "countries": int(df.loc[df[column].notna(), "Country"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def feature_missing_summary(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in catalog["feature"]:
        missing = int(df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "group": catalog.loc[catalog["feature"] == feature, "group"].iloc[0],
                "missing_rows": missing,
                "missing_rate": missing / len(df) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "feature"], ascending=[False, True])


def safe_name(column: str) -> str:
    return (
        column.replace(".", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def infer_group(column: str) -> str:
    if column in ["total_records", "total_locations", "total_cities", "total_sources", "observed_pollutants"]:
        return "coverage"
    if "_value_" in column:
        return "pollutant"
    if column.startswith(("year", "month", "quarter", "time_index")):
        return "calendar"
    return "other"


def feature_record(feature: str, group: str, description: str) -> dict[str, str]:
    return {"feature": feature, "group": group, "description": description}


def build_real_features(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_country_month(dataset_name)
    value_columns = [column for column, _, _ in TARGET_SPECS if column in df.columns]

    feature_records: list[dict[str, str]] = []
    df, records = add_calendar_features(df)
    feature_records.extend(records)

    df = add_exact_future_values(df, value_columns)
    df = add_historical_thresholds_and_targets(df)

    df, records = add_lag_features(df)
    feature_records.extend(records)

    df, records = add_rolling_features(df)
    feature_records.extend(records)

    df, records = add_local_normalized_features(df)
    feature_records.extend(records)

    catalog = build_feature_catalog(df, feature_records)
    target_summary = summarize_targets(df)
    missing_summary = feature_missing_summary(df, catalog)
    return df, catalog, target_summary, missing_summary


def main() -> None:
    PROCESSED_REAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    target_frames = []
    missing_frames = []
    catalog_frames = []

    for dataset_name in DATASET_NAMES:
        matrix, catalog, target_summary, missing_summary = build_real_features(dataset_name)

        matrix.to_csv(PROCESSED_REAL_DIR / f"{dataset_name}_real_feature_matrix.csv", index=False)
        catalog.to_csv(TABLES_DIR / f"{dataset_name}_real_feature_catalog.csv", index=False)

        target_summary.insert(0, "dataset", dataset_name)
        missing_summary.insert(0, "dataset", dataset_name)
        catalog_with_dataset = catalog.copy()
        catalog_with_dataset.insert(0, "dataset", dataset_name)

        target_frames.append(target_summary)
        missing_frames.append(missing_summary)
        catalog_frames.append(catalog_with_dataset)

        print(f"{dataset_name}: feature matrix rows={len(matrix)}, features={len(catalog)}")
        print(target_summary.to_string(index=False))
        print()

    pd.concat(target_frames, ignore_index=True).to_csv(
        TABLES_DIR / "real_target_prevalence_summary.csv",
        index=False,
    )
    pd.concat(missing_frames, ignore_index=True).to_csv(
        TABLES_DIR / "real_feature_missing_summary.csv",
        index=False,
    )
    pd.concat(catalog_frames, ignore_index=True).to_csv(
        TABLES_DIR / "real_feature_catalog.csv",
        index=False,
    )

    print("Real target and feature construction completed.")
    print(f"Processed matrices saved to: {PROCESSED_REAL_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
