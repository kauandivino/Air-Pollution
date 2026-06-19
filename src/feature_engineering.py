from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.target_builder import AQI_COLUMN, DATE_COLUMN, LOCATION_COLUMNS


CURRENT_POLLUTANT_COLUMNS = [
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "NO (ug/m3)",
    "NO2 (ug/m3)",
    "NOx (ppb)",
    "NH3 (ug/m3)",
    "CO (mg/m3)",
    "SO2 (ug/m3)",
    "O3 (ug/m3)",
    "Benzene (ug/m3)",
    "Toluene (ug/m3)",
    "Xylene (ug/m3)",
]

CURRENT_METEOROLOGY_COLUMNS = [
    "Wind_Speed (km/h)",
    "Humidity (%)",
]

CURRENT_SOCIO_ENVIRONMENTAL_COLUMNS = [
    "Deforestation_Rate_%",
    "Industry_Growth_%",
    "CO2_Emission_MT",
    "Population_Density_per_SqKm",
]

DEFAULT_LAG_PLAN = {
    "AQI": [1, 2, 3, 6, 12],
    "PM2.5 (ug/m3)": [1, 2, 3],
    "PM10 (ug/m3)": [1, 2, 3],
    "CO (mg/m3)": [1],
    "NO2 (ug/m3)": [1],
    "SO2 (ug/m3)": [1],
    "O3 (ug/m3)": [1],
    "Wind_Speed (km/h)": [1],
    "Humidity (%)": [1],
    "Deforestation_Rate_%": [1],
}

DEFAULT_ROLLING_PLAN = {
    "AQI": [3, 6],
    "PM2.5 (ug/m3)": [3],
    "PM10 (ug/m3)": [3],
}

DEFAULT_LOCAL_NORMALIZATION_COLUMNS = [
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


def build_supervised_feature_matrix(
    df: pd.DataFrame,
    lag_plan: dict[str, Iterable[int]] = DEFAULT_LAG_PLAN,
    rolling_plan: dict[str, Iterable[int]] = DEFAULT_ROLLING_PLAN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create calendar, lag, rolling, and interaction features.

    Rolling statistics are computed from shifted series, so a rolling feature at
    month t summarizes months before t and does not consume future observations.
    """
    result = _sort_city_time(df)
    feature_records: list[dict[str, str]] = []

    result, calendar_features = add_calendar_features(result)
    feature_records.extend(
        _feature_record(name, "calendar", "Temporal calendar encoding")
        for name in calendar_features
    )

    lag_features = add_lag_features(result, lag_plan=lag_plan)
    feature_records.extend(
        _feature_record(name, "lag", "City-level lagged observation")
        for name in lag_features
    )

    rolling_features = add_rolling_features(result, rolling_plan=rolling_plan)
    feature_records.extend(
        _feature_record(name, "rolling", "City-level rolling statistic using past months")
        for name in rolling_features
    )

    interaction_features = add_interaction_features(result)
    feature_records.extend(
        _feature_record(name, "interaction", "Domain-informed interaction")
        for name in interaction_features
    )

    current_features = available_current_features(result)
    feature_records.extend(
        _feature_record(name, _current_feature_group(name), "Current city-month observation")
        for name in current_features
    )

    feature_catalog = pd.DataFrame(feature_records).drop_duplicates("feature")
    feature_catalog = feature_catalog.sort_values(["group", "feature"]).reset_index(drop=True)

    return result, feature_catalog


def add_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _validate_required_columns(df, [DATE_COLUMN])
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="raise")

    min_date = df[DATE_COLUMN].min()
    df["year"] = df[DATE_COLUMN].dt.year
    df["month"] = df[DATE_COLUMN].dt.month
    df["quarter"] = df[DATE_COLUMN].dt.quarter
    df["time_index_months"] = (
        (df[DATE_COLUMN].dt.year - min_date.year) * 12
        + (df[DATE_COLUMN].dt.month - min_date.month)
    )
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_aug_oct"] = df["month"].isin([8, 9, 10]).astype("int8")
    df["is_dec_feb"] = df["month"].isin([12, 1, 2]).astype("int8")

    features = [
        "year",
        "month",
        "quarter",
        "time_index_months",
        "month_sin",
        "month_cos",
        "is_aug_oct",
        "is_dec_feb",
    ]
    return df, features


def add_lag_features(
    df: pd.DataFrame,
    lag_plan: dict[str, Iterable[int]] = DEFAULT_LAG_PLAN,
) -> list[str]:
    grouped = df.groupby(LOCATION_COLUMNS, observed=True)
    created_features: list[str] = []

    for column, lags in lag_plan.items():
        _validate_required_columns(df, [column])
        base_name = _safe_feature_name(column)
        for lag in lags:
            _validate_positive_window(lag, "lag")
            feature = f"{base_name}_lag_{lag}"
            df[feature] = grouped[column].shift(lag)
            created_features.append(feature)

    return created_features


def add_rolling_features(
    df: pd.DataFrame,
    rolling_plan: dict[str, Iterable[int]] = DEFAULT_ROLLING_PLAN,
) -> list[str]:
    created_features: list[str] = []

    for column, windows in rolling_plan.items():
        _validate_required_columns(df, [column])
        base_name = _safe_feature_name(column)
        shifted = df.groupby(LOCATION_COLUMNS, observed=True)[column].shift(1)

        for window in windows:
            _validate_positive_window(window, "rolling window")
            grouped_shifted = shifted.groupby([df[col] for col in LOCATION_COLUMNS], observed=True)

            mean_feature = f"{base_name}_roll_mean_{window}"
            max_feature = f"{base_name}_roll_max_{window}"
            std_feature = f"{base_name}_roll_std_{window}"

            df[mean_feature] = grouped_shifted.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            df[max_feature] = grouped_shifted.transform(
                lambda values: values.rolling(window, min_periods=1).max()
            )
            df[std_feature] = grouped_shifted.transform(
                lambda values: values.rolling(window, min_periods=2).std()
            )
            created_features.extend([mean_feature, max_feature, std_feature])

    if "AQI_lag_1" in df.columns and "AQI_lag_3" in df.columns:
        df["AQI_trend_3"] = df[AQI_COLUMN] - df["AQI_lag_3"]
        df["AQI_delta_1"] = df[AQI_COLUMN] - df["AQI_lag_1"]
        created_features.extend(["AQI_trend_3", "AQI_delta_1"])

    return created_features


def add_interaction_features(df: pd.DataFrame) -> list[str]:
    required = [
        "PM2.5 (ug/m3)",
        "PM10 (ug/m3)",
        "Wind_Speed (km/h)",
        "Humidity (%)",
        "Deforestation_Rate_%",
        "Industry_Growth_%",
        "month",
    ]
    _validate_required_columns(df, required)

    df["PM25_PM10_ratio"] = df["PM2.5 (ug/m3)"] / df["PM10 (ug/m3)"].replace(0, np.nan)
    df["low_wind_PM25"] = df["PM2.5 (ug/m3)"] / (df["Wind_Speed (km/h)"] + 1.0)
    df["humidity_PM25"] = df["Humidity (%)"] * df["PM2.5 (ug/m3)"]
    df["deforestation_aug_oct"] = df["Deforestation_Rate_%"] * df["is_aug_oct"]
    df["industry_NO2"] = df["Industry_Growth_%"] * df["NO2 (ug/m3)"]

    return [
        "PM25_PM10_ratio",
        "low_wind_PM25",
        "humidity_PM25",
        "deforestation_aug_oct",
        "industry_NO2",
    ]


def available_current_features(df: pd.DataFrame) -> list[str]:
    candidates = (
        [AQI_COLUMN]
        + CURRENT_POLLUTANT_COLUMNS
        + CURRENT_METEOROLOGY_COLUMNS
        + CURRENT_SOCIO_ENVIRONMENTAL_COLUMNS
    )
    return [column for column in candidates if column in df.columns]


def feature_columns_from_catalog(feature_catalog: pd.DataFrame) -> list[str]:
    return feature_catalog["feature"].drop_duplicates().tolist()


def summarize_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for feature in feature_columns:
        missing = int(df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "missing_values": missing,
                "missing_rate": missing / total_rows if total_rows else 0.0,
                "non_missing_values": total_rows - missing,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "feature"], ascending=[False, True])


def add_past_city_normalized_features(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_LOCAL_NORMALIZATION_COLUMNS,
    min_history: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add local anomaly and z-score features using only past city observations."""
    result = _sort_city_time(df)
    created: list[dict[str, str]] = []
    grouped = result.groupby(LOCATION_COLUMNS, observed=True)

    for column in columns:
        _validate_required_columns(result, [column])
        base_name = _safe_feature_name(column)
        shifted = grouped[column].shift(1)
        shifted_grouped = shifted.groupby([result[col] for col in LOCATION_COLUMNS], observed=True)

        past_count = shifted_grouped.cumcount()
        past_mean = shifted_grouped.transform(
            lambda values: values.expanding(min_periods=min_history).mean()
        )
        past_std = shifted_grouped.transform(
            lambda values: values.expanding(min_periods=min_history).std()
        )

        anomaly_feature = f"{base_name}_city_past_anomaly"
        zscore_feature = f"{base_name}_city_past_zscore"
        history_feature = f"{base_name}_city_history_count"

        result[anomaly_feature] = result[column] - past_mean
        result[zscore_feature] = result[anomaly_feature] / past_std.replace(0, np.nan)
        result[history_feature] = past_count

        created.extend(
            [
                _feature_record(
                    anomaly_feature,
                    "local_normalization",
                    "Current value minus past city expanding mean",
                ),
                _feature_record(
                    zscore_feature,
                    "local_normalization",
                    "Current value z-score using past city expanding statistics",
                ),
                _feature_record(
                    history_feature,
                    "local_normalization",
                    "Number of previous city observations available",
                ),
            ]
        )

    catalog = pd.DataFrame(created).drop_duplicates("feature")
    return result, catalog


def _sort_city_time(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, LOCATION_COLUMNS + [DATE_COLUMN, AQI_COLUMN])
    return df.sort_values(LOCATION_COLUMNS + [DATE_COLUMN]).reset_index(drop=True).copy()


def _safe_feature_name(column: str) -> str:
    replacements = {
        "PM2.5 (ug/m3)": "PM25",
        "PM10 (ug/m3)": "PM10",
        "NO (ug/m3)": "NO",
        "NO2 (ug/m3)": "NO2",
        "NOx (ppb)": "NOx",
        "NH3 (ug/m3)": "NH3",
        "CO (mg/m3)": "CO",
        "SO2 (ug/m3)": "SO2",
        "O3 (ug/m3)": "O3",
        "Wind_Speed (km/h)": "Wind_Speed",
        "Humidity (%)": "Humidity",
        "Deforestation_Rate_%": "Deforestation_Rate",
    }
    return replacements.get(column, column.replace(" ", "_").replace(".", "").replace("%", "pct"))


def _current_feature_group(feature: str) -> str:
    if feature == AQI_COLUMN:
        return "current_aqi"
    if feature in CURRENT_POLLUTANT_COLUMNS:
        return "current_pollutant"
    if feature in CURRENT_METEOROLOGY_COLUMNS:
        return "current_meteorology"
    if feature in CURRENT_SOCIO_ENVIRONMENTAL_COLUMNS:
        return "current_socio_environmental"
    return "current"


def _feature_record(feature: str, group: str, description: str) -> dict[str, str]:
    return {"feature": feature, "group": group, "description": description}


def _validate_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _validate_positive_window(value: int, label: str) -> None:
    if value < 1:
        raise ValueError(f"{label} must be positive. Received: {value}")
