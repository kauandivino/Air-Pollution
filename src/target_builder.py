from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


LOCATION_COLUMNS = ["Country", "State", "City"]
DATE_COLUMN = "Date"
AQI_COLUMN = "AQI"

DEFAULT_HORIZONS = [1, 3]
DEFAULT_ABSOLUTE_THRESHOLDS = [151, 201]
DEFAULT_CITY_PERCENTILES = [0.90, 0.95]


def add_future_aqi_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Add future AQI columns for each horizon using city-level time ordering."""
    result = _sort_city_time(df)

    for horizon in horizons:
        _validate_positive_horizon(horizon)
        result[f"future_aqi_h{horizon}"] = result.groupby(LOCATION_COLUMNS, observed=True)[
            AQI_COLUMN
        ].shift(-horizon)

    return result


def compute_city_percentile_thresholds(
    reference_df: pd.DataFrame,
    percentiles: Iterable[float] = DEFAULT_CITY_PERCENTILES,
) -> pd.DataFrame:
    """Compute city-level AQI percentile thresholds from a reference dataframe.

    In exploratory analysis the reference dataframe may be the full dataset. During
    model evaluation it must be restricted to the training split to avoid leakage.
    """
    _validate_required_columns(reference_df, LOCATION_COLUMNS + [AQI_COLUMN])
    thresholds = reference_df[LOCATION_COLUMNS].drop_duplicates().reset_index(drop=True)

    grouped = reference_df.groupby(LOCATION_COLUMNS, observed=True)[AQI_COLUMN]
    for percentile in percentiles:
        _validate_percentile(percentile)
        column = _percentile_threshold_column(percentile)
        values = grouped.quantile(percentile).reset_index(name=column)
        thresholds = thresholds.merge(values, on=LOCATION_COLUMNS, how="left")

    return thresholds


def add_extreme_event_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    absolute_thresholds: Iterable[int] = DEFAULT_ABSOLUTE_THRESHOLDS,
    city_percentiles: Iterable[float] = DEFAULT_CITY_PERCENTILES,
    percentile_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create absolute and city-relative future extreme-event targets."""
    result = add_future_aqi_targets(df, horizons=horizons)

    if percentile_reference_df is None:
        percentile_reference_df = df

    thresholds = compute_city_percentile_thresholds(
        percentile_reference_df,
        percentiles=city_percentiles,
    )
    result = result.merge(thresholds, on=LOCATION_COLUMNS, how="left")

    for horizon in horizons:
        future_column = f"future_aqi_h{horizon}"
        valid_future = result[future_column].notna()

        for threshold in absolute_thresholds:
            target_column = f"extreme_abs_{threshold}_h{horizon}"
            result[target_column] = pd.NA
            result.loc[valid_future, target_column] = (
                result.loc[valid_future, future_column] >= threshold
            ).astype("int8")

        for percentile in city_percentiles:
            threshold_column = _percentile_threshold_column(percentile)
            target_column = f"extreme_city_p{int(percentile * 100):02d}_h{horizon}"
            valid_relative = valid_future & result[threshold_column].notna()
            result[target_column] = pd.NA
            result.loc[valid_relative, target_column] = (
                result.loc[valid_relative, future_column]
                >= result.loc[valid_relative, threshold_column]
            ).astype("int8")

    return result


def list_target_columns(
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    absolute_thresholds: Iterable[int] = DEFAULT_ABSOLUTE_THRESHOLDS,
    city_percentiles: Iterable[float] = DEFAULT_CITY_PERCENTILES,
) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        for threshold in absolute_thresholds:
            columns.append(f"extreme_abs_{threshold}_h{horizon}")
        for percentile in city_percentiles:
            columns.append(f"extreme_city_p{int(percentile * 100):02d}_h{horizon}")
    return columns


def summarize_target_prevalence(
    df: pd.DataFrame,
    target_columns: Iterable[str],
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize valid rows, event counts, and prevalence for target columns."""
    rows = []

    if groupby is None:
        for target_column in target_columns:
            valid = df[target_column].dropna()
            rows.append(_target_summary_row(target_column, valid))
        return pd.DataFrame(rows)

    for group_values, group_df in df.groupby(groupby, observed=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_info = dict(zip(groupby, group_values, strict=True))

        for target_column in target_columns:
            valid = group_df[target_column].dropna()
            rows.append({**group_info, **_target_summary_row(target_column, valid)})

    return pd.DataFrame(rows)


def _target_summary_row(target_column: str, valid: pd.Series) -> dict[str, object]:
    valid_rows = int(valid.shape[0])
    events = int(valid.astype("int8").sum()) if valid_rows else 0
    prevalence = events / valid_rows if valid_rows else 0.0

    return {
        "target": target_column,
        "valid_rows": valid_rows,
        "events": events,
        "non_events": valid_rows - events,
        "prevalence": prevalence,
    }


def _sort_city_time(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, LOCATION_COLUMNS + [DATE_COLUMN, AQI_COLUMN])
    return df.sort_values(LOCATION_COLUMNS + [DATE_COLUMN]).reset_index(drop=True).copy()


def _validate_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _validate_positive_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"Horizon must be positive. Received: {horizon}")


def _validate_percentile(percentile: float) -> None:
    if percentile <= 0 or percentile >= 1:
        raise ValueError(f"Percentile must be between 0 and 1. Received: {percentile}")


def _percentile_threshold_column(percentile: float) -> str:
    return f"city_aqi_p{int(percentile * 100):02d}"

