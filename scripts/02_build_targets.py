from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.data_loading import load_global_air_quality_data
from src.target_builder import (
    DEFAULT_ABSOLUTE_THRESHOLDS,
    DEFAULT_CITY_PERCENTILES,
    DEFAULT_HORIZONS,
    add_extreme_event_targets,
    list_target_columns,
    summarize_target_prevalence,
)


TARGETED_EXPORT_COLUMNS = [
    "Country",
    "State",
    "City",
    "Date",
    "AQI",
    "future_aqi_h1",
    "future_aqi_h3",
    "city_aqi_p90",
    "city_aqi_p95",
]


def main() -> None:
    ensure_results_dirs()
    df = load_global_air_quality_data()

    target_df = add_extreme_event_targets(
        df,
        horizons=DEFAULT_HORIZONS,
        absolute_thresholds=DEFAULT_ABSOLUTE_THRESHOLDS,
        city_percentiles=DEFAULT_CITY_PERCENTILES,
    )
    target_columns = list_target_columns(
        horizons=DEFAULT_HORIZONS,
        absolute_thresholds=DEFAULT_ABSOLUTE_THRESHOLDS,
        city_percentiles=DEFAULT_CITY_PERCENTILES,
    )

    summary = summarize_target_prevalence(target_df, target_columns)
    by_country = summarize_target_prevalence(
        target_df,
        target_columns,
        groupby=["Country"],
    )

    summary.to_csv(TABLES_DIR / "target_prevalence_summary.csv", index=False)
    by_country.to_csv(TABLES_DIR / "target_prevalence_by_country.csv", index=False)

    export_columns = TARGETED_EXPORT_COLUMNS + target_columns
    target_df[export_columns].to_csv(
        PROCESSED_DATA_DIR / "target_preview_global.csv",
        index=False,
    )

    print("Target construction completed.")
    print(summary.to_string(index=False))
    print()
    print("Most frequent country-target combinations:")
    print(
        by_country.sort_values("prevalence", ascending=False)
        .head(12)
        .to_string(index=False)
    )
    print()
    print(f"Target prevalence tables saved to: {TABLES_DIR}")
    print(f"Target preview saved to: {PROCESSED_DATA_DIR / 'target_preview_global.csv'}")


if __name__ == "__main__":
    main()
