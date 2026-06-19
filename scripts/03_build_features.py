from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.data_loading import load_global_air_quality_data
from src.feature_engineering import (
    build_supervised_feature_matrix,
    feature_columns_from_catalog,
    summarize_feature_matrix,
)
from src.target_builder import (
    DEFAULT_ABSOLUTE_THRESHOLDS,
    DEFAULT_CITY_PERCENTILES,
    DEFAULT_HORIZONS,
    add_extreme_event_targets,
    list_target_columns,
)


IDENTIFIER_COLUMNS = ["Country", "State", "City", "Date"]


def main() -> None:
    ensure_results_dirs()

    df = load_global_air_quality_data()
    target_df = add_extreme_event_targets(
        df,
        horizons=DEFAULT_HORIZONS,
        absolute_thresholds=DEFAULT_ABSOLUTE_THRESHOLDS,
        city_percentiles=DEFAULT_CITY_PERCENTILES,
    )
    feature_df, feature_catalog = build_supervised_feature_matrix(target_df)

    feature_columns = feature_columns_from_catalog(feature_catalog)
    target_columns = list_target_columns(
        horizons=DEFAULT_HORIZONS,
        absolute_thresholds=DEFAULT_ABSOLUTE_THRESHOLDS,
        city_percentiles=DEFAULT_CITY_PERCENTILES,
    )
    future_columns = [f"future_aqi_h{horizon}" for horizon in DEFAULT_HORIZONS]
    threshold_columns = ["city_aqi_p90", "city_aqi_p95"]

    export_columns = (
        IDENTIFIER_COLUMNS
        + future_columns
        + threshold_columns
        + target_columns
        + feature_columns
    )
    export_columns = [column for column in export_columns if column in feature_df.columns]

    feature_summary = summarize_feature_matrix(feature_df, feature_columns)
    target_validity = feature_df[target_columns].notna().sum().reset_index()
    target_validity.columns = ["target", "valid_rows"]

    feature_catalog.to_csv(TABLES_DIR / "feature_catalog.csv", index=False)
    feature_summary.to_csv(TABLES_DIR / "feature_missing_summary.csv", index=False)
    target_validity.to_csv(TABLES_DIR / "feature_matrix_target_validity.csv", index=False)
    feature_df[export_columns].to_csv(
        PROCESSED_DATA_DIR / "feature_matrix_global.csv",
        index=False,
    )

    print("Feature matrix construction completed.")
    print(f"Rows: {len(feature_df)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Target columns: {len(target_columns)}")
    print()
    print("Feature groups:")
    print(feature_catalog.groupby("group").size().sort_values(ascending=False).to_string())
    print()
    print("Features with highest missing rates:")
    print(feature_summary.head(12).to_string(index=False))
    print()
    print(f"Feature matrix saved to: {PROCESSED_DATA_DIR / 'feature_matrix_global.csv'}")
    print(f"Feature catalog saved to: {TABLES_DIR / 'feature_catalog.csv'}")


if __name__ == "__main__":
    main()
