from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.feature_engineering import (
    DEFAULT_LOCAL_NORMALIZATION_COLUMNS,
    add_past_city_normalized_features,
    summarize_feature_matrix,
)


BASE_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
LOCAL_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global_local.csv"


def main() -> None:
    ensure_results_dirs()
    df = pd.read_csv(BASE_FEATURE_MATRIX, parse_dates=["Date"])

    local_df, local_catalog = add_past_city_normalized_features(
        df,
        columns=DEFAULT_LOCAL_NORMALIZATION_COLUMNS,
        min_history=6,
    )
    local_features = local_catalog["feature"].tolist()
    local_summary = summarize_feature_matrix(local_df, local_features)

    local_catalog.to_csv(TABLES_DIR / "local_feature_catalog.csv", index=False)
    local_summary.to_csv(TABLES_DIR / "local_feature_missing_summary.csv", index=False)
    local_df.to_csv(LOCAL_FEATURE_MATRIX, index=False)

    print("Local normalized feature matrix completed.")
    print(f"Rows: {len(local_df)}")
    print(f"Local features: {len(local_features)}")
    print()
    print("Local feature missing summary:")
    print(local_summary.head(15).to_string(index=False))
    print()
    print(f"Local feature matrix saved to: {LOCAL_FEATURE_MATRIX}")
    print(f"Local feature catalog saved to: {TABLES_DIR / 'local_feature_catalog.csv'}")


if __name__ == "__main__":
    main()

