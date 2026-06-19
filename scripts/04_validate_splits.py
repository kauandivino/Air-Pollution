from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.splits import (
    iter_leave_one_country_out_splits,
    make_random_split,
    make_temporal_split,
    summarize_split,
    validate_leave_one_country_integrity,
    validate_split_integrity,
)


TARGET_COLUMN = "extreme_abs_151_h1"
FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DIAGNOSTIC_COLUMNS = ["Country", "Date", TARGET_COLUMN]


def main() -> None:
    ensure_results_dirs()
    df = pd.read_csv(FEATURE_MATRIX_PATH, usecols=DIAGNOSTIC_COLUMNS, parse_dates=["Date"])

    random_split = make_random_split(df, TARGET_COLUMN)
    temporal_split = make_temporal_split(df, TARGET_COLUMN)
    validate_split_integrity(random_split)
    validate_split_integrity(temporal_split)

    summaries = [
        summarize_split(df, random_split, TARGET_COLUMN),
        summarize_split(df, temporal_split, TARGET_COLUMN),
    ]

    loco_summaries = []
    for split in iter_leave_one_country_out_splits(df, TARGET_COLUMN):
        validate_leave_one_country_integrity(df, split)
        loco_summaries.append(summarize_split(df, split, TARGET_COLUMN))

    split_summary = pd.concat(summaries, ignore_index=True)
    loco_summary = pd.concat(loco_summaries, ignore_index=True)

    split_summary.to_csv(TABLES_DIR / "split_summary_random_temporal.csv", index=False)
    loco_summary.to_csv(TABLES_DIR / "split_summary_leave_one_country_out.csv", index=False)

    loco_test_summary = loco_summary[loco_summary["part"] == "test"].copy()
    loco_test_summary = loco_test_summary.sort_values("prevalence", ascending=False)
    loco_test_summary.to_csv(
        TABLES_DIR / "split_summary_loco_test_by_country.csv",
        index=False,
    )

    print("Split validation completed.")
    print()
    print("Random and temporal split summary:")
    print(
        split_summary[
            [
                "split_name",
                "part",
                "rows",
                "countries",
                "min_date",
                "max_date",
                "events",
                "prevalence",
            ]
        ].to_string(index=False)
    )
    print()
    print("Leave-one-country-out test summary:")
    print(
        loco_test_summary[
            [
                "held_out_country",
                "rows",
                "countries",
                "min_date",
                "max_date",
                "events",
                "prevalence",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )
    print()
    print(f"Split summaries saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
