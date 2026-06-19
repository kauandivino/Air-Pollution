from __future__ import annotations

import pandas as pd

from src.config import NUMERIC_COLUMNS, RAW_GLOBAL_DATA, REQUIRED_COLUMNS


def load_global_air_quality_data(path=RAW_GLOBAL_DATA) -> pd.DataFrame:
    """Load the global air-quality CSV and apply basic type normalization."""
    path = path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    validate_required_columns(df)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = ["Country", "State", "City", "AQI_Bucket"]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")

