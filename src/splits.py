from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


COUNTRY_COLUMN = "Country"
DATE_COLUMN = "Date"
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class SplitResult:
    name: str
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    metadata: dict[str, object]


def filter_valid_target_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    _validate_required_columns(df, [target_column])
    return df[df[target_column].notna()].copy()


def make_random_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
    stratify: bool = True,
) -> SplitResult:
    """Create a stratified random split over valid target rows."""
    _validate_split_sizes(train_size, val_size, test_size)
    valid_df = filter_valid_target_rows(df, target_column)
    stratify_values = _stratify_values(valid_df, target_column, stratify)

    train_val_idx, test_idx = train_test_split(
        valid_df.index,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_values,
    )

    train_val_df = valid_df.loc[train_val_idx]
    relative_val_size = val_size / (train_size + val_size)
    train_val_stratify = _stratify_values(train_val_df, target_column, stratify)

    train_idx, val_idx = train_test_split(
        train_val_df.index,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=train_val_stratify,
    )

    return SplitResult(
        name="random_split",
        train_idx=pd.Index(train_idx),
        val_idx=pd.Index(val_idx),
        test_idx=pd.Index(test_idx),
        metadata={
            "target": target_column,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "random_seed": random_seed,
            "stratified": stratify,
        },
    )


def make_temporal_split(
    df: pd.DataFrame,
    target_column: str,
    train_end: str = "2021-12-31",
    val_start: str = "2022-01-01",
    val_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2025-12-31",
) -> SplitResult:
    """Create a chronological split over valid target rows."""
    _validate_required_columns(df, [DATE_COLUMN, target_column])
    valid_df = filter_valid_target_rows(df, target_column)
    dates = pd.to_datetime(valid_df[DATE_COLUMN], errors="raise")

    train_end_dt = pd.Timestamp(train_end)
    val_start_dt = pd.Timestamp(val_start)
    val_end_dt = pd.Timestamp(val_end)
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)

    if not (train_end_dt < val_start_dt <= val_end_dt < test_start_dt <= test_end_dt):
        raise ValueError("Temporal split boundaries must be ordered and non-overlapping.")

    train_idx = valid_df.loc[dates <= train_end_dt].index
    val_idx = valid_df.loc[(dates >= val_start_dt) & (dates <= val_end_dt)].index
    test_idx = valid_df.loc[(dates >= test_start_dt) & (dates <= test_end_dt)].index

    return SplitResult(
        name="temporal_split",
        train_idx=pd.Index(train_idx),
        val_idx=pd.Index(val_idx),
        test_idx=pd.Index(test_idx),
        metadata={
            "target": target_column,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        },
    )


def iter_leave_one_country_out_splits(
    df: pd.DataFrame,
    target_column: str,
    countries: Iterable[str] | None = None,
    val_size: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
    stratify: bool = True,
) -> Iterator[SplitResult]:
    """Yield one geographical split per held-out country.

    The held-out country appears only in the test split. Validation is sampled
    from the remaining countries, preserving the geographical isolation of test.
    """
    _validate_required_columns(df, [COUNTRY_COLUMN, target_column])
    if val_size <= 0 or val_size >= 1:
        raise ValueError(f"val_size must be between 0 and 1. Received: {val_size}")

    valid_df = filter_valid_target_rows(df, target_column)
    selected_countries = (
        sorted(valid_df[COUNTRY_COLUMN].dropna().unique().tolist())
        if countries is None
        else sorted(countries)
    )

    for country in selected_countries:
        test_mask = valid_df[COUNTRY_COLUMN] == country
        test_idx = valid_df.loc[test_mask].index
        train_val_df = valid_df.loc[~test_mask]

        if test_idx.empty:
            raise ValueError(f"No valid test rows for held-out country: {country}")
        if train_val_df.empty:
            raise ValueError(f"No training rows after holding out country: {country}")

        train_val_stratify = _stratify_values(train_val_df, target_column, stratify)
        train_idx, val_idx = train_test_split(
            train_val_df.index,
            test_size=val_size,
            random_state=random_seed,
            stratify=train_val_stratify,
        )

        yield SplitResult(
            name="leave_one_country_out",
            train_idx=pd.Index(train_idx),
            val_idx=pd.Index(val_idx),
            test_idx=pd.Index(test_idx),
            metadata={
                "target": target_column,
                "held_out_country": country,
                "val_size": val_size,
                "random_seed": random_seed,
                "stratified": stratify,
            },
        )


def summarize_split(df: pd.DataFrame, split: SplitResult, target_column: str) -> pd.DataFrame:
    _validate_required_columns(df, [COUNTRY_COLUMN, DATE_COLUMN, target_column])
    rows = []
    for part, index in [
        ("train", split.train_idx),
        ("validation", split.val_idx),
        ("test", split.test_idx),
    ]:
        part_df = df.loc[index]
        valid_target = part_df[target_column].dropna()
        events = int(valid_target.astype("int8").sum()) if len(valid_target) else 0
        rows.append(
            {
                "split_name": split.name,
                "part": part,
                "rows": len(part_df),
                "countries": part_df[COUNTRY_COLUMN].nunique(),
                "min_date": _date_or_empty(part_df[DATE_COLUMN].min()),
                "max_date": _date_or_empty(part_df[DATE_COLUMN].max()),
                "target": target_column,
                "events": events,
                "non_events": len(valid_target) - events,
                "prevalence": events / len(valid_target) if len(valid_target) else 0.0,
                **split.metadata,
            }
        )
    return pd.DataFrame(rows)


def validate_split_integrity(split: SplitResult) -> None:
    train = set(split.train_idx)
    val = set(split.val_idx)
    test = set(split.test_idx)

    if train & val:
        raise ValueError(f"{split.name}: train and validation sets overlap.")
    if train & test:
        raise ValueError(f"{split.name}: train and test sets overlap.")
    if val & test:
        raise ValueError(f"{split.name}: validation and test sets overlap.")
    if not train or not val or not test:
        raise ValueError(f"{split.name}: all split parts must be non-empty.")


def validate_leave_one_country_integrity(df: pd.DataFrame, split: SplitResult) -> None:
    validate_split_integrity(split)
    held_out_country = split.metadata.get("held_out_country")
    if held_out_country is None:
        raise ValueError("Missing held_out_country metadata.")

    train_countries = set(df.loc[split.train_idx, COUNTRY_COLUMN].dropna().unique())
    val_countries = set(df.loc[split.val_idx, COUNTRY_COLUMN].dropna().unique())
    test_countries = set(df.loc[split.test_idx, COUNTRY_COLUMN].dropna().unique())

    if test_countries != {held_out_country}:
        raise ValueError(
            f"Test split must contain only {held_out_country}. Found: {sorted(test_countries)}"
        )
    if held_out_country in train_countries or held_out_country in val_countries:
        raise ValueError(f"Held-out country leaked into train/validation: {held_out_country}")


def _stratify_values(
    df: pd.DataFrame,
    target_column: str,
    stratify: bool,
) -> pd.Series | None:
    if not stratify:
        return None

    values = df[target_column].astype("int8")
    if values.nunique() < 2:
        return None
    if values.value_counts().min() < 2:
        return None
    return values


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    sizes = [train_size, val_size, test_size]
    if any(size <= 0 for size in sizes):
        raise ValueError("All split sizes must be positive.")
    if abs(sum(sizes) - 1.0) > 1e-9:
        raise ValueError(
            f"Split sizes must sum to 1. Received: {train_size + val_size + test_size}"
        )


def _validate_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _date_or_empty(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()

