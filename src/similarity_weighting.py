from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PROFILE_COLUMNS = [
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


def country_environmental_profiles(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    country_column: str = "Country",
) -> pd.DataFrame:
    """Return one robust environmental profile per country."""
    columns = columns or DEFAULT_PROFILE_COLUMNS
    missing = [column for column in [country_column, *columns] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for country profiles: {missing}")

    aggregations = {}
    for column in columns:
        aggregations[f"{column}__mean"] = (column, "mean")
        aggregations[f"{column}__std"] = (column, "std")
        aggregations[f"{column}__p90"] = (column, lambda values: values.quantile(0.90))

    profiles = df.groupby(country_column, observed=True).agg(**aggregations)
    return profiles.fillna(0.0)


def country_similarity_weights(
    df: pd.DataFrame,
    held_out_country: str,
    train_index: pd.Index,
    columns: list[str] | None = None,
    country_column: str = "Country",
    temperature: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Create row-level sample weights from country-profile similarity.

    The held-out country's unlabeled covariate profile is compared with every
    training-country profile. Closer countries receive larger weights.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if min_weight <= 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("Weight bounds must be positive and ordered.")

    profiles = country_environmental_profiles(df, columns=columns, country_column=country_column)
    if held_out_country not in profiles.index:
        raise ValueError(f"Held-out country not found in profiles: {held_out_country}")

    train_countries = sorted(df.loc[train_index, country_column].dropna().unique().tolist())
    if not train_countries:
        raise ValueError("No training countries available for similarity weighting.")

    profile_means = profiles.mean(axis=0)
    profile_stds = profiles.std(axis=0).replace(0, np.nan)
    standardized = (profiles - profile_means) / profile_stds
    standardized = standardized.fillna(0.0)

    target_profile = standardized.loc[held_out_country]
    train_profiles = standardized.loc[train_countries]
    distances = ((train_profiles - target_profile) ** 2).mean(axis=1).pow(0.5)
    raw_weights = np.exp(-distances / temperature)

    if raw_weights.mean() == 0 or np.isnan(raw_weights.mean()):
        country_weights = pd.Series(1.0, index=raw_weights.index)
    else:
        country_weights = raw_weights / raw_weights.mean()
    country_weights = country_weights.clip(lower=min_weight, upper=max_weight)
    country_weights = country_weights / country_weights.mean()

    row_weights = df.loc[train_index, country_column].map(country_weights).astype(float)
    row_weights.index = train_index

    diagnostics = pd.DataFrame(
        {
            "held_out_country": held_out_country,
            "train_country": country_weights.index,
            "profile_distance": distances.loc[country_weights.index].to_numpy(),
            "similarity_weight": country_weights.to_numpy(),
        }
    ).sort_values("profile_distance")
    return row_weights, diagnostics.reset_index(drop=True)
