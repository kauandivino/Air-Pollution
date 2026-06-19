from __future__ import annotations

import pandas as pd


FEATURE_SET_ORDER = [
    "aqi_history",
    "aqi_pollutants",
    "aqi_pollutants_meteorology",
    "all_features",
]


def load_feature_catalog(path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    required = {"feature", "group"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Feature catalog is missing columns: {', '.join(sorted(missing))}")
    return catalog


def build_feature_sets(feature_catalog: pd.DataFrame) -> dict[str, list[str]]:
    """Build cumulative feature sets for ablation experiments."""
    all_features = feature_catalog["feature"].drop_duplicates().tolist()

    calendar = _features_by_group(feature_catalog, ["calendar"])
    current_aqi = _features_by_group(feature_catalog, ["current_aqi"])
    aqi_temporal = _features_matching_prefix(
        all_features,
        prefixes=["AQI_lag_", "AQI_roll_", "AQI_trend_", "AQI_delta_"],
    )

    pollutant_features = _features_by_group(feature_catalog, ["current_pollutant"])
    pollutant_features += _features_matching_prefix(
        all_features,
        prefixes=[
            "PM25_lag_",
            "PM10_lag_",
            "PM25_roll_",
            "PM10_roll_",
            "CO_lag_",
            "NO2_lag_",
            "SO2_lag_",
            "O3_lag_",
            "PM25_PM10_ratio",
        ],
    )

    meteorology_features = _features_by_group(feature_catalog, ["current_meteorology"])
    meteorology_features += _features_matching_prefix(
        all_features,
        prefixes=["Wind_Speed_lag_", "Humidity_lag_", "low_wind_PM25", "humidity_PM25"],
    )

    socio_environmental_features = _features_by_group(
        feature_catalog,
        ["current_socio_environmental"],
    )
    socio_environmental_features += _features_matching_prefix(
        all_features,
        prefixes=["Deforestation_Rate_lag_", "deforestation_aug_oct", "industry_NO2"],
    )

    feature_sets = {
        "aqi_history": _deduplicate(calendar + current_aqi + aqi_temporal),
        "aqi_pollutants": _deduplicate(calendar + current_aqi + aqi_temporal + pollutant_features),
        "aqi_pollutants_meteorology": _deduplicate(
            calendar + current_aqi + aqi_temporal + pollutant_features + meteorology_features
        ),
        "all_features": _deduplicate(
            calendar
            + current_aqi
            + aqi_temporal
            + pollutant_features
            + meteorology_features
            + socio_environmental_features
        ),
    }

    missing_from_all = set(all_features) - set(feature_sets["all_features"])
    if missing_from_all:
        raise ValueError(
            "Some catalog features were not assigned to all_features: "
            + ", ".join(sorted(missing_from_all))
        )

    return feature_sets


def summarize_feature_sets(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_set": name, "n_features": len(features)}
            for name, features in feature_sets.items()
        ]
    )


def _features_by_group(feature_catalog: pd.DataFrame, groups: list[str]) -> list[str]:
    return feature_catalog.loc[feature_catalog["group"].isin(groups), "feature"].tolist()


def _features_matching_prefix(features: list[str], prefixes: list[str]) -> list[str]:
    return [feature for feature in features if any(feature.startswith(prefix) for prefix in prefixes)]


def _deduplicate(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))

