from __future__ import annotations

import pandas as pd


def inverse_country_size_weights(
    df: pd.DataFrame,
    train_index: pd.Index,
    country_column: str = "Country",
) -> tuple[pd.Series, pd.DataFrame]:
    """Give each training country the same total mass."""
    train_countries = df.loc[train_index, country_column]
    country_counts = train_countries.value_counts()
    n_train = len(train_index)
    n_countries = country_counts.shape[0]
    country_weights = n_train / (n_countries * country_counts)
    row_weights = train_countries.map(country_weights).astype(float)
    row_weights.index = train_index
    row_weights = row_weights / row_weights.mean()

    diagnostics = (
        pd.DataFrame(
            {
                "train_country": country_counts.index,
                "country_rows": country_counts.to_numpy(),
                "country_weight": country_weights.loc[country_counts.index].to_numpy(),
            }
        )
        .sort_values("country_weight", ascending=False)
        .reset_index(drop=True)
    )
    return row_weights, diagnostics


def country_class_balanced_weights(
    df: pd.DataFrame,
    train_index: pd.Index,
    target: str,
    country_column: str = "Country",
    min_weight: float = 0.10,
    max_weight: float = 10.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Balance country mass and class mass inside each country.

    Each country receives equal total importance. Within each country, positive
    and negative classes receive equal total importance when both classes exist.
    """
    train = df.loc[train_index, [country_column, target]].copy()
    train[target] = train[target].astype(int)
    n_train = len(train)
    n_countries = train[country_column].nunique()

    group_counts = train.groupby([country_column, target], observed=True).size()
    country_class_counts = train.groupby(country_column, observed=True)[target].nunique()

    weights = []
    diagnostics_rows = []
    for idx, row in train.iterrows():
        country = row[country_column]
        klass = int(row[target])
        class_count = group_counts.loc[(country, klass)]
        class_groups = country_class_counts.loc[country]
        weight = n_train / (n_countries * class_groups * class_count)
        weights.append((idx, weight))

    row_weights = pd.Series(dict(weights), dtype=float).reindex(train_index)
    row_weights = row_weights.clip(lower=min_weight, upper=max_weight)
    row_weights = row_weights / row_weights.mean()

    for (country, klass), count in group_counts.items():
        class_groups = country_class_counts.loc[country]
        raw_weight = n_train / (n_countries * class_groups * count)
        diagnostics_rows.append(
            {
                "train_country": country,
                "class": klass,
                "country_class_rows": count,
                "country_observed_classes": class_groups,
                "raw_country_class_weight": raw_weight,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["train_country", "class"]
    ).reset_index(drop=True)
    return row_weights, diagnostics


def domain_weights(
    df: pd.DataFrame,
    train_index: pd.Index,
    target: str,
    strategy: str,
    country_column: str = "Country",
) -> tuple[pd.Series, pd.DataFrame]:
    if strategy == "inverse_country_size":
        return inverse_country_size_weights(df, train_index, country_column=country_column)
    if strategy == "country_class_balanced":
        return country_class_balanced_weights(
            df,
            train_index,
            target=target,
            country_column=country_column,
        )
    raise ValueError(f"Unknown domain-weighting strategy: {strategy}")
