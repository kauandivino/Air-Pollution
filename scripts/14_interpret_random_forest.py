from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.evaluation import evaluate_model_on_parts
from src.feature_sets import build_feature_sets, load_feature_catalog
from src.models import get_model_specs
from src.splits import make_temporal_split, validate_split_integrity


FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_global.csv"
DEFAULT_TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
FEATURE_SET = "all_features"
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret Random Forest alert models.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_modeling_frame(targets: list[str], feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date"] + targets + feature_columns
    return pd.read_csv(FEATURE_MATRIX_PATH, usecols=usecols, parse_dates=["Date"])


def stratified_sample_index(
    y: pd.Series,
    sample_size: int,
    random_seed: int = RANDOM_SEED,
) -> pd.Index:
    if len(y) <= sample_size:
        return y.index
    stratify = y.astype("int8") if y.nunique() == 2 else None
    _, sample_idx = train_test_split(
        y.index,
        test_size=sample_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return pd.Index(sample_idx)


def native_feature_importance(estimator, feature_columns: list[str], target: str) -> pd.DataFrame:
    forest = estimator.named_steps["model"]
    return (
        pd.DataFrame(
            {
                "target": target,
                "feature": feature_columns,
                "importance": forest.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance(
    estimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
    sample_size: int,
    repeats: int,
) -> pd.DataFrame:
    sample_idx = stratified_sample_index(y_test, sample_size=sample_size)
    x_sample = x_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx].astype(int)

    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    result = permutation_importance(
        estimator,
        x_sample,
        y_sample,
        scoring=scorer,
        n_repeats=repeats,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "target": target,
                "feature": x_test.columns,
                "permutation_importance_mean": result.importances_mean,
                "permutation_importance_std": result.importances_std,
                "sample_rows": len(x_sample),
                "repeats": repeats,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def add_feature_groups(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    grouped = df.merge(catalog[["feature", "group"]], on="feature", how="left")
    grouped["group"] = grouped["group"].fillna("unknown")
    return grouped


def summarize_groups(native: pd.DataFrame, permutation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_group = (
        native.groupby(["target", "group"], observed=True)
        .agg(
            importance_sum=("importance", "sum"),
            importance_mean=("importance", "mean"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
        .sort_values(["target", "importance_sum"], ascending=[True, False])
    )
    permutation_group = (
        permutation.groupby(["target", "group"], observed=True)
        .agg(
            permutation_importance_sum=("permutation_importance_mean", "sum"),
            permutation_importance_mean=("permutation_importance_mean", "mean"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
        .sort_values(["target", "permutation_importance_sum"], ascending=[True, False])
    )
    return native_group, permutation_group


def compare_horizons(native: pd.DataFrame, permutation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_wide = native.pivot_table(
        index=["feature", "group"],
        columns="target",
        values="importance",
        aggfunc="mean",
    ).reset_index()
    permutation_wide = permutation.pivot_table(
        index=["feature", "group"],
        columns="target",
        values="permutation_importance_mean",
        aggfunc="mean",
    ).reset_index()

    if set(DEFAULT_TARGETS).issubset(native_wide.columns):
        native_wide["delta_h3_minus_h1_importance"] = (
            native_wide["extreme_abs_151_h3"] - native_wide["extreme_abs_151_h1"]
        )
    if set(DEFAULT_TARGETS).issubset(permutation_wide.columns):
        permutation_wide["delta_h3_minus_h1_permutation"] = (
            permutation_wide["extreme_abs_151_h3"] - permutation_wide["extreme_abs_151_h1"]
        )
    return native_wide, permutation_wide


def save_figures(native: pd.DataFrame, permutation: pd.DataFrame, native_group: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    for target in native["target"].unique():
        target_native = native[native["target"] == target].head(20)
        plt.figure(figsize=(11, 8))
        sns.barplot(data=target_native, y="feature", x="importance", hue="group", dodge=False)
        plt.title(f"Random Forest feature importance - {target}")
        plt.xlabel("Importancia nativa")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"rf_native_importance_top20_{target}.png", dpi=170)
        plt.close()

        target_perm = permutation[permutation["target"] == target].head(20)
        plt.figure(figsize=(11, 8))
        sns.barplot(
            data=target_perm,
            y="feature",
            x="permutation_importance_mean",
            hue="group",
            dodge=False,
        )
        plt.title(f"Random Forest permutation importance - {target}")
        plt.xlabel("Queda media de PR-AUC ao embaralhar")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"rf_permutation_importance_top20_{target}.png", dpi=170)
        plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=native_group,
        x="group",
        y="importance_sum",
        hue="target",
        palette=["#3d5a80", "#b23a48"],
    )
    plt.title("Importancia nativa agregada por grupo de features")
    plt.xlabel("Grupo")
    plt.ylabel("Soma das importancias")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_native_importance_by_group.png", dpi=170)
    plt.close()


def run_target(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    sample_size: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_df = df[df[target].notna()].copy()
    split = make_temporal_split(target_df, target)
    validate_split_integrity(split)

    spec = get_model_specs(["random_forest"])[0]
    estimator = clone(spec.estimator)
    x = target_df[feature_columns]
    y = target_df[target]

    print(f"Training Random Forest for interpretability: {target}")
    estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx].astype(int))

    metrics = evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={"validation": split.val_idx, "test": split.test_idx},
        base_metadata={
            "target": target,
            "split_name": "temporal_split",
            "model": "random_forest",
            "feature_set": FEATURE_SET,
        },
    )

    native = native_feature_importance(estimator, feature_columns, target)
    permutation = compute_permutation_importance(
        estimator,
        x.loc[split.test_idx],
        y.loc[split.test_idx],
        target=target,
        sample_size=sample_size,
        repeats=repeats,
    )
    return metrics, native, permutation


def main() -> None:
    args = parse_args()
    ensure_results_dirs()

    catalog = load_feature_catalog(TABLES_DIR / "feature_catalog.csv")
    feature_sets = build_feature_sets(catalog)
    feature_columns = feature_sets[FEATURE_SET]
    df = load_modeling_frame(args.targets, feature_columns)

    metrics_frames = []
    native_frames = []
    permutation_frames = []

    for target in args.targets:
        metrics, native, permutation = run_target(
            df=df,
            target=target,
            feature_columns=feature_columns,
            sample_size=args.sample_size,
            repeats=args.permutation_repeats,
        )
        metrics_frames.append(metrics)
        native_frames.append(native)
        permutation_frames.append(permutation)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    native = add_feature_groups(pd.concat(native_frames, ignore_index=True), catalog)
    permutation = add_feature_groups(pd.concat(permutation_frames, ignore_index=True), catalog)
    native_group, permutation_group = summarize_groups(native, permutation)
    native_horizon, permutation_horizon = compare_horizons(native, permutation)

    metrics.to_csv(METRICS_DIR / "rf_interpretability_temporal_metrics.csv", index=False)
    native.to_csv(TABLES_DIR / "rf_native_feature_importance.csv", index=False)
    permutation.to_csv(TABLES_DIR / "rf_permutation_importance.csv", index=False)
    native_group.to_csv(TABLES_DIR / "rf_native_importance_by_group.csv", index=False)
    permutation_group.to_csv(TABLES_DIR / "rf_permutation_importance_by_group.csv", index=False)
    native_horizon.to_csv(TABLES_DIR / "rf_native_importance_h1_vs_h3.csv", index=False)
    permutation_horizon.to_csv(TABLES_DIR / "rf_permutation_importance_h1_vs_h3.csv", index=False)

    save_figures(native, permutation, native_group)

    print()
    print("Random Forest interpretability completed.")
    print()
    print("Temporal test metrics:")
    print(metrics[["target", "part", "rows", "event_rate", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print()
    print("Top native importances:")
    print(native.groupby("target").head(12)[["target", "feature", "group", "importance"]].to_string(index=False))
    print()
    print("Top permutation importances:")
    print(
        permutation.groupby("target")
        .head(12)[["target", "feature", "group", "permutation_importance_mean", "permutation_importance_std"]]
        .to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
