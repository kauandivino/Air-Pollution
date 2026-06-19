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
from src.domain_weighting import domain_weights
from src.evaluation import evaluate_model_on_parts
from src.models import get_model_specs
from src.splits import iter_leave_one_country_out_splits, validate_leave_one_country_integrity


LOCAL_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global_local.csv"
TARGET = "extreme_abs_151_h3"
MODEL = "xgboost"
DEFAULT_COUNTRIES = ["Brazil", "China", "Vietnam"]
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret final XGBoost h3 model with local features and country-class weights."
    )
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_feature_columns_and_catalog() -> tuple[list[str], pd.DataFrame]:
    base_catalog = pd.read_csv(TABLES_DIR / "feature_catalog.csv")
    local_catalog = pd.read_csv(TABLES_DIR / "local_feature_catalog.csv")
    catalog = pd.concat([base_catalog, local_catalog], ignore_index=True)
    catalog = catalog.drop_duplicates(subset=["feature"], keep="first")
    features = catalog["feature"].tolist()
    return features, catalog


def load_modeling_frame(target: str, feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date", target, *feature_columns]
    usecols = list(dict.fromkeys(usecols))
    return pd.read_csv(LOCAL_FEATURE_MATRIX, usecols=usecols, parse_dates=["Date"])


def stratified_sample_index(y: pd.Series, sample_size: int) -> pd.Index:
    if len(y) <= sample_size:
        return y.index
    stratify = y.astype("int8") if y.nunique() == 2 else None
    _, sample_idx = train_test_split(
        y.index,
        test_size=sample_size,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    return pd.Index(sample_idx)


def native_importance(estimator, feature_columns: list[str], country: str, target: str) -> pd.DataFrame:
    booster = estimator.named_steps["model"]
    return (
        pd.DataFrame(
            {
                "target": target,
                "held_out_country": country,
                "feature": feature_columns,
                "native_importance": booster.feature_importances_,
            }
        )
        .sort_values("native_importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance(
    estimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    country: str,
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
                "held_out_country": country,
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


def checkpoint_paths(country: str, target: str) -> tuple[Path, Path, Path]:
    safe_country = country.replace(" ", "_").replace("/", "_")
    prefix = f"{target}__xgboost_final_interpretability__{safe_country}"
    checkpoint_dir = METRICS_DIR / "final_xgboost_interpretability_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return (
        checkpoint_dir / f"{prefix}__metrics.csv",
        checkpoint_dir / f"{prefix}__native.csv",
        checkpoint_dir / f"{prefix}__permutation.csv",
    )


def run_country(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    country: str,
    sample_size: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_df = df[df[target].notna()].copy()
    splits = {
        split.metadata["held_out_country"]: split
        for split in iter_leave_one_country_out_splits(target_df, target, countries=[country])
    }
    split = splits[country]
    validate_leave_one_country_integrity(target_df, split)

    spec = get_model_specs([MODEL])[0]
    estimator = clone(spec.estimator)
    x = target_df[feature_columns]
    y = target_df[target]
    sample_weight, _ = domain_weights(
        df=target_df,
        train_index=split.train_idx,
        target=target,
        strategy="country_class_balanced",
    )

    print(f"Training final XGBoost interpretability fold: held out {country}")
    estimator.fit(
        x.loc[split.train_idx],
        y.loc[split.train_idx].astype(int),
        **{"model__sample_weight": sample_weight.to_numpy()},
    )

    metrics = evaluate_model_on_parts(
        estimator,
        x=x,
        y=y,
        parts={"validation": split.val_idx, "test": split.test_idx},
        base_metadata={
            "target": target,
            "model": MODEL,
            "feature_set": "all_plus_local_normalization",
            "weighting_strategy": "country_class_balanced",
            "held_out_country": country,
        },
    )
    native = native_importance(estimator, feature_columns, country, target)
    permutation = compute_permutation_importance(
        estimator=estimator,
        x_test=x.loc[split.test_idx],
        y_test=y.loc[split.test_idx],
        country=country,
        target=target,
        sample_size=sample_size,
        repeats=repeats,
    )
    return metrics, native, permutation


def add_groups(data: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    grouped = data.merge(catalog[["feature", "group"]], on="feature", how="left")
    grouped["group"] = grouped["group"].fillna("unknown")
    return grouped


def summarize(native: pd.DataFrame, permutation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_summary = (
        native.groupby(["feature", "group"], observed=True)
        .agg(
            native_importance_mean=("native_importance", "mean"),
            native_importance_std=("native_importance", "std"),
            folds=("held_out_country", "nunique"),
        )
        .reset_index()
        .sort_values("native_importance_mean", ascending=False)
    )
    permutation_summary = (
        permutation.groupby(["feature", "group"], observed=True)
        .agg(
            permutation_importance_mean=("permutation_importance_mean", "mean"),
            permutation_importance_std=("permutation_importance_mean", "std"),
            folds=("held_out_country", "nunique"),
        )
        .reset_index()
        .sort_values("permutation_importance_mean", ascending=False)
    )
    return native_summary, permutation_summary


def summarize_groups(native: pd.DataFrame, permutation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_group = (
        native.groupby("group", observed=True)
        .agg(native_importance_sum=("native_importance", "sum"), n_features=("feature", "nunique"))
        .reset_index()
        .sort_values("native_importance_sum", ascending=False)
    )
    permutation_group = (
        permutation.groupby("group", observed=True)
        .agg(
            permutation_importance_sum=("permutation_importance_mean", "sum"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
        .sort_values("permutation_importance_sum", ascending=False)
    )
    return native_group, permutation_group


def save_figures(permutation_summary: pd.DataFrame, native_summary: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    top_perm = permutation_summary.head(25)
    plt.figure(figsize=(11, 8))
    sns.barplot(
        data=top_perm,
        y="feature",
        x="permutation_importance_mean",
        hue="group",
        dodge=False,
    )
    plt.title("Final XGBoost h3 - permutation importance")
    plt.xlabel("Queda media de PR-AUC ao embaralhar")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_xgboost_h3_permutation_importance_top25.png", dpi=180)
    plt.close()

    top_native = native_summary.head(25)
    plt.figure(figsize=(11, 8))
    sns.barplot(data=top_native, y="feature", x="native_importance_mean", hue="group", dodge=False)
    plt.title("Final XGBoost h3 - native feature importance")
    plt.xlabel("Importancia nativa media")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_xgboost_h3_native_importance_top25.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=group_summary,
        x="group",
        y="permutation_importance_sum",
        color="#2a9d8f",
    )
    plt.title("Final XGBoost h3 - importancia por grupo")
    plt.xlabel("Grupo de features")
    plt.ylabel("Soma da permutation importance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_xgboost_h3_permutation_importance_by_group.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    feature_columns, catalog = load_feature_columns_and_catalog()
    df = load_modeling_frame(args.target, feature_columns)

    metric_frames = []
    native_frames = []
    permutation_frames = []
    resume = not args.no_resume

    for country in args.countries:
        metric_path, native_path, permutation_path = checkpoint_paths(country, args.target)
        if resume and metric_path.exists() and native_path.exists() and permutation_path.exists():
            print(f"Skipping {country}: interpretability checkpoints exist.")
            metric_frames.append(pd.read_csv(metric_path))
            native_frames.append(pd.read_csv(native_path))
            permutation_frames.append(pd.read_csv(permutation_path))
            continue

        metrics, native, permutation = run_country(
            df=df,
            feature_columns=feature_columns,
            target=args.target,
            country=country,
            sample_size=args.sample_size,
            repeats=args.permutation_repeats,
        )
        metrics.to_csv(metric_path, index=False)
        native.to_csv(native_path, index=False)
        permutation.to_csv(permutation_path, index=False)
        metric_frames.append(metrics)
        native_frames.append(native)
        permutation_frames.append(permutation)

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    native_all = add_groups(pd.concat(native_frames, ignore_index=True), catalog)
    permutation_all = add_groups(pd.concat(permutation_frames, ignore_index=True), catalog)
    native_summary, permutation_summary = summarize(native_all, permutation_all)
    native_group, permutation_group = summarize_groups(native_all, permutation_all)

    metrics_path = METRICS_DIR / "final_xgboost_h3_interpretability_metrics.csv"
    native_path = TABLES_DIR / "final_xgboost_h3_native_importance.csv"
    permutation_path = TABLES_DIR / "final_xgboost_h3_permutation_importance.csv"
    native_summary_path = TABLES_DIR / "final_xgboost_h3_native_importance_summary.csv"
    permutation_summary_path = TABLES_DIR / "final_xgboost_h3_permutation_importance_summary.csv"
    group_path = TABLES_DIR / "final_xgboost_h3_permutation_importance_by_group.csv"

    metrics_all.to_csv(metrics_path, index=False)
    native_all.to_csv(native_path, index=False)
    permutation_all.to_csv(permutation_path, index=False)
    native_summary.to_csv(native_summary_path, index=False)
    permutation_summary.to_csv(permutation_summary_path, index=False)
    permutation_group.to_csv(group_path, index=False)
    native_group.to_csv(TABLES_DIR / "final_xgboost_h3_native_importance_by_group.csv", index=False)

    save_figures(permutation_summary, native_summary, permutation_group)

    print("Final XGBoost interpretability completed.")
    print()
    print("Fold metrics:")
    print(
        metrics_all[metrics_all["part"] == "test"][
            ["held_out_country", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )
    print()
    print("Top permutation features:")
    print(
        permutation_summary[
            ["feature", "group", "permutation_importance_mean", "permutation_importance_std", "folds"]
        ]
        .head(20)
        .to_string(index=False)
    )
    print()
    print("Permutation importance by group:")
    print(permutation_group.to_string(index=False))
    print()
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
