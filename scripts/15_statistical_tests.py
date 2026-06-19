from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, TABLES_DIR, ensure_results_dirs


TARGETS = ["extreme_abs_151_h1", "extreme_abs_151_h3"]
MODELS = ["majority_class", "logistic_regression", "decision_tree", "random_forest"]
METRICS = ["pr_auc", "f1", "recall", "balanced_accuracy"]
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 5000


def load_loco_metrics() -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        path = METRICS_DIR / f"loco_models_{target}_test.csv"
        frame = pd.read_csv(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired_model_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, metric in itertools.product(TARGETS, METRICS):
        target_df = df[df["target"] == target]
        wide = target_df.pivot_table(
            index="held_out_country",
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        for model_a, model_b in itertools.combinations(MODELS, 2):
            paired = wide[[model_a, model_b]].dropna()
            if paired.empty:
                continue
            diff = paired[model_a] - paired[model_b]
            rows.append(
                {
                    "test": "wilcoxon_paired_models",
                    "target": target,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": len(paired),
                    "mean_a": paired[model_a].mean(),
                    "mean_b": paired[model_b].mean(),
                    "mean_diff_a_minus_b": diff.mean(),
                    "median_diff_a_minus_b": diff.median(),
                    **_safe_wilcoxon(diff),
                }
            )
    return pd.DataFrame(rows)


def paired_horizon_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, metric in itertools.product(MODELS, METRICS):
        model_df = df[df["model"] == model]
        wide = model_df.pivot_table(
            index="held_out_country",
            columns="target",
            values=metric,
            aggfunc="mean",
        )
        paired = wide[TARGETS].dropna()
        if paired.empty:
            continue
        diff = paired[TARGETS[0]] - paired[TARGETS[1]]
        rows.append(
            {
                "test": "wilcoxon_h1_vs_h3",
                "metric": metric,
                "model": model,
                "n_pairs": len(paired),
                "h1_mean": paired[TARGETS[0]].mean(),
                "h3_mean": paired[TARGETS[1]].mean(),
                "mean_diff_h1_minus_h3": diff.mean(),
                "median_diff_h1_minus_h3": diff.median(),
                **_safe_wilcoxon(diff),
            }
        )
    return pd.DataFrame(rows)


def friedman_model_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, metric in itertools.product(TARGETS, METRICS):
        target_df = df[df["target"] == target]
        wide = target_df.pivot_table(
            index="held_out_country",
            columns="model",
            values=metric,
            aggfunc="mean",
        )[MODELS].dropna()
        if len(wide) < 2:
            continue
        statistic, p_value = friedmanchisquare(*(wide[model] for model in MODELS))
        ranks = wide.rank(axis=1, ascending=False, method="average")
        row = {
            "test": "friedman_models",
            "target": target,
            "metric": metric,
            "n_countries": len(wide),
            "statistic": statistic,
            "p_value": p_value,
        }
        for model in MODELS:
            row[f"mean_rank_{model}"] = ranks[model].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for (target, model), group in df.groupby(["target", "model"], observed=True):
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            boot_means = np.empty(BOOTSTRAP_ITERATIONS)
            for i in range(BOOTSTRAP_ITERATIONS):
                sample = rng.choice(values, size=len(values), replace=True)
                boot_means[i] = sample.mean()
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "ci_lower_95": np.quantile(boot_means, 0.025),
                    "ci_upper_95": np.quantile(boot_means, 0.975),
                }
            )
    return pd.DataFrame(rows)


def summarize_key_results(
    model_tests: pd.DataFrame,
    horizon_tests: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
) -> pd.DataFrame:
    key_model_tests = model_tests[
        (model_tests["metric"].isin(["pr_auc", "f1"]))
        & (
            model_tests["model_a"].eq("random_forest")
            | model_tests["model_b"].eq("random_forest")
        )
    ].copy()
    key_horizon_tests = horizon_tests[
        (horizon_tests["metric"].isin(["pr_auc", "f1", "recall"]))
        & (horizon_tests["model"].isin(["logistic_regression", "decision_tree", "random_forest"]))
    ].copy()
    key_ci = bootstrap_ci[
        (bootstrap_ci["metric"].isin(["pr_auc", "f1", "recall"]))
        & (bootstrap_ci["model"].isin(["logistic_regression", "decision_tree", "random_forest"]))
    ].copy()

    key_model_tests.to_csv(TABLES_DIR / "stat_tests_key_model_wilcoxon.csv", index=False)
    key_horizon_tests.to_csv(TABLES_DIR / "stat_tests_key_horizon_wilcoxon.csv", index=False)
    key_ci.to_csv(TABLES_DIR / "stat_tests_key_bootstrap_ci.csv", index=False)

    return key_horizon_tests


def _safe_wilcoxon(diff: pd.Series) -> dict[str, float]:
    diff = diff.dropna()
    if len(diff) == 0 or np.allclose(diff.to_numpy(), 0):
        return {"statistic": np.nan, "p_value": np.nan}
    statistic, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(statistic), "p_value": float(p_value)}


def main() -> None:
    ensure_results_dirs()
    df = load_loco_metrics()

    model_tests = paired_model_wilcoxon(df)
    horizon_tests = paired_horizon_wilcoxon(df)
    friedman_tests = friedman_model_tests(df)
    bootstrap_ci = bootstrap_confidence_intervals(df)

    model_tests.to_csv(TABLES_DIR / "stat_tests_model_wilcoxon.csv", index=False)
    horizon_tests.to_csv(TABLES_DIR / "stat_tests_horizon_wilcoxon.csv", index=False)
    friedman_tests.to_csv(TABLES_DIR / "stat_tests_friedman_models.csv", index=False)
    bootstrap_ci.to_csv(TABLES_DIR / "stat_tests_bootstrap_ci.csv", index=False)

    key_horizon = summarize_key_results(model_tests, horizon_tests, bootstrap_ci)

    print("Statistical tests completed.")
    print()
    print("Friedman tests:")
    print(friedman_tests.to_string(index=False))
    print()
    print("Key h1 vs h3 Wilcoxon tests:")
    print(
        key_horizon[
            [
                "metric",
                "model",
                "n_pairs",
                "h1_mean",
                "h3_mean",
                "mean_diff_h1_minus_h3",
                "p_value",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
