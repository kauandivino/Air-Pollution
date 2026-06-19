from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, TABLES_DIR, ensure_results_dirs


EXPERIMENTS = [
    ("extreme_abs_151_h1", "xgboost"),
    ("extreme_abs_151_h3", "xgboost"),
    ("extreme_abs_151_h1", "random_forest"),
]

METRICS = [
    "pr_auc",
    "roc_auc",
    "f1",
    "recall",
    "precision",
    "balanced_accuracy",
    "false_alarm_rate",
    "missed_event_rate",
]


def load_base_results(target: str, model: str) -> pd.DataFrame:
    path = METRICS_DIR / f"loco_models_{target}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing base LOCO metrics: {path}")

    data = pd.read_csv(path)
    data = data[
        (data["target"] == target)
        & (data["model"] == model)
        & (data["part"] == "test")
    ].copy()
    data["feature_regime"] = "base"
    return data


def load_local_results(target: str, model: str) -> pd.DataFrame:
    path = METRICS_DIR / "loco_local_features_metrics.csv"
    if path.exists():
        data = pd.read_csv(path)
        data = data[
            (data["target"] == target)
            & (data["model"] == model)
            & (data["part"] == "test")
        ].copy()
    else:
        data = pd.DataFrame()

    if data.empty:
        checkpoint_dir = METRICS_DIR / "loco_local_feature_checkpoints"
        checkpoint_paths = sorted(checkpoint_dir.glob(f"{target}__{model}__*.csv"))
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"Missing local rows for {target} | {model} in {path} "
                f"and no checkpoints found in {checkpoint_dir}"
            )
        data = pd.concat(
            [pd.read_csv(checkpoint_path) for checkpoint_path in checkpoint_paths],
            ignore_index=True,
        )
        data = data[
            (data["target"] == target)
            & (data["model"] == model)
            & (data["part"] == "test")
        ].copy()

    data["feature_regime"] = "local"
    return data


def paired_table(base: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    columns = ["target", "model", "held_out_country", *METRICS]
    merged = base[columns].merge(
        local[columns],
        on=["target", "model", "held_out_country"],
        suffixes=("_base", "_local"),
        validate="one_to_one",
    )

    for metric in METRICS:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_local"] - merged[f"{metric}_base"]
        )
    return merged


def summarize_pair(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        base_values = deltas[f"{metric}_base"]
        local_values = deltas[f"{metric}_local"]
        delta_values = deltas[f"{metric}_delta"]
        rows.append(
            {
                "target": deltas["target"].iloc[0],
                "model": deltas["model"].iloc[0],
                "metric": metric,
                "n_countries": len(deltas),
                "base_mean": base_values.mean(),
                "local_mean": local_values.mean(),
                "mean_delta": delta_values.mean(),
                "median_delta": delta_values.median(),
                "relative_delta_pct": (
                    100.0 * delta_values.mean() / base_values.mean()
                    if base_values.mean() != 0
                    else np.nan
                ),
                "countries_improved": int((delta_values > 0).sum()),
                "countries_worse": int((delta_values < 0).sum()),
                "countries_equal": int((delta_values == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_tests(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        delta_values = deltas[f"{metric}_delta"].dropna().to_numpy()
        if len(delta_values) < 2 or np.allclose(delta_values, 0):
            statistic = np.nan
            p_value = np.nan
        else:
            result = wilcoxon(delta_values, alternative="two-sided", zero_method="wilcox")
            statistic = result.statistic
            p_value = result.pvalue

        rows.append(
            {
                "target": deltas["target"].iloc[0],
                "model": deltas["model"].iloc[0],
                "metric": metric,
                "n_countries": len(delta_values),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "mean_delta": np.mean(delta_values) if len(delta_values) else np.nan,
                "median_delta": np.median(delta_values) if len(delta_values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    deltas: pd.DataFrame,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    rows = []
    for metric in METRICS:
        values = deltas[f"{metric}_delta"].dropna().to_numpy()
        if len(values) == 0:
            lower = upper = np.nan
        else:
            indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
            boot_means = values[indices].mean(axis=1)
            lower, upper = np.percentile(boot_means, [lower_q, upper_q])

        rows.append(
            {
                "target": deltas["target"].iloc[0],
                "model": deltas["model"].iloc[0],
                "metric": metric,
                "mean_delta": values.mean() if len(values) else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence": confidence,
                "n_resamples": n_resamples,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_results_dirs()

    all_deltas = []
    all_summaries = []
    all_tests = []
    all_cis = []

    for target, model in EXPERIMENTS:
        base = load_base_results(target, model)
        local = load_local_results(target, model)

        if base.empty:
            print(f"Skipping {target} | {model}: no base rows found.")
            continue
        if local.empty:
            print(f"Skipping {target} | {model}: no local rows found.")
            continue

        deltas = paired_table(base, local)
        if len(deltas) != 24:
            print(
                f"Warning: {target} | {model} has {len(deltas)} paired countries "
                "instead of 24."
            )

        all_deltas.append(deltas)
        all_summaries.append(summarize_pair(deltas))
        all_tests.append(wilcoxon_tests(deltas))
        all_cis.append(bootstrap_ci(deltas))

    if not all_deltas:
        raise RuntimeError("No paired base/local experiments were available.")

    delta_by_country = pd.concat(all_deltas, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    tests = pd.concat(all_tests, ignore_index=True)
    ci = pd.concat(all_cis, ignore_index=True)

    delta_path = TABLES_DIR / "local_features_base_vs_local_delta_by_country.csv"
    summary_path = TABLES_DIR / "local_features_base_vs_local_summary.csv"
    tests_path = TABLES_DIR / "local_features_base_vs_local_wilcoxon.csv"
    ci_path = TABLES_DIR / "local_features_base_vs_local_bootstrap_ci.csv"

    delta_by_country.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    ci.to_csv(ci_path, index=False)

    headline = summary[summary["metric"].isin(["pr_auc", "f1"])].copy()
    headline = headline[
        [
            "target",
            "model",
            "metric",
            "base_mean",
            "local_mean",
            "mean_delta",
            "relative_delta_pct",
            "countries_improved",
            "countries_worse",
        ]
    ].sort_values(["target", "model", "metric"])

    print("\nBase vs local feature comparison:")
    print(headline.to_string(index=False))
    print(f"\nSaved country deltas to: {delta_path}")
    print(f"Saved summaries to: {summary_path}")
    print(f"Saved Wilcoxon tests to: {tests_path}")
    print(f"Saved bootstrap CIs to: {ci_path}")


if __name__ == "__main__":
    main()
