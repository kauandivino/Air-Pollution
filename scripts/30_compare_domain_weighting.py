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


def load_local_baseline(target: str, model: str) -> pd.DataFrame:
    checkpoint_dir = METRICS_DIR / "loco_local_feature_checkpoints"
    paths = sorted(checkpoint_dir.glob(f"{target}__{model}__*.csv"))
    if not paths:
        raise FileNotFoundError(f"No local-feature checkpoints found for {target} | {model}")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    return data[
        (data["target"] == target)
        & (data["model"] == model)
        & (data["part"] == "test")
    ].copy()


def load_domain_results() -> pd.DataFrame:
    checkpoint_dir = METRICS_DIR / "loco_domain_weighting_checkpoints"
    paths = sorted(checkpoint_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No domain-weighting checkpoints found.")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    return data[data["part"] == "test"].copy()


def paired_comparison(local: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    columns = ["target", "model", "held_out_country", *METRICS]
    merged = local[columns].merge(
        weighted[columns + ["feature_regime", "weighting_strategy"]],
        on=["target", "model", "held_out_country"],
        suffixes=("_local", "_weighted"),
        validate="one_to_one",
    )
    for metric in METRICS:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_weighted"] - merged[f"{metric}_local"]
        )
    return merged


def summarize(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, strategy), group in deltas.groupby(
        ["target", "model", "weighting_strategy"],
        observed=True,
    ):
        for metric in METRICS:
            local = group[f"{metric}_local"]
            weighted = group[f"{metric}_weighted"]
            delta = group[f"{metric}_delta"]
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "weighting_strategy": strategy,
                    "metric": metric,
                    "n_countries": len(group),
                    "local_mean": local.mean(),
                    "weighted_mean": weighted.mean(),
                    "mean_delta": delta.mean(),
                    "median_delta": delta.median(),
                    "relative_delta_pct": (
                        100 * delta.mean() / local.mean() if local.mean() != 0 else np.nan
                    ),
                    "countries_improved": int((delta > 0).sum()),
                    "countries_worse": int((delta < 0).sum()),
                    "countries_equal": int((delta == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_tests(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, strategy), group in deltas.groupby(
        ["target", "model", "weighting_strategy"],
        observed=True,
    ):
        for metric in METRICS:
            values = group[f"{metric}_delta"].dropna().to_numpy()
            if len(values) < 2 or np.allclose(values, 0):
                statistic = p_value = np.nan
            else:
                result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
                statistic = result.statistic
                p_value = result.pvalue
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "weighting_strategy": strategy,
                    "metric": metric,
                    "n_countries": len(values),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "mean_delta": values.mean() if len(values) else np.nan,
                    "median_delta": np.median(values) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_results_dirs()
    weighted = load_domain_results()
    all_deltas = []

    for (target, model, strategy), group in weighted.groupby(
        ["target", "model", "weighting_strategy"],
        observed=True,
    ):
        local = load_local_baseline(target, model)
        all_deltas.append(paired_comparison(local, group))

    if not all_deltas:
        raise RuntimeError("No domain-weighting experiments available for comparison.")

    deltas = pd.concat(all_deltas, ignore_index=True)
    summary = summarize(deltas)
    tests = wilcoxon_tests(deltas)

    delta_path = TABLES_DIR / "domain_weighting_vs_local_delta_by_country.csv"
    summary_path = TABLES_DIR / "domain_weighting_vs_local_summary.csv"
    tests_path = TABLES_DIR / "domain_weighting_vs_local_wilcoxon.csv"

    deltas.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)

    headline = summary[summary["metric"].isin(["pr_auc", "f1", "recall", "precision"])]
    print("Domain weighting vs local features:")
    print(
        headline[
            [
                "target",
                "model",
                "weighting_strategy",
                "metric",
                "local_mean",
                "weighted_mean",
                "mean_delta",
                "countries_improved",
                "countries_worse",
            ]
        ].to_string(index=False)
    )
    print()
    print("Wilcoxon tests:")
    print(
        tests[tests["metric"].isin(["pr_auc", "f1", "recall", "precision"])][
            ["target", "model", "weighting_strategy", "metric", "mean_delta", "p_value"]
        ].to_string(index=False)
    )
    print()
    print(f"Country deltas saved to: {delta_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Wilcoxon tests saved to: {tests_path}")


if __name__ == "__main__":
    main()
