from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_results_dirs
from src.domain_weighting import domain_weights
from src.evaluation import prediction_scores
from src.models import get_model_specs
from src.splits import iter_leave_one_country_out_splits, validate_leave_one_country_integrity


LOCAL_FEATURE_MATRIX = PROCESSED_DATA_DIR / "feature_matrix_global_local.csv"
TARGET = "extreme_abs_151_h3"
MODEL = "xgboost"
REGIMES = ["local_features", "country_class_balanced"]
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operational threshold analysis for the best LOCO model.")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--regimes", nargs="+", default=REGIMES)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_feature_columns() -> list[str]:
    base = pd.read_csv(TABLES_DIR / "feature_catalog.csv")["feature"].drop_duplicates().tolist()
    local = pd.read_csv(TABLES_DIR / "local_feature_catalog.csv")["feature"].drop_duplicates().tolist()
    return base + local


def load_modeling_frame(target: str, feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Country", "Date", "City", target, *feature_columns]
    usecols = list(dict.fromkeys(usecols))
    return pd.read_csv(LOCAL_FEATURE_MATRIX, usecols=usecols, parse_dates=["Date"])


def checkpoint_path(target: str, model: str, regime: str, country: str) -> Path:
    safe_country = country.replace(" ", "_").replace("/", "_")
    return (
        PREDICTIONS_DIR
        / "operational_best_model_checkpoints"
        / f"{target}__{model}__{regime}__{safe_country}.csv"
    )


def fit_estimator(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    regime: str,
    split,
):
    spec = get_model_specs([model_name])[0]
    estimator = clone(spec.estimator)
    x = df[feature_columns]
    y = df[target].astype(int)

    if regime == "country_class_balanced":
        sample_weight, _ = domain_weights(
            df=df,
            train_index=split.train_idx,
            target=target,
            strategy="country_class_balanced",
        )
        estimator.fit(
            x.loc[split.train_idx],
            y.loc[split.train_idx],
            **{"model__sample_weight": sample_weight.to_numpy()},
        )
    elif regime == "local_features":
        estimator.fit(x.loc[split.train_idx], y.loc[split.train_idx])
    else:
        raise ValueError(f"Unknown operational regime: {regime}")

    return estimator


def predict_one(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    regime: str,
    split,
) -> pd.DataFrame:
    estimator = fit_estimator(
        df=df,
        feature_columns=feature_columns,
        target=target,
        model_name=model_name,
        regime=regime,
        split=split,
    )
    x_test = df.loc[split.test_idx, feature_columns]
    y_test = df.loc[split.test_idx, target].astype(int)
    scores = prediction_scores(estimator, x_test)
    if scores is None:
        raise RuntimeError(f"Model does not expose prediction scores: {model_name}")

    test_meta = df.loc[split.test_idx, ["Country", "City", "Date"]].copy()
    test_meta["target"] = target
    test_meta["model"] = model_name
    test_meta["regime"] = regime
    test_meta["held_out_country"] = split.metadata["held_out_country"]
    test_meta["y_true"] = y_test.to_numpy()
    test_meta["y_score"] = scores
    return test_meta


def generate_predictions(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_name: str,
    regimes: list[str],
    resume: bool,
) -> pd.DataFrame:
    checkpoint_dir = PREDICTIONS_DIR / "operational_best_model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    target_df = df[df[target].notna()].copy()
    for split in iter_leave_one_country_out_splits(target_df, target):
        validate_leave_one_country_integrity(target_df, split)
        country = split.metadata["held_out_country"]
        for regime in regimes:
            path = checkpoint_path(target, model_name, regime, country)
            if resume and path.exists():
                print(f"Skipping {path.name}: checkpoint exists.")
                continue
            print(f"Training operational model | {target} | {model_name} | {regime} | {country}")
            predictions = predict_one(
                df=target_df,
                feature_columns=feature_columns,
                target=target,
                model_name=model_name,
                regime=regime,
                split=split,
            )
            predictions.to_csv(path, index=False)

    paths = []
    for regime in regimes:
        paths.extend(sorted(checkpoint_dir.glob(f"{target}__{model_name}__{regime}__*.csv")))
    if not paths:
        raise RuntimeError("No operational prediction checkpoints found.")
    predictions = pd.concat((pd.read_csv(path, parse_dates=["Date"]) for path in paths), ignore_index=True)
    predictions = predictions[
        (predictions["target"] == target)
        & (predictions["model"] == model_name)
        & (predictions["regime"].isin(regimes))
    ]
    return predictions


def threshold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, regime), group in predictions.groupby(["target", "model", "regime"], observed=True):
        y_true = group["y_true"].astype(int).to_numpy()
        y_score = group["y_score"].astype(float).to_numpy()
        for threshold in THRESHOLDS:
            y_pred = (y_score >= threshold).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "regime": regime,
                    "threshold": threshold,
                    "rows": len(group),
                    "events": int(y_true.sum()),
                    "predicted_alerts": int(y_pred.sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_alarm_rate": false_alarm_rate,
                    "false_alerts_per_100_city_months": 100 * fp / len(group),
                    "alerts_per_100_city_months": 100 * y_pred.sum() / len(group),
                    "missed_events_per_100_events": 100 * fn / y_true.sum() if y_true.sum() else 0.0,
                }
            )
    return pd.DataFrame(rows)


def select_operational_points(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    recall_targets = [0.50, 0.60, 0.70, 0.80]
    for (target, model, regime), group in curves.groupby(["target", "model", "regime"], observed=True):
        for recall_target in recall_targets:
            candidates = group[group["recall"] >= recall_target].copy()
            if candidates.empty:
                continue
            selected = candidates.sort_values(
                ["false_alerts_per_100_city_months", "threshold"],
                ascending=[True, False],
            ).iloc[0]
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "regime": regime,
                    "recall_target": recall_target,
                    "threshold": selected["threshold"],
                    "precision": selected["precision"],
                    "recall": selected["recall"],
                    "f1": selected["f1"],
                    "false_alarm_rate": selected["false_alarm_rate"],
                    "false_alerts_per_100_city_months": selected["false_alerts_per_100_city_months"],
                    "alerts_per_100_city_months": selected["alerts_per_100_city_months"],
                }
            )
    return pd.DataFrame(rows)


def save_figures(curves: pd.DataFrame, pr_points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    palette = {"local_features": "#466b8f", "country_class_balanced": "#b23a48"}

    plt.figure(figsize=(8.5, 6))
    sns.lineplot(
        data=curves,
        x="false_alerts_per_100_city_months",
        y="recall",
        hue="regime",
        palette=palette,
        linewidth=2.2,
    )
    plt.title("Recall vs falsos alertas por 100 cidade-mes")
    plt.xlabel("Falsos alertas por 100 cidade-mes")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "operational_recall_vs_false_alerts_best_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    sns.lineplot(
        data=pr_points,
        x="recall",
        y="precision",
        hue="regime",
        palette=palette,
        linewidth=2.2,
    )
    plt.title("Curva precision-recall operacional")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "operational_precision_recall_best_model.png", dpi=180)
    plt.close()

    fixed = curves[curves["threshold"] == 0.50].copy()
    fixed = fixed.melt(
        id_vars=["regime"],
        value_vars=["precision", "recall", "f1", "false_alerts_per_100_city_months"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9.5, 5.5))
    sns.barplot(data=fixed, x="metric", y="value", hue="regime", palette=palette)
    plt.title("Operacao em threshold padrao 0.50")
    plt.xlabel("")
    plt.ylabel("Valor")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "operational_threshold_050_best_model.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    feature_columns = load_feature_columns()
    df = load_modeling_frame(args.target, feature_columns)
    predictions = generate_predictions(
        df=df,
        feature_columns=feature_columns,
        target=args.target,
        model_name=args.model,
        regimes=args.regimes,
        resume=not args.no_resume,
    )

    predictions_path = PREDICTIONS_DIR / "operational_best_model_predictions.csv"
    curves_path = TABLES_DIR / "operational_threshold_curves_best_model.csv"
    pr_path = TABLES_DIR / "operational_precision_recall_points_best_model.csv"
    operating_points_path = TABLES_DIR / "operational_recall_targets_best_model.csv"

    curves = threshold_metrics(predictions)
    pr_points = curves[
        ["target", "model", "regime", "threshold", "precision", "recall"]
    ].copy()
    operating_points = select_operational_points(curves)

    predictions.to_csv(predictions_path, index=False)
    curves.to_csv(curves_path, index=False)
    pr_points.to_csv(pr_path, index=False)
    operating_points.to_csv(operating_points_path, index=False)
    save_figures(curves, pr_points)

    print("Operational analysis completed.")
    print()
    print("Threshold 0.50 comparison:")
    print(
        curves[curves["threshold"] == 0.50][
            [
                "regime",
                "precision",
                "recall",
                "f1",
                "false_alarm_rate",
                "false_alerts_per_100_city_months",
                "alerts_per_100_city_months",
            ]
        ].to_string(index=False)
    )
    print()
    print("Operating points by recall target:")
    print(operating_points.to_string(index=False))
    print()
    print(f"Predictions saved to: {predictions_path}")
    print(f"Threshold curves saved to: {curves_path}")
    print(f"Precision-recall points saved to: {pr_path}")
    print(f"Operating points saved to: {operating_points_path}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
