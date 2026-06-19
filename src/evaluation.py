from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute robust binary-classification metrics for rare-event detection."""
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    ).ravel()

    metrics: dict[str, float | int] = {
        "rows": int(len(y_true_array)),
        "events": int(y_true_array.sum()),
        "event_rate": float(y_true_array.mean()) if len(y_true_array) else 0.0,
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "balanced_accuracy": balanced_accuracy_score(y_true_array, y_pred_array),
        "precision": precision_score(y_true_array, y_pred_array, zero_division=0),
        "recall": recall_score(y_true_array, y_pred_array, zero_division=0),
        "f1": f1_score(y_true_array, y_pred_array, zero_division=0),
        "macro_f1": f1_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "missed_event_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }

    if y_score is not None:
        y_score_array = np.asarray(y_score, dtype=float)
        metrics["roc_auc"] = _safe_roc_auc(y_true_array, y_score_array)
        metrics["pr_auc"] = _safe_average_precision(y_true_array, y_score_array)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def prediction_scores(estimator, x: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class scores when the estimator exposes them."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities[:, -1]

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x)

    return None


def evaluate_model_on_parts(
    estimator,
    x: pd.DataFrame,
    y: pd.Series,
    parts: dict[str, pd.Index],
    base_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = []
    metadata = base_metadata or {}

    for part_name, index in parts.items():
        x_part = x.loc[index]
        y_part = y.loc[index].astype(int)
        y_pred = estimator.predict(x_part)
        y_score = prediction_scores(estimator, x_part)
        rows.append(
            {
                **metadata,
                "part": part_name,
                **evaluate_binary_classifier(y_part, y_pred, y_score),
            }
        )

    return pd.DataFrame(rows)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

