from __future__ import annotations


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def classification_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    if min(tp, fp, tn, fn) < 0:
        raise ValueError("confusion matrix counts must be non-negative.")
    accuracy = _safe_divide(tp + tn, tp + fp + tn + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
