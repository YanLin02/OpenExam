from __future__ import annotations

import math


def softmax(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("values must not be empty.")
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    denominator = sum(exp_values)
    return [value / denominator for value in exp_values]


def cross_entropy_from_probs(probs: list[float], target_index: int) -> float:
    if not probs:
        raise ValueError("probs must not be empty.")
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range.")
    epsilon = 1e-12
    target_prob = min(max(probs[target_index], epsilon), 1.0)
    return -math.log(target_prob)


def cross_entropy_from_logits(logits: list[float], target_index: int) -> float:
    return cross_entropy_from_probs(softmax(logits), target_index)


def mse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        raise ValueError("y_true must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred, strict=True)) / len(y_true)
