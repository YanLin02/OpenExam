from __future__ import annotations


def gradient_descent_step(w: float, grad: float, lr: float) -> float:
    return w - lr * grad
