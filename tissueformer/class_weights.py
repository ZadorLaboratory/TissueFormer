"""Reusable class weight computation for imbalanced classification tasks."""

import numpy as np


def calculate_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
    beta: float = 0.999,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Calculate class weights based on label frequencies.

    Args:
        labels: Array of integer labels (can be non-consecutive).
        method: Weighting method:
            - "inverse": 1/frequency, normalized so weights sum to n_classes.
            - "balanced": sklearn-style  n_samples / (n_classes * counts).
            - "effective": Effective number of samples (1-beta)/(1-beta^count).
        beta: Parameter for the "effective" method.
        n_classes: Total number of classes (size of the returned array).
            Defaults to max(labels) + 1.

    Returns:
        Array of shape (n_classes,) with per-class weights.
    """
    n_samples = len(labels)
    if n_classes is None:
        n_classes_total = int(max(labels)) + 1
    else:
        n_classes_total = n_classes

    counts = np.bincount(labels, minlength=n_classes_total)
    present_classes = counts > 0
    n_classes = np.sum(present_classes)

    if method == "inverse":
        with np.errstate(divide="ignore"):
            weights = 1 / counts
        weights[counts == 0] = 0
        weights = weights * (n_classes / weights.sum())

    elif method == "balanced":
        with np.errstate(divide="ignore"):
            weights = n_samples / (n_classes * counts)
        weights[counts == 0] = 0

    elif method == "effective":
        weights = np.zeros_like(counts, dtype=float)
        non_zero = counts > 0
        weights[non_zero] = (1 - beta) / (1 - beta ** counts[non_zero])
        weights = weights * (n_classes / weights.sum())

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Handle classes with zero samples
    zero_sample_classes = np.where(counts == 0)[0]
    if len(zero_sample_classes) > 0:
        print(f"Warning: Classes {zero_sample_classes} have no samples in the training set")
        print(f"Setting their weights to max weight: {np.max(weights[counts > 0]):.3f}")
        weights[zero_sample_classes] = np.max(weights[counts > 0])

    return weights
