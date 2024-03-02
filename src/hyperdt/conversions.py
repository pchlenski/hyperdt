"""Convert between hyperbolic coordinates"""

from typing import Literal

import numpy as np


def _validate_X(X: np.ndarray) -> np.ndarray:
    """Validate X input"""

    # Input validation
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    return X


def _rearrange_X(X: np.ndarray, timelike_dim: int = 0) -> np.ndarray:
    """Put timelike dimension first"""

    # Put timelike dimension first
    X = np.roll(X, timelike_dim, axis=1)

    return X


def _hyperboloid_to_klein(X: np.ndarray, timelike_dim: int = 0) -> np.ndarray:
    """Convert hyperboloid coordinates to Klein coordinates."""

    return _poincare_to_klein(_hyperboloid_to_poincare(X, timelike_dim))


def _hyperboloid_to_poincare(X: np.ndarray, timelike_dim: int = 0) -> np.ndarray:
    """Convert hyperboloid coordinates to Poincare ball coordinates."""
    X = _rearrange_X(X, timelike_dim)

    return X[:, 1:] / (1 + X[:, 0, None])


def _poincare_to_klein(X: np.ndarray) -> np.ndarray:
    """Convert Poincare ball coordinates to Klein coordinates."""
    return 2 * X / (1 + np.linalg.norm(X, axis=1) ** 2)[:, None]


def _poincare_to_hyperboloid(X: np.ndarray) -> np.ndarray:
    """Convert Poincare ball coordinates to hyperboloid coordinates."""
    # Compute squared norm for each sample
    norm_squared = np.linalg.norm(X, axis=1) ** 2
    denominator = 1 - norm_squared

    # Compute X0 for each sample
    X0 = ((1 + norm_squared) / denominator).reshape(-1, 1)

    # Compute xi for each sample
    Xi = 2 * X / denominator[:, None]

    # Concatenate results
    return np.concatenate((X0, Xi), axis=1)


def convert(
    X: np.ndarray,
    initial: Literal["hyperboloid", "poincare"],
    final: Literal["klein", "poincare", "hyperboloid"],
    timelike_dim: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Convert between embeddings.

    Args:
    -----
    X: np.ndarray (n_samples, n_features)
        Data matrix
    initial: str
        Initial coordinates type. One of "hyperboloid" or "poincare"
    final: str
        Final coordinates type. One of "klein", "poincare", or "hyperboloid"
    timelike_dim: int
        Index of timelike dimension, if initial geometry is "hyperboloid." (default: 0)

    Returns:
    --------
    X_new: np.ndarray (n_samples, n_features)
        Data matrix in new coordinates
    """

    # Input validation
    X = _validate_X(X)

    # Convert
    if initial == final:
        X_new = X
    elif initial == "hyperboloid" and final == "klein":
        X_new = _hyperboloid_to_klein(X, timelike_dim=timelike_dim)
    elif initial == "hyperboloid" and final == "poincare":
        X_new = _hyperboloid_to_poincare(X, timelike_dim=timelike_dim)
    elif initial == "poincare" and final == "klein":
        X_new = _poincare_to_klein(X)
    elif initial == "poincare" and final == "hyperboloid":
        X_new = _poincare_to_hyperboloid(X)
    else:
        raise ValueError(f"Cannot convert from {initial} to {final}")

    return X_new
