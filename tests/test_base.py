"""Tests for HyperbolicDecisionTree base-class behaviors.

Covers input-geometry handling (hyperboloid / Klein / Poincare), input-geometry
validation, midpoint methods, and error handling that the model-type and
equivalence suites don't exercise.
"""

import numpy as np
import pytest

from hyperdt import HyperbolicDecisionTreeClassifier
from hyperdt._base import HyperbolicDecisionTree
from hyperdt.toy_data import wrapped_normal_mixture


def _hyperboloid_data(n=200, classes=3, dims=4, seed=42):
    """Hyperboloid-embedded data (timelike dimension 0)."""
    return wrapped_normal_mixture(num_points=n, num_classes=classes, num_dims=dims, seed=seed)


def _to_klein(X):
    """Hyperboloid -> Klein (project through the origin onto the t=1 plane)."""
    return X[:, 1:] / X[:, 0:1]


def _to_poincare(X):
    """Hyperboloid -> Poincare ball (stereographic projection from (-1, 0...))."""
    return X[:, 1:] / (1.0 + X[:, 0:1])


def test_validate_hyperboloid_accepts_valid_data():
    """validate_input_geometry=True should accept genuine hyperboloid points."""
    X, y = _hyperboloid_data()
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, timelike_dim=0, validate_input_geometry=True)
    clf.fit(X, y)
    assert clf.predict(X).shape == y.shape


def test_validate_hyperboloid_rejects_offmanifold_data():
    """Points that violate the Minkowski-norm constraint should be rejected."""
    X, y = _hyperboloid_data()
    bad = X.copy()
    bad[:, 0] += 0.5  # break -x0^2 + |x_space|^2 = -1
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, timelike_dim=0, validate_input_geometry=True)
    with pytest.raises(AssertionError):
        clf.fit(bad, y)


def test_klein_input_geometry():
    """A model can be trained directly on Klein-coordinate inputs."""
    X, y = _hyperboloid_data()
    K = _to_klein(X)
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, input_geometry="klein", validate_input_geometry=True)
    clf.fit(K, y)
    assert clf.predict(K).shape == y.shape


def test_poincare_input_geometry():
    """A model can be trained directly on Poincare-ball inputs."""
    X, y = _hyperboloid_data()
    P = _to_poincare(X)
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, input_geometry="poincare", validate_input_geometry=True)
    clf.fit(P, y)
    assert clf.predict(P).shape == y.shape


@pytest.mark.parametrize("method", ["einstein", "naive", "zero", "random"])
def test_midpoint_methods(method):
    """All supported midpoint methods produce a usable model."""
    X, y = _hyperboloid_data()
    clf = HyperbolicDecisionTreeClassifier(
        max_depth=4, timelike_dim=0, validate_input_geometry=False, midpoint_method=method
    )
    clf.fit(X, y)
    assert clf.predict(X).shape == y.shape


def test_unknown_backend_raises():
    """An unrecognized backend is rejected at construction time."""
    with pytest.raises(ValueError):
        HyperbolicDecisionTree(backend="not_a_backend")


def test_unknown_midpoint_method_raises():
    """An unrecognized midpoint method is rejected when thresholds are adjusted."""
    X, y = _hyperboloid_data()
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, validate_input_geometry=False, midpoint_method="bogus")
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_unknown_input_geometry_raises():
    """An unrecognized input geometry is rejected during preprocessing."""
    X, y = _hyperboloid_data()
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, input_geometry="bogus", validate_input_geometry=False)
    with pytest.raises(ValueError):
        clf.fit(X, y)
