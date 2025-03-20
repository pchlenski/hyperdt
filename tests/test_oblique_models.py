"""
Tests for the Hyperbolic Oblique Decision Tree models.
This test suite verifies that each oblique model type can properly fit and predict data.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hyperdt import OBLIQUE_AVAILABLE
from hyperdt.toy_data import wrapped_normal_mixture

# Skip all tests if oblique trees are not available
pytestmark = pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")

# Import oblique models if available
if OBLIQUE_AVAILABLE:
    from hyperdt import (
        HyperbolicContinuouslyOptimizedClassifier,
        HyperbolicContinuouslyOptimizedRegressor,
        HyperbolicHouseHolderClassifier,
        HyperbolicHouseHolderRegressor,
    )


# Prepare test data
def prepare_test_data(task="classification", n_samples=200, n_features=5, n_classes=3, random_state=42):
    """Generate and prepare test data for classifiers or regressors."""
    # Add 1 to n_features since wrapped_normal_mixture produces data with dimension n+1
    manifold_dim = n_features - 1

    # Generate data
    X, y_class = wrapped_normal_mixture(
        num_points=n_samples, num_classes=n_classes, num_dims=manifold_dim, seed=random_state
    )

    # Use a simple function of the coordinates for regression
    if task == "regression":
        y = np.sin(X[:, 1]) + np.cos(X[:, 2]) + 0.1 * np.random.randn(len(X))
    else:
        y = y_class

    return train_test_split(X, y, test_size=0.2, random_state=random_state)


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_householder_classifier():
    """Test HyperbolicHouseHolderClassifier."""
    print("Testing HyperbolicHouseHolderClassifier...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification")

    # Create and fit model
    clf = HyperbolicHouseHolderClassifier(
        max_depth=5, timelike_dim=0, validate_input_geometry=False, midpoint_method="einstein"
    )
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_householder_regressor():
    """Test HyperbolicHouseHolderRegressor."""
    print("Testing HyperbolicHouseHolderRegressor...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="regression")

    # Create and fit model
    reg = HyperbolicHouseHolderRegressor(
        max_depth=5, timelike_dim=0, validate_input_geometry=False, midpoint_method="einstein"
    )
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_co2_classifier():
    """Test HyperbolicContinuouslyOptimizedClassifier."""
    print("Testing HyperbolicContinuouslyOptimizedClassifier...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification", n_classes=2)

    # Create and fit model
    clf = HyperbolicContinuouslyOptimizedClassifier(
        max_depth=5, timelike_dim=0, validate_input_geometry=False, midpoint_method="einstein"
    )
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_co2_regressor():
    """Test HyperbolicContinuouslyOptimizedRegressor."""
    print("Testing HyperbolicContinuouslyOptimizedRegressor...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="regression")

    # Create and fit model
    reg = HyperbolicContinuouslyOptimizedRegressor(
        max_depth=5, timelike_dim=0, validate_input_geometry=False, midpoint_method="einstein"
    )
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_midpoint_methods():
    """Test different midpoint methods for oblique models."""
    print("Testing different midpoint methods for oblique models...")

    # Generate and prepare data for binary classification
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification", n_classes=2)

    midpoint_used = {}
    for method in ["einstein", "naive", "zero"]:
        # Create and fit model with different midpoint methods
        clf = HyperbolicHouseHolderClassifier(
            max_depth=3, timelike_dim=0, validate_input_geometry=False, midpoint_method=method
        )
        clf.fit(X_train, y_train)

        # Just check that the model can be trained with different midpoint methods
        # Make predictions and verify they're the correct shape
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        midpoint_used[method] = True

    for method in midpoint_used:
        print(f"  {method} midpoint test: passed")

    # Check that all methods were tested
    assert len(midpoint_used) == 3, "All midpoint methods should be tested"


if __name__ == "__main__":
    print("=== Testing Hyperbolic Oblique Decision Tree Models ===")

    if OBLIQUE_AVAILABLE:
        # Test classifiers
        test_householder_classifier()
        test_co2_classifier()

        # Test regressors
        test_householder_regressor()
        test_co2_regressor()

        # Test midpoint methods
        test_midpoint_methods()

        print("\nAll oblique model tests completed successfully!")
    else:
        print("Skipping oblique model tests as scikit-obliquetree is not installed.")
