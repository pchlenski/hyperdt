"""
Tests for the HyperbolicDecisionTree models with different backends.
This test suite verifies that each model type can properly fit and predict data.
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hyperdt import (
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
    HyperbolicXGBoostClassifier,
    HyperbolicXGBoostRegressor,
)
from hyperdt.toy_data import wrapped_normal_mixture


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


def test_decision_tree_classifier():
    """Test HyperbolicDecisionTreeClassifier."""
    print("Testing HyperbolicDecisionTreeClassifier...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification")

    # Create and fit model
    clf = HyperbolicDecisionTreeClassifier(max_depth=5, timelike_dim=0, validate_input_geometry=False)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y_train))), "Wrong probability shape"


def test_decision_tree_regressor():
    """Test HyperbolicDecisionTreeRegressor."""
    print("Testing HyperbolicDecisionTreeRegressor...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="regression")

    # Create and fit model
    reg = HyperbolicDecisionTreeRegressor(max_depth=5, timelike_dim=0, validate_input_geometry=False)
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


def test_random_forest_classifier():
    """Test HyperbolicRandomForestClassifier."""
    print("Testing HyperbolicRandomForestClassifier...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification")

    # Create and fit model with fewer trees for faster testing
    clf = HyperbolicRandomForestClassifier(n_estimators=10, max_depth=3, timelike_dim=0, validate_input_geometry=False)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y_train))), "Wrong probability shape"


def test_random_forest_regressor():
    """Test HyperbolicRandomForestRegressor."""
    print("Testing HyperbolicRandomForestRegressor...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="regression")

    # Create and fit model with fewer trees for faster testing
    reg = HyperbolicRandomForestRegressor(n_estimators=10, max_depth=3, timelike_dim=0, validate_input_geometry=False)
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


def test_xgboost_classifier():
    """Test HyperbolicXGBoostClassifier if available."""
    print("Testing HyperbolicXGBoostClassifier...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="classification")

    # Create and fit model with fewer trees for faster testing
    clf = HyperbolicXGBoostClassifier(
        n_estimators=10, max_depth=3, learning_rate=0.1, timelike_dim=0, validate_input_geometry=False
    )
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y_train))), "Wrong probability shape"


def test_xgboost_regressor():
    """Test HyperbolicXGBoostRegressor if available."""
    print("Testing HyperbolicXGBoostRegressor...")

    # Generate and prepare data
    X_train, X_test, y_train, y_test = prepare_test_data(task="regression")

    # Create and fit model with fewer trees for faster testing
    reg = HyperbolicXGBoostRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1, timelike_dim=0, validate_input_geometry=False
    )
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


if __name__ == "__main__":
    print("=== Testing HyperbolicDecisionTree Models ===")

    # Test classifiers
    test_decision_tree_classifier()
    test_random_forest_classifier()
    test_xgboost_classifier()

    # Test regressors
    test_decision_tree_regressor()
    test_random_forest_regressor()
    test_xgboost_regressor()

    print("\nAll tests completed successfully!")
