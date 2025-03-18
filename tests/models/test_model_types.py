"""
Tests for the refactored HyperbolicDecisionTree models with different backends.
This test suite verifies that each model type can properly fit and predict data.
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import os

# Add the parent directory to the path to import hyperdt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hyperdt import (
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
    XGBOOST_AVAILABLE,
)

# Check if XGBoost is available
if XGBOOST_AVAILABLE:
    import xgboost as xgb
    from hyperdt import HyperbolicXGBoostClassifier, HyperbolicXGBoostRegressor


# Generate synthetic hyperbolic data for testing
def generate_hyperbolic_data(n_samples=200, n_features=5, n_classes=3, task="classification", random_state=None):
    """Generate synthetic data on the hyperboloid for testing."""
    if random_state is not None:
        np.random.seed(random_state)

    # Generate points in ambient space
    X_ambient = np.random.randn(n_samples, n_features - 1)

    # Normalize ambient coordinates
    X_norm = np.sqrt(np.sum(X_ambient**2, axis=1))
    X_ambient_normalized = X_ambient / X_norm[:, np.newaxis]

    # Add randomness proportional to feature count
    noise_factor = 0.1 * np.sqrt(n_features)
    X_ambient_noisy = X_ambient_normalized + np.random.randn(n_samples, n_features - 1) * noise_factor

    # Compute timelike coordinate to place points on hyperboloid (x₀² - x₁² - ... - xₙ² = 1)
    spacelike_norm_squared = np.sum(X_ambient_noisy**2, axis=1)
    timelike = np.sqrt(spacelike_norm_squared + 1.0)

    # Combine to form hyperboloid points
    X = np.column_stack([timelike, X_ambient_noisy])

    # Generate target values
    if task == "classification":
        y = np.random.randint(0, n_classes, size=n_samples)
    else:  # regression
        # Use a simple function of the coordinates for regression
        y = np.sin(X[:, 1]) + np.cos(X[:, 2]) + 0.1 * np.random.randn(n_samples)

    return X, y


def test_decision_tree_classifier():
    """Test HyperbolicDecisionTreeClassifier."""
    print("Testing HyperbolicDecisionTreeClassifier...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, n_classes=3, task="classification", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model
    clf = HyperbolicDecisionTreeClassifier(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Just check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y))), "Wrong probability shape"


def test_decision_tree_regressor():
    """Test HyperbolicDecisionTreeRegressor."""
    print("Testing HyperbolicDecisionTreeRegressor...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, task="regression", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model
    reg = HyperbolicDecisionTreeRegressor(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Just check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


def test_random_forest_classifier():
    """Test HyperbolicRandomForestClassifier."""
    print("Testing HyperbolicRandomForestClassifier...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, n_classes=3, task="classification", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model with fewer trees for faster testing
    clf = HyperbolicRandomForestClassifier(n_estimators=10, max_depth=3, timelike_dim=0, skip_hyperboloid_check=True)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Just check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y))), "Wrong probability shape"


def test_random_forest_regressor():
    """Test HyperbolicRandomForestRegressor."""
    print("Testing HyperbolicRandomForestRegressor...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, task="regression", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model with fewer trees for faster testing
    reg = HyperbolicRandomForestRegressor(n_estimators=10, max_depth=3, timelike_dim=0, skip_hyperboloid_check=True)
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Just check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"


def test_xgboost_classifier():
    """Test HyperbolicXGBoostClassifier if available."""
    if not XGBOOST_AVAILABLE:
        print("Skipping XGBoost classifier test (XGBoost not installed)")
        return

    print("Testing HyperbolicXGBoostClassifier...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, n_classes=3, task="classification", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model with fewer trees for faster testing
    clf = HyperbolicXGBoostClassifier(
        n_estimators=10, max_depth=3, learning_rate=0.1, timelike_dim=0, skip_hyperboloid_check=True
    )
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Probability shape: {y_pred_proba.shape}")

    # Just check that the model runs and makes predictions
    assert y_pred.shape == y_test.shape, "Wrong prediction shape"
    assert y_pred_proba.shape == (X_test.shape[0], len(np.unique(y))), "Wrong probability shape"


def test_xgboost_regressor():
    """Test HyperbolicXGBoostRegressor if available."""
    if not XGBOOST_AVAILABLE:
        print("Skipping XGBoost regressor test (XGBoost not installed)")
        return

    print("Testing HyperbolicXGBoostRegressor...")

    # Generate data
    X, y = generate_hyperbolic_data(n_samples=200, n_features=5, task="regression", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit model with fewer trees for faster testing
    reg = HyperbolicXGBoostRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1, timelike_dim=0, skip_hyperboloid_check=True
    )
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Mean Squared Error: {mse:.4f}")

    # Just check that the model runs and makes predictions
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
