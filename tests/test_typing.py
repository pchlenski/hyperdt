"""
Test typing annotations for the HyperDT library.
This file contains tests to check that the interfaces work as expected.
"""

import pytest
import numpy as np

from hyperdt import (
    HyperbolicDecisionTree,
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
    HyperbolicXGBoostClassifier,
    HyperbolicXGBoostRegressor,
)


# Create a fixture for the data
@pytest.fixture
def hyperbolic_data():
    # Generate data that satisfies hyperboloid constraints
    n_samples = 100
    n_features = 5

    # First generate points in ambient space (excluding timelike dimension)
    np.random.seed(42)  # For reproducibility
    X_ambient = np.random.randn(n_samples, n_features - 1)
    # Compute timelike coordinate to place points on hyperboloid (x₀² - x₁² - ... - xₙ² = 1)
    spacelike_norm_squared = np.sum(X_ambient**2, axis=1)
    timelike = np.sqrt(spacelike_norm_squared + 1.0)
    # Combine to form hyperboloid points
    X = np.column_stack([timelike, X_ambient])

    y_class = np.random.randint(0, 3, size=n_samples)
    y_reg = np.random.random(n_samples)

    return X, y_class, y_reg


def test_decision_tree_classifier(hyperbolic_data):
    """Test that HyperbolicDecisionTreeClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test classifier
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True)
    clf.fit(X, y_class)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    assert y_pred.shape == y_class.shape
    assert y_proba.shape[0] == y_class.shape[0]
    assert clf.feature_importances_ is not None


def test_decision_tree_regressor(hyperbolic_data):
    """Test that HyperbolicDecisionTreeRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test regressor
    reg = HyperbolicDecisionTreeRegressor(max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True)
    reg.fit(X, y_reg)
    y_pred_reg = reg.predict(X)

    assert y_pred_reg.shape == y_reg.shape
    assert reg.feature_importances_ is not None


def test_random_forest_classifier(hyperbolic_data):
    """Test that HyperbolicRandomForestClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test random forest
    rf_clf = HyperbolicRandomForestClassifier(
        n_estimators=3, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True
    )
    rf_clf.fit(X, y_class)
    rf_y_pred = rf_clf.predict(X)
    rf_y_proba = rf_clf.predict_proba(X)

    assert rf_y_pred.shape == y_class.shape
    assert rf_y_proba.shape[0] == y_class.shape[0]
    assert rf_clf.feature_importances_ is not None


def test_random_forest_regressor(hyperbolic_data):
    """Test that HyperbolicRandomForestRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test random forest regressor
    rf_reg = HyperbolicRandomForestRegressor(
        n_estimators=3, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True
    )
    rf_reg.fit(X, y_reg)
    rf_y_pred_reg = rf_reg.predict(X)

    assert rf_y_pred_reg.shape == y_reg.shape
    assert rf_reg.feature_importances_ is not None


def test_xgboost_classifier(hyperbolic_data):
    """Test that HyperbolicXGBoostClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test XGBoost
    xgb_clf = HyperbolicXGBoostClassifier(
        n_estimators=3, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True
    )
    xgb_clf.fit(X, y_class)
    xgb_y_pred = xgb_clf.predict(X)
    xgb_y_proba = xgb_clf.predict_proba(X)

    assert xgb_y_pred.shape == y_class.shape
    assert xgb_y_proba.shape[0] == y_class.shape[0]
    assert xgb_clf.feature_importances_ is not None


def test_xgboost_regressor(hyperbolic_data):
    """Test that HyperbolicXGBoostRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test XGBoost regressor
    xgb_reg = HyperbolicXGBoostRegressor(
        n_estimators=3, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True
    )
    xgb_reg.fit(X, y_reg)
    xgb_y_pred_reg = xgb_reg.predict(X)

    assert xgb_y_pred_reg.shape == y_reg.shape
    assert xgb_reg.feature_importances_ is not None
