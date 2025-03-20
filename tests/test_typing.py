"""
Test typing annotations for the HyperDT library.
This file contains tests to check that the interfaces work as expected.
"""

import numpy as np
import pytest

from hyperdt import (
    OBLIQUE_AVAILABLE,
    HyperbolicDecisionTree,
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)
from hyperdt.toy_data import wrapped_normal_mixture
from hyperdt.xgboost import HyperbolicXGBoostClassifier, HyperbolicXGBoostRegressor

# Import oblique models if available
if OBLIQUE_AVAILABLE:
    from hyperdt import (
        HyperbolicContinuouslyOptimizedClassifier,
        HyperbolicContinuouslyOptimizedRegressor,
        HyperbolicHouseHolderClassifier,
        HyperbolicHouseHolderRegressor,
    )


# Create a fixture for the data
@pytest.fixture
def hyperbolic_data():
    """Generate hyperbolic data for testing."""
    # Generate data that satisfies hyperboloid constraints
    n_samples = 100
    n_features = 5
    manifold_dim = n_features - 1

    # Use the wrapped_normal_mixture function from toy_data
    X, y_class = wrapped_normal_mixture(num_points=n_samples, num_classes=3, num_dims=manifold_dim, seed=42)

    # Create regression targets
    y_reg = np.sin(X[:, 1]) + np.cos(X[:, 2]) + 0.1 * np.random.randn(len(X))

    return X, y_class, y_reg


def test_hyperbolic_decision_tree(hyperbolic_data):
    """Test that base HyperbolicDecisionTree works."""
    X, y_class, _ = hyperbolic_data

    # Test base class with default settings (classification)
    base_tree = HyperbolicDecisionTree(max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False)
    base_tree.fit(X, y_class)
    y_pred = base_tree.predict(X)

    assert y_pred.shape == y_class.shape


def test_decision_tree_classifier(hyperbolic_data):
    """Test that HyperbolicDecisionTreeClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test classifier
    clf = HyperbolicDecisionTreeClassifier(max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False)
    clf.fit(X, y_class)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    assert y_pred.shape == y_class.shape
    assert y_proba.shape[0] == y_class.shape[0]
    assert hasattr(clf.estimator_, "feature_importances_")


def test_decision_tree_regressor(hyperbolic_data):
    """Test that HyperbolicDecisionTreeRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test regressor
    reg = HyperbolicDecisionTreeRegressor(max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False)
    reg.fit(X, y_reg)
    y_pred_reg = reg.predict(X)

    assert y_pred_reg.shape == y_reg.shape
    assert hasattr(reg.estimator_, "feature_importances_")


def test_random_forest_classifier(hyperbolic_data):
    """Test that HyperbolicRandomForestClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test random forest
    rf_clf = HyperbolicRandomForestClassifier(
        n_estimators=3, max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    rf_clf.fit(X, y_class)
    rf_y_pred = rf_clf.predict(X)
    rf_y_proba = rf_clf.predict_proba(X)

    assert rf_y_pred.shape == y_class.shape
    assert rf_y_proba.shape[0] == y_class.shape[0]
    assert hasattr(rf_clf.estimator_, "feature_importances_")


def test_random_forest_regressor(hyperbolic_data):
    """Test that HyperbolicRandomForestRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test random forest regressor
    rf_reg = HyperbolicRandomForestRegressor(
        n_estimators=3, max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    rf_reg.fit(X, y_reg)
    rf_y_pred_reg = rf_reg.predict(X)

    assert rf_y_pred_reg.shape == y_reg.shape
    assert hasattr(rf_reg.estimator_, "feature_importances_")


def test_xgboost_classifier(hyperbolic_data):
    """Test that HyperbolicXGBoostClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test XGBoost
    xgb_clf = HyperbolicXGBoostClassifier(
        n_estimators=3, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    xgb_clf.fit(X, y_class)
    xgb_y_pred = xgb_clf.predict(X)
    xgb_y_proba = xgb_clf.predict_proba(X)

    assert xgb_y_pred.shape == y_class.shape
    assert xgb_y_proba.shape[0] == y_class.shape[0]
    assert hasattr(xgb_clf.estimator_, "feature_importances_")


def test_xgboost_regressor(hyperbolic_data):
    """Test that HyperbolicXGBoostRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test XGBoost regressor
    xgb_reg = HyperbolicXGBoostRegressor(
        n_estimators=3, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    xgb_reg.fit(X, y_reg)
    xgb_y_pred_reg = xgb_reg.predict(X)

    assert xgb_y_pred_reg.shape == y_reg.shape
    assert hasattr(xgb_reg.estimator_, "feature_importances_")


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_householder_classifier(hyperbolic_data):
    """Test that HyperbolicHouseHolderClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test HouseHolder classifier
    hh_clf = HyperbolicHouseHolderClassifier(max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False)
    hh_clf.fit(X, y_class)
    hh_y_pred = hh_clf.predict(X)

    assert hh_y_pred.shape == y_class.shape


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_householder_regressor(hyperbolic_data):
    """Test that HyperbolicHouseHolderRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test HouseHolder regressor
    hh_reg = HyperbolicHouseHolderRegressor(max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False)
    hh_reg.fit(X, y_reg)
    hh_y_pred_reg = hh_reg.predict(X)

    assert hh_y_pred_reg.shape == y_reg.shape


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_co2_classifier(hyperbolic_data):
    """Test that HyperbolicContinuouslyOptimizedClassifier works."""
    X, y_class, _ = hyperbolic_data

    # Test CO2 classifier
    co2_clf = HyperbolicContinuouslyOptimizedClassifier(
        max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    co2_clf.fit(X, y_class)
    co2_y_pred = co2_clf.predict(X)

    assert co2_y_pred.shape == y_class.shape


@pytest.mark.skipif(not OBLIQUE_AVAILABLE, reason="scikit-obliquetree not installed")
def test_co2_regressor(hyperbolic_data):
    """Test that HyperbolicContinuouslyOptimizedRegressor works."""
    X, _, y_reg = hyperbolic_data

    # Test CO2 regressor
    co2_reg = HyperbolicContinuouslyOptimizedRegressor(
        max_depth=3, curvature=1.0, timelike_dim=0, validate_input_geometry=False
    )
    co2_reg.fit(X, y_reg)
    co2_y_pred_reg = co2_reg.predict(X)

    assert co2_y_pred_reg.shape == y_reg.shape
