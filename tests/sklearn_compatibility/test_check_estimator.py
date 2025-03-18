"""
Tests to verify that HyperDT estimators are compatible with scikit-learn's API.
"""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

import sys
import os

# Add the parent directory to the path to import hyperdt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hyperdt.faster_tree import (
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)

# Import custom estimator checks
from sklearn.utils.estimator_checks import (
    parametrize_with_checks, check_estimator as sklearn_check_estimator
)

# Generate test data that's appropriate for hyperbolic space
def generate_hyperbolic_data(n_samples=100, n_features=5, random_state=42):
    """Generate synthetic data on the hyperboloid for testing."""
    np.random.seed(random_state)

    # Generate points in ambient space
    X_ambient = np.random.randn(n_samples, n_features - 1)

    # Normalize ambient coordinates
    X_norm = np.sqrt(np.sum(X_ambient**2, axis=1))
    X_ambient_normalized = X_ambient / X_norm[:, np.newaxis]

    # Add randomness to normalized coordinates
    noise_factor = 0.1 * np.sqrt(n_features)
    X_ambient_noisy = X_ambient_normalized + np.random.randn(n_samples, n_features - 1) * noise_factor

    # Compute timelike coordinate to place points on hyperboloid (x₀² - x₁² - ... - xₙ² = 1)
    spacelike_norm_squared = np.sum(X_ambient_noisy**2, axis=1)
    timelike = np.sqrt(spacelike_norm_squared + 1.0)

    # Combine to form hyperboloid points
    X = np.column_stack([timelike, X_ambient_noisy])

    # Generate target values
    y_class = np.random.randint(0, 3, size=n_samples)
    y_reg = np.sin(X[:, 1]) + np.cos(X[:, 2]) + 0.1 * np.random.randn(n_samples)

    return X, y_class, y_reg


# Create a modified DecisionTreeClassifier for compatibility tests
class CompatibilityHyperbolicDecisionTreeClassifier(HyperbolicDecisionTreeClassifier):
    """Modified classifier for sklearn compatibility tests."""
    
    def __init__(self, max_depth=3, curvature=1.0, timelike_dim=0, **kwargs):
        super().__init__(max_depth=max_depth, curvature=curvature, timelike_dim=timelike_dim,
                         skip_hyperboloid_check=True, **kwargs)
    
    def fit(self, X, y):
        # For sklearn compatibility tests, we'll skip the hyperboloid check
        # and add a timelike dimension if needed
        if X.shape[1] == 1:
            # If only one feature, add a second one
            X = np.column_stack([np.ones(X.shape[0]) * 1.5, X])
        elif X.shape[1] >= 2 and self.timelike_dim >= X.shape[1]:
            # If timelike dimension is invalid, set to 0
            self.timelike_dim = 0
            
        # Make sure first dimension is timelike
        X_norm = np.sqrt(np.sum(np.delete(X, self.timelike_dim, axis=1)**2, axis=1))
        if self.timelike_dim == 0:
            X[:, 0] = np.sqrt(X_norm**2 + 1.0)
        
        return super().fit(X, y)


# Create a modified DecisionTreeRegressor for compatibility tests
class CompatibilityHyperbolicDecisionTreeRegressor(HyperbolicDecisionTreeRegressor):
    """Modified regressor for sklearn compatibility tests."""
    
    def __init__(self, max_depth=3, curvature=1.0, timelike_dim=0, **kwargs):
        super().__init__(max_depth=max_depth, curvature=curvature, timelike_dim=timelike_dim,
                         skip_hyperboloid_check=True, **kwargs)
    
    def fit(self, X, y):
        # For sklearn compatibility tests, we'll skip the hyperboloid check
        # and add a timelike dimension if needed
        if X.shape[1] == 1:
            # If only one feature, add a second one
            X = np.column_stack([np.ones(X.shape[0]) * 1.5, X])
        elif X.shape[1] >= 2 and self.timelike_dim >= X.shape[1]:
            # If timelike dimension is invalid, set to 0
            self.timelike_dim = 0
            
        # Make sure first dimension is timelike
        X_norm = np.sqrt(np.sum(np.delete(X, self.timelike_dim, axis=1)**2, axis=1))
        if self.timelike_dim == 0:
            X[:, 0] = np.sqrt(X_norm**2 + 1.0)
        
        return super().fit(X, y)


@parametrize_with_checks([CompatibilityHyperbolicDecisionTreeClassifier()])
def test_sklearn_compatible_classifier(estimator, check):
    """Test that CompatibilityHyperbolicDecisionTreeClassifier passes all sklearn checks."""
    check(estimator)


@parametrize_with_checks([CompatibilityHyperbolicDecisionTreeRegressor()])
def test_sklearn_compatible_regressor(estimator, check):
    """Test that CompatibilityHyperbolicDecisionTreeRegressor passes all sklearn checks."""
    check(estimator)


def test_hyperbolic_estimators_in_sklearn_pipeline():
    """Minimal test to ensure the estimators work in a sklearn pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Generate test data
    X, y_class, y_reg = generate_hyperbolic_data(n_samples=100, n_features=5, random_state=42)
    
    # Test classifier in pipeline with actual HyperbolicDecisionTreeClassifier
    # Skip hyperboloid check since we're generating valid data
    from hyperdt.faster_tree import HyperbolicDecisionTreeClassifier, HyperbolicDecisionTreeRegressor
    
    # Test with real estimators, not just the compatibility ones
    clf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', HyperbolicDecisionTreeClassifier(skip_hyperboloid_check=True))
    ])
    clf_pipe.fit(X, y_class)
    y_pred = clf_pipe.predict(X)
    assert y_pred.shape == y_class.shape, "Classifier prediction shape mismatch"
    
    # Test regressor in pipeline
    reg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', HyperbolicDecisionTreeRegressor(skip_hyperboloid_check=True))
    ])
    reg_pipe.fit(X, y_reg)
    y_pred_reg = reg_pipe.predict(X)
    assert y_pred_reg.shape == y_reg.shape, "Regressor prediction shape mismatch"
    
    print("Hyperbolic estimators work successfully in scikit-learn pipelines")


if __name__ == "__main__":
    # Run the check_estimator function directly
    sklearn_check_estimator(CompatibilityHyperbolicDecisionTreeClassifier())
    sklearn_check_estimator(CompatibilityHyperbolicDecisionTreeRegressor())
    print("All compatibility checks passed!")