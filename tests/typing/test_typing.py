"""
Test typing annotations for the HyperDT library.
This file doesn't actually run tests, it just imports things to check type annotations with mypy.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from hyperdt.faster_tree import (
    HyperbolicDecisionTree,
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)

# Import XGBoost types if available
try:
    from hyperdt.faster_tree import (
        HyperbolicXGBoostClassifier,
        HyperbolicXGBoostRegressor,
    )
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Create dummy data with proper shapes for type checking
n_samples = 100
n_features = 5
X = np.random.random((n_samples, n_features))
y_class = np.random.randint(0, 3, size=n_samples)
y_reg = np.random.random(n_samples)

# Test classifier
clf = HyperbolicDecisionTreeClassifier(max_depth=3, curvature=1.0, timelike_dim=0)
clf.fit(X, y_class)
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)

# Test regressor
reg = HyperbolicDecisionTreeRegressor(max_depth=3, curvature=1.0, timelike_dim=0)
reg.fit(X, y_reg)
y_pred_reg = reg.predict(X)

# Test random forest
rf_clf = HyperbolicRandomForestClassifier(
    n_estimators=10, max_depth=3, curvature=1.0, timelike_dim=0
)
rf_clf.fit(X, y_class)
rf_y_pred = rf_clf.predict(X)
rf_y_proba = rf_clf.predict_proba(X)

# Test random forest regressor
rf_reg = HyperbolicRandomForestRegressor(
    n_estimators=10, max_depth=3, curvature=1.0, timelike_dim=0
)
rf_reg.fit(X, y_reg)
rf_y_pred_reg = rf_reg.predict(X)

# Test XGBoost models if available
if XGBOOST_AVAILABLE:
    xgb_clf = HyperbolicXGBoostClassifier(
        n_estimators=10, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0
    )
    xgb_clf.fit(X, y_class)
    xgb_y_pred = xgb_clf.predict(X)
    xgb_y_proba = xgb_clf.predict_proba(X)

    xgb_reg = HyperbolicXGBoostRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1, curvature=1.0, timelike_dim=0
    )
    xgb_reg.fit(X, y_reg)
    xgb_y_pred_reg = xgb_reg.predict(X)