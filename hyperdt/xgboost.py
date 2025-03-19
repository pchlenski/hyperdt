"""
XGBoost implementations for hyperbolic space.

This module provides XGBoost classifiers and regressors that operate
natively in hyperbolic space.
"""

from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree


class HyperbolicXGBoostClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """This is just HyperbolicDecisionTree with backend="xgboost" and task="classification"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="xgboost", task="classification", *args, **kwargs)


class HyperbolicXGBoostRegressor(HyperbolicDecisionTree, RegressorMixin):
    """This is just HyperbolicDecisionTree with backend="xgboost" and task="regression"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="sklearn_rf", task="regression", *args, **kwargs)
