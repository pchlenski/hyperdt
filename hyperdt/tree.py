"""
Decision tree implementations for hyperbolic space.

This module provides decision tree classifiers and regressors that operate
natively in hyperbolic space.
"""

from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree


class HyperbolicDecisionTreeClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """This is just HyperbolicDecisionTree with backend="sklearn_dt" and task="classification"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="sklearn_dt", task="classification", *args, **kwargs)


class HyperbolicDecisionTreeRegressor(HyperbolicDecisionTree, RegressorMixin):
    """This is just HyperbolicDecisionTree with backend="sklearn_dt" and task="regression"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="sklearn_dt", task="regression", *args, **kwargs)
