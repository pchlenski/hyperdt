"""
Ensemble implementations for hyperbolic space.

This module provides decision tree classifiers and regressors that operate
natively in hyperbolic space.
"""

from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree


class HyperbolicRandomForestClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """This is just HyperbolicDecisionTree with backend="sklearn_rf" and task="classification"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="sklearn_rf", task="classification", *args, **kwargs)


class HyperbolicRandomForestRegressor(HyperbolicDecisionTree, RegressorMixin):
    """This is just HyperbolicDecisionTree with backend="sklearn_rf" and task="regression"."""

    def __init__(self, *args, **kwargs):
        super().__init__(backend="sklearn_rf", task="regression", *args, **kwargs)
