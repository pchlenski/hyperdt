"""
Ensemble implementations for hyperbolic space.

This module provides random forest classifiers and regressors that operate
natively in hyperbolic space.
"""

from typing import Any, Optional
import numpy as np

from .tree import HyperbolicDecisionTreeClassifier, HyperbolicDecisionTreeRegressor


class HyperbolicRandomForestClassifier(HyperbolicDecisionTreeClassifier):
    """Hyperbolic Random Forest for classification.

    Parameters
    ----------
    max_depth : int, default=3
        The maximum depth of the tree.
    curvature : float, default=1.0
        The curvature of the hyperbolic space.
    timelike_dim : int, default=0
        The index of the timelike dimension in the input data.
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : str, default="gini"
        The function to measure the quality of a split.
    skip_hyperboloid_check : bool, default=False
        Whether to skip the validation that points lie on a hyperboloid.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    **kwargs :
        Additional parameters passed to the sklearn RandomForestClassifier.

    Attributes
    ----------
    estimator_ : RandomForestClassifier
        The underlying fitted forest estimator.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances from the underlying forest.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        The number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Only defined if X has feature names.
    """

    def __init__(
        self,
        max_depth: int = 3,
        curvature: float = 1.0,
        timelike_dim: int = 0,
        n_estimators: int = 100,
        criterion: str = "gini",
        skip_hyperboloid_check: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        # Add n_estimators to kwargs
        kwargs["n_estimators"] = n_estimators

        # Don't pass splitter parameter to RandomForestClassifier as it doesn't accept it
        # Construct kwargs dict without passing splitter for RandomForest models
        # Note: We intentionally don't pass splitter here since RandomForest doesn't support it
        #       It will be filtered in _init_estimator anyway
        super().__init__(
            backend="sklearn_rf",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            criterion=criterion,
            skip_hyperboloid_check=skip_hyperboloid_check,
            random_state=random_state,
            **kwargs,
        )


class HyperbolicRandomForestRegressor(HyperbolicDecisionTreeRegressor):
    """Hyperbolic Random Forest for regression.

    Parameters
    ----------
    max_depth : int, default=3
        The maximum depth of the tree.
    curvature : float, default=1.0
        The curvature of the hyperbolic space.
    timelike_dim : int, default=0
        The index of the timelike dimension in the input data.
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : str, default="squared_error"
        The function to measure the quality of a split.
    skip_hyperboloid_check : bool, default=False
        Whether to skip the validation that points lie on a hyperboloid.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    **kwargs :
        Additional parameters passed to the sklearn RandomForestRegressor.

    Attributes
    ----------
    estimator_ : RandomForestRegressor
        The underlying fitted forest estimator.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances from the underlying forest.
    n_features_in_ : int
        The number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Only defined if X has feature names.
    """

    def __init__(
        self,
        max_depth: int = 3,
        curvature: float = 1.0,
        timelike_dim: int = 0,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        skip_hyperboloid_check: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        # Add n_estimators to kwargs
        kwargs["n_estimators"] = n_estimators

        # Don't pass splitter parameter to RandomForestRegressor as it doesn't accept it
        # Construct kwargs dict without passing splitter for RandomForest models
        # Note: We intentionally don't pass splitter here since RandomForest doesn't support it
        #       It will be filtered in _init_estimator anyway
        super().__init__(
            backend="sklearn_rf",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            criterion=criterion,
            skip_hyperboloid_check=skip_hyperboloid_check,
            random_state=random_state,
            **kwargs,
        )
