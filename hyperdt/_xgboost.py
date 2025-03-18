"""
XGBoost implementations for hyperbolic space.

This module provides XGBoost classifiers and regressors that operate
natively in hyperbolic space.
"""

from typing import Any, Optional
import numpy as np

from .tree import HyperbolicDecisionTreeClassifier, HyperbolicDecisionTreeRegressor

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

if XGBOOST_AVAILABLE:
    class HyperbolicXGBoostClassifier(HyperbolicDecisionTreeClassifier):
        """Hyperbolic XGBoost for classification.
        
        Parameters
        ----------
        max_depth : int, default=3
            The maximum depth of the tree.
        curvature : float, default=1.0
            The curvature of the hyperbolic space.
        timelike_dim : int, default=0
            The index of the timelike dimension in the input data.
        n_estimators : int, default=100
            The number of boosting rounds.
        learning_rate : float, default=0.1
            The learning rate.
        skip_hyperboloid_check : bool, default=False
            Whether to skip the validation that points lie on a hyperboloid.
        random_state : int, RandomState instance, default=None
            Controls the randomness of the estimator.
        **kwargs : 
            Additional parameters passed to the XGBClassifier.
        
        Attributes
        ----------
        estimator_ : XGBClassifier
            The underlying fitted XGBoost estimator.
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances from the underlying XGBoost model.
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
            learning_rate: float = 0.1,
            skip_hyperboloid_check: bool = False,
            random_state: Optional[int] = None,
            **kwargs: Any,
        ):
            # Add XGBoost specific parameters
            kwargs["n_estimators"] = n_estimators
            kwargs["learning_rate"] = learning_rate
            
            super().__init__(
                backend="xgboost",
                max_depth=max_depth,
                curvature=curvature,
                timelike_dim=timelike_dim,
                skip_hyperboloid_check=skip_hyperboloid_check,
                random_state=random_state,
                **kwargs,
            )

    class HyperbolicXGBoostRegressor(HyperbolicDecisionTreeRegressor):
        """Hyperbolic XGBoost for regression.
        
        Parameters
        ----------
        max_depth : int, default=3
            The maximum depth of the tree.
        curvature : float, default=1.0
            The curvature of the hyperbolic space.
        timelike_dim : int, default=0
            The index of the timelike dimension in the input data.
        n_estimators : int, default=100
            The number of boosting rounds.
        learning_rate : float, default=0.1
            The learning rate.
        skip_hyperboloid_check : bool, default=False
            Whether to skip the validation that points lie on a hyperboloid.
        random_state : int, RandomState instance, default=None
            Controls the randomness of the estimator.
        **kwargs : 
            Additional parameters passed to the XGBRegressor.
        
        Attributes
        ----------
        estimator_ : XGBRegressor
            The underlying fitted XGBoost estimator.
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances from the underlying XGBoost model.
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
            learning_rate: float = 0.1,
            skip_hyperboloid_check: bool = False,
            random_state: Optional[int] = None,
            **kwargs: Any,
        ):
            # Add XGBoost specific parameters
            kwargs["n_estimators"] = n_estimators
            kwargs["learning_rate"] = learning_rate
            
            super().__init__(
                backend="xgboost",
                max_depth=max_depth,
                curvature=curvature,
                timelike_dim=timelike_dim,
                skip_hyperboloid_check=skip_hyperboloid_check,
                random_state=random_state,
                **kwargs,
            )