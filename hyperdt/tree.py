"""
Decision tree implementations for hyperbolic space.

This module provides decision tree classifiers and regressors that operate
natively in hyperbolic space.
"""

from typing import Any, Optional
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._base import HyperbolicDecisionTree


class HyperbolicDecisionTreeClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """Hyperbolic Decision Tree for classification.
    
    Parameters
    ----------
    backend : str, default="sklearn_dt"
        The backend to use for the tree estimator.
        Available backends: "sklearn_dt", "sklearn_rf", "xgboost" (if installed)
    max_depth : int, default=3
        The maximum depth of the tree.
    curvature : float, default=1.0
        The curvature of the hyperbolic space.
    timelike_dim : int, default=0
        The index of the timelike dimension in the input data.
    criterion : str, default="gini"
        The function to measure the quality of a split. 
        Supported criteria are "gini" for the Gini impurity and "entropy" for information gain.
    splitter : str, default="best"
        The strategy used to choose the split at each node. 
        Supported strategies are "best" to choose the best split and "random" to choose
        the best random split.
    skip_hyperboloid_check : bool, default=False
        Whether to skip the validation that points lie on a hyperboloid.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    **kwargs : 
        Additional parameters passed to the backend estimator.
    
    Attributes
    ----------
    estimator_ : object
        The underlying fitted estimator.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        The number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Only defined if X has feature names.
    """
    
    def __init__(
        self,
        backend: str = "sklearn_dt",
        max_depth: int = 3,
        curvature: float = 1.0,
        timelike_dim: int = 0,
        criterion: str = "gini",
        splitter: str = "best",
        skip_hyperboloid_check: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        # Add criterion and splitter to kwargs for decision tree backends
        kwargs["criterion"] = criterion
        kwargs["splitter"] = splitter
        
        # Add random_state to kwargs if provided
        if random_state is not None:
            kwargs["random_state"] = random_state
            
        super().__init__(
            backend=backend,
            task="classification",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from underlying estimator."""
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.feature_importances_
        
    def predict_proba(
        self, 
        X: ArrayLike
    ) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fitted
        check_is_fitted(self, ["estimator_", "n_features_in_"])
        
        # Validate input data
        X_array = self._validate_data(
            X, 
            accept_sparse=False, 
            dtype=np.float64, 
            ensure_2d=True, 
            force_all_finite=True, 
            reset=False
        )
        
        # Check hyperboloid constraints if needed
        if not self.skip_hyperboloid_check:
            try:
                self._validate_hyperbolic(X_array)
            except (ValueError, AssertionError) as e:
                raise ValueError(f"Input data does not satisfy hyperboloid constraints: {str(e)}")
            
        # Convert to Klein coordinates
        x0 = X_array[:, self.timelike_dim]
        X_klein = np.delete(X_array, self.timelike_dim, axis=1) / x0[:, None]
            
        # Predict probabilities using backend estimator
        return self.estimator_.predict_proba(X_klein)
    
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ):
        """
        Fit hyperbolic decision tree classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)


class HyperbolicDecisionTreeRegressor(HyperbolicDecisionTree, RegressorMixin):
    """Hyperbolic Decision Tree for regression.
    
    Parameters
    ----------
    backend : str, default="sklearn_dt"
        The backend to use for the tree estimator.
        Available backends: "sklearn_dt", "sklearn_rf", "xgboost" (if installed)
    max_depth : int, default=3
        The maximum depth of the tree.
    curvature : float, default=1.0
        The curvature of the hyperbolic space.
    timelike_dim : int, default=0
        The index of the timelike dimension in the input data.
    criterion : str, default="squared_error"
        The function to measure the quality of a split. 
        Supported criteria are "squared_error" for mean squared error,
        "friedman_mse" for Friedman's improvement score, "absolute_error" for mean
        absolute error, and "poisson" for mean Poisson deviance.
    splitter : str, default="best"
        The strategy used to choose the split at each node.
        Supported strategies are "best" to choose the best split and "random" to choose
        the best random split.
    skip_hyperboloid_check : bool, default=False
        Whether to skip the validation that points lie on a hyperboloid.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    **kwargs : 
        Additional parameters passed to the backend estimator.
    
    Attributes
    ----------
    estimator_ : object
        The underlying fitted estimator.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    n_features_in_ : int
        The number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Only defined if X has feature names.
    """
    
    def __init__(
        self,
        backend: str = "sklearn_dt",
        max_depth: int = 3,
        curvature: float = 1.0,
        timelike_dim: int = 0,
        criterion: str = "squared_error",
        splitter: str = "best",
        skip_hyperboloid_check: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        # Add criterion and splitter to kwargs for decision tree backends
        kwargs["criterion"] = criterion
        kwargs["splitter"] = splitter
        
        # Add random_state to kwargs if provided
        if random_state is not None:
            kwargs["random_state"] = random_state
            
        super().__init__(
            backend=backend,
            task="regression",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from underlying estimator."""
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.feature_importances_
    
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ):
        """
        Fit hyperbolic decision tree regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates.
        y : array-like of shape (n_samples,)
            The target values (real numbers).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)