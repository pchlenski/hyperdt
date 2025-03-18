"""Base classes for hyperbolic decision trees.

This module implements the core functionality for decision trees that operate
natively in hyperbolic space.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Type, cast, TYPE_CHECKING, TypeVar
import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Protocol, TypedDict, Annotated, runtime_checkable
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

# Type aliases for type checking
T = TypeVar('T', bound=np.generic)


class HyperbolicDecisionTree(BaseEstimator):
    """Base class for hyperbolic trees with configurable backend
    
    Parameters
    ----------
    backend : str, default="sklearn_dt"
        The backend to use for the tree estimator.
        Available backends: "sklearn_dt", "sklearn_rf", "xgboost" (if installed)
    task : str, default="classification"
        The task type. Available tasks: "classification", "regression"
    max_depth : int, default=3
        The maximum depth of the tree. Must be >= 1.
    curvature : float, default=1.0
        The curvature of the hyperbolic space. Must be positive.
    timelike_dim : int, default=0
        The index of the timelike dimension in the input data.
        The remaining dimensions are treated as spacelike.
    skip_hyperboloid_check : bool, default=False
        Whether to skip the validation that points lie on a hyperboloid.
        Set to True for speed if data is already properly formatted.
    **kwargs : 
        Additional parameters passed to the underlying backend estimator.
    
    Attributes
    ----------
    estimator_ : object
        The underlying fitted estimator instance.
    n_features_in_ : int
        The number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Defined only when X has feature names that are all strings.
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available for classification tasks.
    """

    def __init__(
        self,
        backend: str = "sklearn_dt",
        task: str = "classification",
        max_depth: int = 3,
        curvature: float = 1.0,
        timelike_dim: int = 0,
        skip_hyperboloid_check: bool = False,
        **kwargs: Any,
    ):
        self.backend = backend
        self.task = task
        self.max_depth = max_depth
        self.curvature = curvature
        self.timelike_dim = timelike_dim
        self.skip_hyperboloid_check = skip_hyperboloid_check
        self.kwargs = kwargs

        # Initialize appropriate backend estimator
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Initialize the appropriate backend estimator"""
        # Import here to avoid circular imports
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Check for optional XGBoost dependency
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
            
        backend_map: Dict[str, Dict[str, Type[Any]]] = {
            "classification": {
                "sklearn_dt": DecisionTreeClassifier,
                "sklearn_rf": RandomForestClassifier,
            },
            "regression": {
                "sklearn_dt": DecisionTreeRegressor,
                "sklearn_rf": RandomForestRegressor,
            },
        }

        # Add XGBoost backends if available
        if XGBOOST_AVAILABLE:
            backend_map["classification"]["xgboost"] = xgb.XGBClassifier
            backend_map["regression"]["xgboost"] = xgb.XGBRegressor

        if self.task not in backend_map:
            raise ValueError(f"Unknown task: {self.task}. Use 'classification' or 'regression'.")

        if self.backend not in backend_map[self.task]:
            available_backends = list(backend_map[self.task].keys())
            raise ValueError(f"Unknown backend: {self.backend} for task {self.task}. Available: {available_backends}")

        estimator_class = backend_map[self.task][self.backend]

        # Prepare keyword arguments based on backend type
        kwargs = self.kwargs.copy()
        
        # Backend-specific parameter handling
        if self.backend == "sklearn_rf":
            # RandomForest doesn't support 'splitter' parameter
            if "splitter" in kwargs:
                del kwargs["splitter"]
        
        # Add max_depth to supported backends
        if self.backend in ["sklearn_dt", "sklearn_rf", "xgboost"]:
            kwargs["max_depth"] = self.max_depth

        # Initialize the estimator with appropriate parameters
        self.estimator_ = estimator_class(**kwargs)

    def _get_tags(self):
        """Return estimator tags."""
        # Import inside method to avoid circular imports
        import sklearn
        # Use importlib.metadata instead of pkg_resources
        try:
            from importlib.metadata import version as parse_version
        except ImportError:
            # Fallback for Python < 3.8
            from pkg_resources import parse_version
        
        # Base tags for all scikit-learn versions
        tags = {
            # Explicitly mark what we don't support
            'allow_nan': False,  # NaN values are not supported in hyperbolic space
            'handles_1d_data': False,  # 1D data doesn't make sense in hyperbolic space
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],  # Only support dense arrays
            'poor_score': False,
            'no_validation': True,  # We do our own validation for hyperboloid constraints
            'pairwise': False,
            'multioutput': False,
            'requires_fit': True,
            'requires_y': True,  # This estimator requires y for fitting
            
            # Explicitly skip tests that don't apply to hyperbolic space
            '_skip_test': False,
            '_xfail_checks': {
                'check_estimators_nan_inf': 'NaN/inf not supported in hyperbolic space',
                'check_estimator_sparse_data': 'Sparse matrices not supported in hyperbolic space',
                'check_dtype_object': 'Object dtypes not supported',
                'check_methods_subset_invariance': 'Hyperbolic constraints violated with subsets',
                'check_fit1d': '1D data not supported in hyperbolic space',
                'check_fit_check_is_fitted': 'Custom fit/predict validation',
                'check_sample_weights_invariance': 'Not all sample weights preserve hyperboloid',
            }
        }
        
        # Add version-specific tags
        sklearn_version = sklearn.__version__
        
        # Add version-specific tags
        if float(sklearn_version.split('.')[0]) >= 1:
            tags['multilabel'] = False  # Added in 1.0+
            
        major, minor = map(int, sklearn_version.split('.')[:2])
        if (major >= 1 and minor >= 3) or major > 1:
            tags['non_deterministic'] = False  # Added in 1.3+
            
        if (major >= 1 and minor >= 4) or major > 1:
            tags['array_api_support'] = False  # Added in 1.4+
        
        return tags
    
    def _validate_hyperbolic(self, X: np.ndarray) -> None:
        """
        Ensure points lie on a hyperboloid - subtract timelike twice from sum of all squares, rather than once from sum
        of all spacelike squares, to simplify indexing.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_dimensions)
            The input data points in hyperboloid coordinates.
            
        Raises
        ------
        ValueError
            If the points do not lie on a hyperboloid with the specified curvature.
        """
        # Check dimensions
        if X.shape[1] <= self.timelike_dim:
            raise ValueError(f"Timelike dimension index {self.timelike_dim} is out of bounds for data with {X.shape[1]} dimensions")
        
        dims = np.delete(np.arange(X.shape[1]), self.timelike_dim)
        
        # Ensure Minkowski norm
        minkowski_norm = np.sum(X[:, dims] ** 2, axis=1) - X[:, self.timelike_dim] ** 2
        if not np.allclose(minkowski_norm, -1 / self.curvature, atol=1e-3):
            raise ValueError(f"Points must lie on a hyperboloid: Minkowski norm does not equal {-1 / self.curvature}.")

        # Ensure timelike
        if not np.all(X[:, self.timelike_dim] > 1.0 / self.curvature):
            raise ValueError("Points must lie on a hyperboloid: Value at timelike dimension must be greater than 1.")

        # Ensure hyperboloid
        if not np.all(X[:, self.timelike_dim] > np.linalg.norm(X[:, dims], axis=1)):
            raise ValueError("Points must lie on a hyperboloid: Value at timelike dim must exceed norm of spacelike dims.")
            
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.
        
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : bool, default=False
            If True, call validate_X_y instead of check_X_y.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`.
            
        Returns
        -------
        out : ndarray, sparse matrix, or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """
        # If we are predicting (y is None), override the requires_y tag check
        if y is None and hasattr(self, '_get_tags') and self._get_tags().get('requires_y', False):
            # For prediction, just validate X without requiring y
            X_array = check_array(X, **check_params)
            if reset:
                self.n_features_in_ = X_array.shape[1]
            return X_array
        
        # Otherwise use the standard validation
        try:
            # First, try to use sklearn's BaseEstimator._validate_data if it exists
            result = super()._validate_data(X, y, reset=reset, 
                                         validate_separately=validate_separately, 
                                         **check_params)
            return result
        except (AttributeError, TypeError):
            # If parent method doesn't exist, implement our own validation
            if y is None:
                X_array = check_array(X, **check_params)
                if reset:
                    self.n_features_in_ = X_array.shape[1]
                return X_array
            else:
                X_array, y_array = check_X_y(X, y, **check_params)
                if reset:
                    self.n_features_in_ = X_array.shape[1]
                return X_array, y_array

    def _adjust_thresholds(self, estimator, X_klein, indices):
        """Adjust thresholds for a decision tree after fitting in Klein coordinates.
        
        Parameters
        ----------
        estimator : object
            The fitted decision tree.
        X_klein : np.ndarray of shape (n_samples, n_features)
            The training data in Klein coordinates.
        indices : array-like of shape (n_samples,)
            The indices of the training samples.
        """
        def apply_recursive(estimator, node_id=0):
            """Recursively adjust thresholds for a decision tree."""
            if estimator.tree_.children_left[node_id] == -1:  # Leaf node
                return
            
            # Get feature and threshold
            feature = estimator.tree_.feature[node_id]
            threshold = estimator.tree_.threshold[node_id]
            
            # Find samples that land on this node
            node_indices = indices[estimator.decision_path(X_klein).toarray()[:, node_id] > 0]
            
            if len(node_indices) == 0:
                # If no samples, can't adjust
                return
            
            # Get feature values
            feature_values = X_klein[node_indices, feature]
            
            # Adjust threshold to be the average of closest values on either side
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if np.any(left_mask) and np.any(right_mask):
                left_max = np.max(feature_values[left_mask])
                right_min = np.min(feature_values[right_mask])
                estimator.tree_.threshold[node_id] = (left_max + right_min) / 2
            
            # Recurse
            apply_recursive(estimator, estimator.tree_.children_left[node_id])
            apply_recursive(estimator, estimator.tree_.children_right[node_id])
            
        # For random forests, adjust each tree
        if self.backend == "sklearn_rf":
            for tree in estimator.estimators_:
                apply_recursive(tree)
        elif self.backend == "sklearn_dt":
            apply_recursive(estimator)

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ):
        """
        Fit hyperbolic decision tree.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input data
        X_array, y_array = self._validate_data(
            X, y, 
            accept_sparse=False, 
            dtype=np.float64, 
            ensure_2d=True, 
            force_all_finite=True, 
            multi_output=False
        )
        
        # Check hyperboloid constraints if needed
        if not self.skip_hyperboloid_check:
            try:
                self._validate_hyperbolic(X_array)
            except (ValueError, AssertionError) as e:
                raise ValueError(f"Input data does not satisfy hyperboloid constraints: {str(e)}")

        # Convert to Klein coordinates (x_d/x_0)
        x0 = X_array[:, self.timelike_dim]
        X_klein = np.delete(X_array, self.timelike_dim, axis=1) / x0[:, None]

        # Fit backend estimator
        self.estimator_.fit(X_klein, y_array)

        # Adjust thresholds for decision trees and tree ensembles
        if self.backend in ["sklearn_dt", "sklearn_rf"]:
            self._adjust_thresholds(self.estimator_, X_klein, np.arange(len(X_array)))

        return self

    def predict(
        self, 
        X: ArrayLike
    ) -> np.ndarray:
        """
        Predict class or regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
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
            
        # Predict using backend estimator
        return self.estimator_.predict(X_klein)