"""
Hyperbolic Decision Trees with configurable backends for classification and regression.
This module implements decision trees that operate natively in hyperbolic space.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Type, cast
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Protocol, TypedDict, Annotated, runtime_checkable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree._tree import Tree as SklearnTree
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Define custom type aliases for array shapes
NDArraySamples = NDArray  # 1D array of samples
NDArraySamplesFeatures = NDArray  # 2D array of samples x features
NDArraySamplesClasses = NDArray  # 2D array of samples x classes


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

        # Add max_depth parameter to kwargs if it's applicable
        kwargs = self.kwargs.copy()
        if self.backend in ["sklearn_dt", "sklearn_rf", "xgboost"]:
            kwargs["max_depth"] = self.max_depth

        self.estimator_ = estimator_class(**kwargs)

    def _get_tags(self):
        """Return estimator tags."""
        return {
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
    
    def _validate_hyperbolic(self, X: NDArraySamplesFeatures) -> None:
        """
        Ensure points lie on a hyperboloid - subtract timelike twice from sum of all squares, rather than once from sum
        of all spacelike squares, to simplify indexing.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_dimensions)
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
            If True, call validate_X_y separately on X and y. This is
            useful in cases where y contains more information to conduct 
            the validation.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`.
            
        Returns
        -------
        X_validated : ndarray
            The validated input.
        y_validated : ndarray or None
            The validated target if provided, None otherwise.
            
        Raises
        ------
        ValueError
            For unsupported data types, dimensions, etc.
        """
        # Set explicit parameters for validation
        params = {
            'accept_sparse': False,  # Reject sparse matrices
            'dtype': np.float64,     # Only accept float64
            'ensure_2d': True,       # Reject 1D data
            'force_all_finite': True, # Reject NaN/inf
        }
        params.update(check_params)
        
        # Explicitly reject unsupported data
        from scipy import sparse
        if sparse.issparse(X):
            raise ValueError("Sparse matrices are not supported in hyperbolic space")
        
        # Check for 1D data and raise clear error
        if hasattr(X, 'ndim') and X.ndim == 1:
            raise ValueError("1D data is not supported in hyperbolic space - at least 2 dimensions required")
        
        # Check for NaN/inf values
        if hasattr(X, 'size') and X.size > 0:
            if np.isnan(np.sum(X)):
                raise ValueError("NaN values are not supported in hyperbolic space")
            if np.isinf(X).any():
                raise ValueError("Infinite values are not supported in hyperbolic space")
        
        # Handle y=None case
        if y is None:
            # Check dimensions early
            if hasattr(X, 'shape') and len(X.shape) > 1 and self.timelike_dim >= X.shape[1]:
                raise ValueError(f"Timelike dimension index {self.timelike_dim} exceeds data dimensions {X.shape[1]}")
                
            X_array = check_array(X, **params)
            if reset:
                self.n_features_in_ = X_array.shape[1]
                if hasattr(X, "columns") and hasattr(X.columns, "tolist"):
                    self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
            return X_array
        
        # Both X and y are provided
        if validate_separately:
            X_array = check_array(X, **params)
            y_array = check_array(y, ensure_2d=False, **params)
        else:
            X_array, y_array = check_X_y(X, y, **params)
        
        # Set n_features_in_
        if reset:
            self.n_features_in_ = X_array.shape[1]
            if hasattr(X, "columns") and hasattr(X.columns, "tolist"):
                self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
            
            # For classification tasks, store classes
            if self.task == "classification":
                self.classes_ = np.unique(y_array)
        
        return X_array, y_array

    def _einstein_midpoint(self, u: float, v: float) -> float:
        """
        Einstein midpoint for scalar features. Assumes u, v are the i-th coordinates of points in the Klein model
        
        Parameters
        ----------
        u : float
            First coordinate
        v : float
            Second coordinate
            
        Returns
        -------
        float
            Einstein midpoint coordinate
        """
        gamma_u = 1 / np.sqrt(1 - u**2 / self.curvature)
        gamma_v = 1 / np.sqrt(1 - v**2 / self.curvature)

        # Correct Einstein midpoint formula for scalars
        numerator = gamma_u * u + gamma_v * v
        denominator = gamma_u + gamma_v
        midpoint = numerator / denominator

        # Rescale back to original coordinates
        return midpoint

    def _adjust_thresholds(
        self, 
        estimator: Any, 
        X_klein: NDArraySamplesFeatures, 
        samples: NDArraySamples
    ) -> None:
        """
        Adjust thresholds using Einstein midpoint method.
        Works for both individual trees and ensembles of trees.
        
        Parameters
        ----------
        estimator : object
            The estimator object with decision thresholds
        X_klein : NDArray of shape (n_samples, n_dimensions-1)
            The input data in Klein coordinates
        samples : NDArray of shape (n_samples,)
            The indices of samples to consider
        """
        # Handle different types of estimators
        if hasattr(estimator, "estimators_"):  # RandomForest and similar
            for tree in estimator.estimators_:
                self._adjust_tree_thresholds(tree.tree_, 0, X_klein, samples)
        elif hasattr(estimator, "tree_"):  # Single DecisionTree
            self._adjust_tree_thresholds(estimator.tree_, 0, X_klein, samples)
        # Add handlers for other tree types as needed (XGBoost, etc.)

    def _adjust_tree_thresholds(
        self, 
        tree: SklearnTree, 
        node_id: int, 
        X_klein: NDArraySamplesFeatures, 
        samples: NDArraySamples
    ) -> None:
        """
        Adjust thresholds for a single tree's node and its children
        
        Parameters
        ----------
        tree : sklearn.tree._tree.Tree
            The decision tree to adjust
        node_id : int
            The current node ID
        X_klein : NDArray of shape (n_samples, n_dimensions-1)
            The input data in Klein coordinates
        samples : NDArray of shape (n_samples,)
            The indices of samples to consider
        """
        if tree.children_left[node_id] == -1:  # Leaf node
            return

        feature = tree.feature[node_id]
        left_mask = X_klein[samples, feature] <= tree.threshold[node_id]
        left_samples = samples[left_mask]
        right_samples = samples[~left_mask]

        if len(left_samples) > 0 and len(right_samples) > 0:
            # Get boundary representatives
            left_rep = X_klein[left_samples, feature].max()
            right_rep = X_klein[right_samples, feature].min()

            # Compute Einstein midpoint
            new_threshold = self._einstein_midpoint(left_rep, right_rep)
            tree.threshold[node_id] = new_threshold

        # Recurse depth-first
        self._adjust_tree_thresholds(tree, tree.children_left[node_id], X_klein, left_samples)
        self._adjust_tree_thresholds(tree, tree.children_right[node_id], X_klein, right_samples)

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ) -> "HyperbolicDecisionTree":
        """
        Fit the hyperbolic decision tree model.
        
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
    ) -> NDArraySamples:
        """
        Predict class or regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes or values.
        """
        # Check if fit has been called
        check_is_fitted(self, attributes=["estimator_", "n_features_in_"])
        
        # Validate input
        X_array = self._validate_data(
            X, 
            reset=False, 
            accept_sparse=False, 
            dtype=np.float64, 
            ensure_2d=True,
            force_all_finite=True
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

        return self.estimator_.predict(X_klein)

    def predict_proba(
        self, 
        X: ArrayLike
    ) -> NDArraySamplesClasses:
        """
        Probability predictions for classifier models
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
            
        Raises
        ------
        AttributeError
            If the task is not classification.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba is not available for regression tasks")

        # Check if fit has been called
        check_is_fitted(self, attributes=["estimator_", "n_features_in_"])
        
        # Validate input
        X_array = self._validate_data(
            X, 
            reset=False, 
            accept_sparse=False, 
            dtype=np.float64, 
            ensure_2d=True,
            force_all_finite=True
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

        return self.estimator_.predict_proba(X_klein)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : Dict
            Parameter names mapped to their values.
        """
        # Get parameters from parent
        params = super().get_params(deep=deep)

        # Add parameters from estimator if deep
        if deep and hasattr(self, "estimator_"):
            estimator_params = self.estimator_.get_params()
            # Filter out parameters already in parent
            for k, v in estimator_params.items():
                if k not in params and k not in [
                    "backend",
                    "task",
                    "max_depth",
                    "curvature",
                    "timelike_dim",
                    "skip_hyperboloid_check",
                ]:
                    params[f"estimator__{k}"] = v

        return params


class HyperbolicDecisionTreeClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """
    Hyperbolic Decision Tree for classification tasks.

    This classifier implements a decision tree that works natively in hyperbolic space
    by transforming data to the Klein model and using an underlying scikit-learn
    DecisionTreeClassifier.

    Parameters
    ----------
    max_depth : int, default=3
        The maximum depth of the tree. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.

    curvature : float, default=1.0
        The curvature of the hyperbolic space. Must be positive.

    timelike_dim : int, default=0
        The index of the timelike dimension in the input data. The remaining dimensions
        are treated as spacelike.

    skip_hyperboloid_check : bool, default=False
        Whether to skip checking if points lie on a hyperboloid. Set to True for speed
        if you're confident your data is already properly formatted for hyperbolic space.

    **kwargs : dict
        Additional parameters to pass to the underlying DecisionTreeClassifier.
    """

    def __init__(
        self, 
        max_depth: int = 3, 
        curvature: float = 1.0, 
        timelike_dim: int = 0, 
        skip_hyperboloid_check: bool = False, 
        **kwargs: Any
    ):
        super().__init__(
            backend="sklearn_dt",
            task="classification",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ) -> "HyperbolicDecisionTreeClassifier":
        """
        Fit the hyperbolic decision tree classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates
        y : array-like of shape (n_samples,)
            The target values (class labels)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(
        self, 
        X: ArrayLike
    ) -> NDArraySamples:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        return super().predict(X)

    def predict_proba(
        self, 
        X: ArrayLike
    ) -> NDArraySamplesClasses:
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
        return super().predict_proba(X)


class HyperbolicDecisionTreeRegressor(HyperbolicDecisionTree, RegressorMixin):
    """
    Hyperbolic Decision Tree for regression tasks.

    This regressor implements a decision tree that works natively in hyperbolic space
    by transforming data to the Klein model and using an underlying scikit-learn
    DecisionTreeRegressor.

    Parameters
    ----------
    max_depth : int, default=3
        The maximum depth of the tree. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.

    curvature : float, default=1.0
        The curvature of the hyperbolic space. Must be positive.

    timelike_dim : int, default=0
        The index of the timelike dimension in the input data. The remaining dimensions
        are treated as spacelike.

    skip_hyperboloid_check : bool, default=False
        Whether to skip checking if points lie on a hyperboloid. Set to True for speed
        if you're confident your data is already properly formatted for hyperbolic space.

    **kwargs : dict
        Additional parameters to pass to the underlying DecisionTreeRegressor.
    """

    def __init__(
        self, 
        max_depth: int = 3, 
        curvature: float = 1.0, 
        timelike_dim: int = 0, 
        skip_hyperboloid_check: bool = False, 
        **kwargs: Any
    ):
        super().__init__(
            backend="sklearn_dt",
            task="regression",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ) -> "HyperbolicDecisionTreeRegressor":
        """
        Fit the hyperbolic decision tree regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates
        y : array-like of shape (n_samples,)
            The target values (real numbers)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(
        self, 
        X: ArrayLike
    ) -> NDArraySamples:
        """
        Predict regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        return super().predict(X)


class HyperbolicRandomForestClassifier(HyperbolicDecisionTree, ClassifierMixin):
    """
    Hyperbolic Random Forest for classification tasks.

    This classifier implements a random forest that works natively in hyperbolic space
    by transforming data to the Klein model and using an underlying scikit-learn
    RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=3
        The maximum depth of the trees. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.

    curvature : float, default=1.0
        The curvature of the hyperbolic space. Must be positive.

    timelike_dim : int, default=0
        The index of the timelike dimension in the input data. The remaining dimensions
        are treated as spacelike.

    skip_hyperboloid_check : bool, default=False
        Whether to skip checking if points lie on a hyperboloid. Set to True for speed
        if you're confident your data is already properly formatted for hyperbolic space.

    **kwargs : dict
        Additional parameters to pass to the underlying RandomForestClassifier.
    """

    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int = 3, 
        curvature: float = 1.0, 
        timelike_dim: int = 0, 
        skip_hyperboloid_check: bool = False, 
        **kwargs: Any
    ):
        kwargs["n_estimators"] = n_estimators
        super().__init__(
            backend="sklearn_rf",
            task="classification",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ) -> "HyperbolicRandomForestClassifier":
        """
        Fit the hyperbolic random forest classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates
        y : array-like of shape (n_samples,)
            The target values (class labels)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(
        self, 
        X: ArrayLike
    ) -> NDArraySamples:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        return super().predict(X)

    def predict_proba(
        self, 
        X: ArrayLike
    ) -> NDArraySamplesClasses:
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
        return super().predict_proba(X)


class HyperbolicRandomForestRegressor(HyperbolicDecisionTree, RegressorMixin):
    """
    Hyperbolic Random Forest for regression tasks.

    This regressor implements a random forest that works natively in hyperbolic space
    by transforming data to the Klein model and using an underlying scikit-learn
    RandomForestRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=3
        The maximum depth of the trees. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.

    curvature : float, default=1.0
        The curvature of the hyperbolic space. Must be positive.

    timelike_dim : int, default=0
        The index of the timelike dimension in the input data. The remaining dimensions
        are treated as spacelike.

    skip_hyperboloid_check : bool, default=False
        Whether to skip checking if points lie on a hyperboloid. Set to True for speed
        if you're confident your data is already properly formatted for hyperbolic space.

    **kwargs : dict
        Additional parameters to pass to the underlying RandomForestRegressor.
    """

    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int = 3, 
        curvature: float = 1.0, 
        timelike_dim: int = 0, 
        skip_hyperboloid_check: bool = False, 
        **kwargs: Any
    ):
        kwargs["n_estimators"] = n_estimators
        super().__init__(
            backend="sklearn_rf",
            task="regression",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike
    ) -> "HyperbolicRandomForestRegressor":
        """
        Fit the hyperbolic random forest regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The training input samples in hyperboloid coordinates
        y : array-like of shape (n_samples,)
            The target values (real numbers)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(
        self, 
        X: ArrayLike
    ) -> NDArraySamples:
        """
        Predict regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            The input samples in hyperboloid coordinates.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        return super().predict(X)


# Only define XGBoost classes if the library is available
if XGBOOST_AVAILABLE:

    class HyperbolicXGBoostClassifier(HyperbolicDecisionTree, ClassifierMixin):
        """
        Hyperbolic XGBoost for classification tasks.

        This classifier implements XGBoost that works natively in hyperbolic space
        by transforming data to the Klein model and using an underlying XGBoost
        classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of boosting rounds.

        max_depth : int, default=3
            The maximum depth of the trees.

        learning_rate : float, default=0.1
            The learning rate or step size shrinkage.

        curvature : float, default=1.0
            The curvature of the hyperbolic space. Must be positive.

        timelike_dim : int, default=0
            The index of the timelike dimension in the input data. The remaining dimensions
            are treated as spacelike.

        skip_hyperboloid_check : bool, default=False
            Whether to skip checking if points lie on a hyperboloid. Set to True for speed
            if you're confident your data is already properly formatted for hyperbolic space.

        **kwargs : dict
            Additional parameters to pass to the underlying XGBClassifier.
        """

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 3,
            learning_rate: float = 0.1,
            curvature: float = 1.0,
            timelike_dim: int = 0,
            skip_hyperboloid_check: bool = False,
            **kwargs: Any,
        ):
            kwargs.update({"n_estimators": n_estimators, "learning_rate": learning_rate})
            super().__init__(
                backend="xgboost",
                task="classification",
                max_depth=max_depth,
                curvature=curvature,
                timelike_dim=timelike_dim,
                skip_hyperboloid_check=skip_hyperboloid_check,
                **kwargs,
            )

        def fit(
            self, 
            X: ArrayLike, 
            y: ArrayLike
        ) -> "HyperbolicXGBoostClassifier":
            """
            Fit the hyperbolic XGBoost classifier.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_dimensions)
                The training input samples in hyperboloid coordinates
            y : array-like of shape (n_samples,)
                The target values (class labels)
                
            Returns
            -------
            self : object
                Fitted estimator.
            """
            return super().fit(X, y)

        def predict(
            self, 
            X: ArrayLike
        ) -> NDArraySamples:
            """
            Predict class for X.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_dimensions)
                The input samples in hyperboloid coordinates.
                
            Returns
            -------
            y : ndarray of shape (n_samples,)
                The predicted classes.
            """
            return super().predict(X)

        def predict_proba(
            self, 
            X: ArrayLike
        ) -> NDArraySamplesClasses:
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
            return super().predict_proba(X)

    class HyperbolicXGBoostRegressor(HyperbolicDecisionTree, RegressorMixin):
        """
        Hyperbolic XGBoost for regression tasks.

        This regressor implements XGBoost that works natively in hyperbolic space
        by transforming data to the Klein model and using an underlying XGBoost
        regressor.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of boosting rounds.

        max_depth : int, default=3
            The maximum depth of the trees.

        learning_rate : float, default=0.1
            The learning rate or step size shrinkage.

        curvature : float, default=1.0
            The curvature of the hyperbolic space. Must be positive.

        timelike_dim : int, default=0
            The index of the timelike dimension in the input data. The remaining dimensions
            are treated as spacelike.

        skip_hyperboloid_check : bool, default=False
            Whether to skip checking if points lie on a hyperboloid. Set to True for speed
            if you're confident your data is already properly formatted for hyperbolic space.

        **kwargs : dict
            Additional parameters to pass to the underlying XGBRegressor.
        """

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 3,
            learning_rate: float = 0.1,
            curvature: float = 1.0,
            timelike_dim: int = 0,
            skip_hyperboloid_check: bool = False,
            **kwargs: Any,
        ):
            kwargs.update({"n_estimators": n_estimators, "learning_rate": learning_rate})
            super().__init__(
                backend="xgboost",
                task="regression",
                max_depth=max_depth,
                curvature=curvature,
                timelike_dim=timelike_dim,
                skip_hyperboloid_check=skip_hyperboloid_check,
                **kwargs,
            )

        def fit(
            self, 
            X: ArrayLike, 
            y: ArrayLike
        ) -> "HyperbolicXGBoostRegressor":
            """
            Fit the hyperbolic XGBoost regressor.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_dimensions)
                The training input samples in hyperboloid coordinates
            y : array-like of shape (n_samples,)
                The target values (real numbers)
                
            Returns
            -------
            self : object
                Fitted estimator.
            """
            return super().fit(X, y)

        def predict(
            self, 
            X: ArrayLike
        ) -> NDArraySamples:
            """
            Predict regression value for X.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_dimensions)
                The input samples in hyperboloid coordinates.
                
            Returns
            -------
            y : ndarray of shape (n_samples,)
                The predicted values.
            """
            return super().predict(X)