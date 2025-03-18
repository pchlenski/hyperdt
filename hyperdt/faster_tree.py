import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class HyperbolicDecisionTree(BaseEstimator):
    """Base class for hyperbolic trees with configurable backend"""

    def __init__(
        self,
        backend="sklearn_dt",
        task="classification",
        max_depth=3,
        curvature=1.0,
        timelike_dim=0,
        skip_hyperboloid_check=False,
        **kwargs,
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

    def _init_estimator(self):
        """Initialize the appropriate backend estimator"""
        backend_map = {
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

    def _validate_hyperbolic(self, X):
        """
        Ensure points lie on a hyperboloid - subtract timelike twice from sum of all squares, rather than once from sum
        of all spacelike squares, to simplify indexing.
        """
        dims = np.delete(np.arange(X.shape[1]), self.timelike_dim)
        # Ensure Minkowski norm
        assert np.allclose(
            np.sum(X[:, dims] ** 2, axis=1) - X[:, self.timelike_dim] ** 2, -1 / self.curvature, atol=1e-3
        ), "Points must lie on a hyperboloid: Minkowski norm does not equal {-1 / self.curvature}."

        # Ensure timelike
        assert np.all(
            X[:, self.timelike_dim] > 1.0 / self.curvature
        ), "Points must lie on a hyperboloid: Value at timelike dimension must be greater than 1."

        # Ensure hyperboloid
        assert np.all(
            X[:, self.timelike_dim] > np.linalg.norm(X[:, dims], axis=1)
        ), "Points must lie on a hyperboloid: Value at timelike dim must exceed norm of spacelike dims."

    def _einstein_midpoint(self, u, v):
        """Einstein midpoint for scalar features. Assumes u, v are the i-th coordinates of points in the Klein model"""
        gamma_u = 1 / np.sqrt(1 - u**2 / self.curvature)
        gamma_v = 1 / np.sqrt(1 - v**2 / self.curvature)

        # Correct Einstein midpoint formula for scalars
        numerator = gamma_u * u + gamma_v * v
        denominator = gamma_u + gamma_v
        midpoint = numerator / denominator

        # Rescale back to original coordinates
        return midpoint

    def _adjust_thresholds(self, estimator, X_klein, samples):
        """
        Adjust thresholds using Einstein midpoint method.
        Works for both individual trees and ensembles of trees.
        """
        # Handle different types of estimators
        if hasattr(estimator, "estimators_"):  # RandomForest and similar
            for tree in estimator.estimators_:
                self._adjust_tree_thresholds(tree.tree_, 0, X_klein, samples)
        elif hasattr(estimator, "tree_"):  # Single DecisionTree
            self._adjust_tree_thresholds(estimator.tree_, 0, X_klein, samples)
        # Add handlers for other tree types as needed (XGBoost, etc.)

    def _adjust_tree_thresholds(self, tree, node_id, X_klein, samples):
        """Adjust thresholds for a single tree's node and its children"""
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

    def fit(self, X, y):
        if not self.skip_hyperboloid_check:
            self._validate_hyperbolic(X)

        # Convert to Klein coordinates (x_d/x_0)
        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]

        # Fit backend estimator
        self.estimator_.fit(X_klein, y)

        # Adjust thresholds for decision trees and tree ensembles
        if self.backend in ["sklearn_dt", "sklearn_rf"]:
            self._adjust_thresholds(self.estimator_, X_klein, np.arange(len(X)))

        return self

    def predict(self, X):
        if not self.skip_hyperboloid_check:
            self._validate_hyperbolic(X)

        # Convert to Klein coordinates
        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]

        return self.estimator_.predict(X_klein)

    def predict_proba(self, X):
        """Probability predictions for classifier models"""
        if self.task != "classification":
            raise AttributeError("predict_proba is not available for regression tasks")

        if not self.skip_hyperboloid_check:
            self._validate_hyperbolic(X)

        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]

        return self.estimator_.predict_proba(X_klein)

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
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

    def __init__(self, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=False, **kwargs):
        super().__init__(
            backend="sklearn_dt",
            task="classification",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )


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

    def __init__(self, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=False, **kwargs):
        super().__init__(
            backend="sklearn_dt",
            task="regression",
            max_depth=max_depth,
            curvature=curvature,
            timelike_dim=timelike_dim,
            skip_hyperboloid_check=skip_hyperboloid_check,
            **kwargs,
        )


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
        self, n_estimators=100, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=False, **kwargs
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
        self, n_estimators=100, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=False, **kwargs
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
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            curvature=1.0,
            timelike_dim=0,
            skip_hyperboloid_check=False,
            **kwargs,
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
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            curvature=1.0,
            timelike_dim=0,
            skip_hyperboloid_check=False,
            **kwargs,
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
