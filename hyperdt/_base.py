"""Base classes for hyperbolic decision trees.

This module implements the core functionality for decision trees that operate
natively in hyperbolic space.
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator


class HyperbolicDecisionTree(BaseEstimator):
    """Base class for Klein wrapper"""

    def __init__(
        self,
        backend: Literal["sklearn_dt", "sklearn_rf", "xgboost"] = "sklearn_dt",
        task: Literal["classification", "regression"] = "classification",
        curvature: float = 1.0,
        timelike_dim: int = 0,
        validate_input_geometry: bool = True,
        input_geometry: Literal["hyperboloid", "klein", "poincare"] = "hyperboloid",
        **kwargs: Any,
    ):
        self.backend = backend
        self.task = task
        self.curvature = curvature
        self.timelike_dim = timelike_dim
        self.validate_input_geometry = validate_input_geometry
        self.input_geometry = input_geometry
        self.kwargs = kwargs

        # Import the appropriate estimator based on task and backend
        estimator_mapping = {
            ("classification", "sklearn_dt"): ("sklearn.tree", "DecisionTreeClassifier"),
            ("classification", "sklearn_rf"): ("sklearn.ensemble", "RandomForestClassifier"),
            ("classification", "xgboost"): ("xgboost", "XGBClassifier"),
            ("regression", "sklearn_dt"): ("sklearn.tree", "DecisionTreeRegressor"),
            ("regression", "sklearn_rf"): ("sklearn.ensemble", "RandomForestRegressor"),
            ("regression", "xgboost"): ("xgboost", "XGBRegressor"),
        }
        if (self.task, self.backend) not in estimator_mapping:
            raise ValueError(f"Unknown task: {self.task} and/or backend: {self.backend}.")

        # Only import the estimator we need; add it to the class
        module_name, class_name = estimator_mapping[self.task, self.backend]
        module = __import__(module_name, fromlist=[class_name])
        estimator_class = getattr(module, class_name)
        self.estimator_ = estimator_class(**self.kwargs)  # sklearn compatible namings

    def _validate_hyperboloid(self, X: ArrayLike) -> None:
        """Validate the input data: ensure points lie on a hyperboloid."""
        assert X.ndim == 2, "Input must be a 2D array."
        X_spacelike = np.delete(X, self.timelike_dim, axis=1)
        X_timelike = X[:, self.timelike_dim]
        assert np.all(
            X_timelike >= 1.0 / self.curvature
        ), "Points must lie on a hyperboloid: Value at timelike dimension must be greater than 1 / K."
        assert np.all(
            X_timelike > np.linalg.norm(X_spacelike, axis=1)
        ), "Points must lie on a hyperboloid: Value at timelike dim must exceed norm of spacelike dims."
        assert np.allclose(
            np.linalg.norm(X_spacelike, axis=1) ** 2 - X_timelike**2, -1 / self.curvature
        ), "Points must lie on a hyperboloid: Minkowski norm must equal -1/curvature."

    def _validate_klein(self, X: ArrayLike) -> None:
        """Validate the input data: ensure points lie on a hyperboloid."""
        assert np.all(
            np.linalg.norm(X, axis=1) <= 1 / self.curvature**0.5
        ), "Points must lie inside a disk: norms must be <= than 1/sqrt(K)."

    def _validate_poincare(self, X: ArrayLike) -> None:
        """Validate the input data: ensure points lie on a hyperboloid."""
        assert np.all(
            np.linalg.norm(X, axis=1) <= 1 / self.curvature**0.5
        ), "Points must lie inside a disk: norms must be <= than 1/sqrt(K)."

    def _preprocess(self, X: ArrayLike) -> np.ndarray:
        """Preprocess the input data: convert to Klein coordinates."""
        if self.input_geometry == "hyperboloid":
            if self.validate_input_geometry:
                self._validate_hyperboloid(X)
            X_spacelike = np.delete(X, self.timelike_dim, axis=1)
            X_timelike = X[:, self.timelike_dim]
            X_klein = X_spacelike / X_timelike.reshape(-1, 1)
        elif self.input_geometry == "klein":
            if self.validate_input_geometry:
                self._validate_klein(X)
            X_klein = X
        elif self.input_geometry == "poincare":
            if self.validate_input_geometry:
                self._validate_poincare(X)
            coefs = 1 / (1 + (1 - self.curvature * np.linalg.norm(X, axis=1) ** 2) ** 0.5)
            X_klein = X * coefs.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown input geometry: {self.input_geometry}.")
        self._validate_klein(X_klein)
        return X_klein

    def _einstein_midpoint(self, u: float, v: float) -> float:
        """Calculate the Einstein midpoint between two values."""

        # Get the Lorentz factor for each value
        gamma_u = 1 / np.sqrt(1 - u**2 / self.curvature)
        gamma_v = 1 / np.sqrt(1 - v**2 / self.curvature)

        # Calculate the Einstein midpoint
        numerator = gamma_u * u + gamma_v * v
        denominator = gamma_u + gamma_v

        return numerator / denominator

    def _fix_node_recursive(self, estimator: Any, node_id: int, X_klein: np.ndarray) -> None:
        """Fix the tree to use Einstein midpoints."""

        if estimator.tree_.children_left[node_id] == -1:  # Leaf node
            return

        # Get feature and threshold
        feature = estimator.tree_.feature[node_id]
        threshold = estimator.tree_.threshold[node_id]

        # Get feature values
        feature_values = X_klein[:, feature]

        # Adjust threshold to be the average of closest values on either side
        left_mask = feature_values <= threshold
        right_mask = ~left_mask

        # Adjust this node's threshold using Einstein midpoints instead of naive averages as in base sklearn
        left_max = np.max(feature_values[left_mask])  # Closest point from left
        right_min = np.min(feature_values[right_mask])  # Closest point from right
        estimator.tree_.threshold[node_id] = self._einstein_midpoint(left_max, right_min)

        # Recurse
        self._fix_node_recursive(estimator, estimator.tree_.children_left[node_id], X_klein[left_mask])
        self._fix_node_recursive(estimator, estimator.tree_.children_right[node_id], X_klein[right_mask])

    def _postprocess(self, X_klein: np.ndarray) -> np.ndarray:
        """Postprocess estimator: change to Einstein midpoints"""
        if self.backend == "sklearn_dt":
            self._fix_node_recursive(self.estimator_, 0, X_klein)
        elif self.backend == "sklearn_rf":
            rf = self.estimator_
            for tree_idx, tree in enumerate(rf.estimators_):
                # Restrict X to the points that were used to train this tree
                X_klein_tree = X_klein[rf.estimators_samples_[tree_idx]]
                self._fix_node_recursive(tree, 0, X_klein_tree)
        elif self.backend == "xgboost":
            # This gets overwritten by the XGBoost implementation
            pass

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HyperbolicDecisionTree":
        """
        Fit the hyperbolic decision tree.

        Args:
        -----
        X: ArrayLike
            The input data.
        y: ArrayLike
            The target data.

        Returns:
        --------
        self: HyperbolicDecisionTree
            The fitted hyperbolic decision tree predictor.
        """
        X_klein = self._preprocess(X)
        self.estimator_.fit(X_klein, y)
        self._postprocess(X_klein)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the output for the input data.

        Args:
        -----
        X: ArrayLike
            The input data.

        Returns:
        --------
        y_pred: np.ndarray
            The predicted output.
        """
        X_klein = self._preprocess(X)
        return self.estimator_.predict(X_klein)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict the output for the input data.

        Args:
        -----
        X: ArrayLike
            The input data.

        Returns:
        --------
        y_pred: np.ndarray
            The predicted output.
        """
        X_klein = self._preprocess(X)
        return self.estimator_.predict_proba(X_klein)
