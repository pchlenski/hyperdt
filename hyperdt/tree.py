"""Hyperbolic decision tree model"""

from typing import Tuple, List, Union, Literal

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from .hyperbolic_trig import get_candidates
from .cache import SplitCache


class DecisionNode:
    """Node in a decision tree. If value is not None, then the node is a leaf. Otherwise, it is an internal node."""

    def __init__(
        self, value: Union[int, float, None] = None, probs: np.ndarray = None, feature: int = None, theta: float = None
    ) -> None:
        self.value = value
        self.probs = probs  # predicted class probabilities of all samples in the leaf
        self.feature = feature  # feature index
        self.theta = theta  # threshold
        self.left = None
        self.right = None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Basic CART decision tree classifier. Used to define regressors and hyperDT trees."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        criterion: Literal["gini", "entropy", "misclassification"] = "gini",
        weights: np.ndarray = None,
        cache: SplitCache = None,
    ) -> None:

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion = criterion
        self.weights = weights
        self.min_impurity_decrease = 0.0

        self.ndim = None
        self.dims = None
        self.classes_ = None

        # Set loss
        if criterion == "gini":
            self._loss = self._gini
        else:
            raise ValueError("Invalid criterion")

        # Cache management (not used in Euclidean case)
        self.cache = SplitCache() if cache is None else cache

    def _get_probs(self, y: np.ndarray) -> np.ndarray:
        """Get the class probabilities"""
        # Convert y labels into indices in self.classes_
        # self.classes_ maintains the indexing that we want for all of our datapoints
        y = np.searchsorted(self.classes_, y)
        return np.bincount(y, minlength=len(self.classes_)) / len(y)

    def _gini(self, y: np.ndarray) -> float:
        """Gini impurity"""
        if self.weights is not None:
            return 1 - np.sum(self._get_probs(y) ** 2 * self.weights)
        else:
            return 1 - np.sum(self._get_probs(y) ** 2)

    def _information_gain(self, left: List[int], right: List[int], y: np.ndarray) -> float:
        """Get the information gain from splitting on a given dimension"""
        y_l, y_r = y[left], y[right]
        w_l, w_r = len(y_l) / len(y), len(y_r) / len(y)
        parent_loss = self._loss(y)
        child_loss = w_l * self._loss(y_l) + w_r * self._loss(y_r)
        return parent_loss - child_loss

    def _get_split(self, X: np.ndarray, dim: int, theta: float) -> Tuple[List[int], List[int]]:
        """Get the indices of the split"""
        return X[:, dim] < theta, X[:, dim] >= theta

    def _get_candidates(self, X: np.ndarray, dim: int) -> np.ndarray:
        """Get candidate angles for a given dimension"""
        unique_vals = np.unique(X[:, dim])  # already sorted
        return (unique_vals[:-1] + unique_vals[1:]) / 2

    def _fit_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        """Recursively fit a node of the tree"""

        # Base case
        if depth == self.max_depth or len(y) <= self.min_samples_split or len(np.unique(y)) == 1:
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Recursively find the best split:
        best_dim, best_theta, best_score = None, None, -1
        for dim in self.dims:
            for theta in self._get_candidates(X=X, dim=dim):
                left, right = self._get_split(X=X, dim=dim, theta=theta)
                min_len = np.min([len(y[left]), len(y[right])])
                if min_len >= self.min_samples_leaf:
                    score = self._information_gain(left, right, y)
                    if score >= best_score + self.min_impurity_decrease:
                        best_dim, best_theta, best_score = dim, theta, score

        # Fallback case:
        if best_score == -1:
            value, probs = self._leaf_values(y)
            return DecisionNode(value=value, probs=probs)

        # Populate:
        node = DecisionNode(feature=best_dim, theta=best_theta)
        node.score = best_score
        left, right = self._get_split(X=X, dim=best_dim, theta=best_theta)
        node.left = self._fit_node(X=X[left], y=y[left], depth=depth + 1)
        node.right = self._fit_node(X=X[right], y=y[right], depth=depth + 1)
        return node

    def _leaf_values(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        """Return the value and probability of a leaf node"""
        probs = self._get_probs(y)
        return np.argmax(probs), probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Fit a decision tree to the data

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to fit tree to
        y: np.ndarray (n_samples,)
            Labels for data

        Returns:
        -------
        DecisionTreeClassifier
            Fitted decision tree model
        """

        # Some attributes we need:
        self.ndim = X.shape[1]
        self.dims = list(range(self.ndim))
        self.classes_, y = np.unique(y, return_inverse=True)

        # Weight classes
        if self.weights == "balanced":
            self.weights = 1 / np.bincount(y)
            self.weights /= np.sum(self.weights)

        # Validate data and fit tree:
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _left(self, x: np.ndarray, node: DecisionNode) -> bool:
        """Boolean: Go left?"""
        return x[node.feature] < node.theta

    def _traverse(self, x: np.ndarray, node: DecisionNode = None) -> DecisionNode:
        """Traverse a decision tree for a single point"""
        # Root case
        if node is None:
            node = self.tree

        # Leaf case
        if node.value is not None:
            return node

        return self._traverse(x, node.left) if self._left(x, node) else self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples in X

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to predict labels for

        Returns:
        -------
        np.ndarray (n_samples,)
            Predicted labels for each sample in X
        """
        return np.array([self.classes_[self._traverse(x).value] for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to predict class probabilities for

        Returns:
        -------
        np.ndarray (n_samples, n_classes)
            Predicted class probabilities for each sample in X
        """
        return np.array([self._traverse(x).probs for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to predict class probabilities for
        y: np.ndarray (n_samples,)
            Labels for data

        Returns:
        -------
        float
            Mean accuracy of the model on the given data
        """
        return np.mean(self.predict(X) == y)


class HyperbolicDecisionTreeClassifier(DecisionTreeClassifier):
    """Modify DecisionTreeClassifier to handle hyperbolic data."""

    def __init__(
        self,
        candidates: Literal["data", "grid"] = "data",
        timelike_dim: int = 0,
        dot_product: Literal["sparse", "dense", "sparse_minkowski"] = "sparse",
        curvature: float = 1,
        skip_hyperboloid_check: bool = False,
        angle_midpoint_method: Literal["hyperbolic", "bisect"] = "hyperbolic",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.candidates = candidates
        self.timelike_dim = timelike_dim
        self.hyperbolic = True
        self.dot_product = dot_product
        self.curvature = abs(curvature)
        self.skip_hyperboloid_check = skip_hyperboloid_check
        self.angle_midpoint_method = angle_midpoint_method  # 'hyperbolic' or 'bisect'

    def _dot(self, X: np.ndarray, dim: int, theta: float) -> np.ndarray:
        """Get the dot product(s) of the normal vector and the data"""
        if self.dot_product == "sparse":
            return np.sin(theta) * X[:, dim] - np.cos(theta) * X[:, self.timelike_dim]
        elif self.dot_product == "dense":
            v = np.zeros(self.ndim)
            v[self.timelike_dim], v[dim] = -np.cos(theta), np.sin(theta)
            return X @ v
        elif self.dot_product == "sparse_minkowski":
            return np.sin(theta) * X[:, dim] + np.cos(theta) * X[:, self.timelike_dim]
        else:
            raise ValueError("Invalid dot product")

    def _get_split(self, X: np.ndarray, dim: int, theta: float) -> Tuple[List[int], List[int]]:
        """Get the indices of the split"""
        p = self._dot(X, dim, theta)
        return p < 0, p >= 0

    def _get_candidates(self, X: np.ndarray, dim: int) -> np.ndarray:
        if self.candidates == "data":
            return get_candidates(
                X=X, dim=dim, timelike_dim=self.timelike_dim, method=self.angle_midpoint_method, cache=self.cache
            )

        elif self.candidates == "grid":
            return np.linspace(np.pi / 4, 3 * np.pi / 4, 1000)

    def _validate_hyperbolic(self, X: np.ndarray) -> None:
        """
        Ensure points lie on a hyperboloid - subtract timelike twice from sum of all squares, rather than once from sum
        of all spacelike squares, to simplify indexing.
        """
        if self.skip_hyperboloid_check:
            return

        # Ensure Minkowski norm
        if not np.allclose(
            np.sum(X[:, self.dims] ** 2, axis=1) - X[:, self.timelike_dim] ** 2, -1 / self.curvature, atol=1e-3
        ):
            raise ValueError(f"Points must lie on a hyperboloid: Minkowski norm does not equal {-1 / self.curvature}.")

        # Ensure timelike
        if not np.all(X[:, self.timelike_dim] > 1.0 / self.curvature):
            raise ValueError("Points must lie on a hyperboloid: Value at timelike dimension must be greater than 1.")

        # Ensure hyperboloid
        if not np.all(X[:, self.timelike_dim] > np.linalg.norm(X[:, self.dims], axis=1)):
            raise ValueError(
                "Points must lie on a hyperboloid: Value at timelike dimension must exceed norm of spacelike dimensions."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HyperbolicDecisionTreeClassifier":
        """
        Fit a decision tree to the data

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to fit tree to
        y: np.ndarray (n_samples,)
            Labels for data

        Returns:
        -------
        HyperbolicDecisionTreeClassifier
            Fitted decision tree model
        """

        # Some attributes we need:
        self.ndim = X.shape[1]
        self.classes_ = np.unique(y)
        self.dims = list(range(self.ndim))

        # Weight classes
        if self.weights == "balanced":
            self.weights = 1 / np.bincount(y)
            self.weights /= np.sum(self.weights)

        # Hyperboloid specific code:
        if self.timelike_dim == -1:
            self.dims.remove(self.ndim - 1)  # Not clean but whatever
        else:
            self.dims.remove(self.timelike_dim)
        self._validate_hyperbolic(X)

        # Fit tree
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _left(self, x: np.ndarray, node: DecisionNode) -> bool:
        """Boolean: Go left?"""
        return self._dot(x.reshape(1, -1), node.feature, node.theta).item() < 0


class DecisionTreeRegressor(DecisionTreeClassifier):
    """Modify DecisionTreeClassifier to handle regression."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = self._mse

    def _mse(self, y: np.ndarray) -> float:
        """Mean squared error"""
        return np.mean((y - np.mean(y)) ** 2)

    def _leaf_values(self, y: np.ndarray) -> Tuple[float, None]:
        """Return the value and probability (dummy) of a leaf node"""
        return np.mean(y), None  # TODO: probs?

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression values for samples in X

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to predict regression values for

        Returns:
        -------
        np.ndarray (n_samples,)
            Predicted regression values for each sample in X
        """
        return np.array([self._traverse(x).value for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X (dummy)"""
        raise NotImplementedError("Regression does not support predict_proba")

    def score(self, X: np.ndarray, y: np.ndarray, metric: Literal["mse", "rmse", "mae", "r2", "R2"] = "mse") -> float:
        """
        Return the mean accuracy/score on the given test data and labels

        Args:
        ----
        X: np.ndarray (n_samples, n_features)
            Data to predict class probabilities for
        y: np.ndarray (n_samples,)
            Labels for data
        metric: str
            Metric to use for scoring. One of ["mse", "rmse", "mae", "r2", "R2"]

        Returns:
        -------
        float
            Mean accuracy/score of the model on the given data
        """
        y_hat = self.predict(X)
        if metric == "mse":
            return np.mean((y - y_hat) ** 2)
        elif metric == "rmse":
            return np.sqrt(np.mean((y - y_hat) ** 2))
        elif metric == "mae":
            return np.mean(np.abs(y - y_hat))
        elif metric in ["r2", "R2"]:
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """
        Fit a decision tree to the data. Wrapper for DecisionTreeClassifier's fit method but with a dummy
        self.classes_ attribute.
        """
        super().fit(X, y)
        self.classes_ = None
        return self


class HyperbolicDecisionTreeRegressor(DecisionTreeRegressor, HyperbolicDecisionTreeClassifier):
    """Hacky multiple inheritance constructor - seems to work though"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
