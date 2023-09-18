"""Hyperbolic decision tree model"""

import numpy as np
from warnings import warn
from sklearn.base import BaseEstimator, ClassifierMixin
from .hyperbolic_trig import get_candidates_hyperbolic


class DecisionNode:
    def __init__(self, value=None, probs=None, feature=None, theta=None):
        self.value = value
        self.probs = probs
        self.feature = feature
        self.theta = theta
        self.left = None
        self.right = None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        criterion="gini",  # 'gini', 'entropy', or 'misclassification'
        weights=None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion = criterion
        self.weights = weights
        self.min_impurity_decrease = 0.0

        # Set loss
        if criterion == "gini":
            self._loss = self._gini

    def _get_probs(self, y):
        """Get the class probabilities"""
        return np.bincount(y, minlength=len(self.classes_)) / len(y)

    def _gini(self, y):
        """Gini impurity"""
        if self.weights is not None:
            return 1 - np.sum(self._get_probs(y) ** 2 * self.weights)
        else:
            return 1 - np.sum(self._get_probs(y) ** 2)

    def _information_gain(self, left, right, y):
        """Get the information gain from splitting on a given dimension"""
        y_l, y_r = y[left], y[right]
        w_l, w_r = len(y_l) / len(y), len(y_r) / len(y)
        parent_loss = self._loss(y)
        child_loss = w_l * self._loss(y_l) + w_r * self._loss(y_r)
        return parent_loss - child_loss

    def _get_split(self, X, dim, theta):
        """Get the indices of the split"""
        return X[:, dim] < theta, X[:, dim] >= theta

    def _get_candidates(self, X, dim):
        """Get candidate angles for a given dimension"""
        unique_vals = np.unique(X[:, dim])  # already sorted
        return (unique_vals[:-1] + unique_vals[1:]) / 2

    def _fit_node(self, X, y, depth):
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

    def _leaf_values(self, y):
        """Return the value and probability of a leaf node"""
        probs = self._get_probs(y)
        return np.argmax(probs), probs

    def fit(self, X, y):
        """Fit a decision tree to the data"""

        # Some attributes we need:
        self.ndim = X.shape[1]
        self.dims = list(range(self.ndim))
        self.classes_, y = np.unique(y, return_inverse=True)

        # Weight classes
        if self.weights == "balanced":
            self.weights = 1 / np.bincount(y)
            self.weights /= np.sum(self.weights)

        # Validate data and fit tree:
        # self.label_names, y = np.unique(y, return_inverse=True)
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _left(self, x, node):
        """Boolean: Go left?"""
        return x[node.feature] < node.theta

    def _traverse(self, x, node=None):
        """Traverse a decision tree for a single point"""
        # Root case
        if node is None:
            node = self.tree

        # Leaf case
        if node.value is not None:
            return node

        return self._traverse(x, node.left) if self._left(x, node) else self._traverse(x, node.right)

    def predict(self, X):
        """Predict labels for samples in X"""
        return np.array([self.classes_[self._traverse(x).value] for x in X])

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        return np.array([self._traverse(x).probs for x in X])

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return np.mean(self.predict(X) == y)


class HyperbolicDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, candidates="data", timelike_dim=0, dot_product="sparse", **kwargs):
        super().__init__(**kwargs)
        self.candidates = candidates
        self.timelike_dim = timelike_dim
        self.hyperbolic = True
        self.dot_product = dot_product

    def _dot(self, X, dim, theta):
        """Get the dot product of the normal vector and the data"""
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

    def _get_split(self, X, dim, theta):
        """Get the indices of the split"""
        p = self._dot(X, dim, theta)
        return p < 0, p >= 0

    def _get_candidates(self, X, dim):
        if self.candidates == "data":
            return get_candidates_hyperbolic(
                X=X,
                dim=dim,
                timelike_dim=self.timelike_dim,
                dot_product=self.dot_product,
            )

        elif self.candidates == "grid":
            return np.linspace(np.pi / 4, 3 * np.pi / 4, 1000)

    def _validate_hyperbolic(self, X):
        """
        Ensure points lie on a hyperboloid - subtract timelike twice from sum of all
        squares, rather than once from sum of all spacelike squares, to simplify
        indexing
        """
        X_spacelike = X[:, self.dims]  # Nice and clean
        try:
            assert np.allclose(
                np.sum(X_spacelike**2, axis=1) - X[:, self.timelike_dim] ** 2,
                -1.0,
                atol=1e-3,  # Don't be too strict
            )
            assert np.all(X[:, self.timelike_dim] > 1.0)  # Ensure timelike
            assert np.all(X[:, self.timelike_dim] > np.linalg.norm(X_spacelike, axis=1))
        except AssertionError:
            raise ValueError("Points must lie on a hyperboloid")

    def fit(self, X, y):
        """Fit a decision tree to the data"""

        # Some attributes we need:
        self.ndim = X.shape[1]
        self.classes_ = np.unique(y)
        self.dims = list(range(self.ndim))
        if self.timelike_dim == -1:
            self.dims.remove(self.ndim - 1)  # Not clean but whatever
        else:
            self.dims.remove(self.timelike_dim)

        # Validate data and fit tree:
        self._validate_hyperbolic(X)
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _left(self, x, node):
        """Boolean: Go left?"""
        return self._dot(x.reshape(1, -1), node.feature, node.theta).item() < 0


class DecisionTreeRegressor(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = self._mse

    def _mse(self, y):
        """Mean squared error"""
        return np.mean((y - np.mean(y)) ** 2)

    def _leaf_values(self, y):
        """Return the value and probability (dummy) of a leaf node"""
        return np.mean(y), None  # TODO: probs?

    def predict_proba(self, X):
        """Predict class probabilities for samples in X (dummy)"""
        raise NotImplementedError("Regression does not support predict_proba")

    def score(self, X, y, metric="mse"):
        """Return the mean accuracy/score on the given test data and labels"""
        y_hat = self.predict(X)
        if metric == "mse":
            return np.mean((y - y_hat) ** 2)
        elif metric == "rmse":
            return np.sqrt(np.mean((y - y_hat) ** 2))
        elif metric == "mae":
            return np.mean(np.abs(y - y_hat))
        elif metric in ["r2", "R2"]:
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


class HyperbolicDecisionTreeRegressor(DecisionTreeRegressor, HyperbolicDecisionTreeClassifier):
    """Hacky multiple inheritance constructor - seems to work though"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
