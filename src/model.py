"""Hyperbolic decision tree model"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# from geomstats.geometry.hyperboloid import Hyperboloid

# Gini impurity
from itertools import combinations


class HyperbolicDecisionNode:
    def __init__(
        self,
        leaf=False,
        id=None,
        feature=None,
        theta=None,
        left=None,
        right=None,
        value=None,
    ):
        """Init node"""
        self.leaf = leaf
        self.id = id
        self.feature = feature
        self.theta = theta
        self.left = left
        self.right = right
        self.value = value


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class HyperbolicDecisionNode:
    def __init__(
        self, leaf=False, value=None, feature=None, theta=None, id=None
    ):
        self.leaf = leaf
        self.value = value
        self.feature = feature
        self.theta = theta
        self.id = id
        self.left = None
        self.right = None


class HyperbolicDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, min_samples=2, hyperbolic=True):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.hyperbolic = hyperbolic
        self.tree = None

    def _normal(self, dim, theta):
        v = np.zeros(self.ndim)
        v[0], v[dim] = np.sin(theta), np.cos(theta)
        return v

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _information_gain(self, left, right, y):
        n = len(y)
        n_l, n_r = len(y[left]), len(y[right])
        if np.min([n_l, n_r]) < self.min_samples:
            return -1
        parent_loss = self._gini(y)
        child_loss = (
            n_l * self._gini(y[left]) + n_r * self._gini(y[right])
        ) / n
        return parent_loss - child_loss

    def _get_split(self, X, dim, theta):
        if self.hyperbolic:
            prods = X @ self._normal(dim, theta)
            return prods < 0.0, prods >= 0.0
        else:
            return X[:, dim] < theta, X[:, dim] >= theta

    def _get_candidates(self, X, dim):
        X[:, dim][X[:, dim] == 0.0] = 1e-6
        thetas = np.arctan(X[:, 0] / X[:, dim])
        thetas = np.unique(np.sort(thetas))
        return (thetas[1:] + thetas[:-1]) / 2.0

    def _fit_node(self, X, y, depth, id="ROOT"):
        if (
            depth == self.max_depth
            or len(y) <= self.min_samples
            or len(set(y)) == 1
        ):
            return HyperbolicDecisionNode(
                leaf=True, value=self._most_common_label(y), id=id
            )
        # Loop through all possible splits:
        best_dim, best_theta, best_score = None, None, -1
        for dim in self.dims:
            for theta in self._get_candidates(X=X, dim=dim):
                left, right = self._get_split(X=X, dim=dim, theta=theta)
                score = self._information_gain(left, right, y)
                if score > best_score:
                    best_dim, best_theta, best_score = dim, theta, score

        # Contingency for no split found:
        if best_score == -1:
            return HyperbolicDecisionNode(
                leaf=True, value=self._most_common_label(y), id=id
            )

        # Populate:
        node = HyperbolicDecisionNode(feature=best_dim, theta=best_theta, id=id)
        l, r = self._get_split(X=X, dim=best_dim, theta=best_theta)
        d = depth + 1
        node.left = self._fit_node(X=X[l], y=y[l], depth=d, id=f"{id}_L")
        node.right = self._fit_node(X=X[r], y=y[r], depth=d, id=f"{id}_R")
        node.X = None
        node.y = None
        return node

    def _most_common_label(self, y):
        _, counts = np.unique(y, return_counts=True)
        return np.argmax(counts) if len(counts) > 0 else None

    def fit(self, X, y):
        self.ndim = X.shape[1]
        self.dims = range(1, self.ndim) if self.hyperbolic else range(self.ndim)
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _traverse(self, x, node):
        if node.leaf:
            return node
        if self.hyperbolic:
            v = self._normal(node.feature, node.theta)
            return (
                self._traverse(x, node.left)
                if x @ v < 0.0
                else self._traverse(x, node.right)
            )
        else:
            return (
                self._traverse(x, node.left)
                if x[node.feature] < node.theta
                else self._traverse(x, node.right)
            )

    def predict(self, X):
        return np.array([self._traverse(x, self.tree).value for x in X])

    def predict_probs(self, X):
        """Predict class probabilities for samples in X"""
        raise NotImplementedError
