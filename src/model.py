"""Hyperbolic decision tree model"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# from geomstats.geometry.hyperboloid import Hyperboloid

# Gini impurity
from itertools import combinations


def _gini(x):
    """Gini impurity: 1 - sum(p_i^2)"""
    counts = np.bincount(x, minlength=2)
    p = counts / len(x)
    return 1.0 - np.sum(p ** 2)


class HyperbolicDecisionNode:
    def __init__(
        self, X, y, depth, dim=None, threshold=None, leaf=False, parent=None
    ):
        self.X = X
        self.y = y
        self.depth = depth
        self.dim = dim
        self.threshold = threshold
        self.leaf = leaf
        self.parent = parent
        self.left = None
        self.right = None


class HyperbolicDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # Initialize tree
        self.tree = None

    def _get_split(self, X, y, dim, theta):
        """
        Get indices to left and right of decision hyperplane at angle theta
        """

        # Normal vector for decision hyperplane (passing through origin)
        normal = np.zeros(X.shape[1])
        normal[0], normal[dim] = np.sin(theta), np.cos(theta)

        # Get left and right nodes
        prods = X @ normal
        left = prods < 0.0
        right = prods >= 0.0
        return left, right

    def _get_impurity(self, left, right, y):
        """Get impurity of a split"""
        y_left, y_right = y[left], y[right]
        return _gini(y_left) + _gini(y_right)

    def _get_best_threshold_dim(self, X, y, dim):
        """
        Find best split dimension for a given dimension. This is done by
        finding argmin_t of a normal vector (t, 0, ..., 1, ..., 0), where x_0 is
        the time-like coordinate and x_dim is our split dimension, with respect
        to the impurity of the resulting nodes.
        """

        # First, compute tangents along (x_0, x_dim) plane, then angle:
        X[:, dim][X[:, dim] == 0.0] = 1e-6 # Avoid division by zero
        tans = X[:, 0] / X[:, dim]
        tans[tans > 1e6] = 1e6 # Avoid overflow
        thetas = np.arctan(tans)
        thetas = sorted(thetas)

        # Candidates are midpoints between thetas
        candidates = (thetas[1:] + thetas[:-1]) / 2.0

        # Get best split
        best_theta = None
        best_score = np.inf
        for theta in candidates:
            left, right = self._get_split(X=X, y=y, dim=dim, theta=theta)
            score = _get_impurity(y[left]) + _get_impurity(y[right])
            if score < best_score:
                best_theta = theta
                best_score = score

        return best_theta, best_score

    def get_best_dim(self, X, y):
        """Find best split dimension for a given node"""
        ndim = X.shape[1]
        dim_thetas_scores = [
            _get_best_threshold_dim(X, y, dim) for dim in range(1, ndim)
        ]
        dim_thetas = [dim for dim, _ in dim_thetas_scores]
        dim_scores = [score for _, score in dim_thetas_scores]
        best_idx = np.argmin(dim_scores)
        return dim_thetas[best_idx], dim_scores[best_idx]

    def _fit_node(self, X, y, depth, parent=None):
        """
        Recursively fit nodes to the data
        """

        # Check for stopping conditions: max depth reached, node size is too 
        # small, or all labels are the same
        if (
            (depth == self.max_depth)
            or (len(y) < self.min_samples_split)
            or (len(set(y)) == 1)
        ):
            return HyperbolicDecisionNode(X, y, depth, leaf=True, parent=parent)

        # Find the best split for the current node
        best_dim, best_threshold = self.get_best_dim(X, y)

        # Initialize current node
        node = HyperbolicDecisionNode(
            X=X,
            y=y,,
            depth=depth,
            dim=best_dim,
            threshold=best_threshold,
            leaf=False,
            parent=parent,
        )

        # Split the data into left and right partitions
        left, right = self._get_split(
            X=X, y=y, dim=best_dim, theta=best_threshold
        )

        # Recursive calls to _fit_node for left and right child nodes
        node.left = self._fit_node(
            X=X[left], y=y[left], depth=depth + 1, parent=node
        )
        node.right = self._fit_node(
            X=X[right], y=y[right], depth=depth + 1, parent=node
        )

        # Remove X and y from parent node to save memory
        node.X = None
        node.y = None

        return node

    def fit(self, X, y):
        """Fit hyperbolic decision tree to training data"""
        # Initialize tree by fitting the root node to the data
        self.tree = self._fit_node(X=X, y=y, depth=0)

        return self

    def predict(self, X):
        """Predict class labels for samples in X"""
        raise NotImplementedError

    def predict_probs(self, X):
        """Predict class probabilities for samples in X"""
        raise NotImplementedError
