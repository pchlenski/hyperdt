"""Hyperbolic decision tree model"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# from geomstats.geometry.hyperboloid import Hyperboloid

# Gini impurity
from itertools import combinations


def _gini(x):
    """Gini impurity: 1 - sum(p_i^2)"""
    if x.dtype == int:
        counts = np.bincount(x, minlength=2)
    else:
        _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return 1.0 - np.sum(p ** 2)


class HyperbolicDecisionNode:
    def __init__(
        self,
        X,
        y,
        depth,
        dim=None,
        theta=None,
        leaf=False,
        parent=None,
        id=None,
    ):
        self.X = X
        self.y = y
        self.depth = depth
        self.dim = dim
        self.theta = theta
        self.leaf = leaf
        self.parent = parent
        self.left = None
        self.right = None
        self.id = id

        # If this is a leaf node, compute and store its prediction
        if self.leaf and len(y) > 0:
            classes, counts = np.unique(y, return_counts=True)
            self.prediction = classes[np.argmax(counts)]  # most common class
            self.probs = dict(zip(classes, counts / len(y)))
        else:
            self.prediction = None
            self.probs = None


class HyperbolicDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, min_samples=2, hyperbolic=True):
        """Init classifier"""
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.hyperbolic = hyperbolic

        # Initialize tree
        self.tree = None

    def _get_split(self, X, dim, theta):
        """
        Get indices to left and right of decision hyperplane at angle theta
        """

        # Get left and right nodes
        # Note that this approach does not include a bias term. We do not
        # allow bias terms in the hyperboloid case, since they do not intersect
        # the manifold at geodesics.

        if self.hyperbolic:
            # Normal vector for decision hyperplane (passing through origin)
            normal = np.zeros(X.shape[1])
            normal[0], normal[dim] = np.sin(theta), np.cos(theta)
            prods = X @ normal

        else:
            # Normal vector for decision hyperplane (with bias)
            normal = np.zeros(X.shape[1])
            normal[dim] = 1.0
            prods = X @ normal - theta

        left = prods < 0.0
        right = prods >= 0.0
        print("LEFT", left)
        return left, right

    def _get_impurity(self, left, right, y, leaf=False):
        """Get impurity of a split"""
        y_left, y_right = y[left], y[right]
        min_samples = np.min([len(y_left), len(y_right)])
        print(f"min_samples: {min_samples}")
        if min_samples < self.min_samples:
            return np.inf
        else:
            return _gini(y_left) + _gini(y_right)

    def _get_best_theta_dim(self, X, y, dim):
        """
        Find best split dimension for a given dimension. This is done by
        finding argmin_t of a normal vector (t, 0, ..., 1, ..., 0), where x_0 is
        the time-like coordinate and x_dim is our split dimension, with respect
        to the impurity of the resulting nodes.
        """

        # Input validation
        assert dim > 0, "dim must be greater than 0"
        assert dim < X.shape[1], "dim must be less than X.shape[1]"
        assert len(y) == len(X), "X and y must have the same length"
        dim = int(dim)

        # First, compute tangents along (x_0, x_dim) plane, then angle:
        X[:, dim][X[:, dim] == 0.0] = 1e-6  # Avoid division by zero
        tans = X[:, 0] / X[:, dim]
        tans[tans > 1e6] = 1e6  # Avoid overflow
        tans[tans < -1e6] = -1e6  # Avoid overflow
        thetas = np.arctan(tans)
        thetas = np.sort(thetas)
        thetas = np.unique(thetas)  # Remove duplicates)

        # Candidates are midpoints between thetas
        # candidates = (thetas[1:] + thetas[:-1]) / 2.0
        # OVERRIDE
        if self.hyperbolic:
            candidates = np.linspace(0, 2 * np.pi, 1000)
        else:
            candidates = np.linspace(np.min(X[:, dim]), np.max(X[:, dim]), 1000)
        print(f"Got {len(candidates)} candidates for dim {dim}")
        print(
            f"Max: {np.max(candidates)/np.pi:.3f}*pi, min: {np.min(candidates)/np.pi:.3f}*pi"
        )

        # Get best split
        best_theta = None
        best_score = np.inf
        for theta in candidates:
            left, right = self._get_split(X=X, dim=dim, theta=theta)
            score = self._get_impurity(left, right, y)
            if score < best_score:
                best_theta = theta
                best_score = score

        return best_theta, best_score

    def get_best_dim(self, X, y):
        """Find best split dimension for a given node"""
        ndim = X.shape[1]
        dim_thetas_scores = [
            self._get_best_theta_dim(X, y, dim) for dim in range(1, ndim)
        ]
        print("Got dim thetas scores")
        dim_thetas = [theta for theta, _ in dim_thetas_scores]
        dim_scores = [score for _, score in dim_thetas_scores]
        best_idx = np.argmin(dim_scores)
        print("Finishing get_best_dim")
        return best_idx, dim_thetas[best_idx], dim_scores[best_idx]

    def _fit_node(self, X, y, depth, parent=None, id="ROOT"):
        """
        Recursively fit nodes to the data
        """

        print(f"Fitting node at depth {depth}")

        # Check for stopping conditions:
        if (
            (depth == self.max_depth)
            or (len(y) < self.min_samples)
            or (len(set(y)) == 1)
        ):
            return HyperbolicDecisionNode(
                X, y, depth, leaf=True, parent=parent, id=id
            )

        # Find the best split for the current node
        best_dim, best_theta, best_score = self.get_best_dim(X, y)

        # Initialize current node
        node = HyperbolicDecisionNode(
            X=X,
            y=y,
            depth=depth,
            dim=best_dim,
            theta=best_theta,
            leaf=False,
            parent=parent,
            id=id,
        )

        # Split the data into left and right partitions
        left, right = self._get_split(X=X, dim=best_dim, theta=best_theta)
        print(f"Split data into {len(X[left])} and {len(X[right])}")

        # Recursive calls to _fit_node for left and right child nodes
        node.left = self._fit_node(
            X=X[left], y=y[left], depth=depth + 1, parent=node, id=f"{id}_L"
        )
        node.right = self._fit_node(
            X=X[right], y=y[right], depth=depth + 1, parent=node, id=f"{id}_R"
        )

        # Remove X and y from parent node to save memory
        node.X = None
        node.y = None

        return node

    def _validate(self, X):
        """Ensure data lies on hyperboloid"""
        if np.allclose(X[:, 0], np.sqrt(1 + np.sum(X[:, 1:] ** 2, axis=1))):
            return True
        else:
            raise ValueError("Data must lie on hyperboloid")

    def fit(self, X, y):
        """Fit hyperbolic decision tree to training data"""
        # Initialize tree by fitting the root node to the data
        self._validate(X)

        self.tree = self._fit_node(X=X, y=y, depth=0)

        return self

    def _traverse(self, x, node):
        """
        Traverse the tree for a single instance.
        """

        # If this is a leaf node, return its prediction
        if node.leaf:
            return node

        # Compute dot product of instance and normal vector
        normal = np.zeros(len(x))
        normal[0], normal[node.dim] = np.sin(node.theta), np.cos(node.theta)
        dot_product = x @ normal

        # Traverse to the left or right child node
        if dot_product < 0.0:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """

        # Initialize array for predictions
        y_pred = np.empty(len(X))

        # Traverse tree for each instance
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.tree)
            y_pred[i] = leaf.prediction

        return y_pred

    def predict_probs(self, X):
        """Predict class probabilities for samples in X"""
        raise NotImplementedError
