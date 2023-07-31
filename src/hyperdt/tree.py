"""Hyperbolic decision tree model"""

import numpy as np
from warnings import warn
from sklearn.base import BaseEstimator, ClassifierMixin


class HyperbolicDecisionNode:
    def __init__(self, value=None, probs=None, feature=None, theta=None):
        self.value = value
        self.probs = probs
        self.feature = feature
        self.theta = theta
        self.left = None
        self.right = None


class HyperbolicDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        hyperbolic=True,
        min_dist=0,
        candidates="data",  # 'data' or 'grid'
        criterion="gini",  # 'gini', 'entropy', or 'misclassification'
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.hyperbolic = hyperbolic
        self.tree = None
        self.min_dist = min_dist
        self.candidates = candidates
        self.criterion = criterion
        if criterion == "gini":
            self._loss = self._gini
        elif criterion == "entropy":
            self._loss = self._entropy
        elif criterion == "misclassification":
            self._loss = self._misclassification
        else:
            raise ValueError(
                "criterion must be one of 'gini', 'entropy', or 'misclassification'"
            )

    def _get_probs(self, y):
        _, counts = np.unique(y, return_counts=True)
        return counts / len(y)

    def _gini(self, y):
        return 1 - np.sum(self._get_probs(y) ** 2)

    def _entropy(self, y):
        return -np.sum(self._get_probs(y) * np.log2(self._get_probs(y)))

    def _misclassification(self, y):
        return 1 - np.max(self._get_probs(y))

    def _information_gain(self, left, right, y):
        n = len(y)
        n_l, n_r = len(y[left]), len(y[right])
        parent_loss = self._loss(y)
        child_loss = (
            n_l * self._loss(y[left]) + n_r * self._loss(y[right])
        ) / n
        return parent_loss - child_loss

    def _normal(self, dim, theta):
        v = np.zeros(self.ndim)
        v[0], v[dim] = np.sin(theta), np.cos(theta)
        return v

    def _get_split(self, X, dim, theta):
        if self.hyperbolic:
            prods = X @ self._normal(dim, theta)
            return prods < 0.0, prods >= 0.0
        else:
            return X[:, dim] < theta, X[:, dim] >= theta

    def _get_candidates(self, X, dim):
        if self.candidates == "data":
            X[:, dim][X[:, dim] == 0.0] = 1e-6
            thetas = np.arctan(X[:, 0] / X[:, dim])
            thetas = np.unique(np.sort(thetas))

            # Keep only those that are sufficiently far apart; take midpoints:
            thetas = thetas[
                np.where(np.abs(np.diff(thetas)) > self.min_dist)[0]
            ]
            return (thetas[:-1] + thetas[1:]) / 2.0
        elif self.candidates == "grid":
            return np.linspace(-np.pi / 4, np.pi / 4, 1000)

    def _fit_node(self, X, y, depth):
        if (
            depth == self.max_depth
            or len(y) <= self.min_samples_split
            or len(set(y)) == 1
        ):
            value, probs = self._leaf_values(y)
            return HyperbolicDecisionNode(value=value, probs=probs)
        best_dim, best_theta, best_score = None, None, -1
        for dim in self.dims:
            for theta in self._get_candidates(X=X, dim=dim):
                left, right = self._get_split(X=X, dim=dim, theta=theta)
                if (
                    np.min([len(y[left]), len(y[right])])
                    >= self.min_samples_leaf
                ):
                    score = self._information_gain(left, right, y)
                    if score > best_score:
                        best_dim, best_theta, best_score = dim, theta, score

        # Contingency for no split found:
        if best_score == -1:
            value, probs = self._leaf_values(y)
            return HyperbolicDecisionNode(value=value, probs=probs)

        # Populate:
        node = HyperbolicDecisionNode(feature=best_dim, theta=best_theta)
        left, right = self._get_split(X=X, dim=best_dim, theta=best_theta)
        node.left = self._fit_node(X=X[left], y=y[left], depth=depth + 1)
        node.right = self._fit_node(X=X[right], y=y[right], depth=depth + 1)
        return node

    def _leaf_values(self, y):
        # probs = self._get_probs(y)
        # return np.argmax(probs), probs
        _, inverse_y = np.unique(y, return_inverse=True)
        probs = np.bincount(inverse_y, minlength=len(self.classes_)) / len(y)
        return np.argmax(probs), probs

    def fit(self, X, y):
        self.ndim = X.shape[1]
        self.dims = range(1, self.ndim) if self.hyperbolic else range(self.ndim)
        self.classes_ = np.unique(y)
        self.tree = self._fit_node(X=X, y=y, depth=0)
        return self

    def _traverse(self, x, node):
        if node.value is not None:
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        return np.array([self._traverse(x, self.tree).probs for x in X])

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return np.mean(self.predict(X) == y)


class HyperbolicDecisionTreeRegressor(HyperbolicDecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = self._mse

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _information_gain(self, left, right, y):
        return self._loss(y) - (
            len(left) * self._loss(y[left]) + len(right) * self._loss(y[right])
        ) / len(y)

    def _leaf_values(self, y):
        return np.mean(y), None

    def predict_proba(self, X):
        raise NotImplementedError("Regression does not support predict_proba")

    def score(self, X, y, metric="mse"):
        """Return the mean accuracy on the given test data and labels"""
        y_hat = self.predict(X)
        if metric == "mse":
            return np.mean((y - y_hat) ** 2)
        elif metric == "rmse":
            return np.sqrt(np.mean((y - y_hat) ** 2))
        elif metric == "mae":
            return np.mean(np.abs(y - y_hat))
        elif metric in ["r2", "R2"]:
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
