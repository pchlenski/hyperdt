import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class HyperbolicDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Einstein midpoint implementation using gyrovector operations"""

    def __init__(self, max_depth=3, curvature=1.0, timelike_dim=0, skip_hyperboloid_check=True, **kwargs):
        self.max_depth = max_depth
        self.curvature = abs(curvature)
        self.timelike_dim = timelike_dim
        self.skip_hyperboloid_check = skip_hyperboloid_check  # TODO: use this
        self.tree = DecisionTreeClassifier(max_depth=max_depth, **kwargs)

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

    def _adjust_thresholds(self, node_id, X_klein, samples):
        tree = self.tree.tree_
        if tree.children_left[node_id] == -1:
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
        self._adjust_thresholds(tree.children_left[node_id], X_klein, left_samples)
        self._adjust_thresholds(tree.children_right[node_id], X_klein, right_samples)

    def fit(self, X, y):
        # Convert to Klein coordinates (x_d/x_0)
        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]

        self.tree.fit(X_klein, y)
        self._adjust_thresholds(0, X_klein, np.arange(len(X)))
        return self

    def predict(self, X):
        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]
        return self.tree.predict(X_klein)

    def predict_proba(self, X):
        x0 = X[:, self.timelike_dim]
        X_klein = np.delete(X, self.timelike_dim, axis=1) / x0[:, None]
        return self.tree.predict_proba(X_klein)
