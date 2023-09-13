"""Hyperbolic random forest"""

import numpy as np
from scipy import stats

from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from .tree import DecisionTreeClassifier, HyperbolicDecisionTreeClassifier


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        weights=None,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.tree_type = DecisionTreeClassifier
        self.n_jobs = n_jobs
        self.tree_params = {}
        self.max_depth = self.tree_params["max_depth"] = max_depth
        self.min_samples_split = self.tree_params["min_samples_split"] = min_samples_split
        self.min_samples_leaf = self.tree_params["min_samples_leaf"] = min_samples_leaf
        self.criterion = self.tree_params["criterion"] = criterion
        self.weights = self.tree_params["weights"] = weights
        self.trees = self._get_trees()

    def _get_trees(self):
        return [self.tree_type(**self.tree_params) for _ in range(self.n_estimators)]

    def _generate_subsample(self, X, y):
        """Generate a random subsample of the data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y, use_tqdm=False):
        """Fit a decision tree to subsamples"""
        self.classes_ = np.unique(y)

        # Fit decision trees individually (parallelized):
        trees = tqdm(self.trees) if use_tqdm else self.trees
        if self.n_jobs != 1:
            fitted_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.fit)(*self._generate_subsample(X, y)) for tree in trees
            )
            self.trees = fitted_trees
        else:
            for tree in trees:
                X_sample, y_sample = self._generate_subsample(X, y)
                tree.fit(X_sample, y_sample)
        return self

    def predict(self, X):
        """Predict the class of each sample in X"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return stats.mode(predictions, axis=0, keepdims=False)[0]

    def predict_proba(self, X):
        """Predict the class probabilities of each sample in X"""
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return np.mean(self.predict(X) == y)


class HyperbolicRandomForestClassifier(RandomForestClassifier):
    def __init__(self, timelike_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.timelike_dim = self.tree_params["timelike_dim"] = timelike_dim
        self.tree_type = HyperbolicDecisionTreeClassifier
