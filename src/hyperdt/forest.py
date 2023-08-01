"""Hyperbolic random forest"""

import numpy as np
from scipy import stats

from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from .tree import HyperbolicDecisionTreeClassifier


class HyperbolicRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_dist=0,
        hyperbolic=True,
        criterion="gini",
        n_jobs=-1,
        timelike_dim=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.hyperbolic = hyperbolic
        self.min_dist = min_dist
        self.criterion = criterion
        self.timelike_dim = timelike_dim
        self.trees = [
            HyperbolicDecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_dist=min_dist,
                hyperbolic=hyperbolic,
                criterion=criterion,
                timelike_dim=timelike_dim,
            )
            for _ in range(n_estimators)
        ]
        self.n_jobs = n_jobs

    def _generate_subsample(self, X, y):
        """Generate a random subsample of the data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y, use_tqdm=False):
        """Fit a decision tree to subsamples"""
        trees = tqdm(self.trees) if use_tqdm else self.trees
        if self.n_jobs != 1:
            fitted_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.fit)(*self._generate_subsample(X, y))
                for tree in trees
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
        return np.squeeze(stats.mode(predictions, axis=0).mode)

    def predict_proba(self, X):
        """Predict the class probabilities of each sample in X"""
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return np.mean(self.predict(X) == y)
