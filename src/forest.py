"""Hyperbolic random forest"""

import numpy as np
from scipy import stats

from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from .tree import HyperbolicDecisionTreeClassifier


class HyperbolicRandomForestClassifier:
    def __init__(
        self, n_estimators=100, max_depth=3, min_samples=2, hyperbolic=True
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.hyperbolic = hyperbolic
        self.trees = [
            HyperbolicDecisionTreeClassifier(max_depth, min_samples, hyperbolic)
            for _ in range(n_estimators)
        ]

    def _generate_subsample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y, use_tqdm=False, njobs=False):
        trees = tqdm(self.trees) if use_tqdm else self.trees
        if tqdm:
            trees = tqdm(trees)
        if njobs > 1:
            Parallel(n_jobs=njobs)(
                delayed(tree.fit)(*self._generate_subsample(X, y))
                for tree in trees
            )
        else:
            for tree in trees:
                X_sample, y_sample = self._generate_subsample(X, y)
                tree.fit(X_sample, y_sample)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.squeeze(stats.mode(predictions, axis=0).mode)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
