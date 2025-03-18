"""Hyperbolic random forest"""

from typing import Tuple, Literal, Type, Optional

import numpy as np
from scipy import stats

from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin

from .tree import (
    DecisionTreeClassifier,
    HyperbolicDecisionTreeClassifier,
    DecisionTreeRegressor,
    HyperbolicDecisionTreeRegressor,
)


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Base class for random forests"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal["gini"] = "gini",
        weights: Optional[np.ndarray] = None,
        n_jobs: int = -1,
        tree_type: Type[DecisionTreeClassifier] = DecisionTreeClassifier,
        random_state: Optional[int] = None,
    ) -> None:
        # Random forest parallelization parameters
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        # Base decision tree parameters
        self.tree_params = {}
        self.max_depth = self.tree_params["max_depth"] = max_depth
        self.min_samples_split = self.tree_params["min_samples_split"] = min_samples_split
        self.min_samples_leaf = self.tree_params["min_samples_leaf"] = min_samples_leaf
        self.criterion = self.tree_params["criterion"] = criterion
        self.weights = self.tree_params["weights"] = weights

        # Actually initialize the forest
        self.tree_type = tree_type
        self.trees = self._get_trees()
        self.random_state = random_state

        # Check that the tree type is correct
        assert isinstance(self.trees[0], self.tree_type), "Tree type mismatch"

        # Curvature
        self.curvature: Optional[float] = None

    def _get_trees(self):
        return [self.tree_type(**self.tree_params) for _ in range(self.n_estimators)]

    def _generate_subsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random subsample of the data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(
        self, X: np.ndarray, y: np.ndarray, use_tqdm: bool = False, seed: Optional[int] = None
    ) -> "RandomForestClassifier":
        """
        Fit a decision tree to subsamples

        Args:
        -----
        X: np.ndarray (n_samples, n_features)
            Training data
        y: np.ndarray (n_samples,)
            Target labels
        use_tqdm: bool
            Use tqdm for progress bar (default: False)
        seed: int
            Random seed (default: None)

        Returns:
        --------
        self: RandomForestClassifier
            The fitted random forest
        """
        self.classes_ = np.unique(y)

        if seed is not None:
            self.random_state = seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of each sample in X.

        Args:
        -----
        X: np.ndarray (n_samples, n_features)
            Test data

        Returns:
        --------
        predictions: np.ndarray (n_samples,)
            Predicted classes
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return stats.mode(predictions, axis=0, keepdims=False)[0]

    def predict_proba(self, X):
        """
        Predict the class probabilities of each sample in X.

        Args:
        -----
        X: np.ndarray (n_samples, n_features)
            Test data

        Returns:
        --------
        predictions: np.ndarray (n_samples, n_classes)
            Predicted class probabilities
        """
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Args:
        -----
        X: np.ndarray (n_samples, n_features)
            Test data
        y: np.ndarray (n_samples,)
            True labels

        Returns:
        --------
        score: float
            Mean accuracy
        """
        return np.mean(self.predict(X) == y)


class RandomForestRegressor(RandomForestClassifier):
    """Random forest for regression problems. Inherits from RandomForestClassifier."""

    def __init__(self, **kwargs):
        super().__init__(tree_type=DecisionTreeRegressor, **kwargs)
        assert isinstance(self.trees[0], DecisionTreeRegressor)


class HyperbolicRandomForestClassifier(RandomForestClassifier):
    """Random forest for hyperbolic decision trees. Inherits from RandomForestClassifier."""

    def __init__(
        self,
        timelike_dim: int = 0,
        curvature: float = -1,
        skip_hyperboloid_check: bool = False,
        angle_midpoint_method: Literal["hyperbolic", "bisect"] = "hyperbolic",
        **kwargs
    ):
        super().__init__(tree_type=HyperbolicDecisionTreeClassifier, **kwargs)

        # Set special params
        self.skip_hyperboloid_check = self.tree_params["skip_hyperboloid_check"] = skip_hyperboloid_check
        self.angle_midpoint_method = self.tree_params["angle_midpoint_method"] = angle_midpoint_method

        # Fix curvature
        self.curvature = np.abs(curvature)
        for tree in self.trees:
            tree.curvature = np.abs(curvature)
        self.timelike_dim = self.tree_params["timelike_dim"] = timelike_dim
        assert isinstance(self.trees[0], HyperbolicDecisionTreeClassifier)


class HyperbolicRandomForestRegressor(RandomForestClassifier):
    """Random forest for hyperbolic regression problems. Inherits from RandomForestClassifier."""

    def __init__(
        self,
        timelike_dim: int = 0,
        curvature: float = -1,
        skip_hyperboloid_check: bool = False,
        angle_midpoint_method: Literal["hyperbolic", "bisect"] = "hyperbolic",
        **kwargs
    ):
        super().__init__(tree_type=HyperbolicDecisionTreeRegressor, **kwargs)

        # Set special params
        self.skip_hyperboloid_check = self.tree_params["skip_hyperboloid_check"] = skip_hyperboloid_check
        self.angle_midpoint_method = self.tree_params["angle_midpoint_method"] = angle_midpoint_method

        # Fix curvature
        self.curvature = np.abs(curvature)
        for tree in self.trees:
            tree.curvature = np.abs(curvature)
        self.timelike_dim = self.tree_params["timelike_dim"] = timelike_dim
        assert isinstance(self.trees[0], HyperbolicDecisionTreeRegressor)
