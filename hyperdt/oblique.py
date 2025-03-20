from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scikit_obliquetree.segmentor import MSE, MeanSegmentor
from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree

# Flag to indicate oblique trees are available
OBLIQUE_AVAILABLE = True


class HyperbolicHouseHolder(HyperbolicDecisionTree):
    """Hyperbolic version of oblique decision trees that uses Einstein midpoints for splits."""

    def __init__(self, *args, **kwargs):
        if "impurity" not in kwargs:
            kwargs["impurity"] = MSE()
        if "segmentor" not in kwargs:
            kwargs["segmentor"] = MeanSegmentor()

        super().__init__(backend="hhcart", *args, **kwargs)

    def _fix_node_recursive(self, node: Any, X_klein: np.ndarray) -> None:
        if node.is_leaf:
            return

        weights = node._weights
        projections = X_klein.dot(weights[:-1])

        left_mask = projections < weights[-1]
        right_mask = ~left_mask

        if np.any(left_mask) and np.any(right_mask):
            left_max = np.max(projections[left_mask])
            right_min = np.min(projections[right_mask])
            node._weights[-1] = self._midpoint(left_max, right_min)

        self._fix_node_recursive(node.left_child, X_klein[left_mask])
        self._fix_node_recursive(node.right_child, X_klein[right_mask])

    def _postprocess(self, X_klein: np.ndarray) -> None:
        self._fix_node_recursive(self.estimator_._root, X_klein)


class HyperbolicHouseHolderClassifier(HyperbolicHouseHolder, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="classification", **kwargs)


class HyperbolicHouseHolderRegressor(HyperbolicHouseHolder, RegressorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="regression", **kwargs)


class HyperbolicContinuouslyOptimized(HyperbolicDecisionTree):
    """Hyperbolic version of continuously optimized oblique decision trees."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, backend="co2", **kwargs)

    def _fix_node_recursive(self, node: Any, X_klein: np.ndarray) -> None:
        if node.is_leaf:
            return

        X_aug = np.hstack((X_klein, np.ones((X_klein.shape[0], 1))))
        weights = node._weights
        projections = X_aug @ weights

        left_mask = projections >= 0
        right_mask = ~left_mask

        left_min = np.min(projections[left_mask])
        right_max = np.max(projections[right_mask])
        midpoint = self._midpoint(right_max, left_min)

        scale_factor = abs(midpoint / weights[-1])
        node._weights = node._weights * scale_factor

        projections = X_aug @ node._weights
        left_mask = projections >= 0
        right_mask = ~left_mask

        self._fix_node_recursive(node.left_child, X_klein[left_mask])
        self._fix_node_recursive(node.right_child, X_klein[right_mask])


class HyperbolicContinuouslyOptimizedClassifier(HyperbolicContinuouslyOptimized, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        if "impurity" not in kwargs:
            kwargs["impurity"] = MSE()
        if "segmentor" not in kwargs:
            kwargs["segmentor"] = MeanSegmentor()
        super().__init__(*args, task="classification", **kwargs)

    def predict(self, X: ArrayLike) -> ArrayLike:
        return (super().predict(X) > 0.5).astype(int)


class HyperbolicContinuouslyOptimizedRegressor(HyperbolicContinuouslyOptimized, RegressorMixin):
    def __init__(self, *args, **kwargs):
        if "impurity" not in kwargs:
            kwargs["impurity"] = MSE()
        if "segmentor" not in kwargs:
            kwargs["segmentor"] = MeanSegmentor()
        super().__init__(*args, task="regression", **kwargs)
