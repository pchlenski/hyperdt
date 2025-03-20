"""
XGBoost implementations for hyperbolic space.

This module provides XGBoost classifiers and regressors that operate
natively in hyperbolic space.
"""

from typing import Any
import tempfile
import os
import json
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree


class HyperbolicXGBoost(HyperbolicDecisionTree):
    """Wrapper around XGBoost trees to make them more amenable to hyperbolic space."""

    def __init__(self, *args, **kwargs):
        # Prevent bootstrap resampling, which is not supported currently.
        if "subsample" in kwargs:
            print("Warning: subsample is not supported currently. Setting to 1.0.")
        kwargs["subsample"] = 1.0
        super().__init__(backend="xgboost", *args, **kwargs)

    def _fix_node_recursive(self, tree: Any, node_id: int, X_klein: np.ndarray) -> None:
        """Fix the tree to use Einstein midpoints.

        Key to converting SKlearn names to XGBoost names:
        children_left -> left_children
        children_right -> right_children
        feature -> split_indices
        threshold -> split_conditions
        """

        if tree["left_children"][node_id] == -1:  # Leaf node
            return

        # Get feature and threshold
        feature = tree["split_indices"][node_id]
        threshold = tree["split_conditions"][node_id]

        # Get feature values
        feature_values = X_klein[:, feature]

        # Adjust threshold to be the average of closest values on either side
        left_mask = feature_values <= threshold
        right_mask = ~left_mask

        # Adjust this node's threshold using Einstein midpoints instead of naive averages as in base sklearn
        left_max = np.max(feature_values[left_mask])  # Closest point from left
        right_min = np.min(feature_values[right_mask])  # Closest point from right
        tree["split_conditions"][node_id] = self._einstein_midpoint(left_max, right_min)

        # Recurse
        self._fix_node_recursive(tree, tree["left_children"][node_id], X_klein[left_mask])
        self._fix_node_recursive(tree, tree["right_children"][node_id], X_klein[right_mask])

    def _postprocess(self, X_klein: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Dump the model to a temporary file and load it as a JSON
            self.estimator_.save_model(os.path.join(temp_dir, "xgb.json"))
            with open(os.path.join(temp_dir, "xgb.json"), "r") as f:
                model_json = json.load(f)

            # Actually fix the trees
            for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]:
                self._fix_node_recursive(tree, 0, X_klein)

            # Save the fixed model
            with open(os.path.join(temp_dir, "xgb_fixed.json"), "w") as f:
                json.dump(model_json, f)

            # Load the fixed model back into the estimator
            self.estimator_.load_model(os.path.join(temp_dir, "xgb_fixed.json"))


class HyperbolicXGBoostClassifier(HyperbolicXGBoost, ClassifierMixin):
    """This is just HyperbolicDecisionTree with backend="xgboost" and task="classification"."""

    def __init__(self, *args, **kwargs):
        super().__init__(task="classification", *args, **kwargs)


class HyperbolicXGBoostRegressor(HyperbolicXGBoost, RegressorMixin):
    """This is just HyperbolicDecisionTree with backend="xgboost" and task="regression"."""

    def __init__(self, *args, **kwargs):
        super().__init__(task="regression", *args, **kwargs)
