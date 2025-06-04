"""
LightGBM implementations for hyperbolic space.

This module provides LightGBM classifiers and regressors that operate
natively in hyperbolic space.
"""

import warnings
import re
from typing import Any, List

import lightgbm as lgb
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from ._base import HyperbolicDecisionTree

# Flag indicating this module could be imported successfully
LIGHTGBM_AVAILABLE = True


class HyperbolicLightGBM(HyperbolicDecisionTree):
    """Wrapper around LightGBM trees to make them more amenable to hyperbolic space."""

    def __init__(self, override_subsample: bool = True, *args, **kwargs):
        # Prevent bootstrap resampling, which is not supported currently.
        self.override_subsample = override_subsample
        if self.override_subsample and "bagging_fraction" in kwargs:
            warnings.warn(
                "Warning: bagging_fraction is not supported currently. Setting to 1.0.",
                UserWarning,
            )
            kwargs["bagging_fraction"] = 1.0
        elif not self.override_subsample and "bagging_fraction" in kwargs:
            warnings.warn(
                "Warning: bagging_fraction is not supported currently. Postprocessing will use entire dataset. Your results may be suboptimal.",
                UserWarning,
            )
            kwargs["bagging_fraction"] = 1.0
        super().__init__(backend="lightgbm", *args, **kwargs)

    def _fix_node_recursive(self, node: Any, X_klein: np.ndarray) -> None:
        if "left_child" not in node:  # Leaf node
            return

        # Get feature and threshold
        feature = node["split_feature"]
        threshold = node["threshold"]

        # Get threshold
        feature_values = X_klein[:, feature]

        # Adjust threshold to be the average of closest values on either side
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        left_max = np.max(feature_values[left_mask])
        right_min = np.min(feature_values[right_mask])
        node["threshold"] = float(self._midpoint(left_max, right_min))

        # Recurse
        self._fix_node_recursive(node["left_child"], X_klein[left_mask])
        self._fix_node_recursive(node["right_child"], X_klein[right_mask])

    def _postprocess(self, X_klein: np.ndarray) -> None:
        booster = self.estimator_.booster_
        model_dict = booster.dump_model()

        # Compute new thresholds by recursing through each tree
        new_thresholds: List[List[float]] = []
        for tree in model_dict["tree_info"]:
            struct = tree["tree_structure"]
            self._fix_node_recursive(struct, X_klein)
            new_thresholds.append(self._extract_thresholds_bfs(struct))

        # Split the text model into header, per-tree text, and tail
        model_str = booster.model_to_string()
        m = re.search(r"tree_sizes=([0-9 ]+)", model_str)
        if not m:
            return

        sizes = list(map(int, m.group(1).split()))
        prefix = model_str[: m.start()]
        start = m.end() + 2  # Skip the blank line after tree_sizes
        after = model_str[m.end() : start]
        idx = start
        trees: List[str] = []
        for sz in sizes:
            trees.append(model_str[idx : idx + sz])
            idx += sz
        tail = model_str[idx:]

        # Update threshold lines while respecting original formatting
        updated_trees: List[str] = []
        for tree_text, thr in zip(trees, new_thresholds):
            lines = tree_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("threshold="):
                    orig_vals = line.split("=")[1].split()
                    fmt_vals = []
                    for val, orig in zip(thr, orig_vals):
                        decimals = len(orig.split(".")[1]) if "." in orig else 0
                        fmt_vals.append(f"{val:.{decimals}f}")
                    lines[i] = "threshold=" + " ".join(fmt_vals)
                    break
            updated_trees.append("\n".join(lines))

        # Recompute tree sizes and rebuild the model string
        new_sizes = [len(t) for t in updated_trees]
        header = prefix + "tree_sizes=" + " ".join(str(s) for s in new_sizes) + after
        new_model_str = header + "".join(updated_trees) + tail

        new_booster = lgb.Booster(model_str=new_model_str)
        self.estimator_._Booster = new_booster

    def _extract_thresholds_bfs(self, node: Any) -> List[float]:
        """Return thresholds in LightGBM's node ID order."""
        # First, collect all nodes with their IDs
        node_thresholds = {}

        def collect_with_id(n, node_id):
            if "left_child" in n:
                node_thresholds[node_id] = float(n["threshold"])
                # Recurse to children
                if isinstance(n["left_child"], dict):
                    collect_with_id(n["left_child"], n["left_child"].get("split_index", len(node_thresholds)))
                if isinstance(n["right_child"], dict):
                    collect_with_id(n["right_child"], n["right_child"].get("split_index", len(node_thresholds)))

        # Start from root with ID 0
        collect_with_id(node, 0)

        # Return thresholds in order of node IDs
        return [node_thresholds[i] for i in sorted(node_thresholds.keys())]


class HyperbolicLGBMClassifier(HyperbolicLightGBM, ClassifierMixin):
    """Hyperbolic LightGBM classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(task="classification", *args, **kwargs)


class HyperbolicLGBMRegressor(HyperbolicLightGBM, RegressorMixin):
    """Hyperbolic LightGBM regressor."""

    def __init__(self, *args, **kwargs):
        super().__init__(task="regression", *args, **kwargs)
