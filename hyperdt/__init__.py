"""HyperDT - Fast Hyperboloid Decision Tree Algorithms.

This package implements decision trees that operate natively in hyperbolic space.
The main implementation is in faster_tree.py, which provides a scikit-learn compatible
interface and supports multiple backends (sklearn, xgboost).

Legacy implementations that use geomstats are available in the legacy submodule.
"""

# Import the main implementations
from .faster_tree import (
    HyperbolicDecisionTree,
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)

# Check if XGBoost is available and import optional XGBoost implementations
try:
    from .faster_tree import (
        HyperbolicXGBoostClassifier,
        HyperbolicXGBoostRegressor,
    )
    XGBOOST_AVAILABLE = True
except (ImportError, AttributeError):
    XGBOOST_AVAILABLE = False

# Check if legacy module is available (requires geomstats)
try:
    from .legacy import LEGACY_AVAILABLE
except ImportError:
    LEGACY_AVAILABLE = False

__all__ = [
    "HyperbolicDecisionTree",
    "HyperbolicDecisionTreeClassifier",
    "HyperbolicDecisionTreeRegressor",
    "HyperbolicRandomForestClassifier",
    "HyperbolicRandomForestRegressor",
    "XGBOOST_AVAILABLE",
    "LEGACY_AVAILABLE",
]

# Add XGBoost classes to __all__ if available
if XGBOOST_AVAILABLE:
    __all__.extend(["HyperbolicXGBoostClassifier", "HyperbolicXGBoostRegressor"])