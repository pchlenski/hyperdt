"""HyperDT - Fast Hyperboloid Decision Tree Algorithms.

This package implements decision trees that operate natively in hyperbolic space.
It provides a scikit-learn compatible interface and supports multiple backends
(sklearn decision trees, random forests, and optionally XGBoost).

Legacy implementations that use geomstats are available in the legacy submodule.
"""

# Import base and core implementations
from ._base import HyperbolicDecisionTree
from .tree import (
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
)
from .ensemble import (
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)

# Check if XGBoost is available and import optional XGBoost implementations
try:
    from .xgboost import (
        HyperbolicXGBoostClassifier,
        HyperbolicXGBoostRegressor,
        XGBOOST_AVAILABLE,
    )
except ImportError:
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
