"""
HyperDT - Fast Hyperboloid Decision Tree Algorithms.

This package implements decision trees that operate natively in hyperbolic space.
It provides a scikit-learn compatible interface and supports multiple backends
(sklearn decision trees, random forests, and optionally XGBoost).

Legacy implementations that use geomstats are available in the legacy submodule.
"""

# Import base and core implementations
from ._base import HyperbolicDecisionTree
from .ensemble import (
    HyperbolicRandomForestClassifier,
    HyperbolicRandomForestRegressor,
)
from .tree import (
    HyperbolicDecisionTreeClassifier,
    HyperbolicDecisionTreeRegressor,
)

# Check if XGBoost is available and import optional XGBoost implementations
try:
    from .xgboost import (
        XGBOOST_AVAILABLE,
        HyperbolicXGBoostClassifier,
        HyperbolicXGBoostRegressor,
    )
except ImportError:
    XGBOOST_AVAILABLE = False

# Check if legacy module is available (requires geomstats)
try:
    from .legacy import LEGACY_AVAILABLE
except ImportError:
    LEGACY_AVAILABLE = False

# Check if scikit-obliquetree is available and import oblique implementations
try:
    from .oblique import (
        OBLIQUE_AVAILABLE,
        HyperbolicContinuouslyOptimizedClassifier,
        HyperbolicContinuouslyOptimizedRegressor,
        HyperbolicHouseHolderClassifier,
        HyperbolicHouseHolderRegressor,
    )
except ImportError:
    OBLIQUE_AVAILABLE = False

__all__ = [
    "HyperbolicDecisionTree",
    "HyperbolicDecisionTreeClassifier",
    "HyperbolicDecisionTreeRegressor",
    "HyperbolicRandomForestClassifier",
    "HyperbolicRandomForestRegressor",
    "XGBOOST_AVAILABLE",
    "LEGACY_AVAILABLE",
    "OBLIQUE_AVAILABLE",
]

# Add XGBoost classes to __all__ if available
if XGBOOST_AVAILABLE:
    __all__.extend(["HyperbolicXGBoostClassifier", "HyperbolicXGBoostRegressor"])

# Add Oblique decision tree classes to __all__ if available
if OBLIQUE_AVAILABLE:
    __all__.extend(
        [
            "HyperbolicHouseHolderClassifier",
            "HyperbolicHouseHolderRegressor",
            "HyperbolicContinuouslyOptimizedClassifier",
            "HyperbolicContinuouslyOptimizedRegressor",
        ]
    )
