"""Legacy implementations of hyperbolic decision trees.

This module includes the original implementation that requires geomstats,
and other legacy algorithms that may have different dependencies.
"""

try:
    from .tree import (
        HyperbolicDecisionTree,
        HyperbolicDecisionTreeClassifier,
        HyperbolicDecisionTreeRegressor,
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False