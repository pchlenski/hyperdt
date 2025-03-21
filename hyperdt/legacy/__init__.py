"""
Legacy implementations of hyperbolic decision trees.

This module includes the original implementation that requires geomstats,
and other legacy algorithms that may have different dependencies.
"""

try:
    # Import core tree components
    from .conversions import *
    from .ensemble import *

    # Import utility modules if available
    from .hyperbolic_trig import *
    from .tree import HyperbolicDecisionTreeClassifier, HyperbolicDecisionTreeRegressor
    from .visualization import *

    # Mark as available
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Try to import dataloaders
try:
    from .dataloaders import *

    DATALOADERS_AVAILABLE = True
except ImportError:
    DATALOADERS_AVAILABLE = False
