"""Utilities for visualizing hyperbolic decision trees."""

from typing import Any, List, Literal, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.interpolate import interp1d

from .conversions import convert

# Constants
GRID_SIZE = 2001
UNIT_CIRCLE_POINTS = 1000
PLOT_SIZE = 10
GRID_RANGE = np.linspace(-1, 1, GRID_SIZE)
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]
BOUNDARY_PLOT_POINTS = 1000
BOUNDARY_PARAM_RANGE = (-10, 10)
GeometryType = Literal["poincare", "klein"]

# Default set of distinct colors for classes
DEFAULT_COLORS = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#CCB974",  # yellow/tan
    "#64B5CD",  # light blue
    "#A95D00",  # brown
    "#CE78B3",  # pink
    "#B3B3B3",  # gray
    "#2F8AC4",  # darker blue
]


def _get_geodesic(
    dim: int,
    theta: float,
    t_dim: int = 1,
    n_dim: int = 3,
    geometry: GeometryType = "poincare",
    timelike_dim: int = 0,
) -> np.ndarray:
    """
    Get points from intersection of a hyperplane and a geodesic.

    This is a special case for axis-aligned hyperplanes parameterized by a
    single dimension and angle.
    """
    start_t, end_t = BOUNDARY_PARAM_RANGE
    _t = np.linspace(start_t, end_t, BOUNDARY_PLOT_POINTS)
    geodesic = np.zeros((BOUNDARY_PLOT_POINTS, n_dim))

    # t dimension gets sinh
    geodesic[:, t_dim] = np.sinh(_t)

    # Coefficient stretches unit vector to hit the manifold
    coef = np.sqrt(-1 / np.cos(2 * theta))  # sqrt(-sec(2 theta))
    geodesic[:, dim] = np.cosh(_t) * coef * np.cos(theta)
    geodesic[:, timelike_dim] = np.cosh(_t) * coef * np.sin(theta)

    return convert(geodesic, initial="hyperboloid", final=geometry, timelike_dim=timelike_dim)


def _get_mask(boundary_dim: int, geodesic: np.ndarray) -> np.ndarray:
    """
    Return points where <x, boundary> < 0 (left side of boundary).
    """
    _xx, _yy = np.meshgrid(GRID_RANGE, GRID_RANGE)

    # Adjust dimensions for interpolation (convert from 1-indexed to 0-indexed)
    boundary_dim = boundary_dim - 1  # Input is {1, 2} but want {0, 1}
    independent_dim = 1 - boundary_dim  # Assume {0, 1}
    
    # Interpolate geodesic as a function of independent dimension
    geodesic_interp = interp1d(
        geodesic[:, independent_dim],
        geodesic[:, boundary_dim],
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )
    geodesic_boundary = geodesic_interp(_yy)
    mask = _xx < geodesic_boundary
    if boundary_dim == 1:
        mask = mask.T
    return mask


def _apply_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a mask to x and y coordinates."""
    x_out, y_out = [], []
    for i, j in zip(x, y):
        if -1 <= i <= 1 and -1 <= j <= 1:  # Check bounds
            idx_i = int(round((i + 1) * (GRID_SIZE - 1) / 2))
            idx_j = int(round((j + 1) * (GRID_SIZE - 1) / 2))
            if 0 <= idx_i < GRID_SIZE and 0 <= idx_j < GRID_SIZE and mask[idx_j, idx_i]:
                x_out.append(i)
                y_out.append(j)
    return np.array(x_out), np.array(y_out)


def plot_boundary(
    boundary_dim: int,
    boundary_theta: float,
    geometry: GeometryType = "poincare",
    ax: Optional[Axes] = None,
    t_dim: int = -1,
    timelike_dim: int = 0,
    color: str = "red",
    mask: Optional[np.ndarray] = None,
    return_mask: bool = False,
    style: str = "solid",
    **kwargs,
) -> Union[Axes, Tuple[Axes, np.ndarray]]:
    """
    Plot a single decision boundary of a hyperbolic decision tree
    
    Args:
        boundary_dim: Which dimension for the decision hyperplane (0, 1, 2)
        boundary_theta: Inclination angle of the hyperplane (radians, pi/4 to 3pi/4)
        geometry: Which geometry to plot in ("poincare", "klein")
        ax: Axes to plot on. If None, a new figure is created
        t_dim: Free dimension for parameterizing geodesic (0, 1, 2)
        timelike_dim: Which dimension is timelike (0, 1, 2)
        color: Color of the decision boundary
        mask: Mask to apply to the boundary (for recursive plotting)
        return_mask: Whether to return the mask after plotting
        style: Line style for the boundary
        **kwargs: Additional parameters
    """
    # Set t_dim if not provided
    if t_dim == -1:
        dims = [0, 1, 2]
        dims.remove(boundary_dim)
        dims.remove(timelike_dim)
        t_dim = dims[0]

    # Get geodesics
    geodesic_points = _get_geodesic(
        dim=boundary_dim, 
        theta=boundary_theta, 
        geometry=geometry, 
        t_dim=t_dim, 
        timelike_dim=timelike_dim
    )

    # Get new mask
    new_mask = _get_mask(boundary_dim=boundary_dim, geodesic=geodesic_points)

    # Apply mask to geodesic points if provided
    if mask is not None:
        geodesic_points = np.stack(
            _apply_mask(geodesic_points[:, 0], geodesic_points[:, 1], mask), 
            axis=1
        )

    # Initialize figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    ax = cast(Axes, ax)  # For type checker

    # Plot geodesics if inside unit circle
    norm = np.linalg.norm(geodesic_points, axis=1)
    valid_points = geodesic_points[norm <= 1]
    if len(valid_points) > 0:
        ax.plot(valid_points[:, 0], valid_points[:, 1], c=color, linestyle=style)
    else:
        print(f"Warning: Geodesic points lie outside unit circle: {boundary_dim} {boundary_theta/np.pi:.3f}*pi {t_dim}")

    if return_mask:
        return ax, new_mask
    return ax


def _plot_tree_recursive(
    node: Any,
    ax: Axes,
    boundary_colors: List[str],
    class_colors: List[str],
    mask: Optional[np.ndarray],
    depth: int,
    geometry: GeometryType = "poincare",
    timelike_dim: int = 0,
    minkowski: bool = False,
    class_mapping: Optional[dict] = None,
    **kwargs,
) -> Axes:
    """Plot a node and its children recursively."""
    # Leaf node case
    if node.value is not None:
        _xx, _yy = np.meshgrid(GRID_RANGE, GRID_RANGE)
        
        # Get the color for this node's class
        majority_class = node.value
        if class_mapping is not None:
            color_idx = class_mapping[majority_class]
        else:
            color_idx = majority_class
            
        # Get RGBA values for the color
        rgba_color = to_rgba(class_colors[color_idx])

        # Apply unit circle mask
        if mask is not None:
            mask_circle = _xx**2 + _yy**2 <= 1
            mask = mask & mask_circle

        # Create and display image
        image = np.zeros(shape=(GRID_SIZE, GRID_SIZE, 4))
        if mask is not None:
            # Set the color with some transparency
            image[mask] = (rgba_color[0], rgba_color[1], rgba_color[2], 0.5)
        ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), aspect="auto")
        return ax
    
    # Internal node case
    else:
        assert node.theta is not None and node.feature is not None
        
        # Adjust theta for Minkowski space if needed
        theta = -node.theta if minkowski else node.theta
        
        # Get color for this depth level
        color = boundary_colors[depth % len(boundary_colors)]
        
        # Plot boundary and get new mask
        ax, new_mask = cast(Tuple[Axes, np.ndarray], plot_boundary(
            boundary_dim=node.feature,
            boundary_theta=theta,
            color=color,
            mask=mask,
            return_mask=True,
            ax=ax,
            style=LINE_STYLES[depth % len(LINE_STYLES)],
            geometry=geometry,
            timelike_dim=timelike_dim,
            **kwargs,
        ))
        
        # Common parameters for recursive calls
        reuse = {
            "ax": ax,
            "boundary_colors": boundary_colors,
            "class_colors": class_colors,
            "depth": depth + 1,
            "geometry": geometry,
            "timelike_dim": timelike_dim,
            "minkowski": minkowski,
            "class_mapping": class_mapping,
            **kwargs,
        }

        # Apply masks for child nodes
        if mask is not None:
            mask_left = mask & new_mask
            mask_right = mask & ~new_mask
            # Flip masks for Minkowski space
            if minkowski:
                mask_left, mask_right = mask_right, mask_left
        else:
            mask_left = mask_right = None

        # Recursively plot children
        assert node.left is not None and node.right is not None
        ax = _plot_tree_recursive(node.left, mask=mask_left, **reuse)
        ax = _plot_tree_recursive(node.right, mask=mask_right, **reuse)
        
        return ax


def plot_tree(
    hdt: Any,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    geometry: GeometryType = "poincare",
    timelike_dim: int = 0,
    masked: bool = True,
    ax: Optional[Axes] = None,
    class_colors: Optional[List[str]] = None,
    boundary_colors: Optional[List[str]] = None,
    **kwargs,
) -> Axes:
    """
    Plot data and all decision boundaries of a hyperbolic decision tree.

    Args:
        hdt: Hyperbolic decision tree to plot
        X: Data points to visualize (optional)
        y: Labels for data points (optional)
        geometry: Which geometry to use ("poincare", "klein")
        timelike_dim: Which dimension is timelike (0, 1, 2)
        masked: Whether to apply masks to decision boundaries
        ax: Axes to plot on (creates new figure if None)
        class_colors: List of colors for each class (uses DEFAULT_COLORS if None)
        boundary_colors: List of colors for decision boundaries (uses ["red"] by default)
        **kwargs: Additional parameters for boundary plotting
    """
    # Initialize plot
    if ax is None:
        _, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    ax = cast(Axes, ax)

    # Use default colors if none provided
    if class_colors is None:
        class_colors = DEFAULT_COLORS
    
    if boundary_colors is None:
        boundary_colors = ["red"]

    # Create mapping of original class values to sequential indices
    unique_classes = sorted(hdt.classes_)
    class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
    
    # Plot data points if provided
    if X is not None and y is not None:
        X_converted = convert(X, initial="hyperboloid", final=geometry, timelike_dim=timelike_dim)
        
        # Map class values to colors
        color_list = [class_colors[class_mapping[cls]] for cls in y]
        
        ax.scatter(
            X_converted[:, 0], X_converted[:, 1], 
            c=color_list, marker="o", s=49, 
            edgecolors="k", linewidths=1
        )

    # Initialize mask
    mask = np.full(shape=(GRID_SIZE, GRID_SIZE), fill_value=True) if masked else None

    # Plot tree recursively
    ax = _plot_tree_recursive(
        hdt.tree,
        ax=ax,
        boundary_colors=boundary_colors,
        class_colors=class_colors,
        depth=0,
        mask=mask,
        geometry=geometry,
        timelike_dim=timelike_dim,
        minkowski=(hdt.dot_product == "sparse_minkowski"),
        class_mapping=class_mapping,
        **kwargs,
    )
    
    # Add legend for boundary depths
    depth_handles = [
        Line2D([0], [0], color=boundary_colors[i % len(boundary_colors)], 
               linestyle=LINE_STYLES[i % len(LINE_STYLES)], label=f"Depth {i}") 
        for i in range(min(hdt.max_depth, 4))  # Only show first few depths to avoid clutter
    ]
    
    # Add legend for classes
    class_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], 
               markersize=8, label=f"Class {cls}", markeredgecolor='k', linestyle='')
        for i, cls in enumerate(unique_classes)
    ]
    
    # Combine legends
    ax.legend(handles=depth_handles + class_handles, loc='best')

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, UNIT_CIRCLE_POINTS)
    ax.plot(np.cos(theta), np.sin(theta), c="black", lw=2)

    # Set axis limits
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    return ax