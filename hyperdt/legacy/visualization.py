"""Utilities for visualizing hyperbolic decision trees."""

from typing import List, Tuple, Union, Literal, Optional


import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

from .conversions import convert

GRID_SIZE = 2001
STYLES = ["solid", "dashed", "dotted", "dashdot", "solid", "dashed", "dotted", "dashdot"]  # Whatever


def _get_geodesic(
    dim: int,
    theta: float,
    t_dim: int = 1,
    n_dim: int = 3,
    start_t: float = -10,
    end_t: float = 10,
    num_points: int = 1000,
    geometry: Literal["poincare", "klein"] = "poincare",
    timelike_dim: int = 0,
) -> np.ndarray:
    """
    Get num_points points from intersection of a hyperplane and a geodesic.

    This is a special case when we have axis-aligned hyperplanes parameterized by a
    single dimension and angle. The more general case is in 'geodesics.py'.
    """
    _t = np.linspace(start_t, end_t, num_points)
    geodesic = np.zeros((num_points, n_dim))

    # t dimension just gets sinh
    geodesic[:, t_dim] = np.sinh(_t)

    # Coefficient stretches unit vector to hit the manifold
    coef = np.sqrt(-1 / np.cos(2 * theta))  # sqrt(-sec(2 theta))
    geodesic[:, dim] = np.cosh(_t) * coef * np.cos(theta)
    geodesic[:, timelike_dim] = np.cosh(_t) * coef * np.sin(theta)

    return convert(geodesic, initial="hyperboloid", final=geometry, timelike_dim=timelike_dim)


def _get_mask(boundary_dim: int, geodesic: np.ndarray) -> np.ndarray:
    """
    Return all points such that <x, boundary> < 0 (left side of boundary).

    This is used to restrict where decision boundaries are plotted, so that we can
    visualize boundaries only where they are actually relevant (e.g. if you're on the
    right side of split 1, don't plot split 2 in the left half)
    """
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE))

    # Interpolate geodesic as a function of the independent dimension
    boundary_dim = boundary_dim - 1  # Input is {1, 2} but want {0, 1}
    independent_dim = 1 - boundary_dim  # Assume {0, 1}
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


def plot_boundary(
    boundary_dim: int,
    boundary_theta: float,
    t_dim: int = -1,
    geometry: Literal["poincare", "klein"] = "poincare",
    ax: Optional["Axes"] = None,
    timelike_dim: int = 0,
    color: str = "red",
    mask: Optional[np.ndarray] = None,
    return_mask: bool = False,
    style: str = "solid",
) -> Union["Axes", Tuple["Axes", np.ndarray]]:
    """
    Plot a single decision boundary of a hyperbolic decision tree

    Args:
    ----
    boundary_dim: int
        Which timelike dimension is nonzero for the deecision hyperplane (0, 1, 2)
    boundary_theta: float
        Inclination angle of the decision hyperplane (in radians, from pi/4 to 3pi/4)
    t_dim: int
        Which dimension is free for parameterizing a geodesic? (i.e. not timelike, not boundary_dim) (0, 1, 2)
    geometry: str
        Which geometry are we plotting in? ("poincare", "klein")
    ax: matplotlib.axes.Axes
        Axes to plot on. If None, a new figure is created
    timelike_dim: int
        Which dimension is timelike? (0, 1, 2)
    color: str
        Color of the decision boundary
    mask: np.ndarray
        Mask to apply to the decision boundary. If None, no mask is applied. Useful for recursive plotting.
    return_mask: bool
        Whether to return the mask after plotting. Useful for recursive plotting.
    style: str
        Line style for the decision boundary
    """
    # Set t_dim: we assume total number of dims is 3
    if t_dim == -1:
        dims = [0, 1, 2]
        dims.remove(boundary_dim)
        dims.remove(timelike_dim)
        t_dim = dims[0]

    # Get geodesics; project
    geodesic_points = _get_geodesic(
        dim=boundary_dim, theta=boundary_theta, geometry=geometry, t_dim=t_dim, timelike_dim=timelike_dim
    )

    # Get new mask
    new_mask = _get_mask(boundary_dim=boundary_dim, geodesic=geodesic_points)

    # Apply mask to geodesic points
    if mask is not None:
        # This is equivalent to throwing out rows outside the mask grid:
        geodesic_points = np.stack(_apply_mask(geodesic_points[:, 0], geodesic_points[:, 1], mask), axis=1)

    # Init figure
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    assert ax is not None  # For type checker

    # Verify geodesics lie inside unit circle:
    if np.all(np.linalg.norm(geodesic_points, axis=1) <= 1):
        ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], c=color, linestyle=style)  # type: ignore
    else:
        print(f"Geodesic points lie outside unit circle:\t{boundary_dim} {boundary_theta/np.pi:.3f}*pi {t_dim}")

    if return_mask:
        return ax, new_mask
    else:
        return ax


def _plot_tree_recursive(
    node: "DecisionNode",
    ax: "Axes",
    colors: List[str],
    mask: Optional[np.ndarray],
    depth: int,
    n_classes: int,
    minkowski: bool = False,
    **kwargs,
) -> "Axes":
    """Plot the decision boundary of a node and its children recursively."""
    if node.value is not None:  # Leaf case
        _xx, _yy = np.meshgrid(np.linspace(-1, 1, 2001), np.linspace(-1, 1, 2001))
        # Match scatterplot colors
        majority_class = node.value
        class_colors = plt.get_cmap("viridis", n_classes)
        color = class_colors(majority_class)

        # Don't extend past unit circle
        if mask is not None:
            mask_circle = _xx**2 + _yy**2 <= 1
            mask = mask & mask_circle

        # Make image
        image = np.zeros(shape=(2001, 2001, 4))
        if mask is not None:
            image[mask] = (color[0], color[1], color[2], 0.5)
            image[~mask] = (0, 0, 0, 0)
        ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), aspect="auto")
        return ax
    else:
        assert node.theta is not None  # For type checker
        theta = -node.theta if minkowski else node.theta
        ax, new_mask = plot_boundary(
            boundary_dim=node.feature,  # type: ignore
            boundary_theta=theta,  # type: ignore
            color=colors[depth],
            mask=mask,
            return_mask=True,
            ax=ax,
            style=STYLES[depth % len(STYLES)],
            **kwargs,
        )  # type: ignore (type checker seems to break here)
        reuse = {
            "ax": ax,
            "colors": colors,
            "depth": depth + 1,
            "n_classes": n_classes,
            "minkowski": minkowski,
            **kwargs,
        }

        # "Mask is None" = don't use mask at all
        if mask is not None:
            mask_left = mask & new_mask
            mask_right = mask & ~new_mask
        else:
            mask_left = mask_right = None

        # Negated angles = flipped dot-product = flipped mask
        if minkowski:
            mask_left, mask_right = mask_right, mask_left

        assert node.left is not None and node.right is not None  # For type checker
        ax = _plot_tree_recursive(node.left, mask=mask_left, **reuse)
        ax = _plot_tree_recursive(node.right, mask=mask_right, **reuse)
        return ax


def _val_to_index(val: float):
    """Convert a 2-place decimal to an index in (0, 200)"""
    assert val >= -1.0 and val <= 1.0
    val = np.round(val, 3)
    return int(val * 1000) - 1001


def _apply_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a mask to x and y coordinates."""
    assert len(x) == len(y)
    x_out = []
    y_out = []
    for i, j in zip(x, y):
        # Note flipped coordinates: (row, column as expected)
        if mask[_val_to_index(j), _val_to_index(i)]:
            x_out.append(i)
            y_out.append(j)

    return np.array(x_out), np.array(y_out)


def plot_tree(
    hdt: "HyperbolicDecisionTreeClassifier",
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    geometry: Literal["poincare", "klein"] = "poincare",
    timelike_dim: int = 0,
    masked: bool = True,
    ax: Optional["Axes"] = None,
    **kwargs,
) -> "Axes":
    """
    Plot data and all decision boundaries of a hyperbolic decision tree.

    Args:
    ----
    hdt: hyperdt.tree.HyperbolicDecisionTree
        Hyperbolic decision tree to plot
    X: np.ndarray (n_samples, 3)
        Data to plot (2D, hyperbolic coordinates)
    y: np.ndarray (n_samples,)
        Labels for data
    geometry: str
        Which geometry are we converting X to? ("poincare", "klein")
    timelike_dim: int
        Which dimension is timelike? (0, 1, 2)
    masked: bool
        Whether to apply a mask to the decision boundaries
    ax: matplotlib.axes.Axes
        Axes to plot on. If None, a new figure is created
    kwargs: dict
        Additional keyword arguments to pass to plot_boundary
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    assert ax is not None  # For type checker

    # Plot data
    if X is not None and y is not None:
        X = convert(X, initial="hyperboloid", final=geometry, timelike_dim=timelike_dim)
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap="viridis", marker="o", s=49, edgecolors="k", linewidths=1  # type: ignore
        )

    # Get colors
    colors = ["red"] * hdt.max_depth

    # Initialize mask: 2001x2001 grid makes rounding work in the _apply_mask lookups
    mask = np.full(shape=(2001, 2001), fill_value=True) if masked else None

    # Plot recursively; get legend
    ax = _plot_tree_recursive(
        hdt.tree,
        ax=ax,
        geometry=geometry,
        timelike_dim=0,
        colors=colors,
        depth=0,
        mask=mask,
        n_classes=len(hdt.classes_),
        minkowski=(hdt.dot_product == "sparse_minkowski"),
        **kwargs,
    )
    ax.legend(handles=[Line2D([0], [0], color=c, label=f"Depth {i}") for i, c in enumerate(colors)])

    # Draw unit circle
    _x = np.linspace(-1, 1, 1000)
    _y = np.sqrt(1 - _x**2)
    ax.plot(_x, _y, c="black")
    ax.plot(_x, -_y, c="black")

    # Set axis limits
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    return ax
