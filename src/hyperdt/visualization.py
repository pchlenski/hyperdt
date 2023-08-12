import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton
from scipy.interpolate import interp1d

from .conversions import convert


def _find_intersection(D, theta, t, t_dim=1, n_dim=3):
    """Get point on the intersection of a hyperplane and a hyperboloid."""
    # Define the hyperplane equation
    def hyperplane(x0, xD):
        return x0 * np.sin(theta) + xD * np.cos(theta)

    # Define the hyperboloid equation: assume all other coordinates are 0
    def hyperboloid_simplified(x0, x1, xD):
        return x1 ** 2 + xD ** 2 - x0 ** 2 + 1

    # Solve for x_0 using the hyperplane equation
    def solve_x0(xD):
        return -xD / np.tan(theta)

    # Substitute x_0 into the hyperboloid equation and solve for x_D
    def equation_to_solve(xD):
        return hyperboloid_simplified(x0=solve_x0(xD), x1=t, xD=xD)

    # Input validation:
    if D == t_dim:
        raise ValueError("dim and t_dim must be different")

    xD_solution = newton(equation_to_solve, 0)
    x0_solution = solve_x0(xD_solution)

    # return x_0_solution, x_D_solution, t
    out_vec = np.zeros(n_dim)
    out_vec[t_dim] = t
    if theta < 0:
        out_vec[0] = x0_solution
        out_vec[D] = xD_solution
    else:
        out_vec[0] = -x0_solution
        out_vec[D] = -xD_solution

    return out_vec


def _get_geodesic(
    dim,
    theta,
    t_dim=1,
    n_dim=3,
    start_t=-100,
    end_t=100,
    num_points=1000,
    geometry="poincare",
    timelike_dim=0,
):
    """Get num_points points from intersection of a hyperplane and a geodesic."""
    geodesic = np.stack(
        [
            _find_intersection(dim, theta, t, t_dim=t_dim, n_dim=n_dim)
            for t in np.linspace(start_t, end_t, num_points)
        ]
    )
    return convert(
        geodesic,
        initial="hyperboloid",
        final=geometry,
        timelike_dim=timelike_dim,
    )


def _get_mask(boundary_dim, geodesic):
    """Return all points such that <x, boundary> < 0 (left side of boundary)"""
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, 2001), np.linspace(-1, 1, 2001))

    # Interpolate geodesic as a function of the independent dimension
    boundary_dim = boundary_dim - 1  # Input is {1, 2} but want {0, 1}
    independent_dim = 1 - boundary_dim  # Assume {0, 1}
    geodesic_interp = interp1d(
        geodesic[:, independent_dim],
        geodesic[:, boundary_dim],
        bounds_error=False,
        fill_value="extrapolate",
    )
    geodesic_boundary = geodesic_interp(_yy)
    mask = _xx < geodesic_boundary
    if boundary_dim == 1:
        mask = mask.T
    return mask


def plot_boundary(
    hdt_node,
    t_dim=None,
    geometry="poincare",
    ax=None,
    timelike_dim=0,
    color="red",
    mask=None,
    return_mask=False,
):
    """Plot decision boundaries of a hyperbolic decision tree"""
    # Get decision boundary parameters
    boundary_dim = hdt_node.feature
    boundary_theta = hdt_node.theta

    # Set t_dim: we assume total number of dims is 3
    if t_dim is None:
        dims = [0, 1, 2]
        dims.remove(boundary_dim)
        dims.remove(timelike_dim)
        t_dim = dims[0]

    # Get geodesics; project
    geodesic_points = _get_geodesic(
        dim=boundary_dim,
        theta=boundary_theta,
        geometry=geometry,
        t_dim=t_dim,
        timelike_dim=timelike_dim,
    )

    # Get new mask
    new_mask = _get_mask(boundary_dim=boundary_dim, geodesic=geodesic_points)

    # Apply mask to geodesic points
    if mask is not None:
        # This is equivalent to throwing out rows outside the mask grid:
        geodesic_points = np.stack(
            _apply_mask(geodesic_points[:, 0], geodesic_points[:, 1], mask),
            axis=1,
        )

    # Init figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Verify geodesics lie inside unit circle:
    if np.all(np.linalg.norm(geodesic_points, axis=1) <= 1):
        ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], c=color)
    else:
        print(
            f"Geodesic points lie outside unit circle:\t{boundary_dim} {boundary_theta/np.pi:.3f}*pi {t_dim}"
        )

    if return_mask:
        return ax, new_mask
    else:
        return ax


def _plot_tree_recursive(node, ax, colors, mask, depth, n_classes, **kwargs):
    """Plot the decision boundary of a node and its children recursively."""
    if node.value is not None:  # Leaf case
        _xx, _yy = np.meshgrid(
            np.linspace(-1, 1, 2001), np.linspace(-1, 1, 2001)
        )
        # Match scatterplot colors
        majority_class = node.value
        class_colors = plt.get_cmap("Paired", n_classes)
        color = class_colors(majority_class)

        # Don't extend past unit circle
        mask_circle = _xx ** 2 + _yy ** 2 <= 1
        mask = mask & mask_circle

        # Make image
        image = np.zeros(shape=(2001, 2001, 4))
        image[mask] = (color[0], color[1], color[2], 0.2)
        image[~mask] = (0, 0, 0, 0)
        ax.imshow(image, origin="lower", extent=[-1, 1, -1, 1], aspect="auto")
        return ax
    else:
        ax, new_mask = plot_boundary(
            node,
            color=colors[depth],
            mask=mask,
            return_mask=True,
            ax=ax,
            **kwargs,
        )
        reuse = {
            "ax": ax,
            "colors": colors,
            "depth": depth + 1,
            "n_classes": n_classes,
            **kwargs,
        }

        # "Mask is None" = don't use mask at all
        if mask is not None:
            mask_left = mask & new_mask
            mask_right = mask & ~new_mask
        else:
            mask_left = mask_right = None

        ax = _plot_tree_recursive(node.left, mask=mask_left, **reuse)
        ax = _plot_tree_recursive(node.right, mask=mask_right, **reuse)
        return ax


def _val_to_index(val):
    """Convert a 2-place decimal to an index in (0, 200)"""
    assert val >= -1.0 and val <= 1.0
    val = np.round(val, 3)
    return int(val * 1000) - 1001


def _apply_mask(x, y, mask):
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
    hdt,
    X=None,
    y=None,
    geometry="poincare",
    timelike_dim=0,
    masked=True,
    **kwargs,
):
    """Plot data and all decision boundaries of a hyperbolic decision tree."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit circle
    _x = np.linspace(-1, 1, 1000)
    _y = np.sqrt(1 - _x ** 2)
    ax.plot(_x, _y, c="black")
    ax.plot(_x, -_y, c="black")

    # Plot data
    if X is not None and y is not None:
        X = convert(
            X, initial="hyperboloid", final=geometry, timelike_dim=timelike_dim
        )
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired")

    # Get colors
    colors = list(plt.cm.get_cmap("tab10", hdt.max_depth).colors)

    # Initialize mask
    if masked:
        mask = np.full(shape=(2001, 2001), fill_value=True)
        # 2001x2001 grid makes rounding work in the _apply_mask lookups
    else:
        mask = None

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
        **kwargs,
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=c, label=f"Depth {i}")
            for i, c in enumerate(colors)
        ]
    )

    # Set axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    return ax
