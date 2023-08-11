import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton

from .conversions import convert


def _find_intersection(D, theta, t, t_dim=1, n_dim=3):
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
    """Get the intersection of a hyperplane and a geodesic."""
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


def plot_boundary(
    hdt_node,
    X=None,
    y=None,
    t_dim=None,
    geometry="poincare",
    ax=None,
    timelike_dim=0,
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

    # Init figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Verify geodesics lie inside unit circle:
    if np.all(np.linalg.norm(geodesic_points, axis=1) <= 1):
        ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], c="red")
    else:
        print(
            f"Geodesic points lie outside unit circle:\t{boundary_dim} {boundary_theta/np.pi:.3f}*pi {t_dim}"
        )

    # Set axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    return ax


def _plot_tree_recursive(node, **kwargs):
    if node.value is not None:  # Leaf case
        return kwargs["ax"]
    else:
        ax = plot_boundary(node, **kwargs)
        ax = _plot_tree_recursive(node.left, **kwargs)
        ax = _plot_tree_recursive(node.right, **kwargs)
        return ax


def plot_tree(hdt, X=None, y=None, geometry="poincare", timelike_dim=0):
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
        ax.scatter(X[:, 0], X[:, 1], c=y)

    return _plot_tree_recursive(
        hdt.tree, ax=ax, geometry=geometry, timelike_dim=0
    )
