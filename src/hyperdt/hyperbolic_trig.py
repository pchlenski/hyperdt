import numpy as np
from scipy.optimize import root_scalar


def _dist(x1, x2):
    """Closed form for distance between two points on intersection of
    hyperboloid and plane such that all other coordinates are 0"""
    a1 = np.sqrt(-1 / np.cos(2 * x1))  # sqrt(-sec(2x_1))
    a2 = np.sqrt(-1 / np.cos(2 * x2))  # sqrt(-sec(2x_2))
    dist = np.arccosh(-a1 * a2 * np.cos(x1 + x2))

    # Deal with really close values - numerical weirdness
    if np.abs(dist) < 1e-6 or np.abs(x1 - x2) < 1e-6:
        dist = 0
    return dist


def _dist_aberration(m, x1, x2):
    """This is 0 when d(theta1, m) = d(theta2, m) = d(theta1, theta2)/2"""
    # return np.cos(x1) * np.cos(x2 + m) ** -np.cos(x2) * np.cos(x1 + m) ** 2
    return _dist(x1, m) - _dist(m, x2)


def get_midpoint(theta1, theta2):
    """Find hyperbolic midpoint of two angles"""
    theta_min = np.min([theta1, theta2])
    theta_max = np.max([theta1, theta2])
    root = root_scalar(
        _dist_aberration, args=(theta1, theta2), bracket=[theta_min, theta_max]
    ).root
    assert np.abs(_dist_aberration(root, theta1, theta2)) < 1e-6
    assert root >= theta_min and root <= theta_max
    return root


def get_candidates_hyperbolic(X, dim, timelike_dim):
    """Get candidate split points for hyperbolic decision tree"""
    # X[:, dim][X[:, dim] == 0.0] = 1e-6
    thetas = np.arctan2(X[:, timelike_dim], X[:, dim])
    thetas[thetas < np.pi / 4] += 2 * np.pi
    thetas = np.unique(thetas)

    # Get all pairs of angles
    candidates = np.array(
        [
            get_midpoint(theta1, theta2)
            for theta1, theta2 in zip(thetas[:-1], thetas[1:])
        ]
    )
    assert (candidates >= np.pi / 4).all()
    assert (candidates <= 3 * np.pi / 4).all()
    return candidates
