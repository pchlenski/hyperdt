"""Toy data for hyperboloid classification"""

import numpy as np
from geomstats.geometry.hyperbolic import Hyperbolic


def _project_to_hyperboloid(points: np.ndarray) -> np.ndarray:
    """Take points in ambient space and project them onto the hyperboloid"""
    points[:, 0] = np.sqrt(1.0 + np.sum(points[:, 1:] ** 2, axis=1))
    return points


def generate_points_on_branch(
    root: np.ndarray,
    direction: np.ndarray,
    num_points: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Generate points along a geodesic branch in the hyperboloid model"""

    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)

    # Randomly sample distances from the root
    distances = np.random.uniform(0.0, 1.0, size=num_points)

    # Generate points along the geodesic - use outer product
    points = root + np.outer(distances, direction)

    # Add noise
    if noise_std > 0.0:
        points += np.random.normal(scale=noise_std, size=points.shape)

    # Project onto the hyperboloid
    points = _project_to_hyperboloid(points)

    return points


def generate_gaussian_mixture_hyperboloid(
    num_points: int,
    num_classes: int,
    noise_std: float = 1.0,
    n_dim: int = 2,
    default_coords_type: str = "extrinsic",
    seed: int = None,
) -> np.ndarray:
    """Generate points from a mixture of Gaussians on the hyperboloid"""

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Generate random means
    means = np.random.normal(size=(num_classes, n_dim + 1))

    # Generate random covariance matrices
    covs = np.zeros((num_classes, n_dim + 1, n_dim + 1))
    for i in range(num_classes):
        covs[i] = np.random.normal(size=(n_dim + 1, n_dim + 1))
        covs[i] = covs[i] @ covs[i].T
    covs = noise_std * covs

    # Generate random class probabilities
    probs = np.random.uniform(size=num_classes)
    probs = probs / np.sum(probs)

    # Generate points
    points = np.zeros((num_points, n_dim + 1))
    labels = np.zeros(num_points, dtype=int)
    for i in range(num_points):
        # Sample class
        c = np.random.choice(num_classes, p=probs)
        labels[i] = c

        # Sample point
        points[i] = np.random.multivariate_normal(means[c], covs[c])

    # Make manifold
    hyp = Hyperbolic(dim=n_dim, default_coords_type=default_coords_type)

    # Make tangent vectors; take exp map
    origin = np.zeros(n_dim + 1)
    origin[0] = 1.0
    tangent_vecs = hyp.to_tangent(points, base_point=origin)
    keep1 = not np.isclose(hyp.metric.squared_norm(tangent_vecs), 0.0)
    tangent_vecs = tangent_vecs[keep1]
    labels = labels[keep1]
    points = hyp.metric.exp(tangent_vecs, base_point=origin)

    # Throw out off-manifold points
    keep2 = np.isclose(hyp.metric.squared_norm(points), -1)
    points = points[keep2]
    labels = labels[keep2]

    # Finally, ensure timelike > norm(spacelike):
    keep3 = points[:, 0] > np.linalg.norm(points[:, 1:], axis=1)
    points = points[keep3]
    labels = labels[keep3]

    return points, labels
