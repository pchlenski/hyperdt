"""Toy data for hyperboloid classification"""

import numpy as np
from geomstats.geometry.hyperbolic import Hyperbolic


def wrapped_normal_mixture(
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

    # Make manifold
    hyp = Hyperbolic(dim=n_dim, default_coords_type=default_coords_type)
    origin = np.array([1.0] + [0.0] * n_dim)

    # Generate random means; parallel transport from origin
    means = np.concatenate(
        [
            np.zeros(shape=(num_classes, 1)),
            np.random.normal(size=(num_classes, n_dim)),
        ],
        axis=1,
    )
    means = hyp.metric.exp(tangent_vec=means, base_point=origin)

    # Generate random covariance matrices
    covs = np.zeros((num_classes, n_dim, n_dim))
    for i in range(num_classes):
        covs[i] = np.random.normal(size=(n_dim, n_dim))
        covs[i] = covs[i] @ covs[i].T
    covs = noise_std * covs

    # Generate random class probabilities
    probs = np.random.uniform(size=num_classes)
    probs = probs / np.sum(probs)

    # First, determine class assignments
    classes = np.random.choice(num_classes, size=num_points, p=probs)

    # Sample the appropriate covariance matrix and make tangent vectors
    vecs = [np.random.multivariate_normal(np.zeros(n_dim), covs[c]) for c in classes]
    tangent_vecs = np.concatenate([np.zeros(shape=(num_points, 1)), vecs], axis=1)

    # Transport each tangent vector to its corresponding mean on the hyperboloid
    tangent_vecs_transported = hyp.metric.parallel_transport(
        tangent_vec=tangent_vecs, base_point=origin, end_point=means[classes]
    )

    # Exponential map to hyperboloid at the class mean
    tangent_vecs_transported = tangent_vecs_transported[~np.isclose(hyp.metric.norm(tangent_vecs_transported), 0)]
    points = hyp.metric.exp(tangent_vec=tangent_vecs_transported, base_point=means[classes])

    return points, classes
