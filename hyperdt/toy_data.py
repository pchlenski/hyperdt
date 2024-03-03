"""Toy data for hyperboloid classification"""

from typing import Tuple, Literal, Optional

import numpy as np
from geomstats.geometry.hyperbolic import Hyperbolic

# Need for bad_points:
import geomstats.backend as gs
import geomstats.algebra_utils as utils


def bad_points(points: np.ndarray, base_points: np.ndarray, manifold: Hyperbolic) -> np.ndarray:
    """Avoid the 'Minkowski norm of 0' error by using this"""
    sq_norm_tangent_vec = manifold.embedding_space.metric.squared_norm(points)
    sq_norm_tangent_vec = gs.clip(sq_norm_tangent_vec, 0, np.inf)

    coef_1 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.cosh_close_0, order=5)
    coef_2 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.sinch_close_0, order=5)

    exp = gs.einsum("...,...j->...j", coef_1, base_points) + gs.einsum("...,...j->...j", coef_2, points)
    return manifold.metric.squared_norm(exp) == 0


def wrapped_normal_mixture(
    num_points: int = 1000,
    num_classes: int = 2,
    num_dims: int = 2,
    noise_std: float = 1.0,
    default_coords_type: Literal["extrinsic", "ball", "half-space"] = "extrinsic",
    seed: Optional[int] = None,
    adjust_for_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points from a mixture of Gaussians on the hyperboloid.

    Args:
    -----
    num_points: int
        Number of points to generate (default: 1000)
    num_classes: int
        Number of classes in the mixture (default: 2)
    num_dims: int
        Dimension of the hyperboloid (default: 2)
    noise_std: float
        Scalar multiplier for the covariance matrices of each class (default: 1.0)
    default_coords_type: str
        Coordinates type for the hyperboloid (default: "extrinsic")
    seed: int
        Random seed (default: None)
    adjust_for_dim: bool
        Adjust the covariance matrices for the dimension of the hyperboloid (default: True)

    Returns:
    --------
    points: np.ndarray
        Generated points on the hyperboloid
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Make manifold
    hyp = Hyperbolic(dim=num_dims, default_coords_type=default_coords_type)
    origin = np.array([1.0] + [0.0] * num_dims)

    # Generate random means; parallel transport from origin
    means = np.concatenate(
        [
            np.zeros(shape=(num_classes, 1)),
            np.random.normal(size=(num_classes, num_dims)),
        ],
        axis=1,
    )
    means = hyp.metric.exp(tangent_vec=means, base_point=origin)

    # Generate random covariance matrices
    covs = np.zeros((num_classes, num_dims, num_dims))
    for i in range(num_classes):
        covs[i] = np.random.normal(size=(num_dims, num_dims))
        covs[i] = covs[i] @ covs[i].T
    covs = noise_std * covs
    if adjust_for_dim:
        covs = covs / num_dims

    # Generate random class probabilities
    probs = np.random.uniform(size=num_classes)
    probs = probs / np.sum(probs)

    # First, determine class assignments
    classes = np.random.choice(num_classes, size=num_points, p=probs)

    # Sample the appropriate covariance matrix and make tangent vectors
    vecs = [np.random.multivariate_normal(np.zeros(num_dims), covs[c]) for c in classes]
    tangent_vecs = np.concatenate([np.zeros(shape=(num_points, 1)), vecs], axis=1)

    # Transport each tangent vector to its corresponding mean on the hyperboloid
    tangent_vecs_transported = hyp.metric.parallel_transport(
        tangent_vec=tangent_vecs, base_point=origin, end_point=means[classes]
    )

    # Exponential map to hyperboloid at the class mean
    keep = ~bad_points(tangent_vecs_transported, means[classes], hyp)
    tangent_vecs_transported = tangent_vecs_transported[keep]
    classes = classes[keep]
    points = hyp.metric.exp(tangent_vec=tangent_vecs_transported, base_point=means[classes])

    return points, classes
