"""Toy data for hyperboloid classification"""

import numpy as np


def project_to_hyperboloid(points: np.ndarray) -> np.ndarray:
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
    points = project_to_hyperboloid(points)

    return points
