"""Tests for toy data utilities."""

import numpy as np
import pytest

from hyperdt.toy_data import wrapped_normal_mixture


def is_on_hyperboloid(points, dim):
    """Check if points lie on the hyperboloid."""
    # For points on the hyperboloid, we have:
    # -x_0^2 + x_1^2 + ... + x_n^2 = -1
    # Rearranged: x_0^2 - (x_1^2 + ... + x_n^2) = 1
    
    # First coordinate squared
    x0_squared = points[:, 0] ** 2
    
    # Sum of squares of remaining coordinates
    rest_squared_sum = np.sum(points[:, 1:] ** 2, axis=1)
    
    # Check if the hyperboloid constraint is satisfied (with numerical tolerance)
    return np.isclose(x0_squared - rest_squared_sum, 1.0, atol=1e-10)


class TestToyData:
    """Test suite for toy data generation utilities."""

    def test_wrapped_normal_mixture_shapes(self):
        """Test that the generated data has the correct shapes."""
        # Test with default parameters
        points, classes = wrapped_normal_mixture(num_points=100, seed=42)
        
        # Check dimensions
        assert points.shape[1] == 3  # Default is 2D hyperboloid (embedded in 3D)
        assert len(classes) == points.shape[0]  # One class label per point
        
        # Test with different dimensions
        points_4d, classes_4d = wrapped_normal_mixture(num_points=50, num_dims=4, seed=42)
        assert points_4d.shape[1] == 5  # 4D hyperboloid embedded in 5D

    def test_wrapped_normal_mixture_hyperboloid(self):
        """Test that the generated points lie on the hyperboloid."""
        # Generate points with 2D manifold (3D embedding)
        points, _ = wrapped_normal_mixture(num_points=100, num_dims=2, seed=42)
        
        # Check if points are on the hyperboloid
        on_hyperboloid = is_on_hyperboloid(points, dim=2)
        assert np.all(on_hyperboloid), "Some points are not on the hyperboloid"
        
        # Test with higher dimensions
        points_4d, _ = wrapped_normal_mixture(num_points=50, num_dims=4, seed=42)
        on_hyperboloid_4d = is_on_hyperboloid(points_4d, dim=4)
        assert np.all(on_hyperboloid_4d), "Some 4D points are not on the hyperboloid"

    def test_wrapped_normal_mixture_classes(self):
        """Test that class assignments are valid."""
        # Test with 2 classes (default)
        _, classes = wrapped_normal_mixture(num_points=100, num_classes=2, seed=42)
        unique_classes = np.unique(classes)
        assert len(unique_classes) <= 2  # May have fewer than 2 due to bad_points filtering
        assert np.all(unique_classes >= 0) and np.all(unique_classes < 2)
        
        # Test with 3 classes
        _, classes_3 = wrapped_normal_mixture(num_points=100, num_classes=3, seed=42)
        unique_classes_3 = np.unique(classes_3)
        assert len(unique_classes_3) <= 3
        assert np.all(unique_classes_3 >= 0) and np.all(unique_classes_3 < 3)

    def test_wrapped_normal_mixture_reproducibility(self):
        """Test that using the same seed gives the same results."""
        points1, classes1 = wrapped_normal_mixture(num_points=100, seed=42)
        points2, classes2 = wrapped_normal_mixture(num_points=100, seed=42)
        
        np.testing.assert_array_equal(points1, points2)
        np.testing.assert_array_equal(classes1, classes2)