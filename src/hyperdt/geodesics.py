import numpy as np


def generate_hyperboloid_points(n_points=2, n_dim=3):
    """Generate some random points on a hyperboloid such that B(x, x) = -1"""
    points = np.random.rand(n_points, n_dim)
    x0 = np.sqrt(1 + np.sum(points**2, axis=1))
    return np.column_stack((-x0, points))  # np.stack is finnicky


def _B(u, v):
    """Minkowski bilinear form"""
    return -u[0] * v[0] + np.dot(u[1:], v[1:])


def _compute_x_u(p1, p2):
    """
    Given points p1 and p2 on the hyperboloid, compute x and u such that:
    B(x, x) = -1
    B(u, u) = 1
    B(x, u) = B(u, x) = 0

    These vectors are used to compute geodesics on the hyperboloid.

    Based on HGCN paper:
    https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf
    """
    # x is simply p1
    x = p1

    # Compute the tangent vector from p1 to p2
    v = p2 - p1

    # Project the tangent vector onto the tangent space at x
    normal = np.concatenate([[-2 * x[0]], 2 * x[1:]])
    u = v - _B(v, normal) * normal / _B(normal, normal)

    # Scale u to be unit-speed according to B
    print("BILINEAR", _B(u, u))
    u /= np.sqrt(abs(_B(u, u)))

    return x, u


def _test_hyperboloid_points(points):
    """Test that p1 and p2 are on the hyperboloid"""
    points = np.array(points)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    assert np.allclose(
        np.sum(points[:, 1:] ** 2, axis=1) - points[:, 0] ** 2, -1
    )


def _test_x_u(x, u):
    """Test that x, u are valid vectors for computing geodesics"""
    assert np.isclose(_B(x, x), -1), "B(x, x) not equal to -1"
    assert np.isclose(_B(u, u), 1), "B(u, u) not equal to 1"
    assert np.isclose(_B(x, u), 0), "B(x, u) not equal to 0"
    assert np.isclose(_B(u, x), 0), "B(u, x) not equal to 0"
    return True


def _test_x_u_p1_p2(x, u, p1, p2):
    """Verify that p1 and p2 are linear combinations of x and u"""
    # Stack x and u as columns to form the matrix A
    A = np.column_stack((x, u))

    # Solve the linear system for p1 and p2
    coeffs_p1, _, _, _ = np.linalg.lstsq(A, p1, rcond=None)
    coeffs_p2, _, _, _ = np.linalg.lstsq(A, p2, rcond=None)

    # Reconstruct p1 and p2 using the found coefficients
    reconstructed_p1 = coeffs_p1[0] * x + coeffs_p1[1] * u
    reconstructed_p2 = coeffs_p2[0] * x + coeffs_p2[1] * u

    # Check if the reconstructed points are close to the original ones
    assert np.allclose(
        reconstructed_p1, p1
    ), "p1 is not a linear combination of x and u"
    assert np.allclose(
        reconstructed_p2, p2
    ), "p2 is not a linear combination of x and u"

    return True


def get_geodesic(p1, p2, n_points=1000):
    """Get geodesic between two points on the hyperboloid"""
    _test_hyperboloid_points(np.stack([p1, p1], axis=0))
    x, u = _compute_x_u(p1, p2)
    _test_x_u(x, u)
    # _test_x_u_p1_p2(x, u, p1, p2)
    w = np.linspace(-1, 1, n_points).reshape(-1, 1)  # Parameterized by w
    geodesic = u * np.sinh(w) + x * np.cosh(w)
    _test_hyperboloid_points(geodesic)

    return geodesic


# def test_random_hyperboloid_points():
#     """Test that _compute_x_u works for random points on the hyperboloid"""
#     success = True
#     for i in range(10):
#         n_dim = np.random.randint(3, 10)
#         points = _generate_hyperboloid_points(2, n_dim)
#         p1, p2 = points
#         try:
#             assert _test_hyperboloid_points(
#                 p1, p2
#             ), f"Test failed for dimension {n_dim}"
#             print(f"Test passed for dimension {n_dim}")
#         except AssertionError as e:
#             print(str(e))
#             success = False
#     return success


# # Run the tests
# test_random_hyperboloid_points()
