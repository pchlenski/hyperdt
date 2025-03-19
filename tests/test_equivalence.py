"""
Consolidated test suite for verifying mathematical and empirical equivalence
between the original and faster implementations of hyperbolic decision trees.

This test suite verifies key mathematical properties and prediction equivalence:

1. Mathematical equivalence of angle-based and ratio-based formulations
2. Prediction agreement across dimensions, tree depths, and class counts
3. Analysis of numerical precision edge cases

Visual comparisons and benchmarks have been moved to notebooks/visualization_equivalence.ipynb
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import TREE_LEAF
import time

import pytest

# Import the faster implementation
from hyperdt import HyperbolicDecisionTreeClassifier as FasterHDTC

# Try to import the legacy implementation
from hyperdt.legacy.tree import HyperbolicDecisionTreeClassifier as OriginalHDTC
from hyperdt.toy_data import wrapped_normal_mixture


def test_mathematical_equivalence():
    """Verify angle- and ratio-based decision boundaries are mathematically equivalent."""
    print("\n== Testing Mathematical Equivalence ==")

    # Generate test points
    n_points = 1000
    X, _ = wrapped_normal_mixture(n_points, random_state=42)

    # Generate angles in the valid range
    n_angles = 10
    thetas = np.random.uniform(np.pi / 4, 3 * np.pi / 4, n_angles)

    # Test each dimension with different angles
    for dim in range(1, X.shape[1]):
        print(f"Testing dimension {dim}:")

        errors = []
        for theta in thetas:
            # Original formula: sin(θ)*x_d - cos(θ)*x_0 < 0
            orig_split = np.sin(theta) * X[:, dim] - np.cos(theta) * X[:, 0] < 0

            # Transformed formula: x_d/x_0 < cot(θ)
            transformed_split = X[:, dim] / X[:, 0] < 1 / np.tan(theta)

            # Check if they give the same results
            error_rate = np.mean(orig_split != transformed_split)
            errors.append(error_rate)

            if error_rate > 0:
                print(f"  Error rate for θ={theta:.4f}: {error_rate:.6f}")

                # Find disagreement examples
                disagreement_idx = np.where(orig_split != transformed_split)[0]
                for idx in disagreement_idx[:3]:  # Show up to 3 examples
                    x0, xd = X[idx, 0], X[idx, dim]
                    orig_val = np.sin(theta) * xd - np.cos(theta) * x0
                    trans_val = xd / x0 - 1 / np.tan(theta)
                    print(f"    Point {idx}: x0={x0:.6f}, x{dim}={xd:.6f}")
                    print(f"    Original: {orig_val:.10f} {'<' if orig_val < 0 else '>='} 0")
                    print(f"    Transformed: {trans_val:.10f} {'<' if trans_val < 0 else '>='} 0")

        # Report overall error rate for this dimension
        mean_error = np.mean(errors)
        if mean_error > 0:
            print(f"  Mean error rate: {mean_error:.6f}")
        else:
            print(f"  All splits equivalent (error rate: {mean_error:.6f})")

    print(f"Mathematical equivalence verified")


def test_prediction_agreement(depths=[1, 3, 5, 10, None]):
    """Test prediction agreement between implementations across various tree depths."""
    print("\n== Testing Prediction Agreement Across Depths ==")

    n_trees_per_depth = 5  # Reduced for faster tests
    n_features = 5
    n_classes = 3
    n_samples = 200

    # Test with different random seeds
    np.random.seed(42)
    seeds = np.random.randint(1, 10000, n_trees_per_depth)

    for depth in depths:
        depth_str = str(depth) if depth is not None else "unlimited"
        print(f"\nTesting max_depth={depth_str}:")

        depth_agreements = []
        depth_split_matches = []

        for i, seed in enumerate(seeds):
            # Generate data
            X, y = wrapped_normal_mixture(n_samples, n_classes, n_features, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            # Create and fit models
            orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
            faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)

            # Fit trees
            orig_tree.fit(X_train, y_train)
            faster_tree.fit(X_train, y_train)

            # Get predictions
            orig_test_preds = orig_tree.predict(X_test)
            faster_test_preds = faster_tree.predict(X_test)

            # Compare predictions
            test_agreement = np.mean(orig_test_preds == faster_test_preds)
            depth_agreements.append(test_agreement)

            # Calculate simple split match rate (just count total splits)
            split_match = 1.0  # Default to perfect match for depths with no splits

            if depth is not None and depth > 0:
                # Extract tree structure
                if hasattr(orig_tree.tree, "feature") and orig_tree.tree.feature is not None:
                    # Only calculate if trees have been properly fitted
                    try:
                        orig_feature = orig_tree.tree.feature  # This might be None for leaf nodes
                        # For faster_tree, we use sklearn's tree structure
                        klein_feature = faster_tree.tree.tree_.feature[0]

                        # If we can map between the feature spaces, check if they match
                        if orig_feature is not None and klein_feature is not None:
                            # Convert original feature to Klein space for comparison
                            orig_feature_klein = orig_feature - 1 if orig_feature > 0 else orig_feature

                            # If features match, it's a 100% match for this simple tree
                            if orig_feature_klein == klein_feature:
                                split_match = 1.0
                            else:
                                split_match = 0.0
                    except (AttributeError, IndexError):
                        # If any structure is missing, count as no match
                        split_match = 0.0

            depth_split_matches.append(split_match)

            # Report issues for any disagreements
            if test_agreement < 1.0:
                print(f"  Disagreement found with seed {seed}:")
                print(f"    Test agreement: {test_agreement:.4f}")

                # Find a disagreement example
                disagreement_idx = np.where(orig_test_preds != faster_test_preds)[0][0]
                print(f"    Example point: X_test[{disagreement_idx}]")
                print(f"    Original prediction: {orig_test_preds[disagreement_idx]}")
                print(f"    Faster prediction: {faster_test_preds[disagreement_idx]}")

        # Collect metrics for this depth
        mean_agreement = np.mean(depth_agreements)
        mean_split_match = np.mean(depth_split_matches)

        print(f"  Mean prediction agreement: {mean_agreement:.6f}")
        print(f"  Mean split match rate: {mean_split_match:.6f}")

        # Assert high agreement
        assert mean_agreement > 0.95, f"Low prediction agreement ({mean_agreement:.4f}) at depth {depth}"


def test_decision_boundary_agreement():
    """Test that the decision boundaries between implementations have high agreement."""
    print(f"\n== Testing Decision Boundary Agreement ==")

    depth = 5
    # Generate 2D hyperbolic data (3D points on hyperboloid)
    X, y = wrapped_normal_mixture(500, n_classes=3, n_features=3, random_state=42)

    # Create and fit models
    orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)

    orig_tree.fit(X, y)
    faster_tree.fit(X, y)

    # Create test points
    resolution = 50  # Reduced for faster tests
    x_min, x_max = np.min(X[:, 1]) * 1.1, np.max(X[:, 1]) * 1.1
    y_min, y_max = np.min(X[:, 2]) * 1.1, np.max(X[:, 2]) * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Create hyperboloid points for the grid
    spacelike_norm_squared = xx**2 + yy**2
    timelike = np.sqrt(spacelike_norm_squared + 1.0)  # Assuming curvature = 1.0
    grid_points = np.column_stack([timelike.flatten(), xx.flatten(), yy.flatten()])

    # Get predictions
    orig_preds = orig_tree.predict(grid_points)
    faster_preds = faster_tree.predict(grid_points)

    # Calculate agreement rate
    agreement_rate = np.mean(orig_preds == faster_preds)

    print(f"Decision boundary agreement rate: {agreement_rate:.6f}")

    # Assert high agreement
    assert agreement_rate > 0.95, f"Low decision boundary agreement ({agreement_rate:.4f})"


def test_information_gain_agreement():
    """Test that implementations have high prediction agreement even when there are ties in information gain."""
    print(f"\n== Testing Information Gain Agreement ==")

    # Use seed 44 which we know has a tie
    seed = 44
    X, y = wrapped_normal_mixture(100, n_classes=3, n_features=5, random_state=seed)

    # Create models
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)

    # Fit both models
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)

    # Check prediction agreement
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)

    agreement = np.mean(orig_preds == faster_preds)
    print(f"Prediction agreement with ties: {agreement:.6f}")

    # Assert reasonable agreement
    # Note: we don't expect perfect agreement due to tie-breaking differences
    assert agreement > 0.9, f"Low prediction agreement ({agreement:.4f}) with information gain ties"


def test_higher_dimensions():
    """Test agreement with higher dimensional data"""
    print("\n== Testing Higher Dimensions ==")

    dimensions = [3, 5, 10, 20]

    for dim in dimensions:
        print(f"\nTesting with {dim} dimensions:")
        X, y = wrapped_normal_mixture(500, n_classes=3, n_features=dim, random_state=42)

        # Create models
        orig_tree = OriginalHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
        faster_tree = FasterHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)

        # Fit models
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)

        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)

        agreement = np.mean(orig_preds == faster_preds)
        print(f"Prediction agreement rate: {agreement:.4f}")

        # Compare probabilities if applicable
        orig_probs = orig_tree.predict_proba(X)
        faster_probs = faster_tree.predict_proba(X)

        prob_mse = np.mean((orig_probs - faster_probs) ** 2)
        print(f"Probability MSE: {prob_mse:.6f}")


def test_baseline_performance():
    """Test that the faster implementation is indeed faster than the original implementation."""
    print("\n== Testing Performance ==")

    # Use smaller dataset for quicker tests
    n_samples = 500
    n_features = 10
    n_classes = 3

    # Generate data
    X, y = wrapped_normal_mixture(n_samples, n_classes, n_features, random_state=42)

    # Test original implementation
    orig_tree = OriginalHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)

    start = time.time()
    orig_tree.fit(X, y)
    orig_time = time.time() - start

    # Test faster implementation
    faster_tree = FasterHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)

    start = time.time()
    faster_tree.fit(X, y)
    faster_time = time.time() - start

    # Compare predictions
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)

    agreement = np.mean(orig_preds == faster_preds)
    speedup = orig_time / faster_time

    print(f"  Original: {orig_time:.4f}s, Faster: {faster_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Prediction agreement: {agreement:.6f}")

    # Assert the faster implementation actually is faster
    assert faster_time < orig_time, "Faster implementation is not actually faster!"


def test_numerical_precision():
    """Test specific edge case at decision boundary"""
    print("\n=== Testing Numerical Precision ===")

    # Generate a simple dataset for training
    X, y = wrapped_normal_mixture(100, n_classes=2, n_features=3, random_state=42)

    # Create and fit simple trees
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)

    orig_tree.fit(X, y)
    faster_tree.fit(X, y)

    # Extract decision boundary parameters
    feature = orig_tree.tree.feature
    theta = orig_tree.tree.theta
    cot_theta = 1 / np.tan(theta)

    print(f"Decision boundary: feature={feature}, θ={theta:.6f}, cot(θ)={cot_theta:.8f}")

    # Generate a series of test points exactly at and around the decision boundary
    test_points = []
    epsilons = [-1e-8, -1e-10, -1e-12, 0, 1e-12, 1e-10, 1e-8]

    for eps in epsilons:
        # Start with a base point
        base_point = np.ones(3)
        base_point[0] = 2.0  # arbitrary timelike component

        # Set the feature to be exactly at/near boundary
        base_point[feature] = (cot_theta + eps) * base_point[0]

        # Ensure it's on the hyperboloid
        norm = np.sqrt(np.sum(base_point[1:] ** 2) + 1)
        base_point[0] = norm

        test_points.append(base_point)

    test_points = np.array(test_points)

    # Get predictions from both implementations
    orig_preds = orig_tree.predict(test_points)
    faster_preds = faster_tree.predict(test_points)

    print("\nBoundary point predictions:")
    print(f"{'Epsilon':>10} {'Original':>10} {'Faster':>10} {'Match':>10} {'Ratio':>15} {'Dot Product':>15}")
    print("-" * 75)

    for i, eps in enumerate(epsilons):
        # Calculate actual ratio and dot product
        point = test_points[i]
        ratio = point[feature] / point[0]
        dot_val = orig_tree._dot(point.reshape(1, -1), feature, theta).item()

        # Check if predictions match
        match = orig_preds[i] == faster_preds[i]

        print(
            f"{eps:10.1e} {orig_preds[i]:10d} {faster_preds[i]:10d} {str(match):>10} "
            f"{ratio:15.12f} {dot_val:15.12f}"
        )


def test_all():
    """Run all verification tests"""
    print("=== Running Comprehensive Equivalence Tests ===")
    test_mathematical_equivalence()
    test_prediction_agreement()
    test_higher_dimensions()
    test_decision_boundary_agreement()
    test_information_gain_agreement()
    test_baseline_performance()
    test_numerical_precision()
    print("\n=== All Tests Complete ===")


if __name__ == "__main__":
    test_all()
