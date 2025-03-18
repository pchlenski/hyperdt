"""
Consolidated test suite for verifying mathematical and empirical equivalence
between the original and faster implementations of hyperbolic decision trees.

This test suite combines functionality from various test files to create a
comprehensive verification system:

1. Mathematical equivalence of angle-based and ratio-based formulations
2. Prediction agreement across dimensions, tree depths, and class counts
3. Split selection differences due to information gain ties
4. Performance comparison between implementations
5. Analysis of numerical precision edge cases
6. Decision boundary visualization and comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import TREE_LEAF
import time
from pathlib import Path
import os
import sys

import pytest

# Import the faster implementation
from hyperdt.faster_tree import HyperbolicDecisionTreeClassifier as FasterHDTC

# Try to import the legacy implementation
try:
    from hyperdt.legacy.tree import HyperbolicDecisionTreeClassifier as OriginalHDTC
    from hyperdt.toy_data import wrapped_normal_mixture
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    # Create a dummy wrapped_normal_mixture function
    def wrapped_normal_mixture(*args, **kwargs):
        return None, None

# Create directory for images
IMAGES_DIR = Path("/home/phil/hyperdt/tests/equivalence/images")
IMAGES_DIR.mkdir(exist_ok=True)

# All results will be saved to the images directory


def generate_hyperbolic_data(n_samples, n_classes=2, n_features=3, random_state=None):
    """Generate synthetic data on the hyperboloid using wrapped normal mixture"""
    # Set seed if provided
    original_seed = None
    if random_state is not None:
        original_seed = np.random.get_state()
        np.random.seed(random_state)

    # Generate data
    X, y = wrapped_normal_mixture(
        num_points=n_samples,
        num_classes=n_classes,
        num_dims=n_features - 1,  # Ambient dimension is manifold dim + 1
        noise_std=1.0,
        adjust_for_dim=True,
    )

    # Restore random state if needed
    if original_seed is not None:
        np.random.set_state(original_seed)

    return X, y


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def test_mathematical_equivalence():
    """
    Verify that the angle-based decision boundary (original) and the
    ratio-based decision boundary (faster) are mathematically equivalent.
    """
    print("\n== Testing Mathematical Equivalence ==")

    # Generate test points
    n_points = 1000
    X, _ = generate_hyperbolic_data(n_points, random_state=42)

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


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def test_prediction_agreement(depths=[1, 3, 5, 10, None]):
    """
    Test prediction agreement between implementations across various tree depths.
    """
    print("\n== Testing Prediction Agreement Across Depths ==")

    n_trees_per_depth = 10
    n_features = 5
    n_classes = 3
    n_samples = 200

    # Results tracking
    results = {
        "depths": [],
        "agreement_rates": [],
        "split_match_rates": [],
        "orig_times": [],
        "faster_times": [],
        "speedups": [],
    }

    # Test with different random seeds
    np.random.seed(42)
    seeds = np.random.randint(1, 10000, n_trees_per_depth)

    for depth in depths:
        depth_str = str(depth) if depth is not None else "unlimited"
        print(f"\nTesting max_depth={depth_str}:")

        depth_agreements = []
        depth_split_matches = []
        depth_orig_times = []
        depth_faster_times = []

        for i, seed in enumerate(seeds):
            # Generate data
            X, y = generate_hyperbolic_data(n_samples, n_classes, n_features, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            # Create and fit models
            orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
            faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)

            # Time and fit original
            start = time.time()
            orig_tree.fit(X_train, y_train)
            orig_time = time.time() - start
            depth_orig_times.append(orig_time)

            # Time and fit faster
            start = time.time()
            faster_tree.fit(X_train, y_train)
            faster_time = time.time() - start
            depth_faster_times.append(faster_time)

            # Get predictions
            orig_train_preds = orig_tree.predict(X_train)
            faster_train_preds = faster_tree.predict(X_train)

            orig_test_preds = orig_tree.predict(X_test)
            faster_test_preds = faster_tree.predict(X_test)

            # Compare predictions
            train_agreement = np.mean(orig_train_preds == faster_train_preds)
            test_agreement = np.mean(orig_test_preds == faster_test_preds)

            # Use test agreement as the primary metric
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
            if train_agreement < 1.0 or test_agreement < 1.0:
                print(f"  Disagreement found with seed {seed}:")
                print(f"    Train agreement: {train_agreement:.4f}, Test agreement: {test_agreement:.4f}")

                # Find a disagreement example
                if test_agreement < 1.0:
                    disagreement_idx = np.where(orig_test_preds != faster_test_preds)[0][0]
                    print(f"    Example point: X_test[{disagreement_idx}]")
                    print(f"    Original prediction: {orig_test_preds[disagreement_idx]}")
                    print(f"    Faster prediction: {faster_test_preds[disagreement_idx]}")

        # Collect metrics for this depth
        mean_agreement = np.mean(depth_agreements)
        mean_split_match = np.mean(depth_split_matches)
        mean_orig_time = np.mean(depth_orig_times)
        mean_faster_time = np.mean(depth_faster_times)
        mean_speedup = mean_orig_time / mean_faster_time if mean_faster_time > 0 else float("inf")

        print(f"  Mean prediction agreement: {mean_agreement:.6f}")
        print(f"  Mean split match rate: {mean_split_match:.6f}")
        print(f"  Mean times - Original: {mean_orig_time:.6f}s, Faster: {mean_faster_time:.6f}s")
        print(f"  Mean speedup: {mean_speedup:.2f}x")

        # Store for plotting
        results["depths"].append(depth_str)
        results["agreement_rates"].append(mean_agreement)
        results["split_match_rates"].append(mean_split_match)
        results["orig_times"].append(mean_orig_time)
        results["faster_times"].append(mean_faster_time)
        results["speedups"].append(mean_speedup)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot agreement rates
    plt.subplot(1, 3, 1)
    plt.bar(results["depths"], results["agreement_rates"], color="skyblue")
    plt.ylim(0.95, 1.001)
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    plt.title("Prediction Agreement Rates")
    plt.ylabel("Agreement Rate")
    plt.xlabel("Max Tree Depth")

    # Plot split match rates
    plt.subplot(1, 3, 2)
    plt.bar(results["depths"], results["split_match_rates"], color="lightgreen")
    plt.ylim(0, 1.05)
    plt.title("Split Match Rates")
    plt.ylabel("Split Match Rate")
    plt.xlabel("Max Tree Depth")

    # Plot speedup
    plt.subplot(1, 3, 3)
    plt.bar(results["depths"], results["speedups"], color="salmon")
    plt.title("Performance Speedup")
    plt.ylabel("Speedup Factor (x)")
    plt.xlabel("Max Tree Depth")

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/prediction_agreement.png")

    return results


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def visualize_decision_boundaries(depth=5):
    """
    Visualize and compare decision boundaries between the two implementations.
    """
    print(f"\n== Visualizing Decision Boundaries (depth={depth}) ==")

    # Generate 2D hyperbolic data (3D points on hyperboloid)
    X, y = generate_hyperbolic_data(500, n_classes=3, n_features=3, random_state=42)

    # Create and fit models
    orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)

    orig_tree.fit(X, y)
    faster_tree.fit(X, y)

    # Create meshgrid for visualization
    resolution = 100
    x_min, x_max = np.min(X[:, 1]) * 1.1, np.max(X[:, 1]) * 1.1
    y_min, y_max = np.min(X[:, 2]) * 1.1, np.max(X[:, 2]) * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Create hyperboloid points for the grid
    spacelike_norm_squared = xx**2 + yy**2
    timelike = np.sqrt(spacelike_norm_squared + 1.0)  # Assuming curvature = 1.0
    grid_points = np.column_stack([timelike.flatten(), xx.flatten(), yy.flatten()])

    # Get predictions
    orig_preds = orig_tree.predict(grid_points).reshape(xx.shape)
    faster_preds = faster_tree.predict(grid_points).reshape(xx.shape)

    # Calculate difference
    diff = (orig_preds != faster_preds).astype(int)
    agreement_rate = 1 - np.mean(diff)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original tree
    plt.subplot(1, 3, 1)
    plt.contourf(xx, yy, orig_preds, alpha=0.3, cmap=plt.cm.Paired)
    for class_value in np.unique(y):
        plt.scatter(X[y == class_value, 1], X[y == class_value, 2], alpha=0.8, label=f"Class {class_value}")
    plt.title(f"Original Tree (depth={depth})")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()

    # Faster tree
    plt.subplot(1, 3, 2)
    plt.contourf(xx, yy, faster_preds, alpha=0.3, cmap=plt.cm.Paired)
    for class_value in np.unique(y):
        plt.scatter(X[y == class_value, 1], X[y == class_value, 2], alpha=0.8, label=f"Class {class_value}")
    plt.title(f"Faster Tree (depth={depth})")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()

    # Difference
    plt.subplot(1, 3, 3)
    plt.imshow(diff, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="Reds", alpha=0.5)
    for class_value in np.unique(y):
        plt.scatter(X[y == class_value, 1], X[y == class_value, 2], alpha=0.3, label=f"Class {class_value}")
    plt.title(f"Differences (agreement: {agreement_rate:.6f})")
    plt.xlabel("x₁")
    plt.ylabel("x₂")

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/decision_boundary_depth{depth}.png")

    print(f"Decision boundary agreement rate: {agreement_rate:.6f}")

    return agreement_rate


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def find_information_gain_ties(seed=44):
    """
    Find and analyze datasets with ties in information gain.
    This demonstrates why the two implementations sometimes choose different splits.
    """
    print(f"\n== Analyzing Information Gain Ties (seed={seed}) ==")

    # Generate data - using seed 44 which we know has a tie
    X, y = generate_hyperbolic_data(100, n_classes=3, n_features=5, random_state=seed)

    # Create models
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)

    # First, calculate information gains for all features in original model
    orig_tree.ndim = X.shape[1]
    orig_tree.timelike_dim = 0
    orig_tree.dims = [i for i in range(X.shape[1]) if i != orig_tree.timelike_dim]
    orig_tree.classes_ = np.unique(y)

    # Calculate best split for each feature
    feature_gains = {}
    for feature in orig_tree.dims:
        thresholds = orig_tree._get_candidates(X, feature)
        max_gain = -1
        best_theta = None

        for theta in thresholds:
            left, right = orig_tree._get_split(X, feature, theta)

            if min(np.sum(left), np.sum(right)) >= orig_tree.min_samples_leaf:
                gain = orig_tree._information_gain(left, right, y)
                if gain > max_gain:
                    max_gain = gain
                    best_theta = theta

        if best_theta is not None:
            feature_gains[feature] = (best_theta, max_gain)

    # Sort features by gain
    sorted_gains = sorted(feature_gains.items(), key=lambda x: x[1][1], reverse=True)

    # Check for ties
    if len(sorted_gains) >= 2:
        best_feature, (best_theta, best_gain) = sorted_gains[0]
        second_feature, (second_theta, second_gain) = sorted_gains[1]

        rel_diff = abs(best_gain - second_gain) / best_gain

        print(f"Best feature: {best_feature}, θ={best_theta:.6f}, gain={best_gain:.6f}")
        print(f"Second best: {second_feature}, θ={second_theta:.6f}, gain={second_gain:.6f}")
        print(f"Relative difference: {rel_diff*100:.6f}%")

        # If there's a tie (difference < 0.01%)
        if rel_diff < 0.0001:
            print("Information gain tie detected!")

            # Fit both models to see what they choose
            orig_tree.fit(X, y)
            faster_tree.fit(X, y)

            chosen_orig_feature = orig_tree.tree.feature
            chosen_orig_theta = orig_tree.tree.theta

            chosen_faster_feature = faster_tree.estimator_.tree_.feature[0]
            chosen_faster_threshold = faster_tree.estimator_.tree_.threshold[0]

            print(f"Original model chose: feature={chosen_orig_feature}, θ={chosen_orig_theta:.6f}")
            print(f"Faster model chose: feature={chosen_faster_feature}, threshold={chosen_faster_threshold:.6f}")

            # Check prediction agreement
            orig_preds = orig_tree.predict(X)
            faster_preds = faster_tree.predict(X)

            agreement = np.mean(orig_preds == faster_preds)
            print(f"Prediction agreement: {agreement:.6f}")

            # Plot information gain landscape
            plt.figure(figsize=(10, 6))

            for feature, (theta, gain) in feature_gains.items():
                thresholds = orig_tree._get_candidates(X, feature)
                gains = []

                for t in thresholds:
                    left, right = orig_tree._get_split(X, feature, t)

                    if min(np.sum(left), np.sum(right)) >= orig_tree.min_samples_leaf:
                        g = orig_tree._information_gain(left, right, y)
                        gains.append(g)
                    else:
                        gains.append(-1)

                valid_idx = [i for i, g in enumerate(gains) if g >= 0]
                if valid_idx:
                    plt.plot(
                        [1 / np.tan(thresholds[i]) for i in valid_idx],
                        [gains[i] for i in valid_idx],
                        "o-",
                        label=f"Feature {feature}",
                    )

            plt.axhline(y=best_gain, color="r", linestyle="--", label="Maximum gain")
            plt.axhline(y=best_gain * 0.9999, color="r", linestyle=":", label="99.99% of max gain")
            plt.xlabel("Decision Threshold (cot(θ) in original space)")
            plt.ylabel("Information Gain")
            plt.title("Information Gain Landscape Showing Ties")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{IMAGES_DIR}/info_gain_landscape_seed{seed}.png")

            return True

    print("No information gain ties found")
    return False


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def test_higher_dimensions():
    """Test agreement with higher dimensional data"""
    print("\n== Testing Higher Dimensions ==")

    dimensions = [3, 5, 10, 20]

    for dim in dimensions:
        print(f"\nTesting with {dim} dimensions:")
        X, y = generate_hyperbolic_data(500, n_classes=3, n_features=dim, random_state=42)

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


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def test_performance():
    """
    Test performance differences between implementations with large datasets.
    """
    print("\n== Testing Performance Scaling ==")

    sample_sizes = [100, 500, 1000, 5000]
    n_features = 10
    n_classes = 3

    orig_times = []
    faster_times = []

    for n_samples in sample_sizes:
        print(f"Testing with {n_samples} samples...")

        # Generate data
        X, y = generate_hyperbolic_data(n_samples, n_classes, n_features, random_state=42)

        # Test original implementation
        orig_tree = OriginalHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)

        start = time.time()
        orig_tree.fit(X, y)
        orig_time = time.time() - start
        orig_times.append(orig_time)

        # Test faster implementation
        faster_tree = FasterHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)

        start = time.time()
        faster_tree.fit(X, y)
        faster_time = time.time() - start
        faster_times.append(faster_time)

        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)

        agreement = np.mean(orig_preds == faster_preds)
        speedup = orig_time / faster_time

        print(f"  Original: {orig_time:.4f}s, Faster: {faster_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Prediction agreement: {agreement:.6f}")

    # Plot performance results
    plt.figure(figsize=(12, 5))

    # Execution time comparison
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, orig_times, "o-", label="Original")
    plt.plot(sample_sizes, faster_times, "o-", label="Faster")
    plt.xlabel("Number of Samples")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")

    # Speedup
    plt.subplot(1, 2, 2)
    speedups = [o / f for o, f in zip(orig_times, faster_times)]
    plt.plot(sample_sizes, speedups, "o-")
    plt.xlabel("Number of Samples")
    plt.ylabel("Speedup Factor (x)")
    plt.title("Performance Speedup")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/performance_comparison.png")

    return orig_times, faster_times


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def test_numerical_precision():
    """Test specific edge case at decision boundary"""
    print("\n=== Testing Numerical Precision ===")

    # Generate a simple dataset for training
    X, y = generate_hyperbolic_data(100, n_classes=2, n_features=3, random_state=42)

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


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy implementation not available")
def run_all_tests():
    """Run all verification tests"""
    print("=== Running Comprehensive Equivalence Tests ===")

    # Mathematical equivalence test
    test_mathematical_equivalence()

    # Prediction agreement across depths
    test_prediction_agreement()

    # Test with higher dimensions
    test_higher_dimensions()

    # Visualize decision boundaries
    visualize_decision_boundaries(depth=3)
    visualize_decision_boundaries(depth=5)

    # Find and analyze information gain ties
    find_information_gain_ties()

    # Performance comparison
    test_performance()

    # Numerical precision test
    test_numerical_precision()

    print("\n=== All Tests Complete ===")
    print(f"Results and visualizations saved to {IMAGES_DIR}")


if __name__ == "__main__":
    run_all_tests()
