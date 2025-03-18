"""
Test script for comparing more complex models between the two implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

from hyperdt.tree import HyperbolicDecisionTreeClassifier as OriginalHDTC
from hyperdt.faster_tree import HyperbolicDecisionTreeClassifier as FasterHDTC
from hyperdt.toy_data import wrapped_normal_mixture


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


def test_higher_dimensions():
    """Test agreement with higher dimensional data"""
    print("\n=== Testing Higher Dimensions ===")
    
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


def test_deep_trees():
    """Test agreement with deeper tree models"""
    print("\n=== Testing Deep Trees ===")
    
    depths = [1, 3, 5, 10, None]  # None = unlimited depth
    
    for depth in depths:
        depth_str = str(depth) if depth is not None else "unlimited"
        print(f"\nTesting with max_depth={depth_str}:")
        
        X, y = generate_hyperbolic_data(500, n_classes=3, n_features=5, random_state=42)
        
        # Create models
        orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
        faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
        
        # Fit models
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)
        
        agreement = np.mean(orig_preds == faster_preds)
        print(f"Prediction agreement rate: {agreement:.4f}")
        
        # If disagreements exist, analyze
        if agreement < 1.0:
            disagreement_indices = np.where(orig_preds != faster_preds)[0]
            print(f"Number of disagreements: {len(disagreement_indices)}")
            
            # Check test set performance
            X_test, y_test = generate_hyperbolic_data(200, n_classes=3, n_features=5, random_state=43)
            
            orig_score = orig_tree.score(X_test, y_test)
            faster_score = faster_tree.score(X_test, y_test)
            
            print(f"Original tree test accuracy: {orig_score:.4f}")
            print(f"Faster tree test accuracy: {faster_score:.4f}")


def test_numerous_classes():
    """Test agreement with a large number of classes"""
    print("\n=== Testing Numerous Classes ===")
    
    class_counts = [2, 5, 10, 20]
    
    for n_classes in class_counts:
        print(f"\nTesting with {n_classes} classes:")
        
        X, y = generate_hyperbolic_data(500, n_classes=n_classes, n_features=5, random_state=42)
        
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


def test_different_hyperparameters():
    """Test agreement with various hyperparameter settings"""
    print("\n=== Testing Different Hyperparameters ===")
    
    # Generate data once
    X, y = generate_hyperbolic_data(500, n_classes=3, n_features=5, random_state=42)
    
    hyperparams = [
        {"min_samples_leaf": 1, "min_samples_split": 2},
        {"min_samples_leaf": 5, "min_samples_split": 10},
        {"min_samples_leaf": 10, "min_samples_split": 20},
        {"min_samples_leaf": 20, "min_samples_split": 50}
    ]
    
    for params in hyperparams:
        print(f"\nTesting with {params}:")
        
        # Create models with the specified hyperparameters
        orig_tree = OriginalHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True, **params)
        faster_tree = FasterHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True, **params)
        
        # Fit models
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)
        
        agreement = np.mean(orig_preds == faster_preds)
        print(f"Prediction agreement rate: {agreement:.4f}")


def test_stress_data():
    """Test agreement with particularly challenging data distributions"""
    print("\n=== Testing Stress Data Distributions ===")
    
    # Test with a large dataset for numerical stability issues
    print("\nTesting large dataset:")
    X, y = generate_hyperbolic_data(5000, n_classes=10, n_features=10, random_state=42)
    
    # Create models
    orig_tree = OriginalHDTC(max_depth=10, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=10, timelike_dim=0, skip_hyperboloid_check=True)
    
    # Time fitting
    start = time.time()
    orig_tree.fit(X, y)
    orig_time = time.time() - start
    
    start = time.time()
    faster_tree.fit(X, y)
    faster_time = time.time() - start
    
    print(f"Original tree fit time: {orig_time:.4f}s")
    print(f"Faster tree fit time: {faster_time:.4f}s")
    print(f"Speedup: {orig_time / faster_time:.2f}x")
    
    # Compare predictions
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)
    
    agreement = np.mean(orig_preds == faster_preds)
    print(f"Prediction agreement rate: {agreement:.4f}")
    
    # Test points with extreme values
    print("\nTesting extreme values:")
    # Create some extreme points still on the hyperboloid
    extreme_points = []
    
    for _ in range(20):
        # Generate large spacelike components
        spacelike = np.random.uniform(-10, 10, X.shape[1] - 1)
        # Calculate corresponding timelike component to stay on hyperboloid
        timelike = np.sqrt(1 + np.sum(spacelike**2))
        # Create the point
        point = np.concatenate([[timelike], spacelike])
        extreme_points.append(point)
    
    extreme_points = np.array(extreme_points)
    
    # Get predictions
    orig_extreme_preds = orig_tree.predict(extreme_points)
    faster_extreme_preds = faster_tree.predict(extreme_points)
    
    extreme_agreement = np.mean(orig_extreme_preds == faster_extreme_preds)
    print(f"Prediction agreement rate on extreme points: {extreme_agreement:.4f}")


if __name__ == "__main__":
    test_higher_dimensions()
    test_deep_trees()
    test_numerous_classes()
    test_different_hyperparameters()
    test_stress_data()