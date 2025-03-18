"""
Analyze specific points where the two implementations disagree.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import export_text

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


def find_and_save_disagreements():
    """Find cases where the two implementations disagree and save the data"""
    # Try different seeds and dimensions to find disagreements
    test_configs = [
        {"seed": 42, "n_features": 5, "n_classes": 3, "n_samples": 500, "depth": 5},
        {"seed": 43, "n_features": 7, "n_classes": 4, "n_samples": 500, "depth": 7},
        {"seed": 44, "n_features": 10, "n_classes": 5, "n_samples": 500, "depth": 10},
        # Add more configurations for extreme cases
        {"seed": 45, "n_features": 20, "n_classes": 10, "n_samples": 1000, "depth": 15},
    ]
    
    disagreements_data = []
    
    for config in test_configs:
        print(f"\nTesting configuration: {config}")
        
        # Generate data
        X, y = generate_hyperbolic_data(
            config["n_samples"], 
            n_classes=config["n_classes"], 
            n_features=config["n_features"],
            random_state=config["seed"]
        )
        
        # Create models
        orig_tree = OriginalHDTC(
            max_depth=config["depth"], 
            timelike_dim=0, 
            skip_hyperboloid_check=True
        )
        faster_tree = FasterHDTC(
            max_depth=config["depth"], 
            timelike_dim=0, 
            skip_hyperboloid_check=True
        )
        
        # Fit models
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)
        
        # Find disagreement indices
        disagreement_indices = np.where(orig_preds != faster_preds)[0]
        agreement_rate = 1 - len(disagreement_indices) / len(X)
        
        print(f"Prediction agreement rate: {agreement_rate:.4f}")
        print(f"Number of disagreements: {len(disagreement_indices)}")
        
        if len(disagreement_indices) > 0:
            # Save the first few disagreement points for analysis
            for idx in disagreement_indices[:min(5, len(disagreement_indices))]:
                point = X[idx]
                orig_pred = orig_preds[idx]
                faster_pred = faster_preds[idx]
                
                # Calculate the dot product (decision value) from original tree
                # Note: this is specific to the first disagreement point for simplicity
                node = orig_tree.tree
                dot_values = []
                
                def traverse_tree(node, point, path=None):
                    if path is None:
                        path = []
                    
                    if node.value is not None:  # Leaf
                        return path
                    
                    # Calculate dot product
                    feature = node.feature
                    theta = node.theta
                    dot_value = orig_tree._dot(point.reshape(1, -1), feature, theta).item()
                    decision = dot_value < 0
                    
                    path.append({
                        'feature': feature,
                        'theta': theta,
                        'dot_value': dot_value,
                        'decision': "Left" if decision else "Right"
                    })
                    
                    if decision:
                        traverse_tree(node.left, point, path)
                    else:
                        traverse_tree(node.right, point, path)
                    
                    return path
                
                orig_path = traverse_tree(orig_tree.tree, point)
                
                # Convert to Klein coordinates for the faster tree
                X_klein = np.delete(point, 0) / point[0]
                
                # Store disagreement data
                disagreements_data.append({
                    'config': config,
                    'point_idx': idx,
                    'point': point,
                    'klein_point': X_klein,
                    'orig_pred': orig_pred,
                    'faster_pred': faster_pred,
                    'orig_path': orig_path,
                })
    
    # Save the disagreements to a file if any were found
    if disagreements_data:
        with open('/home/phil/hyperdt/faster_tests/disagreements.pkl', 'wb') as f:
            pickle.dump(disagreements_data, f)
        print(f"\nSaved {len(disagreements_data)} disagreement cases to disagreements.pkl")
    else:
        print("\nNo disagreements found across all configurations.")


def analyze_extreme_points():
    """Analyze why extreme points show so much disagreement"""
    print("\n=== Analyzing Extreme Points ===")
    
    # Create a basic dataset and train both models
    X, y = generate_hyperbolic_data(500, n_classes=3, n_features=5, random_state=42)
    
    orig_tree = OriginalHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Generate extreme points on the hyperboloid
    extreme_points = []
    
    for scale in [10, 100, 1000]:
        for _ in range(5):
            # Generate large spacelike components
            spacelike = np.random.uniform(-scale, scale, 4)
            # Calculate corresponding timelike component to stay on hyperboloid
            timelike = np.sqrt(1 + np.sum(spacelike**2))
            # Create the point
            point = np.concatenate([[timelike], spacelike])
            extreme_points.append(point)
    
    extreme_points = np.array(extreme_points)
    
    # Get predictions
    orig_preds = orig_tree.predict(extreme_points)
    faster_preds = faster_tree.predict(extreme_points)
    
    # Analyze disagreements
    agreement = np.mean(orig_preds == faster_preds)
    print(f"Agreement rate on extreme points: {agreement:.4f}")
    
    # Check Klein coordinates
    print("\nComparing decision values for extreme points:")
    for i, point in enumerate(extreme_points):
        # Calculate the ratio values (Klein coordinates)
        ratios = point[1:] / point[0]
        
        # Get original dot product for the first node
        feature = orig_tree.tree.feature
        theta = orig_tree.tree.theta
        dot_value = orig_tree._dot(point.reshape(1, -1), feature, theta).item()
        
        # Get the decision
        orig_decision = dot_value < 0
        
        # For the faster tree, the decision is made on the Klein coordinate directly
        faster_decision = ratios[feature-1] < 1/np.tan(theta)
        
        print(f"Point {i} (scale ~{max(abs(point)):.0f}):")
        print(f"  Original prediction: {orig_preds[i]}, Faster prediction: {faster_preds[i]}")
        print(f"  Original dot value: {dot_value:.8f}, Decision: {'Left' if orig_decision else 'Right'}")
        print(f"  Ratio value: {ratios[feature-1]:.8f}, Threshold: {1/np.tan(theta):.8f}")
        print(f"  Klein decision: {'Left' if faster_decision else 'Right'}")
        print(f"  Agreement: {orig_decision == faster_decision}")
        
        # Calculate theoretical ratio from original formula
        # sin(θ)*x_d - cos(θ)*x_0 < 0  ⟹  x_d/x_0 < cot(θ)
        # Check if the theoretical equivalence holds
        theoretical_ratio = dot_value / (-np.cos(theta) * point[0]) * -1
        ratio_diff = abs(theoretical_ratio - ratios[feature-1])
        print(f"  Ratio difference: {ratio_diff:.8e}")
        print()


if __name__ == "__main__":
    find_and_save_disagreements()
    analyze_extreme_points()