"""
Testing script to identify differences between tree.py and faster_tree.py implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def test_candidates_comparison():
    """Compare the candidate split points between implementations"""
    print("\n=== TEST 1: Candidate Split Points Comparison ===")
    
    # Generate simple dataset
    X, y = generate_hyperbolic_data(20, n_classes=2, n_features=3, random_state=42)
    
    # Extract candidates from original implementation
    original = OriginalHDTC(max_depth=3)
    original.ndim = X.shape[1]
    original.timelike_dim = 0
    original.dims = [i for i in range(X.shape[1]) if i != 0]
    
    # Get candidates for each non-timelike dimension
    for dim in original.dims:
        print(f"\nTesting dimension {dim}:")
        orig_candidates = original._get_candidates(X, dim=dim)
        
        # Calculate Klein model ratios manually
        X_klein = np.delete(X, 0, axis=1) / X[:, 0][:, None]
        klein_values = np.sort(np.unique(X_klein[:, dim-1]))  # adjust index for removed timelike dim
        klein_midpoints = (klein_values[:-1] + klein_values[1:]) / 2
        
        # Calculate cotangent of original theta values for comparison
        orig_cot_theta = 1/np.tan(orig_candidates)
        
        # Sort both for easier comparison
        orig_cot_theta = np.sort(orig_cot_theta)
        klein_midpoints = np.sort(klein_midpoints)
        
        print(f"Original candidates count: {len(orig_candidates)}")
        print(f"Klein midpoints count: {len(klein_midpoints)}")
        
        # Compare values if counts match
        if len(orig_cot_theta) == len(klein_midpoints):
            max_diff = np.max(np.abs(orig_cot_theta - klein_midpoints))
            print(f"Maximum difference between cot(θ) and Klein midpoints: {max_diff}")
            
            # Print first few values for inspection
            print("First few original cot(θ) values:", orig_cot_theta[:5])
            print("First few Klein midpoints:", klein_midpoints[:5])
        else:
            print("Candidate counts don't match, cannot compare directly.")
            
            # Print first few values for inspection
            print("First few original cot(θ) values:", orig_cot_theta[:5])
            print("First few Klein midpoints:", klein_midpoints[:5])


def test_single_split():
    """Test a single-split tree (depth=1) to see if the implementations choose the same split"""
    print("\n=== TEST 2: Single Split Comparison ===")
    
    for i in range(5):  # Try with different random seeds
        seed = 42 + i
        print(f"\nTest with random seed {seed}:")
        
        # Generate data
        X, y = generate_hyperbolic_data(100, n_classes=2, n_features=3, random_state=seed)
        
        # Create trees with identical parameters
        orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, curvature=1.0, 
                                skip_hyperboloid_check=True)
        faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, curvature=1.0, 
                               skip_hyperboloid_check=True)
        
        # Fit trees
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Extract and compare split information
        if orig_tree.tree.feature is not None:
            print(f"Original split: feature={orig_tree.tree.feature}, θ={orig_tree.tree.theta:.6f}")
            cot_theta = 1/np.tan(orig_tree.tree.theta)
            print(f"Original cot(θ)={cot_theta:.6f}")
            
            # For faster tree, extract from sklearn's tree structure
            faster_feature = faster_tree.tree.tree_.feature[0]
            faster_threshold = faster_tree.tree.tree_.threshold[0]
            print(f"Faster split: feature={faster_feature}, threshold={faster_threshold:.6f}")
            
            # Compare predictions
            orig_preds = orig_tree.predict(X)
            faster_preds = faster_tree.predict(X)
            agreement = np.mean(orig_preds == faster_preds)
            
            print(f"Prediction agreement: {agreement:.4f}")
            
            # Visualization of split
            if orig_tree.tree.feature == 1 and faster_feature == 0:  # both using the first non-timelike dimension
                timelike = X[:, 0]
                spacelike = X[:, 1]
                
                plt.figure(figsize=(12, 5))
                
                # Original split
                plt.subplot(1, 2, 1)
                plt.scatter(spacelike, timelike, c=orig_preds)
                # Draw original decision boundary: sin(θ)*x_1 - cos(θ)*x_0 = 0
                x1_vals = np.linspace(np.min(spacelike), np.max(spacelike), 100)
                x0_vals = np.sin(orig_tree.tree.theta) * x1_vals / np.cos(orig_tree.tree.theta)
                plt.plot(x1_vals, x0_vals, 'r-', linewidth=2)
                plt.xlabel('x₁ (spacelike)')
                plt.ylabel('x₀ (timelike)')
                plt.title('Original Tree Split')
                
                # Faster split
                plt.subplot(1, 2, 2)
                plt.scatter(spacelike, timelike, c=faster_preds)
                # Draw faster decision boundary: x_1/x_0 = threshold
                x1_vals = np.linspace(np.min(spacelike), np.max(spacelike), 100)
                x0_vals = x1_vals / faster_threshold
                plt.plot(x1_vals, x0_vals, 'r-', linewidth=2)
                plt.xlabel('x₁ (spacelike)')
                plt.ylabel('x₀ (timelike)')
                plt.title('Faster Tree Split')
                
                plt.tight_layout()
                plt.savefig(f'/home/phil/hyperdt/faster_tests/split_comparison_seed{seed}.png')
                plt.close()


def test_prediction_differences():
    """Identify specific points where predictions differ and analyze why"""
    print("\n=== TEST 3: Prediction Differences Analysis ===")
    
    # Generate larger dataset to find disagreements
    X, y = generate_hyperbolic_data(500, n_classes=2, n_features=3, random_state=42)
    
    # Create and fit trees with higher depth to see more complex differences
    orig_tree = OriginalHDTC(max_depth=3, timelike_dim=0, curvature=1.0, 
                            skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=3, timelike_dim=0, curvature=1.0, 
                           skip_hyperboloid_check=True)
    
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Get predictions
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)
    
    # Find disagreements
    disagreement_indices = np.where(orig_preds != faster_preds)[0]
    agreement_rate = 1 - len(disagreement_indices) / len(X)
    
    print(f"Overall agreement rate: {agreement_rate:.4f}")
    print(f"Number of disagreements: {len(disagreement_indices)} out of {len(X)}")
    
    if len(disagreement_indices) > 0:
        # Analyze a few disagreement points
        for i, idx in enumerate(disagreement_indices[:min(5, len(disagreement_indices))]):
            point = X[idx]
            print(f"\nDisagreement point {i+1}: X[{idx}] = {point}")
            print(f"  Original prediction: {orig_preds[idx]}, Faster prediction: {faster_preds[idx]}")
            
            # Trace through original tree manually
            node = orig_tree.tree
            path = ["Root"]
            while node.value is None:  # Until we reach a leaf
                feature = node.feature
                theta = node.theta
                decision = orig_tree._dot(point.reshape(1, -1), feature, theta).item() < 0
                direction = "Left" if decision else "Right"
                path.append(f"{direction} at feature {feature}, θ={theta:.4f} (cot(θ)={1/np.tan(theta):.4f})")
                node = node.left if decision else node.right
            path.append(f"Leaf: {node.value}")
            
            print("  Original tree path:")
            for step in path:
                print(f"    {step}")
            
            # For faster tree, convert to Klein coordinates
            X_klein = np.delete(point, 0) / point[0]
            
            # Print Klein coordinates
            print(f"  Klein coordinates: {X_klein}")
            
            # No clean way to trace sklearn's decision path directly, but can print tree
            if i == 0:  # Only for the first disagreement
                tree_text = export_text(faster_tree.tree, feature_names=[f'x{j+1}/x0' for j in range(len(X_klein))])
                print(f"  Faster tree structure (in Klein coordinates):")
                print(tree_text)
                
                # Try to find a manual trace for this point
                print("  Manual trace through faster tree:")
                x_klein_dict = {f'x{j+1}/x0': X_klein[j] for j in range(len(X_klein))}
                manual_trace = []
                
                for line in tree_text.split('\n'):
                    if not line.strip():
                        continue
                        
                    indent = len(line) - len(line.lstrip())
                    depth = indent // 4
                    
                    if 'class' in line:  # Leaf node
                        manual_trace.append(f"{' ' * (depth * 2)}Leaf: {line.strip()}")
                        break
                        
                    if '<=' in line:  # Decision node
                        parts = line.strip().split('<=')
                        feature = parts[0].strip()
                        threshold = float(parts[1].strip().split()[0])
                        
                        if feature in x_klein_dict and x_klein_dict[feature] <= threshold:
                            manual_trace.append(f"{' ' * (depth * 2)}Feature {feature} <= {threshold} [TRUE]")
                        else:
                            manual_trace.append(f"{' ' * (depth * 2)}Feature {feature} <= {threshold} [FALSE]")
                            # Skip to the next depth level that's less than or equal to this one
                            skip_depth = depth + 1
                            for i, nextline in enumerate(tree_text.split('\n')[tree_text.split('\n').index(line)+1:]):
                                if not nextline.strip():
                                    continue
                                next_indent = len(nextline) - len(nextline.lstrip())
                                next_depth = next_indent // 4
                                if next_depth <= depth:
                                    break
                                    
                for step in manual_trace:
                    print(f"    {step}")


def test_edge_cases():
    """Test specific edge cases that might cause discrepancies"""
    print("\n=== TEST 4: Edge Cases ===")
    
    # 1. Points close to the decision boundary
    print("\nTesting points near decision boundary:")
    
    # Create simple data with one threshold boundary
    X, y = generate_hyperbolic_data(50, n_classes=2, n_features=3, random_state=42)
    
    # Train depth-1 trees
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0)
    
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Get decision boundary parameters
    orig_feature = orig_tree.tree.feature
    orig_theta = orig_tree.tree.theta
    orig_cot_theta = 1/np.tan(orig_theta)
    
    faster_feature = faster_tree.tree.tree_.feature[0]
    faster_threshold = faster_tree.tree.tree_.threshold[0]
    
    print(f"Original boundary: feature={orig_feature}, θ={orig_theta:.6f}, cot(θ)={orig_cot_theta:.6f}")
    print(f"Faster boundary: feature={faster_feature}, threshold={faster_threshold:.6f}")
    
    # Generate points very close to the boundary
    num_test_points = 20
    boundary_points = []
    
    # Base point template
    base_point = np.ones(X.shape[1])
    base_point[0] = np.sqrt(1 + np.sum(base_point[1:]**2))  # Ensure on hyperboloid
    
    # Create points slightly on both sides of the boundary
    for i in range(num_test_points):
        # Generate a random epsilon
        epsilon = (np.random.random() - 0.5) * 0.0002
        
        # Create a point where x_feature/x_0 is very close to the boundary
        test_point = base_point.copy()
        test_point[orig_feature] = (orig_cot_theta + epsilon) * test_point[0]
        test_point[0] = np.sqrt(1 + np.sum(test_point[1:]**2))  # Recalculate timelike to ensure on hyperboloid
        
        boundary_points.append(test_point)
    
    boundary_points = np.array(boundary_points)
    
    # Get predictions for these points
    orig_preds = orig_tree.predict(boundary_points)
    faster_preds = faster_tree.predict(boundary_points)
    
    # Compare
    agreement_rate = np.mean(orig_preds == faster_preds)
    print(f"Agreement rate for boundary points: {agreement_rate:.4f}")
    
    for i, (op, fp) in enumerate(zip(orig_preds, faster_preds)):
        if op != fp:
            point = boundary_points[i]
            ratio = point[orig_feature] / point[0]
            print(f"  Disagreement at point with ratio {ratio:.8f}")
            orig_dot = orig_tree._dot(point.reshape(1, -1), orig_feature, orig_theta).item()
            print(f"  Original dot product: {orig_dot:.8f}")
            print(f"  Difference from boundary: {ratio - orig_cot_theta:.8f}")


if __name__ == "__main__":
    # Run all tests
    test_candidates_comparison()
    test_single_split()
    test_prediction_differences()
    test_edge_cases()