"""
Examine the specific disagreement cases in detail.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from hyperdt.tree import HyperbolicDecisionTreeClassifier as OriginalHDTC
from hyperdt.faster_tree import HyperbolicDecisionTreeClassifier as FasterHDTC
from hyperdt.toy_data import wrapped_normal_mixture


def load_and_examine_disagreements():
    """Load the saved disagreement data and analyze it in detail"""
    try:
        with open('/home/phil/hyperdt/faster_tests/disagreements.pkl', 'rb') as f:
            disagreements = pickle.load(f)
    except FileNotFoundError:
        print("No disagreements file found. Run analyze_disagreements.py first.")
        return
    
    print(f"Loaded {len(disagreements)} disagreement cases.")
    
    for i, case in enumerate(disagreements):
        print(f"\n=== Disagreement Case {i+1} ===")
        config = case['config']
        print(f"Configuration: {config}")
        print(f"Point index: {case['point_idx']}")
        print(f"Original prediction: {case['orig_pred']}, Faster prediction: {case['faster_pred']}")
        
        point = case['point']
        print(f"Hyperboloid point: {point}")
        print(f"Klein coordinates: {case['klein_point']}")
        
        # Examine the decision path
        print("\nOriginal Tree Decision Path:")
        for j, step in enumerate(case['orig_path']):
            print(f"  Step {j+1}: Feature {step['feature']}, θ={step['theta']:.6f}, "
                  f"dot_value={step['dot_value']:.8f}, Decision: {step['decision']}")
            
            # Calculate the ratio threshold
            cot_theta = 1/np.tan(step['theta'])
            ratio_value = point[step['feature']] / point[0]
            
            print(f"    cot(θ)={cot_theta:.8f}, ratio_value={ratio_value:.8f}")
            print(f"    Theoretical decision: {'Left' if ratio_value < cot_theta else 'Right'}")
            
            # Check for numerical precision issue
            ratio_diff = abs(ratio_value - cot_theta)
            if ratio_diff < 1e-7:
                print(f"    *** POTENTIAL ISSUE: Very close to decision boundary (diff={ratio_diff:.10f}) ***")
                
                # Check alternative calculation
                dot_product = np.sin(step['theta']) * point[step['feature']] - np.cos(step['theta']) * point[0]
                print(f"    Direct dot product: {dot_product:.10f}")
                
                # Calculate the Klein representation of the decision boundary
                klein_decision = case['klein_point'][step['feature']-1] < cot_theta
                print(f"    Klein decision: {'Left' if klein_decision else 'Right'}")
        
        # Regenerate the data and refit both models to verify
        print("\nVerifying with fresh models:")
        X, y = wrapped_normal_mixture(
            num_points=config['n_samples'], 
            num_classes=config['n_classes'], 
            num_dims=config['n_features']-1,
            noise_std=1.0,
            seed=config['seed'],
            adjust_for_dim=True
        )
        
        # Create and fit models
        orig_tree = OriginalHDTC(
            max_depth=config['depth'], 
            timelike_dim=0, 
            skip_hyperboloid_check=True
        )
        faster_tree = FasterHDTC(
            max_depth=config['depth'], 
            timelike_dim=0, 
            skip_hyperboloid_check=True
        )
        
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Check predictions again - convert point to numpy array
        point_np = np.array([point])
        orig_pred = orig_tree.predict(point_np)[0]
        faster_pred = faster_tree.predict(point_np)[0]
        
        print(f"New original prediction: {orig_pred}")
        print(f"New faster prediction: {faster_pred}")
        
        # Check at the exact decision boundary to see the behavior
        print("\nAnalyzing numerical precision at decision boundaries:")
        
        for j, step in enumerate(case['orig_path']):
            theta = step['theta']
            feature = step['feature']
            cot_theta = 1/np.tan(theta)
            
            # Create points very slightly on either side of the boundary
            epsilon = 1e-10
            
            # Make a base point that lies exactly on the decision boundary
            base_point = np.ones(len(point))
            base_point[0] = 2.0  # arbitrary timelike component
            base_point[feature] = cot_theta * base_point[0]  # exactly on boundary
            
            # Make sure it's on the hyperboloid (adjust timelike component)
            spacelike_norm_squared = np.sum(base_point[1:]**2)
            base_point[0] = np.sqrt(1 + spacelike_norm_squared)  # proper hyperboloid point
            
            # Create left and right points
            left_point = base_point.copy()
            left_point[feature] = (cot_theta - epsilon) * left_point[0]
            
            right_point = base_point.copy()
            right_point[feature] = (cot_theta + epsilon) * right_point[0]
            
            # Test with both implementations
            orig_left = orig_tree._dot(left_point.reshape(1, -1), feature, theta).item() < 0
            orig_right = orig_tree._dot(right_point.reshape(1, -1), feature, theta).item() < 0
            
            # For faster tree, convert to Klein first
            left_klein = left_point[feature] / left_point[0]
            right_klein = right_point[feature] / right_point[0]
            
            faster_left = left_klein < cot_theta
            faster_right = right_klein < cot_theta
            
            print(f"\nStep {j+1} boundary test:")
            print(f"  Original left: {'Left' if orig_left else 'Right'}, "
                  f"right: {'Left' if orig_right else 'Right'}")
            print(f"  Faster left: {'Left' if faster_left else 'Right'}, "
                  f"right: {'Left' if faster_right else 'Right'}")
            
            # Check dot product calculation directly
            orig_dot_left = np.sin(theta) * left_point[feature] - np.cos(theta) * left_point[0]
            orig_dot_right = np.sin(theta) * right_point[feature] - np.cos(theta) * right_point[0]
            
            print(f"  Original dot left: {orig_dot_left:.15f} ({'Left' if orig_dot_left < 0 else 'Right'})")
            print(f"  Original dot right: {orig_dot_right:.15f} ({'Left' if orig_dot_right < 0 else 'Right'})")


def test_specific_edge_case():
    """Create a targeted test for the decision boundary edge case"""
    print("\n=== Testing Specific Edge Case ===")
    
    # Generate a simple dataset for training
    X, y = wrapped_normal_mixture(
        num_points=100, 
        num_classes=2, 
        num_dims=2,  # 3D hyperboloid
        noise_std=1.0,
        seed=42,
        adjust_for_dim=True
    )
    
    # Create and fit simple trees
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Extract decision boundary parameters
    feature = orig_tree.tree.feature
    theta = orig_tree.tree.theta
    cot_theta = 1/np.tan(theta)
    
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
        norm = np.sqrt(np.sum(base_point[1:]**2) + 1)
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
        
        print(f"{eps:10.1e} {orig_preds[i]:10d} {faster_preds[i]:10d} {str(match):>10} "
              f"{ratio:15.12f} {dot_val:15.12f}")
    
    # Create more test points with slightly larger deviations
    larger_epsilons = [-1e-6, -1e-5, 1e-5, 1e-6]
    test_points_larger = []
    
    for eps in larger_epsilons:
        base_point = np.ones(3)
        base_point[0] = 2.0
        base_point[feature] = (cot_theta + eps) * base_point[0]
        norm = np.sqrt(np.sum(base_point[1:]**2) + 1)
        base_point[0] = norm
        test_points_larger.append(base_point)
    
    test_points_larger = np.array(test_points_larger)
    
    # Get predictions
    orig_preds_larger = orig_tree.predict(test_points_larger)
    faster_preds_larger = faster_tree.predict(test_points_larger)
    
    print("\nLarger deviation points:")
    print(f"{'Epsilon':>10} {'Original':>10} {'Faster':>10} {'Match':>10} {'Ratio':>15} {'Dot Product':>15}")
    print("-" * 75)
    
    for i, eps in enumerate(larger_epsilons):
        point = test_points_larger[i]
        ratio = point[feature] / point[0]
        dot_val = orig_tree._dot(point.reshape(1, -1), feature, theta).item()
        match = orig_preds_larger[i] == faster_preds_larger[i]
        
        print(f"{eps:10.1e} {orig_preds_larger[i]:10d} {faster_preds_larger[i]:10d} {str(match):>10} "
              f"{ratio:15.12f} {dot_val:15.12f}")


if __name__ == "__main__":
    load_and_examine_disagreements()
    test_specific_edge_case()