"""
Test information gain calculations between the two implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree._tree import TREE_LEAF
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


def calculate_original_gains(X, y, orig_tree, feature_space=None):
    """Calculate information gain for all features and thresholds using original implementation"""
    # Initialize tree parameters
    orig_tree.ndim = X.shape[1]
    orig_tree.timelike_dim = 0
    orig_tree.dims = [i for i in range(X.shape[1]) if i != orig_tree.timelike_dim]
    orig_tree.classes_ = np.unique(y)
    
    # If feature space not specified, use all features
    if feature_space is None:
        feature_space = orig_tree.dims
        
    # Dictionary to store feature -> (thresholds, gains)
    feature_gains = {}
    
    # For each feature, get all candidate thresholds and calculate gain
    for feature in feature_space:
        thresholds = orig_tree._get_candidates(X, feature)
        gains = []
        
        for theta in thresholds:
            left, right = orig_tree._get_split(X, feature, theta)
            
            # Check if split is valid according to min_samples_leaf
            if min(np.sum(left), np.sum(right)) >= orig_tree.min_samples_leaf:
                gain = orig_tree._information_gain(left, right, y)
                gains.append(gain)
            else:
                gains.append(-1)  # Invalid split
                
        feature_gains[feature] = (thresholds, gains)
    
    return feature_gains


def convert_to_klein(X, timelike_dim=0):
    """Convert hyperboloid coordinates to Klein coordinates"""
    x0 = X[:, timelike_dim]
    X_klein = np.delete(X, timelike_dim, axis=1) / x0[:, None]
    return X_klein


def calculate_sklearn_gains(X, y, faster_tree):
    """Calculate information gain for all features and thresholds using sklearn implementation"""
    # Convert to Klein coordinates
    X_klein = convert_to_klein(X, timelike_dim=faster_tree.timelike_dim)
    
    # Create reference to the sklearn tree inside FasterHDTC
    tree = faster_tree.tree
    
    # Dictionary to store feature -> (thresholds, gains)
    feature_gains = {}
    
    # For each feature, get all candidate thresholds and calculate gain
    for feature in range(X_klein.shape[1]):
        # Get unique values and calculate thresholds between them
        unique_vals = np.sort(np.unique(X_klein[:, feature]))
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        
        # We need to fit trees with just this threshold to get the gain
        gains = []
        
        for threshold in thresholds:
            # Create a split
            left = X_klein[:, feature] <= threshold
            right = ~left
            
            # Check if valid according to min_samples_leaf
            if min(np.sum(left), np.sum(right)) >= tree.min_samples_leaf:
                # Calculate Gini impurity of parent
                parent_classes, parent_counts = np.unique(y, return_counts=True)
                parent_probs = parent_counts / len(y)
                parent_gini = 1 - np.sum(parent_probs**2)
                
                # Calculate Gini impurity of children
                left_classes, left_counts = np.unique(y[left], return_counts=True)
                left_probs = left_counts / len(y[left])
                left_gini = 1 - np.sum(left_probs**2)
                
                right_classes, right_counts = np.unique(y[right], return_counts=True)
                right_probs = right_counts / len(y[right])
                right_gini = 1 - np.sum(right_probs**2)
                
                # Calculate weighted sum of child impurities
                child_gini = (np.sum(left) * left_gini + np.sum(right) * right_gini) / len(y)
                
                # Information gain = parent impurity - child impurity
                gain = parent_gini - child_gini
                gains.append(gain)
            else:
                gains.append(-1)  # Invalid split
        
        feature_gains[feature] = (thresholds, gains)
    
    return feature_gains


def find_tiebreakers(seed):
    """Find and analyze cases where information gain values are very close"""
    print(f"\n=== Analyzing Information Gain with Seed {seed} ===")
    
    # Generate data
    X, y = generate_hyperbolic_data(100, n_classes=3, n_features=5, random_state=seed)
    
    # Create models
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    
    # Calculate gains using original implementation
    orig_gains = calculate_original_gains(X, y, orig_tree)
    
    # Find best splits according to original implementation
    best_orig_feature = -1
    best_orig_threshold = -1
    best_orig_gain = -1
    
    for feature, (thresholds, gains) in orig_gains.items():
        for i, gain in enumerate(gains):
            if gain > best_orig_gain:
                best_orig_feature = feature
                best_orig_threshold = thresholds[i]
                best_orig_gain = gain
    
    # Calculate gains using sklearn implementation
    faster_gains = calculate_sklearn_gains(X, y, faster_tree)
    
    # Find best splits according to sklearn implementation
    best_faster_feature = -1
    best_faster_threshold = -1
    best_faster_gain = -1
    
    for feature, (thresholds, gains) in faster_gains.items():
        for i, gain in enumerate(gains):
            if gain > best_faster_gain:
                best_faster_feature = feature
                best_faster_threshold = thresholds[i]
                best_faster_gain = gain
    
    # Convert original feature and threshold to Klein coordinates for comparison
    orig_feature_klein = best_orig_feature - 1 if best_orig_feature > 0 else best_orig_feature
    orig_threshold_klein = 1 / np.tan(best_orig_threshold)
    
    print(f"Best original split: feature={best_orig_feature}, θ={best_orig_threshold:.6f}, "
          f"cot(θ)={orig_threshold_klein:.6f}, gain={best_orig_gain:.6f}")
    print(f"Best faster split: feature={best_faster_feature}, threshold={best_faster_threshold:.6f}, "
          f"gain={best_faster_gain:.6f}")
          
    # Check for close alternatives in the original implementation
    close_alternatives_orig = []
    
    for feature, (thresholds, gains) in orig_gains.items():
        for i, gain in enumerate(gains):
            # Look for gains that are within 1% of the best gain
            if gain > 0 and abs(gain - best_orig_gain) / best_orig_gain < 0.01:
                threshold_klein = 1 / np.tan(thresholds[i])
                close_alternatives_orig.append((feature, thresholds[i], threshold_klein, gain))
    
    # Check for close alternatives in the faster implementation
    close_alternatives_faster = []
    
    for feature, (thresholds, gains) in faster_gains.items():
        for i, gain in enumerate(gains):
            # Look for gains that are within 1% of the best gain
            if gain > 0 and abs(gain - best_faster_gain) / best_faster_gain < 0.01:
                close_alternatives_faster.append((feature, thresholds[i], gain))
    
    print(f"\nFound {len(close_alternatives_orig)} alternatives within 1% of best original gain:")
    for feature, theta, cot_theta, gain in close_alternatives_orig:
        print(f"  Feature {feature}, θ={theta:.6f}, cot(θ)={cot_theta:.6f}, gain={gain:.6f}, "
              f"diff={abs(gain-best_orig_gain):.6f}, rel_diff={(abs(gain-best_orig_gain)/best_orig_gain)*100:.2f}%")
    
    print(f"\nFound {len(close_alternatives_faster)} alternatives within 1% of best faster gain:")
    for feature, threshold, gain in close_alternatives_faster:
        print(f"  Feature {feature}, threshold={threshold:.6f}, gain={gain:.6f}, "
              f"diff={abs(gain-best_faster_gain):.6f}, rel_diff={(abs(gain-best_faster_gain)/best_faster_gain)*100:.2f}%")
    
    # Actually fit the models to see what they chose
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Get chosen splits
    chosen_orig_feature = orig_tree.tree.feature
    chosen_orig_theta = orig_tree.tree.theta
    chosen_orig_cot = 1 / np.tan(chosen_orig_theta)
    
    chosen_faster_feature = faster_tree.tree.tree_.feature[0]
    chosen_faster_threshold = faster_tree.tree.tree_.threshold[0]
    
    print("\nActual chosen splits:")
    print(f"  Original: feature={chosen_orig_feature}, θ={chosen_orig_theta:.6f}, cot(θ)={chosen_orig_cot:.6f}")
    print(f"  Faster: feature={chosen_faster_feature}, threshold={chosen_faster_threshold:.6f}")
    
    # Check prediction agreement
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)
    
    agreement = np.mean(orig_preds == faster_preds)
    print(f"\nPrediction agreement: {agreement:.6f}")
    
    # If there are close alternatives with different features, plot the 
    # information gain landscape for both implementations
    if len(close_alternatives_orig) > 1 or len(close_alternatives_faster) > 1:
        plt.figure(figsize=(15, 5))
        
        # Plot original gains
        plt.subplot(1, 2, 1)
        for feature, (thresholds, gains) in orig_gains.items():
            valid_idx = [i for i, g in enumerate(gains) if g >= 0]
            if valid_idx:
                plt.plot(np.array(thresholds)[valid_idx], np.array(gains)[valid_idx], 
                         'o-', label=f"Feature {feature}")
        
        plt.axhline(y=best_orig_gain * 0.99, color='r', linestyle='--', label="99% of best gain")
        plt.xlabel("Theta (angle)")
        plt.ylabel("Information Gain")
        plt.title("Original Implementation Gains")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot faster gains
        plt.subplot(1, 2, 2)
        for feature, (thresholds, gains) in faster_gains.items():
            valid_idx = [i for i, g in enumerate(gains) if g >= 0]
            if valid_idx:
                plt.plot(np.array(thresholds)[valid_idx], np.array(gains)[valid_idx], 
                         'o-', label=f"Feature {feature}")
        
        plt.axhline(y=best_faster_gain * 0.99, color='r', linestyle='--', label="99% of best gain")
        plt.xlabel("Threshold (Klein coordinate)")
        plt.ylabel("Information Gain")
        plt.title("Faster Implementation Gains")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/home/phil/hyperdt/faster_tests/info_gain_landscape_seed{seed}.png')


def test_multiple_seeds():
    """Test information gain landscapes for multiple seeds to find tiebreakers"""
    # Test a range of seeds
    seeds = [42, 43, 44, 45, 46]
    
    for seed in seeds:
        find_tiebreakers(seed)


if __name__ == "__main__":
    test_multiple_seeds()