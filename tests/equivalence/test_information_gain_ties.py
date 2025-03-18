"""
Test information gain calculations to find close ties between features.
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


def find_ties_in_dataset(seed, tie_threshold=0.001):
    """Find and analyze cases where information gain values are very close"""
    print(f"\n=== Finding Ties with Seed {seed}, Threshold {tie_threshold*100:.3f}% ===")
    
    # Generate data
    X, y = generate_hyperbolic_data(100, n_classes=3, n_features=5, random_state=seed)
    
    # Create models
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    
    # Calculate gains using original implementation
    orig_gains = calculate_original_gains(X, y, orig_tree)
    
    # Find best gains for each feature in original implementation
    best_gains_by_feature_orig = {}
    
    for feature, (thresholds, gains) in orig_gains.items():
        if any(g > 0 for g in gains):
            best_idx = np.argmax(gains)
            best_gains_by_feature_orig[feature] = (thresholds[best_idx], gains[best_idx])
    
    # Sort features by their best gain
    features_by_gain_orig = sorted(best_gains_by_feature_orig.items(), 
                                  key=lambda x: x[1][1], reverse=True)
    
    # Find the global best gain
    if features_by_gain_orig:
        best_feature_orig, (best_threshold_orig, best_gain_orig) = features_by_gain_orig[0]
        
        # Check for ties
        ties = []
        for feature, (threshold, gain) in features_by_gain_orig:
            if feature != best_feature_orig and (best_gain_orig - gain) / best_gain_orig <= tie_threshold:
                ties.append((feature, threshold, gain))
        
        if ties:
            print(f"Found ties in original implementation:")
            print(f"  Best split: feature={best_feature_orig}, θ={best_threshold_orig:.6f}, "
                  f"cot(θ)={1/np.tan(best_threshold_orig):.6f}, gain={best_gain_orig:.6f}")
            
            for feature, threshold, gain in ties:
                print(f"  Tie: feature={feature}, θ={threshold:.6f}, "
                      f"cot(θ)={1/np.tan(threshold):.6f}, gain={gain:.6f}, "
                      f"diff={best_gain_orig-gain:.8f}, rel_diff={((best_gain_orig-gain)/best_gain_orig)*100:.6f}%")
            
            # Now fit the actual tree to see which one it picked
            orig_tree.fit(X, y)
            chosen_feature = orig_tree.tree.feature
            chosen_theta = orig_tree.tree.theta
            
            if chosen_feature == best_feature_orig:
                print(f"  Original implementation chose the highest gain split")
            else:
                print(f"  Original implementation chose feature={chosen_feature}, θ={chosen_theta:.6f}, "
                      f"cot(θ)={1/np.tan(chosen_theta):.6f}")
                
                # Find where this ranks in the gains
                for i, (feature, (threshold, gain)) in enumerate(features_by_gain_orig):
                    if feature == chosen_feature:
                        print(f"  This is the #{i+1} highest gain split")
                        break
            
            # Check if the faster implementation makes the same choice
            faster_tree.fit(X, y)
            faster_feature = faster_tree.tree.tree_.feature[0]
            faster_threshold = faster_tree.tree.tree_.threshold[0]
            
            # Convert original feature to Klein space for comparison
            orig_feature_klein = best_feature_orig - 1 if best_feature_orig > 0 else best_feature_orig
            
            if faster_feature == orig_feature_klein:
                print(f"  Faster implementation chose the same feature")
            else:
                # Find the corresponding feature in the original space
                faster_feature_orig = faster_feature + 1 if faster_feature >= orig_tree.timelike_dim else faster_feature
                
                # See where this ranks in the original gains
                for i, (feature, (threshold, gain)) in enumerate(features_by_gain_orig):
                    if feature == faster_feature_orig:
                        print(f"  Faster implementation chose feature={faster_feature_orig}, "
                              f"threshold={faster_threshold:.6f}")
                        print(f"  This is the #{i+1} highest gain split in original space")
                        break
            
            # Return true if we found a tie
            return True
    
    # No ties found
    return False


def search_for_ties():
    """Search for datasets that exhibit ties in information gain"""
    print("Searching for datasets with ties in information gain...")
    
    # Try multiple seeds and tie thresholds
    tie_thresholds = [0.0001, 0.001, 0.01]
    seed_range = 100
    
    ties_found = 0
    
    for threshold in tie_thresholds:
        print(f"\nSearching with threshold {threshold*100:.4f}%")
        
        for seed in range(seed_range):
            if find_ties_in_dataset(seed, threshold):
                ties_found += 1
                
                # Stop after finding a few examples
                if ties_found >= 5:
                    print(f"\nFound {ties_found} examples with ties. Stopping search.")
                    return


if __name__ == "__main__":
    search_for_ties()