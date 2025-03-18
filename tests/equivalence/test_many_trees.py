"""
Test a large number of shallow trees to investigate disagreements more thoroughly.
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


def extract_original_tree_splits(tree_node, depth=0, feature_list=None, threshold_list=None):
    """Extract all split features and thresholds from original tree"""
    if feature_list is None:
        feature_list = []
    if threshold_list is None:
        threshold_list = []
    
    if tree_node.value is not None:  # Leaf node
        return feature_list, threshold_list
    
    feature_list.append(tree_node.feature)
    threshold_list.append(tree_node.theta)
    
    # Recurse on left and right children
    if tree_node.left is not None:
        extract_original_tree_splits(tree_node.left, depth+1, feature_list, threshold_list)
    if tree_node.right is not None:
        extract_original_tree_splits(tree_node.right, depth+1, feature_list, threshold_list)
    
    return feature_list, threshold_list


def extract_faster_tree_splits(sklearn_tree):
    """Extract all split features and thresholds from sklearn tree"""
    tree = sklearn_tree.tree_
    
    features = []
    thresholds = []
    
    # Iterate through all nodes
    for i in range(tree.node_count):
        # Skip leaf nodes
        if tree.children_left[i] == TREE_LEAF:
            continue
        
        features.append(tree.feature[i])
        thresholds.append(tree.threshold[i])
    
    return features, thresholds


def compare_tree_splits(original_tree, faster_tree):
    """Compare the splits between original and faster tree implementations"""
    # Extract splits from both trees
    orig_features, orig_thetas = extract_original_tree_splits(original_tree.tree)
    fast_features, fast_thresholds = extract_faster_tree_splits(faster_tree.tree)
    
    # Convert original theta to cot(theta) for comparison
    orig_thresholds = [1/np.tan(theta) for theta in orig_thetas]
    
    # Check if the number of splits matches
    if len(orig_features) != len(fast_features):
        print(f"Different number of splits: Original={len(orig_features)}, Faster={len(fast_features)}")
        return False
    
    # Map original features to faster features (accounting for timelike dimension)
    orig_features_mapped = [f-1 if f > original_tree.timelike_dim else f for f in orig_features]
    
    # Compare each split
    matches = 0
    for i in range(len(orig_features)):
        # Find matching feature in faster tree
        matching_indices = [j for j, f in enumerate(fast_features) if f == orig_features_mapped[i]]
        
        if not matching_indices:
            print(f"No matching feature for original feature {orig_features[i]} (mapped to {orig_features_mapped[i]})")
            continue
        
        # Check if any threshold matches within tolerance
        found_match = False
        for j in matching_indices:
            if abs(orig_thresholds[i] - fast_thresholds[j]) < 1e-5:
                matches += 1
                found_match = True
                break
        
        if not found_match:
            print(f"No matching threshold for feature {orig_features[i]}: "
                  f"Original cot(θ)={orig_thresholds[i]:.8f}, Faster options={[fast_thresholds[j] for j in matching_indices]}")
    
    match_rate = matches / len(orig_features) if orig_features else 1.0
    return match_rate


def test_many_shallow_trees():
    """Test a large number of shallow trees with different random seeds"""
    print("\n=== Testing Many Shallow Trees ===")
    
    # Parameters
    n_trees = 100
    max_depth = 1  # Use single-split trees
    n_features = 5
    n_classes = 3
    n_samples = 100
    
    # Results tracking
    agreement_rates = []
    split_match_rates = []
    
    # Start with a reproducible set of disagreements
    np.random.seed(42)
    test_seeds = np.random.randint(1, 10000, n_trees)
    
    print(f"Testing {n_trees} trees with max_depth={max_depth}, {n_features} features, {n_classes} classes")
    
    disagreements_found = 0
    split_mismatches = 0
    
    for i, seed in enumerate(test_seeds):
        # Generate data
        X, y = generate_hyperbolic_data(n_samples, n_classes, n_features, random_state=seed)
        
        # Create models
        orig_tree = OriginalHDTC(max_depth=max_depth, timelike_dim=0, skip_hyperboloid_check=True)
        faster_tree = FasterHDTC(max_depth=max_depth, timelike_dim=0, skip_hyperboloid_check=True)
        
        # Fit models
        orig_tree.fit(X, y)
        faster_tree.fit(X, y)
        
        # Compare predictions
        orig_preds = orig_tree.predict(X)
        faster_preds = faster_tree.predict(X)
        
        agreement_rate = np.mean(orig_preds == faster_preds)
        agreement_rates.append(agreement_rate)
        
        # Compare tree structure (splits)
        split_match_rate = compare_tree_splits(orig_tree, faster_tree)
        split_match_rates.append(split_match_rate)
        
        # Track disagreements
        if agreement_rate < 1.0:
            disagreements_found += 1
            
            # Only print detailed analysis for the first few disagreements
            if disagreements_found <= 3:
                print(f"\nFound disagreement with seed {seed}:")
                print(f"  Prediction agreement rate: {agreement_rate:.4f}")
                print(f"  Split match rate: {split_match_rate:.4f}")
                
                # Original tree info
                orig_feature = orig_tree.tree.feature
                orig_theta = orig_tree.tree.theta
                orig_cot = 1/np.tan(orig_theta)
                print(f"  Original tree split: feature={orig_feature}, θ={orig_theta:.6f}, cot(θ)={orig_cot:.6f}")
                
                # Faster tree info
                fast_feature = faster_tree.tree.tree_.feature[0]
                fast_threshold = faster_tree.tree.tree_.threshold[0]
                print(f"  Faster tree split: feature={fast_feature}, threshold={fast_threshold:.6f}")
                
                # Find a sample point where predictions disagree
                disagreement_idx = np.where(orig_preds != faster_preds)[0][0]
                point = X[disagreement_idx]
                print(f"  Disagreement point: {point}")
                
                # Calculate decision values
                orig_dot = orig_tree._dot(point.reshape(1, -1), orig_feature, orig_theta).item()
                orig_decision = orig_dot < 0
                
                # Get Klein coordinate
                X_klein = np.delete(point, 0) / point[0]
                fast_decision = X_klein[fast_feature] < fast_threshold
                
                print(f"  Original dot product: {orig_dot:.10f}, decision: {'Left' if orig_decision else 'Right'}")
                print(f"  Klein ratio: {X_klein[fast_feature]:.10f}, threshold: {fast_threshold:.10f}, "
                      f"decision: {'Left' if fast_decision else 'Right'}")
        
        if split_match_rate < 1.0:
            split_mismatches += 1
            
        # Progress indicator
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{n_trees} trees. "
                  f"Disagreements: {disagreements_found}, Split mismatches: {split_mismatches}")
    
    # Summary statistics
    mean_agreement = np.mean(agreement_rates)
    mean_split_match = np.mean(split_match_rates)
    
    print(f"\nResults summary:")
    print(f"  Mean prediction agreement rate: {mean_agreement:.6f}")
    print(f"  Mean split match rate: {mean_split_match:.6f}")
    print(f"  Trees with prediction disagreements: {disagreements_found}/{n_trees}")
    print(f"  Trees with split mismatches: {split_mismatches}/{n_trees}")
    
    # Histogram of agreement rates
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(agreement_rates, bins=20, alpha=0.7)
    plt.axvline(mean_agreement, color='r', linestyle='--', label=f'Mean: {mean_agreement:.4f}')
    plt.xlabel('Prediction Agreement Rate')
    plt.ylabel('Number of Trees')
    plt.title('Prediction Agreement Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(split_match_rates, bins=20, alpha=0.7)
    plt.axvline(mean_split_match, color='r', linestyle='--', label=f'Mean: {mean_split_match:.4f}')
    plt.xlabel('Split Match Rate')
    plt.ylabel('Number of Trees')
    plt.title('Split Match Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/phil/hyperdt/faster_tests/shallow_trees_agreement.png')


def test_deeper_trees():
    """Test trees of increasing depth to see where disagreements start to appear"""
    print("\n=== Testing Trees of Increasing Depth ===")
    
    depths = [1, 2, 3, 5, 7, 10]
    n_trees_per_depth = 20
    n_features = 5
    n_classes = 3
    n_samples = 200
    
    # Results tracking by depth
    depth_results = {
        'depths': depths,
        'agreement_means': [],
        'split_match_means': [],
        'disagreement_counts': [],
        'mismatch_counts': []
    }
    
    # Start with a reproducible set of disagreements
    np.random.seed(42)
    test_seeds = np.random.randint(1, 10000, n_trees_per_depth)
    
    for depth in depths:
        print(f"\nTesting with max_depth={depth}:")
        
        agreement_rates = []
        split_match_rates = []
        disagreements_found = 0
        split_mismatches = 0
        
        for i, seed in enumerate(test_seeds):
            # Generate data
            X, y = generate_hyperbolic_data(n_samples, n_classes, n_features, random_state=seed)
            
            # Create models
            orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
            faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
            
            # Fit models
            orig_tree.fit(X, y)
            faster_tree.fit(X, y)
            
            # Compare predictions
            orig_preds = orig_tree.predict(X)
            faster_preds = faster_tree.predict(X)
            
            agreement_rate = np.mean(orig_preds == faster_preds)
            agreement_rates.append(agreement_rate)
            
            # Compare tree structure (splits)
            split_match_rate = compare_tree_splits(orig_tree, faster_tree)
            split_match_rates.append(split_match_rate)
            
            # Track disagreements
            if agreement_rate < 1.0:
                disagreements_found += 1
            
            if split_match_rate < 1.0:
                split_mismatches += 1
        
        # Collect results for this depth
        mean_agreement = np.mean(agreement_rates)
        mean_split_match = np.mean(split_match_rates)
        
        depth_results['agreement_means'].append(mean_agreement)
        depth_results['split_match_means'].append(mean_split_match)
        depth_results['disagreement_counts'].append(disagreements_found)
        depth_results['mismatch_counts'].append(split_mismatches)
        
        print(f"  Mean prediction agreement rate: {mean_agreement:.6f}")
        print(f"  Mean split match rate: {mean_split_match:.6f}")
        print(f"  Trees with prediction disagreements: {disagreements_found}/{n_trees_per_depth}")
        print(f"  Trees with split mismatches: {split_mismatches}/{n_trees_per_depth}")
    
    # Plot results by depth
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(depth_results['depths'], depth_results['agreement_means'], 'o-')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Mean Prediction Agreement Rate')
    plt.title('Prediction Agreement vs Tree Depth')
    plt.ylim(0.95, 1.001)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(depth_results['depths'], depth_results['split_match_means'], 'o-')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Mean Split Match Rate')
    plt.title('Split Match Rate vs Tree Depth')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(depth_results['depths'])), depth_results['disagreement_counts'])
    plt.xticks(range(len(depth_results['depths'])), depth_results['depths'])
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Number of Trees with Disagreements')
    plt.title('Prediction Disagreements vs Tree Depth')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.bar(range(len(depth_results['depths'])), depth_results['mismatch_counts'])
    plt.xticks(range(len(depth_results['depths'])), depth_results['depths'])
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Number of Trees with Split Mismatches')
    plt.title('Split Mismatches vs Tree Depth')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/phil/hyperdt/faster_tests/depth_comparison.png')


def analyze_split_selection():
    """Investigate why the two implementations might choose different splits"""
    print("\n=== Analyzing Split Selection Differences ===")
    
    # Generate a simple dataset
    seed = 42
    n_samples = 50
    n_features = 3
    n_classes = 2
    
    X, y = generate_hyperbolic_data(n_samples, n_classes, n_features, random_state=seed)
    
    # Create models
    orig_tree = OriginalHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    faster_tree = FasterHDTC(max_depth=1, timelike_dim=0, skip_hyperboloid_check=True)
    
    # Analyze the candidate splits from original implementation
    orig_tree.ndim = X.shape[1]
    orig_tree.dims = [i for i in range(X.shape[1]) if i != orig_tree.timelike_dim]
    orig_tree.classes_ = np.unique(y)
    
    # Empty results for each feature
    feature_candidates = {}
    feature_scores = {}
    
    # Get candidate thresholds for each feature and evaluate them
    for dim in orig_tree.dims:
        # Get candidates from original implementation
        candidates = orig_tree._get_candidates(X, dim)
        
        # Evaluate each candidate
        scores = []
        for theta in candidates:
            left, right = orig_tree._get_split(X, dim, theta)
            
            # Check if split is valid
            min_len = min(np.sum(left), np.sum(right))
            if min_len >= orig_tree.min_samples_leaf:
                score = orig_tree._information_gain(left, right, y)
                scores.append((theta, score))
        
        if scores:
            feature_candidates[dim] = [item[0] for item in scores]
            feature_scores[dim] = [item[1] for item in scores]
    
    # Determine best split from original approach
    best_feature, best_score, best_threshold = -1, -1, -1
    for dim, scores in feature_scores.items():
        if scores and max(scores) > best_score:
            best_feature = dim
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_threshold = feature_candidates[dim][best_idx]
    
    print(f"Original best split: feature={best_feature}, θ={best_threshold:.6f}, "
          f"cot(θ)={1/np.tan(best_threshold):.6f}, score={best_score:.6f}")
    
    # Now fit and get the actual splits
    orig_tree.fit(X, y)
    faster_tree.fit(X, y)
    
    # Get chosen splits
    orig_feature = orig_tree.tree.feature
    orig_theta = orig_tree.tree.theta
    
    fast_feature = faster_tree.tree.tree_.feature[0]
    fast_threshold = faster_tree.tree.tree_.threshold[0]
    
    # Convert for comparison
    orig_cot = 1/np.tan(orig_theta)
    
    print(f"Original tree split: feature={orig_feature}, θ={orig_theta:.6f}, cot(θ)={orig_cot:.6f}")
    print(f"Faster tree split: feature={fast_feature}, threshold={fast_threshold:.6f}")
    
    # Compare predictions
    orig_preds = orig_tree.predict(X)
    faster_preds = faster_tree.predict(X)
    
    agreement_rate = np.mean(orig_preds == faster_preds)
    print(f"Prediction agreement rate: {agreement_rate:.6f}")
    
    # Analyze the Klein coordinates and best split selection
    print("\nAnalyzing in Klein coordinates:")
    
    # Convert to Klein
    X_klein = np.delete(X, 0, axis=1) / X[:, 0][:, None]
    
    for dim in range(X_klein.shape[1]):
        unique_vals = np.sort(np.unique(X_klein[:, dim]))
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        
        print(f"Feature {dim} has {len(thresholds)} candidate thresholds in Klein coordinates")
        
        if dim == fast_feature:
            print(f"  Selected threshold: {fast_threshold:.6f}")
            print(f"  All thresholds: {thresholds}")
            
            # Find closest match to original tree's threshold
            if orig_feature - 1 == dim:  # -1 because of timelike removal
                closest_idx = np.argmin(np.abs(thresholds - orig_cot))
                print(f"  Original tree threshold converted: {orig_cot:.6f}")
                print(f"  Closest Klein threshold: {thresholds[closest_idx]:.6f}, "
                      f"difference: {abs(thresholds[closest_idx] - orig_cot):.10f}")


if __name__ == "__main__":
    test_many_shallow_trees()
    test_deeper_trees()
    analyze_split_selection()