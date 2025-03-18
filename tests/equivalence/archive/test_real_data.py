"""
Test agreement on real-world hyperbolic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from horoRF_dataloaders.polblogs_hypll import load_polblogs
    has_polblogs = True
except ImportError:
    has_polblogs = False
    print("WARNING: polblogs dataset not available")

try:
    from horoRF_dataloaders.wordnet import load_wordnet
    has_wordnet = True
except ImportError:
    has_wordnet = False
    print("WARNING: wordnet dataset not available")

from hyperdt.tree import HyperbolicDecisionTreeClassifier as OriginalHDTC
from hyperdt.faster_tree import HyperbolicDecisionTreeClassifier as FasterHDTC
from hyperdt.toy_data import wrapped_normal_mixture


def test_real_world_data():
    """Test agreement on real-world hyperbolic datasets"""
    print("\n=== Testing Real-World Hyperbolic Data ===")
    
    datasets = []
    
    # Add available datasets
    if has_polblogs:
        datasets.append(("PolBlogs", load_polblogs))
    
    if has_wordnet:
        datasets.append(("WordNet", load_wordnet))
    
    if not datasets:
        print("No real-world datasets available. Generating synthetic data instead.")
        X, y = wrapped_normal_mixture(1000, num_classes=3, num_dims=3, noise_std=1.0)
        datasets.append(("Synthetic", lambda: (X, y)))
    
    # Test parameters
    depths = [3, 5, 10, None]  # None = unlimited
    
    for dataset_name, load_fn in datasets:
        print(f"\nTesting on {dataset_name} dataset:")
        
        # Load data
        try:
            X, y = load_fn()
            print(f"  Dataset shape: {X.shape}, {len(np.unique(y))} classes")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Test across different depths
            for depth in depths:
                depth_str = str(depth) if depth is not None else "unlimited"
                print(f"\n  Testing with max_depth={depth_str}:")
                
                # Create models
                orig_tree = OriginalHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
                faster_tree = FasterHDTC(max_depth=depth, timelike_dim=0, skip_hyperboloid_check=True)
                
                # Fit and time original model
                start_time = time.time()
                orig_tree.fit(X_train, y_train)
                orig_time = time.time() - start_time
                
                # Fit and time faster model
                start_time = time.time()
                faster_tree.fit(X_train, y_train)
                faster_time = time.time() - start_time
                
                # Get predictions
                orig_preds_train = orig_tree.predict(X_train)
                faster_preds_train = faster_tree.predict(X_train)
                
                orig_preds_test = orig_tree.predict(X_test)
                faster_preds_test = faster_tree.predict(X_test)
                
                # Calculate accuracy
                orig_train_acc = accuracy_score(y_train, orig_preds_train)
                faster_train_acc = accuracy_score(y_train, faster_preds_train)
                
                orig_test_acc = accuracy_score(y_test, orig_preds_test)
                faster_test_acc = accuracy_score(y_test, faster_preds_test)
                
                # Calculate agreement
                train_agreement = np.mean(orig_preds_train == faster_preds_train)
                test_agreement = np.mean(orig_preds_test == faster_preds_test)
                
                print(f"    Training time - Original: {orig_time:.4f}s, Faster: {faster_time:.4f}s, "
                      f"Speedup: {orig_time/faster_time:.2f}x")
                print(f"    Train accuracy - Original: {orig_train_acc:.4f}, Faster: {faster_train_acc:.4f}, "
                      f"Difference: {abs(orig_train_acc - faster_train_acc):.4f}")
                print(f"    Test accuracy - Original: {orig_test_acc:.4f}, Faster: {faster_test_acc:.4f}, "
                      f"Difference: {abs(orig_test_acc - faster_test_acc):.4f}")
                print(f"    Prediction agreement - Train: {train_agreement:.4f}, Test: {test_agreement:.4f}")
                
                # If there are disagreements, find some examples
                if train_agreement < 1.0:
                    disagreement_indices = np.where(orig_preds_train != faster_preds_train)[0]
                    
                    print(f"    Found {len(disagreement_indices)} train disagreements. First examples:")
                    for i, idx in enumerate(disagreement_indices[:min(3, len(disagreement_indices))]):
                        print(f"      - Point {idx}: Original pred={orig_preds_train[idx]}, "
                              f"Faster pred={faster_preds_train[idx]}")
                        
        except Exception as e:
            print(f"  Error testing {dataset_name}: {str(e)}")


def visualize_decision_boundaries():
    """Visualize decision boundaries for 2D synthetic data"""
    print("\n=== Visualizing Decision Boundaries ===")
    
    # Generate 2D hyperbolic data (3D points on hyperboloid)
    X, y = wrapped_normal_mixture(500, num_classes=3, num_dims=2, noise_std=1.0, seed=42)
    
    # Create and fit models with multiple depths
    for depth in [1, 2, 3, 5]:
        print(f"\nVisualizing trees with max_depth={depth}")
        
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
        
        # Create comparison visualization
        plt.figure(figsize=(15, 5))
        
        # Original tree
        plt.subplot(1, 3, 1)
        plt.contourf(xx, yy, orig_preds, alpha=0.3, cmap=plt.cm.Paired)
        for class_value in np.unique(y):
            plt.scatter(X[y == class_value, 1], X[y == class_value, 2], 
                        alpha=0.8, label=f"Class {class_value}")
        plt.title(f"Original Tree (depth={depth})")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend()
        
        # Faster tree
        plt.subplot(1, 3, 2)
        plt.contourf(xx, yy, faster_preds, alpha=0.3, cmap=plt.cm.Paired)
        for class_value in np.unique(y):
            plt.scatter(X[y == class_value, 1], X[y == class_value, 2], 
                        alpha=0.8, label=f"Class {class_value}")
        plt.title(f"Faster Tree (depth={depth})")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend()
        
        # Difference
        plt.subplot(1, 3, 3)
        diff = (orig_preds != faster_preds).astype(int)
        plt.imshow(diff, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='Reds', alpha=0.5)
        for class_value in np.unique(y):
            plt.scatter(X[y == class_value, 1], X[y == class_value, 2], 
                        alpha=0.3, label=f"Class {class_value}")
        plt.title(f"Differences (white=same, red=different)")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        
        # Print agreement stats
        agreement = 1 - np.mean(diff)
        print(f"  Decision boundary agreement: {agreement:.6f}")
        
        plt.tight_layout()
        plt.savefig(f'/home/phil/hyperdt/faster_tests/decision_boundary_depth{depth}.png')
        
        # Compare predictions on the data
        orig_data_preds = orig_tree.predict(X)
        faster_data_preds = faster_tree.predict(X)
        data_agreement = np.mean(orig_data_preds == faster_data_preds)
        print(f"  Data point prediction agreement: {data_agreement:.6f}")
        
        # If there are disagreements, find some examples
        if data_agreement < 1.0:
            disagreement_indices = np.where(orig_data_preds != faster_data_preds)[0]
            print(f"  Found {len(disagreement_indices)} data disagreements.")
            
            # Visualize a few disagreement points specifically
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, orig_preds, alpha=0.1, cmap=plt.cm.Paired)
            plt.contour(xx, yy, faster_preds, colors='red', alpha=0.5, linewidths=0.5)
            
            # Plot all points with low alpha
            for class_value in np.unique(y):
                plt.scatter(X[y == class_value, 1], X[y == class_value, 2], 
                            alpha=0.1, label=f"Class {class_value}")
            
            # Highlight disagreement points
            for i, idx in enumerate(disagreement_indices[:min(10, len(disagreement_indices))]):
                plt.scatter(X[idx, 1], X[idx, 2], s=100, edgecolor='red', facecolor='none', 
                            linewidth=2, label=f"Disagreement" if i==0 else "")
                
            plt.title(f"Points with Different Predictions (depth={depth})")
            plt.xlabel("x₁")
            plt.ylabel("x₂")
            plt.legend()
            plt.savefig(f'/home/phil/hyperdt/faster_tests/disagreement_points_depth{depth}.png')


if __name__ == "__main__":
    test_real_world_data()
    visualize_decision_boundaries()