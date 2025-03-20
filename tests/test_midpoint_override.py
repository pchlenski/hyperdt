"""
Test that overriding the einstein_midpoint method affects model performance.
This confirms that the threshold adjustments are actually taking effect.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hyperdt import HyperbolicDecisionTreeClassifier
from hyperdt.toy_data import wrapped_normal_mixture


def test_midpoint_override_affects_performance():
    """Test that overriding the midpoint calculation with zeros changes model performance."""
    # Generate test data
    X, y = wrapped_normal_mixture(num_points=200, num_classes=2, num_dims=3, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model with the normal einstein_midpoint method
    normal_model = HyperbolicDecisionTreeClassifier(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    normal_model.fit(X_train, y_train)
    normal_pred = normal_model.predict(X_test)
    normal_acc = accuracy_score(y_test, normal_pred)
    
    # Create a subclass with an overridden einstein_midpoint method that always returns 0
    class ZeroMidpointModel(HyperbolicDecisionTreeClassifier):
        def _einstein_midpoint(self, u, v):
            # Return 0 instead of the actual midpoint
            return 0.0
    
    # Train a model with the overridden method
    zero_model = ZeroMidpointModel(max_depth=5, timelike_dim=0, skip_hyperboloid_check=True)
    zero_model.fit(X_train, y_train)
    zero_pred = zero_model.predict(X_test)
    zero_acc = accuracy_score(y_test, zero_pred)
    
    # The predictions should be different, confirming the midpoint override is taking effect
    assert not np.array_equal(normal_pred, zero_pred), "Midpoint override did not affect predictions"
    print(f"Normal model accuracy: {normal_acc:.4f}")
    print(f"Zero midpoint model accuracy: {zero_acc:.4f}")
    print(f"Accuracy difference: {abs(normal_acc - zero_acc):.4f}")
    
    # The prediction accuracy should be different
    assert normal_acc != zero_acc, "Midpoint override did not affect model accuracy"


if __name__ == "__main__":
    test_midpoint_override_affects_performance()