# HyperDT Model Tests

This directory contains tests for the refactored tree-based models in the HyperDT library.

## Overview

The tests verify that all the different model variants can properly fit and predict data:

- **HyperbolicDecisionTreeClassifier**: Basic decision tree for classification in hyperbolic space
- **HyperbolicDecisionTreeRegressor**: Decision tree for regression in hyperbolic space
- **HyperbolicRandomForestClassifier**: Random forest ensemble for classification in hyperbolic space
- **HyperbolicRandomForestRegressor**: Random forest ensemble for regression in hyperbolic space
- **HyperbolicXGBoostClassifier**: XGBoost for classification in hyperbolic space (requires XGBoost)
- **HyperbolicXGBoostRegressor**: XGBoost for regression in hyperbolic space (requires XGBoost)

## Running the Tests

To run the tests:

```bash
cd /path/to/hyperdt
python tests/models/test_model_types.py
```

## Test Structure

The tests follow a consistent pattern:

1. Generate synthetic data on the hyperboloid
2. Split into training and testing sets
3. Create and fit a model
4. Make predictions
5. Verify the predictions have the correct shape
6. Print basic performance metrics for information

## Notes

- These tests focus on API compatibility and basic functionality
- The synthetic data is not optimized for high accuracy, so performance metrics are just for reference
- XGBoost tests are skipped if the XGBoost library is not installed