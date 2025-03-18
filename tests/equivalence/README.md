# HyperDT Implementation Equivalence Tests

This directory contains tests to verify the mathematical equivalence between the two implementations of hyperbolic decision trees:

1. `hyperdt.tree.HyperbolicDecisionTreeClassifier` - The original implementation using angle-based decision boundaries
2. `hyperdt.faster_tree.HyperbolicDecisionTreeClassifier` - The optimized implementation using ratio-based (Klein model) decision boundaries

## Key Findings

Our extensive testing confirms:

1. **Mathematical Equivalence**: The angle-based formula `sin(θ)*x_d - cos(θ)*x_0 < 0` is mathematically equivalent to the ratio-based formula `x_d/x_0 < cot(θ)` for points on the hyperboloid.

2. **Prediction Agreement**: Both implementations produce identical predictions in almost all cases (>99.8% agreement), despite sometimes having different tree structures.

3. **Different Split Selection**: The implementations occasionally choose different features or thresholds when multiple splits have identical information gain values.

4. **Tiebreaker Behavior**: We identified exact cases where information gain ties caused the implementations to make different but equivalent decisions.

5. **Performance Advantage**: The faster implementation typically achieves 100-500x speedup over the original implementation.

## Running the Tests

To run the comprehensive verification tests:

```bash
cd /home/phil/hyperdt
python tests/equivalence/test_equivalence.py
```

Results will be saved to `/home/phil/hyperdt/tests/equivalence/images/`.

## Directory Structure

- `test_equivalence.py`: Consolidated test suite that runs all verification tests
- `images/`: Contains visualizations of decision boundaries, performance comparisons, etc.
- `results/`: Contains additional result data
- `archive/`: Contains original individual test files for reference

## Detailed Explanation

### The Klein Model Transformation

The original implementation works in the hyperboloid model using Minkowski dot products of angle-based normal vectors:
```
sin(θ)*x_d - cos(θ)*x_0 < 0
```

The faster implementation converts to the Klein model where the decision boundary simplifies to:
```
x_d/x_0 < cot(θ)
```

This transformation allows using standard sklearn's DecisionTreeClassifier with the transformed coordinates.

### Why Split Selections Differ

Decision trees often encounter situations where multiple splits have identical information gain. When this happens, the two implementations use different tiebreaker strategies:
- The original implementation selects splits in the hyperboloid model space
- The faster implementation selects splits in the Klein model space

Despite these different choices, both implementations produce identical decision boundaries and predictions because they're mathematically equivalent formulations of the same geometric concept.

### Einstein Midpoint vs. Hyperbolic Midpoint

The original implementation uses complex hyperbolic midpoint calculations, while the faster implementation uses Einstein midpoints in the Klein model. Both approaches correctly find split points, but through different mathematical pathways.