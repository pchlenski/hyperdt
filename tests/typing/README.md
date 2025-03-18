# Type Annotations in HyperDT

This directory contains tests for type annotations in the HyperDT library. The type annotations are checked using `mypy`.

## Array Shape Conventions

For numpy arrays, we use the following type conventions:

- `ArrayLike`: Any array-like input that can be converted to a numpy array (from numpy.typing)
- `NDArray`: A numpy array with any shape and dtype (from numpy.typing)
- `NDArray[np.float64]`: A numpy array with float64 dtype
- `NDArray[np.int_]`: A numpy array with int dtype

For specific shapes, we use the following conventions:

- `NDArray[Shape["n_samples, *"]]`: Array with first dimension representing samples count
- `NDArray[Shape["n_samples, n_features"]]`: 2D array with samples as rows and features as columns
- `NDArray[Shape["n_samples"]]`: 1D array with one entry per sample (e.g., labels)
- `NDArray[Shape["n_samples, n_classes"]]`: 2D array with probabilities for each class (classifier output)
- `NDArray[Shape["n_features"]]`: 1D array representing a single feature vector
- `NDArray[Shape["n_trees"]]`: 1D array representing an ensemble of trees

For arrays representing hyperbolic space points:
- `NDArray[Shape["n_samples, n_dimensions"]]`: Points in hyperboloid space
- `NDArray[Shape["n_samples, n_dimensions-1"]]`: Points in Klein model (no timelike dimension)

## Running Type Checks

To run the type checks, use:

```bash
mypy hyperdt/
```

Type annotations are compatible with scikit-learn and numpy conventions.