# scikit-learn Compatibility Tests

This directory contains tests to verify that HyperDT estimators are compatible with scikit-learn's API.

## Running the tests

To run the compatibility tests:

```bash
python -m pytest tests/sklearn_compatibility
```

## Test Structure

- `test_check_estimator.py`: Verifies that HyperDT estimators pass scikit-learn's `check_estimator` tests
- `test_pipeline.py`: Verifies that HyperDT estimators work properly in scikit-learn pipelines