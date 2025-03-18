# hyperDT
> Fast hyperboloid decision tree algorithms

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/pchlenski/hyperdt)](https://github.com/pchlenski/hyperdt/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/hyperdt.svg)](https://badge.fury.io/py/hyperdt)
[![Tests](https://github.com/pchlenski/hyperdt/actions/workflows/tests.yml/badge.svg)](https://github.com/pchlenski/hyperdt/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/pchlenski/hyperdt/branch/main/graph/badge.svg)](https://codecov.io/gh/pchlenski/hyperdt)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Mypy: Checked](https://img.shields.io/badge/mypy-checked-brightgreen.svg)](https://github.com/pchlenski/hyperdt/blob/main/tests/typing/README.md)
[![sklearn: Compatible](https://img.shields.io/badge/sklearn-compatible-brightgreen.svg)](https://github.com/pchlenski/hyperdt/blob/main/tests/sklearn_compatibility/README.md)
[![ICLR 2024](https://img.shields.io/badge/ICLR-2024-blueviolet.svg)](https://openreview.net/forum?id=TTonmgTT9X)

This repository contains code for the paper Fast Hyperboloid Decision Tree Algorithms (ICLR 2024), which you can find
at one of these links:
* [OpenReview](https://openreview.net/forum?id=TTonmgTT9X)
* [ArXiv](https://arxiv.org/abs/2310.13841)

## Installation:
### Local install
To install, run this from your repo directory:
```bash
git clone https://github.com/pchlenski/hyperdt
cd hyperdt
pip install -e .
```

All dependencies are managed through pyproject.toml. For development, you can install the package with all development dependencies:

```bash
pip install -e ".[dev]"
```

For testing, you can install the test dependencies:

```bash
pip install -e ".[test]"
```

### Pip install
Additionally, hyperDT is available on PyPI. It can be pip installed as follows:

```bash
pip install hyperdt
```

HyperDT supports optional dependencies:

```bash
# Install with XGBoost support
pip install hyperdt[xgboost]

# Install with legacy implementation support (requires geomstats)
pip install hyperdt[legacy]

# Install all optional dependencies
pip install hyperdt[all]
```

## Tutorial
A basic tutorial demonstrating key HyperDT functionality is available in `notebooks/tutorial.ipynb`.

## Package Structure

The hyperDT package is structured as follows:

- `hyperdt/_base.py`: Base classes for hyperbolic decision trees
- `hyperdt/tree.py`: HyperbolicDecisionTreeClassifier and HyperbolicDecisionTreeRegressor 
- `hyperdt/ensemble.py`: HyperbolicRandomForestClassifier and HyperbolicRandomForestRegressor
- `hyperdt/_xgboost.py`: XGBoost integration (requires `pip install hyperdt[xgboost]`)
- `hyperdt/legacy/`: Original implementation (requires `pip install hyperdt[legacy]`)
- `hyperdt/toy_data.py`: Utilities for generating synthetic hyperbolic datasets

The package has a modular design with optional dependencies:
- Core functionality only requires numpy, scikit-learn, scipy, and matplotlib
- XGBoost backend requires the xgboost package (`pip install hyperdt[xgboost]`)
- Legacy implementation requires geomstats (`pip install hyperdt[legacy]`)

## Reproducibility and data availability
All figures and tables in the paper were generated using a combination of Python scripts and Jupyter notebooks. The notebooks used in development were filtered down to only those that remained relevant to the final paper and moved to the `notebooks/archive` directory. The `notebooks` directory contains a tutorial and symbolic links to notebooks of particular relevance to a figure, table, or section of a paper, named according to the section they reproduce.

`hororf_benchmarks.py` runs the benchmarks contributing to Tables 1, 5, and 6, and `scaling_benchmarks.py` runs the benchmarks contributing to Figures 6 and 7.

All relevant datasets, plus benchmarking code outputs, can be found on [Google Drive](https://drive.google.com/drive/folders/11ORbG_5N1RM54ODzx2pk28CG2SbzPIRy?usp=sharing).

## Citation
To cite HyperDT, please use the following:

```bibtex
@inproceedings{
    chlenski2024fast,
    title={Fast Hyperboloid Decision Tree Algorithms},
    author={Philippe Chlenski and Ethan Turok and Antonio Khalil Moretti and Itsik Pe'er},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=TTonmgTT9X}
}
```
