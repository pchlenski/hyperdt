# hyperDT
> Fast hyperboloid decision tree algorithms

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

If you are installing with e.g. a conda environment or virtualenv, you can find exact dependencies in `requirements.txt`.
These are installable in the usual way:
```bash
pip install -r requirements.txt
```

### Pip install
Additionally, hyperDT is available on PyPI. It can be pip installed as follows:

```bash
pip install hyperdt
```

## Tutorial
A basic tutorial demonstrating key HyperDT functionality is available in `notebooks/tutorial.ipynb`.

## Reproducibility
All figures and tables in the paper were generated using a combination of Python scripts and Jupyter notebooks. The notebooks used in development were filtered down to only those that remained relevant to the final paper and moved to the `notebooks/archive` directory. The `notebooks` directory contains a tutorial and symbolic links to notebooks of particular relevance to a figure, table, or section of a paper, named according to the section they reproduce.

`hororf_benchmarks.py` runs the benchmarks contributing to Tables 1, 5, and 6, and `scaling_benchmarks.py` runs the benchmarks contributing to Figures 6 and 7.

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
