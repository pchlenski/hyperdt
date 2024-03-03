# hyperDT
> Fast hyperboloid decision tree algorithms

This repository contains code for the paper Fast Hyperboloid Decision Tree algorithms (ICLR 2024), which you can find
at one of these links:
(OpenReview)[https://openreview.net/forum?id=TTonmgTT9X]
(ArXiv)[https://arxiv.org/abs/2310.13841]

## Installation:
### Local install
To install, run this from your repo directory:
```bash
git clone https://github.com/pchlenski/hyperdt
cd hyperdt
pip install -e .`
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
A basic tutorial demonstrating key HyperDT functionality is available in `notebooks/tutrial.ipynb`.

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