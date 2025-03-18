import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sys

sys.path.append("..")
from hyperdt.legacy.conversions import convert


def _get(seed, dimension):
    path = f"/home/phil/hdt/data/processed/hypll_embeddings/polblogs_{dimension}_{seed}.tsv"
    data = np.loadtxt(path, delimiter="\t")
    labels = np.loadtxt("/home/phil/hdt/data/processed/polblogs_labels_full.tsv", delimiter="\t", usecols=1)
    return train_test_split(data, labels, test_size=0.2, random_state=seed)


def get_training_data(class_label, seed, num_samples=None, convert_to_poincare=True):
    data, _, labels, _ = _get(seed, dimension=class_label)
    if not convert_to_poincare:
        data = convert(data, initial="poincare", final="hyperboloid")
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_testing_data(class_label, seed, num_samples=None, convert_to_poincare=True):
    data, _, labels, _ = _get(seed, dimension=class_label)
    if not convert_to_poincare:
        data = convert(data, "hyperboloid", "poincare")
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_space():
    return "hyperbolic"
