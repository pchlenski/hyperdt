import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sys

sys.path.append("..")
from src.hyperdt.conversions import convert


def _get(seed, class_label, num_samples=None):
    data = np.load("/home/phil/hdt/data/raw/wordnet/embeddings.npy")
    labels = np.load(f"/home/phil/hdt/data/raw/wordnet/multiclass_wordnet/{class_label}_labels.npy")
    np.random.seed(seed)
    idx = np.random.choice(data.shape[0], num_samples, replace=False)
    return train_test_split(data[idx], labels[idx], test_size=0.2, random_state=seed)


def get_training_data(class_label, seed, num_samples=None, convert_to_poincare=True):
    data, _, labels, _ = _get(seed, class_label=class_label, num_samples=num_samples)
    if not convert_to_poincare:
        data = convert(data, initial="poincare", final="hyperboloid")
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_testing_data(class_label, seed, num_samples=None, convert_to_poincare=True):
    data, _, labels, _ = _get(seed, class_label=class_label, num_samples=num_samples)
    if not convert_to_poincare:
        data = convert(data, initial="poincare", final="hyperboloid")
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_space():
    return "hyperbolic"
