import anndata
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def _get(seed, dimension, num_samples, convert_to_poincare=True, **kwargs):
    # Keep only abundant taxa
    adata = anndata.read_h5ad("data/raw/americangut.h5ad")
    labels = adata.var["taxonomy_1"]
    labels_counts = labels.value_counts()
    keep = labels_counts[labels_counts > 1000].index
    labels_filtered = labels[labels.isin(keep)]

    # Set seed and get indices randomly
    np.random.seed(seed)

    # Sufficient to just draw indices from filtered labels
    indices = np.random.choice(labels_filtered.index, num_samples, replace=False)
    if convert_to_poincare:
        embed_name = f"component_embeddings_poincare_{dimension}"
    else:
        embed_name = f"component_embeddings_hyperboloid_{dimension}"
    data = adata.varm[embed_name].loc[indices]
    labels = adata.var["taxonomy_1"].astype("category").cat.codes.loc[indices]

    # Train-test split at the end
    return train_test_split(data, labels, test_size=0.2, random_state=seed)


def get_training_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    data, _, labels, _ = _get(
        seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
    )
    return torch.as_tensor(data.values), torch.as_tensor(labels, dtype=int).flatten()


def get_testing_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    _, data, _, labels = _get(
        seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
    )
    return torch.as_tensor(data.values), torch.as_tensor(labels, dtype=int).flatten()


def get_space():
    return "hyperbolic"
