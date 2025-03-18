import numpy as np
import torch

from geomstats.geometry.hyperbolic import Hyperbolic
from sklearn.model_selection import train_test_split

import sys

sys.path.append("..")
from hyperdt.toy_data import wrapped_normal_mixture
from hyperdt.legacy.conversions import convert

# Need for bad_points:
import geomstats.backend as gs
import geomstats.algebra_utils as utils
import math


def bad_points(points, base_points, manifold):
    """Avoid the 'Minkowski norm of 0' error by using this"""
    sq_norm_tangent_vec = manifold.embedding_space.metric.squared_norm(points)
    sq_norm_tangent_vec = gs.clip(sq_norm_tangent_vec, 0, math.inf)

    coef_1 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.cosh_close_0, order=5)
    coef_2 = utils.taylor_exp_even_func(sq_norm_tangent_vec, utils.sinch_close_0, order=5)

    exp = gs.einsum("...,...j->...j", coef_1, base_points) + gs.einsum("...,...j->...j", coef_2, points)
    return manifold.metric.squared_norm(exp) == 0


def _get(seed, dimension, num_samples, convert_to_poincare=True, **kwargs):
    data, labels = wrapped_normal_mixture(
        num_dims=dimension, num_classes=2, num_points=num_samples, seed=seed, noise_std=2, **kwargs
    )
    if convert_to_poincare:
        data = convert(data, "poincare", "hyperboloid")
    return train_test_split(data, labels, test_size=0.2, random_state=seed)


def get_training_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    data, _, labels, _ = _get(
        seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
    )
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_testing_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    _, data, _, labels = _get(
        seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
    )
    return torch.as_tensor(data), torch.as_tensor(labels, dtype=int).flatten()


def get_space():
    return "hyperbolic"
