from abc import abstractmethod

import numpy as np
import torch

from source.data_tools.preprocessing import prepare_cv_datasets, get_train_val_test_split
from source.optimizers.torch_optimizers import get_optimal_hypernetwork
from pymoo.util.ref_dirs import get_reference_directions

from tqdm import tqdm

from source.optimizers.torch_optimizers import get_optimal_model  # DELETE THIS


def get_test_rays(dim=2, n_partitions=100):
    """
    Generate rays uniformly from dim-simplex
    From https://github.com/AvivNavon/pareto-hypernetworks
    """
    test_rays = get_reference_directions("das-dennis", dim, n_partitions=n_partitions).astype(
        np.float32
    )
    return torch.from_numpy(test_rays)


PARTITIONS_BY_DIM = {
    2: 100,
    3: 20,
    4: 12,
    5: 9
}


def get_pareto_front_losses(x, y, groups, loss_fncs, exclude_last_loss=True, **kwargs):
    """
    Get (approximation) of a pareto front for loss functions
    :param x: features
    :param y: labels
    :param groups: binary group labels
    :param loss_fncs: a list of loss functions
    :param exclude_last_loss: if True, last loss (typically regulizer) is not included in pareto front
    :param kwargs: additional args, such as eps constant for robustness, etc.
    :return: loss values on the combinations of loss weights sampled from n-simplex
    """
    preprocessed_data = get_train_val_test_split(x, y, groups)

    X_test, y_test, groups_test = preprocessed_data["test"]
    optimal_hypernet = get_optimal_hypernetwork(preprocessed_data, loss_fncs, exclude_last_loss, **kwargs)

    val_loss_funcs = loss_fncs[:-1] if exclude_last_loss else loss_fncs
    n_partitions = PARTITIONS_BY_DIM[len(val_loss_funcs)]
    test_rays = get_test_rays(len(val_loss_funcs), n_partitions)
    losses = [[] for _ in range(len(val_loss_funcs))]
    for ray in tqdm(test_rays):
        theta = optimal_hypernet(ray)
        y_pred = X_test @ theta
        params = {
            "true_labels": y_test,
            "pred_logits": y_pred,
            "groups": groups_test,
            "model": lambda x: x @ theta,
            "data": X_test,
            "theta": theta
        }
        params.update(kwargs)
        for i, l in enumerate(val_loss_funcs):
            losses[i].append(l(params))
    return losses
