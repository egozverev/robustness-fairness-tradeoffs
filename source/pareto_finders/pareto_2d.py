# from tqdm import tqdm
from source.data_tools.preprocessing import prepare_cv_datasets
from source.optimizers.torch_optimizers import get_optimal_model
from source.metrics.metrics_tools import compute_metrics

from source.models.linear_models import TorchLinearModel

import numpy as np
from tqdm import tqdm

from torch.special import expit


def get_pareto_curve(grid_sz=10, grid=None, **kwargs):
    """
    Get (2d) pareto curve for two competing Losses.
    Serves as a wrapper for get_pareto_point when running tasks sequentially.
    :param grid: grid of weights for the first loss
    :param grid_sz: if not None, grid is generated uniformly over [0, 1]
    :param kwargs: arguments for get_pareto_point
    :return:
    """
    first_metrics = []
    second_metrics = []
    for point_id in tqdm(range(1, (len(grid) if grid else grid_sz) + 1)):
        point_res = get_pareto_point(point_id, grid_sz=grid_sz, grid=grid, **kwargs)
        first_metrics.append(point_res[0])
        second_metrics.append(point_res[1])
    return [first_metrics, second_metrics]


def get_pareto_point(task_id, loss_fncs, metrics_fncs, x, y, groups,
                     grid_sz=10, n_folds=5, grid=None,
                     **kwargs):
    """
    Get a point on a pareto curve. point_id indicates index on a weight grid.

    :param task_id: task (or point) id: indicates index of weight in the grid
    :param loss_fncs: list of loss functions
    :param metrics_fncs: list of metric functions
    :param x: features
    :param y: labels
    :param groups: binary group labels
    :param grid_sz: size of a grid
    :param n_folds: number of folds for cross-validation
    :param grid: grid, if provided
    :param kwargs: additional arguments
    :return: averaged metric values across folds
    """
    if grid is not None and len(grid):
        imp = grid[task_id - 1]
    else:
        imp = (task_id - 1) / (grid_sz - 1)  # assuming id \in {1, ... grid_sz}
    if (imp < 0) or (imp > 1):
        raise ValueError('A wrong value found in a grid')
    regulizer_strength = kwargs["regulizer_strength"] if "regulizer_strength" in kwargs else 0.1
    cv_first_metrics = []
    cv_second_metrics = []
    loss_coeffs = [None, None, regulizer_strength]
    loss_coeffs[0] = imp
    loss_coeffs[1] = 1 - imp
    for X_train, y_train, X_test, y_test, groups_train, groups_test in prepare_cv_datasets(x, y, groups, n_folds):
        optimal_model = get_optimal_model(X_train, y_train, loss_fncs,
                                          loss_coeffs, groups_train, **kwargs)
        pred_probs = expit(optimal_model(X_test))
        pred_logits = optimal_model(X_test)
        local_metrics = compute_metrics(metrics_fncs,
                                        {
                                            "pred_probs": pred_probs,
                                            "pred_logits": pred_logits,
                                            "true_labels": y_test,
                                            "groups": groups_test,
                                            "model": optimal_model,
                                            "data": X_test,
                                        } | kwargs)
        cv_first_metrics.append(local_metrics[0])
        cv_second_metrics.append(local_metrics[1])
    return [np.mean(cv_first_metrics), np.mean(cv_second_metrics)]
