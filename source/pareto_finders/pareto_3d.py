# from tqdm import tqdm
from source.data_tools.preprocessing import prepare_cv_datasets
from source.optimizers.torch_optimizers import get_optimal_model
from source.metrics.metrics_tools import compute_metrics

import numpy as np
from tqdm import tqdm

from torch.special import expit


###
### NOTE: this is unrefactored dirty code for 3D.
### Check pareto_2d.py to see how to do this better
###

def get_pareto_3d_point(task_id, loss_fncs, metrics_fncs, x, y, groups,
                     loss_default_coeffs=(None, None, None, 0.1),
                     n_folds=5, grid=None,
                     robust_loss_eps=None):
    """
    grid: array of tuples (w_1, w_2, w_3)
    """
    w1, w2, w3 = grid[task_id - 1]
    assert (0 <= w1 <= 1) and (0 <= w2 <= 1) and (0 <= w3 <= 1) and (np.abs(w1 + w2 + w3 - 1) < 1e-8)
    cv_first_metrics = []
    cv_second_metrics = []
    cv_third_metrics = []
    loss_coeffs = list(loss_default_coeffs)
    loss_coeffs[:3] = [w1, w2, w3]
    for X_train, y_train, X_test, y_test, groups_train, groups_test in prepare_cv_datasets(x, y, groups, n_folds):
        optimal_model = get_optimal_model(X_train, y_train, loss_fncs,
                                           loss_coeffs, groups_train, robust_loss_eps)
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
                                            "robust_loss_eps": robust_loss_eps
                                        })
        cv_first_metrics.append(local_metrics[0])
        cv_second_metrics.append(local_metrics[1])
        cv_third_metrics.append(local_metrics[2])

    return [np.mean(cv_first_metrics), np.mean(cv_second_metrics), np.mean(cv_third_metrics)]


def get_pareto_plane(loss_fncs, metrics_fncs, x, y, groups,
                     loss_default_coeffs=(None, None, None, 0.1),
                     grid_sz=10, n_folds=5, grid=None,
                     robust_loss_eps=None):
    if grid is None:
        grid = []
        grid_1d = np.linspace(0, 1, grid_sz)
        mesh_x,mesh_y  = np.meshgrid(grid_1d, grid_1d)
        # very suboptimal, but for grid_sz < 1000 why not
        for i in range(grid_sz):
            for j in range(grid_sz):
                w1 = mesh_x[i][j]
                w2 = mesh_y[i][j]
                if w1 + w2 <= 1:
                    w3 = 1 - w1 - w2
                    grid.append((w1, w2, w3))
        print(f"Calculated grid size = {len(grid)}")
    grid_sz_full = len(grid)
    first_metrics = []
    second_metrics = []
    third_metrics = []

    for point_id in tqdm(range(1, grid_sz_full + 1)):
        point_res = get_pareto_3d_point(point_id, loss_fncs, metrics_fncs, x, y, groups,
                                     loss_default_coeffs, n_folds, grid, robust_loss_eps)
        first_metrics.append(point_res[0])
        second_metrics.append(point_res[1])
        third_metrics.append(point_res[2])
    return [first_metrics, second_metrics, third_metrics]
