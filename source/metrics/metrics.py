import numpy as np
import scipy.stats as sps

from source.models.torch_compatible_losses import log_loss_wrapper, group_fair_loss, ind_fair_loss
from source.models.torch_compatible_losses import robust_adv_loss, robust_l2_loss

from torch.special import expit
import torch


def compute_accuracy(params):
    """
    Standard accuracy
    :param params: dict with predicted probabilities and true labels
    :return: accuracy
    """
    pred_labels = params["pred_probs"] > 0.5
    true_labels = params["true_labels"]
    return (pred_labels == true_labels).sum() / pred_labels.shape[0]


def compute_demographic_fairness(params):
    """
    Standard demographic fairness for hard labels, squared
    :param params: dict with predicted probabilities and group labels
    :return: demographic fairness
    """
    pred_probs = params['pred_probs']
    groups = params["groups"]
    assert np.unique(groups).size == 2
    first_labels = (pred_probs[groups == 0] > 0.5).double()
    second_labels = (pred_probs[groups == 1] > 0.5).double()
    error = np.abs(first_labels.mean() - second_labels.mean())
    return 1 - error ** 2  # squared or not squared, that's the question --> squared


def compute_individual_fairness(params):
    """
    Computes individual fairness for binary classification problem for hard labels
    Follows "A Convex Framework for Fair Regression" paper https://arxiv.org/pdf/1706.02409.pdf
    :param params: dict with true labels, predicted logits and binary group labels
    :return: individual fairness
    """
    true_labels = params["true_labels"]
    pred_labels = params["pred_probs"] > 0.5
    groups = params["groups"]
    norm_const = groups.sum() * (1 - groups).sum()
    error = 0
    for label in (0, 1):
        msk = true_labels == label
        cur_pred_labels = pred_labels[msk]
        cur_groups = groups[msk]
        n_1 = cur_groups.sum()
        n_2 = (1 - cur_groups).sum()
        first_probs = cur_pred_labels[cur_groups == 0]
        second_probs = cur_pred_labels[cur_groups == 1]
        cur_loss = n_2 * np.square(first_probs).sum() + \
                   n_1 * np.square(second_probs).sum() - \
                   2 * first_probs.sum() * second_probs.sum()
        error += cur_loss / norm_const
    return 1 - error


def compute_l2_robustness(params, default_rob_eps=0.3, shift_sample_num=30):
    """
    Computes l_2 adv mean robustness (consistency).
    :param params:
    :param default_rob_eps: default epsilon (allowed change for adversary)
    :param shift_sample_num:
    :return: l_2 adv mean robustness
    """
    df = params['data']
    model = params['model']
    probs = expit(params["pred_probs"])
    if "robust_loss_eps" in params.keys() and params["robust_loss_eps"]:
        eps = params["robust_loss_eps"]
    else:
        eps = default_rob_eps
    if "shift_sample_num" in params.keys():
        shift_sample_num = params["shift_sample_num"]
    sample_num = df.shape[0]
    features_num = df.shape[1]
    shifted_df = df[:, :, np.newaxis] + sps.uniform(-eps, 2 * eps).rvs(
        size=(sample_num, features_num, shift_sample_num))
    shifted_df = shifted_df.transpose(1, 2).float()
    shifted_probs = expit(model(shifted_df))
    y_pred = (probs > 0.5).float()
    y_shifted_pred = (shifted_probs > 0.5).float()
    error = torch.square(y_pred[:, np.newaxis] - y_shifted_pred).mean()
    return 1 - error


# Following functions are self-explanatory
def compute_adv_robustness(params):
    return 1 - robust_adv_loss(params).detach().numpy()


def log_loss_as_metric(params):
    return log_loss_wrapper(params).detach().numpy()


def group_fair_loss_as_metric(params):
    return group_fair_loss(params).detach().numpy()


def ind_fair_loss_as_metric(params):
    return ind_fair_loss(params).detach().numpy()


def robust_adv_loss_as_metric(params):
    return robust_adv_loss(params).detach().numpy()


def robust_l2_loss_as_metric(params):
    return robust_l2_loss(params).detach().numpy()
