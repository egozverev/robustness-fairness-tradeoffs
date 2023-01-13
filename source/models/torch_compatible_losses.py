# from sklearn.metrics import log_loss
from torch.nn import BCEWithLogitsLoss
import scipy.stats as sps
import numpy as np

import torch
from torch.special import expit


def log_loss_wrapper(params):
    """
    Wrapper for BCEWithLogitsLoss
    :param params: dict with true labels and predicted logits
    :return:
    """
    true_labels = params["true_labels"]
    pred_logits = params["pred_logits"]
    return BCEWithLogitsLoss()(pred_logits, true_labels)


def regulizer_loss(params):
    """
    Wrapper for l2-regularization loss
    :param params: dict with either theta or model that has theta as parameter
    :return:
    """
    if "theta" in params:
        theta = params['theta']
    else:
        theta = params["model"].theta
    return torch.square(theta).sum()


def group_fair_loss(params):
    """
    Standard demographic fairness for soft labels, squared
    :param params: dict with predicted probabilities and group labels
    :return:
    """
    pred_logits = params["pred_logits"]
    pred_probs = expit(pred_logits)
    groups = params["groups"]
    assert np.unique(groups).size == 2
    first_probs = pred_probs[groups == 0]
    second_probs = pred_probs[groups == 1]
    error = (first_probs.mean() - second_probs.mean()) ** 2
    return error


def ind_fair_loss(params):
    """
    Computes individual fairness loss for binary classification problem for soft labels
    Follows "A Convex Framework for Fair Regression" paper https://arxiv.org/pdf/1706.02409.pdf
    :param params: dict with true labels, predicted logits and binary group labels
    :return: individual fairness
    """
    true_labels = params["true_labels"]
    pred_probs = expit(params["pred_logits"])
    groups = params["groups"]
    norm_const = groups.sum() * (1 - groups).sum()
    loss = 0
    for label in (0, 1):
        msk = true_labels == label
        cur_probs = pred_probs[msk]
        cur_groups = groups[msk]
        n_1 = cur_groups.sum()
        n_2 = (1 - cur_groups).sum()
        first_probs = cur_probs[cur_groups == 0]
        second_probs = cur_probs[cur_groups == 1]
        cur_loss = n_2 * torch.square(first_probs).sum() + \
                   n_1 * torch.square(second_probs).sum() - \
                   2 * first_probs.sum() * second_probs.sum()
        loss += cur_loss / norm_const
    return loss


def robust_adv_loss(params, default_rob_eps=0.01):
    """
    Computes adversarial robustness loss for linear model
    :param params: dict with features, model, true labels, (optional) param vector and (optional) epsilon
    :param default_rob_eps: default epsilon for adversary
    :return:
    """
    df = params['data']
    model = params['model']
    true_labels = params["true_labels"]
    if "robust_loss_eps" in params.keys():
        eps = params["robust_loss_eps"]
    else:
        eps = default_rob_eps
    if "theta" in params.keys():
        theta = params['theta']
    else:
        assert model.theta is not None
        theta = model.theta
    delta = eps * theta / torch.linalg.norm(theta)
    shifted_df = torch.clone(df)
    shifted_df[true_labels == 1] -= delta
    shifted_df[true_labels == 0] += delta
    shifted_logits = model(shifted_df)
    loss = BCEWithLogitsLoss()(shifted_logits, true_labels)
    return loss


def robust_l2_loss(params, default_rob_eps=0.01, shift_sample_num=30):
    """
    Computes l2-pairwise mean-robustness (consistency) loss for linear model
    :param params: dict with features, predicted logits, (optional) epsilon and (optional) sample number
    :param default_rob_eps: default epsilon
    :param shift_sample_num: sample num for mean estimation
    :return:
    """
    df = params['data']
    model = params['model']
    logits = params["pred_logits"]
    probs = expit(logits)
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
    shifted_logits = model(shifted_df)
    shifted_probs = expit(shifted_logits)
    loss = torch.square(probs[:, np.newaxis] - shifted_probs).mean()
    return loss
