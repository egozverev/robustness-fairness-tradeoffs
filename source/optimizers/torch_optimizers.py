import numpy as np
import scipy.stats as sps
from source.models.linear_models import TorchLinearModel
from source.models.hypermodels import HyperLinearNet

from tqdm import tqdm

from torch.optim import Adam
import torch


def get_total_error(X, y, model, loss_fncs, loss_coeffs, groups, **kwargs):
    error = 0.0
    y_pred = model(X)
    params = {
        "true_labels": y,
        "pred_logits": y_pred,
        "groups": groups,
        "model": model,
        "data": X,
        #"robust_loss_eps": robust_loss_eps
    }
    # if theta is not None:
    #     params['theta'] = theta
    params.update(kwargs)
    for fnc, coeff in zip(loss_fncs, loss_coeffs):
        error += coeff * fnc(params)
    return error


def get_optimal_model(X, y, loss_fncs, loss_coeffs, groups=None, epoch_num=250,
                      use_mini_batch=True, batch_size=4096, **kwargs):
    "Training loop for torch model."
    # losses = []
    model = TorchLinearModel(X.shape[1])
    optimizer = Adam(model.parameters(), lr=0.001)
    if not use_mini_batch:
        batch_size = X.size()[0]
    for epoch in range(epoch_num):
        permutation = torch.randperm(X.size()[0])
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y, batch_groups = X[indices], y[indices], groups[indices]
            loss = get_total_error(batch_X, batch_y, model, loss_fncs, loss_coeffs, batch_groups, **kwargs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # losses.append(loss)
    return model


### ---------------------------------------------------
### HYPERNETWORKS
### ---------------------------------------------------

def get_optimal_hypernetwork(preprocessed_data,
                             loss_fncs, exclude_last_loss=True,
                             epoch_num=50, use_mini_batch=True, batch_size=4096,
                             dir_alpha=0.2, **kwargs):
    """
    Training procedure for hypernetwork

    :param preprocessed_data: train, tets and val data
    :param loss_fncs: a list of loss functions
    :param exclude_last_loss: if True, last loss (typically regulizer) is not included in pareto front
    :param epoch_num: number of training epochs
    :param use_mini_batch: whether to use mini-matches
    :param batch_size: batch size for mini-batches
    :param dir_alpha: parameter of the Dirichlet distribution used for ray sampling during training
    :param kwargs: additional arguments
    :return: trained hypernetwork
    """
    X_train, y_train, groups_train = preprocessed_data["train"]
    X_val, y_val, groups_val = preprocessed_data["val"]
    train_loss_fncs = loss_fncs[:-1] if exclude_last_loss else loss_fncs
    lr = kwargs["learning_rate"] if "learning_rate" in kwargs else 0.001
    regulizer_strength = kwargs["regulizer_strength"] if "regulizer_strength" in kwargs else 0.1
    hyper_model = HyperLinearNet(ray_size=len(train_loss_fncs), theta_dim=X_train.shape[1])
    optimizer = Adam(hyper_model.parameters(), lr=lr)
    if not use_mini_batch:
        batch_size = X_train.size()[0]
    train_losses = []
    val_losses = []
    for epoch in range(epoch_num):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            ray = torch.from_numpy(
                np.random.dirichlet(np.ones(len(train_loss_fncs)) * dir_alpha, 1).astype(np.float32).flatten()
            )
            theta = hyper_model(ray)
            model = lambda x: x @ theta
            indices = permutation[i:i + batch_size]
            batch_X, batch_y, batch_groups = X_train[indices], y_train[indices], groups_train[indices]
            loss_coeffs = np.hstack((ray, [regulizer_strength])) if exclude_last_loss else ray
            loss = get_total_error(batch_X, batch_y, model, loss_fncs, loss_coeffs, batch_groups,
                                   theta=theta, **kwargs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ray = torch.Tensor(np.ones(len(train_loss_fncs)) / len(train_loss_fncs))
        theta = hyper_model(ray)
        loss_coeffs = np.hstack((ray, [regulizer_strength])) if exclude_last_loss else ray
        train_loss = get_total_error(X_train, y_train, lambda x: x @ theta, loss_fncs, loss_coeffs, groups_train,
                                     theta=theta, **kwargs).item()
        train_losses.append(train_loss)
        optimizer.zero_grad()
        val_loss = get_total_error(X_val, y_val, lambda x: x @ theta, loss_fncs, loss_coeffs, groups_val,
                                   theta=theta, **kwargs).item()
        val_losses.append(val_loss)
        optimizer.zero_grad()
    # print("HYPERNET TRAIN LOSSES")
    # print(train_losses)
    # print("HYPERNET VAL LOSSES")
    # print(val_losses)
    if "return_train_losses" in kwargs and kwargs["return_train_losses"]:
        return hyper_model, train_losses, val_losses
    return hyper_model
