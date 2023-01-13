import numpy as np
from scipy.special import expit as spsexpit
import numpy as np
import torch
from torch import nn

class TorchLinearModel(nn.Module):
    """
    Linear model suitable for torch gradient optimization.
    """

    def __init__(self, theta_sz):
        """
        Initialization with uniform noise.
        :param theta_sz: number of parameters (theta dimension)
        """
        super().__init__()
        delta = 1.0 / np.sqrt(theta_sz)
        theta = torch.distributions.Uniform(-delta, delta).sample((theta_sz,))
        self.theta = nn.Parameter(theta)

    def forward(self, X):
        """
        Linear Model Forward Pass
        :param X: features, assuming last column is ones
        :return: logits
        """
        logits = X @ self.theta
        return logits


def logistic_model(theta, X):
    """
    Standard logistic regression model.
    Used in early scipy experiments, left for compatibility with previous experiments.
    """
    logits = X @ theta
    probs = spsexpit(logits)
    return probs
