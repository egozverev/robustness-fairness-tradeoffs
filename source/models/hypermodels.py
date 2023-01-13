from torch import nn


class HyperLinearNet(nn.Module):
    """
    3-layer MLP hyper network for linear/logistic regression
    """

    def __init__(self, ray_hidden_dim=100, ray_size=2, theta_dim=10):
        """
        Hyper network initialization
        :param ray_hidden_dim: hidden dimension of the hypernetwork linear layers
        :param ray_size: the dimension of input rays
        :param theta_dim: dimension of linear model parameter vector
        """
        super().__init__()
        self.ray_mlp = nn.Sequential(
            nn.Linear(ray_size, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, theta_dim),
        )

    def forward(self, ray):
        """
        Hypernetwork forward pass
        :param ray: input ray
        :return: linear model parameter vector (theta)
        """
        theta = self.ray_mlp(ray)
        return theta
