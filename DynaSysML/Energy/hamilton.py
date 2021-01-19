import torch
import torch.nn as nn


class HNN(nn.Module):
    """Hamiltonian Neural ODE

    :param net: function parametrizing the vector field.
    :type net: nn.Module
    """
    def __init__(self, net:nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True) ; n = x.shape[1] // 2
            gradH = torch.autograd.grad(self.net(x).sum(), x,
                                        create_graph=True)[0]
        return torch.cat([gradH[:, n:], -gradH[:, :n]], 1).to(x)