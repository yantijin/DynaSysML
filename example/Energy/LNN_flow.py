import torch
import torch.nn as nn
from DynaSysML.Energy import LNN


class LNN_flow(nn.Module):
    def __init__(self, net, delta_t):
        super(LNN_flow, self).__init__()
        self.net = net
        self.lnn_layer = LNN(net)
        self.delta_t = delta_t

    def forward(self, x, method='euler', steps=10):
        sol = []
        for i in range(steps):
            x = self.get_method(method=method)(x)
            sol.append(x)
        solution = torch.stack(sol, 0)
        return solution

    def _euler_step(self, x):
        grads = self.lnn_layer(x)
        x_out = x + grads * self.delta_t
        return x_out

    def _rk_step(self, x):
        grads = self.lnn_layer(x)

        x2 = x + grads * self.delta_t / 2
        k2 = self.lnn_layer(x2)

        x3 = x + self.delta_t * k2 / 2
        k3 = self.lnn_layer(x3)

        x4 = x + self.delta_t * k3 / 2
        k4 = self.lnn_layer(x4)
        x_out = x + self.delta_t * (grads/6 + k2/3+k3/3+k4/6)
        return x_out

    def get_method(self, method):
        dict1 = {'euler': self._euler_step,
                 'rk4': self._rk_step}
        return dict1[method]


if __name__ == "__main__":
    hdim=4
    a = torch.randn(32, 2)
    b = torch.randn(10, 32, 2)
    md = LNN_flow(nn.Sequential(
            nn.Linear(2,hdim),
            nn.Softplus(),
            nn.Linear(hdim,hdim),
            nn.Softplus(),
            nn.Linear(hdim,1)), delta_t=0.1)
    optim = torch.optim.Adam(md.parameters(), lr=1e-3)
    optim.zero_grad()
    solution = md(a, method='rk4', steps=10)
    print(solution.shape)
    loss = torch.nn.functional.mse_loss(solution, b)
    loss.backward()
    optim.step()