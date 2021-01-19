import torch
import torch.nn as nn
import numpy as np
from DynaSysML.DE import *
import matplotlib.pyplot as plt

data_size = 1000
batch_time = 20
niters = 3000
batch_size = 16

t_grid = torch.tensor(np.linspace(0, 25, data_size))
true_y0 = torch.tensor([[2., 0.]])
true_A = torch.tensor([[-0.1, 2.0],[-2.0, -0.1]])

# create spiral datasets
class La(torch.nn.Module):
    def forward(self, t, y, **kwargs):
        return torch.matmul(y**3, true_A)
with torch.no_grad():
    yN = odeint_adjoint(La(), true_y0, t_grid)
# print(yN.shape)
yN = torch.squeeze(yN)
# print(yN)
def plot_spiral(trajectories):
    for traj in trajectories:
        plt.plot(traj[:,0], traj[:,1])
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# plot_spiral(yN)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = yN[s]  # (M, D)
    batch_t = t_grid[:batch_time]  # (T)
    batch_y = torch.stack([yN[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)

func = ODEFunc()
model = NeuralODE(func)
# optim = torch.optim.Adam(func.parameters(), lr=1e-3)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# ode int model
for i in range(1, niters+1):
    optim.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    # print(type(batch_y0))
    yout = model(batch_y0, batch_t)
    yout = torch.squeeze(yout)
    # print(yout.shape, batch_y.shape)
    loss = torch.mean(torch.abs(yout - batch_y))
    loss.backward()
    optim.step()
    if i % 50 == 0:
        print(i)
        y_test = model(true_y0, t_grid)
        y_test = torch.squeeze(y_test)
        plot_spiral([yN, y_test.detach().numpy()])




# ode_int function
# for i in range(1, niters +1):
#     optim.zero_grad()
#     batch_y0, batch_t, batch_y = get_batch()
#     yout = odeint_adjoint(func, batch_y0, batch_t)
#     # print(yout.shape)
#     yout = torch.squeeze(yout)
#     loss = torch.mean(torch.abs(yout - batch_y))
#     loss.backward()
#     optim.step()
#
#     if i % 500 == 0:
#         print(i)
#         y_test = odeint_adjoint(func, true_y0, t_grid)
#         y_test = torch.squeeze(y_test)
#         plot_spiral([yN, y_test.detach().numpy()])


