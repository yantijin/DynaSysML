import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from DynaSysML.DE import odeint_adjoint, odeint, NeuralODE
import tqdm

# d^2x/dt^2 - \mu(1-x^2) dx/dt + x = 0;
# x(0) = 0, dx(0)/dt=0

position_0 = 1.
velocity_0 = 0.
data_size = 1000
batch_time = 20
niters = 3000
batch_size = 16
true_y0 = torch.tensor([position_0, velocity_0])
t_grid = torch.tensor(np.linspace(0, 25, data_size))

class La(torch.nn.Module):

    def forward(self, t, y, mu=1,**kwargs):
        b = mu * ( 1- y[0]**2) * y[1] - y[0]
        a = y[1]
        res = torch.tensor([a,b])
        return res
for par in La().parameters():
    print(par)
with torch.no_grad():
    yN = odeint_adjoint(La(), true_y0, t_grid) # [data_size, 2]
print(yN.shape)

font = {
    'size': 12,
    'weight': 'normal'
}
def plot_high_order(trajectoris):
    plt.figure(1)
    plt.subplot(221)
    for trajectory in trajectoris:
        plt.plot(t_grid, trajectory[:, 0])
    plt.xlabel('t')
    plt.ylabel('position')
    # plt.figure(2)
    plt.subplot(222)
    for trajectory in trajectoris:
        plt.plot(t_grid, trajectory[:, 1])
    plt.xlabel('t')
    plt.ylabel('velocity', fontdict=font)

    # plt.figure(3)
    plt.subplot(212)
    for trajectory in trajectoris:
        plt.plot(trajectory[:,0], trajectory[:,1])
    plt.xlabel('position')
    plt.ylabel('velocity')

    plt.show()

plot_high_order([yN])


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
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        a = y[..., 1:]
        b = self.net(y)
        res = torch.cat([a, b], dim=-1)
        return res

func = ODEFunc()
model = NeuralODE(func)
# optim = torch.optim.Adam(func.parameters(), lr=1e-3)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# ode int model
for i in tqdm.tqdm(range(1, niters+1)):
    optim.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    yout = model(batch_y0, batch_t, ) # method='rk4')
    # print(yout.shape, batch_y.shape)
    loss = torch.mean(torch.abs(yout - batch_y))
    loss.backward()
    optim.step()
    if i % 500 == 0:
        print(i)
        y_test = model(true_y0, t_grid)
        y_test = torch.squeeze(y_test)
        plot_high_order([yN, y_test.detach().numpy()])

