import torch
import torch.nn as nn
import torch.utils.data as data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from DynaSysML.DE import DepthCat
import numpy as np
import DynaSysML as dsl
from DynaSysML.DE.galerkin import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytorch_lightning as pl

def generate_moons(n_samples=100, noise=1e-4, **kwargs):
    """Creates a *moons* dataset of `n_samples` data points.

    :param n_samples: number of data points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each data point
    :type noise: float
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        X += np.random.rand(n_samples, 1) * noise

    X, y = torch.Tensor(X), torch.Tensor(y).long()
    return X, y

def plot_2D_depth_trajectory(s_span, trajectory, yn, n_lines):
    color=['orange', 'blue']

    fig = plt.figure(figsize=(8,2))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    for i in range(n_lines):
        ax0.plot(s_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
        ax1.plot(s_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);

    ax0.set_xlabel(r"$s$ [Depth]")
    ax0.set_ylabel(r"$h_0(s)$")
    ax0.set_title("Dimension 0")
    ax1.set_xlabel(r"$s$ [Depth]")
    ax1.set_ylabel(r"$h_1(s)$")
    ax1.set_title("Dimension 1")

def plot_2D_state_space(trajectory, yn, n_lines):
    color=['orange', 'blue']

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    for i in range(n_lines):
        ax.plot(trajectory[:,i,0], trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);

    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")
    ax.set_title("Flows in the state-space")

def plot_2D_space_depth(s_span, trajectory, yn, n_lines):
    colors = ['orange', 'blue']
    fig = plt.figure(figsize=(6,3))
    ax = Axes3D(fig)
    for i in range(n_lines):
        ax.plot(s_span, trajectory[:,i,0], trajectory[:,i,1], color=colors[yn[i].int()], alpha = .1)
        ax.view_init(30, -110)

    ax.set_xlabel(r"$s$ [Depth]")
    ax.set_ylabel(r"$h_0$")
    ax.set_zlabel(r"$h_1$")
    ax.set_title("Flows in the space-depth")
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


X, yn = generate_moons(n_samples=512, noise=0.1)
print (X.shape, yn.shape)
colors = ['orange', 'blue']
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i, 0], X[i, 1], s=1, color=colors[yn[i].int()])
# plt.show()

X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

class basis_learner(pl.LightningModule):
    def __init__(self, model):
        super(basis_learner, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y= batch
        # t = torch.linspace(0, 1, 2)
        y_hat = self.model(x)
        # print(y.shape, y_hat.shape)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'loss': loss}
        return {'loss': loss, 'logs': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return trainloader
name = 'stack'
if name == 'basic':
    class ODEFunc(nn.Module):

        def __init__(self):
            super(ODEFunc, self).__init__()

            self.net = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 2),
            )

        def forward(self, t, y):
            return self.net(y)

    func = ODEFunc()

    basis_model = dsl.DE.NeuralODE(func, t=torch.linspace(0,1,2).to(device), last=True).to(device)
    learn = basis_learner(basis_model)
    trainer = pl.Trainer(min_epochs=200, max_epochs=250, progress_bar_refresh_rate=1)
    trainer.fit(learn)

    s_span = torch.linspace(0, 1, 100)
    trajectory = basis_model.trajectory(X_train, s_span).detach().cpu()
    plot_2D_depth_trajectory(s_span, trajectory, yn, len(X))
    plot_2D_state_space(trajectory, yn, len(X))
    plot_2D_space_depth(s_span, trajectory, yn, len(X))
    plt.show()

elif name == 'gal':
    class ODEFunc(nn.Module):
        def __init__(self):
            super(ODEFunc, self).__init__()
            self.net = nn.Sequential(
                DepthCat(s=1, idx_cat=1),
                GalLinear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 2),
            )

        def forward(self, t, y):
            return self.net(y)


    func = ODEFunc()

    basis_model = dsl.DE.NeuralODE(func, t=torch.linspace(0, 1, 2).to(device), last=True).to(device)
    learn = basis_learner(basis_model)
    trainer = pl.Trainer(min_epochs=150, max_epochs=200, progress_bar_refresh_rate=1)
    trainer.fit(learn)

    s_span = torch.linspace(0, 1, 100)
    trajectory = basis_model.trajectory(X_train, s_span).detach().cpu()
    plot_2D_depth_trajectory(s_span, trajectory, yn, len(X))
    plot_2D_state_space(trajectory, yn, len(X))
    plot_2D_space_depth(s_span, trajectory, yn, len(X))
    plt.show()
elif name == 'stack':
    nde = []
    num_pieces = 5
    class ODEFunc(nn.Module):
        def __init__(self):
            super(ODEFunc, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 2),
            )

        def forward(self, t, y):
            return self.net(y)

    for i in range(num_pieces):
        nde.append(
            dsl.DE.NeuralODE(ODEFunc(), t=torch.linspace(0, 1, 2).to(device), last=True),
        )
    basic_model = nn.Sequential(*nde).to(device)
    learn = basis_learner(basic_model)
    trainer = pl.Trainer(min_epochs=350, max_epochs=400, progress_bar_refresh_rate=1)
    trainer.fit(learn)

    s_span = torch.linspace(0, 1, 20)
    trajectory = [basic_model[0].trajectory(X_train, s_span)]
    for i in range(1, num_pieces):
        trajectory.append(
            basic_model[i].trajectory(trajectory[i - 1][-1, :, :], s_span))
    trajectory = torch.cat(trajectory, 0).detach().cpu()
    tot_s_span = torch.linspace(0, 5, 100)
    plot_2D_depth_trajectory(tot_s_span, trajectory, yn, len(X))
    plot_2D_state_space(trajectory, yn, len(X))
    plot_2D_space_depth(tot_s_span, trajectory, yn, len(X))
    plt.show()