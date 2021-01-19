import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
import DynaSysML as dsl
from datasets import ConcentricSphere
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


data_dim = 2
data_concentric = ConcentricSphere(data_dim, inner_range=(0., .5), outer_range=(1., 1.5),
                                   num_points_inner=1000, num_points_outer=2000)
dataloader = DataLoader(data_concentric, batch_size=64, shuffle=True)

# Visualize a batch of data (use a large batch size for visualization)
dataloader_viz = DataLoader(data_concentric, batch_size=256, shuffle=True)

name = 'augment_2order'

if name == 'augment':

    class aug_learner(pl.LightningModule):
        def __init__(self, model, augment_dim=0):
            super(aug_learner, self).__init__()
            self.augment_dim = augment_dim
            self.model = model

        def forward(self, x):
            if self.augment_dim > 0:
                shape1 = list(x.shape[:-1]) + [self.augment_dim]
                aug = torch.zeros(shape1).to(device)
                x = torch.cat([x, aug], dim=-1)
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # t = torch.linspace(0, 1, 2)
            if self.augment_dim > 0:
                shape1 = list(x.shape[:-1]) + [self.augment_dim]
                aug = torch.zeros(shape1).to(device)
                x = torch.cat([x, aug], dim=-1).to(device)
            y_hat = self.model(x)
            # print(y.shape, y_hat.shape)
            loss = nn.CrossEntropyLoss()(y_hat, y.squeeze())
            logs = {'loss': loss}
            return {'loss': loss, 'logs': logs}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.005)

        def train_dataloader(self):
            return dataloader

    class ODEFunc(nn.Module):
        def __init__(self, in_dim=2, augment_dim=0):
            super(ODEFunc, self).__init__()
            self.in_dim = in_dim
            self.augment_dim = augment_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim+augment_dim, 50),
                nn.Tanh(),
                nn.Linear(50, in_dim+augment_dim),
            )

        def forward(self, t, y):
            return self.net(y)

    class aModel(nn.Module):
        def __init__(self, augment_dim=1, out_dim=2):
            super(aModel, self).__init__()
            self.augment_Dim = augment_dim
            func = ODEFunc(augment_dim=self.augment_Dim)
            self.basis_mdoel = dsl.DE.NeuralODE(func, t=torch.linspace(0, 1, 2).to(device), last=True).to(device)
            self.linear = nn.Linear(self.basis_mdoel.func.in_dim+self.augment_Dim, out_dim)
        def forward(self, x):
            out = self.basis_mdoel(x)
            out = self.linear(out)
            return out
    model = aModel().to(device)

    learn = aug_learner(model, augment_dim=1)
    trainer = pl.Trainer(min_epochs=6, max_epochs=10, progress_bar_refresh_rate=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)

    for inputs, target in dataloader_viz:
        print(inputs.shape, target.shape)
        color = ['orange' if target[i, 0]>0.0 else 'blue' for i in range(len(target))]
        plt.figure()
        plt.scatter(inputs[:,0].numpy(), inputs[:, 1].numpy(), c=color)
        fig = plt.figure(figsize=(6, 3))
        data = torch.tensor([i.numpy() for i in data_concentric.data]).to(device)
        shape1 = list(data.shape[:-1]) + [1]
        aug = torch.zeros(shape1).to(device)
        data = torch.cat([data, aug], dim=-1)
        trajectory = model.basis_mdoel.trajectory(torch.tensor(data), s_span).detach().cpu()
        ax = Axes3D(fig)
        colors = ['orange', 'blue']
        for i in range(len(data)):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2], color=colors[data_concentric.targets[i][0].numpy()], alpha=.1)
        break
    plt.show()

elif name == 'augment1': # test Augmenter Module
    class aug_learner(pl.LightningModule):
        def __init__(self, model, augment_dim=0):
            super(aug_learner, self).__init__()
            self.augment_dim = augment_dim
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # t = torch.linspace(0, 1, 2)
            y_hat = self.model(x)
            # print(y.shape, y_hat.shape)
            loss = nn.CrossEntropyLoss()(y_hat, y.squeeze())
            logs = {'loss': loss}
            return {'loss': loss, 'logs': logs}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.005)

        def train_dataloader(self):
            return dataloader

    class ODEFunc(nn.Module):
        def __init__(self, in_dim=2, augment_dim=0):
            super(ODEFunc, self).__init__()
            self.in_dim = in_dim
            self.augment_dim = augment_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim+augment_dim, 50),
                nn.Tanh(),
                nn.Linear(50, in_dim+augment_dim),
            )

        def forward(self, t, y):
            return self.net(y)

    class aModel(nn.Module):
        def __init__(self, augment_dim=1, out_dim=2):
            super(aModel, self).__init__()
            self.augment_Dim = augment_dim
            self.augment = dsl.DE.Augmenter(augment_dims=self.augment_Dim, order='second')
            func = ODEFunc(augment_dim=self.augment_Dim).to(device)
            self.basis_mdoel = dsl.DE.NeuralODE(func, t=torch.linspace(0, 1, 2).to(device), last=True).to(device)
            self.linear = nn.Linear(self.basis_mdoel.func.in_dim+self.augment_Dim, out_dim)
        def forward(self, x):
            x = self.augment(x)
            out = self.basis_mdoel(x)
            out = self.linear(out)
            return out
    model = aModel().to(device)

    learn = aug_learner(model, augment_dim=1)
    trainer = pl.Trainer(min_epochs=6, max_epochs=10, progress_bar_refresh_rate=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)

    for inputs, target in dataloader_viz:
        print(inputs.shape, target.shape)
        color = ['orange' if target[i, 0]>0.0 else 'blue' for i in range(len(target))]
        plt.figure()
        plt.scatter(inputs[:,0].numpy(), inputs[:, 1].numpy(), c=color)
        fig = plt.figure(figsize=(6, 3))
        data = torch.tensor([i.numpy() for i in data_concentric.data]).to(device)
        shape1 = list(data.shape[:-1]) + [1]
        aug = torch.zeros(shape1).to(device)
        data = torch.cat([data, aug], dim=-1)
        trajectory = model.basis_mdoel.trajectory(torch.tensor(data), s_span).detach().cpu()
        ax = Axes3D(fig)
        colors = ['orange', 'blue']
        for i in range(len(data)):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2], color=colors[data_concentric.targets[i][0].numpy()], alpha=.1)
        break
    plt.show()

elif name == 'il_augment':
    class aug_learner(pl.LightningModule):
        def __init__(self, model, augment_dim=0):
            super(aug_learner, self).__init__()
            self.augment_dim = augment_dim
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # t = torch.linspace(0, 1, 2)
            y_hat = self.model(x)
            # print(y.shape, y_hat.shape)
            loss = nn.CrossEntropyLoss()(y_hat, y.squeeze())
            logs = {'loss': loss}
            return {'loss': loss, 'logs': logs}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.005)

        def train_dataloader(self):
            return dataloader

    class ODEFunc(nn.Module):
        def __init__(self, in_dim=2, augment_dim=0):
            super(ODEFunc, self).__init__()
            self.in_dim = in_dim
            self.augment_dim = augment_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim+augment_dim, 50),
                nn.Tanh(),
                nn.Linear(50, in_dim+augment_dim),
            )

        def forward(self, t, y):
            return self.net(y)

    class aModel(nn.Module):
        def __init__(self, augment_dim=1, out_dim=2):
            super(aModel, self).__init__()
            self.augment_Dim = augment_dim
            self.augment = dsl.DE.Augmenter(augment_dims=self.augment_Dim, order='second',
                                              augment_func=nn.Linear(2, 1).to(device), augment_idx=-1)# note: this is used for plot
            # func = ODEFunc(augment_dim=self.augment_Dim)
            func = nn.Sequential(nn.Linear(3, 50), nn.Tanh(), nn.Linear(50, 3))
            self.basis_mdoel = dsl.DE.NeuralODE(func, t=torch.linspace(0, 1, 2).to(device), last=True).to(device)
            # self.linear = nn.Linear(self.basis_mdoel.func.in_dim+self.augment_Dim, out_dim)
            self.linear = nn.Linear(3, out_dim)
        def forward(self, x):
            x = self.augment(x)
            out = self.basis_mdoel(x)
            out = self.linear(out)
            return out
    model = aModel().to(device)

    learn = aug_learner(model, augment_dim=1)
    trainer = pl.Trainer(min_epochs=6, max_epochs=10, progress_bar_refresh_rate=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)

    for inputs, target in dataloader_viz:
        print(inputs.shape, target.shape)
        color = ['orange' if target[i, 0]>0.0 else 'blue' for i in range(len(target))]
        plt.figure()
        plt.scatter(inputs[:,0].numpy(), inputs[:, 1].numpy(), c=color)
        fig = plt.figure(figsize=(6, 3))

        data = torch.tensor([i.numpy() for i in data_concentric.data]).to(device)

        data = model.augment(data) # note: this is different with that in ANODE
        trajectory = model.basis_mdoel.trajectory(torch.tensor(data), s_span).detach().cpu()
        ax = Axes3D(fig)
        colors = ['orange', 'blue']
        for i in range(len(data)):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2], color=colors[data_concentric.targets[i][0].numpy()], alpha=.1)
        break
    plt.show()

elif name == 'augment_2order':
    class aug_learner(pl.LightningModule):
        def __init__(self, model, augment_dim=0):
            super(aug_learner, self).__init__()
            self.augment_dim = augment_dim
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # t = torch.linspace(0, 1, 2)
            y_hat = self.model(x)
            # print(y.shape, y_hat.shape)
            loss = nn.CrossEntropyLoss()(y_hat, y.squeeze())
            logs = {'loss': loss}
            return {'loss': loss, 'logs': logs}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.005)

        def train_dataloader(self):
            return dataloader

    class ODEFunc(nn.Module):
        def __init__(self, in_dim=2, augment_dim=0):
            super(ODEFunc, self).__init__()
            self.in_dim = in_dim
            self.augment_dim = augment_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim+augment_dim, 50),
                nn.Tanh(),
                nn.Linear(50, in_dim+augment_dim),
            )

        def forward(self, t, y):
            return self.net(y)

    class aModel(nn.Module):
        def __init__(self, augment_dim=1, out_dim=2):
            super(aModel, self).__init__()
            self.augment_Dim = augment_dim
            self.augment = dsl.DE.Augmenter(augment_dims=self.augment_Dim, order='second', # augment_func=nn.Linear(2, 1).to(device),
                                              augment_idx=-1)# note: this is used for plot
            # func = ODEFunc(augment_dim=self.augment_Dim)
            func = nn.Sequential(nn.Linear(4, 50), nn.Tanh(), nn.Linear(50, 2))
            self.basis_mdoel = dsl.DE.NeuralODE(func, t=torch.linspace(0, 1, 2).to(device), last=True, order=2).to(device)
            # self.linear = nn.Linear(self.basis_mdoel.func.in_dim+self.augment_Dim, out_dim)
            self.linear = nn.Linear(2*2, out_dim)
        def forward(self, x):
            x = self.augment(x)
            out = self.basis_mdoel(x)
            out = self.linear(out)
            return out
    model = aModel(2).to(device)

    learn = aug_learner(model, augment_dim=2)
    trainer = pl.Trainer(min_epochs=6, max_epochs=10, progress_bar_refresh_rate=1)
    trainer.fit(learn)
    s_span = torch.linspace(0, 1, 100)

    for inputs, target in dataloader_viz:
        print(inputs.shape, target.shape)
        color = ['orange' if target[i, 0]>0.0 else 'blue' for i in range(len(target))]
        plt.figure()
        plt.scatter(inputs[:,0].numpy(), inputs[:, 1].numpy(), c=color)
        fig = plt.figure(figsize=(6, 3))

        data = torch.tensor([i.numpy() for i in data_concentric.data]).to(device)

        data = model.augment(data) # note: this is different with that in ANODE
        trajectory = model.basis_mdoel.trajectory(torch.tensor(data), s_span).detach().cpu()
        ax = Axes3D(fig)
        colors = ['orange', 'blue']
        for i in range(len(data)):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2], color=colors[data_concentric.targets[i][0].numpy()], alpha=.1)
        break
    plt.show()







