import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.utils.data as data
from DynaSysML.Energy import LNN
from DynaSysML.DE import odeint, defunc
import matplotlib.pyplot as plt

# 计算一个一维滑块系统 m\ddot x + kx=0, [x(0), \dot x(0)]\sim N(0, I)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# construct dataset
m, k, l = 1, 1, 1
X = torch.Tensor(2**14, 2).uniform_(-1, 1).to(device)
Xdd = -k*X[:, 0]/m

train = data.TensorDataset(X, Xdd)
trainloader = data.DataLoader(train, batch_size=64, shuffle=False)


class Learner(pl.LightningModule):
    def __init__(self, model):
        super(Learner, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        yhat = self.model(x)
        loss = F.mse_loss(yhat[:,-1], y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return trainloader


hdim = 128
net = LNN(nn.Sequential(
            nn.Linear(2,hdim),
            nn.Softplus(),
            nn.Linear(hdim,hdim),
            nn.Softplus(),
            nn.Linear(hdim,1))
         ).to(device)
learn = Learner(net)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(learn)


# plot figures
X0 = torch.Tensor(256, 2).uniform_(-1,1).to(device)
s_span = torch.linspace(0, 1, 100)
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
color = ['orange', 'blue']
func = defunc(net)
trajectory = odeint(func, X0, s_span)
# trajectory = model.trajectory(X0, s_span)
trajectory = trajectory.detach().cpu().numpy()
print(trajectory.shape)
for i in range(len(X0)):
    ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], color='blue', alpha=0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    ax.set_title('LNN learned trajectories')
plt.show()