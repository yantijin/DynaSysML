from DynaSysML.EBM.PriorGrad import PriorGrad
import torch
import math
import numpy as np
import torch.nn as nn
from unet import Unet, Cond_Unet
from utils import plot_fig, get_mnist_loaders
from tqdm import tqdm
from torchvision.utils import save_image

# 需要一个cond_net,输出为mu(c), logstd(c)
# 需要一个denoise_fn，输入为x_t, t，输出为x_{t-1}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class cond_Net(nn.Module):
    def __init__(self, in_dim):
        super(cond_Net, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = in_dim**2
        self.l = nn.Embedding(10, self.emb_dim//2)
        self.l1 = nn.Sequential(
            nn.Linear(self.emb_dim//2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim*2),
            nn.Tanh()
        )
        self.mean = nn.Linear(self.emb_dim*2, self.emb_dim)
        self.logstd = nn.Linear(self.emb_dim*2, self.emb_dim)

    def forward(self, x):
        x = self.l(x)
        h = self.l1(x)
        mean = self.mean(h).reshape(-1, 1, self.in_dim, self.in_dim)
        logstd = self.logstd(h).reshape(-1, 1, self.in_dim, self.in_dim)
        return mean, logstd

class cond_Net1(nn.Module):
    def __init__(self, in_dim):
        super(cond_Net1, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = in_dim**2
        self.l = nn.Embedding(10, self.emb_dim//2)
        self.l1 = nn.Sequential(
            nn.Linear(self.emb_dim//2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim*2),
            nn.Tanh()
        )
        self.mean = nn.Linear(self.emb_dim*2, self.emb_dim)
        # self.logstd = nn.Linear(self.emb_dim*2, self.emb_dim)

    def forward(self, x):
        x = self.l(x)
        h = self.l1(x)
        mean = self.mean(h).reshape(-1, 1, self.in_dim, self.in_dim)
        # logstd = self.logstd(h).reshape(-1, 1, self.in_dim, self.in_dim)
        logstd = torch.zeros_like(mean)
        return mean, logstd


def train(batch=128, epochs=10):
    # denoise_fn = Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    denoise_fn = Cond_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    cond_fn = cond_Net(in_dim=28).cuda()
    # cond_fn = cond_Net1(in_dim=28).cuda()
    betas = np.linspace(1e-4, 5*1e-2, 200)
    model = PriorGrad(denoise_fn, betas, cond_fn, loss_type='l1').cuda()
    train_loader, test_loader, val_loader = get_mnist_loaders(batch_size=batch)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, label in it:
                optim.zero_grad()
                model.get_cond_par(label.to(device))
                loss = model(x.to(device), cond=label.to(device))
                loss.backward()
                it.set_postfix(
                    ordered_dict={
                        "epoch": epoch,
                        'loss': loss.item()
                    }
                )
                optim.step()


    # 生成过程
    shape = (100, 1, 28, 28)
    label = torch.tensor(list(range(10))*10).long().to(device)
    model.eval()
    model.get_cond_par(label.to(device))
    img1 = model.mu
    img2 = torch.exp(model.logstd)
    img = model.sample(shape, cond=label.to(device))
    save_image(img, '../figures/priorgrad.jpg', nrow=10)
    save_image(img1, '../figures/priorgrad_mu.jpg', nrow=10)
    save_image(img2, '../figures/priorgrad_std.jpg', nrow=10)


if __name__ == "__main__":
    # train_loader, test_loader, val_loader = get_mnist_loaders()
    # flag = 0
    # for x, label in train_loader:
    #     print(x.shape, label.shape, x.device, label.device)
    #     flag += 1
    #     if flag > 1:
    #         break
    train(batch=128, epochs=20)