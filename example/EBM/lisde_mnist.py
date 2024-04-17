import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from unet import Unet, Cond_Unet
from DynaSysML.EBM.lisde import LISDE
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_fig, get_mnist_loaders
from torchvision.utils import save_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 10
sigma_min = 1e-3
eps = 1e-2
num_steps=100
denoise = True
train = False
save = False
model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).to(device)
diffusion = LISDE(model, sigma_min, eps=eps)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader, test_loader, val_loader = get_mnist_loaders()

num = 0
if train:
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, label in it:
                num += 1
                optim.zero_grad()
                loss = diffusion(x.to(device))
                # loss = torch.mean(loss)
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train_loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )
else:
    model.load_state_dict(torch.load('./lisde.pkl'))
    print('load model success')
if save:
    torch.save(model.state_dict(), './lisde.pkl')


# 下面是采样过程
shape = (36, 1, 28, 28)
res = diffusion.sample(shape, num_steps, device=device, denoise=denoise, clamp=True)
print(res[0,0])
save_image(res, '../figures/lisde.jpg', nrow=6)

