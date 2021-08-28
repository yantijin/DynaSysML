import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from unet import Unet, Cond_Unet
from DynaSysML.EBM.ddpm import gaussian_ddpm, cond_gaussian_ddpm, mix_gaussian_ddpm
from DynaSysML.EBM.utils import cosine_beta_schedule, EMA
import copy
import numpy as np
import matplotlib.pyplot as plt

def plot_fig(res):
    n = 6
    digit_size = 28
    figure = np.zeros((28 * n, 28 * n))
    for i in range(n):
        for j in range(n):
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = res[n * i + j, 0, :, :]

    plt.figure()
    plt.imshow(figure, cmap='gray')

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='../data/', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='../data/', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='../data/', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

tp = 'mix_gau_ddpm'

if tp == 'ddpm':
    epochs = 20
    ema_decay = 0.9999
    update_ema_every = 10

    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    # betas = cosine_beta_schedule(1000, )
    betas = np.linspace(1e-4, 1e-2, 1000)
    diffusion = gaussian_ddpm(model, loss_type='l2', betas=betas).cuda()
    ema_model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader, val_loader = get_mnist_loaders()
    ema = EMA(ema_decay)



    num = 0
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, label in it:
                num += 1
                optim.zero_grad()
                loss = diffusion(x.cuda())
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )

            if num % update_ema_every == 0:
                if num < 1000:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    ema.update_model_average(ema_model, model)


    # sample images  100
    shape = (36, 1, 28, 28)
    eta=0.
    res = diffusion.sample(shape)
    res = res.detach().cpu().numpy()
    res1 = diffusion.ddim_sample(shape, eta).detach().cpu().numpy()
    plot_fig(res)
    plot_fig(res1)
    plt.show()


elif tp == 'cond_ddpm':
    epochs = 20
    ema_decay = 0.999
    update_ema_every = 10


    model = Cond_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    # betas = cosine_beta_schedule(1000, )
    betas = np.linspace(1e-4, 1e-2, 1000)
    diffusion = cond_gaussian_ddpm(model, loss_type='l2', betas=betas).cuda()
    ema_model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader, val_loader = get_mnist_loaders()
    ema = EMA(ema_decay)

    num = 0
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, label in it:
                num += 1
                optim.zero_grad()
                loss = diffusion(x.cuda(), label.cuda())
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )

            if num % update_ema_every == 0:
                if num < 1000:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    ema.update_model_average(ema_model, model)

    # sample images
    shape = (36, 1, 28, 28)
    l = 5
    cond = (torch.ones(36) * l).cuda()
    eta = 0.
    res = diffusion.sample(shape, cond=cond)
    res = res.detach().cpu().numpy()
    res1 = diffusion.ddim_sample(shape, eta, cond).detach().cpu().numpy()
    plot_fig(res)
    plot_fig(res1)
    plt.show()


elif tp == 'mix_gau_ddpm':
    epochs = 20
    ema_decay = 0.9999
    update_ema_every = 10
    p=0.5

    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    # betas = cosine_beta_schedule(1000, )
    betas = np.linspace(1e-4, 1e-2, 1000)
    diffusion = mix_gaussian_ddpm(model, loss_type='l2', betas=betas, phi_start=1., phi_end=0.5, p=p).cuda()
    ema_model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader, val_loader = get_mnist_loaders()
    ema = EMA(ema_decay)

    num = 0
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, label in it:
                num += 1
                optim.zero_grad()
                loss = diffusion(x.cuda())
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )

            if num % update_ema_every == 0:
                if num < 1000:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    ema.update_model_average(ema_model, model)

    # sample images  100
    shape = (36, 1, 28, 28)
    eta = 0.
    res = diffusion.sample(shape)
    res = res.detach().cpu().numpy()
    # res1 = diffusion.ddim_sample(shape, eta).detach().cpu().numpy()
    plot_fig(res)
    # plot_fig(res1)
    plt.show()