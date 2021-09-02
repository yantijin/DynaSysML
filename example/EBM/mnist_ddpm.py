import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from unet import Unet, Cond_Unet
from DynaSysML.EBM.ddpm import gaussian_ddpm, mix_gaussian_ddpm
from DynaSysML.EBM.vdm import gau_vdm_ddpm
from DynaSysML.EBM.utils import cosine_beta_schedule, EMA, extract
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_fig, get_mnist_loaders



tp = 'cond_mix_gau_ddpm' #

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
                loss = diffusion(x.cuda(), cond=label.cuda())
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
    res1 = diffusion.ddim_sample(shape, eta, cond=cond).detach().cpu().numpy()
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


elif tp == 'cond_mix_gau_ddpm':
    epochs = 20
    ema_decay = 0.999
    update_ema_every = 10
    p=0.5


    model = Cond_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
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
                loss = diffusion(x.cuda(), cond=label.cuda())
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
    # res1 = diffusion.ddim_sample(shape, eta, cond=cond).detach().cpu().numpy()
    plot_fig(res)
    # plot_fig(res1)
    plt.show()


elif tp == 'snr_gaussian_ddpm':
    epochs = 10
    ema_decay = 0.999
    update_ema_every = 10
    p = 0.5
    SNR_min = 4.5397 * 1e-5#1 / (1-(1-1e-2)**2) - 1
    SNR_max = 9998.3408 #1 / (1-(1-1e-4)**2) - 1

    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    # diffusion = gaussian_vdm_ddpm(denoise_fn=model,
    #                               SNR_min=SNR_min,
    #                               SNR_max=SNR_max,
    #                               num_steps=1000,
    #                               device='cuda:0').cuda()
    diffusion = gau_vdm_ddpm(denoise_fn=model,
                             snr_min=SNR_min,
                             snr_max=SNR_max,
                             num_steps=1000,
                             device='cuda:0').cuda()
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


    # sample
    shape = (36, 1, 28, 28)
    res = diffusion.sample(shape)
    res = res.detach().cpu().numpy()
    # diffusion.get_total_par()
    # res2 = diffusion.sample_1(shape).detach().cpu().numpy()
    plot_fig(res)
    # plot_fig(res2)
    plt.show()
