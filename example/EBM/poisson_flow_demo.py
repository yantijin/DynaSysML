import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import *
from DynaSysML.EBM.PoissonFlow import *
from tqdm import tqdm
from utils import plot_fig, get_mnist_loaders
from torchvision.utils import save_image


epochs = 10
beta_min = 0.1
beta_max = 20.
num_steps = 1000
sigma_data = 0.5
C_1 = 0.001
sigma_min = 0.02
sigma_max = 100
N = 28*28 # x的维度
D = 128 # 扩维的维度
epsilon_t = 1e-3

md_name = 'VE_PFGMPP'
TRAIN = True
if md_name == 'VP':
    # VPSDE 还是原来的形式，这里并不是PDGMPP的代码

    model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    md = WrapEpsNet(model, num_steps=num_steps, beta_min=beta_min,
                    beta_max=beta_max, sigma_data=sigma_data, C_1=C_1, diff_type='VPSDE',
                    epsilon_t=epsilon_t)
    poisson_vp = PoissonVP(beta_min=beta_min, beta_max=beta_max, epsilon_t=epsilon_t)
    optim = torch.optim.Adam(md.parameters(), lr=1e-3)
    train_loader, _, _ = get_mnist_loaders()

    if TRAIN:
        for epoch in range(epochs):
            with tqdm(train_loader) as it:
                for x, _ in it:
                    optim.zero_grad()
                    loss = poisson_vp(md, x.cuda()).mean()
                    loss.backward()
                    optim.step()
                    it.set_postfix(
                        ordered_dict={
                            'train_loss': loss.item(),
                            'epoch': epoch
                        },
                        refresh=False
                    )
        torch.save(model.state_dict(), '../pts/vpsde_base.pt')
        torch.save(md.state_dict(), '../pts/vpsde.pt')
    else:
        md.load_state_dict(torch.load('../pts/vpsde.pt'))
        print('load model success!')
    latents = torch.randn(36, 1, 28, 28).cuda()
    with torch.no_grad():
        sample = ablation_sampler(md, latents, num_steps=100, solver='euler',
                                   discretization='vp', schedule='vp',
                                   scaling='vp')

        # plot_fig(sample.detach().cpu().numpy())
        # plt.show()
        save_image(sample,'../figures/vp_pfgm.jpg', nrow=6)

elif md_name == 'VE_PFGMPP':
    model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    md = WrapEpsNet(model, num_steps=num_steps, beta_min=beta_min,
                    beta_max=beta_max, sigma_min=sigma_min, sigma_max=sigma_max,
                    sigma_data=sigma_data, C_1=C_1, diff_type='VESDE')
    poisson_ve = PoissonVE(sigma_min=sigma_min, sigma_max=sigma_max, D=D, N=N)
    optim = torch.optim.Adam(md.parameters(), lr=1e-3)
    train_loader, test_loader, val_loader = get_mnist_loaders()

    if TRAIN:
        for epoch in range(epochs):
            with tqdm(train_loader) as it:
                for x, _ in it:
                    optim.zero_grad()
                    loss = poisson_ve(md, x.cuda()).mean()
                    loss.backward()
                    optim.step()
                    it.set_postfix(
                        ordered_dict={
                            'train_loss': loss.item(),
                            'epoch': epoch
                        },
                        refresh=False
                    )
        torch.save(md.state_dict(), '../pts/vesde.pt')
    else:
        md.load_state_dict(torch.load('../pts/vesde.pt'))
        print('load model success!')
    # sampler
    seeds = list(range(1, 37))
    latent_sampler = StackedRandomGenerator('cuda:0', seeds)
    latent = latent_sampler.rand_beta_prime(size=(36, 1, 28, 28), N=N, D=D,
                                            device='cuda:0', sigma_max=sigma_max)
    # latent = torch.randn(36, 1, 28, 28).cuda()
    with torch.no_grad():
        sample = ablation_sampler(md, latent, num_steps=100, sigma_min=sigma_min, sigma_max=sigma_max,
                                  solver='heun', discretization='ve', schedule='ve', scaling='none',
                                  epsilon_s=epsilon_t, C_1=C_1, pfgmpp=True)
        # sample = sample.detach().cpu().numpy()
        save_image(sample, '../figures/VE_PFGMPP.jpg', nrow=6)

elif md_name == 'EDM':
    model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    md = WrapEpsNet(model, num_steps=num_steps, beta_min=beta_min,
                    beta_max=beta_max, sigma_min=0.002, sigma_max=80,
                    sigma_data=sigma_data, C_1=C_1, diff_type='EDM')
    poisson_edm = PoissonEDM( P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=D, N=N, gamma=5)
    optim = torch.optim.Adam(md.parameters(), lr=1e-3)
    train_loader, test_loader, val_loader = get_mnist_loaders()

    if TRAIN:
        for epoch in range(epochs):
            with tqdm(train_loader) as it:
                for x, _ in it:
                    optim.zero_grad()
                    loss = poisson_edm(md, x.cuda()).mean()
                    loss.backward()
                    optim.step()
                    it.set_postfix(
                        ordered_dict={
                            'train_loss': loss.item(),
                            'epoch': epoch
                        },
                        refresh=False
                    )
        torch.save(md.state_dict(), '../pts/edm_pfgmpp.pt')
    else:
        md.load_state_dict(torch.load('../pts/edm_pfgmpp.pt'))
        print('load model success!')
    # sampler
    seeds = list(range(1, 37))
    latent_sampler = StackedRandomGenerator('cuda:0', seeds)
    latent = latent_sampler.rand_beta_prime(size=(36, 1, 28, 28), N=N, D=D,
                                            device='cuda:0', sigma_max=80)
    # latent = torch.randn(36, 1, 28, 28).cuda()
    with torch.no_grad():
        sample = ablation_sampler(md, latent, num_steps=16, sigma_min=0.002, sigma_max=80,
                                  solver='heun', discretization='edm', schedule='linear', scaling='none',
                                  epsilon_s=epsilon_t, C_1=C_1, pfgmpp=True)
        # sample = sample.detach().cpu().numpy()
        save_image(sample, '../figures/EDM_PFGMPP.jpg', nrow=6)