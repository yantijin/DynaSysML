import torch
import numpy as np
from tqdm import tqdm
from unet import Dis, Cond_logistic_gen
from DynaSysML.EBM.D3PMGAN import d3pm_gan
from utils import plot_fig, get_mnist_loaders
from torchvision.utils import save_image

def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
timesteps = 4
batch = 100
epochs = 20
z_dim=20
gamma=0.999
# betas = np.linspace(1e-4, 1e-2, timesteps)
betas = np.array([vpsde_beta_t(t, timesteps, 0.1, 20) for t in range(1, timesteps+1)])
print(betas)
denoise_fn = Cond_logistic_gen(dim=16, dim_mults=(1,2,4), channels=1, latent_dim=z_dim).to(device)
dis = Dis(dim=16, channels=2, dim_mults=(1,2,4)).to(device)
gan = d3pm_gan(dis, denoise_fn, betas=betas, device=device, gamma=0.3)

gen_optim = torch.optim.Adam(denoise_fn.parameters(), lr=2*1e-4, betas=(0.5, 0.9))
dis_optim = torch.optim.Adam(dis.parameters(), lr=1e-5, betas=(0.9, 0.9))
sdlG = torch.optim.lr_scheduler.ExponentialLR(gen_optim, gamma)
sdlD = torch.optim.lr_scheduler.ExponentialLR(dis_optim, gamma)

# load data
train_loader, test_loader, val_loader = get_mnist_loaders(batch_size=batch, num=0, test_batch_size=batch)

step = 0
gloss, dloss = 0, 0

for i in range(2):
    with tqdm(train_loader) as it:
        for data, _ in it:
            step += 1
            data *= 255
            data = data.long().cuda()
            z = torch.rand(batch, z_dim).to(device)
            gloss = gan.train_gen_step(data, gen_optim, z)
            it.set_postfix(
                ordered_dict={
                    'gloss': gloss,
                    'dloss': dloss,
                    'epoch': i
                },
                refresh=False
            )
for i in range(epochs):
    with tqdm(train_loader) as it:
        for data, _ in it:
            step += 1
            data *= 255
            data = data.long().cuda()
            if step % 5 == 0:
                z = torch.rand(batch, z_dim).to(device)
                dloss = gan.train_dis_step(data, dis_optim, grad_penal=False, cond=z)
            else:
                z = torch.rand(batch, z_dim).to(device)
                gloss = gan.train_gen_step(data, gen_optim, cond=z)
            it.set_postfix(
                ordered_dict={
                    'gloss': gloss,
                    'dloss': dloss,
                    'epoch': i
                },
                refresh=False
            )
    sdlG.step()
    sdlD.step()

    if (i+1) % 5 == 0:
        torch.save(denoise_fn.state_dict(), '../model/3_df_' + str(i+1) + '.pt')
        torch.save(dis.state_dict(), '../model/3_dis_' + str(i+1) + '.pt')
        print('epoch reaches: ', i+1)
        shape = (100, 1, 28, 28)
        with torch.no_grad():
            cond = torch.rand(batch, z_dim).to(device)
            res = gan.diffusion_gen.p_sample_loop(denoise_fn, shape, cond=cond).detach().cpu()
            save_image(res/255., '../figures/d3pmgan_' + str(i+1) +'.jpg', nrow=10)

        for i in range(len(gan.diffusion_gen.xs)):
            save_image(gan.diffusion_gen.xs[i], '../figures/3s_' + str(i+1) +'_d3pmgan.jpg', nrow=10)
        for i in range(len(gan.diffusion_gen.xo_pred_ls)):
            save_image(gan.diffusion_gen.xo_pred_ls[i], '../figures/3xo_' + str(i+1) +'_d3pmgan.jpg', nrow=10)
        gan.diffusion_gen.xs = []
        gan.diffusion_gen.xo_pred_ls = []