import torch
import numpy as np
from tqdm import tqdm
from unet import c_Gen, c_Dis
from DynaSysML.EBM.DPMGAN import diffusion_gan
from utils import plot_fig, get_mnist_loaders
from torchvision.utils import save_image

def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
timesteps = 4
batch = 100
epochs = 30
z_dim=20
gamma=0.999
# betas = np.linspace(1e-4, 1e-2, timesteps)
betas = np.array([vpsde_beta_t(t, timesteps, 0.1, 20) for t in range(1, timesteps+1)])
print(betas)
denoise_fn = c_Gen(dim=16, dim_mults=(1,2,4), channels=1, latent_dim=z_dim).to(device)
dis = c_Dis(dim=16, channels=2, dim_mults=(1,2,4)).to(device)
gan = diffusion_gan(dis, denoise_fn, betas=betas, z_dim=z_dim, device=device)

gen_optim = torch.optim.Adam(denoise_fn.parameters(), lr=2*1e-4, betas=(0.5, 0.9))
dis_optim = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.9))
sdlG = torch.optim.lr_scheduler.ExponentialLR(gen_optim, gamma)
sdlD = torch.optim.lr_scheduler.ExponentialLR(dis_optim, gamma)

# load data
train_loader, test_loader, val_loader = get_mnist_loaders(batch_size=batch, num=0, test_batch_size=batch)

step = 0
gloss, dloss = 0, 0
for i in range(epochs):
    with tqdm(train_loader) as it:
        for data, label in it:
            step += 1
            z = torch.randn(batch, z_dim).to(device)
            dloss = gan.train_dis_step(data.to(device), dis_optim, z, grad_penal=True, label=label.to(device))
            z = torch.randn(batch, z_dim).to(device)
            gloss = gan.train_gen_step(data.to(device), gen_optim, z, label=label.to(device))
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
        torch.save(denoise_fn.state_dict(), '../model/cdf_' + str(i+1) + '.pt')
        torch.save(dis.state_dict(), '../model/cdis_' + str(i+1) + '.pt')
        print('epoch reaches: ', i+1)
        shape = (100, 1, 28, 28)
        label = torch.tensor(list(range(10))*10).long().to(device)
        with torch.no_grad():
            # cond = torch.randn(batch, z_dim).to(device)
            res = gan.diffusion_gen.sample(shape, clip_denoised=True, repeat_noise=False,label=label).detach().cpu()
            save_image(res, '../figures/cdpmgan_' + str(i+1) +'.jpg', nrow=10)

        for i in range(len(gan.diffusion_gen.xs)):
            save_image(gan.diffusion_gen.xs[i], '../figures/cs_' + str(i+1) +'_dpmgan.jpg', nrow=10)
        for i in range(len(gan.diffusion_gen.xo_pred_ls)):
            save_image(gan.diffusion_gen.xo_pred_ls[i], '../figures/cxo_' + str(i+1) +'_dpmgan.jpg', nrow=10)
        gan.diffusion_gen.xs = []
        gan.diffusion_gen.xo_pred_ls = []
        step = 1
        # with torch.no_grad():
        for data, _ in test_loader:
            if step > 1:
                break

            data = data.to(device)
            forward_state_ls = [data]
            save_image(forward_state_ls[-1], '../figures/cfd_0.jpg', nrow=10)

            for i in range(1, timesteps):
                forward_state_ls.append(gan.diffusion_gen.q_sample(data, t=torch.full((batch,), i, device=device, dtype=torch.long)))
                save_image(forward_state_ls[-1], '../figures/cfd_' + str(i) + '.jpg', nrow=10)


            img = forward_state_ls[-1].detach()
            rev_state_ls = [img]
            rev_state_ls1 = [img]
            mn_data_ls1 = []
            mn_data_ls = []
            save_image(rev_state_ls[-1], '../figures/cori_deter_3.jpg', nrow=10)
            save_image(rev_state_ls1[-1], '../figures/cori_sto_3.jpg', nrow=10)
            z = torch.randn(batch, z_dim).to(device)

            for i in reversed(range(0, timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
                img = gan.diffusion_gen.p_sample(img, torch.full((batch,), i, device=device, dtype=torch.long),clip_denoised=True, repeat_noise=False, label=label)# cond=z)
                rev_state_ls.append(img)
                save_image(rev_state_ls[-1], '../figures/cori_deter_' + str(i) +'.jpg', nrow=10)

            img = forward_state_ls[-1].detach()
            for i in reversed(range(0, timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
                z = torch.randn(batch, z_dim).to(device)
                img = gan.diffusion_gen.p_sample(img, torch.full((batch,), i, device=device, dtype=torch.long), clip_denoised=True, repeat_noise=False, label=label)#cond=z)
                rev_state_ls1.append(img)
                save_image(rev_state_ls1[-1], '../figures/cori_sto_' + str(i) + '.jpg', nrow=10)
            step += 1


# load model
# denoise_fn.load_state_dict(torch.load('../model/cdf_5.pt'))
# dis.load_state_dict(torch.load('../model/cdis_5.pt'))
# gan = diffusion_gan(dis, denoise_fn, betas=betas, device=device)
# label = torch.tensor(list(range(10))*10).long().to(device)
# step = 1
# # with torch.no_grad():
# for data, _ in test_loader:
#     if step > 1:
#         break
#
#     data = data.to(device)
#     forward_state_ls = [data]
#     save_image(forward_state_ls[-1], '../figures/fd_0.jpg', nrow=10)
#
#     for i in range(1, timesteps):
#         forward_state_ls.append(gan.diffusion_gen.q_sample(data, t=torch.full((batch,), i, device=device, dtype=torch.long)))
#         save_image(forward_state_ls[-1], '../figures/fd_' + str(i) + '.jpg', nrow=10)
#
#
#     img = forward_state_ls[-1].detach()
#     rev_state_ls = [img]
#     rev_state_ls1 = [img]
#     mn_data_ls1 = []
#     mn_data_ls = []
#     save_image(rev_state_ls[-1], '../figures/cori_deter_3.jpg', nrow=10)
#     save_image(rev_state_ls1[-1], '../figures/cori_sto_3.jpg', nrow=10)
#     z = torch.randn(batch, z_dim).to(device)
#
#     # for i in reversed(range(0, timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
#     #     img = gan.diffusion_gen.p_sample(img, torch.full((batch,), i, device=device, dtype=torch.long), label)
#     #     rev_state_ls.append(img)
#     #     save_image(rev_state_ls[-1], '../figures/cori_deter_' + str(i) +'.jpg', nrow=10)
#
#     img = forward_state_ls[-1].detach()
#     for i in reversed(range(0, timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
#         z = torch.randn(batch, z_dim).to(device)
#         img = gan.diffusion_gen.p_sample(img, torch.full((batch,), i, device=device, dtype=torch.long), clip_denoised=True, repeat_noise=False, label=label)
#         rev_state_ls1.append(img)
#         save_image(rev_state_ls1[-1], '../figures/cori_sto_' + str(i) + '.jpg', nrow=10)
#
#     for i in range(len(rev_state_ls)):
#         mn_data_ls.append(rev_state_ls[i] - forward_state_ls[timesteps-1-i])
#         mn_data_ls1.append(rev_state_ls1[i] - forward_state_ls[timesteps-1-i])
#
#     # for i in range(timesteps):
#     #     save_image(mn_data_ls1[i], '../figures/cmn1_' + str(timesteps-1-i) + '.jpg', nrow=10)
#     #     save_image(mn_data_ls[i], '../figures/cmn_' + str(timesteps - 1 - i) + '.jpg', nrow=10)
#
#     # z = torch.randn(batch, z_dim).to(device)
#     # print(gan.gloss(data, cond=z), gan.dloss(data, cond=z))
#     step += 1