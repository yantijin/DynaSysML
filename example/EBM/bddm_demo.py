import torch
import torch.nn as nn
import copy
import numpy as np
from DynaSysML.EBM.ddpm import gaussian_ddpm
from DynaSysML.EBM.utils import EMA
from DynaSysML.EBM.utils import extract
from unet import cont_Unet
from utils import plot_fig, get_mnist_loaders, write_to_file, read_file
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from DynaSysML.EBM.noise_schedule import bddm_schedule

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")


class bddm(gaussian_ddpm):
    def get_denoise_par(self, t, **kwargs):
        alpha_t = extract(torch.sqrt(self.alphas_cumprod), t, t.shape)
        return alpha_t
    
class sigma_phi(nn.Module):
    def __init__(self, ):
        super(sigma_phi, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.l = nn.Sequential(
            nn.Linear(64*9, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = self.l(conv3_out.view(conv3_out.shape[0], -1))
        return res


epochs = 20
schedule_epochs = 10
ema_decay = 0.9999
update_ema_every = 10
train = False
train1 = True
load_res = False
beta_path = father_path + '/beta.txt'
schedule_path = father_path + '/schedule.pkl'
path = father_path + '/bddm_par.pkl'


model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
s_phi = sigma_phi().cuda()
betas = np.linspace(1e-4, 1e-2, 1000)
diffusion = bddm(model, loss_type='l2', betas=betas).cuda()
ema_model = copy.deepcopy(model)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader, test_loader, val_loader = get_mnist_loaders()
ema = EMA(ema_decay)

if train:
    print('train is true')
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
    torch.save(model.state_dict(), path)
else:
    print('train is false')
    model.load_state_dict(torch.load(path))


# sample images  100
shape = (36, 1, 28, 28)
eta=0.
# res = diffusion.sample(shape)
# res = res.detach().cpu().numpy()
# res1 = diffusion.ddim_sample(shape, eta).detach().cpu().numpy()
# plot_fig(res, father_path+'/fig/1.jpg')
#
# plot_fig(res1, father_path+'/fig/2.jpg')


schedule = bddm_schedule(s_phi, ddpm_net=diffusion, T=1000, tao=200, betas=betas).cuda()
hat_alphan = torch.tensor(np.linspace(0.1, 0.9, 9), dtype=torch.float32, device='cuda:0')
hat_betan = torch.tensor(np.linspace(0.1, 0.9, 9), dtype=torch.float32, device='cuda:0')
optim1 = torch.optim.Adam(s_phi.parameters(), lr=1e-3)

if train1:
    for epoch in range(schedule_epochs):
        with tqdm(train_loader) as it:
            for x, _ in it:
                optim1.zero_grad()
                loss1, loss2 = schedule(x.cuda())
                loss_t = loss1 + loss2
                loss_t.backward()
                optim1.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss_t.item(),
                        'cn': loss1.item(),
                        'fs': loss2.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )
    torch.save(s_phi.state_dict(), schedule_path)
else:
    s_phi.load_state_dict(torch.load(schedule_path))

if not load_res:
    for i in range(len(hat_alphan)):
        for j in range(len(hat_betan)):
            schedule_beta = schedule.noise_schedule(alpha=hat_alphan[i], beta=hat_betan[j], shape=(1,1,28,28))
            write_to_file(schedule_beta.detach().cpu().numpy(), beta_path)
            img = schedule.sample(shape=shape, betas=schedule_beta)
            fig = plot_fig(img.detach().cpu().numpy(), father_path + '/fig/fig_' + str(round(hat_alphan[i].item(),2)) + '_' + str(round(hat_betan[j].item(), 2)) + '_' + str(len(schedule_beta)) + '.jpg')
            # plt.imsave(father_path + '/fig/fig_' + str(i) + '_' + str(j) + '.jpg', fig)
            plt.close()
else:
    schedule_betas = read_file(beta_path).to('cuda:0')
    for i in range(schedule_betas.shape[0]):
        schedule_beta = schedule_betas[i]
        img = schedule.sample(shape=shape, betas=schedule_beta)
        fig = plot_fig(img.detach().cpu().numpy(), father_path + '/fig/fig_' + str(i) + '_' + str(i+10) + '.jpg')
        # plt.imsave(father_path + '/fig/fig_' + str(i) + '_' + str(i+10) + '.jpg', fig)
        plt.close()






