import torch.nn as nn
from DynaSysML.EBM.sde_lib import *
from DynaSysML.EBM.sample import *
from DynaSysML.EBM.losses import *
from DynaSysML.EBM.utils import EMA, get_score_fn
from utils import get_mnist_loaders, plot_fig, write_to_file, read_file
from unet import cont_Unet
import ml_collections
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from DynaSysML.EBM.noise_schedule import bddm_schedule
import os

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

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


def config_mnist():
    config = ml_collections.ConfigDict()
    config.model = modeling = ml_collections.ConfigDict()
    modeling.beta_min = 0.1
    modeling.beta_max = 10
    modeling.num_scales = 1000

    config.train = training = ml_collections.ConfigDict()
    training.epochs = 10
    training.ema_decay = 0.999
    training.update_ema_every = 10

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'
    sampling.eps = 1e-3
    sampling.snr = 0.16

    config.shape = (36, 1, 28, 28)
    return config


def train(config, model, sde, train_loader):
    ema_model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=True, continuous=True, likelihood_weighting=False)
    ema = EMA(config.train.ema_decay)
    num = 0
    for epoch in range(config.train.epochs):
        with tqdm(train_loader) as it:
            for data, _ in it:
                num += 1
                optim.zero_grad()
                loss = loss_fn(model, data.cuda())
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )
                if num % config.train.update_ema_every == 0:
                    if num < 1000:
                        ema_model.load_state_dict(model.state_dict())
                    else:
                        ema.update_model_average(ema_model, model)


    # sample
    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)
    sampling_fn = get_pc_sampler(sde, config.shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=lambda x: x,
                                 snr=config.sampling.snr,
                                 eps=config.sampling.eps)

    samples, n = sampling_fn(model)
    samples = samples.detach().cpu().numpy()
    plot_fig(samples)
    # plt.show()
    return sde


train1 = False #True
train2 = False #True
load_res = False
schedule_epochs = 10
schedule_path = father_path + '/sc_vpsde.pkl'
path1= father_path + '/vpsde.pkl'
beta_path = father_path + '/beta_candi.txt'
config = config_mnist()
train_loader, _, _ = get_mnist_loaders()
model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
s_phi = sigma_phi().cuda()
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
if train1:
    train(config, model, sde, train_loader)
    torch.save(model.state_dict(), path1)
else:
    model.load_state_dict(torch.load(path1))


schedule = bddm_schedule(s_phi, denoise_fn=model, T=1000, tao=200, betas=sde.discrete_betas).cuda()
hat_alphan = torch.tensor(np.linspace(0.1, 0.9, 9), dtype=torch.float32, device='cuda:0')
hat_betan = torch.tensor(np.linspace(0.1, 0.9, 9), dtype=torch.float32, device='cuda:0')

if train2:
    optim1 = torch.optim.Adam(s_phi.parameters(), lr=1e-3)
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
            img = schedule.sample(shape=config.shape, betas=schedule_beta)
            fig = plot_fig(img.detach().cpu().numpy(), father_path + '/fig/fig_' + str(round(hat_alphan[i].item(),2)) + '_' + str(round(hat_betan[j].item(), 2)) + '_' + str(len(schedule_beta)) + '.jpg')
            # plt.imsave(father_path + '/fig/fig_' + str(i) + '_' + str(j) + '.jpg', fig)
            plt.close()
else:
    schedule_betas = read_file(beta_path).to('cuda:0')
    for i in range(schedule_betas.shape[0]):
        schedule_beta = schedule_betas[i]
        img = schedule.sample(shape=config.shape, betas=schedule_beta)
        fig = plot_fig(img.detach().cpu().numpy(), father_path + '/fig/fig_' + str(i) + '_' + str(i+10) + '.jpg')
        # plt.imsave(father_path + '/fig/fig_' + str(i) + '_' + str(i+10) + '.jpg', fig)
        plt.close()

