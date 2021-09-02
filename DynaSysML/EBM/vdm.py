import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from .utils import extract, noise_like


'''
refer to variational diffusion models  https://arxiv.org/abs/2107.00630
introduce learnable SNR and fourier features
'''
def reshape(out, x_shape):
    b = out.shape[0]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class tilde_gamma_eta(nn.Module):
    def __init__(self, in_dim):
        super(tilde_gamma_eta, self).__init__()
        self.l1 = nn.Linear(in_dim, 1)
        self.l2 = nn.Linear(1, 1024)
        self.l3 = nn.Linear(1024, 1)
        self._init_weight()

    def forward(self, x):
        o1 = self.l1(x) + self.l3(F.sigmoid(self.l2(self.l1(x))))
        return o1

    def _init_weight(self):
        self.l1.weight.data.uniform_(0, 1)
        self.l2.weight.data.uniform_(0, 1)
        self.l3.weight.data.uniform_(0, 1)
        self.l1.bias.data.zero_()
        self.l2.bias.data.zero_()
        self.l3.bias.data.zero_()

    def reset_parameters(self):
        ls = [self.l1, self.l2, self.l3]
        for i in range(len(ls)):
            ls[i].weight.data = (ls[i].weight.data>0)*ls[i].weight.data + (ls[i].weight.data<0) * torch.ones(ls[i].weight.shape, dtype=torch.float32, device=ls[i].weight.device) * 1e-6
            ls[i].bias.data = (ls[i].bias.data>0)*ls[i].bias.data + (ls[i].bias.data<0) * torch.ones(ls[i].bias.shape, device=ls[i].weight.device, dtype=torch.float32)*1e-6

def tilde_gamma_eta_approx(t):
    gamma = torch.log(torch.exp(1e-4 + 10. * t**2) - 1)
    return gamma

class monotonic_net(nn.Module):
    '''
    calculate -log(snr(t))
    '''
    def __init__(self, in_dim, min_value, max_value):
        super(monotonic_net, self).__init__()
        self.gamma_eta = tilde_gamma_eta(in_dim)
        # self.gamma_eta = tilde_gamma_eta_approx
        self.gamma_0 = - torch.tensor(np.log(max_value))
        self.gamma_1 = - torch.tensor(np.log(min_value))
        self.in_dim = in_dim

    def forward(self, x):
        gamma_0 = self.gamma_eta(torch.zeros(x.shape, dtype=torch.float32).to(x.device))
        gamma_1 = self.gamma_eta(torch.ones(x.shape, dtype=torch.float32).to(x.device))
        gamma_t = self.gamma_eta(x)
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (gamma_t - gamma_0) / (gamma_1 - gamma_0)
        return gamma


    def reset_parameters(self):
        self.gamma_eta.reset_parameters()


class gau_vdm_ddpm(nn.Module):
    def __init__(self, denoise_fn, snr_min, snr_max, num_steps, device=None, in_dim=1):
        super(gau_vdm_ddpm, self).__init__()
        self.denoise_fn = denoise_fn
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.num_steps = num_steps
        self.gamma = monotonic_net(in_dim, snr_min, snr_max)
        self.device = device


    def loss(self, x):
        self.gamma.reset_parameters()
        b = x.shape[0]
        t = torch.randint(1, self.num_steps+1, (b,), device=x.device).long()
        noise = noise_like(x.shape, device=self.device, repeat=False)
        snr_t = reshape(torch.exp(-self.gamma((t/self.num_steps).unsqueeze(-1))), x.shape)
        alpha_t_2 = snr_t / (1. + snr_t)
        sigma_t_2 = 1. / (1. + snr_t)
        z = torch.sqrt(alpha_t_2) * x + torch.sqrt(sigma_t_2) * noise
        epsilon_theta = self.denoise_fn(z, t)
        snr_s = reshape(torch.exp(-self.gamma(((t-1.)/self.num_steps).unsqueeze(-1))), x.shape)
        loss = self.num_steps / 2 * (snr_s - snr_t) / snr_t * (epsilon_theta - noise) ** 2
        loss = torch.mean(loss)
        return loss

    def forward(self, x):
        return self.loss(x)

    # def get_par(self, t, shape):
    #     gamma_s = reshape(self.gamma(((t-1.)/self.num_steps).unsqueeze(-1)), shape)
    #     gamma_t = reshape(self.gamma((t / self.num_steps).unsqueeze(-1)), shape)
    #     sigma_ts_2 = - torch.exp(F.softplus(gamma_s) - F.softplus(gamma_t)) + 1. # sigma_t|s^2
    #     snr_t = torch.exp(-gamma_t)
    #     snr_s = torch.exp(-gamma_s)
    #     alpha_t_2 = snr_t / (1. + snr_t)
    #     sigma_t_2 = 1. / (1. + snr_t)
    #     sigma_s_2 = 1. / (1. + snr_s)
    #     alpha_s_2 = snr_s / (1. + snr_s)
    #     alpha_t_s = torch.sqrt(alpha_t_2 / alpha_s_2) # alpha_t|s
    #     return sigma_ts_2, sigma_t_2, sigma_s_2, alpha_t_s
    #
    # @torch.no_grad()
    # def p_sample(self, x, t,):
    #     noise = self.denoise_fn(x, t)
    #     sigma_ts_2, sigma_t_2, sigma_s_2, alpha_t_s = self.get_par(t, x.shape)
    #     mu_q = x / alpha_t_s - sigma_ts_2 / (alpha_t_s * torch.sqrt(sigma_t_2)) * noise
    #     sigma_q = torch.sqrt(sigma_ts_2 * sigma_s_2 / sigma_t_2)
    #     z = noise_like(x.shape, device=self.device, repeat=False)
    #     return mu_q + z * sigma_q
    #
    # @torch.no_grad()
    # def sample(self, shape):
    #     b = shape[0]
    #     img = torch.randn(shape, device=self.device)
    #     for i in tqdm(reversed(range(1, self.num_steps+1)), desc='sampling loop time step', total=self.num_steps):
    #         img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long))
    #     return img

    def get_total_par(self):
        t = torch.tensor(np.linspace(1, self.num_steps + 1, self.num_steps), device=self.device).long()
        gamma_s = self.gamma(((t-1.)/self.num_steps).unsqueeze(-1)).squeeze()
        gamma_t = self.gamma((t / self.num_steps).unsqueeze(-1)).squeeze()
        self.sigma_ts_2 = - torch.exp(F.softplus(gamma_s) - F.softplus(gamma_t)) + 1. # sigma_t|s^2
        snr_t = torch.exp(-gamma_t)
        snr_s = torch.exp(-gamma_s)
        alpha_t_2 = snr_t / (1. + snr_t)
        self.sigma_t_2 = 1. / (1. + snr_t)
        self.sigma_s_2 = 1. / (1. + snr_s)
        alpha_s_2 = snr_s / (1. + snr_s)
        self.alpha_t_s = torch.sqrt(alpha_t_2 / alpha_s_2) # alpha_t|s

    @torch.no_grad()
    def p_sample_s(self, x, t):
        noise = self.denoise_fn(x, t)
        mu_q = x / extract(self.alpha_t_s, t-1, x.shape) - \
               extract(self.sigma_ts_2, t-1, x.shape) / extract((self.alpha_t_s * torch.sqrt(self.sigma_t_2)), t-1, x.shape) * noise
        sigma_q = extract(torch.sqrt(self.sigma_ts_2 * self.sigma_s_2 / self.sigma_t_2), t-1, x.shape)
        z = noise_like(x.shape, device=self.device, repeat=False)
        return mu_q + z * sigma_q

    @torch.no_grad()
    def sample(self, shape):
        self.get_total_par()
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(1, self.num_steps + 1)), desc='sampling loop time step', total=self.num_steps):
            img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long))
        return img


