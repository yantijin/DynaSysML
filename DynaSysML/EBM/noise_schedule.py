import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .utils import extract
from .vdm import reshape

class bddm_schedule(nn.Module):
    '''
    refer to Bilateral denoising diffusion models  http://arxiv.org/abs/2108.11514
    NOTE:
        (1) 需要在DDPM训练完之后再训练此网络
        (2) 注意训练时,不要直接用本网络的所有参数,只需要对sigma_phi的参数进行训练即可
    '''
    def __init__(self, sigma_phi, denoise_fn, T, tao, betas):
        super(bddm_schedule, self).__init__()
        self.T = T
        self.tao = tao
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))

        self.register_buffer("sqrt_cumprod_alpha",
                             torch.tensor(np.sqrt(np.cumprod(1. - betas, axis=0)), dtype=torch.float32))
        self.sigma_phi = sigma_phi

        self.denoise_fn = denoise_fn # NOTE:注意训练的时候这里参数不要加到optim中去,只加sigma_phi的参数
        for p in self.denoise_fn.parameters():
            p.requires_grad = False

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())

        return res

    def forward(self, x, *args, **kwargs):
        b, device = x.shape[0], x.device
        D = np.cumprod(x.shape[1:])[-1]
        shape = x.shape
        t = torch.randint(1, self.T-self.tao, (b,), device=device).long()
        # print(t.device, self.sqrt_cumprod_alpha.device, self.betas.device)
        alpha_n = extract(self.sqrt_cumprod_alpha, t, shape)
        beta_np1 = 1. - (extract(self.sqrt_cumprod_alpha, t+self.tao, shape) / alpha_n)**2
        delta_n = torch.sqrt(1. - alpha_n**2)
        noise = torch.randn_like(x)
        xn = alpha_n * x + delta_n * noise
        denoise_par = self.get_denoise_par(alpha_n, *args, **kwargs)
        epsilon_theta = self.denoise_fn(xn, *denoise_par)
        beta_n = torch.minimum(delta_n**2, beta_np1) * reshape(self.sigma_phi(xn), delta_n.shape)
        Cn = 0.25 * torch.log(delta_n**2/ beta_n) + 0.5 * (beta_n / delta_n**2 - 1.)
        # print(delta_n.shape, beta_n.shape, noise.shape, epsilon_theta.shape, beta_np1.shape, self.sigma_phi(xn).shape)
        term1 = 0.5 / (delta_n**2 - beta_n) * (delta_n * noise - beta_n / delta_n * epsilon_theta)**2
        # print(Cn.shape, term1.shape)
        loss1 = torch.mean(Cn.squeeze())
        # loss2 = torch.mean(torch.sum(term1, dim=[-i for i in range(1, len(shape))]))
        loss2 = torch.mean(term1)
        # return torch.mean(Cn.squeeze() + torch.sum(term1, dim=[-i for i in range(1, len(shape))]))
        return loss1, loss2

    @torch.no_grad()
    def noise_schedule(self, alpha, beta, shape, *args, **kwargs): # 用一个sample来求解
        beta_ls = [beta]
        x = torch.randn(shape, device=self.betas.device)
        for i in tqdm(reversed(range(1, self.T)), desc='get noise schedule', total=self.T-1):
            x = self.p_sample(x, alpha, beta, *args, **kwargs)
            alpha = alpha / torch.sqrt(1. - beta)
            beta = torch.minimum(1-alpha**2, beta) * torch.squeeze(self.sigma_phi(x))
            beta_ls.append(beta)
            if beta < self.betas[0] :
                return torch.tensor(list(reversed(beta_ls[:-1])), dtype=torch.float32, device=self.betas.device)
            if torch.isnan(beta):
                return torch.tensor(list(reversed(beta_ls[:-1])), dtype=torch.float32, device=self.betas.device)
        return torch.tensor(list(reversed(beta_ls)), dtype=torch.float32, device=self.betas.device)

    @torch.no_grad()
    def p_sample(self, x, alpha, beta, *args, **kwargs): # 这里x的batch可以设为1
        alpha_m1 = alpha / torch.sqrt(1. - beta)
        in_alpha = alpha * torch.full((x.shape[0],), 1., dtype=torch.float32, device=self.betas.device)
        denoise_par = self.get_denoise_par(in_alpha, *args, **kwargs)
        mean = 1. / torch.sqrt(1. - beta) * (x - beta/torch.sqrt(1-alpha**2) * self.denoise_fn(x, *denoise_par)) # TODO: 这里维度有问题
        std = torch.sqrt((1-alpha_m1**2)/ (1-alpha**2) * beta)
        noise = torch.randn_like(x)
        return mean + std * noise

    @torch.no_grad()
    def sample(self, shape, betas, *args, **kwargs):
        alpha = torch.cumprod(torch.sqrt(1. - betas), dim=-1)
        # alpha_all = torch.cat([torch.tensor([1.], device=self.betas.device, dtype=torch.float32), alpha], dim=-1)
        x = torch.randn(shape, device=self.betas.device)

        for i in tqdm(reversed(range(len(betas)))):
            x = self.p_sample(x, alpha[i], betas[i], *args, **kwargs)
        return x
