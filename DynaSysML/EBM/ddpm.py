import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial
from .utils import extract, default, noise_like

__all__ = [
    'gaussian_ddpm',
    'mix_gaussian_ddpm'
]

'''
refer to denoising diffusion probabilistic models. 
        https://arxiv.org/abs/2006.11239
'''

class gaussian_ddpm(nn.Module):
    '''
    refer to denoising diffusion probabilistic models.
        https://arxiv.org/abs/2006.11239
    '''
    def __init__(self, denoise_fn, betas, loss_type='l1'):
        super(gaussian_ddpm, self).__init__()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.denoise_fn = denoise_fn

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x_start, t, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_recon = self.denoise_fn(x_noisy, *denoise_par)
        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, *args, **kwargs):
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, *denoise_par))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, *args, **kwargs)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, *args, **kwargs):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):#, desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), *args, **kwargs)
        return img

    @torch.no_grad()
    def sample(self, shape, *args, **kwargs):
        return self.p_sample_loop(shape, *args, **kwargs)

    @torch.no_grad()
    def ddim_sample(self, shape, eta, *args, **kwargs):
        '''
        refer to denoising diffusion implicit models
        :param shape: sample shape
        :param eta: eta for control sigma_t
        :return:
        '''
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)): #, desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_t_1 = extract(self.alphas_cumprod_prev, t, x.shape)
            denoise_par = self.get_denoise_par(t, *args, **kwargs)
            et = self.denoise_fn(x, *denoise_par)
            x_recon = (x-torch.sqrt(1 -alpha_t) * et) / torch.sqrt(alpha_t)
            sigma_t = eta * torch.sqrt((1 - alpha_t / alpha_t_1) * (1 - alpha_t_1) / (1 - alpha_t))
            c2 = torch.sqrt(1 - alpha_t_1 - sigma_t ** 2)
            xt_recon = c2 * et
            noise = noise_like(x.shape, device)
            x = torch.sqrt(alpha_t_1) * x_recon + xt_recon + sigma_t * noise

        return x

    def forward(self, x, *args, **kwargs):
        b, *_ = x.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        loss = self.loss(x, t, *args, **kwargs)
        return loss


class mix_gaussian_ddpm(nn.Module):
    def __init__(self, denoise_fn, betas, phi_start, phi_end, p =0.5, loss_type='l1'):
        super(mix_gaussian_ddpm, self).__init__()


        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.denoise_fn = denoise_fn
        phis = phi_start + (phi_end - phi_start) * np.sqrt(alphas_cumprod)
        self.p = p
        m1 = np.sqrt((1 - phis ** 2) / (p * (1 - p) + p ** 3 / (1 - p) + 2 * p ** 2))
        m2 = - p / (1 - p) * m1

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("phis", to_torch(phis))
        self.register_buffer("m1", to_torch(m1))
        self.register_buffer("m2", to_torch(m2))
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def get_noise(self, t, shape):
        phis = extract(self.phis, t, shape)
        m1 = extract(self.m1, t, shape)
        m2 = extract(self.m2, t, shape)

        B1 = torch.distributions.Bernoulli(self.p)
        p = B1.sample()
        sample = p * (m1 + torch.randn(shape, device=m1.device) * phis) + (1 - p) * (m2 + torch.randn(shape, device=m2.device) * phis)
        return sample

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, self.get_noise(t, x_start.shape))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x, t, noise=None, *args, **kwargs):
        noise = default(noise, self.get_noise(t, x.shape))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_recon = self.denoise_fn(x_noisy, *denoise_par)
        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()
        return loss


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, *args, **kwargs):
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, *denoise_par))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, *args, **kwargs)
        # noise = noise_like(x.shape, device, repeat_noise)
        noise = self.get_noise(t, shape=x.shape)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, *args, **kwargs):
        device = self.betas.device

        b = shape[0]
        T = torch.full((b,), self.num_timesteps-1, device=device, dtype=torch.long)
        img = self.get_noise(T, shape)

        for i in reversed(range(0, self.num_timesteps)): #, desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), *args, **kwargs)
        return img

    @torch.no_grad()
    def sample(self, shape, *args, **kwargs):
        return self.p_sample_loop(shape, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        b, *_ = x.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        loss = self.loss(x, t, *args, **kwargs)
        return loss




