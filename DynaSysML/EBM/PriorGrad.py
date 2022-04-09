import torch
import torch.nn.functional as F
import torch.nn as nn
from DynaSysML.EBM import gaussian_ddpm, noise_like, default, extract


class PriorGrad(gaussian_ddpm):
    def __init__(self, denoise_fn, betas, cond_fn, loss_type='l1'):
        super().__init__(denoise_fn=denoise_fn,
                         betas=betas,
                         loss_type=loss_type)
        self.cond_fn = cond_fn


    def get_cond_par(self, inputs, *args, **kwargs):
        self.mu, self.logstd = self.cond_fn(inputs, *args, **kwargs)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start) * torch.exp(self.logstd))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - self.mu) +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, *args, **kwargs)
        noise = noise_like(x.shape, device, repeat_noise) * torch.exp(self.logstd)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, *args, **kwargs):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)*torch.exp(self.logstd)

        for i in reversed(range(0, self.num_timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),  clip_denoised=True,
                                repeat_noise=False, *args, **kwargs)

        img += self.mu
        return img

    def loss(self, x_start, t, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start) * torch.exp(self.logstd))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_recon = self.denoise_fn(x_noisy, *denoise_par)
        if self.loss_type == "l1":
            # loss = F.l1_loss(x_recon, noise)
            loss = torch.mean(torch.abs((x_recon - noise) / torch.exp(self.logstd)))
        elif self.loss_type == "l2":
            # loss = F.mse_loss(x_recon, noise)/torch.exp(self.logstd)**2
            loss = torch.mean((x_recon - noise) / torch.exp(self.logstd)**2 * (x_recon - noise))
        else:
            raise NotImplementedError()
        return loss

