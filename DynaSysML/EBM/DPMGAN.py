import torch
from DynaSysML.EBM import gaussian_ddpm, noise_like, default, extract


class gaussian_ddpm_gen(gaussian_ddpm):
    def __init__(self, denoise_fn, betas, z_dim=100):
        super(gaussian_ddpm_gen, self).__init__(
            denoise_fn=denoise_fn,
            betas=betas
        )
        self.z_dim = z_dim
        self.xs = []
        self.xo_pred_ls = []

    def q_posterior_sample(self, x_start, x_t, t, repeat=False):
        b, device = x_start.shape[0], x_t.device
        mean, var, logvar = self.q_posterior(x_start, x_t, t)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        noise = noise_like(x_t.shape, device, repeat=repeat)
        x_t_1_pred = mean + nonzero_mask * (0.5 * logvar).exp() * noise
        return x_t_1_pred

    def q_sample(self, x_start, t, noise=None):
        zero_idx = t<0
        t[zero_idx] = 0

        noise = default(noise, lambda: torch.randn_like(x_start))
        sample = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
                 extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        sample[zero_idx] = x_start[zero_idx]
        return  sample


    def forward(self, x, *args, **kwargs):
        b, device = x.shape[0], x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # t = torch.full((b,), 1, device=device, dtype=torch.long)
        x_t = self.q_sample(x_start=x, t=t)
        x_t_1 = self.q_sample(x_start=x, t=t-1)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        x_0_pred = self.denoise_fn(x_t, *denoise_par)
        x_t_1_pred = self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)
        return x_0_pred, x_t, x_t_1, x_t_1_pred, t

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, *args, **kwargs):
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        denoise_par = tuple([denoise_par[0], torch.randn(x.shape[0], self.z_dim).to(x.device), *denoise_par[1:]])
        x_0_pred = self.denoise_fn(x, *denoise_par)

        if clip_denoised:
            x_0_pred.clamp_(-1., 1.)
        self.xo_pred_ls.append(x_0_pred)
        x_t_1 = self.q_posterior_sample(x_0_pred, x, t)
        self.xs.append(x_t_1)
        return x_t_1

    @torch.no_grad()
    def p_sample_loop(self, shape, *args, **kwargs):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        self.xs.append(img)

        for i in reversed(range(0, self.num_timesteps)):  # , desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), *args, **kwargs)
        return img



class diffusion_gan():
    def __init__(self, dis, denoise_fn, betas, gamma=0.05, z_dim=20, device='cpu'):
        self.dis = dis
        self.denoise_fn = denoise_fn
        self.diffusion_gen = gaussian_ddpm_gen(denoise_fn, betas, z_dim=z_dim).to(device)
        self.gamma = gamma

    def gloss(self, x, z, *args, **kwargs):
        x_0_pred, x_t, _, x_t_1_pred, t = self.diffusion_gen(x, z, *args, **kwargs)
        out = self.dis(x_t, x_t_1_pred, t, *args, **kwargs)
        # print(out.shape)
        gloss = -torch.mean(torch.log(out + 1e-8))
        return gloss

    def dloss(self, x, z, grad_penal=True, *args, **kwargs):
        x_0_pred, x_t, x_t_1, x_t_1_pred, t = self.diffusion_gen(x, z, *args, **kwargs)
        out1 = self.dis(x_t, x_t_1, t, *args, **kwargs)
        out2 = self.dis(x_t, x_t_1_pred, t, *args, **kwargs)
        dloss = -(torch.mean(torch.log(out1 + 1e-8)) + torch.mean(torch.log(1-out2+1e-8)))
        if grad_penal:
            dloss += self.gamma / 2 * torch.mean(self.gradient_panalty(x_t, x_t_1, t, *args, **kwargs))
        return dloss

    def gradient_panalty(self, x_t, x_t_1, t, *args, **kwargs):
        x_t_1.requires_grad_(True)
        out = self.dis(x_t, x_t_1, t, *args, **kwargs)
        # print(out.requires_grad, x_t_1.requires_grad)
        grad = torch.autograd.grad(out, x_t_1, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True)[0]
        return grad**2

    def train_gen_step(self, x, optim, z, *args, **kwargs):
        optim.zero_grad()
        gloss = self.gloss(x, z, *args, **kwargs)
        gloss.backward()
        optim.step()
        return gloss

    def train_dis_step(self, x, optim, z, *args, **kwargs):
        optim.zero_grad()
        dloss = self.dloss(x, z, *args, **kwargs)
        dloss.backward()
        optim.step()
        return dloss