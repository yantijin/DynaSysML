import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Beta

# 对epsNet进行包装，变成EDM中要求的形式
class WrapEpsNet(nn.Module):
    def __init__(self, eps_model, 
                 num_steps,
                 beta_min, beta_max,
                 sigma_min=None, sigma_max=None,
                 sigma_data      = 0.5,              # Expected standard deviation of the training data.
                 C_1             = 0.001,            # Timestep adjustment at low noise levels.
                 C_2             = 0.008,            # Timestep adjustment at high noise levels.
                 epsilon_t       = 1e-5,             # Minimum t-value used during training
                 diff_type       = 'VPSDE'
                 ):
        super().__init__()
        self.model = eps_model
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        self.C_2 = C_2
        self.diff_type = diff_type
        self.epsilon_t = epsilon_t
        if diff_type in ('EDM', 'VESDE'):
            assert sigma_min is not None
            assert sigma_max is not None
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
        elif diff_type == 'VPSDE':
            def get_sigma(t):
                t = torch.as_tensor(t)
                return ((0.5 * self.beta_d * t**2 + self.beta_min * t).exp() - 1).sqrt()
            self.sigma_min = float(get_sigma(epsilon_t))
            self.sigma_max = float(get_sigma(1))
        else:
            raise NotImplementedError

        # 给EDM用的
        self.sigma_data = sigma_data

        # 给iDDPM用的
        u = torch.zeros(num_steps + 1)
        for j in range(num_steps, 0, -1): # num_steps, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)

    # VPSDE 中 get_par 使用
    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    # 四个都用
    def round_sigma(self, sigma, return_index=False):
        if self.diff_type == 'iDDPM':
            sigma = torch.as_tensor(sigma)
            index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
            result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
            return result.reshape(sigma.shape).to(sigma.device)
        elif self.diff_type in ('VPSDE', 'VESDE', 'EDM'):
            return torch.as_tensor(sigma)

    # 给iDDPM用的
    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.num_steps / (self.C_2 + 1)).sin() ** 2

    # 四个都用
    def get_par(self, sigma):
        if self.diff_type == 'VPSDE':
            c_skip = 1
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_noise = (self.num_steps - 1) * self.sigma_inv(sigma)
        elif self.diff_type == 'VESDE':
            c_skip = 1
            c_out = sigma
            c_in = 1
            c_noise = (0.5 * sigma).log()
        elif self.diff_type == 'iDDPM':
            c_skip = 1
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_noise = self.num_steps - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        elif self.diff_type == 'EDM':
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = sigma.log() / 4
        else:
            raise NotImplementedError
        return c_skip, c_out, c_in, c_noise
    
    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res
    
    def forward(self, x, sigma, *args, **kwargs):
        if len(sigma.shape) != len(x.shape):
            sigma = sigma.reshape((len(sigma),) + (1,)* (len(x.shape) - 1))
        
        c_skip, c_out, c_in, c_noise = self.get_par(sigma=sigma)

        x_in = c_in * x
        denoise_par = self.get_denoise_par(c_noise.flatten(), *args, **kwargs)
        F_x = self.model(x_in, *denoise_par)
        D_x = c_skip * x + c_out * F_x
        return D_x

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand_beta_prime(self, size, N=3072, D=128, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).float()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)

            if N < 256 * 256 * 3:
                sigma_max = 80
            else:
                sigma_max = kwargs['sigma_max']

            sample_norm = torch.sqrt(inverse_beta) * sigma_max * np.sqrt(D)
            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

class PoissonVE(nn.Module):
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")
        
    
    def forward(self, wrap_net, inputs, *args, **kwargs):
        bs = inputs.shape[0]
        rnd_uniform = torch.rand(bs, device=inputs.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

        r = sigma.double() * np.sqrt(self.D).astype(np.float64)
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                      size=bs).astype(np.double)

        samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(inputs.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        
        # Uniformly sample the angle direction
        gaussian = torch.randn(bs, self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()
        
        sigma = sigma.reshape((len(sigma),)+ (1,) * (len(inputs.shape) - 1))
        noisy_samples = inputs + perturbation_x.view_as(inputs)
        D_yn = wrap_net(noisy_samples, sigma, *args, **kwargs)
        weight = 1 / sigma ** 2
        loss = weight * (D_yn - inputs) ** 2
        return loss


class PoissonVP(nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20., epsilon_t=1e-5):
        super().__init__()
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def forward(self, wrap_net, inputs, *args, **kwargs):
        rand_shape = (inputs.shape[0], ) + (1,) * (len(inputs.shape) - 1)
        rnd_uniform = torch.rand(rand_shape, device=inputs.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1)) # \sqrt{(1-\alpha_t)/ \alpha_t}
        weight = 1 / sigma ** 2
        n = torch.randn_like(inputs) * sigma
        D_yn = wrap_net(inputs + n, sigma, *args, **kwargs)
        loss = weight * ((D_yn - inputs) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    

class PoissonEDM(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, gamma=5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N
        self.gamma = gamma
        # self.opts = opts
        print(f"In EDM loss: D:{self.D}, N:{self.N}")
    
    def forward(self, wrap_net, inputs, *args, **kwargs):
        rnd_normal = torch.randn(inputs.shape[0], device=inputs.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # if self.opts.lsun:
        #     # use larger sigma for high-resolution datasets
        #     sigma *= 380. / 80

        r = sigma.double() * np.sqrt(self.D).astype(np.float64)
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                        size=inputs.shape[0]).astype(np.double)

        samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(inputs.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        # Uniformly sample the angle direction
        gaussian = torch.randn(inputs.shape[0], self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()

        sigma = sigma.reshape((len(sigma), 1, 1, 1))

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = perturbation_x.view_as(inputs)
        D_yn = wrap_net(inputs + n, sigma, *args, **kwargs)
        loss = weight * ((D_yn - inputs) ** 2)
        return loss


def ablation_sampler(
    wrap_net, latents, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, pfgmpp=False, *args, **kwargs
):
    
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    # diff_type_dict = {'vp': 'VPSDE', 've': 'VESDE', 'iddpm': 'iDDPM', 'edm': 'EDM'}

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d

    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, wrap_net.sigma_min)
    sigma_max = min(sigma_max, wrap_net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float32, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(wrap_net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    if pfgmpp:
        x_next = latents.to(torch.float32)
    else:
        x_next = latents.to(torch.float32) * (sigma(t_next) * s(t_next))
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(wrap_net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        # import pdb; pdb.set_trace()
        sig_in = sigma(t_hat).to(torch.float32).reshape(-1,).repeat(x_hat.shape[0]).reshape((x_hat.shape[0], ) + (1, ) * (len(x_hat.shape) - 1))
        denoised = wrap_net(x_hat / s(t_hat), sig_in, *args, **kwargs).to(torch.float32)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            sig_new_in = sigma(t_prime).to(torch.float32).reshape(-1,).repeat(x_hat.shape[0]).reshape((x_hat.shape[0], ) + (1, ) * (len(x_hat.shape) - 1))
            denoised = wrap_net(x_prime / s(t_prime), sig_new_in, *args, **kwargs).to(torch.float32)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next
