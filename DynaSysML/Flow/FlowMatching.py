import abc
import numpy as np
import torch
from DynaSysML.Flow.utils import extract_coeff

__all__ = [
    'vector_field', 
    'VPSDE_vector_field', 'VESDE_vector_field', 'OT_vector_field',
    'CFM_vector_field', 'SB_CFM_vector_field', 'stochastic_interpolants'
]

# u_t(x|x_1) = \frac{\sigma_t'}{\sigma_t}(x-\mu_t) + \mu_t'
# 注意，在使用diffusion模型的时候，不同点在于flow matching在t=1时为数据，t=0时为噪声；
# 而diffusion则恰恰相反
class vector_field(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_x_t(self, x, x_0, t):
        pass

    @abc.abstractmethod
    def get_vf_t(self, x_t, x_1, t, x_0=None):
        pass

# see Flow Matching
class VPSDE_vector_field(vector_field):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min/N, beta_max/N, N)
        self.alphas = 1. - self.discrete_betas

    def get_mean_std(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean_coeff, std

    # 注意，这里输入t，对应的x_t相当于diffusion里面的x_{1-t},所以下面按照1-t来计算
    def get_x_t(self, x, x_0, t):
        mean_coeff, std = self.get_mean_std(1-t)
        if isinstance(x, torch.Tensor):
            noise = torch.randn(x.shape).to(x.device)
        else:
            noise = np.random.randn(*x.shape)
        return extract_coeff(mean_coeff, x.shape) * x + noise * extract_coeff(std, x.shape)

    # 注意这里对应原文中的T(t)
    def _compute_T(self, t):
        return self.beta_0 * t + 0.5 * t ** 2 * (self.beta_1 - self.beta_0)
    # 计算T'(x)
    def _compute_T_derivative(self, t):
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        T = extract_coeff(self._compute_T(1-t), x_t.shape)
        T_der = extract_coeff(self._compute_T_derivative(1-t), x_t.shape)
        scalar = -0.5 * T_der
        num = np.exp(-T) * x_t - np.exp(-0.5 * T) * x_1
        den = 1 - np.exp(-T)
        u_t = scalar * num / den
        return u_t

# see Flow Matching
class VESDE_vector_field(vector_field):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    def get_mean_std(self, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return 1, std

    def get_x_t(self, x, x_0, t):
        mean_coeff, std = self.get_mean_std(1-t)
        if isinstance(x, torch.Tensor):
            noise = torch.randn(x.shape).to(x.device)
        else:
            noise = np.random.randn(*x.shape)
        return extract_coeff(mean_coeff, x.shape) * x + noise * extract_coeff(std, x.shape)

    def _compute_std_der(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t * np.log(self.sigma_max / self.sigma_min)

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        _, std_1_minus_t = self.get_mean_std(1-t)
        std_1_minus_t_der = self._compute_std_der(1-t)
        u_t = - extract_coeff(std_1_minus_t_der / std_1_minus_t, x_t.shape) * (x_t - x_1)
        return u_t

# see Flow Matching
class OT_vector_field(vector_field):
    def __init__(self, sigma_min):
        super().__init__()
        self.sigma_min = sigma_min

    def get_mean_std(self, t):
        mean_coeff = t
        std = 1 - (1 - self.sigma_min) * t
        return mean_coeff, std

    def get_x_t(self, x, x_1, t):
        mean_coeff, std = self.get_mean_std(t)
        if isinstance(x, torch.Tensor):
            noise = torch.randn(x.shape).to(x.device)
        else:
            noise = np.random.randn(*x.shape)
        return extract_coeff(mean_coeff, x.shape) * x + noise * extract_coeff(std, x.shape)

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        num = x_1 - (1 - self.sigma_min) * x_t
        den = 1 - (1 - self.sigma_min) * extract_coeff(t, x_t.shape)
        u_t = num / den
        return u_t

# see conditional flow matching: https://arxiv.org/abs/2302.00482
class CFM_vector_field(vector_field):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def get_x_t(self, x, x_0, t):
        mu_x_t = extract_coeff(t, x.shape) * x + (1 - extract_coeff(t, x.shape)) * x_0
        sigma_t = self.sigma
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x)
        else:
            noise = np.random.randn(*x.shape)
        return mu_x_t + noise * sigma_t

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        assert x_0 != None
        return x_1 - x_0

# see conditional flow matching: https://arxiv.org/abs/2302.00482
class SB_CFM_vector_field(vector_field):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def get_mean_std(self, x, x_0, t):
        mu_x_t = extract_coeff(t, x.shape) * x + (1 - extract_coeff(t, x.shape)) * x_0
        sigma_t = extract_coeff(self.sigma ** 2 * t * (1 - t), x.shape)
        return mu_x_t, sigma_t

    def get_x_t(self, x, x_0, t):
        mu_x_t = extract_coeff(t, x.shape) * x + (1 - extract_coeff(t, x.shape)) * x_0
        sigma_t = self.sigma * torch.sqrt(t *(1-t))
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x)
        else:
            noise = np.random.randn(*x.shape)
        return mu_x_t + noise * extract_coeff(sigma_t, noise.shape)

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        '''
        :param x_0: 注意这里和上述vf不同之处在于不是用x_t，而是直接用x_0
        :param x_1: 目标分布的采样
        :param t: 对应的时间
        :param kwargs: 其他参数
        :return: vector_field在t时刻，给定输入的取值
        '''
        mu_x_t, _ = self.get_mean_std(x_1, x_0, t)
        ut = extract_coeff((1-2*t)/(2*t*(1-t)), x_0.shape) * (x_t - mu_x_t) + (x_1 - x_0)
        return ut

# see building nf with stochastic interpolants https://arxiv.org/abs/2209.15571
class stochastic_interpolants(vector_field):
    def __init__(self):
        super().__init__()

    def get_x_t(self, x, x_0, t):
        return extract_coeff(torch.cos(0.5 * np.pi * t), x_0.shape) * x_0 + \
            extract_coeff(torch.sin(0.5 * np.pi * t), x.shape) * x

    def get_vf_t(self, x_t, x_1, t, x_0=None):
        assert x_0 != None
        return 0.5 * np.pi * (extract_coeff(torch.cos(0.5 * np.pi * t), x_1.shape) * x_1 -
                              extract_coeff(torch.sin(0.5 * np.pi * t), x_0.shape) * x_0)