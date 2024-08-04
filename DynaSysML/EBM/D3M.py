import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    "beta_t",
    "D3M_base",
    "D3M_constant",
    "D3M_Linear",
    "D3M_StoInt"
]

# You can define your noise schema here.
# Note that the design of beta_t should satisfy the contraint in Table 1.
def beta_t(t, beta_type='sqrt'):
    if beta_type == 'sqrt':
        return t.sqrt()
    elif beta_type == 'linear': # equivalent to rectified flow if h(t) is constant
        return t
    elif beta_type == 'square':
        return t ** 2
    elif beta_type == 'exp':
        return (torch.exp(t) - 1) / (np.exp(1) - 1)
    elif beta_type == 'sin': # equivalent to stochastic interpolant if h(t) is trigonometric
        return torch.sin(np.pi / 2 * t)
    elif beta_type == 'flow_matching': # equivalent to flow matching if h(t) is constant
        return (1e-4 + (1 - 1e-4) * t) ** 2
    else:
        raise NotImplementedError


class D3M_base(nn.Module):
    '''
        D3M: Build Generative Models based on the Explicit Solution of Linear SDEs
        Note: the covariance of target distribution of this base function is an identity matrix.
        You can change the target distribution as N(\mu, \Sigma) by modifying a few lines.
    '''

    def __init__(self, net, beta, eps=1e-4, weight1=1, weight2=1):
        super().__init__()
        '''
        net: eps_theta
        beta: function\beta_t, used for N(\alpha_t x_0, \beta_t^2 I)
        eps: used for denoise sampling
        '''
        self.net = net
        self.eps = eps
        self.beta = beta
        self.weight1 = weight1
        self.weight2 = weight2

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def get_H_t(self, t, *psi, x_0=None):
        '''
            Get H_t at time t
        '''
        pass

    def get_psi(self, x_0, t=None):
        '''
            Get coefficients of h(t)
        '''
        pass

    def get_beta_t(self, t):
        '''
            Get noise at time t
        '''
        return self.beta(t)

    def get_noisy_x(self, x_0, t, noise):
        t = t.reshape(t.shape[0], *((1,) * (len(x_0.shape) - 1)))
        psi = self.get_psi(x_0=x_0)

        x_t = x_0 + self.get_H_t(t, *psi) + self.get_beta_t(t) * noise
        return x_t

    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],), ).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        psi = self.get_psi(x_0)
        psi_theta, eps_theta = self.net(x_t, *denoise_par)
        # loss_psi_terms = [F.mse_loss(psi_term, hat_psi_term) for psi_term, hat_psi_term in zip(psi, psi_theta)]
        # loss = sum(loss_psi_terms) + F.mse_loss(noise, eps_theta)
        loss_psi_terms = [(psi_term - hat_psi_term) ** 2 for
                          psi_term, hat_psi_term in zip(psi, psi_theta)]
        l1 = sum(loss_psi_terms)
        l2 = (noise - eps_theta) ** 2
        loss = self.weight1 * l1 + self.weight2 * l2
        return loss

    @torch.no_grad()
    def sample(self, shape, num_steps, device, denoise=True, clamp=True, *args, **kwargs):
        x = torch.randn(shape).to(device)
        x = self.sample_loop(x, num_steps, denoise=denoise, clamp=clamp, *args, **kwargs)
        return x

    def predict_xtm1_xt(self, xt, psi, noise, t, delta_t):
        '''
        Input:
            xt: the state at time t
            psi: coefficients list
            noise: noise learned at t
            t: integration time t
            s: Delta t
        Return:
            x_{t-1}: predicted x_{t-\\Delta t}
        '''
        t = t.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
        delta_t = delta_t.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))

        beta_t = self.get_beta_t(t)
        beta_t_s = self.get_beta_t(t - delta_t)

        mean = xt + self.get_H_t(t - delta_t, *psi) - self.get_H_t(t, *psi) - (beta_t ** 2 - beta_t_s ** 2) / beta_t * noise
        sigma = (beta_t_s / beta_t) * ((beta_t ** 2 - beta_t_s ** 2)).sqrt()
        eps = torch.randn_like(mean, device=xt.device)
        return mean + sigma * eps

    def pred_x_start(self, x_t, noise, psi, t):
        t = t.reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        x_0 = x_t - self.get_H_t(t, *psi) - self.get_beta_t(t) * noise
        return x_0

    def sample_loop(self, x, num_steps, denoise=True, clamp=True, *args, **kwargs):
        '''
            General sampling procedure, see Algorithm 2 in the paper.
            Note: this sampler is actually not used for samplng in the paper because it cannot guarantee
            that the constraints in Table 1 are satisfied.
            We use the constraint-satisfying sampler when h(t) takes different types. See appendix C in the paper.
        '''
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs,), device=x.device)
        for i, time_step in enumerate(time_steps):
            delta_t = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                delta_t = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            psi_theta, eps_theta = self.net(x, *denoise_par)

            x = self.predict_xtm1_xt(x, psi_theta, eps_theta, cur_time, delta_t)
            if clamp:
                x.clamp_(-1., 1.)
            cur_time = cur_time - delta_t

        return x


class D3M_constant(D3M_base):
    '''
        Note:
            When beta is `linear`, the probability path of D3M (Constant-Linear) is equivalent to Rectified Flow [ICLR'23]
            See https://arxiv.org/abs/2209.03003
    '''
    def __init__(self, net, beta, weight_fn=None, eps=1e-4, weight1=1, weight2=1):
        D3M_base.__init__(self, net, beta, eps=eps, weight1=weight1, weight2=weight2)
        self.weight_fn = weight_fn

    def get_H_t(self, t, *psi, x_0=None):
        C = psi[0]
        t = t.reshape(t.shape[0], *((1,) * (len(C.shape) - 1)))
        return t * C

    def get_psi(self, x_0, t=None):
        return (- x_0,)

    def sample_loop(self, x, num_steps, denoise=True, clamp=True, *args, **kwargs):
        '''
            Weighted version of sampler, see Algorithm 3 in Appendix C
        '''
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs,), device=x.device)
        for i, time_step in enumerate(time_steps):
            delta_t = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                delta_t = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            psi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, psi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)
            if self.weight_fn:
                w1 = self.weight_fn(time_steps[0].numpy())
                term = w1 * psi_theta[0] - (1 - w1) * x_0
                psi_new = (term, )
            else:
                # directly use estimated initial value for sampling
                psi_new = (-x_0,)

            x = self.predict_xtm1_xt(x, psi_new, eps_theta, cur_time, delta_t)
            cur_time = cur_time - delta_t

        return x


class D3M_Linear(D3M_base):

    def get_H_t(self, t, *psi, x_0=None):
        a, b = psi[0], psi[1]
        t = t.reshape(t.shape[0], *((1,) * (len(a.shape) - 1)))
        return a / 2 * t ** 2 + b * t

    def get_psi(self, x_0, t=None):
        '''
            You can set these two parameters to determine the forward process.
            Note:
                The settings should ensure the constraints in Table 1 are satisfied.
                If the settings for coefficients are changed, the sampling procedure in `sample_loop` should be
                changed accordingly.
        '''
        # In the paper, we set the first term of \psi the same as D3M(Constant-*).
        a = - x_0
        b = - x_0 / 2
        return (a, b)

    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],), ).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)

        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        psi_theta, eps_theta = self.net(x_t, *denoise_par)

        psi = self.get_psi(x_0)
        loss_psi_term = F.mse_loss(psi[0], psi_theta[0])
        loss = self.weight1 * loss_psi_term + self.weight2 * F.mse_loss(noise, eps_theta)
        return loss

    def pred_x_start(self, x_t, noise, psi, t):
        t = t.reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        if t[0] == 1.:
            den = 1 - t + self.eps
        else:
            den = 1 - t
        x_0 = (x_t - self.get_beta_t(t) * noise + psi[0] * t * (1 - t) / 2) / den
        return x_0

    def sample_loop(self, x, num_steps, denoise=True, clamp=True, *args, **kwargs):
        '''
            Constraint-satisfying sampler for linear type, see Algorithm 4 in Appendix C
        '''
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs,), device=x.device)
        for i, time_step in enumerate(time_steps):
            delta_t = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                delta_t = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            psi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, psi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)

            # NOTE: If the settings in `get_psi` are changed, do not forget to change this line accordingly
            # you can alternate following two lines and observe the diferences.
            # psi_new = (psi_theta[0], -x_0 / 2)
            psi_new = (psi_theta[0], -x_0 - psi_theta[0] / 2)

            x = self.predict_xtm1_xt(x, psi_new, eps_theta, cur_time, delta_t)
            cur_time = cur_time - delta_t

        return x

# The settings that make the probability path of D3M equivalent to stochastic interpolant [ICLR'23]
# https://arxiv.org/pdf/2209.15571v3
class D3M_StoInt(D3M_base):
    def __init__(self, net, beta, weight_fn=None, eps=1e-4, weight1=1, weight2=1):
        D3M_base.__init__(self, net, beta, eps=eps, weight1=weight1, weight2=weight2)
        self.weight_fn = weight_fn

    def get_psi(self, x_0, t=None):
        assert t is not None
        t = t.reshape(t.shape[0], *((1,) * (len(x_0.shape) - 1)))
        term = - np.pi / 2 * torch.sin(np.pi / 2 * t) * x_0
        return (term, )

    def get_H_t(self, t, *psi, x_0=None):
        assert x_0 is not None
        a = psi[0]
        t = t.reshape(t.shape[0], *((1,) * (len(a.shape) - 1)))
        if not x_0:
            # estimate x_0
            x_0 = - 2 / np.pi / torch.sin(np.pi / 2 * t) * a
        return (torch.cos(np.pi / 2 * t) - 1) * x_0

    def get_noisy_x(self, x_0, t, noise):
        t = t.reshape(t.shape[0], *((1,) * (len(x_0.shape) - 1)))
        psi = self.get_psi(x_0=x_0, t=t)

        x_t = x_0 + self.get_H_t(t, *psi, x_0=x_0) + self.get_beta_t(t) * noise
        return x_t

    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],), ).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)

        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        psi_theta, eps_theta = self.net(x_t, *denoise_par)

        psi = self.get_psi(x_0, t=t)
        loss_psi_term = F.mse_loss(psi[0], psi_theta[0])
        loss = self.weight1 * loss_psi_term + self.weight2 * F.mse_loss(noise, eps_theta)
        return loss

    def pred_x_start(self, x_t, noise, psi, t):
        t = t.reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        x_0 = x_t - self.get_H_t(t, *psi) - self.get_beta_t(t) * noise
        return x_0

    def sample_loop(self, x, num_steps, denoise=True, clamp=True, *args, **kwargs):
        '''
            Weighted incremental sampler for D3M <=> Stochastic-Interpolant
        '''
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs,), device=x.device)
        for i, time_step in enumerate(time_steps):
            delta_t = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                delta_t = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            psi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, psi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)
            if self.weight_fn:
                w1 = self.weight_fn(time_steps[0].numpy())
                term = w1 * psi_theta[0] + (1 - w1) * self.get_psi(x_0, cur_time)[0]
                psi_new = (term, )
            else:
                # directly use estimated initial value for sampling
                psi_new = self.get_psi(x_0, cur_time)

            x = self.predict_xtm1_xt(x, psi_new, eps_theta, cur_time, delta_t)
            cur_time = cur_time - delta_t

        return x