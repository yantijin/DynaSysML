import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Callable, Tuple
from DynaSysML.Flow.inter_utils import *

Time = torch.tensor
Sample = torch.tensor
Velocity = nn.Module
Score = nn.Module


def compute_div(
    f: Callable[[Time, Sample], torch.tensor],
    x: torch.tensor,
    t: torch.tensor  # [batch x dim]
) -> torch.tensor:
    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched."""
    bs = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        divergence = 0.0
        for i in range(x.shape[1]):
            divergence += \
                    torch.autograd.grad(
                            f_val[:, i].sum(), x, create_graph=True
                        )[0][:, i]
    return divergence.view(bs)


class Interpolant(nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1) + \gamma(t)z.

    path: str,    what type of interpolant to use, e.g. 'linear' for linear interpolant. see fabrics for options
    gamma_type:   what type of gamma function to use, e.g. 'brownian' for $\gamma(t) = \sqrt{t(1-t)}
    """
    def __init__(self,
                 path: str,
                 gamma_type: str,
                 It: Callable[[Time, Sample, Sample], torch.Tensor] = None,
                 dtIt: Callable[[Time, Sample, Sample], torch.Tensor] = None
                 ) -> None:
        super(Interpolant, self).__init__()
        self.gamma, self.gamma_dot, self.gg_dot = make_gamma(gamma_type)
        if path == 'custom':
            print('Assuming interpolatn was passed in directly')
            self.It = It
            self.dtIt = dtIt
            assert self.It != None
            assert self.dtIt != None
        else:
            self.It, self.dtIt = make_It(path, self.gamma, self.gg_dot)

    def get_xt(self, t: Time, x0: Sample, x1: Sample):
        z = torch.randn_like(x0)
        gamma_t = self.gamma(t)
        It = self.It(t, x0, x1)
        return It + gamma_t * z, z

    # page 50
    def get_antithetic_xts(self, t: Time, x0: Sample, x1: Sample):
        z = torch.randn_like(x0)
        gamma_t = self.gamma(t)
        It = self.It(t, x0, x1)
        return It + gamma_t * z, It - gamma_t * z, z

    def forward(self, ):
        raise NotImplementedError("Interpolant does not need forward func")


class ProbabilityFlow(nn.Module):
    def __init__(self, v: Velocity, s: Score, interpolant: Interpolant, sample_only: bool = False):
        super(ProbabilityFlow, self).__init__()
        self.v = v
        self.s = s
        self.interpolant = interpolant
        self.sample_only = sample_only

    def set_vector_filed(self):
        def vector_field(x: torch.tensor, t: torch.tensor):
            self.v.to(x)
            self.s.to(x)
            return self.v(x, t) - self.interpolant.gg_dot(t) * self.s(x, t) # eq. 2.23

        self.vector_field = vector_field

    def forward(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.vector_field(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.vector_field(x, t), -compute_div(self.vector_field, x, t)) # eq 2.47


class pfIntergrater:
    def __init__(self,
                 v: Velocity,
                 s: Score,
                 interpolant: Interpolant,
                 n_step: int,
                 sample_only: bool = False,
                 method: str = 'dopri5',
                 atol: torch.tensor = 5e-4,
                 rtol: torch.tensor = 5e-4):
        self.v = v
        self.s = s
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.n_step = n_step
        self.pf = ProbabilityFlow(v, s, interpolant=interpolant, sample_only=sample_only)
        self.pf.set_vector_filed()
    def rollout(self, x0: Sample):
        t = torch.linspace(0., 1., self.n_step).to(x0)
        dlogp = torch.zeros(x0.shape[0]).to(x0)

        state = odeint(
            self.pf,
            (x0, dlogp),
            t,
            method=self.method,
            atol=[self.atol, self.atol],
            rtol=[self.rtol, self.rtol]
        )
        x, dlogp = state
        return x, dlogp


class SDEFlow(nn.Module):
    def __init__(self, v: Velocity, s: Score, interpolant: Interpolant, dt: torch.tensor, eps: torch.tensor, ):
        super(SDEFlow, self).__init__()
        self.v = v
        self.s = s
        self.interpolant = interpolant
        self.dt = dt
        self.eps = eps
        self.b = self.set_b
        self.bf = self.set_bf
        self.br = self.set_br
        self.dt_dlogp = self.get_dt_dlogp

    def set_b(self, x: torch.tensor, t: torch.tensor):
        self.v.to(x)
        self.s.to(x)
        return self.v(x, t) - self.interpolant.gg_dot(t) * self.s(x, t)

    def set_bf(self, x: torch.tensor, t: torch.tensor):
        self.v.to(x)
        self.s.to(x)
        return self.b(x, t) + self.eps * self.s(x, t)

    def set_br(self, x: torch.tensor, t: torch.tensor):
        self.v.to(x)
        self.s.to(x)
        with torch.no_grad():
            return self.b(x, t) - self.eps * self.s(x, t)

    # 存疑，按照作者所说，这里是2.65
    def get_dt_dlogp(self, x: torch.tensor, t: torch.tensor):
        score = self.s(x, t)
        s_norm = torch.linalg.norm(score, axis=-1)**2
        return -compute_div(self.bf, x, t) - self.eps * s_norm

    def step_forward(self, x: Sample, t: torch.tensor, method: str = 'heun'):
        dW = torch.sqrt(self.dt) * torch.randn_like(x)
        if method == 'heun':
            '''https://arxiv.org/pdf/2206.00364.pdf Alg. 2'''
            xhat = x + torch.sqrt(2 * self.eps) * dW
            K1 = self.bf(xhat, t + self.dt)
            xp = xhat + self.dt * K1
            K2 = self.bf(xp, t + self.dt)
            return xhat + 0.5 * self.dt * (K1 + K2)
        elif method == 'euler':
            return x + self.set_bf(x, t) * self.dt + torch.sqrt(2 * self.eps) * dW
        else:
            raise NotImplementedError("type should be in `heun` and `euler`")

    def step_backward(self, x: Sample, t: torch.tensor, method: str = 'heun'):
        dW = torch.sqrt(self.dt) * torch.randn_like(x)
        if method == 'heun':
            """https://arxiv.org/pdf/2206.00364.pdf Alg. 2"""
            xhat = x + torch.sqrt(2 * self.eps) * dW
            K1 = self.br(xhat, t - self.dt)
            xp = xhat - self.dt * K1
            K2 = self.br(xp, t - self.dt)
            return xhat - 0.5 * self.dt * (K1 + K2)
        elif method == 'euler':
            return x - self.set_br(x, t) * self.dt + torch.sqrt(2 * self.eps) * dW
        else:
            raise NotImplementedError("type should be in `heun` and `euler`")

    def step_likelihood(self, like: torch.tensor, x: Sample, t: torch.tensor):
        return like - self.dt_dlogp(x, t) * self.dt

    def rollout_forward(self, init: Sample, method: str = 'heun', save_num: int = 1):
        n_step = int(torch.ceil(1.0 / self.dt))
        assert n_step * self.dt == 1.0

        x = init
        save_every = int(n_step / save_num)
        xs = torch.zeros((save_num, *init.shape)).to(init)
        save_counter = 0
        for i in range(n_step):
            # t = torch.tensor(i * self.dt).to(x)
            # t = t.unsqueeze(0)
            t = (torch.ones((x.shape[0], 1))*i*self.dt).to(x)
            x = self.step_forward(x, t, method)
            if save_every > 0:
                if (i + 1) % save_every == 0:
                    xs[save_counter] = x
                    save_counter += 1

        return xs

    def rollout_likelihood(self, init: Sample, method: str='heun', save_num: int = 1, sample_num: int=1,):
        n_step = int(torch.ceil(1.0 / self.dt))
        assert n_step * self.dt == 1.0
        bs = init.shape[0]
        # likes = torch.zeros((sample_num, bs)).to(init)
        # xs = torch.zeros((sample_num, *init.shape)).to(init)

        assert n_step * self.dt == 1.
        assert n_step % save_num == 0

        x = init.repeat((sample_num, *[1]*len(init.shape))).reshape(bs*sample_num, *init.shape[1:])
        like = torch.zeros(sample_num*bs).to(x)

        for i in range(n_step):
            t = (1 - torch.ones((x.shape[0], 1)) * i * self.dt).to(x)
            x = self.step_backward(x, t, method=method)
            like = self.step_likelihood(like, x, t-self.dt)

        xs, likes = x.reshape((sample_num, *init.shape)), like.reshape((sample_num, bs))
        return xs, torch.mean(likes, dim=0)


def interpolant_loss(
        v: Velocity,
        s: Velocity,
        x0: Sample,
        x1: Sample,
        t: torch.tensor,
        interpolant: Interpolant,
        loss_fac: float
):
    '''compute the (variance-reduced loss on individual sample via anthentic sampling) page 50'''
    xtp, xtm, z = interpolant.get_antithetic_xts(t, x0, x1)
    # xtp, xtm, t, = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dIdt = interpolant.dtIt(t, x0, x1)
    vtp = v(xtp, t)
    vtm = v(xtm, t)
    loss_v = 0.5 * torch.sum(vtp**2, dim=-1) - torch.sum(dIdt * vtp, dim=-1)
    loss_v += 0.5 * torch.sum(vtm**2, dim=-1) - torch.sum(dIdt * vtm, dim=-1)

    stp = s(xtp, t)
    stm = s(xtm, t)

    loss_s = 0.5 * torch.sum(stp**2, dim=-1) + torch.sum(1 / interpolant.gamma(t) * (stp * z), dim=-1)
    loss_s += 0.5 * torch.sum(stm**2, dim=-1) - torch.sum(1/ interpolant.gamma(t) * (stm*z), dim=-1)
    return loss_v.mean(), loss_fac * loss_s.mean()
