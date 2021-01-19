import torch
import torch.nn as nn
import numpy as np

# wrap dy/dt = diffeq(t, y_0) to [dy/dt, logp(y(t))/dt] = [diffeq(t, y_0), -e^T dy/y e]
__all__= [
    'ODEfunc', 'divergence_bf', 'divergence_approx',
    'sample_gaussian_like', 'sample_rademacher_like'
]

# dx/y 的迹,暴力解法
def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


# e * dz/dx * e  对dz/dx的迹进行近似
def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


# 满足均值为0，方差为1的两种分布
def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


# wrap the logp(z(t))/dt - Tr(df/dz)
class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        '''
        :param diffeq: dy/dt = diffeq(t, y_0)
        :param divergence_fn: 迹逼近函数
        :param residual: 是否为residual形式
        :param rademacher: 迹估计向量采样分布

        :return
            dy:
            divergence: approximate for Tr(dy/dt)
        '''
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class defunc(nn.Module):
    def __init__(self, net, order=1):
        super(defunc, self).__init__()
        self.net = net
        self.order = order

    def forward(self, s, y):
        if self.order > 1:
            return self.high_order_forward(s, y)
        else:
            return self.net(y)

    def high_order_forward(self, s, y):
        y_new = []
        size_order = y.shape[-1] // self.order
        for i in range(1, self.order):
            y_new.append(y[..., size_order*i:size_order*(i+1)])
        y_new.append(self.net(y))
        out = torch.cat(y_new, dim=-1).to(y)
        return out