import torch
from .base import BaseFlow, FeatureMappingFlow, SequentialFlow
import DynaSysML.Layers.initializer as init
from DynaSysML.typing_ import *
from typing import *
from DynaSysML.core import reduce_sum, flatten_to_ndims, unflatten_from_ndims, add_parameter, variable
from torchdiffeq import odeint_adjoint as odeint
from .utils import _flip


class Planar(FeatureMappingFlow):
    '''
    .. math::

        \\begin{aligned}
                \\mathbf{y} &= \\mathbf{x} +
                    \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b)\\\\
                \\mathbf{\hat{u}} &= \mathbf{u} +
                    \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
                           (\\mathbf{w}^\\top \\mathbf{u})\\right]
                    \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\
                m(a) &= -1 + \\log(1+\\exp(a))
        \\end{aligned}

    '''
    def __init__(self,
                 num_features,
                 event_ndims: int = 1,
                 axis: int =-1,
                 w_init: TensorInitArgType = init.normal,
                 b_init: TensorInitArgType =init.zeros,
                 u_init: TensorInitArgType =init.normal,):
        super().__init__(
            axis=axis,
            event_ndims=event_ndims,
            explicitly_invertible=False
        )
        add_parameter(self, 'w',
                      value=variable([1, num_features], initializer=w_init),)
        add_parameter(self, 'b',
                      value=variable([1], initializer=b_init))
        add_parameter(self, 'u',
                      value=variable([1, num_features], initializer=u_init))
        self.num_features = num_features

        self.u_hat = None


    def get_uhat(self):
        if self.u_hat == None:
            wu = torch.matmul(self.w, self.u.T)
            self.u_hat = self.u + (-1. + torch.nn.functional.softplus(wu) - wu) * \
                        self.w / reduce_sum(self.w ** 2)
            return self.u_hat
        return self.u_hat


    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:

        w = self.w
        b = self.b
        _ = self.get_uhat()
        input_shape = list(input.shape)
        if input_shape[self.axis] != self.num_features:
            raise ValueError('`num_features` is not equal to the `axis` of `input`, '
                             'please check that!')
        if inverse == True:
            raise ValueError('`inverse` for planar nf should never be `True`')
        x_flatten, front_shape = flatten_to_ndims(input, 2)
        wxb = torch.matmul(x_flatten, w.T) + b
        tanh_wb = torch.tanh(wxb)
        out = x_flatten + self.u_hat * tanh_wb
        out = unflatten_from_ndims(out, front_shape=front_shape)

        output_log_det = input_log_det
        if compute_log_det:
            grad = 1. - tanh_wb**2
            phi = grad * w # shape == [?, n_units]
            u_phi = torch.matmul(phi, self.u_hat.T)
            log_det = torch.log(torch.abs(1. + u_phi)) # [? 1]
            log_det = unflatten_from_ndims(log_det, front_shape)
            log_det = torch.squeeze(log_det)
            if output_log_det is None:
                output_log_det = log_det
            else:
                output_log_det += log_det

        return out, output_log_det


class BaseContinuousNF(FeatureMappingFlow):
    def __init__(self, odefunc, integration_times=torch.tensor([0., 1.]), axis=-1, event_ndims=1, solver='dopri5', atol=1e-5, rtol=1e-5):
        '''
        :param odefunc: [dz/dt, dlogp(z)/dt] = odefunc(t, (z_0, dlogp(z_0)/dt_0))
        :param integration_times: 1-D time grid
        :param axis: feature axises
        :param event_ndims:  number of event_ndims in data
        :param solver: numerical integration methods, default=dopri5
        :param atol:
        :param rtol:
        '''
        super().__init__(
            axis=axis,
            event_ndims=event_ndims,
            explicitly_invertible=True
        )
        self.odefunc = odefunc # [dz/dt, dlogp(z)/dt] = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.integration_times = integration_times

    def _forward(self,
                 z: Tensor,
                 input_log_pz: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            self.integration_times = _flip(self.integration_times.to(z), 0)

        if input_log_pz is None:
            input_log_pz = torch.zeros(z.shape[0], 1).to(z)

        # 对于迹估计中的服从均值为0，方差为1的向量进行重置
        self.odefunc.before_odeint()

        state_t = odeint(
            self.odefunc,
            (z, input_log_pz),
            self.integration_times.to(z),
            atol = [self.atol, self.atol] if self.solver == 'dopri5' else self.atol,
            rtol = [self.rtol, self.rtol] if self.solver == 'dopri5' else self.rtol,
            method = self.solver,
            options={}
        )

        z_t, logpz_t = state_t[:2]

        return z_t, logpz_t