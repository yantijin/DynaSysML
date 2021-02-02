import torch
from torchcde import cdeint
from torchcde.interpolation_cubic import NaturalCubicSpline, natural_cubic_spline_coeffs, natural_cubic_coeffs
from torchcde.interpolation_linear import LinearInterpolation, linear_interpolation_coeffs
from torchdiffeq import odeint_adjoint, odeint, odeint_event
from DynaSysML.Layers import BaseLayer
from .odefunc import defunc


__all__ = [
    'odeint_adjoint', 'odeint', 'odeint_event', 'NeuralODE', 'cdeint',
    'natural_cubic_coeffs', 'natural_cubic_spline_coeffs', 'linear_interpolation_coeffs',
    'LinearInterpolation', 'NaturalCubicSpline'
]

class NeuralODE(BaseLayer):
    def __init__(self, func, t=None, last=False, order=1):
        super(NeuralODE, self).__init__()
        # if not isinstance(func, nn.Module):
        #     raise ValueError('func is required to be an instance of nn.Module.')
        if func.forward.__code__.co_argcount == 3:
            self.func = func
        else:
            self.func = defunc(func, order=order)
            print(self.func)
        self.t = t
        self.last = last

    def forward(self, y0, t=None, **kwargs):
        if self.t is not None and t is None:
            t = self.t
        elif self.t is None and t is not None:
            pass
        else:
            raise ValueError('you should add `t` when define NeuralODE or call a NeuralODE object')
        if 'event_fn' not in kwargs or ('event_fn' in kwargs and kwargs['event_fn'] is None):
            solution = odeint_adjoint(self.func, y0=y0, t=t, **kwargs)
            if self.last:
                solution = solution[-1]
            return solution
        else:
            event_t, solution = odeint_adjoint(self.func, y0=y0, t=t, **kwargs)
            return event_t, solution

    def trajectory(self, x: torch.Tensor, s_span: torch.Tensor, method='odeint', **kwargs):
        if method == 'adjoint':
            solution = odeint_adjoint(self.func, x, s_span, **kwargs)
        elif method == 'odeint':
            solution = odeint(self.func, x, s_span, **kwargs)
        else:
            raise ValueError('Please check parameters `method`, it should be `adjoint` or `odeint`')

        return solution