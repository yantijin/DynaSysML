from ..Layers import BaseLayer
from typing import *
from DynaSysML.typing_ import *
from DynaSysML.core import broadcast_to, reduce_sum, broadcast_shape
import torch

__all__ = [
    'BaseScale', 'ExpScale', 'SigmoidScale',
    'LinearScale'
]

class BaseScale(BaseLayer):
    """Base class for scaling `input`."""

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                pre_scale: Tensor,
                event_ndims: int,
                input_log_det: Optional[Tensor] = None,
                compute_log_det: bool = True,
                inverse: bool = False
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # validate the argument
        if input.dim() < event_ndims:
            raise ValueError(
                '`rank(input) >= event_ndims` does not hold: the `input` shape '
                'is {}, while `event_ndims` is {}.'.
                    format(list(input.shape), event_ndims)
            )
        if pre_scale.dim() > input.dim():
            raise ValueError(
                '`rank(input) >= rank(pre_scale)` does not hold: the `input` '
                'shape is {}, while the shape of `pre_scale` is {}.'.
                    format(list(input.shape), list(pre_scale.shape))
            )

        input_shape = list((input.shape))
        event_ndims_start = len(input_shape) - event_ndims
        event_shape = input_shape[event_ndims_start:]
        log_det_shape = input_shape[: event_ndims_start]

        if input_log_det is not None:
            if list(input_log_det.shape) != log_det_shape:
                raise ValueError(
                    'The shape of `input_log_det` is not expected: '
                    'expected to be {}, but got {}'.
                        format(log_det_shape, list(input_log_det.shape))
                )

        scale, log_scale = self._scale_and_log_scale(
            pre_scale, inverse, compute_log_det)
        output = input * scale # 注意是点乘！

        if log_scale is not None:
            log_scale = broadcast_to(
                log_scale,
                broadcast_shape(list(log_scale.shape), event_shape)
            )

            # the last `event_ndims` dimensions must match the `event_shape`
            log_scale_shape = list(log_scale.shape)
            log_scale_event_shape = \
                log_scale_shape[len(log_scale_shape) - event_ndims:]
            if log_scale_event_shape != event_shape:
                raise ValueError(
                    'The shape of the final {}d of `log_scale` is not expected: '
                    'expected to be {}, but got {}.'.
                        format(event_ndims, event_shape, log_scale_event_shape)
                )

            # reduce the last `event_ndims` of log_scale
            log_scale = reduce_sum(log_scale, axis=list(range(-event_ndims, 0)))

            # now add to input_log_det, or broadcast `log_scale` to `log_det_shape`
            if input_log_det is not None:
                output_log_det = input_log_det + log_scale
                if list(output_log_det.shape) != log_det_shape:
                    raise ValueError(
                        'The shape of the computed `output_log_det` is not expected: '
                        'expected to be {}, but got {}.'.
                            format(list(output_log_det.shape), log_det_shape)
                    )
            else:
                output_log_det = broadcast_to(log_scale, log_det_shape)
        else:
            output_log_det = None

        return output, output_log_det


class ExpScale(BaseScale):
    """
    Scaling `input` with `exp` activation.

    ::

        if inverse:
            output = input / exp(pre_scale)
            output_log_det = -pre_scale
        else:
            output = input * exp(pre_scale)
            output_log_det = pre_scale
    """

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        log_scale: Optional[Tensor] = None
        if inverse:
            neg_pre_scale = -pre_scale
            scale = torch.exp(neg_pre_scale)
            if compute_log_scale:
                log_scale = neg_pre_scale
        else:
            scale = torch.exp(pre_scale)
            if compute_log_scale:
                log_scale = pre_scale
        return scale, log_scale


class SigmoidScale(BaseScale):
    """
    Scaling `input` with `sigmoid` activation.

    ::

        if inverse:
            output = input / sigmoid(pre_scale)
            output_log_det = -log(sigmoid(pre_scale))
        else:
            output = input * sigmoid(pre_scale)
            output_log_det = log(sigmoid(pre_scale))
    """

    __constants__ = ('pre_scale_bias',)

    pre_scale_bias: float

    def __init__(self, pre_scale_bias: float = 0.):
        super().__init__()
        self.pre_scale_bias = pre_scale_bias

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.pre_scale_bias != 0.:
            pre_scale = pre_scale + self.pre_scale_bias

        log_scale: Optional[Tensor] = None
        if inverse:
            neg_pre_scale = -pre_scale
            scale = torch.exp(neg_pre_scale) + 1.
            if compute_log_scale:
                log_scale = torch.nn.functional.softplus(neg_pre_scale)
        else:
            scale = torch.sigmoid(pre_scale)
            if compute_log_scale:
                log_scale = -torch.nn.functional.softplus(-pre_scale)

        return scale, log_scale


class LinearScale(BaseScale):
    """
    Scaling `input` with `linear` activation.

    ::

        if inverse:
            output = input / pre_scale
            output_log_det = -log(abs(pre_scale))
        else:
            output = input * pre_scale
            output_log_det = log(abs(pre_scale))
    """

    __constants__ = ('epsilon',)

    epsilon: float

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        log_scale: Optional[Tensor] = None
        if inverse:
            scale = 1. / pre_scale
            if compute_log_scale:
                epsilon = torch.as_tensor(self.epsilon, dtype=pre_scale.dtype)
                log_scale = -torch.log(torch.max(abs(pre_scale), epsilon))
        else:
            scale = pre_scale
            if compute_log_scale:
                epsilon = torch.as_tensor(self.epsilon, dtype=pre_scale.dtype)
                log_scale = torch.log(torch.max(abs(pre_scale), epsilon))

        return scale, log_scale
