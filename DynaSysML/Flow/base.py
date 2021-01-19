import torch
import torch.nn as nn
import scipy.linalg as la
import numpy as np
from DynaSysML.Layers import BaseLayer
from DynaSysML.typing_ import *
from typing import *
from DynaSysML.core import broadcast_to, add_buffer, add_parameter, as_tensor

__all__ = [
    'BaseFlow', 'FeatureMappingFlow', 'InverseFlow', 'SequentialFlow',

    # Invertible Matrix utils
    'LooseInvertibleMatrix', 'StrictInvertibleMatrix'
]

class BaseFlow(BaseLayer):
    """
    Base class for normalizing flows.

    A normalizing flow transforms a random variable `x` into `y` by an
    (implicitly) invertible mapping :math:`y = f(x)`, whose Jaccobian matrix
    determinant :math:`\\det \\frac{\\partial f(x)}{\\partial x} \\neq 0`, thus
    can derive :math:`\\log p(y)` from given :math:`\\log p(x)`.
    """

    __constants__ = ('x_event_ndims', 'y_event_ndims', 'explicitly_invertible')

    x_event_ndims: int
    """Number of event dimensions in `x`."""

    y_event_ndims: int
    """Number of event dimensions in `y`."""

    explicitly_invertible: bool
    """
    Whether or not this flow is explicitly invertible?

    If a flow is not explicitly invertible, then it only supports to
    transform `x` into `y`, and corresponding :math:`\\log p(x)` into
    :math:`\\log p(y)`.  It cannot compute :math:`\\log p(y)` directly
    without knowing `x`, nor can it transform `x` back into `y`.
    """

    def __init__(self,
                 x_event_ndims: int,
                 y_event_ndims: int,
                 explicitly_invertible: bool):
        super().__init__()

        self.x_event_ndims = int(x_event_ndims)
        self.y_event_ndims = int(y_event_ndims)
        self.explicitly_invertible = bool(explicitly_invertible)

    def invert(self) -> 'BaseFlow':
        """
        Get the inverse flow from this flow.

        Specifying `inverse = True` when calling the inverse flow will be
        interpreted as having `inverse = False` in the original flow, and
        vise versa.

        If the current flow requires to be initialized by calling it
        with `inverse = False`, then the inversed flow will require to be
        initialized by calling it with `inverse = True`, and vise versa.

        Returns:
            The inverse flow.
        """
        return InverseFlow(self)

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                input_log_det: Optional[Tensor] = None,
                inverse: bool = False,
                compute_log_det: bool = True
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Transform `x` into `y` and compute the log-determinant of `f` at `x`
        (if `inverse` is False); or transform `y` into `x` and compute the
        log-determinant of `f^{-1}` at `y` (if `inverse` is True).

        Args:
            input: `x` (if `inverse` is False) or `y` (if `inverse` is True).
            input_log_det: The log-determinant of the previous layer.
                Will add the log-determinant of this layer to `input_log_det`,
                to obtain the output log-determinant.  If no previous layer,
                will start from zero log-det.
            inverse: See above.
            compute_log_det: Whether or not to compute the log-determinant?

        Returns:
            The transformed tensor, and the summed log-determinant of
            the previous flow layer and this layer.
        """
        if inverse:
            event_ndims = self.y_event_ndims
        else:
            event_ndims = self.x_event_ndims

        if input.dim() < event_ndims:
            raise ValueError(
                '`input` is required to be at least {}d, but the input shape '
                'is {}.'.format(event_ndims, list(input.shape))
            )

        input_shape = list(input.shape)
        log_det_shape = input_shape[: len(input_shape) - event_ndims]

        if input_log_det is not None:
            if list(input_log_det.shape) != log_det_shape:
                raise ValueError(
                    'The shape of `input_log_det` is not expected: '
                    'expected to be {}, but got {}.'.
                    format(log_det_shape, list(input_log_det.shape))
                )

        # compute the transformed output and log-det
        output, output_log_det = self._forward(
            input, input_log_det, inverse, compute_log_det)

        if output_log_det is not None:
            if output_log_det.dim() < len(log_det_shape):
                output_log_det = broadcast_to(output_log_det, log_det_shape)

            if list(output_log_det.shape) != log_det_shape:
                raise ValueError(
                    'The shape of `output_log_det` is not expected: '
                    'expected to be {}, but got {}.'.
                    format(log_det_shape, list(output_log_det.shape))
                )

        return output, output_log_det


class FeatureMappingFlow(BaseFlow):
    """Base class for flows mapping input features to output features."""

    __constants__ = BaseFlow.__constants__ + ('axis',)

    axis: int
    """The feature axis (negative index)."""

    def __init__(self,
                 axis: int,
                 event_ndims: int,
                 explicitly_invertible: bool):
        """
        Construct a new :class:`FeatureMappingFlow`.

        Args:
            axis: The feature axis, on which to apply the transformation.
                It must be a negative integer, and included in the
                event dimensions.
            event_ndims: Number of event dimensions in both `x` and `y`.
                `x.ndims - event_ndims == log_det.ndims` and
                `y.ndims - event_ndims == log_det.ndims`.
            explicitly_invertible: Whether or not this flow is explicitly
                invertible?
        """
        # check the arguments
        axis = int(axis)
        event_ndims = int(event_ndims)

        if event_ndims < 1:
            raise ValueError(f'`event_ndims` must be at least 1: '
                             f'got {event_ndims}')

        if axis >= 0 or axis < -event_ndims:
            raise ValueError(
                f'`-event_ndims <= axis < 0` does not hold: '
                f'`axis` is {axis}, while `event_ndims` is {event_ndims}.')

        # construct the layer
        super().__init__(x_event_ndims=event_ndims,
                         y_event_ndims=event_ndims,
                         explicitly_invertible=explicitly_invertible)
        self.axis = axis

    @property
    def event_ndims(self) -> int:
        """Get the number of event dimensions in both `x` and `y`."""
        return self.x_event_ndims


# ---- composite flows ----
class InverseFlow(BaseFlow):
    """A flow that inverts another given flow."""

    __constants__ = BaseFlow.__constants__ + ('original_flow',)

    original_flow: Module
    """The original flow, which is inverted by this :class:`InverseFlow`."""

    def __init__(self, flow: Module):
        # if (not isinstance(flow, BaseFlow) and not is_jit_layer(flow)) or \
        #         not flow.explicitly_invertible:
        if not isinstance(flow, BaseFlow) or not flow.explicitly_invertible:
            raise TypeError(
                f'`flow` must be an explicitly invertible flow: '
                f'got {flow!r}'
            )

        super().__init__(
            x_event_ndims=flow.y_event_ndims,
            y_event_ndims=flow.x_event_ndims,
            explicitly_invertible=flow.explicitly_invertible,
        )
        self.original_flow = flow

    def invert(self) -> BaseFlow:
        return self.original_flow

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        return self.original_flow(
            input, input_log_det, not inverse, compute_log_det)


class _NotInvertibleFlow(Module):

    def forward(self,
                input: Tensor,
                input_log_det: Optional[Tensor],
                inverse: bool,
                compute_log_det: bool
                ) -> Tuple[Tensor, Optional[Tensor]]:
        raise RuntimeError('Not an explicitly invertible flow.')


class SequentialFlow(BaseFlow):

    __constants__ = BaseFlow.__constants__ + ('_chain', '_inverse_chain')

    _chain: torch.nn.ModuleList

    # The inverse chain is provided, such that JIT support is still okay.
    # TODO: This separated inverse chain will cause `state_dict()` to have
    #       duplicated weights.  Deal with this issue.
    _inverse_chain: torch.nn.ModuleList

    def __init__(self, *flows: Union[Module, Sequence[Module]]):
        from ..Layers.utils import flatten_nested_layers

        # validate the arguments
        flows = flatten_nested_layers(flows)
        if not flows:
            raise ValueError('`flows` must not be empty.')

        for i, flow in enumerate(flows):
            if not isinstance(flow, BaseFlow): #  and not is_jit_layer(flow):
                raise TypeError(f'`flows[{i}]` is not a flow: got {flow!r}')

        for i, (flow1, flow2) in enumerate(zip(flows[:-1], flows[1:])):
            if flow2.x_event_ndims != flow1.y_event_ndims:
                raise ValueError(
                    f'`x_event_ndims` of `flows[{i + 1}]` != '
                    f'`y_event_ndims` of `flows[{i}]`: '
                    f'{flow2.x_event_ndims} vs {flow1.y_event_ndims}.'
                )

        super().__init__(
            x_event_ndims=flows[0].x_event_ndims,
            y_event_ndims=flows[-1].y_event_ndims,
            explicitly_invertible=all(
                flow.explicitly_invertible for flow in flows)
        )

        self._chain = torch.nn.ModuleList(flows)
        if self.explicitly_invertible:
            self._inverse_chain = torch.nn.ModuleList(reversed(flows))
        else:
            self._inverse_chain = torch.nn.ModuleList([_NotInvertibleFlow()])

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        output, output_log_det = input, input_log_det

        if inverse:
            for flow in self._inverse_chain:
                output, output_log_det = flow(
                    output, output_log_det, True, compute_log_det)
        else:
            for flow in self._chain:
                output, output_log_det = flow(
                    output, output_log_det, False, compute_log_det)

        return output, output_log_det


class InvertibleMatrix(nn.Module):
    def __init__(self, size):
        super(InvertibleMatrix, self).__init__()
        self.size = size

    def __repr__(self):
        return f'{self.__class__.__qualname__}(size={self.size})'


class LooseInvertibleMatrix(InvertibleMatrix):
    '''
    A matrix initialized to be an invertible, orthogonal matrix.

    There is no guarantee that the matrix will keep invertible during training.
    But according to the measure theory, the non-inbvertible n by n real matrix are
    of measure 0. Thus this class is generally enough for use.
    '''
    def __init__(self, seed_matrix, dtype=torch.float32):
        initial_matrix = la.qr(seed_matrix)[0] # 获取正交矩阵
        super().__init__(initial_matrix.shape[0])
        add_parameter(self, 'matrix', as_tensor(seed_matrix, dtype=dtype, force_copy=True))

    def forward(self, inverse=False, compute_log_det=True):
        '''
        :param inverse: bool
        :param compute_log_det: bool
        :return: an invertible, orthogonal matrix, log_det
        '''
        log_det = None
        if inverse:
            matrix = torch.inverse(self.matrix)
            if compute_log_det:
                log_det = - torch.slogdet(self.matrix)[1] # 计算矩阵行列式的绝对值的对数
        else:
            matrix = self.matrix
            if compute_log_det:
                log_det = torch.slogdet(self.matrix)[1]
        return matrix, log_det


class StrictInvertibleMatrix(InvertibleMatrix):
    def __init__(self, seed_matrix, dtype=torch.float32, epsilon=1e-5):
        initial_matrix = la.qr(seed_matrix)[0]
        super().__init__(initial_matrix.shape[0])

        matrix_shape = list(initial_matrix.shape)
        # self.size = matrix_shape[0]
        initial_P, initial_L, initial_U = la.lu(initial_matrix)
        initial_s = np.diag(initial_U)
        initial_sign = np.sign(initial_s)
        initial_log_s = np.log(np.maximum(np.abs(initial_s), epsilon))
        initial_U = np.triu(initial_U, k=1) # 上三角阵，对角线元素为0
        add_buffer(self, 'P', as_tensor(initial_P, dtype=dtype, force_copy=True))
        add_parameter(self, 'pre_L', as_tensor(initial_L, dtype=dtype, force_copy=True))
        add_buffer(self, 'L_mask', as_tensor(np.tril(np.ones(matrix_shape), k=-1), dtype=dtype, force_copy=True))
        add_parameter(self, 'pre_U', as_tensor(initial_U, dtype=dtype, force_copy=True))
        add_buffer(self, 'U_mask', as_tensor(np.triu(np.ones(matrix_shape), k=1), dtype=dtype, force_copy=True))
        add_buffer(self, 'sign', as_tensor(initial_sign, dtype=dtype, force_copy=True))
        add_parameter(self, 'log_s', as_tensor(initial_log_s, dtype=dtype, force_copy=True))

    def forward(self, inverse=False, compute_log_det=True):
        P = self.P
        L = (self.L_mask * self.pre_L + torch.eye(self.size, dtype=P.dtype))
        U = self.U_mask * self.pre_U + torch.diag(self.sign * torch.exp(self.log_s))
        log_det = None
        if inverse:
            matrix = torch.matmul(torch.inverse(U), torch.matmul(torch.inverse(L), torch.inverse(P)))
            if compute_log_det:
                log_det = -torch.sum(self.log_s)
        else:
            matrix = torch.matmul(P, torch.matmul(L, U))
            if compute_log_det:
                log_det = torch.sum(self.log_s)
        return matrix, log_det
