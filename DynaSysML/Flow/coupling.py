import torch
from typing import *
from DynaSysML.typing_ import *
from .scale import *
from .base import FeatureMappingFlow
from DynaSysML.Layers.base import IS_CHANNEL_LAST

__all__ = [
    'CouplingLayer', 'CouplingLayer1d',
    'CouplingLayer2d', 'CouplingLayer3d'
]

class CouplingLayer(FeatureMappingFlow):
    """
    A general implementation of the coupling layer (Dinh et al., 2016).

    Basically, a :class:`CouplingLayer` does the following transformation::

        x1, x2 = split(x)
        if secondary:
            x1, x2 = x2, x1

        y1 = x1

        shift, pre_scale = shift_and_scale(x1)
        if scale_type == 'exp':
            y2 = (x2 + shift) * exp(pre_scale)
        elif scale_type == 'sigmoid':
            y2 = (x2 + shift) * sigmoid(pre_scale + sigmoid_scale_bias)
        elif scale_type == 'linear':
            y2 = (x2 + shift) * pre_scale
        else:
            y2 = x2 + shift

        if secondary:
            y1, y2 = y2, y1
        y = tf.concat([y1, y2], axis=axis)

    The inverse transformation, and the log-determinants are computed
    according to the above transformation, respectively.
    """

    __constants__ = FeatureMappingFlow.__constants__ + (
        'shift_and_pre_scale', 'scale', 'secondary',
    )

    shift_and_pre_scale: Module
    scale: Module
    secondary: bool

    def __init__(self,
                 shift_and_pre_scale: Module,
                 axis: int = -1,
                 event_ndims: int = 1,
                 scale: Union[str, BaseScale, Type[BaseScale],
                              Callable[[], BaseScale]] = 'exp',
                 secondary: bool = False,
                 sigmoid_scale_bias: float = 2.,
                 epsilon: float = 1e-5):
        """
        Construct a new :class:`BaseCouplingLayer`.

        Args:
            shift_and_pre_scale: A layer which maps `x` to `shift` and
                `pre_scale`.  `pre_scale` will then be fed into `scale`
                to obtain the final scale tensor.

                You may construct a single-input, double-output layer by
                composite basic layers with :class:`tk.layers.Branch` and
                :class:`tk.layers.Sequential`, or with your own layer.
            scale: A str, one of {"exp", "sigmoid", "linear"}, an instance
                of :class:`BaseScale` object, a sub-class of :class:`BaseScale`,
                a factory to construct the :class:`BaseScale` object, or None.

                If it is "exp", an :class:`ExpScale` will be constructed.

                If it is "sigmoid", a :class:`SigmoidScale` will be constructed,
                with `sigmoid_scale_bias` used as the `pre_scale_bias` argument
                of :class:`SigmoidScale` constructor.

                If it is "linear", a :class:`LinearScale` will be constructed.
            axis: The feature axis, which to apply the transformation.
            event_ndims: Number of dimensions to be considered as the
                event dimensions.  `x.ndims - event_ndims == log_det.ndims`.
            secondary: Whether or not to swap the left and right part
                after splitting the input?  See above.
            sigmoid_scale_bias: Used as the `pre_scale_bias` argument
                for constructing :class:`SigmoidScale`, when `scale`
                is "sigmoid" or the `SigmoidScale` class.
            epsilon: The infinitesimal constant to avoid dividing by zero or
                taking logarithm of zero.
        """
        # validate the argument
        INVALID = object()

        if isinstance(scale, str):
            scale_name = scale.lower()
            if scale_name == 'exp':
                scale = ExpScale
            elif scale_name == 'sigmoid':
                scale = SigmoidScale
            elif scale_name == 'linear':
                scale = LinearScale
            else:
                scale = INVALID

        if isinstance(scale, Module):
            if not isinstance(scale, BaseScale):
                scale = INVALID
        elif isinstance(scale, type) or callable(scale):
            if scale is SigmoidScale:
                scale = scale(pre_scale_bias=sigmoid_scale_bias)
            else:
                scale = scale()
        else:
            scale = INVALID

        if scale is INVALID:
            raise ValueError(f'`scale` must be a `BaseScale` class, '
                             f'an instance of `BaseScale`, a factory to '
                             f'construct a `BaseScale` instance, or one of '
                             f'{{"exp", "sigmoid", "linear"}}: got {scale!r}')

        super().__init__(
            axis=int(axis), event_ndims=event_ndims, explicitly_invertible=True)

        self.shift_and_pre_scale = shift_and_pre_scale
        self.scale = scale
        self.secondary = bool(secondary)
        self.sigmoid_scale_bias = sigmoid_scale_bias
        self.epsilon = epsilon

    def _forward(self,
              input: Tensor,
              input_log_det: Optional[Tensor],
              inverse: bool,
              compute_log_det: bool
              ) -> Tuple[Tensor, Optional[Tensor]]:
        # split the tensor
        n_features = input.shape[self.axis]
        n1 = n_features // 2
        n2 = n_features - n1
        x1, x2 = torch.split(input, [n1, n2], self.axis)
        if self.secondary:
            x1, x2 = x2, x1

        # do transform
        y1 = x1
        shift, pre_scale = self.shift_and_pre_scale(x1)
        if inverse:
            y2, output_log_det = self.scale(
                input=x2,
                pre_scale=pre_scale,
                event_ndims=self.event_ndims,
                input_log_det=input_log_det,
                compute_log_det=compute_log_det,
                inverse=True,
            )
            y2 = y2 - shift
        else:
            y2, output_log_det = self.scale(
                input=x2 + shift,
                pre_scale=pre_scale,
                event_ndims=self.event_ndims,
                input_log_det=input_log_det,
                compute_log_det=compute_log_det,
                inverse=False,
            )

        # merge the tensor
        if self.secondary:
            y1, y2 = y2, y1
        output = torch.cat([y1, y2], dim=self.axis)

        return output, output_log_det


class CouplingLayerNd(CouplingLayer):

    def __init__(self,
                 shift_and_pre_scale: Module,
                 scale: Union[str, BaseScale, Type[BaseScale],
                              Callable[[], BaseScale]] = 'exp',
                 secondary: bool = False,
                 sigmoid_scale_bias: float = 2.,
                 epsilon: float = 1e-5):
        spatial_ndims = self._get_spatial_ndims()
        feature_axis = -1 if IS_CHANNEL_LAST else -(spatial_ndims + 1)

        super().__init__(
            shift_and_pre_scale=shift_and_pre_scale,
            axis=feature_axis,
            event_ndims=spatial_ndims + 1,
            scale=scale,
            secondary=secondary,
            sigmoid_scale_bias=sigmoid_scale_bias,
            epsilon=epsilon,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class CouplingLayer1d(CouplingLayerNd):
    """1D convolutional coupling layer flow."""

    def _get_spatial_ndims(self) -> int:
        return 1


class CouplingLayer2d(CouplingLayerNd):
    """2D convolutional coupling layer flow."""

    def _get_spatial_ndims(self) -> int:
        return 2


class CouplingLayer3d(CouplingLayerNd):
    """3D convolutional coupling layer flow."""

    def _get_spatial_ndims(self) -> int:
        return 3

