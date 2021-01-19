from .base import FeatureMappingFlow, StrictInvertibleMatrix, LooseInvertibleMatrix
from DynaSysML.typing_ import *
from typing import *
from DynaSysML.Layers import initializer
from DynaSysML.core import variable, to_numpy, flatten_to_ndims, unflatten_from_ndims
import torch

__all__ = [
    'InvertibleDense', 'InvertibleConv1d',
    'InvertibleConv2d', 'InvertibleConv3d'
]

class InvertibleLinearNd(FeatureMappingFlow):
    """Base class for invertible linear transformation flows."""

    __constants__ = FeatureMappingFlow.__constants__ + (
        'invertible_matrix', 'num_features', 'strict', 'epsilon',
    )

    invertible_matrix: Module
    num_features: int
    strict: bool
    epsilon: float

    def __init__(self,
                 num_features: int,
                 strict: bool = False,
                 weight_init: TensorInitArgType = initializer.kaming_uniform,
                 dtype: str = 'float32',
                 epsilon: float = 1e-5):
        """
        Construct a new linear transformation flow.

        Args:
            num_features: The number of features to be transformed.
                The invertible transformation matrix will have the shape
                ``[num_features, num_features]``.
            strict: Whether or not to use the strict invertible matrix?
                Defaults to :obj:`False`.  See :class:`LooseInvertibleMatrix`
                and :class:`StrictInvertibleMatrix`.
            weight_init: The weight initializer for the seed matrix.
            dtype: The dtype of the invertible matrix.
            epsilon: The infinitesimal constant to avoid having numerical issues.
        """
        spatial_ndims = self._get_spatial_ndims()
        super().__init__(
            axis=-(spatial_ndims + 1),
            event_ndims=(spatial_ndims + 1),
            explicitly_invertible=True,
        )

        self.num_features = int(num_features)
        self.strict = bool(strict)
        self.epsilon = float(epsilon)

        # Using the backend random generator instead of numpy generator
        # will allow the backend random seed to have effect on the initialization
        # step of the invertible matrix.
        seed_matrix = variable(
            shape=[num_features, num_features], dtype=dtype,
            initializer=weight_init, requires_grad=False,
        )

        if strict:
            self.invertible_matrix = StrictInvertibleMatrix(
                to_numpy(seed_matrix), dtype=dtype, epsilon=epsilon)
        else:
            self.invertible_matrix = LooseInvertibleMatrix(
                to_numpy(seed_matrix), dtype=dtype)

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        raise NotImplementedError()

    # @jit_method
    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        # obtain the weight
        weight, log_det = self.invertible_matrix(
            inverse=inverse, compute_log_det=compute_log_det)
        spatial_ndims = self.x_event_ndims - 1
        weight = weight.reshape(list(weight.shape) + [1] * spatial_ndims)

        # compute the output
        output, front_shape = flatten_to_ndims(input, spatial_ndims + 2)
        output = self._linear_transform(output, weight)
        output = unflatten_from_ndims(output, front_shape)

        # compute the log_det
        output_log_det = input_log_det
        if log_det is not None:
            for axis in list(range(-spatial_ndims, 0)):
                log_det = log_det * float(input.shape[axis])
            if input_log_det is not None:
                output_log_det = input_log_det + log_det
            else:
                output_log_det = log_det

        return output, output_log_det


class InvertibleDense(InvertibleLinearNd):
    """An invertible linear transformation."""

    def _get_spatial_ndims(self) -> int:
        return 0

    # @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, weight)


class InvertibleConv1d(InvertibleLinearNd):
    """An invertible 1d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 1

    # @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv1d(input, weight)


class InvertibleConv2d(InvertibleLinearNd):
    """An invertible 2d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 2

    # @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv2d(input, weight)


class InvertibleConv3d(InvertibleLinearNd):
    """An invertible 3d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 3

    # @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv3d(input, weight)
