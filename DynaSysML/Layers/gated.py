from .base import BaseSingleVariateLayer
from DynaSysML.typing_ import Tensor, Module
import torch

__all__ = [
    'Gated', 'GatedWithActivation',
]


class BaseGated(BaseSingleVariateLayer):

    __constants__ = ('feature_axis', 'num_features', 'gate_bias', 'activation')

    feature_axis: int
    num_features: int
    gate_bias: float

    def __init__(self,
                 feature_axis: int,
                 num_features: int,
                 gate_bias: float):
        super().__init__()
        self.feature_axis = feature_axis
        self.num_features = num_features
        self.gate_bias = gate_bias

    def _apply_activation(self, input: Tensor) -> Tensor:
        raise NotImplementedError()

    # @jit_method
    def _forward(self, input: Tensor) -> Tensor:
        if input.shape[self.feature_axis] != self.num_features * 2:
            raise ValueError(
                'The shape of the pre-gated output is invalid: '
                'the size of axis {} should be {}, '
                'but the pre-gated output shape is {}.'.format(
                    self.feature_axis,
                    self.num_features * 2,
                    list(input.shape),
                    )
            )

        output, gate = torch.split(
            input,
            [self.num_features, self.num_features],
            self.feature_axis
        )
        output = self._apply_activation(output)
        output = output * torch.sigmoid(gate + self.gate_bias)

        return output


class Gated(BaseGated):

    # @jit_method
    def _apply_activation(self, input: Tensor) -> Tensor:
        return input


class GatedWithActivation(BaseGated):

    activation: Module

    def __init__(self,
                 feature_axis: int,
                 num_features: int,
                 gate_bias: float,
                 activation: Module):
        super().__init__(feature_axis=feature_axis,
                         num_features=num_features,
                         gate_bias=gate_bias)
        self.activation = activation

    # @jit_method
    def _apply_activation(self, input: Tensor) -> Tensor:
        return self.activation(input)
