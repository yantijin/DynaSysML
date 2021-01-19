import torch
from .base import BaseSingleVariateLayer
from DynaSysML.typing_ import *

__all__ = [
    'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid',
]
LEAKY_RELU_DEFAULT_SLOPE = 0.01

class ReLU(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return torch.relu(input)


class LeakyReLU(BaseSingleVariateLayer):

    __constants__ = ('negative_slope',)

    negative_slope: float

    def __init__(self, negative_slope=LEAKY_RELU_DEFAULT_SLOPE):
        super().__init__()
        self.negative_slope = negative_slope

    def _forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.leaky_relu(input, negative_slope=self.negative_slope)


class Tanh(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input)


class Sigmoid(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input)
