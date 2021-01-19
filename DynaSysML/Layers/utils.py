from typing import *

from DynaSysML.arg_check import *
# from ..tensor import Module
from DynaSysML.typing_ import *
from .activation import *

__all__ = [
    'flatten_nested_layers', 'get_activation_class',
    'get_deconv_output_padding',
]


def flatten_nested_layers(nested_layers: Sequence[
    Union[
        Module,
        Sequence[
            Union[
                Module,
                Sequence[Module]
            ]
        ]
    ]
]) -> List[Module]:
    """
    Flatten a nested list of layers into a list of layers.

    Args:
        nested_layers: Nested list of layers.

    Returns:
        The flatten layer list.
    """
    def do_flatten(target, layer_or_layers):
        if isinstance(layer_or_layers, Module):
            target.append(layer_or_layers)
        elif hasattr(layer_or_layers, '__iter__') and not \
                isinstance(layer_or_layers, (str, bytes, dict)):
            for layer in layer_or_layers:
                do_flatten(target, layer)
        else:
            raise TypeError('`nested_layers` is not a nested list of layers.')

    ret = []
    do_flatten(ret, nested_layers)
    return ret


_activation_classes: Dict[str, Optional[Type[Module]]] = {
    'linear': None,
    'relu': ReLU,
    'leakyrelu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}


def get_activation_class(activation: Optional[str]) -> Optional[Type[Module]]:
    """
    Get the activation module class according to `name`.

    Args:
        activation: The activation name, or None (indicating no activation).

    Returns:
        The module class, or None (indicating no activation).
    """
    if activation is not None:
        canonical_name = activation.lower().replace('_', '')
        if canonical_name not in _activation_classes:
            raise ValueError(f'Unsupported activation: {activation}')
        return _activation_classes[canonical_name]


def get_deconv_output_padding(input_size: List[int],
                              output_size: List[int],
                              kernel_size: Union[int, List[int]] = 1,
                              stride: Union[int, List[int]] = 1,
                              padding: Union[int, List[int], str, PaddingMode] = PaddingMode.NONE,
                              dilation: Union[int, List[int]] = 1
                              ) -> List[int]:
    """
    Calculate the `output_padding` for deconvolution (conv_transpose).

    Args:
        input_size: The input size (shape) of the spatial dimensions.
        output_size: The output size (shape) of the spatial dimensions.
        kernel_size: The kernel size.
        stride: The stride.
        padding: The padding.
        dilation: The dilation.

    Returns:
        The output padding, can be used to construct a deconvolution
        (conv transpose) layer.

    Raises:
        ValueError: If any argument is invalid, or no output padding
            can satisfy the specified arguments.
    """
    if len(input_size) != len(output_size):
        raise ValueError(
            f'The length of `input_size` != the length of `output_size`: '
            f'got `input_size` {input_size}, and `output_size` {output_size}.'
        )
    if len(input_size) not in (1, 2, 3):
        raise ValueError(
            f'Only 1d, 2d, or 3d `input_size` and `output_size` is supported: '
            f'got `input_size` {input_size}, and `output_size` {output_size}.'
        )

    spatial_ndims = len(input_size)
    kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
    stride = validate_conv_size('stride', stride, spatial_ndims)
    dilation = validate_conv_size('dilation', dilation, spatial_ndims)
    padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)

    def f(i, o, k, s, d, p):
        # o = op + (i - 1) * s - p * 2 + (k - 1) * d + 1
        op = o - ((i - 1) * s - (p[0] + p[1]) + (k - 1) * d + 1)
        if op < 0 or op >= max(s, d):
            raise ValueError(
                f'No `output_padding` can satisfy the deconvolution task: '
                f'input_size == {input_size}, output_size == {output_size}, '
                f'kernel_size == {kernel_size}, stride == {stride}, '
                f'padding == {padding}, dilation == {dilation}.'
            )
        return op

    return [f(*args) for args in zip(
        input_size, output_size, kernel_size, stride, dilation, padding)]
