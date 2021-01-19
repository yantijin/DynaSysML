import torch
from typing import *
from .typing_ import *
import math

__all__ = [
    # conv utils
    'validate_conv_size', 'validate_positive_int', 'validate_padding',
    'maybe_as_symmetric_padding', 'validate_output_padding',
    # factory method
    'get_layer_from_layer_or_factory',
    # assert utils
    'is_finite', 'is_all', 'assert_finite',

    # layer argument validators
    'validate_layer', 'validate_layer_factory'
]

# 如果输入为单值或列表， 变成List， 保证每个元素大于0，长度和spatial_ndims相同
def validate_conv_size(name: str,
                       value: Union[int, Sequence[int]],
                       spatial_ndims: int
                       ) -> List[int]:
    if not hasattr(value, '__iter__'):
        value = [int(value)] * spatial_ndims
        value_ok = value[0] > 0
    else:
        value = list(map(int, value))
        value_ok = (len(value) == spatial_ndims) and all(v > 0 for v in value)

    if not value_ok:
        raise ValueError(
            f'`{name}` must be either a positive integer, or a sequence of '
            f'positive integers of length `{spatial_ndims}`: got {value}.'
        )
    return value

# 确保值大于0
def validate_positive_int(arg_name: str, arg_value) -> int:
    ret = int(arg_value)
    if ret <= 0:
        raise ValueError(f'`{arg_name}` must be a positive int: '
                         f'got {arg_value!r}')
    return ret


def validate_padding(padding: 'PaddingArgType',
                     kernel_size: List[int],
                     dilation: List[int],
                     spatial_ndims: int
                     ) -> List[Tuple[int, int]]:
    if padding == 'none':
        return [(0, 0)] * spatial_ndims
    elif padding == 'full':
        ret = []
        for i in range(spatial_ndims):
            p = (kernel_size[i] - 1) * dilation[i]
            ret.append((p, p))
        return ret
    elif padding == 'half':
        padding = []
        for i in range(spatial_ndims):
            val = (kernel_size[i] - 1) * dilation[i]
            p1 = int(val // 2)
            p2 = val - p1
            padding.append((p1, p2))
        return padding
    else:
        if hasattr(padding, '__iter__'):
            ret = list(padding)
        else:
            ret = [padding] * spatial_ndims

        if len(ret) == spatial_ndims:
            for i in range(len(ret)):
                if isinstance(ret[i], tuple):
                    p1, p2 = ret[i]
                    ret[i] = (int(p1), int(p2))
                else:
                    ret[i] = (int(ret[i]),) * 2
        if len(ret) != spatial_ndims or \
                not all(p1 >= 0 and p2 >= 0 for p1, p2 in ret):
            raise ValueError(
                f'`padding` must be a non-negative integer, a sequence of '
                f'non-negative integers of length `{spatial_ndims}`, "none", '
                f'"half" or "full": got {padding}.'
            )
        return ret


def maybe_as_symmetric_padding(padding: List[Tuple[int, int]]
                               ) -> Optional[List[int]]:
    if all(p1 == p2 for p1, p2 in padding):
        return [p1 for p1, _ in padding]

def validate_output_padding(output_padding: Union[int, Sequence[int]],
                            stride: List[int],
                            dilation: List[int],
                            spatial_ndims: int
                            ) -> List[int]:
    if hasattr(output_padding, '__iter__'):
        ret = list(map(int, output_padding))
    else:
        ret = [int(output_padding)] * spatial_ndims

    value_ok = (len(ret) == spatial_ndims)
    if value_ok:
        for i in range(spatial_ndims):
            if ret[i] < 0 or ret[i] >= max(stride[i], dilation[i]):
                value_ok = False
                break

    if not value_ok:
        raise ValueError(
            f'`output_padding` must be a non-negative integer, or a sequence '
            f'of non-negative integers, and must be smaller than either '
            f'`stride` or `dilation`: got `output_padding` {output_padding}, '
            f'`stride` {stride}, and `dilation` {dilation}.'
        )

    return ret



def get_layer_from_layer_or_factory(arg_name: str,
                                    layer_or_layer_factory,
                                    args=(),
                                    kwargs=None) -> 'Module':
    if isinstance(layer_or_layer_factory, Module):
        return layer_or_layer_factory
    elif isinstance(layer_or_layer_factory, type) or \
            callable(layer_or_layer_factory):
        return layer_or_layer_factory(*args, **(kwargs or {}))
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer or a '
                        f'layer factory: got {layer_or_layer_factory!r}.')


def is_finite(input: Tensor) -> Tensor:
    if not input.is_floating_point():
        return torch.ones_like(input).to(torch.bool)
    return (input == input) & (input.abs() != math.inf)


def is_all(condition: Tensor) -> bool:
    return bool(torch.all(condition).item())


def assert_finite(input: Tensor, message: str) -> Tensor:
    if not is_all(is_finite(input)):
        raise ValueError('Infinity or NaN value encountered: {}'.
                         format(message))
    return input


def validate_layer(arg_name: str, layer) -> 'Module':
    # from tensorkit.layers import is_jit_layer
    if isinstance(layer, Module): # or is_jit_layer(layer):
        return layer
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer: got {layer!r}')


def validate_layer_factory(arg_name: str, layer_factory):
    if isinstance(layer_factory, type) or callable(layer_factory):
        return layer_factory
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer factory: '
                        f'got {layer_factory!r}.')



# ######
from .typing_ import *