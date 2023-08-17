from DynaSysML.typing_ import *
from typing import *
import torch
import numpy as np


__all__ = [
    'depth_to_space1d', 'depth_to_space2d', 'depth_to_space3d',
    'space_to_depth1d', 'space_to_depth2d', 'space_to_depth3d',
    'reshape_tail',

    '_flip'
]



def depth_to_space1d(input: Tensor, block_size: int) -> Tensor:
    # input: [batch_shape, depth, spatial_dim], where depth% block_size == 0
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 3:
        raise ValueError('`input` must be at-least 3d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-2]
    if channel_size % block_size != 0:
        raise ValueError('`channel_size` is not multiples of `block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -2]
    L = input_shape[-1]

    output = input.reshape(batch_shape + [block_size, -1, L])
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-2, -1, -3]
    )
    output = output.reshape(batch_shape + [-1, L * block_size])

    return output


def depth_to_space2d(input: Tensor, block_size: int) -> Tensor:
    # inputs: [batch_shape, depth, spatial_dim1, spatial_dim2]
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 4:
        raise ValueError('`input` must be at-least 4d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-3]
    if channel_size % (block_size * block_size) != 0:
        raise ValueError('`channel_size` is not multiples of '
                         '`block_size * block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -3]
    H = input_shape[-2]
    W = input_shape[-1]

    output = input.reshape(batch_shape + [block_size, block_size, -1, H, W])
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-3, -2, -5, -1, -4]
    )
    output = output.reshape(batch_shape + [-1, H * block_size, W * block_size])

    return output


def depth_to_space3d(input: Tensor, block_size: int) -> Tensor:
    # input: [batch_shape, depth, saptial_dim1, spatial_dim2, spatial_dim3]
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 5:
        raise ValueError('`input` must be at-least 5d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-4]
    if channel_size % (block_size * block_size * block_size) != 0:
        raise ValueError('`channel_size` is not multiples of '
                         '`block_size * block_size * block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -4]
    D = input_shape[-3]
    H = input_shape[-2]
    W = input_shape[-1]

    output = input.reshape(
        batch_shape + [block_size, block_size, block_size, -1, D, H, W])
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-4, -3, -7, -2, -6, -1, -5]
    )
    output = output.reshape(
        batch_shape + [-1, D * block_size, H * block_size, W * block_size])

    return output


def space_to_depth1d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 3:
        raise ValueError('`input` must be at-least 3d: got input shape `{}`'.
                         format(input_shape))

    L = input_shape[-1]
    if L % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-1:], block_size))

    # do transformation
    batch_shape = input_shape[: -2]
    channel_size = input_shape[-2]
    L_reduced = L // block_size

    output = input.reshape(batch_shape + [channel_size, L_reduced, block_size])
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-1, -3, -2]
    )
    output = output.reshape(batch_shape + [-1, L_reduced])

    return output


def space_to_depth2d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 4:
        raise ValueError('`input` must be at-least 4d: got input shape `{}`'.
                         format(input_shape))

    H = input_shape[-2]
    W = input_shape[-1]
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-2:], block_size))

    # do transformation
    batch_shape = input_shape[: -3]
    channel_size = input_shape[-3]
    H_reduced = H // block_size
    W_reduced = W // block_size

    output = input.reshape(
        batch_shape +
        [channel_size, H_reduced, block_size, W_reduced, block_size]
    )
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-3, -1, -5, -4, -2]
    )
    output = output.reshape(batch_shape + [-1, H_reduced, W_reduced])

    return output


def space_to_depth3d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < 5:
        raise ValueError('`input` must be at-least 5d: got input shape `{}`'.
                         format(input_shape))

    D = input_shape[-3]
    H = input_shape[-2]
    W = input_shape[-1]
    if D % block_size != 0 or H % block_size != 0 or W % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-3:], block_size))

    # do transformation
    batch_shape = input_shape[: -4]
    channel_size = input_shape[-4]
    D_reduced = D // block_size
    H_reduced = H // block_size
    W_reduced = W // block_size

    output = input.reshape(
        batch_shape +
        [channel_size, D_reduced, block_size, H_reduced, block_size,
         W_reduced, block_size]
    )
    output = output.permute(
        list(range(0, len(batch_shape))) +
        [-5, -3, -1, -7, -6, -4, -2]
    )
    output = output.reshape(batch_shape + [-1, D_reduced, H_reduced, W_reduced])

    return output


def reshape_tail(input: Tensor, ndims: int, shape: List[int]) -> Tensor:
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < ndims:
        raise ValueError(
            '`input` must be at least `ndims`-dimensional: '
            '`shape(input)` is {}, while `ndims` is {}'.
            format(input_shape, ndims)
        )
    left_shape = input_shape[: input_rank - ndims]
    return input.reshape(left_shape + shape)



def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def extract_coeff(a, x_shape):
    if isinstance(a, torch.Tensor) or isinstance(a, np.ndarray):
        return a.reshape(x_shape[0], *((1,)*(len(x_shape)-1)))
    else:
        raise ValueError('the first arg must be tensor for ndarray')