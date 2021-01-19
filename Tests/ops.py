import copy
from itertools import product
from typing import *

# import numba
import numpy as np

# from tensorkit import tensor as T


__all__ = [
    # dense op
    'dense',

    # convolution shape ops
    'get_spatial_axis', 'get_channel_axis',
    'channel_to_last_nd', 'channel_to_first_nd', 'space_to_depth_nd',
    'make_conv_shape',

    # convolution ops
    'dense', 'conv_nd', 'conv_transpose_nd',

    # pool
    'avg_pool_nd', 'max_pool_nd',

    # other ops
    'pad_constant', #'norm_except_axis',
]
IS_CHANNEL_LAST = False
backend_name = 'PyTorch'

# ---- utilities ----
def list_add(arr, additive):
    if not hasattr(additive, '__iter__'):
        additive = [additive] * len(arr)
    else:
        assert(len(arr) == len(additive))
    return [a + b for a, b in zip(arr, additive)]


def list_mul(arr, multiplicative):
    if not hasattr(multiplicative, '__iter__'):
        multiplicative = [multiplicative] * len(arr)
    else:
        assert(len(arr) == len(multiplicative))
    return [a * b for a, b in zip(arr, multiplicative)]


# ---- configurations according to the backend ----
if IS_CHANNEL_LAST:
    channel_axis = [-1, -1, -1, -1]
    spatial_axis = [[], [-2], [-3, -2], [-4, -3, -2]]


    def get_pixel(im, spatial_pos: Sequence[int], channel: int):
        if len(spatial_pos) == 3:
            x, y, z = spatial_pos
            return im[..., x, y, z, channel]
        elif len(spatial_pos) == 2:
            x, y = spatial_pos
            return im[..., x, y, channel]
        elif len(spatial_pos) == 1:
            x, = spatial_pos
            return im[..., x, channel]


    def set_pixel(im, value, spatial_pos: Sequence[int], channel: int):
        if len(spatial_pos) == 3:
            x, y, z = spatial_pos
            im[..., x, y, z, channel] = value
        elif len(spatial_pos) == 2:
            x, y = spatial_pos
            im[..., x, y, channel] = value
        elif len(spatial_pos) == 1:
            x, = spatial_pos
            im[..., x, channel] = value


    def make_conv_shape(batch_shape, n_channels, spatial_shape):
        return list(batch_shape) + list(spatial_shape) + [n_channels]
else:
    channel_axis = [-1, -2, -3, -4]
    spatial_axis = [[], [-1], [-2, -1], [-3, -2, -1]]


    def get_pixel(im, spatial_pos: Sequence[int], channel: int):
        if len(spatial_pos) == 3:
            x, y, z = spatial_pos
            return im[..., channel, x, y, z]
        elif len(spatial_pos) == 2:
            x, y = spatial_pos
            return im[..., channel, x, y]
        elif len(spatial_pos) == 1:
            x, = spatial_pos
            return im[..., channel, x]


    def set_pixel(im, value, spatial_pos: Sequence[int], channel: int):
        if len(spatial_pos) == 3:
            x, y, z = spatial_pos
            im[..., channel, x, y, z] = value
        elif len(spatial_pos) == 2:
            x, y = spatial_pos
            im[..., channel, x, y] = value
        elif len(spatial_pos) == 1:
            x, = spatial_pos
            im[..., channel, x] = value


    def make_conv_shape(batch_shape, n_channels, spatial_shape):
        return list(batch_shape) + [n_channels] + list(spatial_shape)


def get_channel_axis(spatial_ndims: int):
    return channel_axis[spatial_ndims]


def get_spatial_axis(spatial_ndims: int):
    return spatial_axis[spatial_ndims]


if backend_name == 'PyTorch':
    weight_out_feature_axis = 0
    weight_in_feature_axis = 1


    def get_kernel_pixel(kernel, out_channel, in_channel, spatial_offset):
        k = kernel[out_channel][in_channel]
        for offset in spatial_offset:
            k = k[offset]
        return k
else:
    weight_out_feature_axis = 1
    weight_in_feature_axis = 0


    def get_kernel_pixel(kernel, out_channel, in_channel, spatial_offset):
        k = kernel[in_channel][out_channel]
        for offset in spatial_offset:
            k = k[offset]
        return k

# 对于spatial的维度进行迭代  返回一个迭代器
def iter_spatial_pos(shape,
                     spatial_ndims: int,
                     strip: Union[int, Sequence[int]] = 0
                     ) -> Iterator[Tuple[int, ...]]:
    if not hasattr(strip, '__iter__'):
        strip = (int(strip),) * spatial_ndims
    else:
        assert(len(strip) == spatial_ndims)
        strip = tuple(map(int, strip))
    return product(*(
        range(shape[a] - o)
        for a, o in zip(spatial_axis[spatial_ndims], strip)
    ))


def iter_channel(shape, spatial_ndims: int) -> Iterator[int]:
    return range(shape[channel_axis[spatial_ndims]])


def iter_kernel_offset(kernel_shape):
    return product(*(range(a) for a in kernel_shape))


# ---- convolution shape ops ----
def channel_to_last_nd(x, spatial_ndims: int):
    return np.transpose(
        x,
        list(range(0, len(x.shape) - spatial_ndims - 1)) +
        list(range(len(x.shape) - spatial_ndims, len(x.shape))) +
        [len(x.shape) - spatial_ndims - 1]
    )


def channel_to_first_nd(x, spatial_ndims: int):
    return np.transpose(
        x,
        list(range(0, len(x.shape) - spatial_ndims - 1)) +
        [-1] +
        list(range(len(x.shape) - spatial_ndims - 1, len(x.shape) - 1))
    )


def space_to_depth_nd(x, block_size: int, spatial_ndims: int):
    src_shape = list(x.shape)
    dst_shape = list(src_shape)
    for k in spatial_axis[spatial_ndims]:
        dst_shape[k] //= block_size
    dst_shape[channel_axis[spatial_ndims]] *= block_size ** spatial_ndims

    src = x.reshape([-1] + list(x.shape)[-spatial_ndims - 1:])
    dst = np.zeros([len(src)] + dst_shape[-spatial_ndims - 1:])

    for i in range(len(src)):
        for spatial_pos in iter_spatial_pos(dst.shape, spatial_ndims):
            for channel in iter_channel(dst.shape, spatial_ndims):
                src_channel = channel % x.shape[channel_axis[spatial_ndims]]
                tmp = channel // x.shape[channel_axis[spatial_ndims]]
                tmp_offset = []
                for j in range(spatial_ndims):
                    tmp_offset.append(tmp % block_size)
                    tmp //= block_size
                tmp_offset = list(reversed(tmp_offset))
                src_spatial_pos = [0] * spatial_ndims
                for j in range(spatial_ndims):
                    src_spatial_pos[j] = spatial_pos[j] * block_size + tmp_offset[j]
                set_pixel(dst[i], get_pixel(src[i], src_spatial_pos, src_channel),
                          spatial_pos, channel)

    dst = dst.reshape(dst_shape)
    return dst


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    return 1 + (input_size +
                (padding[0] + padding[1]) -
                (kernel_size - 1) * dilation - 1) // stride


def get_deconv_output_size(input_size, kernel_size, stride, padding, output_padding, dilation):
    return output_padding + (
            (input_size - 1) * stride -
            (padding[0] + padding[1]) +
            (kernel_size - 1) * dilation +
            1
    )


# ---- core linear op ----
def dense(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]
          ) -> np.ndarray:
    if weight_out_feature_axis == 0:
        weight = weight.transpose()
    output = np.dot(x, weight)
    if bias is not None:
        output += bias
    return output


def _conv_nd(input: np.ndarray,
             input_spatial_shape: List[int],
             in_channels: int,
             out_channels: int,
             spatial_ndims: int,
             kernel_size: List[int],
             stride: List[int],
             padding: List[Tuple[int, int]],
             dilation: List[int],
             output_spatial_shape: List[int],
             output_shape: List[int],
             kernel: np.ndarray,
             bias: Optional[np.ndarray]):
    output = np.zeros(output_shape, dtype=input.dtype)

    if IS_CHANNEL_LAST:
        pad = [(0, 0)] + padding + [(0, 0)]
    else:
        pad = [(0, 0), (0, 0)] + padding
    input = np.pad(input, pad, mode='constant', constant_values=0.)

    for i in range(len(output)):
        for out_pos in iter_spatial_pos(output_spatial_shape, spatial_ndims):
            in_pos_base = list_mul(out_pos, stride)
            for out_channel in range(out_channels):
                val = 0.
                for in_channel in range(in_channels):
                    for kernel_offset in iter_kernel_offset(kernel_size):
                        in_pos = list_add(
                            in_pos_base,
                            list_mul(kernel_offset, dilation)
                        )
                        kernel_pixel = get_kernel_pixel(
                            kernel, out_channel, in_channel, kernel_offset)
                        in_pixel = get_pixel(input[i], in_pos, in_channel)
                        val += kernel_pixel * in_pixel
                set_pixel(output[i], val, out_pos, out_channel)

    if bias is not None:
        if not IS_CHANNEL_LAST:
            bias = bias.reshape([-1] + [1] * spatial_ndims)
        output += bias

    return output


def _conv_transpose_nd(input: np.ndarray,
                       input_spatial_shape: List[int],
                       in_channels: int,
                       out_channels: int,
                       spatial_ndims: int,
                       kernel_size: List[int],
                       stride: List[int],
                       padding: List[Tuple[int, int]],
                       output_padding: List[int],
                       dilation: List[int],
                       output_spatial_shape: List[int],
                       output_shape: List[int],
                       kernel: np.ndarray,
                       bias: Optional[np.ndarray]):
    output = np.zeros(output_shape, dtype=input.dtype)
    if IS_CHANNEL_LAST:
        pad = [(0, 0)] + padding + [(0, 0)]
    else:
        pad = [(0, 0), (0, 0)] + padding
    output = np.pad(output, pad, mode='constant', constant_values=0.)

    for i in range(len(output)):
        for in_pos in iter_spatial_pos(input_spatial_shape, spatial_ndims):
            o_pos_base = list_mul(in_pos, stride)
            for in_channel in range(in_channels):
                for out_channel in range(out_channels):
                    for kernel_offset in iter_kernel_offset(kernel_size):
                        out_pos = list_add(
                            o_pos_base,
                            list_mul(kernel_offset, dilation)
                        )
                        kernel_pixel = get_kernel_pixel(
                            kernel, in_channel, out_channel, kernel_offset)
                        in_pixel = get_pixel(input[i], in_pos, in_channel)
                        out_pixel = get_pixel(output[i], out_pos, out_channel)
                        set_pixel(
                            output[i], out_pixel + kernel_pixel * in_pixel,
                            out_pos, out_channel
                        )

    if bias is not None:
        if not IS_CHANNEL_LAST:
            bias = bias.reshape([-1] + [1] * spatial_ndims)
        output += bias

    def unpad_axis(input, axis, left, right):
        if axis == -1:
            return input[..., left: input.shape[-1] - right]
        elif axis == -2:
            return input[..., left: input.shape[-2] - right, :]
        elif axis == -3:
            return input[..., left: input.shape[-3] - right, :, :]
        elif axis == -4:
            return input[..., left: input.shape[-4] - right, :, :, :]
        else:
            raise RuntimeError()

    for axis in spatial_axis[spatial_ndims]:
        output = unpad_axis(output, axis, *pad[axis])

    return output


def conv_nd(input: np.ndarray,
            kernel: np.ndarray,
            bias: Optional[np.ndarray],
            stride: Union[int, List[int]],
            padding: Union[str, int, List[Union[int, Tuple[int, int]]]],
            dilation: Union[int, List[int]],
            ):
    # validate the arguments
    spatial_ndims = len(kernel.shape) - 2
    assert(spatial_ndims in (1, 2, 3))
    assert(len(input.shape) == spatial_ndims + 2)
    kernel_size = list(kernel.shape[2:])
    in_channels = kernel.shape[weight_in_feature_axis]
    assert(in_channels == input.shape[channel_axis[spatial_ndims]])
    out_channels = kernel.shape[weight_out_feature_axis]

    def validate_size(t):
        if hasattr(t, '__iter__'):
            r = list(map(int, t))
            assert(len(r) == spatial_ndims)
            return r
        else:
            return [int(t)] * spatial_ndims

    def validate_size_tuple(t):
        if hasattr(t, '__iter__'):
            r = []
            for w in t:
                if isinstance(w, int):
                    p1, p2 = w, w
                else:
                    p1, p2 = w
                r.append((int(p1), int(p2)))
            assert(len(r) == spatial_ndims)
            return r
        else:
            return [(int(t), int(t))] * spatial_ndims

    stride = validate_size(stride)
    dilation = validate_size(dilation)

    if isinstance(padding, str):
        assert(padding in ('full', 'half', 'none'))
        if padding == 'full':
            padding = [((k - 1) * d, (k - 1) * d)
                       for k, d in zip(kernel_size, dilation)]
        elif padding == 'half':
            padding = []
            for k, d in zip(kernel_size, dilation):
                p1 = (k - 1) * d // 2
                p2 = (k - 1) * d - p1
                padding.append((p1, p2))
        else:
            padding = [(0, 0)] * spatial_ndims
    else:
        padding = validate_size_tuple(padding)

    # do the convolution
    input_spatial_shape = [input.shape[a] for a in spatial_axis[spatial_ndims]]
    output_spatial_shape = [
        get_conv_output_size(a, k, s, p, d)
        for a, k, s, p, d in zip(
            input_spatial_shape, kernel_size, stride, padding, dilation)
    ]

    output_shape = make_conv_shape(
        input.shape[:1],
        out_channels,
        output_spatial_shape
    )
    return _conv_nd(
        input=input,
        input_spatial_shape=input_spatial_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_ndims=spatial_ndims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_spatial_shape=output_spatial_shape,
        output_shape=output_shape,
        kernel=kernel,
        bias=bias,
    )


def conv_transpose_nd(input: np.ndarray,
                      kernel: np.ndarray,
                      bias: Optional[np.ndarray],
                      stride: Union[int, List[int]],
                      padding: Union[str, int, List[Union[int, Tuple[int, int]]]],
                      output_padding: Union[int, List[int]],
                      dilation: Union[int, List[int]],
                      ):
    # validate the arguments
    spatial_ndims = len(kernel.shape) - 2
    assert(spatial_ndims in (1, 2, 3))
    assert(len(input.shape) == spatial_ndims + 2)
    kernel_size = list(kernel.shape[2:])
    in_channels = kernel.shape[weight_out_feature_axis]
    assert(in_channels == input.shape[channel_axis[spatial_ndims]])
    out_channels = kernel.shape[weight_in_feature_axis]

    def validate_size(t):
        if hasattr(t, '__iter__'):
            r = list(map(int, t))
            assert(len(r) == spatial_ndims)
            return r
        else:
            return [int(t)] * spatial_ndims

    def validate_size_tuple(t):
        if hasattr(t, '__iter__'):
            r = []
            for w in t:
                if isinstance(w, int):
                    p1, p2 = w, w
                else:
                    p1, p2 = w
                r.append((int(p1), int(p2)))
            assert(len(r) == spatial_ndims)
            return r
        else:
            return [(int(t), int(t))] * spatial_ndims

    stride = validate_size(stride)
    dilation = validate_size(dilation)

    if isinstance(padding, str):
        assert (padding in ('full', 'half', 'none'))
        if padding == 'full':
            padding = [((k - 1) * d, (k - 1) * d)
                       for k, d in zip(kernel_size, dilation)]
        elif padding == 'half':
            padding = []
            for k, d in zip(kernel_size, dilation):
                p1 = (k - 1) * d // 2
                p2 = (k - 1) * d - p1
                padding.append((p1, p2))
        else:
            padding = [(0, 0)] * spatial_ndims
    else:
        padding = validate_size_tuple(padding)

    if not hasattr(output_padding, '__iter__'):
        output_padding = [int(output_padding)] * spatial_ndims
    else:
        output_padding = list(map(int, output_padding))

    # do the de-convolution
    input_spatial_shape = [input.shape[a] for a in spatial_axis[spatial_ndims]]
    output_spatial_shape = [
        get_deconv_output_size(a, k, s, p, op, d)
        for a, k, s, p, op, d in zip(
            input_spatial_shape, kernel_size, stride, padding, output_padding,
            dilation
        )
    ]

    output_shape = make_conv_shape(
        input.shape[:1],
        out_channels,
        output_spatial_shape
    )
    return _conv_transpose_nd(
        input=input,
        input_spatial_shape=input_spatial_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_ndims=spatial_ndims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        output_spatial_shape=output_spatial_shape,
        output_shape=output_shape,
        kernel=kernel,
        bias=bias,
    )


# ---- pool ----
def _pool_nd(spatial_ndims: int,
             reduce_fn: Callable[[np.ndarray, np.ndarray], float],
             input: np.ndarray,
             kernel_size: Union[int, List[int]],
             stride: Union[int, List[int]],
             padding: Union[int, List[int]],
             dilation: Union[int, List[int]],
             ):
    if not hasattr(kernel_size, '__iter__'):
        kernel_size = [int(kernel_size)] * spatial_ndims
    if not hasattr(stride, '__iter__'):
        stride = [int(stride)] * spatial_ndims
    if not hasattr(padding, '__iter__'):
        padding = [int(padding)] * spatial_ndims
    if not hasattr(dilation, '__iter__'):
        dilation = [int(dilation)] * spatial_ndims

    assert(len(input.shape) == spatial_ndims + 2)
    for arg in (kernel_size, stride, padding, dilation):
        assert(len(arg) == spatial_ndims)

    channel_size = input.shape[channel_axis[spatial_ndims]]

    ##
    output_spatial_shape = [
        get_conv_output_size(a, k, s, (p, p), d)
        for a, k, s, p, d in zip(
            [input.shape[k] for k in spatial_axis[spatial_ndims]],
            kernel_size, stride, padding, dilation
        )
    ]
    output_shape = make_conv_shape(
        [input.shape[0]], channel_size, output_spatial_shape)
    output = np.zeros(output_shape, dtype=input.dtype)

    if IS_CHANNEL_LAST:
        pad = [(0, 0)] + [(p, p) for p in padding] + [(0, 0)]
    else:
        pad = [(0, 0), (0, 0)] + [(p, p) for p in padding]

    mark = np.ones_like(input)
    input = np.pad(input, pad, mode='constant', constant_values=0.)
    mark = np.pad(mark, pad, mode='constant', constant_values=0.)

    for i in range(len(output)):
        for out_pos in iter_spatial_pos(output_spatial_shape, spatial_ndims):
            in_pos_base = list_mul(out_pos, stride)
            for j in range(channel_size):
                val_buf = []
                mark_buf = []

                for kernel_offset in iter_kernel_offset(kernel_size):
                    in_pos = list_add(
                        in_pos_base,
                        list_mul(kernel_offset, dilation)
                    )
                    val_buf.append(get_pixel(input[i], in_pos, j))
                    mark_buf.append(get_pixel(mark[i], in_pos, j))
                val = reduce_fn(np.array(val_buf), np.array(mark_buf))
                set_pixel(output[i], val, out_pos, j)

    return output


def avg_pool_nd(spatial_ndims: int,
                input: np.ndarray,
                kernel_size: Union[int, List[int]],
                stride: Union[int, List[int]],
                padding: Union[int, List[int]],
                count_padded_zeros: bool = False): # T.nn.AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS):
    if count_padded_zeros:
        def reduce_fn(val, mark):
            return np.mean(val)
    else:
        def reduce_fn(val, mark):
            return np.sum(val) / np.sum(mark)

    return _pool_nd(
        spatial_ndims=spatial_ndims,
        reduce_fn=reduce_fn,
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
    )


def max_pool_nd(spatial_ndims: int,
                input: np.ndarray,
                kernel_size: Union[int, List[int]],
                stride: Union[int, List[int]],
                padding: Union[int, List[int]]):
    def reduce_fn(val, mark):
        return np.max(val)

    return _pool_nd(
        spatial_ndims=spatial_ndims,
        reduce_fn=reduce_fn,
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
    )


# ---- other ops ----
def pad_constant(input: np.ndarray,
                 pad: Sequence[Union[int, Tuple[int, int]]],
                 value: float = 0.) -> np.ndarray:
    pad = list(pad)
    for i in range(len(pad)):
        if not isinstance(pad[i], tuple):
            pad[i] = (int(pad[i]),) * 2
        else:
            pad[i] = tuple(map(int, pad[i]))
            assert(len(pad[i]) == 2)
    if len(pad) < len(input.shape):
        pad = [(0, 0)] * (len(input.shape) - len(pad)) + pad
    elif len(pad) > len(input.shape):
        raise ValueError()
    return np.pad(input, pad, mode='constant', constant_values=value)


def norm_except_axis(input: np.ndarray,
                     axis: Union[int, Sequence[int]],
                     p: float = 2.,
                     keepdims: bool = False) -> np.ndarray:
    if axis is None:
        reduce_axis = None
    else:
        if not hasattr(axis, '__iter__'):
            axis = [int(axis)]
        else:
            axis = list(map(int, axis))

        axis_mark = [False] * len(input.shape)
        for a in axis:
            axis_mark[a] = True
        reduce_axis = tuple(
            [i for i, m in zip(range(len(input.shape)), axis_mark) if not m]
        )

    p_inv = 1. / p
    return np.power(
        np.sum(np.power(np.abs(input), p), axis=reduce_axis, keepdims=keepdims),
        p_inv
    )
