import torch
from typing import *
from .typing_ import *
import numpy as np

__all__ = [
    # utils
    'add_parameter', 'get_parameter', 'get_parameters',
    'add_buffer', 'get_buffer', 'get_buffers',
    'set_train_mode',


    'pad', 'index_select', 'shift',

    'to_numpy',

    # variable
    'variable',

    # shape utils
    'squeeze', 'flatten_to_ndims', 'unflatten_from_ndims',
    'broadcast_to', 'broadcast_shape',

    # dtypes
    'get_dtype',

    # reduction calculation
    'reduce_sum', 'reduce_max',
    'reduce_mean', 'reduce_min', 'log_mean_exp', 'log_sum_exp',

    # 'all' or 'any'
    'norm_except_axis',

    # assign to variable
    'fill', 'fill_zeros', 'assign_data',

    # tensor constructors
    'zeros', 'as_tensor', 'ones_like',

    # mean std
    'calculate_mean_and_var',

    'current_device', 'CPU_DEVICE'

]

# ---- utils ----
def add_parameter(layer: Module,
                  name: str,
                  value: Optional[Tensor],
                  requires_grad: bool = True
                  ) -> Optional[torch.Tensor]:
    if value is not None:
        v = torch.nn.Parameter(value, requires_grad=requires_grad)
    # FIXME: 这个位置有点问题，如果是None的话还可以调用吗
    else:
        v = value
    layer.register_parameter(name, v)
    return v


def get_parameter(layer: Module, name: str) -> Optional[torch.Tensor]:
    return getattr(layer, name)


def get_parameters(layer: Module, recursive: bool = True
                   ) -> Iterator[Tuple[str, torch.Tensor]]:
    return layer.named_parameters(recurse=recursive)


def add_buffer(layer: Module,
               name: str,
               value: Optional[Tensor]
               ) -> Optional[Tensor]:
    layer.register_buffer(name, value)
    return value


def get_buffer(layer: Module, name: str) -> Optional[Tensor]:
    return getattr(layer, name)


def get_buffers(layer: Module, recursive: bool = True
                ) -> Iterator[Tuple[str, Tensor]]:
    return layer.named_buffers(recurse=recursive)


def set_train_mode(layer: Module, training: bool = True):
    layer.train(training)
    return layer


def pad(input: torch.Tensor,
        padding: List[Tuple[int, int]],
        value: float = 0.) -> torch.Tensor:
    if len(padding) > input.dim():
        raise ValueError(
            'The length of `padding` must not be larger than `rank(input)`: '
            '`padding` is {}, while `shape(input)` is {}'.
            format(padding, list(input.shape))
        )
    pad: List[int] = []
    for i in range(len(padding) - 1, -1, -1):
        pad.extend(padding[i])
    return torch.nn.functional.pad(input, pad=pad, value=value)


def index_select(input: Tensor, indices: Tensor, axis: int) -> Tensor:
    x_shape = input.shape
    i_shape = indices.shape

    if axis < 0:
        axis += len(x_shape)
    if axis < 0 or axis >= len(x_shape):
        raise ValueError('`axis` out of range: x.shape {} vs axis {}'.
                         format(input.shape, axis))

    if len(i_shape) == 0:
        y = torch.index_select(input, dim=axis, index=indices.reshape([1]))
        y = y.reshape(x_shape[:axis] + x_shape[axis + 1:])

    elif len(i_shape) == 1:
        y = torch.index_select(input, dim=axis, index=indices)

    else:
        y = torch.index_select(input, dim=axis, index=indices.flatten())
        y = y.reshape(x_shape[:axis] + i_shape + x_shape[axis + 1:])

    return y


def to_numpy(input: Tensor) -> np.ndarray:
    if not isinstance(input, Tensor):
        raise TypeError(f'Not a Tensor: got {input!r}')
    return input.detach().cpu().numpy()


# shape utils
def squeeze(input: Tensor, axis: Optional[Union[int,List[int]]] = None) -> Tensor:
    if axis is not None:
        if isinstance(axis, int):
            return torch.squeeze(input, axis)
        elif len(axis) == 1:
            return torch.squeeze(input, axis[0])
        else:
            old_shape = input.shape
            new_shape_mask = [True] * len(old_shape)
            for a in axis:
                if old_shape[a] == 1:
                    new_shape_mask[a] = False
                else:
                    raise ValueError('Axis {} cannot be squeezed, since its '
                                     'size is {} != 1'.format(a, old_shape[a]))
            # new_shape = torch.jit.annotate(List[int], [])
            new_shape = []
            for i in range(len(old_shape)):
                if new_shape_mask[i]:
                    new_shape.append(old_shape[i])
            return input.reshape(new_shape)
    else:
        return torch.squeeze(input)

# @jit
def flatten_to_ndims(input: Tensor, ndims: int
                     ) -> Tuple[Tensor, Optional[List[int]]]:
    if ndims < 1:
        raise ValueError('`ndims` must be at least 1`: got ndims {}'.
                         format(ndims))
    if len(input.shape) < ndims:
        raise ValueError('rank(x) < ndims: x.shape is {}, while '
                         'ndims is {}'.format(input.shape, ndims))

    if ndims == len(input.shape):
        return input, None  # `None` to indicate x is not changed
    elif ndims == 1:
        front_shape = list(input.shape)
        return input.reshape((-1,)), front_shape
    else:
        x_shape = list(input.shape)
        offset = ndims - 1
        front_shape, back_shape = x_shape[: -offset], x_shape[-offset:]
        return input.reshape([-1] + back_shape), front_shape


# @jit
def unflatten_from_ndims(input: Tensor, front_shape: Optional[List[int]]
                         ) -> Tensor:
    x_shape = list(input.shape)
    if front_shape is None:
        return input
    else:
        x_rank = len(x_shape)
        if x_rank < 1:
            raise ValueError(
                'Invalid input: rank(x) < 1, but front_shape is not None.')
        return input.reshape(list(front_shape) + x_shape[1:])


# @jit
def _broadcast_to_sub(t: Tensor,
                      t_shape: List[int],
                      out_shape: List[int]) -> Tensor:
    t_rank = len(t_shape)
    out_rank = len(out_shape)

    if t_rank < out_rank:
        t_shape = [1] * (out_rank - t_rank) + t_shape

    t_repeats = [] # torch.jit.annotate(List[int], [])
    should_repeat = False
    for i in range(out_rank):
        a = t_shape[i]
        b = out_shape[i]
        if a == 1 and b != 1:
            t_repeats.append(b)
            should_repeat = True
        else:
            t_repeats.append(1)

    if should_repeat:
        t = t.repeat(t_repeats)
    return t


# @jit
def broadcast_to(input: Tensor, new_shape: List[int]) -> Tensor:
    x_shape = list(input.shape)
    x_rank = len(x_shape)
    new_rank = len(new_shape)

    if x_rank > new_rank:
        raise ValueError('`x` cannot be broadcast to `new_shape`: shape(x) {} '
                         'vs new_shape {}'.format(x_shape, new_shape))

    for i in range(x_rank):
        a = x_shape[-i - 1]
        b = new_shape[-i - 1]
        if a != 1 and a != b:
            raise ValueError('`x` cannot be broadcast to `new_shape`: '
                             'shape(x) {} vs new_shape {}'.
                             format(x_shape, new_shape))

    return _broadcast_to_sub(input, x_shape, new_shape)


def broadcast_shape(x: List[int], y: List[int]) -> List[int]:
    common_len = min(len(x), len(y))

    right = [] # torch.jit.annotate(List[int], [])
    for i in range(common_len):
        a = x[i - common_len]
        b = y[i - common_len]
        if a == 1:
            right.append(b)
        elif b == 1:
            right.append(a)
        elif a != b:
            raise ValueError('Shape x and y cannot broadcast against '
                             'each other: {} vs {}.'.format(x, y))
        else:
            right.append(a)

    if len(x) > common_len:
        left = x[:len(x)-common_len]
    else:
        left = y[:len(y)-common_len]
    return left + right


# ---- assignment ----
def fill(dst: Tensor, fill_value: float) -> Tensor:
    dst.fill_(fill_value)
    return dst


def fill_zeros(dst: Tensor) -> Tensor:
    dst.zero_()
    return dst


def assign_data(dst: Tensor, src) -> Tensor:
    src = as_tensor(src, force_copy=True).detach()
    if src.shape != dst.shape:
        raise ValueError('`dst.shape` != `src.shape`: {} vs {}'.
                         format(dst.shape, src.shape))
    dst.copy_(src.detach())
    return dst

def variable(shape: List[int],
             dtype: Union[str, torch.dtype] = torch.float32,
             device: Optional[str] = None,
             initializer: Optional[
                 Union[
                     int, float, np.ndarray, Tensor,
                     Callable[[torch.Tensor], None]
                 ]
             ] = None,
             requires_grad: bool = True,
             force_copy: bool = True) -> torch.Tensor:
    """
    Create a new variable.

    Args:
        shape: Shape of the variable.
        dtype: Dtype of the variable.
        initializer: The variable initializer.  It may be a scalar (which
            will be filled into the new variable), an array or another
            `Tensor` with the same shape as specified `shape`, or a callable
            function that can be used to initialize the variable.
        requires_grad: Whether or not that the variable requires gradient
            during back-propagation?  Defaults to :obj:`True`.
        force_copy: Whether or not to force copy the data from `initializer`,
            even if the backend supports sharing memory?
            Defaults to :obj:`True`.

    Returns:
        The created variable.
    """
    if isinstance(dtype, str):
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = dtype

    if device is None:
        device = current_device()

    if isinstance(initializer, (int, float)):
        ret = torch.full(shape, float(initializer), dtype=target_dtype,
                         requires_grad=requires_grad, device=device)
    elif isinstance(initializer, np.ndarray) and initializer.shape == ():
        ret = torch.full(shape, initializer.tolist(), dtype=target_dtype,
                         requires_grad=requires_grad, device=device)
    elif isinstance(initializer, (np.ndarray, Tensor)):
        if list(initializer.shape) != shape:
            raise ValueError(f'`initializer.shape` != `shape`: '
                             f'{list(initializer.shape)} vs {shape}')
        ret = as_tensor(initializer, dtype=target_dtype,
                        force_copy=force_copy, device=device)
        if requires_grad:
            ret.requires_grad_(True)
    elif isinstance(initializer, Callable):
        ret = zeros(shape, dtype=dtype, device=device)
        with torch.no_grad():
            initializer(ret)
        if requires_grad:
            ret.requires_grad_(True)
    elif initializer is None:
        ret = torch.zeros(shape, dtype=target_dtype, device=device,
                          requires_grad=requires_grad)
    else:
        raise TypeError(f'Unsupported initializer: {initializer!r}')

    return ret



# tensor constructors
def as_tensor(data,
              dtype: Optional[Union[torch.dtype, str]] = None,
              force_copy: bool = False,
              device: Optional[str]=None) -> Tensor:
    """
    Construct a new tensor from `data`.

    This method will copy `data` only when it is required to do so, or
    when `force_copy` is set to :obj:`True`.

    Args:
        data: The tensor data.  It might be a Python number, a NumPy array,
            another tensor, a :class:`~tensorkit.StochasticTensor`, or anything
            else that the backend supports.
        dtype: The expected dtype of the constructed tensor.
        force_copy: Force to copy `data` even if it is not necessary.
            The gradient propagation will not be stopped from the copied tensor
            to the original tensor.  The caller may need to use `T.stop_grad()`
            if necessary.

            It should not be necessary to copy the given `data`, if `data`
            is already another tensor with `dtype`; or if `data` is a NumPy
            array with compatible `dtype`, and the backend supports to share
            memory between a tensor and a NumPy array.

    Returns:
        The constructed tensor.
    """
    # from tensorkit import StochasticTensor

    # check the dtype argument
    target_dtype = dtype
    if dtype is not None:
        if not isinstance(dtype, torch.dtype):
            if dtype == 'float32':
                target_dtype = torch.float32
            elif dtype == 'int32':
                target_dtype = torch.int32
            else:
                target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    # if `data` is already a tensor
    # if isinstance(data, StochasticTensor):
    #     data = data.tensor
    if device is None:
        device = current_device()

    if isinstance(data, Tensor):
        # input `data` may be `StochasticTensor`, `Tensor` or `numpy.ndarray`
        from_dev = str(data.device)
        if data.dtype != target_dtype and from_dev != device:
            data = data.to(dtype=target_dtype, device=device)
        elif data.dtype != target_dtype:
            data = data.to(target_dtype)
        elif from_dev != device:
            data = data.to(device=device)

        if force_copy:
            data = data.clone()
        return data

    # or if `data` is other types
    ret = torch.as_tensor(data, dtype=target_dtype, device=device)
    if force_copy:
        ret = ret.clone()
    return ret


def ones_like(input: Tensor,
              dtype: Optional[str] = None,
              shape: Optional[List[int]] = None) -> Tensor:
    if dtype is not None:
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = input.dtype
    if shape is None:
        shape = list(input.shape)
    return torch.ones(shape, dtype=target_dtype, device=input.device)


def zeros(shape: List[int],
          dtype: Union[str, torch.dtype] = 'float32',
          device: Optional[str] = None) -> Tensor:
    if isinstance(dtype, str):
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    elif isinstance(dtype, torch.dtype):
        target_dtype = dtype
    else:
        raise ValueError('`dtype` should be a str or torch.dtype, please check your inputs')

    if device is None:
        device = current_device()
    return torch.zeros(shape, dtype=target_dtype, device=device)


# reduction calculation
def reduce_sum(input: Tensor,
               axis: Optional[Union[List[int], int]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.sum(input).reshape([1] * len(input.shape))
        else:
            return torch.sum(input)
    elif isinstance(axis, int):
        return torch.sum(input, dim=axis, keepdim=keepdims)
    else:
        if len(axis) == 0:
            return input
        else:
            return torch.sum(input, dim=axis, keepdim=keepdims)



def reduce_mean(input: Tensor,
                axis: Optional[Union[List[int], int]] = None,
                keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.mean(input).reshape([1] * len(input.shape))
        else:
            return torch.mean(input)
    elif isinstance(axis, int):
        return torch.mean(input, dim=axis, keepdim=keepdims)
    else:
        if len(axis) == 0:
            return input
        else:
            return torch.mean(input, dim=axis, keepdim=keepdims)



def reduce_max(input: Tensor,
               axis: Optional[Union[int,List[int]]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.max(input).reshape([1] * len(input.shape))
        else:
            return torch.max(input)
    elif isinstance(axis, int):
        return torch.max(input, dim=axis, keepdim=keepdims)[0]
    else:
        if len(axis) == 0:
            return input
        elif len(axis) == 1:
            return torch.max(input, dim=axis[0], keepdim=keepdims)[0]
        else:
            for a in axis:
                input = torch.max(input, dim=a, keepdim=True)[0]
            if not keepdims:
                input = squeeze(input, axis)
            return input



def reduce_min(input: Tensor,
               axis: Optional[Union[int, List[int]]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.min(input).reshape([1] * len(input.shape))
        else:
            return torch.min(input)
    elif isinstance(axis, int):
        return torch.min(input, dim=axis, keepdim=keepdims)[0]
    else:
        if len(axis) == 0:
            return input
        elif len(axis) == 1:
            return torch.min(input, dim=axis[0], keepdim=keepdims)[0]
        else:
            for a in axis:
                input = torch.min(input, dim=a, keepdim=True)[0]
            if not keepdims:
                input = squeeze(input, axis)
            return input



def log_sum_exp(input: Tensor,
                axis: Optional[Union[List[int], int]] = None,
                keepdims: bool = False) -> Tensor:
    if axis is None:
        axis = list(range(0, len(input.shape)))
        if keepdims:
            return torch.logsumexp(input, dim=axis, keepdim=True)
        else:
            return torch.logsumexp(input, dim=axis, keepdim=False)
    elif isinstance(axis, int):
        return torch.logsumexp(input, dim=axis, keepdim=keepdims)
    else:
        if len(axis) == 0:
            raise ValueError('`axis` must not be an empty list.')
        return torch.logsumexp(input, dim=axis, keepdim=keepdims)



def log_mean_exp(input: Tensor,
                 axis: Optional[Union[int, List[int]]] = None,
                 keepdims: bool = False) -> Tensor:
    if axis is not None and not isinstance(axis, int):
        if len(axis) == 0:
            raise ValueError('`axis` must not be an empty list.')
    x_max_keepdims = reduce_max(input, axis=axis, keepdims=True)
    if not keepdims:
        x_max = squeeze(x_max_keepdims, axis=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean(
        torch.exp(input - x_max_keepdims), axis=axis, keepdims=keepdims)
    return x_max + torch.log(mean_exp)




def norm_except_axis(input: Tensor,
                     axis: Optional[Union[int, List[int]]],
                     p: float = 2,
                     keepdims: bool = False) -> Tensor:
    """
    Calculate the Lp-norm of a tensor except for specified axis.

    Args:
        input: The input tensor.
        axis: The axis to keep for computing Lp-norm.
            All other axis will be reduced.  Defaults to :obj:`None`,
            where no axis will be kept.
        p: The `p` of the `Lp` norm.  Defaults to 2.
        keepdims: Whether or not to keep the reduced dimensions?
            Defaults to :obj:`False`.

    Returns:
        The Lp-norm of the tensor.
    """
    r = input.dim()
    if axis is None:
        axis_reduce = None
    elif isinstance(axis, int):
        a = axis
        if a < -r or a >= r:
            raise ValueError(
                f'`axis` out of range: `axis` is {axis}, '
                f'while the shape of `input` is {input.shape}.')
        if a < 0:
            a = a + r
        axis_reduce = list(range(0, a)) + list(range(a + 1, r))
    elif len(axis) == 1:
        # compute the axis to reduce in a fast manner
        a = axis[0]
        if a < -r or a >= r:
            raise ValueError(
                f'`axis` out of range: `axis` is {axis}, '
                f'while the shape of `input` is {input.shape}.')
        if a < 0:
            a = a + r
        axis_reduce = list(range(0, a)) + list(range(a + 1, r))
    else:
        # compute the axis to reduce in a slow manner
        axis_mask: List[bool] = [True] * r
        for a in axis:
            if a < -r or a >= r:
                raise ValueError(
                    f'`axis` out of range: `axis` is {axis}, '
                    f'while the shape of `input` is {input.shape}.')
            axis_mask[a] = False
        axis_reduce: List[int] = []
        for i in range(r):
            if axis_mask[i]:
                axis_reduce.append(i)

    if p == 2:
        return torch.sqrt(reduce_sum(input ** 2, axis=axis_reduce, keepdims=keepdims))
    elif p == 1:
        return reduce_sum(torch.abs(input), axis=axis_reduce, keepdims=keepdims)
    else:
        p_inv = 1. / p
        return torch.pow(
            reduce_sum(torch.pow(torch.abs(input), p), axis=axis_reduce, keepdims=keepdims),
            p_inv
        )


# dtypes
def get_dtype(input: Tensor) -> str:
    if input.dtype == torch.float32:
        return 'float32'
    elif input.dtype == torch.int32:
        return 'int32'
    else:
        return {torch.int8: 'int8', torch.uint8: 'uint8', torch.int16: 'int16', torch.int64: 'int64', torch.float16: 'float16', torch.float64: 'float64', torch.bool: 'bool'}[input.dtype]


def calculate_mean_and_var(input: Tensor,
                           axis: Optional[List[int]] = None,
                           keepdims: bool = False,
                           unbiased: bool = True) -> Tuple[Tensor, Tensor]:
    # compute mean & var
    mean = reduce_mean(input, axis=axis, keepdims=True)
    var = reduce_mean((input - mean) ** 2, axis=axis, keepdims=keepdims)
    if not keepdims:
        mean = mean.reshape(var.shape)

    reduce_size = input.numel() // mean.numel()
    if reduce_size < 2:
        raise RuntimeError(
            'Variance can only be calculated with at least 2 samples.')

    # obtain unbiased estimator from the biased estimator by multiply n / (n-1)
    if unbiased:
        var = var * (float(reduce_size) / (reduce_size - 1.))

    return mean, var

def shift(input: Tensor,
          shift: List[int],
          fill_value: float = 0.) -> Tensor:
    shift_length = len(shift)
    if shift_length > input.dim():
        raise ValueError('`len(shift) <= rank(input)` does not hold: '
                         'got `shift` {}, and `shape(input)` {}.'.
                         format(shift, list(input.shape)))

    padding: List[int] = []
    need_pad: bool = False

    for axis in range(-1, -(shift_length + 1), -1):
        s = shift[axis]
        size = input.shape[axis]
        if s < -size or s > size:
            raise ValueError(
                '`shift` out of range at axis {}: expected to be >= {} '
                'and <= {}.'.format(axis, -size, size)
            )
        if s < 0:
            padding.append(0)
            padding.append(-s)
            input = torch.narrow(input, axis, -s, size + s)
            need_pad = True
        elif s > 0:
            padding.append(s)
            padding.append(0)
            input = torch.narrow(input, axis, 0, size - s)
            need_pad = True
        else:
            padding.append(0)
            padding.append(0)
        axis -= 1

    if need_pad:
        input = torch.nn.functional.pad(
            input, padding, mode='constant', value=fill_value)

    return input


CPU_DEVICE = 'cpu'
_current_device = [CPU_DEVICE]


def current_device() -> str:
    return _current_device[0]