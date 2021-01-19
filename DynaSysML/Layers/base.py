import torch
import torch.nn as nn
from typing import *
from DynaSysML.typing_ import *
from . import initializer
from DynaSysML.arg_check import *
from DynaSysML.core import variable, assign_data, add_parameter, \
    norm_except_axis, as_tensor, get_dtype, flatten_to_ndims, unflatten_from_ndims, \
    pad, current_device, CPU_DEVICE

__all__ = [
    'DEFAULT_BIAS_INIT', 'DEFAULT_WEIGHT_INIT', 'EPSILON', 'DEFAULT_GATE_BIAS',
    'IS_CHANNEL_LAST',
    # Paramstores
    'BaseParamStore', 'SimpleParamStore', 'NormedWeightStore',
    'weight_norm_decompose', 'NormedAndScaledWeightStore',
    'get_weight_store', 'get_bias_store',

    # identity layer
    'Identity',

    # layers
    'BaseLayer', 'BaseSingleVariateLayer', 'BaseMultiVariateLayer',
    'BaseSplitLayer', 'BaseMergeLayer', 'BaseContextualLayer',
    'BaseMultiVariateContextualLayer',

    # sequential
    'Sequential',

    # Linear layers
    'CoreLinear', 'Linear',
    'LinearConv1d', 'LinearConv2d', 'LinearConv3d',
    'LinearConvTranspose1d', 'LinearConvTranspose2d', 'LinearConvTranspose3d',

    # normalizer layers
    'BatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',

    # dropout layers
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
]

DEFAULT_WEIGHT_INIT = initializer.kaming_uniform
DEFAULT_BIAS_INIT = initializer.zeros
EPSILON = 1e-5
DEFAULT_GATE_BIAS: float = 2.0
IS_CHANNEL_LAST = False



def layer_to_device(layer: Module, device: Optional[str] = None) -> Module:
    """
    Move the specified module or layer to the given device.
    The module or layer may be changed in-place.

    Args:
        layer: The module or layer to be moved.
        device: The device, to where move the module or layer.
            If not specified, will move to ``T.current_device()``.

    Returns:
        The layer instance.
    """
    if device is None:
        device = current_device()
    layer = layer.to(device=torch.device(device))
    return layer


# ---- weight wrapper: a simple weight, or a normed weight ----
class BaseParamStore(Module):
    """
    Base class for a component that stores a trainable parameter,
    or a set of trainable parameters that can be used to derive
    virtually "a parameter" (e.g., weight-normed `weight`).
    """

    __constants__ = ('shape',)

    shape: List[int]

    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = list(map(int, shape))

    def extra_repr(self) -> str:
        return f'shape={self.shape}'

    def forward(self) -> Tensor:
        return self.get()

    def get(self) -> Tensor:
        raise NotImplementedError()

    def set(self, value: TensorOrData) -> None:
        raise NotImplementedError()


class SimpleParamStore(BaseParamStore):
    """A module that carries a direct variable as the parameter."""

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 device: Optional[str]=None):
        super().__init__(shape)
        device = device or current_device()
        add_parameter(self, 'value', variable(shape,
                                              initializer=initializer,
                                              device=device))

    def get(self) -> Tensor:
        return self.value

    def set(self, value: TensorOrData) -> None:
        with torch.no_grad():
            assign_data(self.value, value)


def weight_norm_decompose(weight: Tensor,
                          norm_axis: int,
                          epsilon: float
                          ) -> Tuple[Tensor, Tensor]:
    """
    Decompose `weight` by "weight-norm", i.e., into direction part `v`
    and norm part `v_norm`.

    Args:
        weight: The weight to be decomposed.
        norm_axis: The axis, along with to calculate the weight norm.
        epsilon: Infinitesimal constant to avoid dividing zero.

    Returns:
        A tuple of `(v, v_norm)`.
    """
    v_norm = norm_except_axis(weight, axis=[norm_axis], keepdims=True)
    v = weight / torch.max(v_norm, torch.as_tensor(epsilon, dtype=v_norm.dtype))
    return v, v_norm


class NormedWeightStore(BaseParamStore):
    """A module that carries the weight-normed `weight`, without `g`."""

    __constants__ = BaseParamStore.__constants__ + ('feature_axis', 'epsilon')

    norm_axis: int
    epsilon: float

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 norm_axis: int = 1,
                 device: Optional[str]=None,
                 epsilon: float = EPSILON):
        super().__init__(shape)
        self.norm_axis = norm_axis
        device = device or current_device()
        self.epsilon = epsilon

        weight = variable(shape, initializer=initializer, device=device)
        with torch.no_grad():
            v, _ = weight_norm_decompose(weight, norm_axis, epsilon)
        add_parameter(self, 'v', v)

    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.norm_axis, self.epsilon)
        return v

    def set(self, value: TensorOrData) -> None:
        with torch.no_grad():
            v, _ = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v), device=str(self.v.device)),
                self.norm_axis,
                self.epsilon,
            )
            assign_data(self.v, v)


class NormedAndScaledWeightStore(BaseParamStore):
    """A module that carries the weight-normed `weight`, with `v` and `g`."""

    __constants__ = BaseParamStore.__constants__ + ('feature_axis', 'epsilon')

    norm_axis: int
    epsilon: float

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 norm_axis: int = 1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        super().__init__(shape)
        self.norm_axis = norm_axis
        self.epsilon = epsilon
        device = device or current_device()

        weight = variable(shape, initializer=initializer, device=device)
        with torch.no_grad():
            v, g = weight_norm_decompose(weight, norm_axis, epsilon)
        add_parameter(self, 'v', v)
        add_parameter(self, 'g', g)

    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.norm_axis, self.epsilon)
        return self.g * v

    def set(self, value: TensorOrData) -> None:
        with torch.no_grad():
            v, g = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v), device=str(self.v.device)),
                self.norm_axis,
                self.epsilon,
            )
            assign_data(self.v, v)
            assign_data(self.g, g)


def get_weight_store(shape: List[int],
                     initializer: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                     norm_axis: int = 1,
                     weight_norm: WeightNormArgType = False,
                     device: Optional[str]=None
                     ) -> BaseParamStore:
    """
    Create a module which carries the `weight` parameter.

    Args:
        shape: The shape of the weight.
        initializer: The initializer for the weight.
        norm_axis: The axis, along with to normalize the weight.
        weight_norm: The mode of weight norm.
            Use `NormedAndScaledWeightStore` if `True` or `WeightNormMode.FULL`.
            Use `NormedWeightStore` if `WeightNormMode.NO_SCALE`.
            Use `WeightStore` if `False` or `WeightNormMode.NONE`.

    Returns:
        The weight object.
    """
    device = device or current_device()
    if weight_norm is True or weight_norm == WeightNormMode.FULL:
        return NormedAndScaledWeightStore(shape, initializer, norm_axis, device)
    elif weight_norm == WeightNormMode.NO_SCALE:
        return NormedWeightStore(shape, initializer, norm_axis, device)
    elif weight_norm is False or weight_norm == WeightNormMode.NONE:
        return SimpleParamStore(shape, initializer, device)
    else:
        raise ValueError(f'Invalid value for argument `weight_norm`: '
                         f'{weight_norm!r}.')


def get_bias_store(shape: List[int],
                   initializer: TensorInitArgType = DEFAULT_BIAS_INIT,
                   use_bias: bool = True,
                   device: Optional[str]=None
                   ) -> Optional[BaseParamStore]:
    """
    Create a module that carries the `bias` parameter.

    Args:
        shape: The shape of the bias.
        initializer: The initializer for the bias.
        use_bias: Whether or not to use the bias?
            If `False`, will return :obj:`None`.

    Returns:
        The bias object, or :obj:`None` if `use_bias` is False.
    """
    device =device or current_device()
    if use_bias:
        return SimpleParamStore(shape, initializer, device)



class Identity(Module):

    def forward(self, input: Tensor) -> Tensor:
        return input


class BaseLayer(nn.Module):
    '''
    BaseLayer inherited from `nn.Module`.
    '''
    def extra_repr(self) -> str:
        buf = []
        attributes = list(getattr(self, '__constants__', ()))

        for attr in getattr(self, '__annotations__', ()):
            if attr not in attributes:
                attributes.append(attr)

        for attr in attributes:
            if attr.startswith('_'):
                continue
            attr_val = getattr(self, attr, None)
            if attr_val is None or isinstance(attr_val, nn.Module) or \
                    isinstance(attr_val, torch.Tensor):
                continue
            buf.append(f'{attr}={attr_val!r}')

        return ', '.join(buf)


class BaseSingleVariateLayer(BaseLayer):
    """
    Base class for single-input, single-output layers.

    Sub-classes should override `_call(input: Tensor) -> Tensor` to
    actually implement the module.
    """

    def _forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        return self._forward(input)


class BaseMultiVariateLayer(BaseLayer):
    """
    Base class for multiple-input, multiple-output layers.
    The inputs and outputs should be given as a list of Tensors.
    """

    def _forward(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        return self._forward(inputs)


class BaseSplitLayer(BaseLayer):
    """
    Base class for single-input, multiple-output layers.
    The outputs should be given as a list of Tensors.
    """

    def _forward(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> List[Tensor]:
        return self._forward(input)


class BaseMergeLayer(BaseLayer):
    """
    Base class for multiple-input, single-output layers.
    The inputs should be given as a list of Tensors.
    """

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return self._forward(inputs)


class BaseContextualLayer(BaseLayer):
    """
    Base class layers that produces the output according to the input tensor
    and contextual tensors.
    """

    def _forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                context: Optional[List[Tensor]] = None) -> Tensor:
        if context is None:
            context = []
        return self._forward(input, context)


class BaseMultiVariateContextualLayer(BaseLayer):
    """
    Base class layers that produces the output tensors according to the
    input tensors and contextual tensors.
    """

    def _forward(self, inputs: List[Tensor], context: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()

    def forward(self,
                inputs: List[Tensor],
                context: Optional[List[Tensor]] = None) -> List[Tensor]:
        if context is None:
            context = []
        return self._forward(inputs, context)


class Sequential(torch.nn.Sequential):

    def __init__(self, *layers: Union[Module, Sequence[Module]]):
        from .utils import flatten_nested_layers
        super().__init__(*flatten_nested_layers(layers))


class CoreLinear(BaseLayer):
    """Base class for the core linear layers."""

    __constants__ = (
        # modules
        'weight_store', 'bias_store',

        # attributes
        'in_features', 'out_features', 'in_channels', 'out_channels',
        'kernel_size', 'stride', 'padding', '_symmetric_padding',
        'output_padding', 'dilation', 'use_bias',
    )

    weight_store: Module
    bias_store: Optional[Module]

    def __init__(self,
                 weight_shape: List[int],
                 bias_shape: List[int],
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        device = device or current_device()
        weight_store = get_weight_store(
            weight_shape, initializer=weight_init,
            weight_norm=weight_norm, device=device)
        bias_store = get_bias_store(
            bias_shape, initializer=bias_init,
            use_bias=use_bias, device=device)

        if data_init is not None:
            if not isinstance(data_init, initializer.DataDependentInitializer) and \
                    (isinstance(data_init, type) or callable(data_init)):
                data_init = data_init()

            if not isinstance(data_init, initializer.DataDependentInitializer):
                raise TypeError(f'Unsupported data dependent initializer: '
                                f'{data_init!r}')

        super().__init__()
        self.weight_store = weight_store
        self.bias_store = bias_store

        if data_init is not None:
            data_init.register(self)

    def __repr__(self) -> str:
        attributes = []
        for attr in self.__annotations__:
            val = getattr(self, attr, None)
            if val is not None:
                if attr == 'use_bias':
                    if not val:
                        attributes.append(f'{attr}={val}')
                elif not isinstance(val, (Module, Tensor)):
                    attributes.append(f'{attr}={val!r}')
        return f'{self.__class__.__qualname__}({", ".join(attributes)})'

    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_store()
        if self.bias_store is None:
            bias = None
        else:
            bias = self.bias_store()
        return self._forward(input, weight, bias)


class Linear(CoreLinear):

    in_features: int
    out_features: int
    use_bias: bool

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        in_features = validate_positive_int('in_features', in_features)
        out_features = validate_positive_int('out_features', out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[out_features, in_features],
            bias_shape=[out_features],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device,
        )

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        output, front_shape = flatten_to_ndims(input, 2)
        output = torch.nn.functional.linear(output, weight, bias)
        output = unflatten_from_ndims(output, front_shape)
        return output


class LinearConvNd(CoreLinear):

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[Tuple[int, int]]
    _symmetric_padding: Optional[List[int]]
    dilation: List[int]
    use_bias: bool

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        spatial_ndims = self._get_spatial_ndims()
        in_channels = validate_positive_int('in_channels', in_channels)
        out_channels = validate_positive_int('out_channels', out_channels)
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)
        _symmetric_padding = maybe_as_symmetric_padding(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self._symmetric_padding = _symmetric_padding
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[out_channels, in_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class LinearConv1d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv1d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding, dilation=self.dilation
            )
        else:
            output = pad(input, self.padding, value=0.)
            return torch.nn.functional.conv1d(
                input=output, weight=weight, bias=bias, stride=self.stride,
                padding=0, dilation=self.dilation,
            )


class LinearConv2d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv2d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding, dilation=self.dilation
            )
        else:
            output = pad(input, self.padding, value=0.)
            return torch.nn.functional.conv2d(
                input=output, weight=weight, bias=bias, stride=self.stride,
                padding=0, dilation=self.dilation,
            )


class LinearConv3d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv3d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding, dilation=self.dilation
            )
        else:
            output = pad(input, self.padding, value=0.)
            return torch.nn.functional.conv3d(
                input=output, weight=weight, bias=bias, stride=self.stride,
                padding=0, dilation=self.dilation,
            )


class LinearConvTransposeNd(CoreLinear):

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[Tuple[int, int]]
    is_symmetric_padding: bool
    dilation: List[int]
    output_padding: List[int]
    use_bias: bool

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 output_padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        spatial_ndims = self._get_spatial_ndims()
        in_channels = validate_positive_int('in_channels', in_channels)
        out_channels = validate_positive_int('out_channels', out_channels)
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)
        _symmetric_padding = maybe_as_symmetric_padding(padding)
        output_padding = validate_output_padding(
            output_padding, stride, dilation, spatial_ndims)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._symmetric_padding = _symmetric_padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[in_channels, out_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


# @jit
def unpad_output(input: Tensor,
                 axis: List[int],
                 padding: List[Tuple[int, int]]) -> Tensor:
    output = input
    for a in axis:
        p1, p2 = padding[a]
        length = output.shape[a] - p1 - p2
        if length < 0:
            raise ValueError('Too large `padding` at axis {}: {}'.
                             format(axis, padding))
        output = torch.narrow(output, a, p1, length)
    return output


class LinearConvTranspose1d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv_transpose1d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding,
                output_padding=self.output_padding, dilation=self.dilation
            )
        else:
            output = torch.nn.functional.conv_transpose1d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=0, output_padding=self.output_padding,
                dilation=self.dilation
            )
            return unpad_output(output, [-1], self.padding)


class LinearConvTranspose2d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv_transpose2d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding,
                output_padding=self.output_padding, dilation=self.dilation
            )
        else:
            output = torch.nn.functional.conv_transpose2d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=0, output_padding=self.output_padding,
                dilation=self.dilation
            )
            return unpad_output(output, [-1, -2], self.padding)


class LinearConvTranspose3d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    # @jit_method
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
                 ) -> Tensor:
        if self._symmetric_padding is not None:
            return torch.nn.functional.conv_transpose3d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=self._symmetric_padding,
                output_padding=self.output_padding, dilation=self.dilation
            )
        else:
            output = torch.nn.functional.conv_transpose3d(
                input=input, weight=weight, bias=bias, stride=self.stride,
                padding=0, output_padding=self.output_padding,
                dilation=self.dilation
            )
            return unpad_output(output, [-1, -2, -3], self.padding)


# ---- normalizer layers ----
class BatchNorm(torch.nn.BatchNorm1d):
    """Batch normalization for dense layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 device: Optional=None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 2:
            raise ValueError('`BatchNorm` only supports 2d input, '
                             'but the input shape is {}'.format(list(input.shape)))


class BatchNorm1d(torch.nn.BatchNorm1d):
    """Batch normalization for 1D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = EPSILON,
                 device: Optional[str]=None):
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        device = device or current_device()
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 3:
            raise ValueError('`BatchNorm1d` only supports 3d input, '
                             'but the input shape is {}'.format(list(input.shape)))


class BatchNorm2d(torch.nn.BatchNorm2d):
    """Batch normalization for 2D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = EPSILON,
                 device: Optional[str]=None):
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        device = device or current_device()
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 4:
            raise ValueError('`BatchNorm2d` only supports 4d input, '
                             'but the input shape is {}'.format(list(input.shape)))


class BatchNorm3d(torch.nn.BatchNorm3d):
    """Batch normalization for 3D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = EPSILON,
                 device: Optional[str]=None):
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        device = device or current_device()
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 5:
            raise ValueError('`BatchNorm3d` only supports 5d input, '
                             'but the input shape is {}'.format(list(input.shape)))


# ---- dropout layers ----
Dropout = torch.nn.Dropout


class Dropout1d(BaseSingleVariateLayer):
    """Randomly zero out entire channels of the 1d convolution input."""

    __constants__ = ('p', '_keep_prob')

    p: float
    _keep_prob: float

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._keep_prob = 1. - p

    def _forward(self, input: Tensor) -> Tensor:
        if input.dim() < 2:  # pragma: no cover
            raise ValueError('`input` must be at least 2d, but the '
                             'input shape is {}.'.format(list(input.shape)))

        device = input.device
        output = input
        if self.training:
            noise_shape = output.shape[:-1] + (1,)
            noise = torch.zeros(noise_shape, dtype=output.dtype, device=device)
            keep_prob = torch.as_tensor(self._keep_prob, dtype=output.dtype, device=device)
            noise = torch.bernoulli(keep_prob.expand(noise_shape), out=noise)
            noise = noise.detach()
            output = output * noise / keep_prob
        return output


Dropout2d = torch.nn.Dropout2d
Dropout3d = torch.nn.Dropout3d
