from typing import *
from DynaSysML.arg_check import *
from DynaSysML.typing_ import *
from .gated import *
from .base import *
from .contextual import *

__all__ = [
    'ResBlock1d', 'ResBlock2d', 'ResBlock3d',
    'ResBlockTranspose1d', 'ResBlockTranspose2d', 'ResBlockTranspose3d',
]


class ResBlockNd(BaseLayer):
    """
    A general implementation of ResNet block.

    The architecture of this ResNet implementation follows the work
    "Wide residual networks" (Zagoruyko & Komodakis, 2016).  It basically does
    the following things:

    .. code-block:: python

        shortcut = input
        if strides != 1 or (kernel_size != 1 and padding != 'half') or \
                in_channels != out_channels or use_shortcut:
            shortcut_layer = shortcut(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
            )
            shortcut = shortcut_layer(shortcut)

        residual = input
        if resize_at_exit:
            conv0_layer = conv0(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='half',
                output_padding=0,  # for deconvolutional layers only
                dilation=dilation,
            )
            conv1_layer = conv1(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,  # for deconvolutional layers only
                dilation=dilation,
            )
        else:
            conv0_layer = conv0(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,  # for deconvolutional layers only
                dilation=dilation,
            )
            conv1_layer = conv1(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='half'',
                output_padding=0,  # for deconvolutional layers only
                dilation=dilation,
            )
        residual = normalizer0(residual)
        residual = activation0(residual)
        residual = conv0_layer(residual)
        if merge_context0 is not None:
            residual = merge_context0(residual, context)
        residual = dropout(residual)
        residual = normalizer1(residual)
        residual = activation1(residual)
        residual = conv1_layer(residual)
        if merge_context1 is not None:
            residual = merge_context1(residual, context)

        output = shortcut + residual
    """

    __constants__ = (
        'shortcut',
        'pre_conv0', 'merge_context0', 'conv0',
        'pre_conv1', 'merge_context1', 'conv1',
    )

    shortcut: Module
    pre_conv0: Module
    merge_context0: Module
    conv0: Module
    pre_conv1: Module
    merge_context1: Module
    conv1: Module
    post_conv1: Module

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 output_padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 resize_at_exit: bool = False,
                 use_shortcut: Optional[bool] = None,
                 shortcut: Optional[LayerOrLayerFactory] = None,
                 conv0: Optional[LayerOrLayerFactory] = None,
                 conv1: Optional[LayerOrLayerFactory] = None,
                 merge_context0: Optional[Module] = None,
                 merge_context1: Optional[Module] = None,
                 activation: Optional[LayerFactory] = None,
                 normalizer: Optional[NormalizerFactory] = None,
                 dropout: Optional[Union[float, LayerOrLayerFactory]] = None,
                 weight_norm: WeightNormArgType = False,
                 gated: bool = False,
                 gate_bias: float = DEFAULT_GATE_BIAS,
                 use_bias: Optional[bool] = None,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        """
        Construct a new resnet block.

        Args:
            in_channels: The number of channels of the input.
            out_channels: The number of channels of the output.
            kernel_size: The kernel size over spatial dimensions.
            stride: The stride over spatial dimensions.
            padding: The padding over spatial dimensions.
            output_padding: The output padding for de-convolutional resnet
                blocks.  Must not be specified for convolutional resnet blocks.
            dilation: The dilation over spatial dimensions.
            resize_at_exit: If :obj:`True`, resize the spatial dimensions at
                the "conv1" convolutional layer.
                If :obj:`False`, resize at the "conv0" convolutional layer.
                (see above)
            use_shortcut: If :obj:`True`, always applies a linear
                convolution transformation on the shortcut path.
                Defaults to :obj:`None`, only use shortcut if necessary.
            shortcut: The "shortcut" layer, or the factory to construct the layer.
            conv0: The "conv0" layer, or the factory to construct the layer.
            conv1: The "conv1" layer, or the factory to construct the layer.
            merge_context0: Layer after "conv0" to merge the `context`
                argument with the output of "conv0".  (See above)
            merge_context1: Layer after "conv1" to merge the `context`
                argument with the output of "conv1".  (See above)
            activation: The factory of the activation layers.
                It should expect no argument.
            normalizer: The factory of the normalizer layers.  It should accept
                one positional argument, the output channel size.
            dropout: A float, a layer or a factory.
                If it is a float, it will be used as the `p` argument to
                construct an instance of :class:`tensorkit.layers.Dropout`.
                If it is a factory, it should expect no argument.
            weight_norm: The weight norm mode for the convolutional layers.
                If :obj:`True`, will use "full" weight norm for "conv1" and
                "shortcut".  For "conv0", will use "full" if `normalizer`
                is :obj:`None` or `dropout` is not :obj:`None`.

                If :obj:`False`, will not use weight norm for all layers.
            gated: Whether or not to use gate on the output of "conv1"?
                `conv1 = activation(conv1) * sigmoid(gate + gate_bias)`.
            gate_bias: The bias added to `gate` before applying the `sigmoid`
                activation.
            use_bias: Whether or not to use bias in "conv0", "conv1"
                and "shortcut"?  If :obj:`True`, will always use bias.
                If :obj:`False`, will never use bias.

                Defaults to :obj:`None`, where "use_bias" of "shortcut",
                "conv0" and "conv1" is set according to the following rules:

                * "shortcut": `use_bias` is :obj:`True` if `gated` is True.
                * "conv0": `use_bias` is :obj:`True` if `normalizer` is None,
                  or `dropout` is not None.
                * "conv1": `use_bias` is always :obj:`True`.
            weight_init: The weight initializer for the convolutional layers.
            bias_init: The bias initializer for the convolutional layers.
            data_init: The data-dependent initializer for the convolutional layers.
            device: The device where to place new tensors and variables.
        """
        def use_bias_or_else(default_val: bool):
            if use_bias is None:
                return default_val
            return use_bias

        def compile_layer_list(layers: List[Module]) -> Module:
            if len(layers) == 0:
                return Identity()
            elif len(layers) == 1:
                return layers[0]
            else:
                return Sequential(layers)

        spatial_ndims = self._get_spatial_ndims()
        is_deconv = self._is_deconv()

        # validate arguments
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('strides', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)

        if output_padding != 0 and not is_deconv:
            raise ValueError(f'The `output_padding` argument is not allowed '
                             f'by {self.__class__.__qualname__}.')
        output_padding = validate_output_padding(
            output_padding, stride, dilation, spatial_ndims)

        if conv0 is None:
            conv0 = self._default_conv_factory()

        if conv1 is None:
            conv1 = self._default_conv_factory()

        orig_merge_context0 = merge_context0
        if merge_context0 is None:
            merge_context0 = IgnoreContext()
        else:
            merge_context0 = validate_layer('merge_context0', merge_context0)

        if merge_context1 is None:
            merge_context1 = IgnoreContext()
        else:
            merge_context1 = validate_layer('merge_context1', merge_context1)

        if shortcut is not None:
            use_shortcut = True
        if use_shortcut is None:
            use_shortcut = (
                any(s != 1 for s in stride) or
                any(p[0] + p[1] != (k - 1) * d
                    for p, k, d in zip(padding, kernel_size, dilation)) or
                in_channels != out_channels)

        if activation is not None:
            activation_factory = validate_layer_factory('activation', activation)
        else:
            activation_factory = None

        if normalizer is not None:
            normalizer_factory = validate_layer_factory('normalizer', normalizer)
        else:
            normalizer_factory = None

        if isinstance(dropout, float):
            dropout = Dropout(p=dropout)
        elif dropout is not None:
            dropout = get_layer_from_layer_or_factory('dropout', dropout)

        conv0_weight_norm = weight_norm
        if conv0_weight_norm is True:
            conv0_weight_norm = (
                WeightNormMode.FULL if normalizer is None or dropout is not None
                else WeightNormMode.NO_SCALE
            )

        kwargs = {'weight_init': weight_init, 'bias_init': bias_init,
                  'data_init': data_init, 'device': device}

        # build the shortcut path
        if use_shortcut:
            if shortcut is None:
                shortcut = self._default_conv_factory()
            if not isinstance(shortcut, Module):
                shortcut = get_layer_from_layer_or_factory(
                    'shortcut', shortcut, kwargs=dict(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        use_bias=use_bias_or_else(gated),
                        weight_norm=weight_norm,
                        **self._add_output_padding_to_kwargs(output_padding, kwargs)
                    )
                )
        else:
            shortcut = Identity()

        # prepare the arguments for the residual path
        if resize_at_exit:
            conv0_out_channels = in_channels
            conv0_stride = 1
            conv0_padding = PaddingMode.HALF  # such that it can keep the output shape
            conv0_kwargs = kwargs
            conv1_stride = stride
            conv1_padding = padding
            conv1_kwargs = self._add_output_padding_to_kwargs(output_padding, kwargs)
        else:
            conv0_out_channels = out_channels
            conv0_stride = stride
            conv0_padding = padding
            conv0_kwargs = self._add_output_padding_to_kwargs(output_padding, kwargs)
            conv1_stride = 1
            conv1_padding = PaddingMode.HALF  # such that it can keep the output shape
            conv1_kwargs = kwargs

        conv1_out_channels = out_channels
        if gated:
            conv1_out_channels *= 2

        # pre_conv0
        pre_conv0 = []
        if normalizer_factory is not None:
            pre_conv0.append(normalizer_factory(in_channels))
        if activation_factory is not None:
            pre_conv0.append(activation_factory())
        pre_conv0 = compile_layer_list(pre_conv0)

        # conv0
        conv0 = get_layer_from_layer_or_factory(  # conv0
            'conv0', conv0, kwargs=dict(
                in_channels=in_channels,
                out_channels=conv0_out_channels,
                kernel_size=kernel_size,
                stride=conv0_stride,
                padding=conv0_padding,
                dilation=dilation,
                use_bias=use_bias_or_else(normalizer_factory is None or
                                          dropout is not None or
                                          orig_merge_context0 is not None),
                weight_norm=conv0_weight_norm,
                **conv0_kwargs,
            )
        )

        # pre_conv1
        pre_conv1 = []
        if dropout is not None:
            pre_conv1.append(dropout)
        if normalizer_factory is not None:
            pre_conv1.append(normalizer_factory(conv0_out_channels))
        if activation_factory is not None:
            pre_conv1.append(activation_factory())
        pre_conv1 = compile_layer_list(pre_conv1)

        # conv1
        conv1 = get_layer_from_layer_or_factory(
            'conv1', conv1, kwargs=dict(
                in_channels=conv0_out_channels,
                out_channels=conv1_out_channels,
                kernel_size=kernel_size,
                stride=conv1_stride,
                padding=conv1_padding,
                dilation=dilation,
                use_bias=use_bias_or_else(True),
                weight_norm=weight_norm,
                **conv1_kwargs,
            )
        )

        # post_conv1
        if gated:
            post_conv1 = Gated(
                feature_axis=-(spatial_ndims + 1),
                num_features=out_channels,
                gate_bias=gate_bias,
            )
        else:
            post_conv1 = Identity()

        # construct the layer
        super().__init__()
        self.shortcut = shortcut
        self.pre_conv0 = pre_conv0
        self.merge_context0 = merge_context0
        self.conv0 = conv0
        self.pre_conv1 = pre_conv1
        self.merge_context1 = merge_context1
        self.conv1 = conv1
        self.post_conv1 = post_conv1

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def _default_conv_factory(self) -> LayerFactory:
        raise NotImplementedError()

    def _is_deconv(self) -> bool:
        raise NotImplementedError()

    def _add_output_padding_to_kwargs(self, output_padding, kwargs):
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                context: Optional[List[Tensor]] = None) -> Tensor:
        if context is None:
            context = []

        # compute the residual path
        residual = self.pre_conv0(input)
        residual = self.conv0(residual)
        residual = self.merge_context0(residual, context)
        residual = self.pre_conv1(residual)
        residual = self.conv1(residual)
        residual = self.merge_context1(residual, context)
        residual = self.post_conv1(residual)

        # sum up the shortcut path and the residual path as the final output
        return self.shortcut(input) + residual


class ResBlockConvNd(ResBlockNd):

    def _add_output_padding_to_kwargs(self, output_padding, kwargs):
        return kwargs

    def _is_deconv(self) -> bool:
        return False


class ResBlock1d(ResBlockConvNd):
    """1D ResNet convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 1

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConv1d


class ResBlock2d(ResBlockConvNd):
    """2D ResNet convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 2

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConv2d


class ResBlock3d(ResBlockConvNd):
    """3D ResNet convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 3

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConv3d


class ResBlockTransposeNd(ResBlockNd):

    def _add_output_padding_to_kwargs(self, output_padding, kwargs=None):
        kwargs = dict(kwargs or {})
        kwargs['output_padding'] = output_padding
        return kwargs

    def _is_deconv(self) -> bool:
        return True


class ResBlockTranspose1d(ResBlockTransposeNd):
    """1D ResNet de-convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 1

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConvTranspose1d


class ResBlockTranspose2d(ResBlockTransposeNd):
    """2D ResNet de-convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 2

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConvTranspose2d


class ResBlockTranspose3d(ResBlockTransposeNd):
    """3D ResNet de-convolution block."""

    def _get_spatial_ndims(self) -> int:
        return 3

    def _default_conv_factory(self) -> LayerFactory:
        return LinearConvTranspose3d
