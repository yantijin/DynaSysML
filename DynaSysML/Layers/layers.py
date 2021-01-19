# import torch
from typing import *

from DynaSysML.arg_check import *
from DynaSysML.typing_ import *
from .gated import *
from .base import Sequential, Linear, DEFAULT_WEIGHT_INIT, DEFAULT_BIAS_INIT, DEFAULT_GATE_BIAS, \
    LinearConv1d, LinearConv2d, LinearConv3d, LinearConvTranspose1d, LinearConvTranspose2d, LinearConvTranspose3d, \
    IS_CHANNEL_LAST

__all__ = [
    'Dense',
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
]



def _make_layers(linear: Module,
                 out_features: int,
                 out_feature_axis: int,
                 normalizer: Optional[NormalizerOrNormalizerFactory],
                 activation: Optional[LayerOrLayerFactory],
                 gated: bool,
                 gate_bias: float
                 ) -> List[Module]:
    layers = [linear]
    pre_gate_out_features = out_features * (2 if gated else 1)

    # add normalizer
    if normalizer is not None:
        layers.append(
            get_layer_from_layer_or_factory(
                'normalizer', normalizer, args=(pre_gate_out_features,)))

    # add activation and gate
    if activation is not None:
        activation = get_layer_from_layer_or_factory('activation', activation)
        if gated:
            gate_with_activation = GatedWithActivation(
                out_feature_axis, out_features, gate_bias, activation)
            layers.append(gate_with_activation)
        else:
            layers.append(activation)
    elif gated:
        layers.append(Gated(out_feature_axis, out_features, gate_bias))

    return layers


# ---- dense layer ----
class Dense(Sequential):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: Optional[bool] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 gated: bool = False,
                 gate_bias: float = DEFAULT_GATE_BIAS,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        # check the arguments
        if use_bias is None:
            use_bias = normalizer is None

        # construct the layer
        super().__init__(*_make_layers(
            linear=Linear(
                in_features=in_features,
                out_features=out_features * (2 if gated else 1),
                use_bias=use_bias,
                weight_norm=weight_norm,
                weight_init=weight_init,
                bias_init=bias_init,
                data_init=data_init,
                device=device
            ),
            out_features=out_features,
            out_feature_axis=-1,
            normalizer=normalizer,
            activation=activation,
            gated=gated,
            gate_bias=gate_bias,
        ))


# ---- convolution layers ----
class ConvNd(Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: Optional[bool] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 gated: bool = False,
                 gate_bias: float = DEFAULT_GATE_BIAS,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        spatial_ndims = self._get_spatial_ndims()
        linear_factory = self._get_linear_factory()

        # check the arguments
        if use_bias is None:
            use_bias = normalizer is None

        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)

        # construct the layer
        super().__init__(*_make_layers(
            linear=linear_factory(
                in_channels=in_channels,
                out_channels=out_channels * (2 if gated else 1),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                use_bias=use_bias,
                weight_norm=weight_norm,
                weight_init=weight_init,
                bias_init=bias_init,
                data_init=data_init,
                device=device
            ),
            out_features=out_channels,
            out_feature_axis=-1 if IS_CHANNEL_LAST else -(spatial_ndims + 1),
            normalizer=normalizer,
            activation=activation,
            gated=gated,
            gate_bias=gate_bias,
        ))

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def _get_linear_factory(self) -> LayerFactory:
        raise NotImplementedError()


class Conv1d(ConvNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConv1d


class Conv2d(ConvNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConv2d


class Conv3d(ConvNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConv3d


# ---- deconvolution layers ----
class ConvTransposeNd(Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 output_padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: Optional[bool] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 gated: bool = False,
                 gate_bias: float = DEFAULT_GATE_BIAS,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str]=None
                 ):
        spatial_ndims = self._get_spatial_ndims()
        linear_factory = self._get_linear_factory()

        # check the arguments
        if use_bias is None:
            use_bias = normalizer is None

        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)
        output_padding = validate_output_padding(
            output_padding, stride, dilation, spatial_ndims)

        # construct the layer
        super().__init__(*_make_layers(
            linear=linear_factory(
                in_channels=in_channels,
                out_channels=out_channels * (2 if gated else 1),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                use_bias=use_bias,
                weight_norm=weight_norm,
                weight_init=weight_init,
                bias_init=bias_init,
                data_init=data_init,
                device=device
            ),
            out_features=out_channels,
            out_feature_axis=-1 if IS_CHANNEL_LAST else -(spatial_ndims + 1),
            normalizer=normalizer,
            activation=activation,
            gated=gated,
            gate_bias=gate_bias,
        ))

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def _get_linear_factory(self) -> LayerFactory:
        raise NotImplementedError()


class ConvTranspose1d(ConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConvTranspose1d


class ConvTranspose2d(ConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConvTranspose2d


class ConvTranspose3d(ConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    def _get_linear_factory(self) -> LayerFactory:
        return LinearConvTranspose3d
