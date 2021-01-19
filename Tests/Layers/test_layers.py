import pytest
import DynaSysML as dsl
import DynaSysML.Layers.initializer as init
import torch
from Tests.ops import *


def check_composed_layer(ctx, input, layer_cls, linear_cls, normalizer_cls,
                         in_features, out_features, **kwargs):
    # test pure
    for use_bias in [None, True, False]:
        layer = layer_cls(
            in_features, out_features, use_bias=use_bias,
            bias_init=init.uniform, **kwargs
        )
        expected_use_bias = True if use_bias is None else use_bias
        linear = linear_cls(
            in_features, out_features,
            weight_init=layer[0].weight_store.get(),
            bias_init=(layer[0].bias_store.get() if expected_use_bias
                       else None),
            use_bias=expected_use_bias,
            **kwargs
        )
        assert (layer_cls.__qualname__ in repr(layer))
        assert isinstance(layer[0], linear_cls)
        assert (layer[0].use_bias == expected_use_bias)
        assert(
            layer(input)-linear(input) < 1e-6
        ).all()

    # test normalizer
    for use_bias in [None, True, False]:
        for normalizer_arg in [normalizer_cls, normalizer_cls(out_features)]:
            layer = layer_cls(
                in_features, out_features, normalizer=normalizer_arg,
                use_bias=use_bias, bias_init=init.uniform, **kwargs
            )
            expected_use_bias = False if use_bias is None else use_bias
            linear = linear_cls(
                in_features, out_features,
                weight_init=layer[0].weight_store.get(),
                bias_init=(layer[0].bias_store.get() if expected_use_bias
                           else None),
                use_bias=expected_use_bias,
                **kwargs
            )
            normalizer = normalizer_cls(out_features)
            assert isinstance(layer[0], linear_cls)
            assert (layer[0].use_bias == expected_use_bias)
            assert isinstance(layer[1], normalizer_cls)
            assert(
                layer(input)- normalizer(linear(input))< 1e-6
            ).all()

    # test activation
    activation_cls = dsl.Layers.Tanh
    for activation_arg in [activation_cls, activation_cls()]:
        layer = layer_cls(
            in_features, out_features, activation=activation_arg,
            bias_init=init.uniform, **kwargs
        )
        linear = linear_cls(
            in_features, out_features,
            weight_init=layer[0].weight_store.get(),
            bias_init=layer[0].bias_store.get(),
            **kwargs
        )
        assert isinstance(layer[0], linear_cls)
        assert isinstance(layer[1], dsl.Layers.Tanh)
        assert(
            layer(input) - activation_cls()(linear(input)) < 1e-6
        ).all()

    # test gate
    layer = layer_cls(
        in_features, out_features,
        bias_init=init.uniform, gated=True, **kwargs
    )
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        bias_init=layer[0].bias_store.get(),
        **kwargs
    )
    assert isinstance(layer[0], linear_cls)
    out = linear(input)
    assert(
        layer(input)-torch.sigmoid(out[:, out_features:] + 2.0) * out[:, :out_features] < 1e-6
    ).all()

    # test gate + activation
    activation = dsl.Layers.LeakyReLU()
    layer = layer_cls(
        in_features, out_features, activation=activation,
        bias_init=init.uniform, gated=True, **kwargs
    )
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        bias_init=layer[0].bias_store.get(),
        **kwargs
    )
    assert isinstance(layer[0], linear_cls)
    out = linear(input)
    assert( layer(input) - (torch.sigmoid(out[:, out_features:] + 2.0) *
         activation(out[:, :out_features])) <1e-6
    ).all()

    # test normalizer + gate + activation
    normalizer = normalizer_cls(out_features * 2)
    activation = dsl.Layers.LeakyReLU()
    layer = layer_cls(
        in_features, out_features, activation=activation, normalizer=normalizer,
        gated=True, **kwargs
    )
    assert(layer[0].use_bias==False)
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        use_bias=False,
        **kwargs
    )
    assert isinstance(layer[0], linear_cls)
    out = normalizer(linear(input))
    assert(layer(input)-(torch.sigmoid(out[:, out_features:] + 2.0) *
         activation(out[:, :out_features])) <1e-6
    ).all()


class TestLayers(object):

    def test_dense(self):
        check_composed_layer(
            self,
            torch.randn([5, 4]),
            dsl.Layers.Dense,
            dsl.Layers.Linear,
            dsl.Layers.BatchNorm,
            4, 3
        )

    def test_conv_nd(self):
        for spatial_ndims in (1, 2, 3):
            check_composed_layer(
                self,
                torch.randn(make_conv_shape(
                    [5], 4, [16, 15, 14][:spatial_ndims]
                )),
                getattr(dsl.Layers, f'Conv{spatial_ndims}d'),
                getattr(dsl.Layers, f'LinearConv{spatial_ndims}d'),
                getattr(dsl.Layers, f'BatchNorm{spatial_ndims}d'),
                4, 3,
                kernel_size=3, stride=2, dilation=2, padding='half'
            )

    def test_conv_transpose_nd(self):
        for spatial_ndims in (1, 2, 3):
            for output_padding in (0, 1):
                check_composed_layer(
                    self,
                    torch.randn(make_conv_shape(
                        [5], 4, [16, 15, 14][:spatial_ndims]
                    )),
                    getattr(dsl.Layers, f'ConvTranspose{spatial_ndims}d'),
                    getattr(dsl.Layers, f'LinearConvTranspose{spatial_ndims}d'),
                    getattr(dsl.Layers, f'BatchNorm{spatial_ndims}d'),
                    4, 3,
                    kernel_size=3, stride=2, dilation=2, padding='half',
                    output_padding=output_padding,
                )
