from itertools import product
from typing import *
import pytest
import torch

import numpy as np
import pytest
from ..ops import *
import DynaSysML as dsl
from DynaSysML.Layers import BaseLayer
from DynaSysML.core import Tensor



def make_causal_test_input(size: List[int],
                           pos: List[int],
                           single_point: bool = True,
                           ) -> np.ndarray:
    ret = np.zeros(size, dtype=np.float32)
    if single_point:
        tmp = ret
        for p in pos[:-1]:
            tmp = tmp[p]
        tmp[pos[-1]] = 1.
    else:
        tmp = ret
        for p in pos[:-1]:
            tmp[(p+1):] = 1.
            tmp = tmp[p]
        tmp[pos[-1]:] = 1.
    return np.reshape(ret, make_conv_shape([1], 1, size))


def make_causal_mask(size: List[int], pos: List[int]) -> np.ndarray:
    ret = make_causal_test_input(size, pos, single_point=False)
    r_shape = ret.shape
    ret = ret.reshape(size)
    tmp = ret
    for p in pos[:-1]:
        tmp = tmp[p]
    tmp[pos[-1]] = 0.
    return ret.reshape(r_shape)


def iter_causal_test_pos(size: List[int]):
    return list(product(*([0, s // 2, s - 1] for s in size)))


def ensure_stacks_causality(ctx,
                            outputs,
                            size: List[int],
                            pos: List[int]):
    assert(len(outputs) == len(size))
    spatial_ndims = len(outputs)
    for i in range(spatial_ndims):
        output = outputs[i]
        if isinstance(output, dsl.Tensor):
            output = dsl.to_numpy(output)
        output = output.reshape(size)
        this_pos = list(pos)
        this_pos[i] += 1
        k = i
        while k > 0 and this_pos[k] >= size[k]:
            this_pos[k - 1] += 1
            this_pos[k] = 0
            k -= 1
        for j in range(i + 1, spatial_ndims):
            this_pos[j] = 0
        if this_pos[0] >= size[0]:
            mask = np.zeros(size, dtype=np.float32)
        else:
            mask = make_causal_test_input(size, this_pos, single_point=False)
        is_wrong = np.any(
            np.logical_and(
                np.abs(output) > 1e-6,
                np.logical_not(mask.astype(np.bool))
            )
        )
        assert(
            bool(is_wrong) == False        )


def ensure_full_receptive_field(ctx,
                                output,
                                size: List[int],
                                pos: List[int]):
    if isinstance(output, dsl.Tensor):
        output = dsl.to_numpy(output)
    output_true = (np.abs(output.reshape(size)) >= 1e-6).astype(np.int32)
    mask = make_causal_mask(size, pos).astype(np.int32)
    assert(
        bool(np.all(
            np.logical_not(
                np.logical_xor(
                    mask.astype(np.bool),
                    output_true.astype(np.bool)
                )
            )
        )) == True
    )


class _MyAddContext(BaseLayer):

    def forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        if len(context) == 0:
            return input
        elif len(context) == 1:
            return input + context[0]
        else:
            raise ValueError('Expected context to have 0 or 1 element.')


class TestPixelCNN(object):

    def test_causality_and_receptive_field(self):
        for size in [[12], [12, 11], [12, 11, 10]]:
            spatial_ndims = len(size)

            for kernel_size in [3, 5, [5, 3, 5][:spatial_ndims]]:
                # ---- construct the layers ----
                # the input layer
                input_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNInput{spatial_ndims}d')
                input_layer = input_layer_cls(
                    1, 1, kernel_size=kernel_size, add_ones_channel=False,
                    weight_init=dsl.Layers.ones,
                )
                # input_layer = tk.layers.jit_compile(input_layer)

                with pytest.raises(Exception,
                                   match='`input` is expected to be .*d'):
                    _ = input_layer(dsl.zeros([1] * (spatial_ndims + 1)))
                with pytest.raises(Exception,
                                   match='`input` is expected to be .*d'):
                    _ = input_layer(dsl.zeros([1] * (spatial_ndims + 3)))

                # `add_ones_channnel = True`
                input_layer2 = input_layer_cls(
                    1, 1, kernel_size=kernel_size, weight_init=dsl.Layers.ones)

                # the pixelcnn resblock
                resblock_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNResBlock{spatial_ndims}d')

                with pytest.raises(ValueError,
                                   match=r'`kernel_size` is required to be at '
                                         r'least 3'):
                    _ = resblock_layer_cls(1, 1, kernel_size=1)
                with pytest.raises(ValueError,
                                   match=r'`kernel_size` is required to be odd'):
                    _ = resblock_layer_cls(1, 1, kernel_size=[4, 3, 5][:spatial_ndims])

                resblock_layer = resblock_layer_cls(
                    1, 1, kernel_size=kernel_size, weight_init=dsl.Layers.ones
                )
                # resblock_layer = dsl.layers.jit_compile(resblock_layer)

                with pytest.raises(Exception):
                    _ = resblock_layer([dsl.zeros([])] * (spatial_ndims - 1))
                with pytest.raises(Exception):
                    _ = resblock_layer([dsl.zeros([])] * (spatial_ndims + 1))

                # the down-sampling and up-sampling layer
                down_sample_cls = getattr(dsl.Layers, f'PixelCNNConv{spatial_ndims}d')
                down_sample_layer = down_sample_cls(1, 1, kernel_size, stride=2)
                # down_sample_layer = dsl.Layers.jit_compile(down_sample_layer)

                down_sample_output_size = list(down_sample_layer(
                    [dsl.zeros(make_conv_shape([1], 1, size))] * spatial_ndims)[0].shape)
                up_sample_cls = getattr(dsl.Layers, f'PixelCNNConvTranspose{spatial_ndims}d')
                up_sample_layer = up_sample_cls(
                    1, 1, kernel_size, stride=2,
                    output_padding=dsl.Layers.get_deconv_output_padding(
                        input_size=[down_sample_output_size[a]
                                    for a in get_spatial_axis(spatial_ndims)],
                        output_size=size,
                        kernel_size=kernel_size,
                        stride=2,
                        padding='half',  # sum of the both sides == (kernel_size - 1) * dilation
                    )
                )
                # up_sample_layer = tk.layers.jit_compile(up_sample_layer)

                # the output layer
                output_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNOutput{spatial_ndims}d')
                output_layer = output_layer_cls()
                # output_layer = dsl.layers.jit_compile(output_layer)

                with pytest.raises(Exception,
                                   match=r'`len\(inputs\)` is expected to be .*'):
                    _ = output_layer([dsl.zeros([])] * (spatial_ndims - 1))
                with pytest.raises(Exception,
                                   match=r'`len\(inputs\)` is expected to be .*'):
                    _ = output_layer([dsl.zeros([])] * (spatial_ndims + 1))

                # ---- test the causality ----
                for pos, single_point in product(
                            iter_causal_test_pos(size),
                            (True, False)
                        ):
                    x = make_causal_test_input(
                        size, pos, single_point=single_point)
                    x_t = dsl.as_tensor(x)

                    # check the input layer output
                    outputs = input_layer(x_t)
                    ensure_stacks_causality(self, outputs, size, pos)

                    # check the final output
                    assert (output_layer(outputs) - outputs[-1] < 1e-5).all()

                    # check the resblock output
                    resblock_outputs = resblock_layer(outputs)
                    ensure_stacks_causality(self, resblock_outputs, size, pos)

                    outputs2 = resblock_outputs
                    for i in range(4):
                        outputs2 = resblock_layer(outputs2)
                    ensure_full_receptive_field(self, outputs2[-1], size, pos)

                    # check the down-sample and up-sample
                    down_sample_outputs = down_sample_layer(outputs)
                    up_sample_outputs = up_sample_layer(down_sample_outputs)
                    ensure_stacks_causality(self, up_sample_outputs, size, pos)

                # ---- test zero input on different input layers ----
                x_t = dsl.zeros(make_conv_shape([1], 1, size), dtype=torch.float32)
                outputs = input_layer(x_t)
                assert(
                    (np.abs(dsl.to_numpy(outputs[-1])) >= 1e-6).astype(np.int32) ==
                    x_t.cpu().numpy()
                ).all()
                outputs = input_layer2(x_t)
                assert(
                    (np.abs(dsl.to_numpy(outputs[-1])) >= 1e-6).astype(np.int32) ==
                    make_causal_mask(size, [0] * spatial_ndims).astype(np.int32)
                ).all()

    def test_pixelcnn_network(self):
        in_channels = 3
        out_channels = 5

        for size in [[15], [15, 13], [15, 13, 11]]:
            spatial_ndims = len(size)

            for kernel_size in [3, 5, [5, 3, 5][:spatial_ndims]]:
                # ---- construct the layers ----
                # the input layer
                input_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNInput{spatial_ndims}d')
                input_layer = input_layer_cls(
                    in_channels, out_channels, kernel_size=kernel_size)
                # input_layer = dsl.Layers.jit_compile(input_layer)

                # the pixelcnn layers
                resblock_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNResBlock{spatial_ndims}d')
                conv_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNConv{spatial_ndims}d')
                deconv_layer_cls = getattr(
                    dsl.Layers, f'PixelCNNConvTranspose{spatial_ndims}d')
                normalizer_cls = getattr(
                    dsl.Layers, f'BatchNorm{spatial_ndims}d')
                dropout_cls = getattr(
                    dsl.Layers, f'Dropout{spatial_ndims}d')

                pixelcnn_layers = [
                    resblock_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        activation=dsl.Layers.LeakyReLU, normalizer=normalizer_cls,
                        merge_context1=_MyAddContext,
                        data_init=dsl.Layers.StdDataInit,
                    ),
                    conv_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        stride=2, activation=dsl.Layers.Tanh, normalizer=normalizer_cls,
                        data_init=dsl.Layers.StdDataInit,
                    ),
                    deconv_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        stride=2, activation=dsl.Layers.Tanh, normalizer=normalizer_cls,
                        data_init=dsl.Layers.StdDataInit,
                    ),
                    resblock_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        activation=dsl.Layers.Sigmoid, normalizer=normalizer_cls,
                        dropout=0.5, merge_context1=_MyAddContext,
                        data_init=dsl.Layers.StdDataInit,
                    ),
                ]
                # pixelcnn_layers = [t.layers.jit_compile(l) for l in pixelcnn_layers]

                # the pixelcnn network
                network_cls = getattr(dsl.Layers, f'PixelCNN{spatial_ndims}d')

                with pytest.raises(TypeError,
                                   match='`input_layer` must be an instance of'):
                    _ = network_cls(dsl.Layers.Linear(2, 3))

                network1 = network_cls(input_layer)
                network2 = network_cls(input_layer, pixelcnn_layers[0], pixelcnn_layers[1:])

                # ---- test the network ----
                x_t = torch.randn(make_conv_shape([3], in_channels, size))
                context = [torch.randn(make_conv_shape([3], out_channels, size))]

                _ = network2(torch.randn(list(x_t.shape)))  # run the initializers
                dsl.set_train_mode(network1, False)
                dsl.set_train_mode(network2, False)

                # without context
                expected_outputs2 = expected_outputs1 = input_layer(x_t)
                expected_output1 = expected_outputs1[-1]

                for l in pixelcnn_layers:
                    print(l)
                    expected_outputs2 = l(expected_outputs2)
                expected_output2 = expected_outputs2[-1]
                print('*'*56)
                assert (torch.abs(network1(x_t) - expected_output1)< 1e-3).all()
                print(network2)
                print(torch.max(torch.abs(network2(x_t) - expected_output2)))
                assert(torch.abs(network2(x_t) - expected_output2) < 1e-3).all()

                # with context
                expected_outputs2 = expected_outputs1 = input_layer(x_t)
                expected_output1 = expected_outputs1[-1]

                for l in pixelcnn_layers:
                    expected_outputs2 = l(expected_outputs2, context)
                expected_output2 = expected_outputs2[-1]

                assert (torch.abs(network1(x_t, context) - expected_output1) < 1e-3).all()
                print(torch.max(torch.abs(network2(x_t, context) - expected_output2)))

                assert(torch.abs(network2(x_t, context) -  expected_output2) < 1e-3).all()
