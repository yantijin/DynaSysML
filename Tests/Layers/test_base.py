import pytest
from DynaSysML.Layers.base import *
from DynaSysML.Layers import initializer
from DynaSysML.core import *
import numpy as np
import torch
from DynaSysML.core import as_tensor
from DynaSysML.typing_ import *
from typing import *
from Tests.ops import *
import DynaSysML as dsl
from itertools import product

class TestConstant_Param(object):
    def test_constant(self):
        assert DEFAULT_GATE_BIAS == 2.0
        assert DEFAULT_BIAS_INIT == initializer.zeros
        assert DEFAULT_WEIGHT_INIT == initializer.kaming_uniform
        assert EPSILON == 1e-5
        assert IS_CHANNEL_LAST == False

    def test_SimParam(self):
        init1 = 2
        init2 = 2.0
        init3 = np.array([3,4,5])
        init4 = initializer.kaming_normal
        shape = [3]

        res1 = SimpleParamStore(shape, init1)
        res2 = SimpleParamStore(shape, init2)
        res3 = SimpleParamStore(shape, init3)
        res4 = SimpleParamStore([3, 5, 4,4], init4)
        assert  (res1().data - as_tensor([2, 2, 2])< 1e-6).all()
        assert  (res2().data - as_tensor([2.0, 2.0, 2.0])<1e-6) .all()
        assert  (res3().data - as_tensor([3, 4, 5])<1e-6).all()
        assert  (list(res4().shape) == [3, 5, 4, 4])
        res1.set([1,2,3])
        assert (res1().data - as_tensor([1,2,3]) < 1e-6).all()

    def test_weight_norm(self):
        initial_value = np.random.randn(2, 3, 4)
        new_value = np.random.randn(2, 3, 4)
        for norm_axis in [-3, -2, -1,0, 1, 2]:
            store = NormedWeightStore([2,3,4], initializer=initial_value, norm_axis=norm_axis)
            # print(norm_axis, repr(store))
            expected_value = as_tensor(initial_value) / norm_except_axis(
                as_tensor(initial_value), axis=norm_axis, keepdims=True
            )
            assert (store.get().data - expected_value < 1e-6).all()
            assert (store() - expected_value < 1e-6).all()
            assert (store.v - expected_value < 1e-6).all()
            store.set(as_tensor(new_value))
            expected_value = as_tensor(new_value) / norm_except_axis(
                as_tensor(new_value), axis=norm_axis, keepdims=True
            )
            assert (store.get() - expected_value < 1e-6).all()
            assert (store() - expected_value < 1e-6).all()
            assert (store.v - expected_value < 1e-6).all()

    def test_NormedAndScaledWeightStore(self):
        initial_value = np.random.randn(2, 3, 4)
        new_value = np.random.randn(2, 3, 4)
        for norm_axis in [-3, -2, -1, 0, 1, 2]:
            store = NormedAndScaledWeightStore(
                [2,3,4], initializer=initial_value, norm_axis=norm_axis
            )
            assert (store.get().data - initial_value < 1e-3).all()
            assert (store().data - initial_value < 1e-3).all()
            assert (store.g - norm_except_axis(as_tensor(initial_value), norm_axis, keepdims=True) < 1e-3).all()
            assert (store.v - as_tensor(initial_value) / store.g < 1e-3).all()

            store.set(as_tensor(new_value))
            assert (store.get() - as_tensor(new_value) < 1e-3).all()
            assert (store().data - new_value < 1e-3).all()
            assert (store.g - norm_except_axis(as_tensor(new_value), norm_axis, keepdims=True) < 1e-3).all()
            assert (store.v - as_tensor(new_value) / store.g < 1e-3).all()

    def test_get_weight_store(self):
        for term in [True, 'full', WeightNormMode.FULL]:
            store = get_weight_store([2,3,4], weight_norm=term)
            assert isinstance(store, NormedAndScaledWeightStore)
            assert (list(store().shape) == [2,3,4])
        for term in ['no_scale', WeightNormMode.NO_SCALE]:
            store = get_weight_store([2, 3, 4], weight_norm=term)
            assert isinstance(store, NormedWeightStore)
            assert (list(store().shape) == [2, 3, 4])
        for term in [False, 'none', WeightNormMode.NONE]:
            store = get_weight_store([2,3,4], weight_norm=term)
            assert isinstance(store, SimpleParamStore)
            assert (list(store().shape) == [2,3,4])

        with pytest.raises(ValueError, match='Invalid value for argument `weight_norm`'):
            store = get_weight_store([2,3,4], weight_norm='invalid')


    def test_get_bias_store(self):
        store = get_bias_store([2,3,4], use_bias=True)
        assert isinstance(store, SimpleParamStore)
        assert (list(store().shape) == [2,3,4])

        store = get_bias_store([2,3,4], use_bias=False)
        assert (store == None)

class _MySingleVariateLayer(BaseSingleVariateLayer):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def set_bias(self, input: Tensor):
        self.bias = as_tensor(input)

    def _forward(self, input: Tensor):
        return input + self.bias


class _MyMultiVariateLayer(BaseMultiVariateLayer):
    def _forward(self, input: List[Tensor]):
        res = []
        for index in range(len(input) - 1):
            res.append(input[index+1] + input[index])
        return res

class _MyBaseSplitLayer(BaseSplitLayer):
    def _forward(self, input: Tensor):
        return [input, input+1, input+2]

class _MyBaseMergeLayer(BaseMergeLayer):
    def _forward(self, inputs: List[Tensor]):
        res = inputs[0].clone()
        for tensor in inputs[1:]:
            res += tensor
        return res


class TestBaseLayers(object):
    def test_single_variate_layer(self):
        layer = _MySingleVariateLayer(torch.tensor(0.))
        inputs = torch.randn(2,3,4)
        res = layer(inputs)
        assert (res - inputs < 1e-6).all()
        layer.set_bias(7.)
        res = layer(inputs)
        assert (res - 7 - inputs < 1e-4).all()

    def test_multi_variate_layer(self):
        layer = _MyMultiVariateLayer()
        x = torch.randn(2,3,4)
        y = torch.randn(2,3,4)
        z = torch.randn(2,3,4)
        a, b = layer([x, y, z])
        assert (a -x - y < 1e-6).all()
        assert (b-y-z<1e-6).all()

    def test_base_split(self):
        layer = _MyBaseSplitLayer()
        inputs = torch.randn(2,3,4)
        a,b,c = layer(inputs)
        assert (a == inputs).all()
        assert (b == inputs + 1).all()
        assert (c == inputs + 2).all()

    def test_base_merge(self):
        layer = _MyBaseMergeLayer()
        x = torch.randn(2,3,4)
        y = torch.randn(2,3,4)
        zz = torch.randn(2,3,4)
        aa = torch.randn(2,3,4)
        res = layer([x, y, zz, aa])
        # assert (res == x).all()
        assert (res.data == x+y+zz+aa).all()


class TestSequential(object):
    def test_sequential(self):
        layers = [torch.nn.Linear(5, 5) for _ in range(5)]
        a = Sequential(layers[0], layers[1:2],[layers[2], layers[3], layers[4]])
        assert (list(a) == layers)
        inputs = torch.randn([4,5])
        y = a(inputs)
        y2 = inputs
        for layer in layers:
            y2 = layer(y2)
        assert (y - y2 < 1e-6).all()


def check_core_linear(ctx, input, layer_factory, layer_name, numpy_fn):

    # print(layer_name)
    # test with bias
    layer = layer_factory(use_bias=True)
    assert (layer_name in repr(layer))
    assert isinstance(layer.weight_store, SimpleParamStore)
    weight = to_numpy(layer.weight_store())
    bias = to_numpy(layer.bias_store())
    res1 = layer(as_tensor(input, dtype=torch.float32))
    res2 = numpy_fn(input, weight, bias)
    assert (res1 - as_tensor(res2) < 1e-6).all()
    assert ('use_bias=' not in repr(layer))

    # test without bias
    layer = layer_factory(use_bias=False)
    assert isinstance(layer.weight_store, SimpleParamStore)
    weight = to_numpy(layer.weight_store())
    res1 = layer(as_tensor(input, dtype=torch.float32))
    res2 = numpy_fn(input, weight, None)
    assert (res1 - as_tensor(res2) < 1e-6).all()
    assert ('use_bias=False' in repr(layer))

    # test `weight_norm`
    for wn in [True, WeightNormMode.FULL, 'full']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        assert isinstance(layer.weight_store, NormedAndScaledWeightStore)
        weight = to_numpy(layer.weight_store())
        res1 = layer(as_tensor(input, dtype=torch.float32))
        res2 = numpy_fn(input, weight, None)
        assert (res1 - as_tensor(res2) <1e-6).all()

    for wn in [WeightNormMode.NO_SCALE, 'no_scale']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        assert isinstance(layer.weight_store, NormedWeightStore)
        weight = to_numpy(layer.weight_store())
        res1 = layer(as_tensor(input, dtype=torch.float32))
        res2 = numpy_fn(input, weight, None)
        assert (res1 - as_tensor(res2) < 1e-6).all()

    for wn in [False, WeightNormMode.NONE, 'none']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        assert  isinstance(layer.weight_store, SimpleParamStore)

    # TODO: test `data_init`


class TestCoreLinearCase(object):
    def test_linear(self):
        layer = Linear(5, 3)
        check_core_linear(
            self,
            np.random.randn(5,3).astype(np.float32),
            (lambda **kwargs: Linear(3, 4, **kwargs)),
            'Linear',
            dense
        )

        check_core_linear(
            self,
            np.random.randn(10,5,3).astype(np.float32),
            (lambda **kwargs: Linear(3, 4, **kwargs)),
            'Linear',
            dense
        )

    def test_conv_nd(self):
        def do_check(spatial_ndims, kernel_size, stride, dilation, padding):
            cls_name = f'LinearConv{spatial_ndims}d'
            layer_factory = getattr(dsl.Layers, cls_name)
            check_core_linear(
                self,
                np.random.randn(
                    *make_conv_shape(
                        [2], 3, [14, 13, 12][: spatial_ndims]
                    )
                ).astype(np.float32),
                (lambda **kwargs: layer_factory(
                    in_channels=3, out_channels=4,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, **kwargs
                )),
                cls_name,
                (lambda input, weight, bias: conv_nd(
                    input, kernel=weight, bias=bias, stride=stride,
                    padding=padding, dilation=dilation,
                ))
            )

        for spatial_ndims in (1,2):
            for kernel_size, stride, padding, dilation in product(
                    (1, (3,2,1)[:spatial_ndims]),
                    (1, (3,2,1)[:spatial_ndims]),
                    (0, 1, ((4,3),3,(2,1))[:spatial_ndims], PaddingMode.FULL, PaddingMode.HALF, PaddingMode.NONE),
                    (1,(3,2,1)[:spatial_ndims])
            ):
                do_check(spatial_ndims, kernel_size, stride, dilation, padding)

        do_check(3, (3,2,1),(3,2,1), (3,2,1),PaddingMode.HALF)

    def test_conv_transpose_nd(self):
        def is_valid_output_padding(spatial_ndims, output_padding, stride, dilation):
            if not hasattr(output_padding, '__iter__'):
                output_padding = [output_padding] * spatial_ndims
            if not hasattr(stride, '__iter__'):
                stride = [stride] * spatial_ndims
            if not hasattr(dilation, '__iter__'):
                dilation = [dilation] * spatial_ndims
            for op, s, d in zip(output_padding, stride, dilation):
                if op >= s and op >= d:
                    return False
            return True

        def do_check(spatial_ndims, kernel_size, stride,
                     dilation, padding, output_padding):
            cls_name = f'LinearConvTranspose{spatial_ndims}d'
            layer_factory = getattr(dsl.Layers, cls_name)
            fn = lambda: check_core_linear(
                self,
                np.random.randn(
                    *make_conv_shape(
                        [2], 3, [9, 8, 7][: spatial_ndims]
                    )
                ).astype(np.float32),
                (lambda **kwargs: layer_factory(
                    in_channels=3, out_channels=4,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding,
                    dilation=dilation, **kwargs
                )),
                cls_name,
                (lambda input, weight, bias: conv_transpose_nd(
                    input, kernel=weight, bias=bias, stride=stride,
                    padding=padding, output_padding=output_padding,
                    dilation=dilation,
                )),
            )

            if not is_valid_output_padding(
                    spatial_ndims, output_padding, stride, dilation):
                with pytest.raises(Exception, match='`output_padding`'):
                    fn()
            else:
                fn()

        for spatial_ndims in (1, 2):
            for kernel_size, stride, padding, output_padding, dilation in product(
                    (1, (3, 2, 1)[: spatial_ndims]),
                    (1, (3, 2, 1)[: spatial_ndims]),
                    (0, 1, ((4, 3), 3, (2, 1))[: spatial_ndims],
                     PaddingMode.FULL, PaddingMode.HALF, PaddingMode.NONE),
                    (0, 1, (3, 2, 1)[: spatial_ndims]),
                    (1, (3, 2, 1)[: spatial_ndims]),
            ):
                do_check(spatial_ndims, kernel_size, stride, dilation, padding,
                         output_padding)

        # 3d is too slow, just do one particular test
        do_check(3, (3, 2, 1), (3, 2, 1), (3, 2, 1), PaddingMode.HALF, 0)


class TestBatchNorm(object):
    def test_batch_norm(self):
        eps = 1e-5
        for spatial_ndims in (0, 1, 2, 3):
            cls = getattr(dsl.Layers, ('BatchNorm' if not spatial_ndims
                                      else f'BatchNorm{spatial_ndims}d'))
            layer = cls(5, momentum=0.1, epsilon=eps)
            assert ('BatchNorm' in repr(layer))
            # layer = T.jit_compile(layer)

            # layer output
            x = torch.randn(make_conv_shape(
                [3], 5, [6, 7, 8][:spatial_ndims]
            ))

            set_train_mode(layer)
            _ = layer(x)
            set_train_mode(layer, False)
            y = layer(x)

            # manually compute the expected output
            # if T.backend_name == 'PyTorch':
            dst_shape = [-1] + [1] * spatial_ndims
            weight = torch.reshape(layer.weight, dst_shape)
            bias = torch.reshape(layer.bias, dst_shape)
            running_mean = torch.reshape(layer.running_mean, dst_shape)
            running_var = torch.reshape(layer.running_var, dst_shape)
            expected = (((x - running_mean) / torch.sqrt(running_var + eps)) *
                        weight + bias)
            # else:
            #     raise RuntimeError()

            # check output
            assert (y-expected<1e-6).all()

            # check invalid dimensions
            with pytest.raises(Exception, match='only supports .d input'):
                _ = layer(
                    torch.randn(make_conv_shape(
                        [], 5, [6, 7, 8][:spatial_ndims]
                    ))
                )


class TestDropout(object):
    def test_dropout(self):
        n_samples = 10000
        for spatial_ndims in (0, 1, 2, 3):
            cls = getattr(dsl.Layers, ('Dropout' if not spatial_ndims
                                      else f'Dropout{spatial_ndims}d'))
            layer = cls(p=0.3)
            assert('p=0.3' in repr(layer))
            assert ('Dropout' in repr(layer))
            # layer = T.jit_compile(layer)

            x = 1. + torch.randn(
                make_conv_shape([1], n_samples, [2, 2, 2][:spatial_ndims])
            )

            # ---- train mode ----
            set_train_mode(layer, True)
            y = layer(x)

            # check: channels should be all zero or no zero
            spatial_axis = tuple(get_spatial_axis(spatial_ndims))

            all_zero = np.all(to_numpy(y) == 0, axis=spatial_axis, keepdims=True)
            no_zero = np.all(to_numpy(y) != 0, axis=spatial_axis, keepdims=True)
            assert (np.logical_or(all_zero, no_zero).all() == True)

            # check: the probability of not being zero
            assert (
                np.abs(np.mean(all_zero) - 0.3) <=
                5.0 / np.sqrt(n_samples) * 0.3 * 0.7  # 5-sigma
            )

            # check: the value
            assert (y - (to_numpy(x) * no_zero) / 0.7 < 1e-6).all()

            # ---- eval mode ----
            set_train_mode(layer, False)
            y = layer(x)
            assert (np.all(to_numpy(y) != 0) == True)
            assert (y-x<1e-6).all()








