import pytest
from DynaSysML.core import *
from DynaSysML.Layers import BaseLayer
import torch
import numpy as np


class TestModuleBufferParameters(object):
    def test_Module_Buffer(self):
        layer = BaseLayer()
        weight = torch.randn(5, 4)
        bias = torch.randn(4)
        cons1 = torch.randn(3, 3)
        cons2 = torch.randn(4,)
        add_parameter(layer, 'w', weight)
        add_parameter(layer, 'b', bias)

        add_buffer(layer, 'cons1', cons1)
        add_buffer(layer, 'cons2', cons2)

        assert (get_parameter(layer, 'w') - weight < 1e-6).all()
        assert (get_parameter(layer, 'b') - bias < 1e-6).all()
        assert (get_buffer(layer, 'cons1') - cons1 < 1e-6).all()
        assert (get_parameter(layer, 'cons1') - cons1 < 1e-6).all()
        for name, parameter in get_parameters(layer):
            if name == 'w':
                assert (parameter.data - weight < 1e-6).all()
            elif name == 'b':
                assert (parameter.data - bias < 1e-6).all()
            else:
                print(name)

        for name, buffer in get_buffers(layer):
            if name == 'cons1':
                assert (buffer.data - cons1 < 1e-6).all()
            elif name == 'cons2':
                assert  (buffer.data - cons2 < 1e-6).all()
            else:
                print(name)


class TestConvUtils(object):
    def test_pad(self):
        inputs1 = torch.randn(12, 3, 28, 28)
        padding1 = [(2, 2)]
        padding2 = [(1,1), (2,2)]
        padding3 = [(1, 2)]
        padding4 = [(1,2), (2,1)]
        out1 = pad(inputs1, padding1, value=0.)
        out2 = pad(inputs1, padding2, value=0.)
        out3 = pad(inputs1, padding3, value=0.)
        out4 = pad(inputs1, padding4, value=0.)
        assert (list(out1.shape) == [12, 3, 28, 32])
        assert (list(out2.shape) == [12, 3, 30, 32])
        assert (list(out3.shape) == [12, 3, 28, 31])
        assert (list(out4.shape) == [12, 3, 31, 31])


class TestShapeUtils(object):
    def test_squeeze(self):
        inputs1 = torch.randn(2, 1, 5, 3, 1)
        res1 = squeeze(inputs1)
        res2 = squeeze(inputs1, axis=[1])
        res3 = squeeze(inputs1, axis=[1, 4])

        assert  (list(res1.shape) == [2, 5, 3])
        assert  (list(res2.shape) == [2, 5, 3, 1])
        assert  (list(res3.shape) == [2, 5, 3])

    def test_flatten_to_ndims(self):
        inputs1 = torch.randn(13, 3, 28, 28)
        ndims = 2
        out1, front_shape = flatten_to_ndims(inputs1, ndims)
        assert (list(out1.shape) == [13*3*28, 28])
        out = unflatten_from_ndims(out1, front_shape)
        assert (out - inputs1 < 1e-6).all()

    def test_broadcast_to(self):
        inputs1 = torch.randn(4, 1, 15, 3, 1)
        inputs2 = torch.randn( 1, 15, 3, 2)
        new_shape = [4, 3, 15, 3, 2]
        res1 = broadcast_to(inputs1, new_shape)
        res2 = broadcast_to(inputs2, new_shape)
        assert (list(res1.shape) == new_shape)
        assert (list(res2.shape) == new_shape)

    def test_broadcast_shape(self):
        a = [1, 2, 1, 3]
        b = [2, 1, 3, 1]
        c = [2, 3, 3]
        shape1 = broadcast_shape(a, b)
        print(shape1)
        shape2 = broadcast_shape(a, c)
        print(shape2)
        assert (shape1 == [2, 2, 3, 3])
        assert (shape2 == [1, 2, 3, 3])

    def test_assignment(self):
        inputs1 = torch.randn(10, 4)
        inputs2 = torch.empty(3, 4)
        inputs1 = fill(inputs1, 1.0)
        inputs2 = fill(inputs2, 2.)
        assert (inputs1 == torch.ones(10, 4)).all()
        assert (inputs2 == 2 * torch.ones(3, 4)).all()

        inputs1 = fill_zeros(inputs1)
        assert (inputs1 == torch.zeros(10, 4)).all()
        data = torch.randn(3, 4)
        res = assign_data(inputs2, data)
        assert(res - data < 1e-6).all()

        zero_tensor = zeros([3,4], dtype='int8')
        assert (zero_tensor.dtype == torch.int8)
        assert (zero_tensor - torch.zeros([3,4]) < 1e-6).all()

    def test_variable(self):
        shape1 = [3, 4]
        shape2 = [2, 3, 4, 5]
        dtype1 = 'float32'
        dtype2 = torch.float64
        initializer1 = 1
        initializer2 = torch.randn(3, 4)
        a1 = variable(shape1, dtype1, initializer1,)
        a2 = variable(shape2, dtype2)
        a3 = variable(shape1, dtype2, initializer2)
        assert (a1.dtype == torch.float32)
        assert (list(a2.shape) == shape2)
        assert (a2.dtype == dtype2)
        assert (a3.data - initializer2 < 1e-6).all()

    def test_as_tensor(self):
        dtype1 = torch.float32
        dtype2 = 'float32'
        inputs = np.random.randn(3, 4)
        a1 = as_tensor(inputs, dtype1)
        a2 = as_tensor(inputs, dtype2)
        assert (a1.dtype, torch.float32)
        assert (a2.dtype, torch.float32)
        assert (a1.data - inputs < 1e-6).all()
        assert (a2.data - inputs < 1e-6).all()


class TestReduction(object):
    def test_reduce(self):
        input1 = torch.randn(3, 4, 5)
        axises = [1, [1, 2], -1, None]
        keep_dims = [True, False]
        res_sum = []
        res_mean = []
        res_max = []
        res_min = []
        for axis in axises:
            for keep_dim in keep_dims:
                res_sum.append(reduce_sum(input1, axis, keep_dim))
                res_mean.append(reduce_mean(input1, axis, keep_dim))
                res_max.append(reduce_max(input1, axis, keep_dim))
                res_min.append(reduce_min(input1, axis, keep_dim))
        res_shape = [[3,1,5], [3,5], [3,1,1], [3], [3,4,1], [3,4], [1,1,1],[]]
        for i in range(8):
            assert (list(res_sum[i].shape) == res_shape[i])
            assert (list(res_mean[i].shape) == res_shape[i])
            assert (list(res_max[i].shape) == res_shape[i])
            assert (list(res_min[i].shape) == res_shape[i])


class TestCalculation(object):
    def test_log_mean_exp_log_sum_exp(self):
        input1 = torch.randn(3, 4, 5)
        axis = [-1, 1, [1, 2], None]
        keep_dims = [True, False]
        res_log_mean_exp = []
        res_log_sum_exp = []
        for ax in axis:
            for keep_dim in keep_dims:
               res_log_mean_exp.append(log_mean_exp(input1, ax, keep_dim))
               res_log_sum_exp.append(log_sum_exp(input1, ax, keep_dim))
        shape1 = [[3,4,1], [3,4], [3,1,5],[3,5], [3,1,1], [3], [1,1,1], []]
        for i in range(8):
            assert list(res_log_sum_exp[i].shape) == shape1[i]
            assert list(res_log_mean_exp[i].shape) == shape1[i]

    def test_norm_except_axis(self):
        input = torch.randn(3, 4, 5)
        axises = [1, -1, [1,2], None]
        keepdims = [True, False]
        res = []
        for axis in axises:
            for keepdim in keepdims:
               res.append(norm_except_axis(input, axis, keepdims=keepdim))

        shape1 = [[1,4,1], [4], [1,1,5],[5], [1,4,5], [4,5], [1,1,1], []]
        for i in range(8):
            assert list(res[i].shape) == shape1[i]


















