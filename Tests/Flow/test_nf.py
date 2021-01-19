import pytest
import DynaSysML as dsl
import torch
from DynaSysML.Flow import Planar
from ..helper import noninvert_flow_standard_check
from DynaSysML.core import flatten_to_ndims, unflatten_from_ndims

class TestNF(object):
    def test_planar(self):
        input1 = torch.randn(12,5)
        input2 = torch.randn(3, 4, 5)
        model1 = Planar(num_features=5,)
        model2 = Planar(num_features=5, event_ndims=1)
        model3 = Planar(num_features=5, event_ndims=2)
        input_log_det1 = torch.randn(12)
        input_log_det2 = torch.randn(3, 4)

        x_flatten, front_shape = flatten_to_ndims(input1, 2)
        wxb = torch.matmul(x_flatten, model1.w.T) + model1.b
        tanh_wb = torch.tanh(wxb)
        out = x_flatten + model1.get_uhat() * tanh_wb
        expected_y = unflatten_from_ndims(out, front_shape=front_shape)
        grad = 1. - tanh_wb ** 2
        phi = grad * model1.w # shape == [?, n_units]
        u_phi = torch.matmul(phi, model1.get_uhat().T)
        log_det = torch.log(torch.abs(1. + u_phi))  # [? 1]
        expected_log_det = unflatten_from_ndims(log_det, front_shape)
        expected_log_det = expected_log_det.squeeze()
        print('expect', expected_log_det.shape)

        noninvert_flow_standard_check(self, model1, input1, expected_y,
                                      expected_log_det, input_log_det1)

        x_flatten, front_shape = flatten_to_ndims(input2, 2)
        wxb = torch.matmul(x_flatten, model2.w.T) + model2.b
        tanh_wb = torch.tanh(wxb)
        out = x_flatten + model2.get_uhat() * tanh_wb
        expected_y = unflatten_from_ndims(out, front_shape=front_shape)
        grad = 1. - tanh_wb ** 2
        phi = grad * model2.w  # shape == [?, n_units]
        u_phi = torch.matmul(phi, model2.get_uhat().T)
        log_det = torch.log(torch.abs(1. + u_phi))  # [? 1]
        expected_log_det = unflatten_from_ndims(log_det, front_shape)
        expected_log_det = expected_log_det.squeeze()
        print('expect', expected_log_det.shape)

        noninvert_flow_standard_check(self, model2, input2, expected_y,
                                      expected_log_det, input_log_det2)
