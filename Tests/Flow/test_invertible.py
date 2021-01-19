import pytest
import torch
import DynaSysML as dsl
from DynaSysML.Flow.invertible import *
from Tests.helper import flow_standard_check
from DynaSysML.core import flatten_to_ndims, unflatten_from_ndims, reduce_sum


def check_invertible_linear(ctx,
                            spatial_ndims: int,
                            invertible_linear_factory,
                            linear_factory,
                            strict: bool,):
    for batch_shape in ([2], [2, 3]):
        num_features = 4
        spatial_shape = [5, 6, 7][:spatial_ndims]
        x = torch.randn(list(batch_shape)+ [num_features]+ list(spatial_shape))

        # construct the layer
        flow = invertible_linear_factory(num_features, strict=strict)
        assert(f'num_features={num_features}' in repr(flow))

        # derive the expected answer
        weight, log_det = flow.invertible_matrix(
            inverse=False, compute_log_det=True)
        linear_kwargs = {}
        if spatial_ndims > 0:
            linear_kwargs['kernel_size'] = 1
        linear = linear_factory(
            num_features, num_features,
            weight_init=torch.reshape(weight, list(weight.shape) + [1] * spatial_ndims),
            use_bias=False,
            **linear_kwargs
        )
        x_flatten, front_shape = flatten_to_ndims(x, spatial_ndims + 2)
        expected_y = unflatten_from_ndims(linear(x_flatten), front_shape)
        expected_log_det = reduce_sum(log_det.expand(spatial_shape)).expand(batch_shape)

        # check the invertible layer
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            torch.randn(list(batch_shape)))


class TestInvertibleLinear(object):
    def test_invertible_dense(self):
        # T.random.seed(1234)
        for strict in (True, False):
            check_invertible_linear(
                self,
                spatial_ndims=0,
                invertible_linear_factory=InvertibleDense,
                linear_factory=dsl.Layers.Linear,
                strict=strict
            )

    def test_invertible_conv_nd(self):
        # T.random.seed(1234)
        # when strict is `False`, sometimes the test cannot go through
        for spatial_ndims in (1, 2, 3):
            for strict in (True,):
                print(spatial_ndims, strict)
                check_invertible_linear(
                    self,
                    spatial_ndims=spatial_ndims,
                    invertible_linear_factory=getattr(
                        dsl.Flow, f'InvertibleConv{spatial_ndims}d'),
                    linear_factory=getattr(
                        dsl.Layers, f'LinearConv{spatial_ndims}d'),
                    strict=strict
                )


        for spatial_ndims in (1, 2, 3):
            for strict in (False,):
                print(spatial_ndims, strict)
                check_invertible_linear(
                    self,
                    spatial_ndims=spatial_ndims,
                    invertible_linear_factory=getattr(
                        dsl.Flow, f'InvertibleConv{spatial_ndims}d'),
                    linear_factory=getattr(
                        dsl.Layers, f'LinearConv{spatial_ndims}d'),
                    strict=strict
                )
