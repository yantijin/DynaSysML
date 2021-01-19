import pytest
from DynaSysML.Flow.shuffle import *
from Tests.helper import *
from Tests.ops import *
from DynaSysML.core import get_parameter, index_select
import torch
import DynaSysML as dsl




def check_shuffling_flow(ctx,
                         spatial_ndims: int,
                         cls):
    num_features = 5

    for batch_shape in ([2], [2, 3]):
        shape = make_conv_shape(
            batch_shape, num_features, [6, 7, 8][: spatial_ndims])

        # test constructor
        flow = cls(num_features)
        assert(f'num_features={num_features}' in repr(flow))
        permutation = get_parameter(flow, 'permutation')
        inv_permutation = get_parameter(flow, 'inv_permutation')
        assert(torch.argsort(permutation) == inv_permutation).all()
        assert(torch.argsort(inv_permutation) == permutation).all()
        # flow = T.jit_compile(flow)

        # prepare for the answer
        x = torch.randn(shape)
        channel_axis = get_channel_axis(spatial_ndims)
        expected_y = index_select(x, permutation, axis=channel_axis)
        assert(
            index_select(expected_y, inv_permutation, axis=channel_axis) - x < 1e-6
        ).all()
        expected_log_det = torch.zeros(batch_shape)

        # check the flow
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            torch.randn(batch_shape))


class TestRearrangement(object):

    def test_FeatureShuffleFlow(self):
        check_shuffling_flow(self, 0, FeatureShufflingFlow)

    def test_FeatureShuffleFlowNd(self):
        for spatial_ndims in (1, 2, 3):
            check_shuffling_flow(
                self,
                spatial_ndims,
                getattr(dsl.Flow, f'FeatureShufflingFlow{spatial_ndims}d'),
            )
