import pytest
import DynaSysML as dsl
from DynaSysML.Flow.reshape import *
from ..helper import *
from ..ops import *
from .test_base import reshape_tail
from itertools import product
import torch



class TestReshapeFlow(object):

    def test_ReshapeFlow(self):
        flow = ReshapeFlow([4, -1], [-1])
        assert (flow.x_event_shape == [4, -1])
        assert(flow.y_event_shape == [-1])
        assert(flow.x_event_ndims ==2)
        assert(flow.y_event_ndims == 1)
        assert ('x_event_shape=[4, -1]' in repr(flow))
        assert('y_event_shape=[-1]'in repr(flow))

        x = torch.randn([2, 3, 4, 5])
        expected_y = reshape_tail(x, 2, [-1])
        expected_log_det = torch.zeros([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            torch.randn([2, 3]))

        with pytest.raises(ValueError,
                           match='Too many `-1` specified in `x_event_shape`'):
            _ = ReshapeFlow([-1, -1], [-1])

        with pytest.raises(ValueError,
                           match='All elements of `x_event_shape` must be '
                                 'positive integers or `-1`'):
            _ = ReshapeFlow([-1, -2], [-1])

        with pytest.raises(ValueError,
                           match='Too many `-1` specified in `y_event_shape`'):
            _ = ReshapeFlow([-1], [-1, -1])

        with pytest.raises(ValueError,
                           match='All elements of `y_event_shape` must be '
                                 'positive integers or `-1`'):
            _ = ReshapeFlow([-1], [-1, -2])


class TestSpaceDepthTransformFlow():

    def test_space_depth_transform(self):
        # T.random.seed(1234)

        for spatial_ndims, batch_shape, block_size in product(
                    (1, 2, 3),
                    ([2], [2, 3]),
                    (1, 2, 4),
                ):
            # prepare for the data
            n_channels = 5
            x = torch.randn(make_conv_shape(
                batch_shape, n_channels, [4, 8, 12][:spatial_ndims]))
            y = getattr(dsl.Flow, f'space_to_depth{spatial_ndims}d')(x, block_size)
            log_det = torch.zeros(batch_shape)
            input_log_det = torch.randn(batch_shape)

            # construct the classes
            cls = getattr(dsl.Flow, f'SpaceToDepth{spatial_ndims}d')
            inv_cls = getattr(dsl.Flow, f'DepthToSpace{spatial_ndims}d')

            flow = cls(block_size)
            assert(flow.block_size == block_size)
            inv_flow = inv_cls(block_size)
            assert(inv_flow.block_size == block_size)

            assert isinstance(flow.invert(), inv_cls)
            assert(flow.invert().block_size == block_size)
            assert isinstance(inv_flow.invert(), cls)
            assert (inv_flow.invert().block_size == block_size)

            # check call
            flow_standard_check(self, flow, x, y, log_det, input_log_det)
            flow_standard_check(self, inv_flow, y, x, log_det, input_log_det)

            # test error
            with pytest.raises(ValueError,
                               match='`block_size` must be at least 1'):
                _ = cls(0)

            with pytest.raises(ValueError,
                               match='`block_size` must be at least 1'):
                _ = inv_cls(0)
