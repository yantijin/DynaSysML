import pytest
import DynaSysML as tsp
import torch
from ..ops import *
from ..helper import *
from DynaSysML.Flow import *
from DynaSysML.Layers.base import IS_CHANNEL_LAST
from DynaSysML.core import get_dtype


def check_act_norm(ctx, spatial_ndims: int, cls):
    num_features = 4
    channel_axis = get_channel_axis(spatial_ndims)

    def do_check(batch_shape, scale_type, initialized, dtype):
        x = torch.randn(make_conv_shape(
            batch_shape, num_features, [6, 7, 8][: spatial_ndims]), dtype=dtype)

        # check construct
        flow = cls(num_features, scale=scale_type, initialized=initialized,
                   dtype=dtype)
        assert(f'num_features={num_features}' in repr(flow))
        assert (f'axis={-(spatial_ndims + 1)}' in repr(flow))
        assert(f'scale_type={scale_type!r}' in repr(flow))

        # check initialize
        if not initialized:
            # must initialize with sufficient data
            with pytest.raises(Exception,
                               match='at least .* dimensions'):
                _ = flow(torch.randn(
                    make_conv_shape([], num_features, [6, 7, 8][: spatial_ndims]), dtype=dtype))

            # must initialize with inverse = Fale
            with pytest.raises(Exception,
                               match='`ActNorm` must be initialized with '
                                     '`inverse = False`'):
                _ = flow(x, inverse=True)

            # do initialize
            y, _ = flow(x, compute_log_det=False)
            y_mean, y_var = tsp.calculate_mean_and_var(
                y,
                axis=[a for a in range(-y.dim(), 0) if a != channel_axis]
            )
            assert (y_mean - torch.zeros([num_features]) < 1e-3).all()
            assert (y_var - torch.ones([num_features]) < 1e-3).all()
        else:
            y, _ = flow(x, compute_log_det=False)
            assert (y - x < 1e-3).all()

        # prepare for the expected result
        scale_obj = ExpScale() if scale_type == 'exp' else LinearScale()

        if IS_CHANNEL_LAST:
            aligned_shape = [num_features]
        else:
            aligned_shape = [num_features] + [1] * spatial_ndims
        bias = torch.reshape(flow.bias, aligned_shape)
        pre_scale = torch.reshape(flow.pre_scale, aligned_shape)

        expected_y, expected_log_det = scale_obj(
            x + bias, pre_scale, event_ndims=(spatial_ndims + 1), compute_log_det=True)

        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            torch.randn(list(expected_log_det.shape)))

    for batch_shape in ([11], [11, 12]):
        do_check(batch_shape, 'exp', False, torch.float32)

    for scale_type in ('exp', 'linear'):
        do_check([11], scale_type, False, torch.float32)

    for initialized in (True, False):
        do_check([11], 'exp', initialized, torch.float32)

    for dtype in float_dtypes:
        do_check([11], 'exp', False, dtype)


class TestActNorm():

    def test_ActNorm(self):
        # T.random.seed(1234)
        check_act_norm(self, 0, ActNorm)

    def test_ActNormNd(self):
        # T.random.seed(1234)
        for spatial_ndims in (1, 2, 3):
            check_act_norm(
                self,
                spatial_ndims,
                getattr(tsp.Flow, f'ActNorm{spatial_ndims}d')
            )
