import torch
import pytest
import DynaSysML as dsl
from DynaSysML.Flow.scale import *
from  DynaSysML.core import reduce_sum, broadcast_to
from DynaSysML.typing_ import *
from typing import *

def log_sigmoid(x: Tensor) -> Tensor:
    # using `neg_x` and `pos_x` separately can avoid having NaN or Infinity
    # on either of the path.
    neg_x = torch.min(x, torch.as_tensor(0., dtype=x.dtype))
    pos_x = torch.max(x, torch.as_tensor(0., dtype=x.dtype))
    return torch.where(
        x < 0.,
        neg_x - torch.log1p(torch.exp(neg_x)),  # log(exp(x) / (1 + exp(x)))
        -torch.log1p(torch.exp(-pos_x))     # log(1 / (1 + exp(-x)))
    )

def check_scale(ctx,
                scale: BaseScale,
                x,
                pre_scale,
                expected_y,
                expected_log_det):
    assert(list(x.shape) == list(expected_log_det.shape))

    # dimension error
    with pytest.raises(Exception,
                       match=r'`rank\(input\) >= event_ndims` does not hold'):
        _ = scale(torch.randn([1]), torch.randn([1]), event_ndims=2)

    with pytest.raises(Exception,
                       match=r'`rank\(input\) >= rank\(pre_scale\)` does not hold'):
        _ = scale(torch.randn([1]), torch.randn([1, 2]), event_ndims=1)

    with pytest.raises(Exception,
                       match=r'The shape of `input_log_det` is not expected'):
        _ = scale(torch.randn([2, 3]),
                  torch.randn([2, 3]),
                  event_ndims=1,
                  input_log_det=torch.randn([3]))

    with pytest.raises(Exception,
                       match=r'The shape of `input_log_det` is not expected'):
        _ = scale(torch.randn([2, 3]),
                  torch.randn([2, 3]),
                  event_ndims=2,
                  input_log_det=torch.randn([2]))

    # check call
    for event_ndims in range(pre_scale.dim(), x.dim()):
        this_expected_log_det = reduce_sum(
            expected_log_det, axis=list(range(-event_ndims, 0)))
        input_log_det = torch.randn(list(this_expected_log_det.shape))

        # check no compute log_det
        y, log_det = scale(x, pre_scale, event_ndims=event_ndims,
                           compute_log_det=False)
        assert (y- expected_y< 1e-6).all()
        assert (log_det == None)

        # check compute log_det
        y, log_det = scale(x, pre_scale, event_ndims=event_ndims)
        assert(y-expected_y<1e-6).all()
        assert(log_det-this_expected_log_det<1e-6).all()

        # check compute log_det with input_log_det
        y, log_det = scale(
            x, pre_scale, event_ndims=event_ndims, input_log_det=input_log_det)
        assert(y -expected_y<1e-6).all()
        assert(log_det-input_log_det -this_expected_log_det<1e-6).all()

        # check inverse, no compute log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True, compute_log_det=False)
        assert (inv_x -x<1e-6).all()
        assert(log_det == None)

        # check inverse, compute log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True)
        assert(inv_x-x<1e-6).all()
        assert(log_det + this_expected_log_det<1e-6).all()

        # check inverse, compute log_det with input_log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True, input_log_det=input_log_det)
        assert(inv_x -x<1e-6).all()
        assert(log_det-input_log_det +this_expected_log_det<1e-6).all()


class _BadScale1(BaseScale):

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        scale = pre_scale
        if compute_log_scale:
            log_scale: Optional[Tensor] = torch.randn([2, 3, 4])
        else:
            log_scale: Optional[Tensor] = None
        return scale, log_scale


class _BadScale2(BaseScale):

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        scale = pre_scale
        if compute_log_scale:
            log_scale: Optional[Tensor] = torch.randn([1] + list(pre_scale.shape))
        else:
            log_scale: Optional[Tensor] = None
        return scale, log_scale


class TestScale(object):

    def test_ExpScale(self):
        # T.random.seed(1234)

        x = torch.randn([2, 3, 4])
        scale = ExpScale()
        # scale = T.jit_compile(scale)

        for pre_scale in [torch.randn([4]),
                          torch.randn([3, 1]),
                          torch.randn([2, 1, 1]),
                          torch.randn([2, 3, 4])]:
            expected_y = x *  torch.exp(pre_scale)
            expected_log_det = broadcast_to(pre_scale, list(x.shape))
            check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_SigmoidScale(self):
        # T.random.seed(1234)

        x = torch.randn([2, 3, 4])

        for pre_scale_bias in [None, 0., 1.5]:
            scale = SigmoidScale(**(
                {'pre_scale_bias': pre_scale_bias}
                if pre_scale_bias is not None else {}
            ))
            if pre_scale_bias is None:
                pre_scale_bias = 0.
            assert (f'pre_scale_bias={pre_scale_bias}' in repr(scale))
            # scale = T.jit_compile(scale)

            for pre_scale in [torch.randn([4]),
                              torch.randn([3, 1]),
                              torch.randn([2, 1, 1]),
                              torch.randn([2, 3, 4])]:
                expected_y = x * torch.sigmoid(pre_scale + pre_scale_bias)
                expected_log_det = broadcast_to(
                    log_sigmoid(pre_scale + pre_scale_bias), list(x.shape))
                check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_LinearScale(self):
        # T.random.seed(1234)

        x = torch.randn([2, 3, 4])
        scale = LinearScale(epsilon=1e-5)
        assert('epsilon=' in repr(scale))
        # scale = T.jit_compile(scale)

        for pre_scale in [torch.randn([4]),
                          torch.randn([3, 1]),
                          torch.randn([2, 1, 1]),
                          torch.randn([2, 3, 4])]:
            expected_y = x * pre_scale
            expected_log_det = broadcast_to(
                torch.log(torch.abs(pre_scale)), list(x.shape))
            check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_bad_output(self):
        # T.random.seed(1234)
        x = torch.randn([2, 3, 1])

        scale = _BadScale1()
        with pytest.raises(Exception,
                           match='The shape of the final 1d of `log_scale` is '
                                 'not expected'):
            _ = scale(x, x, event_ndims=1)

        scale = _BadScale2()
        with pytest.raises(Exception, match='shape'):
            _ = scale(x, x, event_ndims=0)
        with pytest.raises(Exception,
                           match='The shape of the computed `output_log_det` '
                                 'is not expected'):
            _ = scale(x, x, event_ndims=0, input_log_det=x)
