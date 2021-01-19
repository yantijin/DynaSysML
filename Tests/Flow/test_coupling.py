import pytest
import DynaSysML as dsl
from Tests.ops import *
import torch
from DynaSysML.Flow.coupling import *
from DynaSysML.Flow.scale import *
from ..helper import flow_standard_check
from DynaSysML.typing_ import *
from typing import *

class Branch(dsl.Layers.BaseSplitLayer):
    """
    A module that maps the input tensor into multiple tensors via sub-modules.

    ::

        shared_output = shared(input)
        outputs = [branch(shared_output) for branch in branches]
    """

    __constants__ = ('shared', 'branches')

    shared: Module
    branches: torch.nn.ModuleList

    def __init__(self,
                 branches: Sequence[Module],
                 shared: Optional[Module] = None):
        """
        Construct a enw :class:`Branch` module.

        Args:
            branches: The branch sub-modules.
            shared: The shared module to apply before the branch sub-modules.
        """
        if shared is None:
            shared = torch.nn.Identity()

        super().__init__()
        self.branches = torch.nn.ModuleList(list(branches))
        self.shared = shared

    def _forward(self, input: Tensor) -> List[Tensor]:
        outputs: List[Tensor] = []
        shared_output = self.shared(input)
        for branch in self.branches:
            outputs.append(branch(shared_output))
        return outputs


def check_coupling_layer(ctx,
                         spatial_ndims: int,
                         num_features: int,
                         cls,
                         shift_and_pre_scale_factory):
    batch_shape = [11]
    sigmoid_scale_bias = 1.5

    n1, n2 = (num_features // 2), (num_features - num_features // 2)
    shift_and_pre_scale_1 = shift_and_pre_scale_factory(n1, n2)
    shift_and_pre_scale_2 = shift_and_pre_scale_factory(n2, n1)

    def do_check(secondary, scale_type):
        x = torch.randn(make_conv_shape(
            batch_shape, num_features, [6, 7, 8][:spatial_ndims]))
        n1, n2 = (num_features // 2), (num_features - num_features // 2)

        # construct the instance
        shift_and_pre_scale = (shift_and_pre_scale_2
                               if secondary else shift_and_pre_scale_1)
        flow = cls(
            shift_and_pre_scale, scale=scale_type, secondary=secondary,
            sigmoid_scale_bias=sigmoid_scale_bias
        )
        assert(f'secondary={secondary}' in repr(flow))
        # flow = T.jit_compile(flow)

        # obtain the expected output
        channel_axis = get_channel_axis(spatial_ndims)
        x1, x2 = torch.split(x, [n1, n2], dim=channel_axis)
        if secondary:
            x1, x2 = x2, x1

        y1 = x1
        shift, pre_scale = shift_and_pre_scale(x1)
        if scale_type == 'exp' or scale_type is ExpScale:
            scale = ExpScale()
        elif scale_type == 'sigmoid' or scale_type is SigmoidScale:
            scale = SigmoidScale(pre_scale_bias=sigmoid_scale_bias)
        elif scale_type == 'linear' or scale_type is LinearScale:
            scale = LinearScale()
        elif isinstance(scale_type, BaseScale):
            scale = scale_type
        else:
            raise ValueError(f'Invalid value for `scale`: {scale_type}')
        y2, log_det = scale(x2 + shift, pre_scale,
                            event_ndims=spatial_ndims + 1,
                            compute_log_det=True)

        if secondary:
            y1, y2 = y2, y1
        expected_y = torch.cat([y1, y2], dim=channel_axis)
        expected_log_det = log_det

        # now check the flow
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            torch.randn(batch_shape))

    for secondary in (False, True):
        do_check(secondary, 'exp')

    for scale_type in ('exp', 'sigmoid', 'linear',
                       SigmoidScale, LinearScale()):
        do_check(False, scale_type)

    # test error constructors
    shift_and_pre_scale = shift_and_pre_scale_factory(2, 3)
    for scale in ('invalid', object(), dsl.Layers.Linear(2, 3)):
        with pytest.raises(ValueError,
                           match=r'`scale` must be a `BaseScale` class, '
                                 r'an instance of `BaseScale`, a factory to '
                                 r'construct a `BaseScale` instance, or one of '
                                 r'\{"exp", "sigmoid", "linear"\}'):
            _ = cls(shift_and_pre_scale, scale=scale)


class TestCouplingLayer(object):

    def test_CouplingLayer(self):
        def shift_and_pre_scale_factory(n1, n2):

            return Branch(
                [
                    dsl.Layers.Linear(10, n2),
                    dsl.Layers.Linear(10, n2),
                ],
                shared=dsl.Layers.Linear(n1, 10),
            )

        check_coupling_layer(
            self,
            spatial_ndims=0,
            num_features=5,
            cls=CouplingLayer,
            shift_and_pre_scale_factory=shift_and_pre_scale_factory,
        )

    def test_CouplingLayerNd(self):
        for spatial_ndims in (1, 2, 3):
            print('spatial_ndims', spatial_ndims)
            conv_cls = getattr(dsl.Layers, f'LinearConv{spatial_ndims}d')

            def shift_and_pre_scale_factory(n1, n2):
                return Branch(
                    [
                        conv_cls(10, n2, kernel_size=1),
                        conv_cls(10, n2, kernel_size=1),
                    ],
                    shared=conv_cls(n1, 10, kernel_size=1),
                )

            check_coupling_layer(
                self,
                spatial_ndims=spatial_ndims,
                num_features=5,
                cls=getattr(dsl.Flow, f'CouplingLayer{spatial_ndims}d'),
                shift_and_pre_scale_factory=shift_and_pre_scale_factory,
            )
