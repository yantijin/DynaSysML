import pytest
import DynaSysML as dsl
import numpy as np
from DynaSysML.Flow import *
from DynaSysML.typing_ import *
from typing import *
import torch
import math
from Tests.helper import flow_standard_check

def reshape_tail(input: Tensor, ndims: int, shape: List[int]) -> Tensor:
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < ndims:
        raise ValueError(
            '`input` must be at least `ndims`-dimensional: '
            '`shape(input)` is {}, while `ndims` is {}'.
            format(input_shape, ndims)
        )
    left_shape = input_shape[: input_rank - ndims]
    return input.reshape(left_shape + shape)

class _MyFlow(BaseFlow):

    def __init__(self):
        super().__init__(x_event_ndims=1,
                         y_event_ndims=2,
                         explicitly_invertible=True)

    def _forward(self,
              input: Tensor,
              input_log_det: Optional[Tensor],
              inverse: bool,
              compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = reshape_tail(0.5 * (input - 1.), 2, [-1])
        else:
            output = reshape_tail(input * 2. + 1., 1, [-1, 1])

        output_log_det = input_log_det
        if compute_log_det:
            log_2 = torch.as_tensor(math.log(2.), dtype=output.dtype)
            if output_log_det is None:
                if inverse:
                    output_log_det = -log_2 * input.shape[-2]
                else:
                    output_log_det = log_2 * input.shape[-1]
            else:
                if inverse:
                    output_log_det = output_log_det - log_2 * input.shape[-2]
                else:
                    output_log_det = output_log_det + log_2 * input.shape[-1]
        return output, output_log_det


class _MyBadFlow(BaseFlow):

    def __init__(self):
        super().__init__(x_event_ndims=1,
                         y_event_ndims=1,
                         explicitly_invertible=True)

    def _forward(self,
              input: Tensor,
              input_log_det: Optional[Tensor],
              inverse: bool,
              compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        output = input
        output_log_det = input_log_det
        if compute_log_det:
            if output_log_det is None:
                output_log_det = torch.zeros(output.shape)
            else:
                output_log_det = input_log_det
        return output, output_log_det


class TestaBaseFlow(object):

    def test_constructor(self):
        flow = BaseFlow(x_event_ndims=1,
                        y_event_ndims=2,
                        explicitly_invertible=True)
        assert(flow.x_event_ndims ==1)
        assert(flow.y_event_ndims ==2)
        assert(flow.explicitly_invertible ==True)

        flow = BaseFlow(x_event_ndims=3,
                        y_event_ndims=1,
                        explicitly_invertible=False)
        assert(flow.x_event_ndims ==3)
        assert(flow.y_event_ndims ==1)
        assert(flow.explicitly_invertible ==False)

    def test_invert(self):
        flow = _MyFlow()
        inv_flow = flow.invert()
        assert isinstance(inv_flow, InverseFlow)

    def test_call(self):
        # flow = T.jit_compile(_MyFlow())
        flow = _MyFlow()
        assert(flow.x_event_ndims ==1)
        assert(flow.y_event_ndims ==2)
        assert(flow.explicitly_invertible ==True)

        # test call
        x = torch.randn([2, 3, 4])
        expected_y = torch.reshape(x * 2. + 1., [2, 3, 4, 1])
        expected_log_det = torch.full([2, 3], math.log(2.) * 4)
        input_log_det = torch.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        # test input shape error
        with pytest.raises(Exception,
                           match='`input` is required to be at least .*d'):
            _ = flow(torch.randn([]))
        with pytest.raises(Exception,
                           match='`input` is required to be at least .*d'):
            _ = flow(torch.randn([3]), inverse=True)

        # test input_log_det shape error
        with pytest.raises(Exception,
                           match='The shape of `input_log_det` is not expected'):
            _ = flow(x, torch.randn([2, 4]))
        with pytest.raises(Exception,
                           match='The shape of `input_log_det` is not expected'):
            _ = flow(expected_y, torch.randn([2, 4]), inverse=True)

        # test output_log_det shape error
        # flow = T.jit_compile(_MyBadFlow())
        flow = _MyBadFlow()
        with pytest.raises(Exception,
                           match='The shape of `output_log_det` is not expected'):
            _ = flow(x)
        with pytest.raises(Exception,
                           match='The shape of `output_log_det` is not expected'):
            _ = flow(x, inverse=True)


class TestFeatureMappingFlow(object):

    def test_constructor(self):
        flow = FeatureMappingFlow(axis=-1,
                                  event_ndims=2,
                                  explicitly_invertible=True)
        assert(flow.event_ndims ==2)

        assert(flow.axis ==-1)
        assert(flow.x_event_ndims ==2)
        assert(flow.y_event_ndims ==2)
        assert(flow.explicitly_invertible ==True)

        with pytest.raises(ValueError,
                           match='`event_ndims` must be at least 1'):
            _ = FeatureMappingFlow(axis=-1, event_ndims=0, explicitly_invertible=True)

        with pytest.raises(ValueError,
                           match='`-event_ndims <= axis < 0` does not hold'):
            _ = FeatureMappingFlow(axis=-2, event_ndims=1, explicitly_invertible=True)

        with pytest.raises(ValueError,
                           match='`-event_ndims <= axis < 0` does not hold'):
            _ = FeatureMappingFlow(axis=0, event_ndims=1, explicitly_invertible=True)


class TestInverseFlow(object):

    def test_InverseFlow(self):
        original_flow = _MyFlow()
        flow = InverseFlow(original_flow)
        assert(flow.original_flow is original_flow)
        assert(flow.invert() is original_flow)

        # flow = T.jit_compile(flow)
        assert(flow.x_event_ndims ==2)
        assert(flow.y_event_ndims ==1)
        assert(flow.explicitly_invertible ==True)

        x = torch.randn([2, 3, 4, 1])
        expected_y = torch.reshape((x - 1.) * 0.5, [2, 3, 4])
        expected_log_det = -torch.full([2, 3], math.log(2.) * 4)
        input_log_det = torch.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        with pytest.raises(TypeError,
                           match='`flow` must be an explicitly invertible flow'):
            _ = InverseFlow(dsl.Layers.Linear(5, 3))

        base_flow = _MyFlow()
        base_flow.explicitly_invertible = False
        with pytest.raises(TypeError,
                           match='`flow` must be an explicitly invertible flow'):
            _ = InverseFlow(base_flow)


class _MyFlow1(BaseFlow):

    def __init__(self):
        super().__init__(x_event_ndims=1, y_event_ndims=1,
                         explicitly_invertible=True)

    def _forward(self,
              input: Tensor,
              input_log_det: Optional[Tensor],
              inverse: bool,
              compute_log_det: bool
              ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = (input - 1.) * 0.5
        else:
            output = input * 2. + 1.

        output_log_det = input_log_det
        if compute_log_det:
            log_2 =  torch.as_tensor(math.log(2.), dtype=output.dtype)
            if output_log_det is None:
                if inverse:
                    output_log_det = -log_2 * input.shape[-1]
                else:
                    output_log_det = log_2 * input.shape[-1]
            else:
                if inverse:
                    output_log_det = output_log_det - log_2 * input.shape[-1]
                else:
                    output_log_det = output_log_det + log_2 * input.shape[-1]

        return output, output_log_det


class TestSequentialFlow(object):

    def test_constructor(self):
        flows = [_MyFlow1(), _MyFlow()]
        flow = SequentialFlow(flows)
        assert(flow.x_event_ndims==1)
        assert(flow.y_event_ndims==2)
        assert(flow.explicitly_invertible==True)

        flow2 = _MyFlow()
        flow2.explicitly_invertible = False
        flows = [_MyFlow1(), flow2]
        flow = SequentialFlow(flows)
        assert(flow.explicitly_invertible == False)

        with pytest.raises(ValueError,
                           match='`flows` must not be empty'):
            _ = SequentialFlow([])

        with pytest.raises(TypeError,
                           match=r'`flows\[0\]` is not a flow'):
            _ = SequentialFlow([dsl.Layers.Linear(5, 3), _MyFlow()])

        with pytest.raises(TypeError,
                           match=r'`flows\[1\]` is not a flow'):
            _ = SequentialFlow([_MyFlow(), dsl.layers.Linear(5, 3)])

        with pytest.raises(ValueError,
                           match=r'`x_event_ndims` of `flows\[1\]` != '
                                 r'`y_event_ndims` of `flows\[0\]`: '
                                 r'1 vs 2'):
            _ = SequentialFlow([_MyFlow(), _MyFlow()])

    def test_call(self):
        # test call and inverse call
        flows = [_MyFlow1(), _MyFlow1()]
        flow = SequentialFlow(flows)

        x = torch.randn([2, 3, 4])
        expected_y = (x * 2. + 1.) * 2. + 1.
        expected_log_det = torch.full([2, 3], math.log(2.) * 8)
        input_log_det = torch.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        # test no inverse call
        flows = [_MyFlow1()]
        flows[0].explicitly_invertible = False
        flow = SequentialFlow(flows)

        with pytest.raises(Exception,
                           match='Not an explicitly invertible flow'):
            _ = flow(x, inverse=True)




def check_invertible_matrix(ctx, m):
    matrix, log_det = m(inverse=False, compute_log_det=False)
    assert(log_det == None)

    matrix, log_det = m(inverse=False, compute_log_det=True)
    assert(list(matrix.shape) ==[m.size, m.size])
    assert(torch.inverse(torch.inverse(matrix)) -matrix<1e-6).all()
    assert(torch.slogdet(matrix)[1]-log_det<1e-6).all()

    inv_matrix, inv_log_det = m(inverse=True, compute_log_det=True)
    assert(list(inv_matrix.shape) == [m.size, m.size])
    assert(torch.inverse(inv_matrix) - matrix <1e-6).all()
    assert(torch.inverse(torch.inverse(inv_matrix))-inv_matrix<1e-6).all()
    assert (inv_log_det +log_det <1e-6).all()
    assert(torch.slogdet(inv_matrix)[1] +log_det <1e-6).all()


class TestInvertibleMatrix(object):

    def test_invertible_matrices(self):
        for cls in (LooseInvertibleMatrix, StrictInvertibleMatrix):
            for n in [1, 3, 5]:
                m = cls(np.random.randn(n, n))
                assert(repr(m) == f'{cls.__qualname__}(size={n})')

                # m = T.jit_compile(m)
                assert(m.size ==n)

                # check the initial value is an orthogonal matrix
                matrix, _ = m(inverse=False, compute_log_det=False)
                inv_matrix, _ = m(inverse=True, compute_log_det=False)
                assert (torch.eye(n) -torch.matmul(matrix, inv_matrix) <1e-6).all()
                assert (torch.eye(n)-torch.matmul(inv_matrix, matrix)<1e-6).all()

                # check the invertibility
                check_invertible_matrix(self, m)

                # check the gradient
                matrix, log_det = m(inverse=False, compute_log_det=True)
                params = [v for _, v in dsl.core.get_parameters(m)]
                grads = list(torch.autograd.grad([dsl.core.reduce_sum(matrix), dsl.core.reduce_sum(log_det)], params))

                # update with gradient, then check the invertibility
                if cls is StrictInvertibleMatrix:
                    for param, grad in zip(params, grads):
                        with torch.no_grad():
                            param.copy_(param + 0.001 * grad)
                    check_invertible_matrix(self, m)