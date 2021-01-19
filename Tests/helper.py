import os

import numpy as np
import pytest
import torch
from DynaSysML.typing_ import *
# from tensorkit import tensor as T
# from tensorkit import *

__all__ = [
    'int_dtypes', 'float_dtypes', 'number_dtypes',
    'n_samples',

    # 'assert_allclose', 'assert_not_equal', 'assert_equal',
    #
    # 'slow_test',

    # 'check_distribution_instance',
    'flow_standard_check', 'noninvert_flow_standard_check'
]


# Not all integer or float dtypes are listed as follows.  Just some commonly
# used dtypes, enough for test.
int_dtypes = (torch.int32, torch.int64)
float_dtypes = (torch.float32, torch.float64)
number_dtypes = int_dtypes + float_dtypes

# The number of samples to take for tests which requires random samples.
n_samples = 10000


# def wrap_numpy_testing_assertion_fn(fn):
#     def wrapper(x, y, **kwargs):
#         if isinstance(x, (Tensor, StochasticTensor)):
#             x = T.to_numpy(T.as_tensor(x))
#         if isinstance(y, (T.Tensor, StochasticTensor)):
#             y = T.to_numpy(T.as_tensor(y))
#         return fn(x, y, **kwargs)
#     return wrapper


# assert_allclose = wrap_numpy_testing_assertion_fn(np.testing.assert_allclose)
#
#
# @wrap_numpy_testing_assertion_fn
# def assert_not_equal(x, y, err_msg=''):
#     if np.all(np.equal(x, y)):
#         msg = f'`x != y` not hold'
#         if err_msg:
#             msg += f': {err_msg}'
#         msg += f'\nx = {x}\ny = {y}'
#         raise AssertionError(msg)
#
#
# assert_equal = wrap_numpy_testing_assertion_fn(np.testing.assert_equal)
#
#
# decorate a test that is slow
# def slow_test(fn):
#     fn = pytest.mark.skipif(
#         os.environ.get('FAST_TEST', '0').lower() in ('1', 'on', 'yes', 'true'),
#         reason=f'slow test: {fn}'
#     )(fn)
#     return fn


# def check_distribution_instance(ctx,
#                                 d,
#                                 event_ndims,
#                                 batch_shape,
#                                 min_event_ndims,
#                                 max_event_ndims,
#                                 log_prob_fn,
#                                 transform_origin_distribution=None,
#                                 transform_origin_group_ndims=None,
#                                 **expected_attrs):
#     ctx.assertLessEqual(max_event_ndims - event_ndims, d.batch_ndims)
#
#     event_shape = expected_attrs.get('event_shape', None)
#     ctx.assertEqual(d.min_event_ndims, min_event_ndims)
#     ctx.assertEqual(d.value_ndims, len(batch_shape) + event_ndims)
#     if event_shape is not None:
#         ctx.assertEqual(d.value_shape, batch_shape + event_shape)
#     ctx.assertEqual(d.batch_shape, batch_shape)
#     ctx.assertEqual(d.batch_ndims, len(batch_shape))
#     ctx.assertEqual(d.event_ndims, event_ndims)
#     ctx.assertEqual(d.event_shape, event_shape)
#
#     for attr, val in expected_attrs.items():
#         ctx.assertEqual(getattr(d, attr), val)
#     ctx.assertEqual(
#         d.validate_tensors,
#         expected_attrs.get('validate_tensors', False)
#     )
#
    # check sample
    # for n_samples in (None, 5):
    #     for group_ndims in (None, 0,
    #                         -(event_ndims - min_event_ndims),
    #                         max_event_ndims - event_ndims):
    #         for reparameterized2 in (None, True, False):
    #             if reparameterized2 and not d.reparameterized:
    #                 continue
    #
    #             sample()
    #             sample_kwargs = {}
    #             if n_samples is not None:
    #                 sample_kwargs['n_samples'] = n_samples
    #                 sample_shape = [n_samples]
    #             else:
    #                 sample_shape = []
    #
    #             if group_ndims is not None:
    #                 sample_kwargs['group_ndims'] = group_ndims
    #             else:
    #                 group_ndims = 0
    #
    #             if reparameterized2 is not None:
    #                 sample_kwargs['reparameterized'] = reparameterized2
    #             else:
    #                 reparameterized2 = d.reparameterized
    #
    #             t = d.sample(**sample_kwargs)
    #             ctx.assertEqual(t.group_ndims, group_ndims)
    #             ctx.assertEqual(t.reparameterized, reparameterized2)
    #             ctx.assertEqual(
    #                 T.rank(t.tensor),
    #                 d.value_ndims + len(sample_shape))
    #             ctx.assertEqual(
    #                 T.shape(t.tensor)[:(d.batch_ndims +
    #                                     len(sample_shape))],
    #                 sample_shape + d.batch_shape
    #             )
    #
    #             if transform_origin_distribution is not None:
    #                 ctx.assertIsInstance(t.transform_origin, StochasticTensor)
    #                 ctx.assertIs(
    #                     t.transform_origin.distribution,
    #                     transform_origin_distribution
    #                 )
    #                 ctx.assertIs(
    #                     t.transform_origin.group_ndims,
    #                     transform_origin_group_ndims
    #                 )
    #
                # log_prob()
                # expected_log_prob = log_prob_fn(t)
                # for group_ndims2 in (None, 0,
                #                      -(event_ndims - min_event_ndims),
                #                      max_event_ndims - event_ndims):
                #     if group_ndims2 is not None:
                #         log_prob_kwargs = {'group_ndims': group_ndims2}
                #     else:
                #         log_prob_kwargs = {}
                #         group_ndims2 = group_ndims
                #
                #     log_prob = t.log_prob(**log_prob_kwargs)
                #     ctx.assertEqual(
                #         T.shape(log_prob),
                #         T.shape(t.tensor)[: T.rank(t.tensor) - (group_ndims2 + event_ndims)]
                #     )

                    # assert_allclose(
                    #     log_prob,
                    #     T.reduce_sum(
                    #         expected_log_prob,
                    #         T.int_range(-(group_ndims2 + (event_ndims - min_event_ndims)), 0)
                    #     ),
                    #     rtol=1e-4, atol=1e-6,
                    # )

                    # prob = t.prob(**log_prob_kwargs)
                    # assert_allclose(prob, T.exp(log_prob), rtol=1e-4, atol=1e-6)
                    #
                    # if transform_origin_distribution is not None:
                    #     for p in (log_prob, prob):
                    #         ctx.assertIsInstance(p.transform_origin,
                    #                              StochasticTensor)
                    #         ctx.assertIs(
                    #             p.transform_origin.distribution,
                    #             transform_origin_distribution
                    #         )
                    #         ctx.assertIs(
                    #             p.transform_origin.group_ndims,
                    #             transform_origin_group_ndims
                    #         )


def flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                        input_log_det):
    # test call
    y, log_det = flow(x)
    assert(y - expected_y < 1e-6).all()
    assert(log_det-expected_log_det < 1e-6).all()

    y, log_det = flow(x, input_log_det)
    assert(y -expected_y<1e-6).all()
    assert(log_det -input_log_det - expected_log_det<1e-6).all()

    y, log_det = flow(x, compute_log_det=False)
    assert (y - expected_y <1e-6).all()
    assert (log_det == None)

    # test call inverse
    y = expected_y
    expected_x = x
    expected_log_det = -expected_log_det

    x, log_det = flow(y, inverse=True)
    print('1',torch.max(x - expected_x))
    assert (x- expected_x<1e-4).all()
    assert(log_det -expected_log_det<1e-6).all()

    x, log_det = flow(y, input_log_det, inverse=True)
    assert (x -expected_x < 1e-6).all()
    assert (log_det-input_log_det - expected_log_det<1e-4).all()

    x, log_det = flow(y, inverse=True, compute_log_det=False)
    print(torch.max(x-expected_x))
    assert(x-expected_x<1e-4).all()
    assert(log_det == None)

def noninvert_flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                                  input_log_det):
    # test call
    y, log_det = flow(x)
    assert (list(y.shape) == list(x.shape))
    assert (y - expected_y < 1e-6).all()
    assert (log_det - expected_log_det < 1e-3).all()

    y, log_det = flow(x, input_log_det)
    assert (y - expected_y < 1e-6).all()
    print(log_det - input_log_det - expected_log_det)
    assert (log_det - input_log_det - expected_log_det < 1e-3).all()

    y, log_det = flow(x, compute_log_det=False)
    assert (y - expected_y < 1e-6).all()
    assert (log_det == None)