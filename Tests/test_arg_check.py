import pytest
from DynaSysML.arg_check import *
from DynaSysML.typing_ import PaddingMode

class TestConvUtils(object):
    def test_validate_conv_size(self):
        spatial_ndims = 2
        name = 'test1'
        value1 = [1,1]
        value2 = 2
        res1 = validate_conv_size(name, value1, spatial_ndims)
        res2 = validate_conv_size(name, value2, spatial_ndims)
        assert res1 == [1, 1]
        assert res2 == [2, 2]

    def test_validate_padding(self):
        padding1 = 1
        padding2 = 'full'
        padding3 = 'half'
        padding4 = 'none'
        padding5 = [1, 1]
        padding6 = PaddingMode.NONE
        dilation = [1, 1]
        spatial_ndims = 2
        kernel_size = [5,5]
        res1 = validate_padding(padding1, kernel_size, dilation, spatial_ndims)
        res2 = validate_padding(padding2, kernel_size, dilation, spatial_ndims)
        res3 = validate_padding(padding3, kernel_size, dilation, spatial_ndims)
        res4 = validate_padding(padding4, kernel_size, dilation, spatial_ndims)
        res5 = validate_padding(padding5, kernel_size, dilation, spatial_ndims)
        res6 = validate_padding(padding6, kernel_size, dilation, spatial_ndims)
        assert res1 == [(1,1),(1,1)]
        assert res4 == [(0,0),(0,0)]
        assert res2 == [(4, 4), (4, 4)]
        assert res3 == [(2, 2), (2, 2)]
        assert res5 == [(1,1), (1,1)]
        assert res6 == res4

    def test_maybe_as_symmetric_padding(self):
        a = [(1,1),(1,1)]
        b =[(1,2),(1,2)]
        res1 = maybe_as_symmetric_padding(a)
        res2 = maybe_as_symmetric_padding(b)
        assert res1 == [1, 1]
        assert res2 == None

    def test_validate_output_padding(self):
        stride = [3, 3]
        dilation = [1, 1]
        spatial_ndims = 2
        padding1 = 2
        padding2 = [2, 2]
        res1 = validate_output_padding(padding1, stride, dilation, spatial_ndims)
        res2 = validate_output_padding(padding2, stride, dilation, spatial_ndims)
        assert res1 == [2, 2]
        assert res2 == [2, 2]




