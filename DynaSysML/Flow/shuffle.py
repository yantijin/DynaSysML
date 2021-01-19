import torch
from typing import *
from DynaSysML.typing_ import *
from DynaSysML.Flow import FeatureMappingFlow
from DynaSysML.core import add_parameter, index_select


__all__ = [
    'FeatureShufflingFlow', 'FeatureShufflingFlow1d', 'FeatureShufflingFlow2d',
    'FeatureShufflingFlow3d',
]


class FeatureShufflingFlow(FeatureMappingFlow):
    """
    An invertible flow which shuffles the order of input features.

    This type of flow is proposed in (Kingma & Dhariwal, 2018), as a possible
    replacement to the alternating pattern of coupling layers proposed in
    (Dinh et al., 2016).
    """

    __constants__ = FeatureMappingFlow.__constants__ + (
        'num_features',
    )

    num_features: int

    def __init__(self,
                 num_features: int,
                 axis: int = -1,
                 event_ndims: int = 1):
        """
        Construct a new :class:`FeatureShufflingFlow`.

        Args:
            num_features: The size of the feature axis.
            axis: The feature axis, to apply the transformation.
            event_ndims: Number of dimensions to be considered as the
                event dimensions.  `x.ndims - event_ndims == log_det.ndims`.
        """
        super().__init__(axis=int(axis), event_ndims=event_ndims,
                         explicitly_invertible=True)
        self.num_features = num_features

        # initialize the permutation variable, and the inverse permutation
        permutation = torch.randperm(num_features, dtype=torch.int64)
        inv_permutation = torch.argsort(permutation)

        # register the permutation as layer parameter, such that it could be
        # persisted by Model checkpoint.
        add_parameter(self, 'permutation', permutation, requires_grad=False)
        add_parameter(self, 'inv_permutation', inv_permutation,
                      requires_grad=False)

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = index_select(input, self.inv_permutation, axis=self.axis)
        else:
            output = index_select(input, self.permutation, axis=self.axis)
        output_log_det = input_log_det
        if compute_log_det and output_log_det is None:
            output_log_det = torch.as_tensor(0., dtype=input.dtype)
        return output, output_log_det


class FeatureShufflingFlow1d(FeatureShufflingFlow):
    """1D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        super().__init__(num_features, axis=-2, event_ndims=2)


class FeatureShufflingFlow2d(FeatureShufflingFlow):
    """2D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        super().__init__(num_features, axis=-3, event_ndims=3)


class FeatureShufflingFlow3d(FeatureShufflingFlow):
    """3D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        super().__init__(num_features, axis=-4, event_ndims=4)
