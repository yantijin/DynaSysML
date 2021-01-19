from DynaSysML.typing_ import *
from .base import *
from typing import *

__all__ = [
    'IgnoreContext', 'AddContext', 'MultiplyContext'
]

class IgnoreContext(BaseLayer):
    """
    A module which simply returns the input, ignoring any context.
    """

    def forward(self,
                input: Tensor,
                context: Optional[List[Tensor]] = None) -> Tensor:
        return input


class AddContext(BaseLayer):
    """
    A module which adds the input with the contexts.
    """

    def forward(self,
                input: Tensor,
                context: Optional[List[Tensor]] = None) -> Tensor:
        if context is not None:
            for t in context:
                input = input + t
        return input


class MultiplyContext(BaseLayer):
    """
    A module which multiplies the input with the contexts.
    """

    def forward(self,
                input: Tensor,
                context: Optional[List[Tensor]] = None) -> Tensor:
        if context is not None:
            for t in context:
                input = input * t
        return input
