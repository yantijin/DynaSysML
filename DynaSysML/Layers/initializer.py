import math
from typing import *
from DynaSysML.typing_ import *
import torch
import numpy as np
from DynaSysML import core
from torch.utils.hooks import RemovableHandle
from torch.jit import ScriptModule


__all__ = [
    # utilities
    'calculate_fan_in_and_fan_out', 'get_activation_gain', 'apply_initializer',

    # data-independent tensor initializers
    'zeros', 'ones', 'fill', 'uniform', 'normal',
    'xavier_uniform', 'xavier_normal',
    'kaming_uniform', 'kaming_normal',

    # data-dependent layer initializers
    'DataDependentInitializer', 'set_initialized',
    'remove_data_dependent_initializers',
]

LEAKY_RELU_DEFAULT_SLOPE = 0.01
# ---- utilities ----
def calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    r = tensor.dim()
    if r < 2:
        raise ValueError('`fan_in` and `fan_out` cannot be calculated '
                         'when `rank(tensor)` < 2.')

    n_output_feature_maps = tensor.shape[0]
    n_input_feature_maps = tensor.shape[1]
    if r > 2:
        receptive_field = tensor[0][0].numel()
        n_input_feature_maps *= receptive_field
        n_output_feature_maps *= receptive_field

    return n_input_feature_maps, n_output_feature_maps


def _leaky_relu_activation_gain(negative_slope=LEAKY_RELU_DEFAULT_SLOPE):
    return math.sqrt(2. / (1. + negative_slope ** 2))


_relu_activation_gain = math.sqrt(2)
_tanh_activation_gain = 5. / 3


def get_activation_gain(activation: Optional[Union[Type[Module], Module]],
                        *args, **kwargs) -> float:
    """
    Get the preferred initialization gain for specified activation.

    Args:
        activation: The class or instance of the activation module.
        *args, \\**kwargs: Arguments to construct the activation module,
            if the specified `activation` is given as its class.
            Ignored if `activation` is already a module.

    Returns:
        The initialization gain.  If `activation` is not recognized, returns 1.
    """
    from .activation import ReLU, LeakyReLU, Tanh

    if activation is not None:
        if isinstance(activation, type):
            if issubclass(activation, ReLU):
                return _relu_activation_gain
            if issubclass(activation, LeakyReLU):
                return _leaky_relu_activation_gain(*args, **kwargs)
            if issubclass(activation, Tanh):
                return _tanh_activation_gain
        else:
            if isinstance(activation, ReLU):
                return _relu_activation_gain
            if isinstance(activation, LeakyReLU):
                return _leaky_relu_activation_gain(activation.negative_slope)
            if isinstance(activation, Tanh):
                return _tanh_activation_gain
    return 1.0


def apply_initializer(tensor: Tensor,
                      initializer: Optional[
                          Union[
                              int, float, np.ndarray, Tensor,
                              Callable[..., None]
                          ]
                      ],
                      gain: Optional[float] = None,
                      activation: Optional[
                          Union[str, Type[Module], Module, Any]
                      ] = None,
                      fan_in_and_fan_out: Optional[Tuple[int, int]] = None,
                      mode: str = 'fan_in'
                      ) -> None:
    """
    Apply an `initializer` on the specified `tensor`.

    Args:
        tensor: The tensor to be initialized.
        initializer: The initializer, may be one of:
            *   A scalar, which will be filled into `tensor`.
            *   A NumPy array or another `Tensor`, whose value will be copied
                to the `tensor`.
            *   A callable function ``(t: Tensor, \\**kwargs) -> None``.
                The `\\**kwargs` must present in order to consume all
                named arguments passed to the initializer.  Currently
                possible named arguments are: `gain`, `fan_in_and_fan_out`,
                and `mode`.
        gain: The gain of the activation.  If not specified, will calculate
            according to `activation` via :func:`get_activation_gain()`.
        activation: The activation of the layer.
        fan_in_and_fan_out: A tuple of ``(fan_in, fan_out)`` of the layer.
            If not specified, and if `rank(tensor)` >= 2, it will be computed
            via :func:`calculate_fan_in_and_fan_out()`.
        mode: Either "fan_in" or "fan_out".  If it is "fan_out", then the
            specified or calculated `fan_in` will be regarded as `fan_out`,
            and `fan_out` regarded as `fan_in`.
    """
    if gain is None:
        gain = get_activation_gain(activation)

    r = tensor.dim()
    if fan_in_and_fan_out is None and r > 1:
        fan_in_and_fan_out = calculate_fan_in_and_fan_out(tensor)
    kwargs = ({} if fan_in_and_fan_out is None
              else {'fan_in_and_fan_out': fan_in_and_fan_out})

    is_scalar = (
            (
                    not isinstance(initializer, bool) and
                    isinstance(initializer, (int, float))
            ) or (
                    isinstance(initializer, np.ndarray) and
                    np.shape(initializer) == ()
            )
    )
    if is_scalar:
        fill(tensor, initializer)
    elif isinstance(initializer, (np.ndarray, Tensor)):
        core.assign_data(tensor, initializer)
    elif callable(initializer):
        with torch.no_grad():
            initializer(tensor, gain=gain, mode=mode, **kwargs)
    else:
        raise TypeError(f'Unsupported initializer: {initializer!r}')


# ---- data-independent tensor initializers ----
# NOTE: all initializer functions must have `**kwargs` in its arguments, to
#       consume all arguments passed from :class:`LayerInit`.  The arguments
#       are listed as follows:
#
#       1.  gain: float
#       2.  fan_in_and_fan_out: Tuple[int, int]
#       3.  mode: str  # either one of {"fan_in", "fan_out"}
_no_grad_uniform_init = torch.nn.init.uniform_
_no_grad_normal_init = torch.nn.init.normal_


def _validate_fan_in_and_fan_out(tensor: Tensor,
                                 fan_in_and_fan_out: Optional[Tuple[int, int]]
                                 ) -> Tuple[int, int]:
    if fan_in_and_fan_out is None:
        fan_in_and_fan_out = calculate_fan_in_and_fan_out(tensor)
    return fan_in_and_fan_out


def _calculate_fan(tensor: Tensor,
                   fan_in_and_fan_out: Optional[Tuple[int, int]],
                   mode: str) -> int:
    if mode not in ('fan_in', 'fan_out'):
        raise ValueError(f'`mode` must be either "fan_in" or "fan_out": '
                         f'got {mode!r}')
    fan_in, fan_out = _validate_fan_in_and_fan_out(tensor, fan_in_and_fan_out)
    return fan_in if mode == 'fan_in' else fan_out


def zeros(tensor: Tensor, **kwargs):
    with torch.no_grad():
        core.fill_zeros(tensor)


def ones(tensor: Tensor, **kwargs):
    with torch.no_grad():
        core.fill(tensor, fill_value=1.)


def fill(tensor: Tensor, fill_value: Union[int, float, np.ndarray], **kwargs):
    with torch.no_grad():
        core.fill(tensor, fill_value=float(fill_value))


def uniform(tensor: Tensor, low: float = 0., high: float = 1.,
            **kwargs):
    _no_grad_uniform_init(tensor, low, high)


def normal(tensor: Tensor, mean: float = 0., std: float = 1.,
           **kwargs):
    _no_grad_normal_init(tensor, mean=mean, std=std)


def xavier_uniform(tensor: Tensor,
                   gain: float = 1.,
                   fan_in_and_fan_out: Optional[Tuple[int, int]] = None,
                   **kwargs):
    fan_in, fan_out = _validate_fan_in_and_fan_out(tensor, fan_in_and_fan_out)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # such that U(-a, a) will have standard deviation `std`

    _no_grad_uniform_init(tensor, -a, a)


def xavier_normal(tensor: Tensor,
                  gain: float = 1.,
                  fan_in_and_fan_out: Optional[Tuple[int, int]] = None,
                  **kwargs):
    fan_in, fan_out = _validate_fan_in_and_fan_out(tensor, fan_in_and_fan_out)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    _no_grad_normal_init(tensor, 0., std)


def kaming_uniform(tensor: Tensor,
                   gain: float = 1.,
                   fan_in_and_fan_out: Optional[Tuple[int, int]] = None,
                   mode: str = 'fan_in',
                   **kwargs):
    fan = _calculate_fan(tensor, fan_in_and_fan_out, mode)
    std = gain / math.sqrt(fan)
    a = math.sqrt(3.0) * std  # such that U(-a, a) will have standard deviation `std`

    _no_grad_uniform_init(tensor, -a, a)


def kaming_normal(tensor: Tensor,
                  gain: float = 1.,
                  fan_in_and_fan_out: Optional[Tuple[int, int]] = None,
                  mode: str = 'fan_in',
                  **kwargs):
    fan = _calculate_fan(tensor, fan_in_and_fan_out, mode)
    std = gain / math.sqrt(fan)

    _no_grad_normal_init(tensor, mean=0., std=std)


# # ---- data-dependent layer initializers ----
class DataDependentInitializer(object):
    """
    Base class for data-dependent initializers.

    A :class:`DataDependentInitializer` initializes the `weight` and `bias` of
    layers according to their inputs.  :class:`DataDependentInitializer` are
    generally stateless, and can be shared among layers.
    """

    def register(self, layer: Module, initialized: bool = False) -> None:
        """
        Register this data-dependent initializer to the specified `layer`.

        Args:
            layer: The layer to be initialized by this initializer.
            initialized: The initial `initialized` flag of the hook.
                Defaults to :obj:`False`.
        """
        _ = DataDependentInitializerForwardPreHook(
            self, layer, initialized=initialized)

    def _forward(self, layer: Module, inputs: List[Tensor]) -> None:
        raise NotImplementedError()

    def __call__(self, layer: Module, inputs: List[Tensor]) -> None:
        self._forward(layer, list(inputs))

    def __repr__(self) -> str:
        buf = []
        for attr in getattr(self, '__annotations__', ()):
            attr_val = getattr(self, attr, None)
            buf.append(f'{attr}={attr_val!r}')
        return f'{self.__class__.__qualname__}({", ".join(buf)})'


class DataDependentInitializerForwardPreHook(object):

    initializer: DataDependentInitializer
    hook_handle: RemovableHandle
    initialized: bool
    is_calling: bool  # whether or not the initializer is being called

    def __init__(self,
                 layer_init: DataDependentInitializer,
                 layer: Module,
                 initialized: bool = False):
        super().__init__()
        self.initializer = layer_init
        self.hook_handle = layer.register_forward_pre_hook(self)
        self.initialized = initialized
        self.is_calling = False

    def __call__(self, layer: Module, inputs: List[Tensor]):
        if not self.initialized:
            if not self.is_calling:
                self.is_calling = True
                try:
                    self.initializer(layer, inputs)
                    self.initialized = True
                finally:
                    self.is_calling = False

    def set_initialized(self, initialized: bool = True):
        self.initialized = initialized


def set_initialized(root: Module, initialized: bool = True) -> None:
    """
    Call `set_initialized` on `root` and all its children layers (recursively),
    as well as their every data-dependent initializer hook.

    Args:
        root: The root layer.
        initialized: The value of the `initialized` flag.
            If :obj:`True`, the data-dependent initializers will be disabled.
            If :obj:`False`, the data-dependent initializes will be enabled
            for the next forward call.
    """
    def set_initialized(layer: Module):
        if hasattr(layer, 'set_initialized'):
            getattr(layer, 'set_initialized')(initialized)
        if not isinstance(layer, ScriptModule):
            for key, hook in layer._forward_pre_hooks.items():
                if isinstance(hook, DataDependentInitializerForwardPreHook):
                    hook.set_initialized(initialized)
    root.apply(set_initialized)


def remove_data_dependent_initializers(root: Module) -> None:
    """
    Remove all data-dependent initializer hooks from the `root` module and all
    its children (recursively).

    Args:
        root: The root module.
    """
    def remove(layer: Module):
        handles_to_remove = []
        for itm, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, DataDependentInitializerForwardPreHook):
                handles_to_remove.append(hook.hook_handle)
        for handle in handles_to_remove:
            handle.remove()
    root.apply(remove)
