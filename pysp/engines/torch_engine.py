"""Torch implementation of the ComputeEngine protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pysp.engines.base import ComputeEngine
from pysp.engines.precision import normalize_torch_dtype
from pysp.utils.optional_deps import require

try:
    import torch
except ImportError:  # pragma: no cover - exercised when optional extra is absent
    torch = None


class TorchEngine(ComputeEngine):
    """Torch tensor engine for device placement, autograd, and optional DTensor sharding."""

    name = "torch"
    supports_autograd = True

    def __init__(
        self,
        device: str | None = None,
        dtype: Any = None,
        compile: bool = False,
        mesh: Any = None,
        shard: str | None = None,
    ) -> None:
        if torch is None:
            require("torch", "torch")
        if shard not in (None, "components"):
            raise ValueError("TorchEngine shard must be None or 'components'.")
        self.device = torch.device(device or "cpu")
        self.dtype = normalize_torch_dtype(dtype, torch) if dtype is not None else torch.float64
        self.compile_enabled = bool(compile)
        self.mesh = mesh
        self.shard = shard

    def with_precision(self, precision: Any) -> TorchEngine:
        """Return a Torch engine with the same placement and a new dtype policy."""
        return TorchEngine(
            device=str(self.device), dtype=precision, compile=self.compile_enabled, mesh=self.mesh, shard=self.shard
        )

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert ``x`` to a Torch tensor or DTensor on the configured device."""
        if torch is None:
            require("torch", "torch")
        if self._is_dtensor(x):
            rv = x
            if dtype is not None:
                rv = rv.to(dtype=dtype)
            return rv
        if isinstance(x, torch.Tensor):
            rv = x.to(device=self.device, dtype=dtype or (self.dtype if x.dtype.is_floating_point else x.dtype))
            return self._replicate_tensor(rv)
        arr = np.asarray(x)
        if dtype is not None:
            dt = dtype
        elif arr.dtype.kind == "f":
            dt = self.dtype
        elif arr.dtype.kind == "b":
            dt = torch.bool
        else:
            dt = torch.int64
        return self._replicate_tensor(torch.as_tensor(arr, dtype=dt, device=self.device))

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate a zero tensor with this engine's dtype/device/placement."""
        return self._replicate_tensor(torch.zeros(shape, dtype=dtype or self.dtype, device=self.device))

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate an uninitialized tensor with this engine's dtype/device/placement."""
        return self._replicate_tensor(torch.empty(shape, dtype=dtype or self.dtype, device=self.device))

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return ``torch.arange`` on the configured device."""
        kwargs.setdefault("device", self.device)
        if "dtype" not in kwargs and any(isinstance(x, (float, np.floating)) for x in args):
            kwargs["dtype"] = self.dtype
        return self._replicate_tensor(torch.arange(*args, **kwargs))

    def to_numpy(self, x: Any) -> np.ndarray:
        """Gather/detach a Torch tensor or DTensor to a host NumPy array."""
        if self._is_dtensor(x):
            return x.full_tensor().detach().cpu().numpy()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack tensors with ``torch.stack`` and apply default placement."""
        return self._replicate_tensor(torch.stack(tuple(arrays), dim=axis))

    def replicate(self, x: Any) -> Any:
        """Return ``x`` replicated across the configured DeviceMesh."""
        if self.mesh is None:
            return self.asarray(x)
        if self._is_dtensor(x):
            return x.redistribute(device_mesh=self.mesh, placements=self._replicated_placements())
        if not isinstance(x, torch.Tensor):
            x = self.asarray(x)
            if self._is_dtensor(x):
                return x.redistribute(device_mesh=self.mesh, placements=self._replicated_placements())
            return x
        return self._replicate_tensor(x)

    def place_component_axis(self, x: Any, axis: int = 0) -> Any:
        """Place a stacked component parameter tensor on ``Shard(axis)``.

        ``TorchEngine(mesh=..., shard='components')`` is the model-parallel
        configuration from the compute-engine design.  Encoded data remains
        replicated, while homogeneous stacked-kernel parameter tensors are
        sharded along their component dimension.  With no mesh, or with no
        component sharding requested, this is just ``asarray``.
        """
        if self.mesh is None or self.shard != "components":
            return self.asarray(x)
        if self._is_dtensor(x):
            rv = x
        else:
            rv = x if isinstance(x, torch.Tensor) else self.asarray(x)
            if self._is_dtensor(rv):
                pass
            else:
                rv = rv.to(device=self.device, dtype=self.dtype if rv.dtype.is_floating_point else rv.dtype)
                shape = tuple(getattr(rv, "shape", ()))
                if len(shape) == 0:
                    return self._replicate_tensor(rv)
                axis = _normalize_axis(axis, len(shape))
                return self._distribute_tensor(rv, self._component_placements(axis))
        shape = tuple(getattr(rv, "shape", ()))
        if len(shape) == 0:
            return rv.redistribute(device_mesh=self.mesh, placements=self._replicated_placements())
        axis = _normalize_axis(axis, len(shape))
        return rv.redistribute(device_mesh=self.mesh, placements=self._component_placements(axis))

    def requires_grad(self, x: Any) -> bool:
        """Return whether ``x`` is a Torch tensor/DTensor requiring gradients."""
        return bool((isinstance(x, torch.Tensor) or self._is_dtensor(x)) and x.requires_grad)

    def compile(self, fn: Callable) -> Callable:
        """Compile ``fn`` with ``torch.compile`` when enabled and available."""
        if self.compile_enabled and hasattr(torch, "compile"):
            return torch.compile(fn)
        return fn

    log = staticmethod(lambda x: torch.log(x))
    exp = staticmethod(lambda x: torch.exp(x))
    sqrt = staticmethod(lambda x: torch.sqrt(x))
    abs = staticmethod(lambda x: torch.abs(x))
    where = staticmethod(lambda *args: torch.where(*args))
    maximum = staticmethod(lambda x, y: torch.maximum(x, y))
    clip = staticmethod(lambda x, a_min=None, a_max=None: torch.clamp(x, min=a_min, max=a_max))
    floor = staticmethod(lambda x: torch.floor(x))
    isnan = staticmethod(lambda x: torch.isnan(x))
    isinf = staticmethod(lambda x: torch.isinf(x))

    @staticmethod
    def sum(x, *args, **kwargs):
        """Return ``torch.sum`` accepting either ``axis`` or ``dim``."""
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return torch.sum(x, *args, **kwargs)

    @staticmethod
    def max(x, *args, **kwargs):
        """Return ``torch.max`` accepting either ``axis`` or ``dim``."""
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        rv = torch.max(x, *args, **kwargs)
        return rv.values if hasattr(rv, "values") else rv

    dot = staticmethod(lambda x, y: torch.dot(x, y))
    matmul = staticmethod(lambda x, y: torch.matmul(x, y))
    cumsum = staticmethod(lambda x, *args, **kwargs: torch.cumsum(x, *args, **kwargs))

    @staticmethod
    def logsumexp(x, *args, **kwargs):
        """Return ``torch.logsumexp`` accepting either ``axis`` or ``dim``."""
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return torch.logsumexp(x, *args, **kwargs)

    bincount = staticmethod(lambda x, *args, **kwargs: torch.bincount(x, *args, **kwargs))
    unique = staticmethod(lambda x, *args, **kwargs: torch.unique(x, *args, **kwargs))
    searchsorted = staticmethod(lambda x, y, *args, **kwargs: torch.searchsorted(x, y, *args, **kwargs))
    gammaln = staticmethod(lambda x: torch.special.gammaln(x))
    digamma = staticmethod(lambda x: torch.special.digamma(x))
    betaln = staticmethod(lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y))
    erf = staticmethod(lambda x: torch.erf(x))

    def index_add(self, out: Any, index: Any, values: Any) -> Any:
        """Add ``values`` into ``out`` along axis 0 using Torch ``index_add``."""
        # index must be a long tensor on out's device; coerce defensively so a
        # numpy index (or one on another device) does not raise a cryptic error
        index = torch.as_tensor(index, dtype=torch.long, device=out.device)
        return out.index_add(0, index, values)

    def _replicate_tensor(self, x: Any) -> Any:
        if self.mesh is None or self._is_dtensor(x):
            return x
        return self._distribute_tensor(x, self._replicated_placements())

    def _distribute_tensor(self, x: Any, placements: Any) -> Any:
        _, _, _, distribute_tensor = _dtensor_api()
        return distribute_tensor(x, self.mesh, placements=placements)

    def _replicated_placements(self) -> Any:
        _, _, Replicate, _ = _dtensor_api()
        return tuple(Replicate() for _ in range(_mesh_ndim(self.mesh)))

    def _component_placements(self, axis: int) -> Any:
        _, Shard, Replicate, _ = _dtensor_api()
        ndim = _mesh_ndim(self.mesh)
        placements = [Replicate() for _ in range(ndim)]
        placements[-1] = Shard(axis)
        return tuple(placements)

    @staticmethod
    def _is_dtensor(x: Any) -> bool:
        try:
            DTensor, _, _, _ = _dtensor_api()
        except ImportError:
            return False
        return isinstance(x, DTensor)


def _dtensor_api():
    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    except ImportError as e:  # pragma: no cover - depends on torch build
        raise ImportError("TorchEngine mesh placement requires torch.distributed.tensor.") from e
    return DTensor, Shard, Replicate, distribute_tensor


def _mesh_ndim(mesh: Any) -> int:
    ndim = getattr(mesh, "ndim", None)
    if ndim is not None:
        return int(ndim)
    shape = getattr(mesh, "shape", None)
    if shape is not None:
        return max(1, len(tuple(shape)))
    return 1


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("component axis %d is out of bounds for tensor rank %d." % (axis, ndim))
    return axis
