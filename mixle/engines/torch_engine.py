"""Torch implementation of the ``ComputeEngine`` protocol.

The engine handles tensor placement, dtype policy, autograd support, optional
compilation, and component sharding for resident scoring and estimation paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.engines.base import ComputeEngine
from mixle.engines.precision import normalize_torch_dtype
from mixle.utils.optional_deps import require

try:
    import torch
except ImportError:  # pragma: no cover - exercised when optional extra is absent
    torch = None


def _require_bool(value: Any, name: str) -> bool:
    """Require an actual Boolean for an execution-path feature flag (MXR-080-1563).

    ``bool(value)`` is the wrong gate for a flag that selects a *compilation or numerical path*:
    every non-empty string is truthy, so a configuration value that round-tripped through YAML/JSON/
    an environment variable as ``"false"`` or ``"0"`` used to switch the path ON -- the exact
    opposite of what it says. Only ``True``/``False`` (or their NumPy scalars) can mean what the flag
    means, so anything else is refused instead of being coerced into a plausible-looking answer.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__} ({value!r}).")
    return bool(value)


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
        compile = _require_bool(compile, "TorchEngine compile")
        if shard not in (None, "components"):
            raise ValueError("TorchEngine shard must be None or 'components'.")
        if mesh is not None and shard == "components" and not _dtensor_ops_supported():
            raise ValueError(
                f"TorchEngine DTensor component-sharding requires torch >= 2.5; this is torch "
                f"{getattr(torch, '__version__', '?')}, whose DTensor lacks sharding strategies for "
                "reductions/guards used in the mixture E-step (logsumexp, isinf, ...). Use "
                "backend='model_parallel' for engine-agnostic component-parallel EM (bit-identical, any "
                "torch), or upgrade torch."
            )
        self.device = torch.device(device or "cpu")
        # MPS (Apple-silicon GPU) has no float64. An unset/default dtype silently settles for MPS's
        # highest supported precision (float32) below -- a low-stakes accommodation, since the caller
        # never asked for a specific precision. An EXPLICIT float64 request is different: MPS genuinely
        # cannot honor it, and a caller relying on it for precision allocation or a scientific tolerance
        # must not receive a silently-downgraded policy with no infeasibility signal, so that case fails
        # closed instead (see the `dtype is not None` branch below).
        self._no_f64 = self.device.type == "mps"
        if isinstance(dtype, torch.dtype) and not dtype.is_floating_point:
            # A concrete non-floating torch dtype (e.g. torch.int64/torch.bool, as surfaced
            # by engine discovery reading an integer/Boolean tensor's own storage dtype) is
            # not a meaningful floating-point policy value -- treat it like "no override" so
            # engine construction/discovery succeeds instead of raising (this used to make
            # engine_of() raise ValueError for ordinary integer and Boolean tensors, breaking
            # indexing, masks, and categorical payload dispatch). A *named* precision that
            # resolves to non-floating (e.g. dtype="int64") is a genuine caller mistake and
            # still raises below, inside normalize_torch_dtype.
            dtype = None
        # Whether the floating-point policy below is a real, caller-requested opinion (True)
        # or just an implicit default because no floating dtype was given/applicable (False).
        # mixle.engines.engine_of's mixed-engine check reads this so an engine discovered from
        # a non-floating leaf (an integer index or Boolean mask tensor) never conflicts with a
        # genuinely floating sibling leaf purely because of this filled-in default.
        self.dtype_explicit = dtype is not None
        if dtype is not None:
            self.dtype = normalize_torch_dtype(dtype, torch)
            if self._no_f64 and self.dtype == torch.float64:
                raise ValueError(
                    f"TorchEngine device 'mps' has no float64 support, but dtype={dtype!r} explicitly "
                    "requested it. MPS's highest supported floating precision is float32, which cannot "
                    "meet a float64 accuracy requirement -- silently downgrading would let a caller "
                    "relying on float64 for precision allocation or a scientific tolerance receive a "
                    "policy that cannot honor it, with no signal the request went unmet. Pass "
                    "device='cpu' (or another float64-capable device), or explicitly accept the reduced "
                    "precision by passing dtype='float32' instead."
                )
        else:
            self.dtype = torch.float32 if self._no_f64 else torch.float64
        self.compile_enabled = compile
        self.mesh = mesh
        self.shard = shard

    @property
    def accumulator_dtype(self) -> Any:
        """High-precision dtype for sufficient-statistic reductions (float64, or float32 on MPS).

        Reductions that aggregate over observations accumulate in float64 even when scoring runs in
        reduced precision, so a float32 fit does not drift on large N. MPS has no float64, so there the
        accumulator falls back to float32 (its max precision) — fits on very large N may drift slightly.
        """
        return torch.float32 if self._no_f64 else torch.float64

    def with_precision(self, precision: Any) -> TorchEngine:
        """Return a Torch engine with the same placement and a new dtype policy."""
        return TorchEngine(
            device=str(self.device), dtype=precision, compile=self.compile_enabled, mesh=self.mesh, shard=self.shard
        )

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert ``x`` to a Torch tensor or DTensor on the configured device.

        Contract: float inputs are force-cast to the engine's float dtype (e.g. a
        float32 numpy array becomes the engine's float64) unless ``dtype`` is given;
        this differs from numpy's ``asarray``, which preserves the input dtype. Complex
        NumPy input is rejected when ``dtype`` is not given explicitly: the dtype
        inference below has no complex-precision policy of its own, and letting it fall
        through to the generic non-floating/non-Boolean case would silently discard the
        imaginary component and collapse the real part to an integer. Pass an explicit
        ``dtype=torch.complex64``/``torch.complex128`` for a genuine complex tensor.
        """
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
        elif arr.dtype.kind == "c":
            raise ValueError(
                f"TorchEngine.asarray received complex-valued input (NumPy dtype {arr.dtype}) with no "
                "explicit dtype= override. This engine's dtype policy has no complex-precision concept, "
                "so the generic non-floating/non-Boolean fallback (torch.int64) would silently discard "
                "the imaginary component and collapse the real part to an integer. Pass "
                "dtype=torch.complex64 or dtype=torch.complex128 explicitly for a genuine complex tensor."
            )
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
        """Stack tensors with ``torch.stack`` under the engine's dtype and device policy.

        Elements route through :meth:`asarray` first. Leaf scorers legitimately hand back float64
        NumPy arrays; stacking them raw produced a float64 tensor that later met float32 engine
        tensors -- invisible on CPU, but a hard "mat1 double, mat2 float" error inside the HMM
        forward pass on the first real CUDA execution (rented RTX 3060, 2026-08-04). The engine's
        dtype policy has to apply at every ingestion point, not only the explicit asarray calls.
        """
        return self._replicate_tensor(torch.stack(tuple(self.asarray(a) for a in arrays), dim=axis))

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        """Join tensors with ``torch.cat`` under the same ingestion policy as :meth:`stack`."""
        return self._replicate_tensor(torch.cat(tuple(self.asarray(a) for a in arrays), dim=axis))

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

    def sum(self, x, *args, **kwargs):
        """Return ``torch.sum`` accepting either ``axis`` or ``dim``.

        Promotes a floating input to :attr:`accumulator_dtype` when the caller doesn't pass an explicit
        ``dtype=`` -- mirroring :meth:`NumpyEngine.sum`. Without this, ``torch.sum`` accumulates in the
        input tensor's own dtype by default, so a float32-precision fit on this engine would silently
        drift on large N (the exact catastrophic-cancellation risk ``accumulator_dtype`` exists to
        guard against) while the numpy engine, which already promotes, stayed accurate.
        """
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        if kwargs.get("dtype") is None and torch.is_tensor(x) and x.dtype.is_floating_point:
            kwargs["dtype"] = self.accumulator_dtype
        return torch.sum(x, *args, **kwargs)

    @staticmethod
    def max(x, *args, **kwargs):
        """Return ``torch.max`` accepting either ``axis`` or ``dim`` (incl. a tuple of axes)."""
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        dim = kwargs.pop("dim", None)
        if isinstance(dim, (tuple, list)):
            # torch.max reduces a single dim; fold over a tuple of axes to match numpy. Every axis is
            # normalized against the ORIGINAL rank BEFORE sorting (MXR-080-1562): the fold drops a
            # dimension each pass, so a still-negative axis names a different dimension on the second
            # pass than numpy's contract says it does -- reducing a (2,3,4) tensor over (-1,-2) used to
            # sort to [-1,-2] and return shape (3,) where numpy returns (2,). Duplicates are rejected
            # exactly as numpy does rather than silently reducing two different dimensions.
            rv = x
            normalized: list[int] = []
            for one_dim in dim:
                axis = int(one_dim)
                if axis < 0:
                    axis += x.dim()
                if not 0 <= axis < x.dim():
                    raise IndexError(
                        "Dimension out of range (expected to be in range of [%d, %d], but got %d)"
                        % (-x.dim(), x.dim() - 1, int(one_dim))
                    )
                if axis in normalized:
                    raise ValueError("duplicate value in 'axis'")
                normalized.append(axis)
            for one_dim in sorted(normalized, reverse=True):
                rv = torch.max(rv, dim=one_dim, **kwargs)
                rv = rv.values if isinstance(rv, tuple) else rv
            return rv
        if dim is not None:
            kwargs["dim"] = dim
        rv = torch.max(x, *args, **kwargs)
        # torch.max(x) with no dim returns a plain Tensor, and Tensor.values is the SPARSE accessor
        # method — hasattr would hand that method back uncalled. Only the dim reduction returns the
        # (values, indices) named tuple, so unwrap on tuple-ness, never on attribute presence.
        return rv.values if isinstance(rv, tuple) else rv

    dot = staticmethod(lambda x, y: torch.dot(x, y))

    def matmul(self, x: Any, y: Any) -> Any:
        """Matrix-multiply under the engine's dtype policy.

        Both operands route through :meth:`asarray`. Mixed float32/float64 operands reached this
        point through paths that never crossed an explicit ingestion call, and torch.matmul
        promotes on CPU but hard-errors on CUDA ("mat1 double, mat2 float") -- found and probed on
        the first real CUDA execution of the HMM forward pass (rented RTX 3060, 2026-08-04:
        mat1 float64 [100, 2] against mat2 float32 [2, 2]). The engine's declared dtype IS the
        policy for floating operands, so applying it here is a correctness boundary, not a cast of
        convenience.
        """
        return torch.matmul(self.asarray(x), self.asarray(y))

    @staticmethod
    def cumsum(x, *args, **kwargs):
        """Return ``torch.cumsum`` accepting ``axis``/``dim`` and defaulting to a flattened scan."""
        if "axis" in kwargs and "dim" not in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        if not args and "dim" not in kwargs:
            # numpy's ``cumsum`` with no axis flattens; torch requires a dim.
            x = torch.reshape(x, (-1,))
            kwargs["dim"] = 0
        return torch.cumsum(x, *args, **kwargs)

    @staticmethod
    def logsumexp(x, *args, **kwargs):
        """Return ``torch.logsumexp`` accepting either ``axis`` or ``dim``.

        torch < 2.5 registers no DTensor sharding strategy for ``logsumexp``, so reducing over a
        *sharded* axis (the mixture E-step's log-partition over component-sharded scores) raises
        ``NotImplementedError``. Fall back to redistributing the DTensor to replicated first -- the
        reduction then runs locally and is correct on any torch version. torch >= 2.5 has the strategy
        and takes the fast native path unchanged; non-DTensor inputs re-raise the original error."""
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
    # optional trig tier (not in REQUIRED_OPS): directional families use these where the engine has them
    cos = staticmethod(lambda x: torch.cos(x))
    sin = staticmethod(lambda x: torch.sin(x))
    arctan2 = staticmethod(lambda x, y: torch.atan2(x, y))
    i0e = staticmethod(lambda x: torch.special.i0e(x))
    erfcx = staticmethod(lambda x: torch.special.erfcx(x))

    # --- fused HMM recurrences (the generic engine-op loop pays ~2.5x indirection overhead per step;
    # hidden_markov.hmm_engine_forward_backward/_ll delegate here when the engine provides these) ---
    def _hmm_inputs(self, log_emit: Any, log_w: Any, log_a: Any, mask: Any):
        le = self.asarray(log_emit)
        lw = self.asarray(log_w)
        la = self.asarray(log_a)
        m = self.asarray(mask)
        return le, lw, la, m

    def hmm_forward_ll(self, log_emit: Any, log_w: Any, log_a: Any, mask: Any) -> Any:
        """Fused log-space forward: per-sequence emission log-likelihood (freeze-alpha padding)."""
        le, lw, la, m = self._hmm_inputs(log_emit, log_w, log_a, mask)
        init = lw if lw.dim() == 2 else lw[None, :]
        alpha = init + le[:, 0, :]
        for t in range(1, le.shape[1]):
            cand = torch.logsumexp(alpha[:, :, None] + la[None, :, :], dim=1) + le[:, t, :]
            alpha = torch.where(m[:, t][:, None] > 0, cand, alpha)
        return torch.logsumexp(alpha, dim=1)

    def hmm_forward_backward(
        self, log_emit: Any, log_w: Any, log_a: Any, mask: Any, weights: Any = None
    ) -> tuple[Any, Any, Any, Any]:
        """Fused log-space Baum-Welch recurrences, exactly mirroring the generic engine-op version:
        alpha freezes across padded steps, beta carries, gamma is mask-SELECTED (NaN-safe for empty
        sequences), xi is computed for all transitions at once, and per-sequence ``weights`` scale
        gamma / xi / pi. Returns ``(ll, gamma, xi_sum, pi)``."""
        le, lw, la, m = self._hmm_inputs(log_emit, log_w, log_a, mask)
        n, tmax, k = le.shape
        init = lw if lw.dim() == 2 else lw[None, :].expand(n, k)

        alpha = init + le[:, 0, :]
        alphas = [alpha]
        for t in range(1, tmax):
            cand = torch.logsumexp(alpha[:, :, None] + la[None, :, :], dim=1) + le[:, t, :]
            alpha = torch.where(m[:, t][:, None] > 0, cand, alpha)
            alphas.append(alpha)
        alpha_stack = torch.stack(alphas, dim=1)
        ll = torch.logsumexp(alpha, dim=1)

        beta = torch.zeros((n, k), dtype=le.dtype, device=le.device)
        betas = [beta]
        for t in range(tmax - 2, -1, -1):
            step = la[None, :, :] + (le[:, t + 1, :] + beta)[:, None, :]
            cand = torch.logsumexp(step, dim=2)
            beta = torch.where(m[:, t + 1][:, None] > 0, cand, beta)
            betas.append(beta)
        beta_stack = torch.stack(betas[::-1], dim=1)

        wvec = torch.ones(n, dtype=le.dtype, device=le.device) if weights is None else self.asarray(weights)
        zero = torch.zeros((), dtype=le.dtype, device=le.device)

        ab = alpha_stack + beta_stack
        log_gamma = ab - torch.logsumexp(ab, dim=2, keepdim=True)
        gamma = torch.exp(log_gamma) * wvec[:, None, None]
        gamma = torch.where(m[:, :, None] > 0, gamma, zero)
        pi = gamma[:, 0, :]

        if tmax > 1:  # all transitions at once: (N, T-1, S, S)
            log_xi = (
                alpha_stack[:, :-1, :, None]
                + la[None, None, :, :]
                + (le[:, 1:, :] + beta_stack[:, 1:, :])[:, :, None, :]
                - ll[:, None, None, None]
            )
            contrib = torch.exp(log_xi) * wvec[:, None, None, None]
            contrib = torch.where((m[:, 1:] > 0)[:, :, None, None], contrib, zero)
            xi_sum = contrib.sum(dim=(0, 1))
        else:
            xi_sum = torch.zeros((k, k), dtype=le.dtype, device=le.device)

        return ll, gamma, xi_sum, pi

    def index_add(self, out: Any, index: Any, values: Any) -> Any:
        """Add ``values`` into ``out`` along axis 0 using Torch ``index_add``.

        Contract: return-value-only -- ``torch.Tensor.index_add`` (no trailing
        underscore) does not mutate ``out`` in place, unlike numpy's ``add.at``;
        callers must use the returned tensor.
        """
        # index must be a long tensor on out's device; coerce defensively so a
        # numpy index (or one on another device) does not raise a cryptic error
        index = torch.as_tensor(index, dtype=torch.long, device=out.device)
        # values must match out's dtype AND device. A float64 numpy array reaching a float32
        # accumulator worked by accident on CPU (torch promotes there) but raises
        # "self (Float) and source (Double) must have the same scalar type" on CUDA -- found by
        # the first real CUDA execution of the HMM float32 path (rented RTX 3060, 2026-08-04),
        # invisible on CPU and MPS runs of the identical fit.
        values = torch.as_tensor(values, dtype=out.dtype, device=out.device)
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
    # torch >= 2.5 exposes DTensor under the public ``torch.distributed.tensor``; torch 2.0-2.4 ship the
    # same symbols only under the private ``torch.distributed._tensor`` (the public module exists but is
    # empty). Try the public path first, then fall back, so multi-GPU sharding works on both.
    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
        except ImportError as e:  # pragma: no cover - depends on torch build
            raise ImportError("TorchEngine mesh placement requires torch.distributed.tensor.") from e
    return DTensor, Shard, Replicate, distribute_tensor


def _dtensor_ops_supported() -> bool:
    """Whether this torch registers DTensor sharding strategies for the mixture E-step's ops.

    torch >= 2.5 does (logsumexp / isinf / ... verified bit-identical on 2.12); torch 2.0-2.4 do not,
    so a component-sharded fit dies deep in the kernel. A version check (rather than a runtime probe,
    which would need a live process group at engine construction) keeps the gate low-overhead and
    side-effect-free."""
    try:
        major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return True  # unparseable (dev build) -> assume capable; the native op will error clearly if not
    return (major, minor) >= (2, 5)


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
