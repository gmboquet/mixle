"""Inference-backend registry — `register, don't branch` for `mixle.inference`.

Mirrors :func:`mixle.stats.compute.kernel.register_kernel_factory`: each engine's NUTS
implementation *self-registers* an :class:`InferenceBackend` at import time, so the dispatcher
(:func:`mixle.inference.nuts`) never grows a central ``if engine == ...`` switch. A backend declares

* ``name`` — the selector string (``"numpy"``, ``"numba"``, ``"torch"``, ``"jax"``).
* ``available`` — a zero-arg predicate: is the engine importable on this host?  Kept lazy so
  ``import mixle.inference`` works with any subset of optional engines installed.
* ``target_kind`` — what *contract* the caller's target must satisfy: a numpy fused
  ``value_and_grad`` (``"numpy_vg"``), an ``@njit`` fused ``value_and_grad`` (``"njit_vg"``), a
  torch scalar ``logp`` (``"torch_logp"``), or a jax scalar ``logp`` (``"jax_logp"``). The kinds
  cannot be auto-converted across autodiff systems, so the *target* is what ultimately picks a
  backend in ``"auto"`` mode (see :func:`select_backend`).
* ``nuts`` — the callable that runs the sampler and returns a :class:`mixle.inference.NutsResult`.

``available_backends()`` lists the installed engines; ``select_backend()`` resolves the
``backend=`` argument (including ``"auto"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mixle.inference import NutsResult

# target_kind -> the backend "auto" prefers for a target declared with that kind. A target is the
# strongest available signal: a torch logp can only run on the torch backend, etc.
_KIND_PREFERENCE: dict[str, tuple[str, ...]] = {
    "numpy_vg": ("numpy", "numba"),  # an analytic numpy vg: numpy first, numba can run it too
    "njit_vg": ("numba",),
    "torch_logp": ("torch",),
    "jax_logp": ("jax",),
}


@dataclass(frozen=True)
class InferenceBackend:
    """A registered inference engine: a name, an availability probe, a target contract, a sampler."""

    name: str
    available: Callable[[], bool]
    target_kind: str
    nuts: Callable[..., NutsResult]


_INFERENCE_BACKENDS: dict[str, InferenceBackend] = {}


def register_inference_backend(backend: InferenceBackend, *, replace: bool = False) -> None:
    """Register an inference backend under ``backend.name``.

    The registry is global process state that :func:`select_backend` dispatches every ``nuts`` call
    through, so one malformed registration is not a local mistake -- it corrupts dispatch for every
    caller. A backend with an empty name and an unrecognized ``target_kind`` used to register
    successfully and become selectable, even though :func:`select_backend` rejects that same kind
    string when a *caller* supplies it.

    Replacing an already-registered name requires ``replace=True``. Silent replacement let a second
    import or a typo'd name swap out a live sampler with no trace; an intentional override (a test
    double, a vendored engine) says so explicitly.

    Raises:
        TypeError: if ``backend`` is not an :class:`InferenceBackend`, or ``available``/``nuts`` are
            not callable.
        ValueError: for an empty/blank name, an unrecognized ``target_kind``, or an unrequested
            replacement of a registered name.
    """
    if not isinstance(backend, InferenceBackend):
        raise TypeError(f"register_inference_backend expects an InferenceBackend, got {type(backend).__name__}.")
    if not isinstance(backend.name, str) or not backend.name.strip():
        raise ValueError(f"inference backend name must be a non-empty string; got {backend.name!r}.")
    if backend.target_kind not in _KIND_PREFERENCE:
        raise ValueError(
            f"inference backend {backend.name!r} declares unknown target kind {backend.target_kind!r}; "
            f"known kinds: {', '.join(sorted(_KIND_PREFERENCE))}. Target kinds are the calling convention "
            "select_backend dispatches on, so an unrecognized one is never selectable for any target."
        )
    if not callable(backend.available):
        raise TypeError(f"inference backend {backend.name!r} needs a callable available() probe.")
    if not callable(backend.nuts):
        raise TypeError(f"inference backend {backend.name!r} needs a callable nuts implementation.")
    if backend.name in _INFERENCE_BACKENDS and not replace:
        raise ValueError(
            f"inference backend {backend.name!r} is already registered; pass replace=True to override it "
            "deliberately (silent replacement swaps out a live sampler for every caller with no trace)."
        )
    _INFERENCE_BACKENDS[backend.name] = backend


def get_inference_backend(name: str) -> InferenceBackend:
    """Return the registered backend named ``name`` (raises if unknown)."""
    try:
        return _INFERENCE_BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(_INFERENCE_BACKENDS)) or "<none registered>"
        raise ValueError(f"unknown inference backend {name!r}; registered: {known}.") from None


def available_backends() -> list[str]:
    """Return the names of registered backends whose engine is importable, in registration order.

    Each probe is isolated: one backend whose ``available()`` raises used to propagate out of this
    call and hide *every* healthy backend, so a single broken third-party registration made the
    always-present numpy path look unavailable too. A raising probe answers "unknown", which is
    reported as a warning and treated as unavailable -- it cannot be silently equivalent to a probe
    that returned ``False``, because the caller may need to know its engine is misconfigured rather
    than absent.
    """
    return [name for name, _ok, _err in _probe_backends() if _ok]


def backend_availability() -> list[tuple[str, bool, str | None]]:
    """Per-backend ``(name, available, probe_error)`` in registration order.

    ``probe_error`` is ``None`` unless that backend's ``available()`` raised, in which case it is the
    formatted exception and ``available`` is ``False``.
    """
    return _probe_backends()


def _probe_backends() -> list[tuple[str, bool, str | None]]:
    import warnings

    out: list[tuple[str, bool, str | None]] = []
    for name, b in _INFERENCE_BACKENDS.items():
        try:
            out.append((name, bool(b.available()), None))
        except Exception as exc:  # noqa: BLE001 - one broken probe must not hide every healthy backend
            warnings.warn(
                f"inference backend {name!r} availability probe raised "
                f"{type(exc).__name__}: {exc}; treating it as unavailable.",
                RuntimeWarning,
                stacklevel=3,
            )
            out.append((name, False, f"{type(exc).__name__}: {exc}"))
    return out


def select_backend(backend: str = "auto", target: str | None = None) -> str:
    """Resolve a ``backend=`` argument to a concrete, available backend name.

    Policy:

    * An explicit ``backend`` (anything but ``"auto"``) is honored — it must be registered and its
      engine importable, else a clear error.
    * ``"auto"`` with a ``target`` *kind* hint picks the first preferred-and-available backend for
      that kind (e.g. ``"torch_logp"`` -> torch; ``"numpy_vg"`` -> numpy, then numba). This keeps
      the always-available numpy path the default for plain numpy targets. A declared kind is a
      *contract*, not a preference: if no backend advertising that contract is available, this
      raises rather than silently handing e.g. a torch/jax scalar ``logp`` to a backend that will
      call it expecting a numpy ``value_and_grad -> (value, gradient)`` pair. The kinds are not
      convertible across autodiff systems, so there is no correct fallback between them.
    * ``"auto"`` with no hint (``target=None``) falls back to the first available backend,
      preferring ``"numpy"`` (the always-present, dependency-free path) when it is available. This
      is the genuinely undetermined case -- a numpy and an ``@njit`` ``value_and_grad`` are
      indistinguishable plain callables -- and stays unchanged.

    Raises:
        ValueError: if ``target`` is not one of the known target kinds.
        RuntimeError: if nothing is available, the explicit choice is unavailable, or no available
            backend implements the declared ``target`` kind.
    """
    avail = available_backends()
    if not avail:
        raise RuntimeError("no inference backends are available.")
    if backend != "auto":
        b = get_inference_backend(backend)  # validates the name
        if not b.available():
            raise RuntimeError(f"inference backend {backend!r} is registered but its engine is not importable.")
        return backend
    if target is not None:
        preferred = _KIND_PREFERENCE.get(target)
        if preferred is None:
            # An unknown or empty kind string used to fall through to numpy, which is exactly the
            # wrong answer: the caller told us something about the target's calling convention and we
            # ignored it. A typo'd kind must not silently become the numpy contract.
            raise ValueError(f"unknown target kind {target!r}; known kinds: {', '.join(sorted(_KIND_PREFERENCE))}.")
        for name in preferred:  # target-kind precedence
            if name in avail:
                return name
        raise RuntimeError(
            f"no available inference backend implements target kind {target!r} "
            f"(it needs one of: {', '.join(preferred)}; available: {', '.join(avail)}). "
            "Target kinds are not convertible across autodiff systems, so 'auto' will not substitute "
            "a backend expecting a different calling convention -- install the required engine or pass "
            "a target of a supported kind."
        )
    if "numpy" in avail:  # dependency-free default
        return "numpy"
    return avail[0]


def _dispatch_target_kind(target: Any, explicit_kind: str | None) -> str | None:
    """Best-effort hint for ``select_backend`` from the *type* of a BYO target.

    Only torch/jax targets are auto-detectable from their type; a numpy ``value_and_grad`` and an
    ``@njit`` ``value_and_grad`` are both plain callables and indistinguishable here, so they fall
    through to ``None`` (and the explicit ``backend=`` / numpy default decides). ``explicit_kind``
    short-circuits the probe.
    """
    if explicit_kind is not None:
        return explicit_kind
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch

        if isinstance(target, torch.nn.Module) or _returns_torch(target):
            return "torch_logp"
    return None


def _returns_torch(target: Any) -> bool:
    # Intentionally conservative: we do not call the target here (it may be expensive / stateful),
    # so unannotated callables stay ``None`` and rely on backend= or the numpy default.
    return False
