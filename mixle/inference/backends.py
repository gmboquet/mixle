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


def register_inference_backend(backend: InferenceBackend) -> None:
    """Register (or replace) an inference backend under ``backend.name``."""
    _INFERENCE_BACKENDS[backend.name] = backend


def get_inference_backend(name: str) -> InferenceBackend:
    """Return the registered backend named ``name`` (raises if unknown)."""
    try:
        return _INFERENCE_BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(_INFERENCE_BACKENDS)) or "<none registered>"
        raise ValueError(f"unknown inference backend {name!r}; registered: {known}.") from None


def available_backends() -> list[str]:
    """Return the names of registered backends whose engine is importable, in registration order."""
    return [name for name, b in _INFERENCE_BACKENDS.items() if b.available()]


def select_backend(backend: str = "auto", target: str | None = None) -> str:
    """Resolve a ``backend=`` argument to a concrete, available backend name.

    Policy:

    * An explicit ``backend`` (anything but ``"auto"``) is honored — it must be registered and its
      engine importable, else a clear error.
    * ``"auto"`` with a ``target`` *kind* hint picks the first preferred-and-available backend for
      that kind (e.g. ``"torch_logp"`` -> torch; ``"numpy_vg"`` -> numpy, then numba). This keeps
      the always-available numpy path the default for plain numpy targets.
    * ``"auto"`` with no hint falls back to the first available backend, preferring ``"numpy"``
      (the always-present, dependency-free path) when it is available.

    Raises if nothing is available or the explicit choice is unavailable.
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
        for name in _KIND_PREFERENCE.get(target, ()):  # target-kind precedence
            if name in avail:
                return name
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
