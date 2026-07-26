"""A single ``sample()`` entry point that draws from any samplable mixle object.

``sample(model, size)`` dispatches on the kind of ``model`` so distributions, conjugate posteriors,
relations, field posteriors and latent posteriors all read the same way::

    mixle.stats.sample(gauss, 100)                 # 100 iid observations
    mixle.stats.sample(posterior, 50)              # 50 parameter draws from a conjugate posterior
    mixle.stats.sample(assignment, 10, temperature=2.0)   # 10 Gibbs-weighted relation members
    mixle.stats.sample(field_post, 100)            # 100 joint field draws (dict per node)
    mixle.stats.sample(q_latent, 5)                # 5 latent-variable draws

A shared ``rng`` (a ``numpy.random.RandomState``) makes a whole pipeline reproducible: it is threaded
into the relation / field / latent draws directly, and for a distribution / conjugate posterior the
per-call stream is seeded from it, so one ``rng`` drives independent reproducible streams across many
``sample()`` calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index
from threading import RLock
from typing import Any

import numpy as np

from mixle.engines.arithmetic import maxrandint

__all__ = [
    "SampleDispatchRegistration",
    "register_sample_dispatch",
    "replace_sample_dispatch",
    "sample",
    "unregister_sample_dispatch",
]


def _resolve_rng(seed: int | None, rng: np.random.RandomState | None) -> np.random.RandomState:
    if rng is not None:
        return rng
    return np.random.RandomState(seed)


def _validate_size(size: Any) -> int | None:
    """Return one exact, non-negative sample count, preserving ``None`` as a scalar draw."""
    if size is None:
        return None
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("sample size must be an integer or None, not a boolean.")
    try:
        count = index(size)
    except TypeError as exc:
        raise TypeError("sample size must be an integer or None.") from exc
    if count < 0:
        raise ValueError("sample size must be non-negative.")
    return count


# Out-of-core samplable handlers. A higher layer (e.g. ``mixle.ppl`` for ``FieldPosterior``) registers a
# dispatcher for its own types here, so this core module never imports upward to name them -- keeping the
# dependency graph strictly ppl -> core. Each handler is ``fn(model, size, *, seed, rng, **kwargs)`` and
# returns a draw, or the ``SAMPLE_UNHANDLED`` sentinel if ``model`` is not its type.
SAMPLE_UNHANDLED: Any = object()
SampleDispatch = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SampleDispatchRegistration:
    """One named extension in the deterministic :func:`sample` dispatch order."""

    dispatch_id: str
    handler: SampleDispatch

    def __post_init__(self) -> None:
        if not isinstance(self.dispatch_id, str) or not self.dispatch_id.strip():
            raise ValueError("sample dispatch_id must be a non-empty string.")
        if not callable(self.handler):
            raise TypeError("sample dispatch handler must be callable.")


_SAMPLE_DISPATCHERS: dict[str, SampleDispatchRegistration] = {}
_SAMPLE_DISPATCH_LOCK = RLock()


def _install_sample_dispatch(
    dispatch_id: str,
    fn: SampleDispatch,
    *,
    replace: bool,
) -> SampleDispatch:
    registration = SampleDispatchRegistration(dispatch_id, fn)
    with _SAMPLE_DISPATCH_LOCK:
        exists = dispatch_id in _SAMPLE_DISPATCHERS
        if exists != replace:
            action = "replace" if exists else "register"
            raise KeyError(f"cannot {action} sample dispatch {dispatch_id!r}: registration {'exists' if exists else 'missing'}.")
        # A dict assignment preserves the existing precedence slot during an explicit replacement.
        _SAMPLE_DISPATCHERS[dispatch_id] = registration
    return fn


def register_sample_dispatch(dispatch_id: str, fn: SampleDispatch | None = None):
    """Register a named out-of-core sampler, rejecting duplicate IDs.

    Registration order defines dispatch precedence. A handler that does not own ``model`` MUST return
    :data:`SAMPLE_UNHANDLED` before mutating the model, RNG, or any other observable state.
    """

    def decorate(handler: SampleDispatch) -> SampleDispatch:
        return _install_sample_dispatch(dispatch_id, handler, replace=False)

    return decorate if fn is None else decorate(fn)


def replace_sample_dispatch(dispatch_id: str, fn: SampleDispatch) -> SampleDispatch:
    """Explicitly replace an existing named handler without changing its precedence slot."""
    return _install_sample_dispatch(dispatch_id, fn, replace=True)


def unregister_sample_dispatch(dispatch_id: str) -> SampleDispatchRegistration:
    """Remove and return an existing named handler."""
    if not isinstance(dispatch_id, str) or not dispatch_id.strip():
        raise ValueError("sample dispatch_id must be a non-empty string.")
    with _SAMPLE_DISPATCH_LOCK:
        try:
            return _SAMPLE_DISPATCHERS.pop(dispatch_id)
        except KeyError:
            raise KeyError(f"sample dispatch {dispatch_id!r} is not registered.") from None


def sample(
    model: Any,
    size: int | None = None,
    *,
    seed: int | None = None,
    rng: np.random.RandomState | None = None,
    **kwargs: Any,
) -> Any:
    """Draw sample(s) from any samplable mixle object.

    Args:
        model: a distribution, conjugate posterior, :class:`~mixle.relations.Relation`,
            ``FieldPosterior`` or ``LatentPosterior``.
        size: ``None`` returns a single draw in the object's natural type; a non-negative integer
            returns a collection (an array for homogeneous leaves, a list / dict-of-arrays for
            structured draws). Zero returns an empty collection in that natural collection type.
        seed: scalar seed for the draw. Mutually exclusive with ``rng``.
        rng: a shared ``RandomState`` for reproducible, composable streams. Mutually exclusive
            with ``seed``.
        **kwargs: forwarded to the underlying sampler -- e.g. ``temperature`` / ``k`` / ``uniform`` for a
            relation, ``nodes`` for a field posterior, ``batched`` for a distribution.

    Returns:
        A single draw (``size=None``) or a collection of ``size`` draws.

    Raises:
        TypeError: if ``model`` is not a recognized samplable object, if ``size`` is not an integer
            or ``None``, or if both ``seed`` and ``rng`` are supplied (two randomness sources are
            ambiguous -- the same double-supply policy as the constructor keyword aliases).
        ValueError: if ``size`` is negative.
    """
    if seed is not None and rng is not None:
        raise TypeError("'seed' and 'rng' are mutually exclusive; pass only one")
    size = _validate_size(size)
    # Relation -- a sampler under a Gibbs measure over its members (temperature/k/uniform are sampler args).
    from mixle.relations import Relation

    if isinstance(model, Relation):
        return model.sampler(seed=seed, rng=rng, **kwargs).sample(size)

    # Out-of-core samplables registered by higher layers (e.g. mixle.ppl FieldPosterior -- joint
    # field/parameter draws). Iterated before LatentPosterior to preserve the original dispatch order.
    with _SAMPLE_DISPATCH_LOCK:
        dispatchers = tuple(_SAMPLE_DISPATCHERS.values())
    for registration in dispatchers:
        out = registration.handler(model, size, seed=seed, rng=rng, **kwargs)
        if out is not SAMPLE_UNHANDLED:
            return out

    # LatentPosterior -- latent-variable draws (one per call; loop for a collection).
    from mixle.stats.compute.posterior import LatentPosterior

    if isinstance(model, LatentPosterior):
        r = _resolve_rng(seed, rng)
        return model.sample(r) if size is None else [model.sample(r) for _ in range(size)]

    # Distribution or ConjugatePosterior -- both expose .sampler(seed).sample(size).
    if hasattr(model, "sampler"):
        draw_seed = int(rng.randint(0, maxrandint)) if rng is not None else seed
        return model.sampler(seed=draw_seed).sample(size, **kwargs)

    raise TypeError(
        f"don't know how to sample from a {type(model).__name__}; expected a distribution, "
        "conjugate posterior, Relation, FieldPosterior, or LatentPosterior."
    )
