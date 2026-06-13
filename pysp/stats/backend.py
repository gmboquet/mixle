"""Generic backend scoring dispatch over distribution-owned math hooks."""
from __future__ import annotations

from typing import Any

from pysp.engines import NUMPY_ENGINE, ComputeEngine


class BackendScoringError(NotImplementedError):
    """Raised when a distribution has no backend scoring hook."""

    pass


def backend_seq_log_density(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return per-row log densities using ``engine`` and distribution-local math."""
    enc = getattr(enc, 'engine_payload', enc)
    fn = getattr(dist, 'backend_seq_log_density', None)
    if callable(fn):
        return fn(enc, engine)
    try:
        from pysp.stats.declarations import generated_log_density
        return generated_log_density(dist, enc, engine)
    except Exception as exc:
        raise BackendScoringError(
            '%s does not implement backend_seq_log_density and could not be generated: %s' %
            (type(dist).__name__, exc)) from exc


def backend_seq_component_log_density(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return component log densities when a distribution exposes them."""
    enc = getattr(enc, 'engine_payload', enc)
    fn = getattr(dist, 'backend_seq_component_log_density', None)
    if not callable(fn):
        raise BackendScoringError('%s does not implement backend_seq_component_log_density.' % type(dist).__name__)
    return fn(enc, engine)


def backend_log_density_sum(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return the total log likelihood for one encoded payload."""
    return engine.sum(backend_seq_log_density(dist, enc, engine))
