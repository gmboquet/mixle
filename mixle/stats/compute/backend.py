"""Backend scoring dispatch over distribution-owned math hooks.

These helpers route encoded data through engine-aware distribution methods or
generated declarations while preserving the legacy accumulator path when a child
distribution cannot stay resident on the active engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.capability import (
    SupportsBackendComponentScoring,
    SupportsBackendScoring,
    supports,
)
from mixle.engines import NUMPY_ENGINE, ComputeEngine


class BackendScoringError(NotImplementedError):
    """Raised when a distribution has no backend scoring hook."""

    pass


def child_seq_update(accumulator: Any, enc: Any, weights: Any, estimate: Any, engine: ComputeEngine) -> None:
    """Route a child accumulator's E-step through the active engine when possible.

    Structural distributions (sequence, composite, optional, ...) delegate accumulation to child
    accumulators. When the engine prefers staying resident (``resident_estep``) this uses the child's
    ``seq_update_engine`` so nested families stay engine-resident; otherwise it falls back to the host
    ``seq_update`` with numpy weights. This is the recursion that pushes engine residency down a model tree.
    """
    if getattr(engine, "resident_estep", True) and callable(getattr(accumulator, "seq_update_engine", None)):
        accumulator.seq_update_engine(enc, weights, estimate, engine)
        return
    w = weights
    if hasattr(engine, "to_numpy"):
        w = np.asarray(engine.to_numpy(weights), dtype=np.float64)
    accumulator.seq_update(enc, w, estimate)


def backend_seq_log_density(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return per-row log densities using ``engine`` and distribution-local math."""
    enc = getattr(enc, "engine_payload", enc)
    if supports(dist, SupportsBackendScoring):
        return dist.backend_seq_log_density(enc, engine)
    try:
        from mixle.stats.compute.declarations import generated_log_density

        return generated_log_density(dist, enc, engine)
    except Exception as exc:
        raise BackendScoringError(
            "%s does not implement backend_seq_log_density and could not be generated: %s" % (type(dist).__name__, exc)
        ) from exc


def backend_seq_component_log_density(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return component log densities when a distribution exposes them."""
    enc = getattr(enc, "engine_payload", enc)
    if not supports(dist, SupportsBackendComponentScoring):
        raise BackendScoringError("%s does not implement backend_seq_component_log_density." % type(dist).__name__)
    return dist.backend_seq_component_log_density(enc, engine)


def backend_log_density_sum(dist: Any, enc: Any, engine: ComputeEngine = NUMPY_ENGINE) -> Any:
    """Return the total log likelihood for one encoded payload."""
    return engine.sum(backend_seq_log_density(dist, enc, engine))
