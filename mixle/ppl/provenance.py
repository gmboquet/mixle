"""Reproducible artifacts for the PPL surface: a provenance header for a fitted RandomVariable.

The estimator path has :func:`mixle.inference.fit_with_provenance`; this is its PPL counterpart. It times a
``rv.fit(...)`` (any ``how`` -- EM / MAP / MCMC / VI / ...), then builds a :class:`~mixle.inference.
provenance.Header` from the fitted model's *lowered* distribution (``rv.dist``), so the header gets the
concrete schema and final log-likelihood alongside the data hash, training settings, timing, resources, and
environment. The explicit result object is authoritative; this function never mutates the fitted object to
attach metadata.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np


class PPLFitResult(NamedTuple):
    """Explicit fitted-object and provenance pair.

    A named tuple preserves the historical ``fitted, header = ...`` unpacking
    contract while avoiding optional, failure-prone mutation of ``fitted``.
    """

    fitted: Any
    header: Any


def _replayable_data(data: Any) -> Any:
    """Materialize a one-shot iterator exactly once; preserve replayable data sources."""
    if isinstance(data, Iterator):
        return list(data)
    return data


def _rng_record(rng: np.random.RandomState, requested_seed: int | None, effective_seed: int) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = rng.get_state()
    state_bytes = (
        algorithm.encode("ascii")
        + keys.tobytes()
        + int(position).to_bytes(8, "big", signed=False)
        + int(has_gauss).to_bytes(1, "big", signed=False)
        + np.float64(cached_gaussian).tobytes()
    )
    return {
        "kind": "numpy.random.RandomState",
        "algorithm": algorithm,
        "requested_seed": requested_seed,
        "effective_seed": effective_seed,
        "initial_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    }


def fit_with_provenance(
    rv: Any,
    data: Any,
    *,
    seed: int | None = None,
    **fit_kw: Any,
) -> PPLFitResult:
    """Fit a PPL random variable and return an explicit provenance result.

    One exact replayable representation of one-shot data is used for both fit
    and hashing. ``seed=None`` has the deterministic effective seed ``0``;
    otherwise the exact non-negative integer seed initializes the
    ``RandomState`` passed to ``rv.fit``. Supplying a separate ``rng`` is
    rejected so the recorded state cannot disagree with the state used.
    """
    from mixle.inference.production.provenance import _resource_usage, build_header

    if "rng" in fit_kw:
        raise ValueError("fit_with_provenance owns the random-state contract; pass seed= instead of rng=")
    if isinstance(seed, (bool, np.bool_)) or (seed is not None and not isinstance(seed, (int, np.integer))):
        raise TypeError("seed must be a non-negative integer or None")
    effective_seed = 0 if seed is None else int(seed)
    if effective_seed < 0:
        raise ValueError("seed must be a non-negative integer or None")
    rng = np.random.RandomState(effective_seed)
    random_state = _rng_record(rng, None if seed is None else int(seed), effective_seed)
    fit_kw["rng"] = rng
    training_data = _replayable_data(data)

    cpu0 = _resource_usage().get("cpu_time_s")
    t0 = time.time()
    fitted = rv.fit(training_data, **fit_kw)
    t1 = time.time()
    usage = _resource_usage()
    if cpu0 is not None and usage.get("cpu_time_s") is not None:
        usage = {"cpu_time_s": round(usage["cpu_time_s"] - cpu0, 3), "peak_rss_mb": usage.get("peak_rss_mb")}

    model = getattr(fitted, "dist", fitted)  # lowered concrete distribution -> schema + scoring
    training = {
        "method": fit_kw.get("how", "auto"),
        "max_its": fit_kw.get("max_its"),
        "delta": fit_kw.get("delta"),
        "backend": fit_kw.get("backend", "local"),
        "seed": effective_seed,
        "random_state": random_state,
        "surface": "ppl",
    }
    header = build_header(
        model,
        training_data,
        training=training,
        started=t0,
        finished=t1,
        resources=usage,
    )
    return PPLFitResult(fitted=fitted, header=header)
