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
from typing import Any, NamedTuple

import numpy as np

from mixle.data.hashing import _records, dataset_hash


class PPLFitResult(NamedTuple):
    """Explicit fitted-object and provenance pair.

    A named tuple preserves the historical ``fitted, header = ...`` unpacking
    contract while avoiding optional, failure-prone mutation of ``fitted``.
    """

    fitted: Any
    header: Any


def _replayable_data(data: Any) -> Any:
    """Take one pre-fit snapshot of ``data`` that both the fit and the hash read.

    The old rule snapshotted only ``isinstance(data, Iterator)`` (MXR-080-1897). A generator is an
    ``Iterator``, but a one-shot *iterable* that is not one -- an object whose ``__iter__`` drains a
    file handle, a cursor, or a socket -- passed straight through: the fit consumed it, and the
    header then hashed the exhausted source, producing the hash of the **empty** dataset and
    ``n_records=None`` for a fit that saw every record. Nothing in the header disclosed that.

    So everything is materialized exactly once, which is what the estimator-surface counterpart
    ``mixle.inference.fit_with_provenance`` already does (``list(_records(data))``) and what
    :class:`mixle.data.core.MaterializedSource` documents as the rule: ``__len__`` is *not* evidence
    that a source is replayable, since one-shot readers routinely expose a remaining-length hint. The
    one exception is an ``ndarray``, recognized by exact type rather than by duck-typing: it is
    already a random-access buffer, so ``.copy()`` is a faithful owned snapshot and the fit keeps
    receiving an array instead of a list of scalars.

    A ``DataSource`` is snapshotted through its own ``records()`` contract, so the fit receives the
    records rather than the source object.
    """
    if isinstance(data, np.ndarray):
        return data.copy()
    return list(_records(data))


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

    One pre-fit snapshot (see :func:`_replayable_data`) is used for both the fit
    and the hash, and the snapshot is fingerprinted *before* the fit runs and
    again when the header is built. Those two fingerprints must agree: if the
    fit edits its input in place the header would otherwise report a hash of
    records the fit never saw, and a provenance header that silently describes
    the wrong dataset is worse than one that refuses to be built. The extra
    hash is one SHA-256 pass over the snapshot and no extra memory. Mutation
    *inside* a record object is deliberately still detected by this check but
    not prevented -- the snapshot is shallow (MXR-080-1897).

    ``seed=None`` has the deterministic effective seed ``0``; otherwise the
    exact non-negative integer seed initializes the ``RandomState`` passed to
    ``rv.fit``. Supplying a separate ``rng`` is rejected so the recorded state
    cannot disagree with the state used.
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
    pre_fit_hash = dataset_hash(training_data)

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
        "data_materialized": True,  # parity with the estimator surface's own training block
    }
    header = build_header(
        model,
        training_data,
        training=training,
        started=t0,
        finished=t1,
        resources=usage,
    )
    if header.dataset_hash != pre_fit_hash:
        raise RuntimeError(
            "the fitted data changed while rv.fit was running, so the provenance header would "
            f"describe records the fit never saw (pre-fit {pre_fit_hash[:16]}..., post-fit "
            f"{header.dataset_hash[:16]}...). A fit that edits its input in place cannot be given a "
            "truthful dataset hash; pass a copy, or use a fitter that does not mutate its input."
        )
    return PPLFitResult(fitted=fitted, header=header)
