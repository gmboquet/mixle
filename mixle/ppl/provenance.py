"""Reproducible artifacts for the PPL surface: a provenance header for a fitted RandomVariable.

The estimator path has :func:`mixle.inference.fit_with_provenance`; this is its PPL counterpart. It times a
``rv.fit(...)`` (any ``how`` -- EM / MAP / MCMC / VI / ...), then builds a :class:`~mixle.inference.
provenance.Header` from the fitted model's *lowered* distribution (``rv.dist``), so the header gets the
concrete schema and final log-likelihood alongside the data hash, training settings, timing, resources, and
environment. The header is returned (and attached as ``rv.header`` when the RV permits attribute setting).
"""

from __future__ import annotations

import time
from typing import Any


def fit_with_provenance(rv: Any, data: Any, *, seed: int | None = None, **fit_kw: Any):
    """Fit a PPL ``RandomVariable`` on ``data`` and return ``(fitted_rv, header)`` with full provenance.

    ``fit_kw`` is passed through to ``rv.fit`` (``how=``, ``max_its=``, ``delta=``, ``backend=``, ...). The
    header is built from the fitted model's lowered distribution so it carries schema + final log-likelihood
    where available; ``method`` records the requested ``how``. The header is returned regardless; it is
    also attached as ``fitted.header`` when the RandomVariable allows it (RVs with ``__slots__`` do not)."""
    from mixle.inference.production.provenance import _resource_usage, build_header

    cpu0 = _resource_usage().get("cpu_time_s")
    t0 = time.time()
    fitted = rv.fit(data, **fit_kw)
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
        "seed": seed,
        "surface": "ppl",
    }
    header = build_header(model, data, training=training, started=t0, finished=t1, resources=usage)
    try:
        fitted.header = header
    except Exception:  # noqa: BLE001
        pass
    return fitted, header
