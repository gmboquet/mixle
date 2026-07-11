"""Posterior summarization for the mixle PPL: highest-density intervals and an ArviZ-style table.

After an MCMC / ensemble fit you want a compact, readable report of each parameter's posterior. The
equal-tailed credible interval in :meth:`RandomVariable.summary` is fine for symmetric posteriors;
:func:`hdi` gives the *highest-density* interval (the narrowest interval holding the mass, the right
choice for skewed or bounded posteriors), and :func:`posterior_summary` assembles the mean / sd / HDI
together with the convergence diagnostics (effective sample size, R-hat) into one per-parameter dict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable


def hdi(samples: Sequence[float], prob: float = 0.94) -> tuple[float, float]:
    """Highest-density interval: the narrowest interval containing ``prob`` of the posterior mass.

    For a unimodal posterior this is the shortest ``(low, high)`` such that ``P(low <= x <= high) =
    prob``; unlike an equal-tailed interval it tracks an asymmetric or bounded posterior correctly.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1).")
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    if n == 0:
        raise ValueError("samples is empty.")
    k = int(np.floor(prob * n))
    if k >= n:
        return float(x[0]), float(x[-1])
    k = max(k, 1)
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def posterior_summary(fitted: RandomVariable, *, hdi_prob: float = 0.94) -> dict[str, dict[str, Any]]:
    """Per-parameter posterior summary table for a fitted PPL model (best after ``how='mcmc'``).

    Returns ``{param_name: {'mean', 'sd', 'hdi_low', 'hdi_high', 'ess', 'r_hat'}}``. ``mean``/``sd`` come
    from the fit's own summary; the HDI is computed from the posterior draws (when the fit exposes them);
    ``ess`` (effective sample size) and ``r_hat`` (Gelman-Rubin, multi-chain) come from the sampler's
    diagnostics when present. A point fit (em/map) yields just ``mean``/``sd``.
    """
    summ = fitted.summary()
    result = getattr(fitted, "_result", None)
    rhat = getattr(result, "rhat", None) if result is not None else None
    ess = getattr(result, "ess", None) if result is not None else None
    out: dict[str, dict[str, Any]] = {}
    for name, stat in summ.items():
        if name.startswith("_") or not isinstance(stat, dict):
            continue
        row: dict[str, Any] = {"mean": stat.get("mean"), "sd": stat.get("std", stat.get("sd"))}
        draws = None
        try:
            draws = np.asarray(fitted.posterior(name), dtype=float).ravel()
        except Exception:  # noqa: BLE001
            draws = None
        if draws is not None and draws.size > 1:
            lo, hi = hdi(draws, hdi_prob)
            row["hdi_low"] = lo
            row["hdi_high"] = hi
        if isinstance(rhat, dict) and name in rhat:
            row["r_hat"] = float(rhat[name])
        if ess is not None and isinstance(ess, (int, float)):
            row["ess"] = float(ess)
        out[name] = row
    return out


__all__ = ["hdi", "posterior_summary"]
