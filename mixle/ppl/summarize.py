"""Posterior summarization for the mixle PPL: highest-density intervals and an ArviZ-style table.

After an MCMC / ensemble fit you want a compact, readable report of each parameter's posterior. The
equal-tailed credible interval in :meth:`RandomVariable.summary` is fine for symmetric posteriors;
:func:`hdi` gives the *highest-density* interval (the narrowest interval holding the mass, the right
choice for skewed or bounded posteriors), and :func:`posterior_summary` assembles the mean / sd / HDI
together with the convergence diagnostics (effective sample size, R-hat) into one per-parameter dict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    try:
        x = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("samples must be a finite one-dimensional numeric sequence") from error
    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional; select one coordinate before computing a multivariate HDI")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must contain only finite posterior draws")
    x = np.sort(x)
    n = x.size
    if n == 0:
        raise ValueError("samples is empty.")
    if n == 1:
        return float(x[0]), float(x[0])
    k = int(np.floor(prob * n))
    if k >= n:
        return float(x[0]), float(x[-1])
    k = max(k, 1)
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def posterior_summary(fitted: RandomVariable, *, hdi_prob: float = 0.94) -> dict[str, dict[str, Any]]:
    """Per-parameter posterior summary table for a fitted PPL model (best after ``how='mcmc'``).

    Returns a fixed per-parameter schema with ``mean``, ``sd``, ``hdi_low``, ``hdi_high``,
    ``ess``, ``ess_tail``, ``r_hat``, ``mcse``, ``diagnostic_status``, and ``diagnostic_error``.
    ``mean``/``sd`` come from the fit's own summary; the HDI is computed from the posterior draws
    (when the fit exposes them); ``ess`` (effective sample size) and ``r_hat`` (Gelman-Rubin,
    multi-chain) come from the sampler's diagnostics when present; ``mcse`` is the Monte Carlo
    standard error of the mean, ``sd / sqrt(ess)`` -- the parameter number's own noise floor,
    published WITH the number instead of dropped (STAT-RR17-11). A point fit (em/map) yields just
    ``mean``/``sd``.

    ``diagnostic_status`` semantics (STAT-RR17-11 -- "ok" used to mean only "the ESS call did not
    raise", so a one-chain fit with ``r_hat=None`` or NaN split-R-hat still published parameter
    numbers under "ok"):

    * ``"ok"`` -- ess, ess_tail, mcse are finite AND a finite multi-chain ``r_hat`` exists. The
      ONLY promotable state.
    * ``"single-chain-mixing-unassessable"`` -- ESS computed, but one chain cannot assess mixing
      (no R-hat); run >= 2 chains before treating the summary as converged evidence.
    * ``"unusable"`` -- a diagnostic evaluated non-finite (NaN/inf ESS or R-hat).
    * ``"unavailable"`` / ``"failed"`` -- draws or diagnostics missing / raised.
    """
    summ = fitted.summary()
    result = getattr(fitted, "_result", None)
    rhat = getattr(result, "split_rhat", None) if result is not None else None
    if not isinstance(rhat, dict):
        rhat = getattr(result, "rhat", None) if result is not None else None
    bulk_by_parameter = getattr(result, "bulk_ess", None) if result is not None else None
    tail_by_parameter = getattr(result, "tail_ess", None) if result is not None else None
    out: dict[str, dict[str, Any]] = {}
    for name, stat in summ.items():
        if name.startswith("_") or not isinstance(stat, dict):
            continue
        row: dict[str, Any] = {
            "mean": stat.get("mean"),
            "sd": stat.get("std", stat.get("sd")),
            "hdi_low": None,
            "hdi_high": None,
            "ess": None,
            "ess_tail": None,
            "r_hat": None,
            "mcse": None,
            "diagnostic_status": "unavailable",
            "diagnostic_error": None,
        }
        try:
            draws = np.asarray(fitted.posterior(name), dtype=float)
            # HDI over the POOLED draws: hdi() is one-dimensional by contract, and passing a
            # (n_chains, n_draws) matrix raised -- which silently routed every multi-chain fit to
            # "failed" before its diagnostics were even computed, leaving single-chain fits as
            # the only ones that could ever read "ok" (STAT-RR17-11's enabling accident).
            lo, hi = hdi(draws.reshape(-1), hdi_prob)
            row["hdi_low"], row["hdi_high"] = lo, hi
            # The ESS estimators take (n_chains, n_draws). A fit that exposes a flat vector of draws
            # is presented to them as one chain -- passing the 1-D array straight through made
            # them raise, and every ess/ess_tail came back None with diagnostic_status "failed".
            chains = draws.reshape(1, -1) if draws.ndim == 1 else draws
            # How many chains ran is a fact about the FIT, not about the shape posterior()
            # returns: the public MCMC path pools chains into a flat vector, so judging by
            # draws.ndim branded every real 4-chain fit "single-chain" while printing its finite
            # multi-chain R-hat in the same row (pass 19: "ok" was unreachable through the public
            # path -- the RR17-11 taxonomy inverted again). _attach_convergence writes a FINITE
            # per-parameter split-R-hat only when >= 2 chains ran (deliberate NaN for one chain),
            # so a finite result-supplied R-hat is multi-chain evidence even for pooled draws.
            rhat_from_result = None
            if isinstance(rhat, dict) and name in rhat:
                try:
                    rhat_from_result = float(rhat[name])
                except (TypeError, ValueError):
                    rhat_from_result = float("nan")
            multi_chain = chains.shape[0] > 1 or (rhat_from_result is not None and np.isfinite(rhat_from_result))

            def _ess_value(
                by_parameter: Any,
                compute: Callable[[np.ndarray], float],
                *,
                key: str = name,
                is_multi: bool = multi_chain,
                chain_matrix: np.ndarray = chains,
            ) -> float:
                if isinstance(by_parameter, dict) and key in by_parameter:
                    supplied = float(by_parameter[key])
                    if np.isfinite(supplied) or is_multi:
                        # a non-finite value for a MULTI-chain fit is a genuine diagnostic
                        # failure and must surface as one (unusable), never be papered over
                        return supplied
                # either no result-supplied value, or the result's deliberate single-chain NaN
                # marker: the single-chain ESS itself is computable and honest -- what one chain
                # cannot assess is MIXING, and the status says that, not "numerically failed"
                return float(compute(chain_matrix))

            from mixle.ppl.diagnostics import bulk_ess, tail_ess

            row["ess"] = _ess_value(bulk_by_parameter, bulk_ess)
            row["ess_tail"] = _ess_value(tail_by_parameter, tail_ess)
            row["diagnostic_status"] = "pending"
        except (KeyError, NotImplementedError) as error:
            row["diagnostic_status"] = "unavailable"
            row["diagnostic_error"] = f"{type(error).__name__}: {error}"
        except (TypeError, ValueError, RuntimeError) as error:
            row["diagnostic_status"] = "failed"
            row["diagnostic_error"] = f"{type(error).__name__}: {error}"
        if isinstance(rhat, dict) and name in rhat:
            try:
                row["r_hat"] = float(rhat[name])
            except (TypeError, ValueError):
                row["r_hat"] = float("nan")
        if row["diagnostic_status"] == "pending":
            # STAT-RR17-11: "ok" means USABLE diagnostics, not "the ESS call returned". mcse is
            # the mean's own Monte Carlo noise floor; a NaN anywhere is unusable, and a single
            # chain cannot assess mixing at all, so it is never "ok".
            sd = row["sd"]
            if row["ess"] is not None and np.isfinite(row["ess"]) and row["ess"] > 0 and sd is not None:
                row["mcse"] = float(sd) / float(np.sqrt(row["ess"]))
            finite_core = all(row[k] is not None and np.isfinite(row[k]) for k in ("ess", "ess_tail", "mcse"))
            rhat_value = row["r_hat"]
            if rhat_value is not None and not np.isfinite(rhat_value) and not multi_chain:
                # the result's deliberate single-chain marker: R-hat is UNDEFINED for one chain,
                # which is the single-chain status below, not a numeric failure
                row["r_hat"] = None
                rhat_value = None
            if not finite_core or (rhat_value is not None and not np.isfinite(rhat_value)):
                row["diagnostic_status"] = "unusable"
                row["diagnostic_error"] = "a diagnostic evaluated non-finite (NaN/inf)"
            elif not multi_chain or rhat_value is None:
                row["diagnostic_status"] = "single-chain-mixing-unassessable"
                # the message states which precondition actually failed instead of asserting
                # "one chain" for a multi-chain fit that merely lacks a usable R-hat
                row["diagnostic_error"] = (
                    "one chain: R-hat undefined; run >= 2 chains before promoting"
                    if not multi_chain
                    else "mixing unassessable: the fit supplied no usable multi-chain R-hat for this parameter"
                )
            else:
                row["diagnostic_status"] = "ok"
        out[name] = row
    return out


__all__ = ["hdi", "posterior_summary"]
