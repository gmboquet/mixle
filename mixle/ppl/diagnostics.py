"""Predictive model-comparison diagnostics for the mixle PPL: WAIC and PSIS-LOO.

Both estimate the expected log pointwise predictive density (elpd) -- how well a fitted Bayesian model
predicts new data -- from the pointwise log-likelihood matrix ``loglik`` of shape ``(n_draws, n_obs)``
(the log-density of each observation under each posterior draw of the parameters).

* ``waic`` -- the Widely Applicable Information Criterion (Watanabe): ``elpd = lppd - p_waic`` with the
  effective parameter count ``p_waic`` the per-observation posterior variance of the log-likelihood.
* ``psis_loo`` -- Pareto-Smoothed Importance-Sampling Leave-One-Out cross-validation (Vehtari, Gelman &
  Gabry 2017). Importance-reweights the full-data posterior to each leave-one-out posterior, smoothing
  the heavy importance-weight tail with a generalized-Pareto fit; it reports the diagnostic shape
  ``khat`` (values above ~0.7 flag unreliable estimates).

Both return results on the deviance scale (``waic``/``loo`` = ``-2 * elpd``, lower is better) with a
standard error, matching the conventions of Stan / ArviZ / the R ``loo`` package.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from mixle.utils.special import logsumexp as _logsumexp


def _lppd_pointwise(loglik: np.ndarray) -> np.ndarray:
    """Log pointwise predictive density per observation: log mean_s exp(loglik[s, i])."""
    s = loglik.shape[0]
    return _logsumexp(loglik, axis=0) - np.log(s)


# ------------------------------------------- convergence diagnostics (Vehtari et al. 2021, Bayesian Analysis)
# Rank-normalized split-R-hat and bulk/tail effective sample size -- the modern Stan/ArviZ standard. The
# inputs are an ``(n_chains, n_draws)`` array of draws for one scalar parameter.


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance of a 1-D series at all lags via FFT."""
    n = x.size
    c = x - x.mean()
    m = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(c, n=m)
    ac = np.fft.irfft(f * np.conjugate(f), n=m)[:n].real
    return ac / n


def _ess_chains(x: np.ndarray) -> float:
    """Stan effective sample size for one parameter, ``x`` of shape ``(n_chains, n_draws)``."""
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    acov = np.array([_autocov(x[c]) for c in range(m)])
    mean_acov = acov.mean(axis=0)
    chain_var = acov[:, 0] * n / (n - 1.0)
    w = float(chain_var.mean())
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    b = n * float(np.var(x.mean(axis=1), ddof=1))
    var_plus = (n - 1.0) / n * w + b / n
    if not np.isfinite(var_plus) or var_plus <= 0:
        return float("nan")
    rho = 1.0 - (w - mean_acov) / var_plus  # rho[0] == 1
    # Geyer initial monotone positive sequence on paired autocorrelations.
    pairs = []
    k = 0
    while 2 * k + 2 < n:
        p = rho[2 * k + 1] + rho[2 * k + 2]
        if p < 0:
            break
        pairs.append(p)
        k += 1
    for i in range(1, len(pairs)):
        pairs[i] = min(pairs[i], pairs[i - 1])  # enforce monotone decreasing
    tau = max(1.0 + 2.0 * float(sum(pairs)), 1.0)
    return float(m * n / tau)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Blom rank-normalization to normal scores: ``Phi^{-1}((rank - 3/8) / (N - 1/4))``."""
    from scipy.stats import norm, rankdata

    r = rankdata(x).reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size - 0.25))


def _classic_rhat(x: np.ndarray) -> float:
    """Potential scale reduction from chains ``x`` of shape ``(n_chains, n_draws)``."""
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    w = float(np.var(x, axis=1, ddof=1).mean())
    b = n * float(np.var(x.mean(axis=1), ddof=1))
    if w <= 0:
        return float("inf") if b > 0 else float("nan")
    return float(np.sqrt(((n - 1.0) / n * w + b / n) / w))


def _validate_chains(draws: np.ndarray, fn_name: str, *, min_chains: int = 2) -> np.ndarray:
    """Return a finite ``(n_chains, n_draws)`` matrix or reject an invalid diagnostic input.

    ``min_chains`` is 2 for R-hat, which compares independent chains and is meaningless with one.
    The ESS estimators pass 1: they consume :func:`_split_chains` output, so a single long chain
    becomes two half-chains -- the standard split-ESS construction -- and a single-chain sampler is
    a perfectly ordinary thing to want an effective sample size for.
    """
    try:
        x = np.asarray(draws, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{fn_name}(): draws must be a rectangular numeric matrix.") from error
    if x.ndim != 2:
        raise ValueError(f"{fn_name}(): draws must have shape (n_chains, n_draws), got {x.shape}.")
    if x.shape[0] < min_chains:
        plural = "chain is" if min_chains == 1 else "independent chains are"
        raise ValueError(f"{fn_name}(): at least {min_chains} {plural} required, got {x.shape[0]}.")
    # Every caller splits each chain in half first, so a half must itself clear the four-draw floor.
    required = 4 if x.shape[0] >= 2 else 8
    if x.shape[1] < required:
        raise ValueError(f"{fn_name}(): at least {required} draws per chain are required, got {x.shape[1]}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{fn_name}(): draws must be finite (no NaN or Inf).")
    return x


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split each validated chain into equal first and last halves."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def split_rhat(draws: np.ndarray) -> float:
    """Maximum rank-normalized and folded split-R-hat for one parameter.

    ``draws`` must be a finite ``(n_chains, n_draws)`` matrix with at least two chains and four draws
    per chain. Splitting catches within-chain non-stationarity; folding about the pooled median also
    catches scale mismatch. Values within ~0.01 of 1.0 indicate convergence; > 1.01 is a warning.
    Constant chains have no estimable scale and return ``NaN`` rather than a healthy-looking value.
    """
    split = _split_chains(_validate_chains(draws, "split_rhat"))
    if np.ptp(split) == 0:
        return float("nan")
    rank_rhat = _classic_rhat(_rank_normalize(split).reshape(split.shape))
    folded = np.abs(split - np.median(split))
    folded_rhat = _classic_rhat(_rank_normalize(folded).reshape(split.shape))
    if np.isnan(rank_rhat) and np.isnan(folded_rhat):
        return float("nan")
    return float(np.fmax(rank_rhat, folded_rhat))


def bulk_ess(draws: np.ndarray) -> float:
    """Bulk ESS of split rank-normalized draws; constant chains return ``NaN``."""
    split = _split_chains(_validate_chains(draws, "bulk_ess", min_chains=1))
    if np.ptp(split) == 0:
        return float("nan")
    return _ess_chains(_rank_normalize(split).reshape(split.shape))


def tail_ess(draws: np.ndarray) -> float:
    """Tail ESS from split 5% and 95% quantile indicators; unavailable cases return ``NaN``."""
    split = _split_chains(_validate_chains(draws, "tail_ess", min_chains=1))
    q05, q95 = np.quantile(split, 0.05), np.quantile(split, 0.95)
    lower = _ess_chains((split <= q05).astype(float))
    upper = _ess_chains((split >= q95).astype(float))
    return float(min(lower, upper))


def convergence_diagnostics(draws: np.ndarray) -> dict:
    """Return modern convergence metrics plus an explicit availability receipt.

    Every unavailability is a RECEIPT, not an exception: a metric that cannot be computed for these
    draws comes back ``NaN``, is listed under ``unavailable``, and gets a plain-language entry in
    ``unavailable_because``. In particular a single chain -- the default configuration of every
    sampling route -- receipts ``split_rhat`` (which compares independent chains) while still
    returning the finite single-chain ``bulk_ess``/``tail_ess`` estimates, instead of raising the
    2-chain ``ValueError`` for one cause and receipting the other (t5 wave-3). Malformed input
    (wrong shape, NaN/Inf draws, too few draws to split) still raises.
    """
    x = _validate_chains(draws, "convergence_diagnostics", min_chains=1)
    because = {}
    if x.shape[0] < 2:
        rhat_value = float("nan")
        because["split_rhat"] = (
            "split_rhat compares at least 2 independent chains, got 1 -- refit with chains=2 or more"
        )
    else:
        rhat_value = split_rhat(x)
    metrics = {"split_rhat": rhat_value, "bulk_ess": bulk_ess(x), "tail_ess": tail_ess(x)}
    unavailable = [name for name, value in metrics.items() if not np.isfinite(value)]
    for name in unavailable:
        because.setdefault(name, "the draws are constant; no scale is estimable")
    return {
        **metrics,
        "available": not unavailable,
        "unavailable": unavailable,
        "unavailable_because": {name: because[name] for name in unavailable},
        "status": "available" if not unavailable else "unavailable",
    }


def _validate_loglik(loglik: np.ndarray, fn_name: str) -> np.ndarray:
    """Validate a strict 2-D pointwise log-likelihood matrix.

    An empty matrix (zero draws and/or zero observations) sums to ``0.0`` in both ``waic`` and
    ``psis_loo`` below -- indistinguishable from "a model with a perfect, trivial fit" rather than
    "no data was actually provided" -- and a NaN/Inf entry propagates silently into the returned
    dict (a NaN comparison is always False, so a downstream ``compare()`` ranking would silently
    misorder models instead of erroring). Both are rejected here, at the shared entry point, rather
    than letting either function return a plausible-looking placeholder.
    """
    try:
        ll = np.asarray(loglik, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{fn_name}(): loglik must be a rectangular numeric matrix.") from error
    if ll.ndim != 2:
        raise ValueError(f"{fn_name}(): loglik must have shape (n_draws, n_obs), got {ll.shape}.")
    if ll.shape[0] == 0 or ll.shape[1] == 0:
        raise ValueError(
            f"{fn_name}(): loglik must be a non-empty (n_draws, n_obs) log-likelihood matrix, got shape {ll.shape}."
        )
    if ll.shape[0] < 2:
        raise ValueError(
            f"{fn_name}(): at least two genuine posterior draws are required; "
            "a one-row plug-in log likelihood is not a Bayesian predictive diagnostic."
        )
    if not np.isfinite(ll).all():
        raise ValueError(f"{fn_name}(): loglik must be finite (no NaN or Inf).")
    return ll


def waic(loglik: np.ndarray) -> dict:
    """Return the WAIC of a ``(n_draws, n_obs)`` pointwise log-likelihood matrix."""
    loglik = _validate_loglik(loglik, "waic")
    s, n = loglik.shape
    lppd_i = _lppd_pointwise(loglik)
    p_waic_i = np.var(loglik, axis=0, ddof=1)
    elpd_i = lppd_i - p_waic_i
    elpd = float(np.sum(elpd_i))
    se = float(2.0 * np.sqrt(n * np.var(elpd_i, ddof=1))) if n > 1 else float("nan")
    return {
        "elpd_waic": elpd,
        "p_waic": float(np.sum(p_waic_i)),
        "waic": -2.0 * elpd,
        "se": se,
        "n_draws": s,
        "pointwise": elpd_i,
    }


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Fit a generalized Pareto distribution to positive exceedances ``x`` (Zhang & Stephens 2009)."""
    x = np.sort(x)
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior_bs * x[int(np.ceil(n / 4.0)) - 1]
    bs += 1.0 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x[None, :]), axis=1)
    log_lik = n * (np.log(-bs / ks) - ks - 1.0)
    weights = np.exp(log_lik - _logsumexp(log_lik))
    b = float(np.sum(bs * weights))
    k = float(np.mean(np.log1p(-b * x)))
    sigma = -k / b
    # weakly informative prior shrinking k toward 0.5
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    return k, sigma


def _gpd_quantile(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """Quantile function of the generalized Pareto distribution (shape k, scale sigma)."""
    if abs(k) < 1.0e-8:
        return sigma * -np.log1p(-p)
    return sigma / k * (np.power(1.0 - p, -k) - 1.0)


def _psis_smooth(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth a 1-D array of log importance weights; return (smoothed log weights, khat)."""
    lw = np.asarray(log_weights, dtype=float).copy()
    s = len(lw)
    lw -= np.max(lw)  # stabilize
    m = int(min(0.2 * s, 3.0 * np.sqrt(s)))
    if m < 5 or s < 25:
        return lw, float("nan")  # too few draws to estimate a tail reliably

    order = np.argsort(lw)
    tail_idx = order[-m:]
    cutoff = lw[order[-m - 1]]  # log threshold below the tail
    exceedances = np.exp(lw[tail_idx]) - np.exp(cutoff)
    if np.any(exceedances <= 0.0):
        return lw, float("nan")

    k, sigma = _gpdfit(exceedances)
    # replace the tail by the smoothed expected order statistics from the fitted GPD
    probs = (np.arange(m) + 0.5) / m
    smoothed = np.log(_gpd_quantile(probs, k, sigma) + np.exp(cutoff))
    # tail_idx is ordered by ascending lw; smoothed is ascending -> assign in that order
    lw[tail_idx] = np.minimum(smoothed, 0.0)  # truncate at the (stabilized) max weight of 0
    return lw, k


def psis_loo(loglik: np.ndarray) -> dict:
    """Return PSIS-LOO of a ``(n_draws, n_obs)`` pointwise log-likelihood matrix."""
    loglik = _validate_loglik(loglik, "psis_loo")
    s, n = loglik.shape
    if s < 2:
        raise ValueError("psis_loo(): at least two posterior draws are required for importance sampling.")

    elpd_i = np.empty(n)
    khat = np.empty(n)
    for i in range(n):
        ll = loglik[:, i]
        lw, k = _psis_smooth(-ll)  # LOO importance weights are proportional to 1 / p(y_i | theta)
        elpd_i[i] = _logsumexp(lw + ll) - _logsumexp(lw)
        khat[i] = k

    elpd = float(np.sum(elpd_i))
    p_loo = float(np.sum(_lppd_pointwise(loglik)) - elpd)
    se = float(2.0 * np.sqrt(n * np.var(elpd_i, ddof=1))) if n > 1 else float("nan")
    return {
        "elpd_loo": elpd,
        "p_loo": p_loo,
        "loo": -2.0 * elpd,
        "se": se,
        "khat_max": float(np.nanmax(khat)) if np.any(np.isfinite(khat)) else float("nan"),
        "n_draws": s,
        "pointwise": elpd_i,
    }


def loo(loglik: np.ndarray) -> dict:
    """PSIS-LOO under its conventional short name (see :func:`psis_loo`)."""
    return psis_loo(loglik)


def _validate_stacking_controls(iters: int, tol: float) -> tuple[int, float]:
    if isinstance(iters, (bool, np.bool_)) or not isinstance(iters, Integral) or iters < 1:
        raise ValueError(f"loo_stacking_weights(): iters must be a positive integer, got {iters!r}.")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real) or not np.isfinite(tol) or tol < 0:
        raise ValueError(f"loo_stacking_weights(): tol must be a finite non-negative number, got {tol!r}.")
    return int(iters), float(tol)


def _validate_stacking_lpd(pointwise_lpd: np.ndarray) -> np.ndarray:
    try:
        lpd = np.asarray(pointwise_lpd, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("loo_stacking_weights(): pointwise_lpd must be a rectangular numeric matrix.") from error
    if lpd.ndim != 2:
        raise ValueError(f"loo_stacking_weights(): pointwise_lpd must have shape (n_obs, n_models), got {lpd.shape}.")
    if lpd.shape[0] == 0 or lpd.shape[1] == 0:
        raise ValueError("loo_stacking_weights(): pointwise_lpd must contain at least one observation and one model.")
    if not np.isfinite(lpd).all():
        raise ValueError("loo_stacking_weights(): pointwise_lpd must be finite (no NaN or Inf).")
    return lpd


def _loo_stacking_fit(pointwise_lpd: np.ndarray, iters: int, tol: float) -> dict:
    """Optimize LOO stacking weights and return weights with an auditable optimizer receipt."""
    lpd = _validate_stacking_lpd(pointwise_lpd)
    iters, tol = _validate_stacking_controls(iters, tol)
    _, k = lpd.shape
    if k == 1:
        return {
            "weights": np.ones(1),
            "converged": True,
            "iterations": 0,
            "objective": float(np.sum(lpd[:, 0])),
            "reason": "single_model",
        }

    w = np.full(k, 1.0 / k)

    def _objective(weights: np.ndarray) -> float:
        log_weights = np.full(k, -np.inf)
        positive = weights > 0
        log_weights[positive] = np.log(weights[positive])
        return float(np.sum(_logsumexp(lpd + log_weights[None, :], axis=1)))

    objective = _objective(w)
    converged = False
    iterations = 0
    for iteration in range(1, iters + 1):
        log_weights = np.full(k, -np.inf)
        positive = w > 0
        log_weights[positive] = np.log(w[positive])
        log_joint = lpd + log_weights[None, :]
        log_mixture = _logsumexp(log_joint, axis=1)
        w_new = np.exp(log_joint - log_mixture[:, None]).mean(axis=0)
        w_new /= w_new.sum()
        new_objective = _objective(w_new)
        iterations = iteration
        if abs(new_objective - objective) <= tol * (abs(objective) + 1.0):
            w = w_new
            objective = new_objective
            converged = True
            break
        w = w_new
        objective = new_objective
    return {
        "weights": w,
        "converged": converged,
        "iterations": iterations,
        "objective": objective,
        "reason": "tolerance" if converged else "iteration_limit",
    }


def loo_stacking_weights(
    pointwise_lpd: np.ndarray,
    iters: int = 2000,
    tol: float = 1.0e-10,
    *,
    return_result: bool = False,
) -> np.ndarray | dict:
    """Return LOO stacking weights (Yao, Vehtari, Simpson & Gelman, 2018).

    ``pointwise_lpd`` is an ``(n_obs, K)`` matrix of per-model pointwise LOO log-predictive
    densities (each column is ``psis_loo(model_k)["pointwise"]``). The returned simplex weights
    ``w`` maximize the LOO log-score of the weighted predictive distribution,
    ``sum_i log(sum_k w_k * exp(lpd_ik))``. This is concave in ``w`` and solved here by the standard
    mixture-weight EM update (no external optimizer), which respects the simplex by construction.
    By default this compatibility surface returns only the weights. Set ``return_result=True`` to
    receive the weights together with convergence, iteration, objective, and termination metadata.
    """
    if not isinstance(return_result, (bool, np.bool_)):
        raise ValueError(f"loo_stacking_weights(): return_result must be boolean, got {return_result!r}.")
    result = _loo_stacking_fit(pointwise_lpd, iters, tol)
    return result if return_result else result["weights"]


def loo_stack(logliks: Sequence[np.ndarray], *, iters: int = 2000, tol: float = 1.0e-10) -> dict:
    """Stack K candidate models by LOO predictive performance.

    ``logliks`` is a sequence of ``(n_draws_k, n_obs)`` pointwise log-likelihood matrices over the
    same, aligned observations. Returns the stacking ``weights``, the ``(n_obs, K)`` per-model
    pointwise LOO densities, each model's ``elpd_loo``, the actually achieved ``stacked_elpd_loo``,
    and an optimizer receipt. A finite iteration limit need not reach the exact optimum.
    """
    _validate_stacking_controls(iters, tol)
    if isinstance(logliks, np.ndarray) or not isinstance(logliks, Sequence) or len(logliks) == 0:
        raise ValueError("loo_stack(): logliks must be a non-empty sequence of log-likelihood matrices.")
    loo_results = [psis_loo(ll) for ll in logliks]
    observation_counts = {len(result["pointwise"]) for result in loo_results}
    if len(observation_counts) != 1:
        raise ValueError("loo_stack(): every model must describe the same number of aligned observations.")
    pointwise = np.column_stack([result["pointwise"] for result in loo_results])
    optimization = _loo_stacking_fit(pointwise, iters, tol)
    weights = optimization["weights"]
    stacked = float(optimization["objective"])
    model_elpds = [float(pointwise[:, j].sum()) for j in range(pointwise.shape[1])]
    return {
        "weights": weights,
        "pointwise": pointwise,
        "model_elpd_loo": model_elpds,
        "stacked_elpd_loo": stacked,
        "best_model_elpd_loo": max(model_elpds),
        "objective_gap_from_best": stacked - max(model_elpds),
        "converged": optimization["converged"],
        "iterations": optimization["iterations"],
        "objective": optimization["objective"],
        "termination_reason": optimization["reason"],
    }
