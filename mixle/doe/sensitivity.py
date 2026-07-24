"""Global sensitivity analysis: variance-based Sobol indices and Morris screening.

Which inputs actually drive a model's output? Sobol indices decompose the output variance into the
contribution of each input (first order) and of each input including all its interactions (total order).
Morris screening is a cheaper one-at-a-time elementary-effects method for an initial factor ranking.
These tell you which survey parameters / forcings to refine and which to fix -- the front of the UQ loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.doe.designs import _as_bounds, _qmc_unit, _require_exact_positive_int

__all__ = ["sobol_indices", "morris_screening", "fast_indices", "dgsm"]


def _scale(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map points in the unit cube to the box ``bounds`` (d, 2)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)


def _sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    """``n`` low-discrepancy points in ``[0,1]^d`` via the shared DoE QMC engine.

    Reuses ``designs._qmc_unit`` (scrambled Sobol', stratified-random fallback on older scipy) so
    the sensitivity sampler and the DoE designs draw from one source.
    """
    try:
        from scipy.stats import qmc

        return _qmc_unit(qmc.Sobol, d, n, True, np.random.RandomState(seed))
    except Exception:  # pragma: no cover - qmc fallback  # noqa: BLE001
        return np.random.RandomState(seed).random((n, d))


def _validate_names(names: Sequence[str] | None, d: int) -> list[str]:
    """Validate optional per-input ``names``: exactly ``d`` entries when given, else ``x0..x{d-1}``.

    Shared by every estimator in this module (MXR-080-0192): a caller-supplied name list whose
    length does not match the model's dimensionality is rejected here, once, instead of being
    silently accepted and then silently mis-zipped against the per-input results downstream.
    """
    if names is None:
        return [f"x{i}" for i in range(d)]
    names_list = list(names)
    if len(names_list) != d:
        raise ValueError(f"names must have exactly {d} entries (one per dimension), got {len(names_list)}.")
    return names_list


def _eval_model(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, *, label: str = "func") -> np.ndarray:
    """Call the vectorized model ``func`` at the ``(m, d)`` points ``x`` and validate its output.

    Every estimator in this module treats ``func`` as ``f(X) -> y`` mapping an ``(m, d)`` batch of
    input rows to an ``(m,)`` array of scalar outputs, and this is the single place that call is
    made (MXR-080-0192): a model that returns the wrong number of outputs or a non-finite value is
    caught immediately at the call site instead of silently corrupting a downstream variance or
    elementary-effect computation via a shape-broadcast or a propagated NaN/inf.
    """
    y = np.asarray(func(x), dtype=float).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"{label} must return one output per input row: called with {x.shape[0]} row(s), got "
            f"{y.shape[0]} output(s)."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{label} returned a non-finite value (inf or nan); every model output must be finite.")
    return y


def sobol_indices(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 4096,
    *,
    seed: int = 0,
    names: Sequence[str] | None = None,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """First- and total-order Sobol sensitivity indices (Saltelli sampling, Jansen estimators).

    Args:
        func: a *vectorized* model ``f(X) -> y`` mapping an ``(m, d)`` array of inputs to an ``(m,)``
            array of scalar outputs.
        bounds: ``[(lo, hi), ...]`` per input -- the input is taken uniform on the box.
        n: base sample size; the total number of model evaluations is ``n * (d + 2)`` (bootstrap
            resampling below is free of additional ``func`` calls, so this count is exact).
        seed: RNG seed for the (Sobol) base samples.
        names: optional input names for the returned dict.
        n_bootstrap: number of bootstrap resamples used for the standard error / confidence interval
            of each index (see "Returns" below).
        confidence: confidence level for the bootstrap interval, in ``(0, 1)`` -- e.g. ``0.95`` for a
            two-sided 95% interval.

    Returns:
        A dict with, for both first- and total-order:

        - ``'S1'`` / ``'ST'``: the **raw, unclipped** point estimate. A finite-sample Sobol estimator
          can legitimately land slightly negative, or (for ``S1``) slightly above 1, purely from Monte
          Carlo noise -- most often when the true index is near 0 or 1. That is not a bug in the
          estimate; it is the honest signal that sampling noise is comparable to the index's true
          magnitude, and this raw value is always what is returned here (never silently clamped).
        - ``'S1_clipped'`` / ``'ST_clipped'``: the same estimate clipped into its theoretically valid
          range (``S1`` into ``[0, 1]``, ``ST`` below at ``0``) -- a separate, explicitly-named
          convenience for e.g. plotting or reporting a single non-negative "share of variance"; never
          the only value returned, and never used in place of the raw estimate above.
        - ``'S1_standard_error'`` / ``'ST_standard_error'``: a bootstrap standard error, from
          ``n_bootstrap`` resamples (with replacement) of the already-evaluated ``A``/``B``/``AB_i``
          rows, each re-run through the same S1/ST formula -- the standard uncertainty-quantification
          approach for Sobol' estimators (Archer, Saltelli & Sobol' 1997), since no exact closed-form
          sampling distribution exists for them in general.
        - ``'S1_ci_low'`` / ``'S1_ci_high'`` and ``'ST_ci_low'`` / ``'ST_ci_high'``: the corresponding
          percentile bootstrap confidence interval at the ``confidence`` level.

        Plus ``'names'`` and ``'var'`` (the total output variance). ``S1[i]`` is the fraction of
        output variance from input ``i`` alone; ``ST[i]`` includes all interactions involving ``i``
        (so ``ST[i] - S1[i]`` measures ``i``'s interaction strength, and ``ST[i] ~ 0`` -- relative to
        its standard error, not just numerically close -- means input ``i`` can be fixed).
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    n = _require_exact_positive_int(n, "n")
    n_bootstrap = _require_exact_positive_int(n_bootstrap, "n_bootstrap")
    if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")
    names_out = _validate_names(names, d)
    a_unit = _sobol_unit(n, 2 * d, seed)  # split one 2d-dimensional Sobol block into A and B (independence)
    a, b = a_unit[:, :d], a_unit[:, d:]
    ya = _eval_model(func, _scale(a, bounds), label="sobol_indices's func")
    yb = _eval_model(func, _scale(b, bounds), label="sobol_indices's func")
    var = np.var(np.concatenate([ya, yb]))
    if var <= 0:  # constant output: every index is exactly (not just clipped-to-) zero, with no
        zero = np.zeros(d)  # sampling uncertainty to quantify -- there is nothing left to estimate.
        return {
            "S1": zero,
            "ST": zero.copy(),
            "S1_clipped": zero.copy(),
            "ST_clipped": zero.copy(),
            "S1_standard_error": zero.copy(),
            "ST_standard_error": zero.copy(),
            "S1_ci_low": zero.copy(),
            "S1_ci_high": zero.copy(),
            "ST_ci_low": zero.copy(),
            "ST_ci_high": zero.copy(),
            "names": names_out,
            "var": 0.0,
        }
    s1 = np.zeros(d)
    st = np.zeros(d)
    yab_all = np.empty((d, n))  # stashed for the bootstrap pass below -- no extra `func` calls needed
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]  # A with column i taken from B
        yab = _eval_model(func, _scale(ab, bounds), label="sobol_indices's func")
        yab_all[i] = yab
        s1[i] = np.mean(yb * (yab - ya)) / var  # Saltelli 2010 first-order estimator
        st[i] = 0.5 * np.mean((ya - yab) ** 2) / var  # Jansen total-order estimator

    # Bootstrap standard errors and confidence intervals: resample the *already-evaluated* rows with
    # replacement and recompute S1/ST on each resample, entirely in numpy (no additional model
    # evaluations). The resampling RNG is derived from `seed` via a SeedSequence so it is reproducible
    # but statistically independent of the RandomState `_sobol_unit` seeded directly from `seed` above.
    boot_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    boot_rng = np.random.RandomState(boot_seed)
    idx = boot_rng.randint(0, n, size=(n_bootstrap, n))
    ya_boot, yb_boot = ya[idx], yb[idx]  # each (n_bootstrap, n)
    var_boot = np.var(np.concatenate([ya_boot, yb_boot], axis=1), axis=1)  # (n_bootstrap,)
    safe_var_boot = np.where(var_boot > 0, var_boot, np.nan)  # a degenerate all-tied resample -> NaN,
    s1_boot = np.empty((n_bootstrap, d))  # excluded below by the nan-aware reductions, not divided by 0
    st_boot = np.empty((n_bootstrap, d))
    for i in range(d):
        yab_boot = yab_all[i][idx]
        s1_boot[:, i] = np.mean(yb_boot * (yab_boot - ya_boot), axis=1) / safe_var_boot
        st_boot[:, i] = 0.5 * np.mean((ya_boot - yab_boot) ** 2, axis=1) / safe_var_boot
    alpha = 1.0 - confidence
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    s1_ci = np.nanpercentile(s1_boot, [lo_q, hi_q], axis=0)  # (2, d)
    st_ci = np.nanpercentile(st_boot, [lo_q, hi_q], axis=0)

    return {
        "S1": s1,
        "ST": st,
        "S1_clipped": np.clip(s1, 0.0, 1.0),
        "ST_clipped": np.clip(st, 0.0, None),
        "S1_standard_error": np.nanstd(s1_boot, axis=0, ddof=1),
        "ST_standard_error": np.nanstd(st_boot, axis=0, ddof=1),
        "S1_ci_low": s1_ci[0],
        "S1_ci_high": s1_ci[1],
        "ST_ci_low": st_ci[0],
        "ST_ci_high": st_ci[1],
        "names": names_out,
        "var": float(var),
    }


def morris_screening(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    *,
    trajectories: int = 20,
    levels: int = 4,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Morris elementary-effects screening -- a low-cost first factor ranking.

    Walks ``trajectories`` one-factor-at-a-time paths on a ``levels``-grid; the mean absolute elementary
    effect ``mu_star[i]`` ranks influence and the spread ``sigma[i]`` flags nonlinearity/interactions.
    Cost: ``trajectories * (d + 1)`` evaluations -- far fewer than Sobol, for an initial screen.
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    trajectories = _require_exact_positive_int(trajectories, "trajectories")
    levels = _require_exact_positive_int(levels, "levels", minimum=2)
    names_out = _validate_names(names, d)
    rng = np.random.RandomState(seed)
    delta = levels / (2.0 * (levels - 1))  # the standard Morris step on the unit grid
    grid = np.linspace(0.0, 1.0, levels)
    effects = [[] for _ in range(d)]
    for _ in range(trajectories):
        base = rng.choice(grid[: levels // 2 + 1] if levels > 1 else grid, size=d)  # room to step up by delta
        order = rng.permutation(d)
        x = base.copy()
        y_prev = _eval_model(func, _scale(x[None, :], bounds), label="morris_screening's func")[0]
        for i in order:
            x_next = x.copy()
            x_next[i] = min(x_next[i] + delta, 1.0)
            y_next = _eval_model(func, _scale(x_next[None, :], bounds), label="morris_screening's func")[0]
            step = x_next[i] - x[i]
            if step != 0:
                effects[i].append((y_next - y_prev) / step)
            x, y_prev = x_next, y_next
    mu_star = np.array([np.mean(np.abs(e)) if e else 0.0 for e in effects])
    sigma = np.array([np.std(e) if e else 0.0 for e in effects])
    return {
        "mu_star": mu_star,
        "sigma": sigma,
        "names": names_out,
    }


def fast_indices(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 600,
    *,
    harmonics: int = 6,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """First-order sensitivity indices via Random Balance Designs FAST (RBD-FAST).

    A Fourier alternative to :func:`sobol_indices` for the *first-order* indices: every input is driven
    along the same triangle-wave search curve but under an independent random permutation, the model is
    evaluated once over the ``n`` points, and for each input the output -- reordered along that input's
    curve -- has its variance concentrated at the base frequency's first ``harmonics`` harmonics. The
    ratio of that power to the total is the first-order index (with the Tarantola bias correction). Cost
    is a single batch of ``n`` evaluations, independent of dimension.

    Returns ``{'S1': (d,), 'names': [...], 'var': float}``.
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    n = _require_exact_positive_int(n, "n")
    m = _require_exact_positive_int(harmonics, "harmonics")
    names_out = _validate_names(names, d)
    rng = np.random.RandomState(seed)
    s = np.linspace(-np.pi, np.pi, n, endpoint=False)
    base = 0.5 + np.arcsin(np.sin(s)) / np.pi  # triangle wave, uniform on [0, 1]
    perms = [rng.permutation(n) for _ in range(d)]
    x = np.column_stack([base[perms[i]] for i in range(d)])
    y = _eval_model(func, _scale(x, bounds), label="fast_indices's func")
    s1 = np.zeros(d)
    var = float(np.var(y))
    if var <= 0:
        return {"S1": s1, "names": names_out, "var": 0.0}
    if 2.0 * m >= n - 1:
        # the Tarantola correction's denominator (1 - 2m/(n-1)) must stay strictly positive; once
        # 2m/(n-1) reaches or exceeds 1, the correction denominator hits zero or goes negative,
        # flipping the sign of a subsequent nonsensical S1 rather than raising. clip(0, 1) at the
        # end would otherwise silently mask this as an ordinary "no sensitivity" result.
        raise ValueError(
            f"fast_indices requires 2*harmonics < n-1 for a well-posed Tarantola correction "
            f"(harmonics={m}, n={n}); increase n or lower harmonics."
        )
    for i in range(d):
        yi = y[np.argsort(perms[i])]  # reorder output along input i's search-curve coordinate
        spectrum = np.abs(np.fft.rfft(yi - yi.mean())) ** 2
        total = float(spectrum[1:].sum())
        raw = float(spectrum[1 : m + 1].sum()) / total if total > 0 else 0.0
        # Tarantola (2006) bias correction: an uninformative input has expected raw ~ 2m/(n-1).
        s1[i] = (raw - 2.0 * m / (n - 1)) / (1.0 - 2.0 * m / (n - 1))
    return {"S1": np.clip(s1, 0.0, 1.0), "names": names_out, "var": var}


def dgsm(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 1024,
    *,
    seed: int = 0,
    rel_step: float = 1.0e-4,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Derivative-based global sensitivity measures (DGSM): mean squared partial derivatives.

    ``nu[i] = E[(df/dx_i)^2]`` over the input box, estimated by central finite differences at ``n``
    low-discrepancy points. Unlike the first-order Sobol index, a DGSM is nonzero whenever an input
    matters *anywhere* -- including purely through interactions -- so it is a low-cost, robust screen that
    upper-bounds the total Sobol index (Sobol & Kucherenko, via the Poincare inequality:
    ``ST[i] <= (L_i / pi)^2 * nu[i] / Var(y)`` for a uniform input of width ``L_i``). The reported
    ``importance`` is ``L_i^2 * nu[i]`` normalized to sum to one -- a dimensionless influence ranking.

    Returns ``{'nu': (d,), 'importance': (d,), 'names': [...]}``.
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    n = _require_exact_positive_int(n, "n")
    names_out = _validate_names(names, d)
    x = _scale(_sobol_unit(n, d, seed), bounds)
    span = bounds[:, 1] - bounds[:, 0]
    nu = np.zeros(d)
    for i in range(d):
        h = rel_step * span[i]
        xp = x.copy()
        xm = x.copy()
        xp[:, i] = np.minimum(x[:, i] + h, bounds[i, 1])
        xm[:, i] = np.maximum(x[:, i] - h, bounds[i, 0])
        step = xp[:, i] - xm[:, i]
        yp = _eval_model(func, xp, label="dgsm's func")
        ym = _eval_model(func, xm, label="dgsm's func")
        nu[i] = float(np.mean(((yp - ym) / np.where(step > 0, step, 1.0)) ** 2))
    weighted = span**2 * nu
    total = float(weighted.sum())
    importance = weighted / total if total > 0 else np.zeros(d)
    return {"nu": nu, "importance": importance, "names": names_out}
