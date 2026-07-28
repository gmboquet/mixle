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


def _sobol_unit(n: int, d: int, seed: int) -> tuple[np.ndarray, str]:
    """``n`` low-discrepancy points in ``[0,1]^d`` via the shared DoE QMC engine.

    Reuses ``designs._qmc_unit`` (scrambled Sobol', stratified-random fallback on older scipy) so
    the sensitivity sampler and the DoE designs draw from one source.
    """
    try:
        from scipy.stats import qmc
    except ImportError:  # pragma: no cover - supported only for minimal installations without scipy QMC
        return np.random.RandomState(seed).random((n, d)), "pseudorandom_fallback"
    return _qmc_unit(qmc.Sobol, d, n, True, np.random.RandomState(seed)), "scrambled_sobol"


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
    raw = np.asarray(func(x), dtype=float)
    if raw.ndim == 1 and raw.shape[0] == x.shape[0]:
        y = raw
    elif raw.ndim == 2 and raw.shape == (x.shape[0], 1):
        y = raw[:, 0]
    else:
        raise ValueError(
            f"{label} must return shape ({x.shape[0]},) or ({x.shape[0]}, 1), preserving the leading "
            f"sample axis; got shape {raw.shape}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{label} returned a non-finite value (inf or nan); every model output must be finite.")
    return y


def _require_finite_derived(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    invalid = ~np.isfinite(array)
    if np.any(invalid):
        if array.ndim == 0:
            location = ""
        else:
            location = f" at index {tuple(int(i) for i in np.argwhere(invalid)[0])}"
        raise ValueError(f"{name}{location} is not representable as finite float64.")
    return array


def _validate_confidence(confidence: Any) -> float:
    if isinstance(confidence, (bool, np.bool_)):
        raise ValueError(f"confidence must be a finite scalar in (0, 1), got {confidence!r}.")
    array = np.asarray(confidence)
    if array.ndim != 0:
        raise ValueError(f"confidence must be a finite scalar in (0, 1), got shape {array.shape}.")
    try:
        value = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence must be a finite scalar in (0, 1), got {confidence!r}.") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")
    return value


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

        Plus ``'names'``, ``'var'`` (the total output variance), and ``'sampling_method'`` (normally
        ``'scrambled_sobol'``; ``'pseudorandom_fallback'`` only when SciPy's QMC capability is absent).
        ``S1[i]`` is the fraction of
        output variance from input ``i`` alone; ``ST[i]`` includes all interactions involving ``i``
        (so ``ST[i] - S1[i]`` measures ``i``'s interaction strength, and ``ST[i] ~ 0`` -- relative to
        its standard error, not just numerically close -- means input ``i`` can be fixed).
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    n = _require_exact_positive_int(n, "n")
    n_bootstrap = _require_exact_positive_int(n_bootstrap, "n_bootstrap", minimum=2)
    confidence = _validate_confidence(confidence)
    names_out = _validate_names(names, d)
    a_unit, sampling_method = _sobol_unit(
        n, 2 * d, seed
    )  # split one 2d-dimensional Sobol block into A and B
    a, b = a_unit[:, :d], a_unit[:, d:]
    ya = _eval_model(func, _scale(a, bounds), label="sobol_indices's func")
    yb = _eval_model(func, _scale(b, bounds), label="sobol_indices's func")
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.var(np.concatenate([ya, yb])))
    _require_finite_derived(var, "sobol_indices output variance")
    if var <= 0:
        # Constant output: every index is exactly (not just clipped-to-) zero, with no sampling
        # uncertainty to quantify -- there is nothing left to estimate.
        zero = np.zeros(d)
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
            "sampling_method": sampling_method,
        }
    s1 = np.zeros(d)
    st = np.zeros(d)
    yab_all = np.empty((d, n))  # stashed for the bootstrap pass below -- no extra `func` calls needed
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]  # A with column i taken from B
        yab = _eval_model(func, _scale(ab, bounds), label="sobol_indices's func")
        yab_all[i] = yab
        with np.errstate(over="ignore", invalid="ignore"):
            s1[i] = np.mean(yb * (yab - ya)) / var  # Saltelli 2010 first-order estimator
            st[i] = 0.5 * np.mean((ya - yab) ** 2) / var  # Jansen total-order estimator
        _require_finite_derived(s1[i], f"sobol_indices S1[{i}]")
        _require_finite_derived(st[i], f"sobol_indices ST[{i}]")

    # Bootstrap standard errors and confidence intervals: resample the *already-evaluated* rows with
    # replacement and recompute S1/ST on each resample, entirely in numpy (no additional model
    # evaluations). The resampling RNG is derived from `seed` via a SeedSequence so it is reproducible
    # but statistically independent of the RandomState `_sobol_unit` seeded directly from `seed` above.
    boot_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    boot_rng = np.random.RandomState(boot_seed)
    idx = boot_rng.randint(0, n, size=(n_bootstrap, n))
    ya_boot, yb_boot = ya[idx], yb[idx]  # each (n_bootstrap, n)
    with np.errstate(over="ignore", invalid="ignore"):
        var_boot = np.var(np.concatenate([ya_boot, yb_boot], axis=1), axis=1)
    valid_bootstrap = np.isfinite(var_boot) & (var_boot > 0.0)
    if np.count_nonzero(valid_bootstrap) < 2:
        raise ValueError("sobol_indices has fewer than two non-degenerate finite bootstrap resamples.")
    ya_boot = ya_boot[valid_bootstrap]
    yb_boot = yb_boot[valid_bootstrap]
    var_boot = var_boot[valid_bootstrap]
    s1_boot = np.empty((var_boot.size, d))
    st_boot = np.empty((var_boot.size, d))
    for i in range(d):
        yab_boot = yab_all[i][idx][valid_bootstrap]
        with np.errstate(over="ignore", invalid="ignore"):
            s1_boot[:, i] = np.mean(yb_boot * (yab_boot - ya_boot), axis=1) / var_boot
            st_boot[:, i] = 0.5 * np.mean((ya_boot - yab_boot) ** 2, axis=1) / var_boot
    _require_finite_derived(s1_boot, "sobol_indices bootstrap S1")
    _require_finite_derived(st_boot, "sobol_indices bootstrap ST")
    alpha = 1.0 - confidence
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    s1_ci = np.percentile(s1_boot, [lo_q, hi_q], axis=0)  # (2, d)
    st_ci = np.percentile(st_boot, [lo_q, hi_q], axis=0)
    s1_se = np.std(s1_boot, axis=0, ddof=1)
    st_se = np.std(st_boot, axis=0, ddof=1)
    _require_finite_derived(s1_ci, "sobol_indices S1 confidence interval")
    _require_finite_derived(st_ci, "sobol_indices ST confidence interval")
    _require_finite_derived(s1_se, "sobol_indices S1 standard error")
    _require_finite_derived(st_se, "sobol_indices ST standard error")

    return {
        "S1": s1,
        "ST": st,
        "S1_clipped": np.clip(s1, 0.0, 1.0),
        "ST_clipped": np.clip(st, 0.0, None),
        "S1_standard_error": s1_se,
        "ST_standard_error": st_se,
        "S1_ci_low": s1_ci[0],
        "S1_ci_high": s1_ci[1],
        "ST_ci_low": st_ci[0],
        "ST_ci_high": st_ci[1],
        "names": names_out,
        "var": float(var),
        "sampling_method": sampling_method,
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

    Walks ``trajectories`` randomized-direction one-factor-at-a-time paths on a ``levels``-point grid
    per dimension (Morris 1991; Campolongo, Cariboni & Saltelli 2007); the mean absolute elementary
    effect ``mu_star[i]`` ranks influence and the spread ``sigma[i]`` flags nonlinearity/interactions.
    Cost: ``trajectories * (d + 1)`` evaluations -- far fewer than Sobol, for an initial screen.

    Each trajectory walks the grid ``{0, 1/(p-1), 2/(p-1), ..., 1}`` (``p = levels``) and takes ``d``
    one-at-a-time steps, one per input in a random order, each changing exactly one coordinate by the
    fixed step ``delta = p / (2*(p-1)) = (p/2) / (p-1)`` -- i.e. exactly ``p/2`` grid points, chosen (the
    standard choice) so the step always lands exactly on another grid point. Both the visit *order* and
    each dimension's step *direction* (up by ``delta`` or down by ``delta``) are independently
    randomized per trajectory. The starting coordinate for a dimension stepping up is drawn only from
    the grid's lower half (indices ``0..p/2-1``, which by construction have room for a ``+delta`` move
    without leaving ``[0, 1]``); for a dimension stepping down, only from the upper half (indices
    ``p/2..p-1``, symmetric room for ``-delta``). Every step therefore lands on the grid and inside
    bounds *by construction* -- never clipped or silently shrunk to a smaller step -- and, because the
    two halves are equal-sized and a dimension's pre-step value is confined to its assigned half while
    its post-step value always lands in the other, every grid level is visited with equal (1/p)
    probability, marginally, by any one coordinate at any position along a trajectory: the design is
    unbiased across the input box, not concentrated away from either boundary.

    ``levels`` must be even, so the grid-index step ``levels // 2`` is exact (an odd level count has no
    step that both equals the standard ``delta`` formula and lands on a grid point).
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    trajectories = _require_exact_positive_int(trajectories, "trajectories")
    levels = _require_exact_positive_int(levels, "levels", minimum=2)
    if levels % 2 != 0:
        raise ValueError(
            f"morris_screening requires an even number of levels, so the standard step "
            f"delta = levels / (2*(levels-1)) lands exactly on a grid point; got levels={levels}."
        )
    names_out = _validate_names(names, d)
    rng = np.random.RandomState(seed)
    grid = np.linspace(0.0, 1.0, levels)
    idx_step = levels // 2  # grid-index step: an UP move adds this, a DOWN move subtracts it
    low_idx = np.arange(0, idx_step)  # starting indices with guaranteed room for an UP step
    high_idx = np.arange(idx_step, levels)  # ... and for a DOWN step, the symmetric other half
    effects: list[list[float]] = [[] for _ in range(d)]
    for _ in range(trajectories):
        order = rng.permutation(d)
        up = rng.randint(0, 2, size=d).astype(bool)  # this trajectory's per-dimension step direction
        n_up = int(np.count_nonzero(up))
        idx = np.empty(d, dtype=int)
        idx[up] = rng.choice(low_idx, size=n_up)
        idx[~up] = rng.choice(high_idx, size=d - n_up)
        x = grid[idx]
        y_prev = _eval_model(func, _scale(x[None, :], bounds), label="morris_screening's func")[0]
        for i in order:
            idx_next = idx.copy()
            idx_next[i] += idx_step if up[i] else -idx_step  # in [0, levels-1] by construction above
            x_next = grid[idx_next]
            y_next = _eval_model(func, _scale(x_next[None, :], bounds), label="morris_screening's func")[0]
            step = x_next[i] - x[i]  # exactly +delta or -delta -- never shrunk, never zero
            with np.errstate(over="ignore", invalid="ignore"):
                effect = (y_next - y_prev) / step
            _require_finite_derived(effect, f"morris_screening elementary effect for dimension {i}")
            effects[i].append(effect)
            idx, x, y_prev = idx_next, x_next, y_next
    with np.errstate(over="ignore", invalid="ignore"):
        mu_star = np.array([np.mean(np.abs(e)) for e in effects])
        sigma = np.array([np.std(e) for e in effects])
    _require_finite_derived(mu_star, "morris_screening mu_star")
    _require_finite_derived(sigma, "morris_screening sigma")
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
    n_bootstrap: int = 200,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """First-order sensitivity indices via Random Balance Designs FAST (RBD-FAST).

    A Fourier alternative to :func:`sobol_indices` for the *first-order* indices: every input is driven
    along the same triangle-wave search curve but under an independent random permutation, the model is
    evaluated once over the ``n`` points, and for each input the output -- reordered along that input's
    curve -- has its variance concentrated at the base frequency's first ``harmonics`` harmonics. The
    ratio of that power to the total is the first-order index (with the Tarantola bias correction). Cost
    is a single batch of ``n`` evaluations, independent of dimension.

    ``S1`` is the raw Tarantola-corrected finite-sample estimate. ``S1_clipped`` is a separate
    presentation convenience. Circular moving-block bootstrap fields report the raw estimator's
    standard error and percentile interval without additional model evaluations.
    """
    bounds = _as_bounds(bounds)
    d = bounds.shape[0]
    n = _require_exact_positive_int(n, "n")
    m = _require_exact_positive_int(harmonics, "harmonics")
    n_bootstrap = _require_exact_positive_int(n_bootstrap, "n_bootstrap", minimum=2)
    confidence = _validate_confidence(confidence)
    names_out = _validate_names(names, d)
    rng = np.random.RandomState(seed)
    s = np.linspace(-np.pi, np.pi, n, endpoint=False)
    base = 0.5 + np.arcsin(np.sin(s)) / np.pi  # triangle wave, uniform on [0, 1]
    perms = [rng.permutation(n) for _ in range(d)]
    x = np.column_stack([base[perms[i]] for i in range(d)])
    y = _eval_model(func, _scale(x, bounds), label="fast_indices's func")
    s1 = np.zeros(d)
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.var(y))
    _require_finite_derived(var, "fast_indices output variance")
    if var <= 0:
        return {
            "S1": s1,
            "S1_clipped": s1.copy(),
            "S1_standard_error": s1.copy(),
            "S1_ci_low": s1.copy(),
            "S1_ci_high": s1.copy(),
            "names": names_out,
            "var": 0.0,
            "uncertainty_method": "circular_block_bootstrap",
        }
    if 2.0 * m >= n - 1:
        # the Tarantola correction's denominator (1 - 2m/(n-1)) must stay strictly positive; once
        # 2m/(n-1) reaches or exceeds 1, the correction denominator hits zero or goes negative,
        # flipping the sign of a subsequent nonsensical S1 rather than raising. clip(0, 1) at the
        # end would otherwise silently mask this as an ordinary "no sensitivity" result.
        raise ValueError(
            f"fast_indices requires 2*harmonics < n-1 for a well-posed Tarantola correction "
            f"(harmonics={m}, n={n}); increase n or lower harmonics."
        )
    correction = 2.0 * m / (n - 1)

    def estimate_ordered(ordered: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            centered = ordered - np.mean(ordered)
        _require_finite_derived(centered, "fast_indices centered output")
        scale = float(np.max(np.abs(centered)))
        if scale == 0.0:
            return 0.0
        spectrum = np.abs(np.fft.rfft(centered / scale)) ** 2
        _require_finite_derived(spectrum, "fast_indices spectrum")
        total = float(spectrum[1:].sum())
        _require_finite_derived(total, "fast_indices spectral power")
        raw_power = float(spectrum[1 : m + 1].sum()) / total if total > 0.0 else 0.0
        estimate = (raw_power - correction) / (1.0 - correction)
        _require_finite_derived(estimate, "fast_indices raw S1")
        return estimate

    bootstrap = np.empty((n_bootstrap, d), dtype=np.float64)
    block_length = max(2, int(np.sqrt(n)))
    block_count = int(np.ceil(n / block_length))
    bootstrap_rng = np.random.RandomState(int(np.random.SeedSequence(seed).generate_state(1)[0]))
    offsets = np.arange(block_length)
    for i in range(d):
        yi = y[np.argsort(perms[i])]  # reorder output along input i's search-curve coordinate
        s1[i] = estimate_ordered(yi)
        for replicate in range(n_bootstrap):
            starts = bootstrap_rng.randint(0, n, size=block_count)
            indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
            bootstrap[replicate, i] = estimate_ordered(yi[indices])
    standard_error = np.std(bootstrap, axis=0, ddof=1)
    alpha = 1.0 - confidence
    interval = np.percentile(bootstrap, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)], axis=0)
    _require_finite_derived(standard_error, "fast_indices standard error")
    _require_finite_derived(interval, "fast_indices confidence interval")
    return {
        "S1": s1,
        "S1_clipped": np.clip(s1, 0.0, 1.0),
        "S1_standard_error": standard_error,
        "S1_ci_low": interval[0],
        "S1_ci_high": interval[1],
        "names": names_out,
        "var": var,
        "uncertainty_method": "circular_block_bootstrap",
    }


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
    if isinstance(rel_step, (bool, np.bool_)):
        raise ValueError(f"rel_step must be a finite strictly positive scalar, got {rel_step!r}.")
    rel_step_array = np.asarray(rel_step)
    if rel_step_array.ndim != 0:
        raise ValueError(f"rel_step must be a finite strictly positive scalar, got shape {rel_step_array.shape}.")
    try:
        rel_step = float(rel_step_array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rel_step must be a finite strictly positive scalar, got {rel_step!r}.") from exc
    if not np.isfinite(rel_step) or rel_step <= 0.0:
        raise ValueError(f"rel_step must be finite and strictly positive, got {rel_step!r}.")
    names_out = _validate_names(names, d)
    unit, sampling_method = _sobol_unit(n, d, seed)
    x = _scale(unit, bounds)
    span = bounds[:, 1] - bounds[:, 0]
    nu = np.zeros(d)
    for i in range(d):
        with np.errstate(over="ignore", invalid="ignore"):
            h = rel_step * span[i]
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(f"dgsm finite-difference scale for dimension {i} is not finite and strictly positive.")
        xp = x.copy()
        xm = x.copy()
        xp[:, i] = np.minimum(x[:, i] + h, bounds[i, 1])
        xm[:, i] = np.maximum(x[:, i] - h, bounds[i, 0])
        step = xp[:, i] - xm[:, i]
        if not np.all(np.isfinite(step)) or np.any(step <= 0.0):
            raise ValueError(f"dgsm finite-difference stencil for dimension {i} has a nonpositive or non-finite step.")
        yp = _eval_model(func, xp, label="dgsm's func")
        ym = _eval_model(func, xm, label="dgsm's func")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            derivative = (yp - ym) / step
            nu[i] = float(np.mean(derivative**2))
        _require_finite_derived(derivative, f"dgsm derivative for dimension {i}")
        _require_finite_derived(nu[i], f"dgsm nu[{i}]")
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = span**2 * nu
        total = float(weighted.sum())
    _require_finite_derived(weighted, "dgsm weighted importance")
    _require_finite_derived(total, "dgsm total importance")
    importance = weighted / total if total > 0 else np.zeros(d)
    _require_finite_derived(importance, "dgsm normalized importance")
    return {"nu": nu, "importance": importance, "names": names_out, "sampling_method": sampling_method}
