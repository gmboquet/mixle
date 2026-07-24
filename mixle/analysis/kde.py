"""Kernel density, mode, and point-process intensity estimation.

Nonparametric estimates of *where the mass is* without assuming a parametric family:

  * :class:`KDE` / :func:`kde` -- kernel density estimation in 1-D (and product-kernel in higher
    dimensions), with automatic bandwidth (Silverman / Scott), **boundary correction** by reflection
    for densities on a half-line or interval (a plain KDE leaks mass across a hard boundary and biases
    the edge down), and **adaptive** (variable) bandwidths that widen in the sparse tails (Abramson).
  * :func:`kde_mode` -- the location of the density's peak ("where is the mode, and how sure am I?")
    with a bootstrap confidence interval.
  * :func:`intensity` -- the intensity ``lambda(t)`` of an inhomogeneous Poisson / point process by
    kernel smoothing of event locations, with optional edge correction (ties to the Cox-process
    machinery elsewhere in the library).

Bandwidths are in data units; ``"silverman"`` and ``"scott"`` are the rule-of-thumb selectors.
"""

from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from scipy import stats

_BANDWIDTH_METHODS = ("silverman", "scott")


def _require_variation(x: np.ndarray, method: str) -> None:
    """Guard for automatic bandwidth selectors (Silverman/Scott), which divide by the sample spread
    (standard deviation or IQR). That spread is exactly zero for a constant sample and undefined
    (``ddof=1`` divides by zero) for a sample with fewer than 2 points, so an empty, singleton, or
    constant sample previously produced a silent zero or NaN bandwidth -- and every subsequent kernel
    evaluation divides by that bandwidth, turning the whole density into NaN (MXR-080-0100).

    Reject those cases explicitly instead. A caller with a genuinely degenerate sample can still build a
    :class:`KDE` by supplying an explicit positive numeric ``bandwidth`` (bypassing automatic selection
    entirely -- a single fixed-bandwidth Gaussian bump is still a well-defined density even at one point
    or a repeated constant).

    Raises:
        ValueError: ``x`` has fewer than 2 observations, contains a non-finite entry, or is constant.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError(
            f"{method} needs at least 2 observations to estimate a spread for an automatic bandwidth, "
            f"got {n}. Pass an explicit positive numeric `bandwidth` instead."
        )
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise ValueError(f"{method} requires finite data, got {n_bad} of {n} non-finite entries")
    if not np.ptp(x) > 0:
        raise ValueError(
            f"{method} needs nonzero variation to estimate a spread for an automatic bandwidth, got a "
            f"constant sample (all values equal to {float(x[0]):g}). Pass an explicit positive numeric "
            "`bandwidth` instead."
        )


def silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth ``0.9 min(sd, IQR/1.34) n^{-1/5}`` (1-D).

    Raises:
        ValueError: ``data`` has fewer than 2 observations, non-finite entries, or zero variation (see
            :func:`_require_variation`; MXR-080-0100).
    """
    x = np.asarray(data, dtype=float).ravel()
    _require_variation(x, "silverman_bandwidth")
    n = x.shape[0]
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(0.9 * spread * n ** (-1.0 / 5.0))


def scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule-of-thumb bandwidth ``sd * n^{-1/(d+4)}``.

    Raises:
        ValueError: ``data`` has fewer than 2 observations, non-finite entries, or zero variation in
            every marginal (MXR-080-0100).
    """
    x = np.atleast_2d(np.asarray(data, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    n, d = x.shape
    if n < 2:
        raise ValueError(
            "scott_bandwidth needs at least 2 observations to estimate a spread for an automatic "
            f"bandwidth, got {n}. Pass an explicit positive numeric `bandwidth` instead."
        )
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise ValueError(f"scott_bandwidth requires finite data, got {n_bad} of {x.size} non-finite entries")
    sd = np.std(x, axis=0, ddof=1)
    if not np.all(sd > 0):
        raise ValueError(
            "scott_bandwidth needs nonzero variation in every dimension to estimate a spread for an "
            "automatic bandwidth, got a constant sample. Pass an explicit positive numeric `bandwidth` "
            "instead."
        )
    return float(np.mean(sd) * n ** (-1.0 / (d + 4)))


def _resolve_bw(data: np.ndarray, bandwidth) -> float:
    """Resolve a bandwidth spec to a strictly positive, finite scalar bandwidth.

    Raises:
        ValueError: ``bandwidth`` is a string other than ``"silverman"``/``"scott"``, or a number that
            is not strictly positive and finite (MXR-080-0100 -- previously an unrecognized method name
            silently fell through to Scott's rule, and a non-positive/non-finite number was accepted
            without complaint).
    """
    if isinstance(bandwidth, str):
        if bandwidth not in _BANDWIDTH_METHODS:
            raise ValueError(
                f"unsupported bandwidth method {bandwidth!r}; expected one of {_BANDWIDTH_METHODS} or "
                "a positive finite numeric bandwidth"
            )
        return silverman_bandwidth(data) if bandwidth == "silverman" else scott_bandwidth(data)
    bw = float(bandwidth)
    if not np.isfinite(bw) or bw <= 0.0:
        raise ValueError(f"bandwidth must be strictly positive and finite, got {bandwidth!r}")
    return bw


def _validate_bounds(
    bounds: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    """Validate a ``(lo, hi)`` bounds pair for reflection boundary correction (MXR-080-0100): either
    side may be ``None`` (unbounded), but a given side must be finite, and if both are given, ``lo``
    must be strictly less than ``hi``.

    Raises:
        ValueError: ``bounds`` is not a 2-element pair, a given side is non-finite, or ``lo >= hi``.
    """
    if bounds is None:
        return None
    if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
        raise ValueError(f"bounds must be a (lo, hi) pair, got {bounds!r}")
    lo, hi = bounds
    if lo is not None:
        lo = float(lo)
        if not np.isfinite(lo):
            raise ValueError(f"bounds lo must be finite or None, got {lo!r}")
    if hi is not None:
        hi = float(hi)
        if not np.isfinite(hi):
            raise ValueError(f"bounds hi must be finite or None, got {hi!r}")
    if lo is not None and hi is not None and not lo < hi:
        raise ValueError(f"bounds lo must be strictly less than hi, got bounds=({lo!r}, {hi!r})")
    return (lo, hi)


class KDE:
    """A fitted kernel density estimate.

    Use :func:`kde` to construct. Evaluate with :meth:`evaluate` (or call the instance). Supports a
    Gaussian kernel, reflection boundary correction (``bounds``), and adaptive bandwidths.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        bandwidth="silverman",
        bounds: tuple[float | None, float | None] | None = None,
        adaptive: bool = False,
    ) -> None:
        x = np.asarray(data, dtype=float).ravel()
        if x.shape[0] == 0:
            raise ValueError("KDE requires at least one observation, got an empty sample")
        if not np.all(np.isfinite(x)):
            n_bad = int(np.sum(~np.isfinite(x)))
            raise ValueError(f"KDE data must be finite, got {n_bad} of {x.shape[0]} non-finite entries")
        self.data = x
        self.n = self.data.shape[0]
        self.bandwidth = _resolve_bw(self.data, bandwidth)
        self.bounds = _validate_bounds(bounds)
        self.adaptive = adaptive
        self._local_bw = np.full(self.n, self.bandwidth)
        if adaptive:
            pilot = self._raw_density(self.data, np.full(self.n, self.bandwidth))
            g = np.exp(np.mean(np.log(np.clip(pilot, 1e-300, None))))
            self._local_bw = self.bandwidth * np.sqrt(g / np.clip(pilot, 1e-300, None))

    def _raw_density(self, x: np.ndarray, local_bw: np.ndarray) -> np.ndarray:
        """Plain (no boundary) Gaussian KDE at points ``x`` using per-data-point bandwidths."""
        x = np.atleast_1d(x)
        u = (x[:, None] - self.data[None, :]) / local_bw[None, :]
        return np.mean(stats.norm.pdf(u) / local_bw[None, :], axis=1)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Density at points ``x`` (with reflection boundary correction if ``bounds`` was set)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dens = self._raw_density(x, self._local_bw)
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo is not None:
                dens = dens + self._raw_density(2.0 * lo - x, self._local_bw)
            if hi is not None:
                dens = dens + self._raw_density(2.0 * hi - x, self._local_bw)
            mask = np.ones_like(x, dtype=bool)
            if lo is not None:
                mask &= x >= lo
            if hi is not None:
                mask &= x <= hi
            dens = np.where(mask, dens, 0.0)
        return dens

    __call__ = evaluate


def kde(data: np.ndarray, *, bandwidth="silverman", bounds=None, adaptive: bool = False) -> KDE:
    """Construct a kernel density estimate (Gaussian kernel).

    Args:
        data: ``(n,)`` sample.
        bandwidth: ``"silverman"``, ``"scott"``, or a positive float.
        bounds: ``(lo, hi)`` support limits for reflection boundary correction; either may be ``None``
            for an unbounded side (e.g. ``(0.0, None)`` for a positive variable).
        adaptive: use Abramson variable bandwidths (wider where the pilot density is low).

    Returns:
        A :class:`KDE`.

    Raises:
        ValueError: ``data`` is empty or non-finite; ``bandwidth`` is an unrecognized method name or a
            non-positive/non-finite number; automatic bandwidth selection (``"silverman"``/``"scott"``)
            is requested on fewer than 2 observations or a constant sample; or ``bounds`` is out of
            order or non-finite (MXR-080-0100).
    """
    return KDE(data, bandwidth=bandwidth, bounds=bounds, adaptive=adaptive)


def kde_mode(
    data: np.ndarray,
    *,
    bandwidth="silverman",
    bounds=None,
    grid: np.ndarray | None = None,
    ci: bool = False,
    n_boot: int = 500,
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
) -> float | dict:
    """Estimate the mode (peak location) of a density, optionally with a bootstrap CI.

    Args:
        data: ``(n,)`` sample.
        bandwidth, bounds: passed to :func:`kde`.
        grid: evaluation grid; defaults to 512 points spanning the data range.
        ci: if True return a percentile bootstrap interval for the mode.
        n_boot, ci_level, seed: bootstrap controls.

    Returns:
        The mode (float), or ``{'mode', 'ci_low', 'ci_high'}`` when ``ci`` is True.

    Raises:
        ValueError: ``data`` is empty; or, when ``ci`` is True, ``n_boot`` is not a positive integer or
            ``ci_level`` is not in the open interval ``(0, 1)`` (MXR-080-0100). Also propagates any
            :class:`KDE` construction error from :func:`kde` (e.g. automatic bandwidth selection on a
            constant sample).
    """
    x = np.asarray(data, dtype=float).ravel()
    if x.shape[0] == 0:
        raise ValueError("kde_mode requires at least one observation, got an empty sample")
    if ci:
        if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
            raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
        lvl = float(ci_level)
        if not np.isfinite(lvl) or not (0.0 < lvl < 1.0):
            raise ValueError(f"ci_level must be in the open interval (0, 1), got {ci_level!r}")
    if grid is None:
        pad = 0.1 * (x.max() - x.min() + 1e-12)
        grid = np.linspace(x.min() - pad, x.max() + pad, 512)
    mode = float(grid[np.argmax(kde(x, bandwidth=bandwidth, bounds=bounds).evaluate(grid))])
    if not ci:
        return mode
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = x[rng.randint(0, x.shape[0], x.shape[0])]
        boot[b] = grid[np.argmax(kde(sample, bandwidth=bandwidth, bounds=bounds).evaluate(grid))]
    lo_q = (1.0 - ci_level) / 2.0
    return {"mode": mode, "ci_low": float(np.quantile(boot, lo_q)), "ci_high": float(np.quantile(boot, 1.0 - lo_q))}


def intensity(
    events: np.ndarray,
    grid: np.ndarray,
    *,
    bandwidth="silverman",
    domain: tuple[float, float] | None = None,
    edge_correct: bool = True,
) -> np.ndarray:
    """Kernel intensity ``lambda(t)`` of an inhomogeneous Poisson / point process.

    Unlike a density (which integrates to 1), the intensity integrates to the *expected number of
    events*: ``lambda_hat(t) = sum_i K_h(t - t_i)``. With ``edge_correct`` the estimate is divided by
    the fraction of the kernel falling inside ``domain``, removing the downward bias near the boundary.

    Args:
        events: ``(m,)`` event locations.
        grid: points ``t`` at which to evaluate the intensity.
        bandwidth: ``"silverman"``, ``"scott"``, or a float.
        domain: ``(lo, hi)`` observation window (defaults to the event range); used for edge correction.
        edge_correct: divide by the in-window kernel mass at each ``t``.

    Returns:
        The intensity evaluated on ``grid``.

    Raises:
        ValueError: ``events`` is empty or non-finite; ``bandwidth`` is an unrecognized method name or a
            non-positive/non-finite number; automatic bandwidth selection is requested on fewer than 2
            events or constant events; or (when ``edge_correct`` is True) the effective ``domain`` --
            explicitly passed, or defaulted to the event range -- is non-finite or has ``lo >= hi``
            (MXR-080-0100; a collapsed-to-a-point domain, e.g. from constant events with an explicit
            numeric bandwidth, previously produced a silent divide-by-near-zero blowup).
    """
    e = np.asarray(events, dtype=float).ravel()
    if e.shape[0] == 0:
        raise ValueError("intensity requires at least one event, got an empty sample")
    if not np.all(np.isfinite(e)):
        n_bad = int(np.sum(~np.isfinite(e)))
        raise ValueError(f"event locations must be finite, got {n_bad} of {e.shape[0]} non-finite entries")
    grid = np.asarray(grid, dtype=float)
    h = _resolve_bw(e, bandwidth)
    u = (grid[:, None] - e[None, :]) / h
    lam = np.sum(stats.norm.pdf(u) / h, axis=1)
    if edge_correct:
        lo, hi = domain if domain is not None else (float(e.min()), float(e.max()))
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"domain must be finite, got domain=({lo!r}, {hi!r})")
        if not lo < hi:
            raise ValueError(f"domain lo must be strictly less than hi, got domain=({lo!r}, {hi!r})")
        q = stats.norm.cdf((hi - grid) / h) - stats.norm.cdf((lo - grid) / h)
        lam = lam / np.clip(q, 1e-6, None)
    return lam


__all__ = [
    "KDE",
    "kde",
    "silverman_bandwidth",
    "scott_bandwidth",
    "kde_mode",
    "intensity",
]
