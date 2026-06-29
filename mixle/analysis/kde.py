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


def silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth ``0.9 min(sd, IQR/1.34) n^{-1/5}`` (1-D)."""
    x = np.asarray(data, dtype=float).ravel()
    n = x.shape[0]
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(0.9 * spread * n ** (-1.0 / 5.0))


def scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule-of-thumb bandwidth ``sd * n^{-1/(d+4)}``."""
    x = np.atleast_2d(np.asarray(data, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    n, d = x.shape
    return float(np.mean(np.std(x, axis=0, ddof=1)) * n ** (-1.0 / (d + 4)))


def _resolve_bw(data: np.ndarray, bandwidth) -> float:
    if isinstance(bandwidth, str):
        return silverman_bandwidth(data) if bandwidth == "silverman" else scott_bandwidth(data)
    return float(bandwidth)


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
        self.data = np.asarray(data, dtype=float).ravel()
        self.n = self.data.shape[0]
        self.bandwidth = _resolve_bw(self.data, bandwidth)
        self.bounds = bounds
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
    """
    x = np.asarray(data, dtype=float).ravel()
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
    """
    e = np.asarray(events, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float)
    h = _resolve_bw(e, bandwidth)
    u = (grid[:, None] - e[None, :]) / h
    lam = np.sum(stats.norm.pdf(u) / h, axis=1)
    if edge_correct:
        lo, hi = domain if domain is not None else (e.min(), e.max())
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
