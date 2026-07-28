"""Kernel density, mode, and point-process intensity estimation.

Nonparametric estimates of *where the mass is* without assuming a parametric family:

  * :class:`KDE` / :func:`kde` -- kernel density estimation for a 1-D ``(n,)`` sample or, via an
    axis-aligned Gaussian **product kernel** with a per-dimension bandwidth, a genuinely multivariate
    ``(n, d)`` sample (rows are observations; row-pairing is preserved, never flattened). Automatic
    bandwidth selection (Silverman / Scott) chooses each dimension's bandwidth from that dimension's own
    marginal spread, using the joint-dimension ``n^{-1/(d+4)}`` rate. **Adaptive** (variable) bandwidths
    that widen in the sparse tails (Abramson) work in any dimension; normalized **boundary correction**
    (reflection on a half-line and per-kernel truncation normalization on a finite interval) is supported
    for 1-D data only.
  * :func:`kde_mode` -- the location of a 1-D density's peak ("where is the mode, and how sure am I?")
    with a bootstrap confidence interval.
  * :func:`intensity` -- the intensity ``lambda(t)`` of a 1-D inhomogeneous Poisson / point process by
    kernel smoothing of event locations, with optional edge correction (ties to the Cox-process
    machinery elsewhere in the library).

Bandwidths are in data units; ``"silverman"`` and ``"scott"`` are the rule-of-thumb selectors.
"""

from __future__ import annotations

import operator

import numpy as np
from numpy.random import RandomState
from scipy import special, stats

_BANDWIDTH_METHODS = ("silverman", "scott")


def _positive_dimension(d: int) -> int:
    """Validate the exact joint dimension used by bandwidth-rate formulas."""
    if isinstance(d, (bool, np.bool_)):
        raise ValueError("d must be a non-Boolean positive integer")
    try:
        dimension = operator.index(d)
    except TypeError as exc:
        raise ValueError("d must be a non-Boolean positive integer") from exc
    if dimension <= 0:
        raise ValueError(f"d must be positive, got {dimension}")
    return int(dimension)


def _as_observations(data: np.ndarray) -> np.ndarray:
    """Interpret ``data`` as ``(n, d)`` observations, preserving row-pairing (MXR-080-0099).

    A 1-D ``(n,)`` input is ``n`` scalar observations of 1 variable, shape ``(n, 1)``. A 2-D ``(n, d)``
    input is ``n`` observations of ``d`` variables, one observation per *row* -- returned unchanged, so
    which values came from the same observation (row) and which axis is which variable are both
    preserved exactly as given. This never reshapes across rows: unlike ``.ravel()`` on an ``(n, d)``
    array (the previous behavior, which turned ``n`` paired ``d``-dimensional observations into ``n*d``
    unrelated scalars, destroying both the sample pairing and which axis was which variable), a genuine
    2-D input is used as-is.

    Note this differs from (and fixes) the disambiguation ``scott_bandwidth`` used to use internally
    (``np.atleast_2d`` then transpose iff ``shape[0] == 1``): that heuristic guessed "the caller must
    have meant a 1-D sample" from the *shape* alone, which is wrong for a genuine single ``d``-variable
    observation of shape ``(1, d)``. Here the caller's intent is read from ``ndim``, which is
    unambiguous: pass a 2-D ``(1, d)`` array (e.g. ``[[1.0, 2.0, 3.0]]``) for one ``d``-dimensional
    observation, never a bare 1-D array of length ``d``.

    Raises:
        ValueError: ``data`` has more than 2 dimensions.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected a 1-D (n,) or 2-D (n, d) array, got shape {x.shape}")


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


def silverman_bandwidth(data: np.ndarray, *, d: int = 1) -> float:
    """Silverman's rule-of-thumb bandwidth ``0.9 min(sd, IQR/1.34) n^{-1/(d+4)}`` for a 1-D sample.

    ``data`` is always treated as one 1-D sample (e.g. a single column of a multivariate dataset).
    ``d`` only sets the exponent: it is the dimensionality of the *joint* sample this bandwidth will be
    used in a product kernel for -- pass it explicitly when selecting one dimension's bandwidth for a
    multivariate :class:`KDE`, so the ``n^{-1/(d+4)}`` rate reflects the true joint dimension rather than
    the univariate default ``d=1`` (``n^{-1/5}``) rate (MXR-080-0099).

    Raises:
        ValueError: ``data`` has fewer than 2 observations, non-finite entries, or zero variation (see
            :func:`_require_variation`; MXR-080-0100).
    """
    d = _positive_dimension(d)
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"silverman_bandwidth expects a one-dimensional sample, got shape {x.shape}")
    _require_variation(x, "silverman_bandwidth")
    n = x.shape[0]
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(0.9 * spread * n ** (-1.0 / (d + 4.0)))


def scott_bandwidth(data: np.ndarray, *, d: int = 1) -> float:
    """Scott's rule-of-thumb bandwidth ``sd * n^{-1/(d+4)}`` for a 1-D sample.

    ``data`` is always treated as one 1-D sample; ``d`` only sets the exponent -- see
    :func:`silverman_bandwidth` (whose ``d`` parameter has exactly the same meaning here).

    Raises:
        ValueError: ``data`` has fewer than 2 observations, non-finite entries, or zero variation (see
            :func:`_require_variation`; MXR-080-0100).
    """
    d = _positive_dimension(d)
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"scott_bandwidth expects a one-dimensional sample, got shape {x.shape}")
    _require_variation(x, "scott_bandwidth")
    n = x.shape[0]
    return float(np.std(x, ddof=1) * n ** (-1.0 / (d + 4.0)))


def _resolve_bandwidth_vector(x: np.ndarray, bandwidth, d: int) -> np.ndarray:
    """Resolve a bandwidth spec against ``(n, d)`` observations ``x`` into a length-``d`` vector of
    strictly positive, finite per-dimension bandwidths for the axis-aligned Gaussian product kernel
    (MXR-080-0099).

    ``bandwidth`` may be:
      * ``"silverman"`` / ``"scott"`` -- each dimension gets its own automatic bandwidth from that
        dimension's own marginal spread (column ``j`` of ``x``), using the joint-dimension
        ``n^{-1/(d+4)}`` rate (not the univariate ``n^{-1/5}`` rate) -- see :func:`silverman_bandwidth`.
      * a single positive finite number -- used as the bandwidth in every dimension.
      * a length-``d`` sequence of positive finite numbers -- one bandwidth per dimension.

    Raises:
        ValueError: ``bandwidth`` is an unrecognized method name; a numeric spec that is not a scalar or
            length-``d``, or is not strictly positive and finite in every dimension; or (string methods
            only) some dimension has fewer than 2 observations or zero variation, named by index.
    """
    if isinstance(bandwidth, str):
        if bandwidth not in _BANDWIDTH_METHODS:
            raise ValueError(
                f"unsupported bandwidth method {bandwidth!r}; expected one of {_BANDWIDTH_METHODS}, a "
                f"positive finite number, or a length-{d} sequence of positive finite numbers"
            )
        selector = silverman_bandwidth if bandwidth == "silverman" else scott_bandwidth
        bw = np.empty(d)
        for j in range(d):
            try:
                bw[j] = selector(x[:, j], d=d)
            except ValueError as exc:
                raise ValueError(f"dimension {j}: {exc}") from exc
        return bw
    arr = np.atleast_1d(np.asarray(bandwidth, dtype=float))
    if arr.shape == (1,) and d > 1:
        arr = np.full(d, arr[0])
    if arr.shape != (d,):
        raise ValueError(
            f"numeric bandwidth must be a positive finite number or a length-{d} vector (one per "
            f"dimension), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        raise ValueError(f"bandwidth must be strictly positive and finite in every dimension, got {arr.tolist()}")
    return arr


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
    """A fitted kernel density estimate, 1-D or multivariate.

    Use :func:`kde` to construct. Evaluate with :meth:`evaluate` (or call the instance). Supports a
    Gaussian kernel -- an axis-aligned **product kernel** with a per-dimension bandwidth for
    multivariate ``(n, d)`` data, rows treated as observations and never reshaped -- adaptive bandwidths
    (any dimension), and, for 1-D data only, normalized boundary correction (``bounds``;
    MXR-080-0099).

    Attributes:
        data: the fitted sample as ``(n, d)`` observations (``d == 1`` for a 1-D sample).
        n: number of observations.
        d: dimensionality (``1`` for a 1-D sample).
        bandwidth: the resolved bandwidth -- a scalar ``float`` when ``d == 1``, else a length-``d``
            :class:`numpy.ndarray` (one bandwidth per dimension).
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        bandwidth="silverman",
        bounds: tuple[float | None, float | None] | None = None,
        adaptive: bool = False,
    ) -> None:
        x = _as_observations(data)
        n, d = x.shape
        if n == 0:
            raise ValueError("KDE requires at least one observation, got an empty sample")
        if not np.all(np.isfinite(x)):
            n_bad = int(np.sum(~np.isfinite(x)))
            raise ValueError(f"KDE data must be finite, got {n_bad} of {x.size} non-finite entries")
        if d == 0:
            raise ValueError("KDE requires at least one dimension, got a shape-(n, 0) sample")
        owned_data = np.array(x, dtype=float, copy=True)
        owned_data.setflags(write=False)
        self._data = owned_data
        self.data = owned_data.view()
        self.data.setflags(write=False)
        self.n = n
        self.d = d
        bw_vec = np.array(_resolve_bandwidth_vector(owned_data, bandwidth, d), copy=True)
        bw_vec.setflags(write=False)
        self.bandwidth = float(bw_vec[0]) if d == 1 else bw_vec.view()
        if isinstance(self.bandwidth, np.ndarray):
            self.bandwidth.setflags(write=False)
        if bounds is not None and d > 1:
            raise ValueError(
                "bounds (reflection boundary correction) is only supported for 1-D KDE, got a "
                f"{d}-dimensional sample; construct separate per-dimension KDEs if you need per-axis "
                "boundary correction"
            )
        self.bounds = _validate_bounds(bounds) if d == 1 else None
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo is not None and np.any(owned_data[:, 0] < lo):
                raise ValueError("KDE data must lie within the declared lower support bound")
            if hi is not None and np.any(owned_data[:, 0] > hi):
                raise ValueError("KDE data must lie within the declared upper support bound")
        self.adaptive = adaptive
        self._local_bw = np.tile(bw_vec, (n, 1))
        if adaptive:
            pilot = self._raw_density(self.data, self._local_bw)
            g = np.exp(np.mean(np.log(np.clip(pilot, 1e-300, None))))
            scale = np.sqrt(g / np.clip(pilot, 1e-300, None))
            self._local_bw = bw_vec[None, :] * scale[:, None]
        self._local_bw.setflags(write=False)

    def _raw_density(self, x: np.ndarray, local_bw: np.ndarray) -> np.ndarray:
        """Plain (no boundary) Gaussian product-kernel KDE at ``(m, d)`` points ``x`` using
        per-data-point, per-dimension bandwidths ``local_bw`` (``(n, d)``)."""
        u = (x[:, None, :] - self.data[None, :, :]) / local_bw[None, :, :]
        kernel = stats.norm.pdf(u) / local_bw[None, :, :]  # (m, n, d)
        return np.mean(np.prod(kernel, axis=2), axis=1)  # (m,)

    def _bounded_density(self, x: np.ndarray) -> np.ndarray:
        """Normalized truncated-Gaussian boundary kernel for a one-dimensional declared support."""
        if self.bounds is None:  # pragma: no cover - guarded by evaluate
            raise RuntimeError("_bounded_density requires declared bounds")
        lo, hi = self.bounds
        centers = self._data[:, 0]
        bandwidths = self._local_bw[:, 0]
        query = x[:, 0]
        support = np.ones(query.shape[0], dtype=bool)
        if lo is not None:
            support &= query >= lo
        if hi is not None:
            support &= query <= hi

        # A single reflected image is exactly normalized on a half-line and retains the conventional
        # reflection estimator's strong edge-bias correction.
        if lo is not None and hi is None:
            reflected = np.array(x, copy=True)
            reflected[:, 0] = 2.0 * lo - query
            density = self._raw_density(x, self._local_bw) + self._raw_density(reflected, self._local_bw)
            return np.where(support, density, 0.0)
        if hi is not None and lo is None:
            reflected = np.array(x, copy=True)
            reflected[:, 0] = 2.0 * hi - query
            density = self._raw_density(x, self._local_bw) + self._raw_density(reflected, self._local_bw)
            return np.where(support, density, 0.0)
        if lo is None and hi is None:
            return self._raw_density(x, self._local_bw)

        standardized = (query[:, None] - centers[None, :]) / bandwidths[None, :]
        kernels = stats.norm.pdf(standardized) / bandwidths[None, :]

        z_lo = (lo - centers) / bandwidths
        z_hi = (hi - centers) / bandwidths
        normalizers = 0.5 * (special.erf(z_hi / np.sqrt(2.0)) - special.erf(z_lo / np.sqrt(2.0)))
        if not np.all(np.isfinite(normalizers)) or np.any(normalizers <= 0.0):
            raise ValueError("bounded KDE kernel normalization is not finite and positive")

        density = np.mean(kernels / normalizers[None, :], axis=1)
        return np.where(support, density, 0.0)

    def _prepare_eval_points(self, x: np.ndarray) -> np.ndarray:
        xo = _as_observations(x)
        if xo.shape[0] == 0:
            raise ValueError("evaluate() requires at least one evaluation point")
        if xo.shape[1] != self.d:
            raise ValueError(
                f"evaluate() expects points of dimension {self.d} (this KDE was fit on {self.d}-D "
                f"data), got dimension {xo.shape[1]}"
            )
        if not np.all(np.isfinite(xo)):
            raise ValueError("evaluate() points must be finite (no NaN/Inf)")
        return xo

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Density at points ``x`` -- ``(m,)`` for 1-D, ``(m, d)`` for multivariate -- with reflection
        boundary correction if ``bounds`` was set (1-D only). Always returns a length-``m`` array."""
        xo = self._prepare_eval_points(x)
        if self.bounds is not None:
            return self._bounded_density(xo)
        return self._raw_density(xo, self._local_bw)

    __call__ = evaluate


def kde(data: np.ndarray, *, bandwidth="silverman", bounds=None, adaptive: bool = False) -> KDE:
    """Construct a kernel density estimate (Gaussian kernel; product kernel for multivariate data).

    Args:
        data: ``(n,)`` sample, or ``(n, d)`` for a ``d``-dimensional sample -- rows are observations and
            must stay paired (never flatten an ``(n, d)`` array before calling this; MXR-080-0099).
        bandwidth: ``"silverman"``, ``"scott"``, a single positive finite number (used in every
            dimension), or -- for ``(n, d)`` data -- a length-``d`` sequence of positive finite numbers
            (one bandwidth per dimension). Automatic selection needs >= 2 observations with nonzero
            variation in every dimension.
        bounds: ``(lo, hi)`` support limits for normalized boundary correction; either may be ``None``
            for an unbounded side (e.g. ``(0.0, None)`` for a positive variable). Half-lines use an
            exactly normalized reflected image; finite intervals normalize every truncated kernel.
            Only supported for 1-D (``d == 1``) data.
        adaptive: use Abramson variable bandwidths (wider where the pilot density is low); supported for
            any dimension.

    Returns:
        A :class:`KDE`.

    Raises:
        ValueError: ``data`` is empty, non-finite, or more than 2-D; ``bandwidth`` is an unrecognized
            method name, the wrong length, or not strictly positive and finite; automatic bandwidth
            selection is requested on fewer than 2 observations or a constant sample in some dimension
            (MXR-080-0100); or ``bounds`` is given together with ``d > 1``, out of order, or non-finite
            (MXR-080-0099 / MXR-080-0100).
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
    """Estimate the mode (peak location) of a 1-D density, optionally with a bootstrap CI.

    Univariate only: a dense grid search over ``d`` dimensions is exponential in ``d`` and is not
    implemented here (unlike :class:`KDE`, this never accepted multivariate data as a documented
    feature, so an ``(n, d)`` input is now rejected explicitly instead of being silently flattened with
    ``.ravel()`` -- the same silent-flattening defect :class:`KDE` itself had; MXR-080-0099).

    Args:
        data: ``(n,)`` sample.
        bandwidth, bounds: passed to :func:`kde`.
        grid: evaluation grid; defaults to 512 points spanning the data range.
        ci: if True return a percentile bootstrap interval for the mode.
        n_boot, ci_level, seed: bootstrap controls.

    Bootstrap policy for degenerate resamples (MXR-080-1587): a resample of a perfectly valid,
    nonconstant sample can itself come out constant -- ``[0, 0, 1]`` resamples to ``[0, 0, 0]`` roughly
    one draw in four -- and an automatic bandwidth selector has no spread to estimate from, so it
    raises. That made the interval fail *at random, depending on the seed*, for a fit that had already
    succeeded. A degenerate resample is now scored at the ORIGINAL fit's bandwidth (the sample the
    bandwidth was selected from is the full one, and a bootstrap replicate is a resample of it, not a
    new dataset), rather than skipped -- skipping would quietly narrow the interval by dropping exactly
    the least-dispersed replicates. The count of replicates that needed the fallback is reported as
    ``n_degenerate_resamples`` so a caller can see when the interval rests largely on them.

    Returns:
        The mode (float), or ``{'mode', 'ci_low', 'ci_high', 'n_boot', 'n_degenerate_resamples'}``
        when ``ci`` is True.

    Raises:
        ValueError: ``data`` is empty or not 1-D; or, when ``ci`` is True, ``n_boot`` is not a positive
            integer or ``ci_level`` is not in the open interval ``(0, 1)`` (MXR-080-0100). Also
            propagates any :class:`KDE` construction error from :func:`kde` on the ORIGINAL sample
            (e.g. automatic bandwidth selection on a constant input) -- that is a genuine property of
            the data, unlike a degenerate resample, which is handled by the policy above.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim > 1:
        raise ValueError(
            f"kde_mode is univariate: expected a 1-D (n,) sample, got shape {x.shape}. Multivariate "
            "mode-finding (a grid search over d dimensions) is not supported."
        )
    x = x.ravel()
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
    fit = kde(x, bandwidth=bandwidth, bounds=bounds)
    mode = float(grid[np.argmax(fit.evaluate(grid))])
    if not ci:
        return mode
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    # The bandwidth the ORIGINAL fit resolved to, used as the documented fallback when a resample is
    # too degenerate for the automatic selector to run on (see the bootstrap policy above).
    fallback_bandwidth = float(np.atleast_1d(np.asarray(fit.bandwidth, dtype=float)).ravel()[0])
    boot = np.empty(n_boot)
    n_degenerate = 0
    for b in range(n_boot):
        sample = x[rng.randint(0, x.shape[0], x.shape[0])]
        try:
            replicate = kde(sample, bandwidth=bandwidth, bounds=bounds)
        except ValueError:
            n_degenerate += 1
            replicate = kde(sample, bandwidth=fallback_bandwidth, bounds=bounds)
        boot[b] = grid[np.argmax(replicate.evaluate(grid))]
    lo_q = (1.0 - ci_level) / 2.0
    return {
        "mode": mode,
        "ci_low": float(np.quantile(boot, lo_q)),
        "ci_high": float(np.quantile(boot, 1.0 - lo_q)),
        "n_boot": int(n_boot),
        "n_degenerate_resamples": int(n_degenerate),
    }


def intensity(
    events: np.ndarray,
    grid: np.ndarray,
    *,
    bandwidth="silverman",
    domain: tuple[float, float] | None = None,
    edge_correct: bool = True,
) -> np.ndarray:
    """Kernel intensity ``lambda(t)`` of a 1-D inhomogeneous Poisson / point process.

    Unlike a density (which integrates to 1), the intensity integrates to the *expected number of
    events*: ``lambda_hat(t) = sum_i K_h(t - t_i)``. With ``edge_correct`` the estimate is divided by
    the fraction of the kernel falling inside ``domain``, removing the downward bias near the boundary.

    Univariate only: like :func:`kde_mode`, this never accepted multivariate data as a documented
    feature, so an ``(m, d)`` input is now rejected explicitly instead of being silently flattened with
    ``.ravel()`` (MXR-080-0099).

    Args:
        events: ``(m,)`` event locations.
        grid: nonempty finite one-dimensional points ``t`` at which to evaluate the intensity.
        bandwidth: ``"silverman"``, ``"scott"``, or a float.
        domain: ``(lo, hi)`` observation window (defaults to the event range); used for edge correction.
        edge_correct: divide by the in-window kernel mass at each ``t``.

    Returns:
        The intensity evaluated on ``grid``.

    Raises:
        ValueError: ``events`` is empty, non-finite, or not 1-D; ``bandwidth`` is an unrecognized method
            name or a non-positive/non-finite number; automatic bandwidth selection is requested on
            fewer than 2 events or constant events; or (when ``edge_correct`` is True) the effective
            ``domain`` -- explicitly passed, or defaulted to the event range -- is non-finite or has
            ``lo >= hi`` (MXR-080-0100; a collapsed-to-a-point domain, e.g. from constant events with an
            explicit numeric bandwidth, previously produced a silent divide-by-near-zero blowup).
    """
    e = np.asarray(events, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"intensity is univariate: expected a 1-D (m,) array of event locations, got shape {e.shape}")
    if e.shape[0] == 0:
        raise ValueError("intensity requires at least one event, got an empty sample")
    if not np.all(np.isfinite(e)):
        n_bad = int(np.sum(~np.isfinite(e)))
        raise ValueError(f"event locations must be finite, got {n_bad} of {e.shape[0]} non-finite entries")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"grid must be a nonempty one-dimensional evaluation array, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid must be finite (no NaN/Inf)")
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
