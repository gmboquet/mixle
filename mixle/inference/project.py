"""Closed-form variational projections — compress a structured teacher onto a smaller student *exactly*.

``mixle.ops.project`` already does the general forward-KL (M-)projection by SAMPLING the source and
fitting the target by maximum likelihood. That is universal but approximate and slow. This module adds
the cases where the projection has a **closed form** and needs no samples and no iteration:

* :func:`collapse_mixture` — moment-match a mixture onto a single component. For Gaussians this is the
  law of total variance, exact to machine precision (the M-projection of the mixture onto one Gaussian).
* :func:`reduce_mixture` — Runnalls' KL-greedy mixture reduction: repeatedly merge the pair of components
  whose merge costs the least KL, until the target count is reached. Each merge is a closed-form
  moment match, and the merge *cost* is Runnalls' analytic dissimilarity — so the whole reduction is
  closed form. Every merge is moment-preserving, so the reduced mixture keeps the original's overall
  mean and covariance exactly (a strong, checkable invariant).
* :func:`gaussian_kl` — the analytic KL between two Gaussians (the metric the above use).

Why Gaussians first: the mixtures that actually need compressing at frontier scale are Gaussian-ish --
mixture-of-experts routed in a latent, Kalman/SSM belief states, VLM latent fusion, GP posteriors. The
same moment-matching idea extends to any exponential family (via
``mixle.stats.compute.exp_family.ExponentialFamilyForm.mean_parameters``); those are added as their
closed-form moment algebra is filled in. Non-Gaussian sources fall back to the sampling projection with
a clear pointer, rather than silently pretending to be exact.

References: Runnalls, "Kullback-Leibler approach to Gaussian mixture reduction" (IEEE T-AES 2007);
the moment-matching M-projection is the standard forward-KL result (Minka; Bishop PRML §10.7).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.capability import CapabilityError

__all__ = [
    "gaussian_kl",
    "collapse_mixture",
    "reduce_mixture",
    "moment_project",
    "fisher_merge",
]


def _gaussian_components(mixture: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Extract ``(w, mus (K,d), covs (K,d,d), univariate)`` from any Gaussian mixture, or raise.

    Accepts a :class:`GaussianMixtureDistribution` (reads ``mu``/``sig2``/``w`` directly) or any
    :class:`MixtureDistribution` whose components are (multivariate) Gaussians. ``univariate`` records
    whether the components were 1-D scalar Gaussians so the result can be returned in the same kind.
    """
    w = np.asarray(getattr(mixture, "w", None), dtype=float)
    if w is None or w.ndim != 1:
        raise CapabilityError("collapse/reduce need a mixture with a 1-D weight vector `w`.")

    if hasattr(mixture, "mu") and hasattr(mixture, "sig2"):  # GaussianMixtureDistribution: full (K,d,d)
        mus = np.asarray(mixture.mu, dtype=float)
        covs = np.asarray(mixture.sig2, dtype=float)
        if mus.ndim == 2 and covs.ndim == 3:
            return w, mus, covs, False

    components = getattr(mixture, "components", None)
    if not components:
        raise CapabilityError("mixture exposes no Gaussian components to project.")
    mus, covs, univariate = [], [], False
    for c in components:
        if hasattr(c, "covar"):  # MultivariateGaussianDistribution(mu, covar)
            mus.append(np.asarray(c.mu, dtype=float).ravel())
            covs.append(np.asarray(c.covar, dtype=float))
        elif hasattr(c, "sigma2") and hasattr(c, "mu"):  # univariate GaussianDistribution(mu, sigma2)
            univariate = True
            mus.append(np.array([float(c.mu)]))
            covs.append(np.array([[float(c.sigma2)]]))
        else:
            raise CapabilityError(
                f"component {type(c).__name__} is not Gaussian; use mixle.ops.project for a sampling "
                "projection onto a target family."
            )
    return w, np.asarray(mus), np.asarray(covs), univariate


def _as_distribution(mu: np.ndarray, cov: np.ndarray, univariate: bool) -> Any:
    """Build the concrete Gaussian for a mean/cov, matching the input's univariate/multivariate kind."""
    if univariate:
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        return GaussianDistribution(float(mu[0]), float(cov[0, 0]))
    from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

    return MultivariateGaussianDistribution(np.asarray(mu, dtype=float), np.asarray(cov, dtype=float))


def _moment_match(w: np.ndarray, mus: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The single Gaussian matching a (sub)mixture's first two moments (law of total variance).

    ``mu = Σ w_k mu_k`` and ``Σ = Σ w_k (Σ_k + (mu_k-mu)(mu_k-mu)^T)`` for weights ``w`` summing to 1 --
    the exact M-projection of the mixture onto one Gaussian (minimizes ``KL(mixture || Gaussian)``).
    """
    wn = w / w.sum()
    mu = wn @ mus
    d = mus - mu
    cov = np.einsum("k,kij->ij", wn, covs) + np.einsum("k,ki,kj->ij", wn, d, d)
    return mu, 0.5 * (cov + cov.T)  # symmetrize against round-off


def gaussian_kl(p: Any, q: Any) -> float:
    """``KL(p || q)`` between two Gaussians (uni- or multivariate), in nats. Accepts distributions.

    ``0.5[ tr(Σq⁻¹ Σp) + (μq-μp)ᵀ Σq⁻¹ (μq-μp) - d + ln(det Σq / det Σp) ]`` -- the analytic Gaussian KL.
    """

    def _mc(dist: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(dist, "covar"):
            return np.asarray(dist.mu, dtype=float).ravel(), np.asarray(dist.covar, dtype=float)
        if hasattr(dist, "sigma2"):
            return np.array([float(dist.mu)]), np.array([[float(dist.sigma2)]])
        raise CapabilityError(f"gaussian_kl needs Gaussian distributions, got {type(dist).__name__}.")

    mp, sp = _mc(p)
    mq, sq = _mc(q)
    d = mp.shape[0]
    sq_inv = np.linalg.inv(sq)
    diff = mq - mp
    _, ldp = np.linalg.slogdet(sp)
    _, ldq = np.linalg.slogdet(sq)
    return float(0.5 * (np.trace(sq_inv @ sp) + diff @ sq_inv @ diff - d + (ldq - ldp)))


def collapse_mixture(mixture: Any) -> Any:
    """Moment-match a mixture onto a **single** distribution, in closed form (exact for Gaussians).

    Returns the Gaussian minimizing ``KL(mixture || Gaussian)`` -- its mean and covariance are the
    mixture's overall mean and covariance (law of total variance). This is the exact, sample-free
    counterpart to ``mixle.ops.project(mixture, GaussianDistribution(...).estimator())``.
    """
    w, mus, covs, univariate = _gaussian_components(mixture)
    mu, cov = _moment_match(w, mus, covs)
    return _as_distribution(mu, cov, univariate)


def _runnalls_cost(wi: float, mi: np.ndarray, ci: np.ndarray, wj: float, mj: np.ndarray, cj: np.ndarray) -> float:
    """Runnalls' KL dissimilarity for merging components i and j (the upper bound on KL increase).

    ``B(i,j) = ½[(wi+wj) ln|Σij| - wi ln|Σi| - wj ln|Σj|]`` with ``Σij`` the moment-matched merge. It is
    ≥ 0, zero iff the two components are identical, and inexpensive -- three log-determinants per pair.
    """
    wij = wi + wj
    _, cov = _moment_match(np.array([wi, wj]), np.stack([mi, mj]), np.stack([ci, cj]))
    _, ldij = np.linalg.slogdet(cov)
    _, ldi = np.linalg.slogdet(ci)
    _, ldj = np.linalg.slogdet(cj)
    return float(0.5 * (wij * ldij - wi * ldi - wj * ldj))


def reduce_mixture(mixture: Any, n_components: int, *, method: str = "runnalls") -> Any:
    """Reduce a Gaussian mixture to ``n_components`` by greedily merging the least-costly pair (Runnalls).

    Repeatedly merges the two components whose merge costs the least KL (:func:`_runnalls_cost`) until
    ``n_components`` remain. Every merge is moment-preserving, so the reduced mixture has the **same
    overall mean and covariance** as the original -- only higher moments are lost. Returns a mixture of
    the same kind as the input (a :class:`GaussianMixtureDistribution` for multivariate input).

    Args:
        mixture: a Gaussian mixture (``GaussianMixtureDistribution`` or a mixture of Gaussian components).
        n_components: target number of components (``>= 1``); no-op if already ``<=`` the current count.
        method: only ``"runnalls"`` for now.
    """
    if method != "runnalls":
        raise ValueError(f"unknown reduction method {method!r} (only 'runnalls').")
    w, mus, covs, univariate = _gaussian_components(mixture)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    w = w / w.sum()
    # active list of (weight, mean, cov); merge in place until the target count is reached
    comps = [[float(w[k]), mus[k].copy(), covs[k].copy()] for k in range(len(w))]
    while len(comps) > n_components:
        best, bi, bj = np.inf, 0, 1
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                cost = _runnalls_cost(comps[i][0], comps[i][1], comps[i][2], comps[j][0], comps[j][1], comps[j][2])
                if cost < best:
                    best, bi, bj = cost, i, j
        wi, mi, ci = comps[bi]
        wj, mj, cj = comps[bj]
        mu, cov = _moment_match(np.array([wi, wj]), np.stack([mi, mj]), np.stack([ci, cj]))
        comps[bi] = [wi + wj, mu, cov]
        comps.pop(bj)

    out_w = np.array([c[0] for c in comps])
    out_mu = np.stack([c[1] for c in comps])
    out_cov = np.stack([c[2] for c in comps])
    if univariate:
        from mixle.stats.latent.mixture import MixtureDistribution
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        components = [GaussianDistribution(float(out_mu[k, 0]), float(out_cov[k, 0, 0])) for k in range(len(out_w))]
        return MixtureDistribution(components, out_w)
    from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution

    return GaussianMixtureDistribution(out_mu, out_cov, out_w)


def moment_project(teacher: Any, target: Any = None, *, exact: bool = True, **sampling_kw: Any) -> Any:
    """Project ``teacher`` onto a smaller student -- exactly when possible, else by sampling.

    If ``teacher`` is a Gaussian mixture and ``target`` is ``None`` (or a single Gaussian family), the
    projection is the closed-form :func:`collapse_mixture` -- no samples, machine-precision. Otherwise
    (or when ``exact=False``) it delegates to :func:`mixle.ops.project`, the sampling M-projection onto
    ``target``'s family. This gives one entry point that is exact where the structure allows and explicit
    about sampling where it does not.
    """
    if exact and target is None:
        try:
            return collapse_mixture(teacher)
        except CapabilityError:
            pass
    if target is None:
        raise CapabilityError(
            "moment_project needs a `target` family unless `teacher` is a Gaussian mixture "
            "(then it collapses in closed form). Pass a target distribution/estimator for the sampling path."
        )
    from mixle.ops import project

    return project(teacher, target, **sampling_kw)


def fisher_merge(estimates: Any, fishers: Any = None) -> np.ndarray:
    """Fisher-weighted merge of parameter estimates -- the closed-form Laplace-posterior combination.

    Given estimates ``θ_i`` (each a flat parameter vector) and their Fisher information ``F_i``, returns
    ``θ* = (Σ F_i)⁻¹ (Σ F_i θ_i)`` -- the point that maximizes the sum of the local Laplace log-posteriors
    ``Σ_i -½(θ-θ_i)ᵀ F_i (θ-θ_i)``. This is Matena & Raffel Fisher merging (diagonal ``F``) and, in
    general, the precision-weighted mean; for Gaussians it coincides with the product-of-experts mean.
    No gradient steps -- a single linear solve.

    Args:
        estimates: sequence of parameter vectors ``θ_i`` (each shape ``(p,)``), or a stack ``(m, p)``.
        fishers: per-estimate Fisher information. Each may be a scalar/1-D vector (**diagonal** Fisher,
            per-coordinate precision) or a ``(p, p)`` matrix (**full** Fisher). ``None`` uses unit Fisher
            (a plain average). A single value is broadcast to every estimate.

    Returns:
        The merged parameter vector ``θ*`` of shape ``(p,)``.
    """
    thetas = [np.asarray(t, dtype=float).ravel() for t in estimates]
    if not thetas:
        raise ValueError("fisher_merge needs at least one estimate.")
    p = thetas[0].shape[0]
    if any(t.shape[0] != p for t in thetas):
        raise ValueError("all estimates must have the same length.")
    m = len(thetas)

    if fishers is None:
        fs: list[Any] = [np.ones(p) for _ in thetas]
    elif isinstance(fishers, (list, tuple)) and len(fishers) == m:
        fs = [np.asarray(f, dtype=float) for f in fishers]
    else:  # a single Fisher broadcast to every estimate
        fs = [np.asarray(fishers, dtype=float) for _ in thetas]

    full = any(f.ndim == 2 for f in fs)
    if full:
        num = np.zeros(p)
        den = np.zeros((p, p))
        for t, f in zip(thetas, fs):
            fm = np.diag(f) if f.ndim <= 1 else f  # promote diagonal Fishers to matrices
            num += fm @ t
            den += fm
        return np.linalg.solve(den + 1e-12 * np.eye(p), num)
    # diagonal fast path: per-coordinate precision weighting, with a guard for zero total precision
    fs2 = [np.broadcast_to(np.atleast_1d(f), (p,)).astype(float) for f in fs]
    den = np.sum(fs2, axis=0)
    num = np.sum([f * t for t, f in zip(thetas, fs2)], axis=0)
    zero = den <= 0
    out = np.where(zero, np.mean(thetas, axis=0), num / np.where(zero, 1.0, den))
    return out
