"""Operations — the verbs that transform distributions, and how they change capabilities.

The missing third axis of mixle: alongside *objects* (distributions) and *capabilities* (what they
support), the **operations** that move an object from one capability set to another. Each verb
documents its **capability signature** — what it requires of its input and what the output gains — so
"what can I do to a distribution, and what does it become?" has one home.

    quantize(dist, bits)      : any Distribution      -> FiniteSupport · Enumerable · RankableByIndex
    truncate(dist, ...)       : Distribution          -> Distribution (Enumerable preserved)
    condition(dist, observed) : Conditionable         -> Distribution
    marginalize(dist, keep)   : Marginalizable        -> Distribution
    mixture(dists, w)         : Distributions         -> LatentStructured
    transform(dist, f)        : Distribution + inv. f -> Distribution (Jacobian-corrected)
    tilt(dist, theta)         : ExponentialFamily     -> Distribution
    project(source, target)   : Sampleable x Fittable -> Distribution (forward-KL M-projection)
    product_of_experts(dists) : tractable family      -> Distribution (geometric/log-linear pool)

See ``docs/ARCHITECTURE.md`` and the operation table in ``docs/CAPABILITIES.md``.
"""

from __future__ import annotations

from typing import Any

from mixle.capability import CapabilityError, Conditionable, ExponentialFamily, Marginalizable, require

__all__ = [
    "quantize",
    "truncate",
    "condition",
    "marginalize",
    "mixture",
    "transform",
    "tilt",
    "project",
    "product_of_experts",
]


def quantize(
    dist: Any, bits: int = 8, *, lo_q: float = 1e-3, hi_q: float = 1.0 - 1e-3, n_samples: int = 200_000, seed: int = 0
):
    """Discretize a continuous ``dist`` into a finite distribution over ``2**bits`` bins.

    Capability signature: ``any Distribution -> FiniteSupport · Enumerable · RankableByIndex``. The
    result is a :class:`CategoricalDistribution` over bin midpoints; the bin masses are exact (via the
    base ``cdf`` when available) or sampled otherwise. This is the concrete answer to "if I quantize a
    distribution, what does it become and how" — ``describe(quantize(g))`` shows the gained capabilities.

    **Target measure.** ``[lo_q, hi_q]`` brackets where the bins are placed, not which mass survives:
    the whole distribution is quantized, with mass below the bracket assigned to the first bin and
    mass above it to the last -- the clipped image of ``dist`` under the bin map. The exact CDF route
    used to instead drop both tails and renormalize the interior to one, i.e. silently return the
    distribution *conditional* on the bracket, while the sampling fallback clipped its draws into the
    edge bins. The two advertised implementations therefore targeted different distributions: with
    ``lo_q=.25, hi_q=.75, bits=2`` on a uniform, the CDF route returned four masses of ``.25`` where
    the clipped answer is ``.375/.125/.125/.375``. Both routes now agree on the clipped measure.
    """
    import numpy as np

    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"bits must be a positive integer, got {bits!r}")
    if not (np.isfinite(lo_q) and np.isfinite(hi_q)) or not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(f"quantile bracket must satisfy 0 <= lo_q < hi_q <= 1, got ({lo_q!r}, {hi_q!r})")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    n = 1 << bits
    # bracket the bulk of the mass with the spatial quantile (inverse CDF) when the family has one,
    # else an empirical quantile of a sample. NOT ``density_quantile``: that index is descending-DENSITY
    # order (q=0 is the mode, q -> 1 walks into a tail on either side), so it cannot bracket a support.
    if callable(getattr(dist, "quantile", None)):
        lo, hi = float(dist.quantile(lo_q)), float(dist.quantile(hi_q))
    else:
        s = np.asarray(dist.sampler(seed).sample(n_samples), dtype=float).ravel()
        lo, hi = float(np.quantile(s, lo_q)), float(np.quantile(s, hi_q))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return CategoricalDistribution({float(lo if np.isfinite(lo) else 0.0): 1.0})

    edges = np.linspace(lo, hi, n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    if callable(getattr(dist, "cdf", None)):
        cvals = np.array([float(dist.cdf(float(e))) for e in edges])
        p = np.diff(cvals)
        # Account for the bracketed-out tails instead of discarding them: the lower tail F(lo) goes
        # to the first bin and the upper tail 1 - F(hi) to the last, exactly what the sampling
        # fallback's np.clip does. Without this the interior was renormalized to one and the result
        # was the conditional distribution inside the bracket, not the quantized distribution.
        p[0] += max(0.0, float(cvals[0]))
        p[-1] += max(0.0, 1.0 - float(cvals[-1]))
    else:
        s = np.asarray(dist.sampler(seed).sample(n_samples), dtype=float).ravel()
        p = np.histogram(np.clip(s, edges[0], edges[-1]), bins=edges)[0].astype(float)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return CategoricalDistribution({float(mids[len(mids) // 2]): 1.0})
    p /= total
    return CategoricalDistribution({float(m): float(pi) for m, pi in zip(mids, p) if pi > 0.0})


def truncate(dist: Any, *, allowed: Any = None, forbidden: Any = None):
    """Restrict ``dist`` to (``allowed``) or away from (``forbidden``) a finite set; mass renormalizes."""
    from mixle.stats.combinator.truncated import TruncatedDistribution

    return TruncatedDistribution(dist, allowed=allowed, forbidden=forbidden)


def condition(dist: Any, observed: dict[int, float]):
    """Condition ``dist`` on a subset of coordinates. Requires the ``Conditionable`` capability."""
    require(dist, Conditionable, "condition")
    return dist.condition(observed)


def marginalize(dist: Any, keep: Any):
    """Marginalize ``dist`` to the kept coordinates. Requires the ``Marginalizable`` capability."""
    require(dist, Marginalizable, "marginalize")
    return dist.marginal(keep)


def mixture(dists: Any, w: Any = None):
    """Weighted mixture of ``dists`` — a latent-variable model (``LatentStructured``)."""
    from mixle.stats.latent.mixture import MixtureDistribution

    dists = list(dists)
    if w is None:
        w = [1.0 / len(dists)] * len(dists)
    return MixtureDistribution(dists, w)


def transform(dist: Any, f: Any):
    """Change of variables ``y = f(x)`` with a Jacobian-corrected density (``f`` is a ``Transform``)."""
    from mixle.stats.combinator.transform import TransformDistribution

    return TransformDistribution(dist, f)


def tilt(dist: Any, theta: Any):
    """Exponentially tilt ``dist`` by ``theta``. Requires the ``ExponentialFamily`` capability."""
    require(dist, ExponentialFamily, "tilt")
    from mixle.stats.combinator.exponential_tilt import ExponentialTiltedDistribution

    return ExponentialTiltedDistribution(dist, theta)


def project(source: Any, target: Any, *, n_samples: int = 20_000, seed: int = 0, max_its: int = 50):
    """Variationally project ``source`` onto the family of ``target`` (the sample-based M-projection).

    Capability signature: ``Sampleable source x Fittable target -> Distribution``. Draws ``n_samples``
    from ``source`` and fits ``target`` to them by maximum likelihood. Works for any ``source`` exposing
    a ``sampler`` (a distribution, mixture, GP, or a trained neural model) onto any fittable ``target``
    family, so e.g. a neural sequence model can be distilled onto an HMM.

    **This is the empirical M-projection, not the exact one.** Maximum likelihood on the *population*
    would minimize the forward divergence ``KL(source || target_family)`` exactly; what runs here is
    maximum likelihood on one finite ``n_samples``-sized Monte Carlo draw, optimized by an iterative
    fitter that is capped at ``max_its`` and, for latent-variable families, only guaranteed to reach a
    local optimum. The result therefore carries sampling error (of order ``1/sqrt(n_samples)``, not
    estimated or reported here), optimization error, and whatever misspecification the target family
    imposes. It converges to the exact projection as ``n_samples`` grows, and nothing in the returned
    object certifies how close this particular fit got. Treat it as an approximation and validate it
    (e.g. score held-out ``source`` draws under the result) before claiming a KL guarantee.

    ``target`` may be a distribution (its :meth:`estimator` supplies the family) or an estimator directly.
    The returned model is a member of the target family; ``describe(project(...))`` shows its capabilities.
    """
    import numpy as np

    from mixle.inference.estimation import fit  # the EM driver -- import from its module to be safe

    if not callable(getattr(source, "sampler", None)):
        raise CapabilityError("project requires a sampleable source (a `.sampler(seed)` method).")
    # The sample budget is the whole accuracy story of an empirical projection, so it is taken
    # exactly as given rather than truncated through int(): 2.9 silently became a two-sample
    # projection, and 0 or a negative count reached the fitter as "no data" with a message about
    # empty input rather than about the budget the caller actually passed.
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer number of draws, got {n_samples!r}")
    if isinstance(max_its, bool) or not isinstance(max_its, int) or max_its < 1:
        raise ValueError(f"max_its must be a positive integer, got {max_its!r}")
    estimator = target.estimator() if callable(getattr(target, "estimator", None)) else target
    data = list(source.sampler(seed).sample(n_samples))
    if len(data) != n_samples:
        raise ValueError(f"source sampler returned {len(data)} draws for n_samples={n_samples}")
    return fit(data, estimator, max_its=max_its, rng=np.random.RandomState(seed), out=None)


def product_of_experts(dists: Any, weights: Any = None):
    """Geometric (log-linear) pooling of densities — a Product of Experts.

    Capability signature: ``tractable family -> Distribution``. Multiplies the expert densities,
    ``p(x) ∝ ∏_k p_k(x)**w_k`` i.e. ``log p(x) = Σ_k w_k·log p_k(x) − log Z``, with raw exponent
    weights ``w_k`` (default ``1.0`` each — *not* normalized to sum to one, since PoE weights are
    exponents). Only the cases with a tractable normalizer are constructed exactly; the general
    continuous case raises :class:`~mixle.capability.CapabilityError`.

    * **Categorical / finite shared support** (the LLM-vocab fusion case): exact over the intersection
      of the *weighted* experts' supports — ``new_pmap[x] ∝ ∏_k p_k(x)**w_k`` for ``x`` with positive
      mass under every expert carrying a nonzero exponent, then renormalized. Returns a
      :class:`CategoricalDistribution`.
    * **Gaussian** (closed form): a Gaussian with precision-weighted combination
      ``1/σ² = Σ_k w_k/σ_k²`` and ``μ = σ²·Σ_k w_k·μ_k/σ_k²``. Returns a :class:`GaussianDistribution`.

    A ``w_k = 0`` expert contributes the constant ``p_k(x)**0 = 1`` and so is not in the pool at all;
    it is dropped before the support intersection is taken, exactly as it drops out of the Gaussian
    route's precision sum. (It used to be intersected first: an expert supported only on ``{"a"}`` at
    weight zero deleted every other label from a weight-one expert's answer.)

    Args:
        dists: the experts to pool (an iterable of distributions, all of the same tractable kind).
        weights: per-expert exponents ``w_k``; ``None`` (default) uses ``1.0`` for each expert. Each
            must be a finite real number; negative exponents are permitted (a "divide by this expert"
            pool) but are the caller's responsibility to keep proper.

    Raises:
        CapabilityError: if the experts have no shared finite support and are not all Gaussian — the
            PoE normalizer ``Z = ∫ ∏_k p_k(x)**w_k dx`` is then intractable in closed form — or if
            every exponent is zero, which leaves the constant function 1 rather than a distribution.
    """
    import numpy as np

    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    dists = list(dists)
    if not dists:
        raise CapabilityError("product_of_experts needs at least one expert distribution.")
    if weights is None:
        weights = [1.0] * len(dists)
    else:
        weights = [float(w) for w in weights]
    if len(weights) != len(dists):
        raise ValueError("product_of_experts: len(weights) must match len(dists).")
    if not all(np.isfinite(w) for w in weights):
        raise ValueError(f"product_of_experts: every weight must be a finite exponent, got {weights!r}")

    # An exponent of zero raises its expert's density to the constant 1: that expert is simply not
    # part of the pool, and must not constrain the support or the normalizer either. Drop it here so
    # both branches below see only the experts that actually participate.
    active = [(w, d) for w, d in zip(weights, dists) if w != 0.0]
    if not active:
        raise CapabilityError(
            "product_of_experts: every expert weight is zero, so the pool is the constant function "
            "1 rather than a distribution. Give at least one expert a nonzero exponent."
        )

    # --- Categorical / finite shared support: exact PoE over the support intersection ---------------
    if all(isinstance(d, CategoricalDistribution) for d in dists):
        if any(d.no_default for _, d in active):
            raise CapabilityError(
                "product_of_experts requires categoricals with default_value=0 (a closed finite "
                "support); a non-zero default makes the shared support unbounded."
            )
        # the support intersection: labels with strictly positive mass under every participating expert
        support = set(k for k, v in active[0][1].pmap.items() if v > 0.0)
        for _, d in active[1:]:
            support &= set(k for k, v in d.pmap.items() if v > 0.0)
        if not support:
            raise CapabilityError(
                "product_of_experts over categoricals has empty support intersection — the pooled "
                "density is zero everywhere (no label has positive mass under every expert)."
            )
        # log p(x) = Σ_k w_k·log p_k(x); normalize on the shared support
        log_unnorm = {}
        for x in support:
            log_unnorm[x] = sum(w * np.log(d.pmap[x]) for w, d in active)
        m = max(log_unnorm.values())
        unnorm = {x: np.exp(lp - m) for x, lp in log_unnorm.items()}
        total = sum(unnorm.values())
        pmap = {x: float(v / total) for x, v in unnorm.items()}
        return CategoricalDistribution(pmap=pmap)

    # --- Gaussian: closed-form precision-weighted combination --------------------------------------
    if all(isinstance(d, GaussianDistribution) for d in dists):
        precision = sum(w / d.sigma2 for w, d in active)
        if precision <= 0.0 or not np.isfinite(precision):
            raise CapabilityError(
                "product_of_experts over Gaussians has non-positive pooled precision "
                "(Σ_k w_k/σ_k² ≤ 0); the pool is not a proper Gaussian. Use positive weights."
            )
        sigma2 = 1.0 / precision
        mu = sigma2 * sum(w * d.mu / d.sigma2 for w, d in active)
        return GaussianDistribution(float(mu), float(sigma2))

    raise CapabilityError(
        "product_of_experts has no tractable normalizer for these experts: the closed-form pool is "
        "implemented for categoricals over a shared finite support and for Gaussians. A general "
        "(non-Gaussian, non-shared-finite-support) continuous pool needs the intractable integral "
        "Z = ∫ ∏_k p_k(x)**w_k dx; pool via sampling/MCMC instead."
    )
