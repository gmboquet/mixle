"""Operations — the verbs that transform distributions, and how they change capabilities.

The missing third axis of pysp: alongside *objects* (distributions) and *capabilities* (what they
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

See ``docs/ARCHITECTURE.md`` and the operation table in ``docs/CAPABILITIES.md``.
"""

from __future__ import annotations

from typing import Any

from pysp.capability import CapabilityError, Conditionable, ExponentialFamily, Marginalizable, require

__all__ = ["quantize", "truncate", "condition", "marginalize", "mixture", "transform", "tilt", "project"]


def quantize(
    dist: Any, bits: int = 8, *, lo_q: float = 1e-3, hi_q: float = 1.0 - 1e-3, n_samples: int = 200_000, seed: int = 0
):
    """Discretize a continuous ``dist`` into a finite distribution over ``2**bits`` bins.

    Capability signature: ``any Distribution -> FiniteSupport · Enumerable · RankableByIndex``. The
    result is a :class:`CategoricalDistribution` over bin midpoints; the bin masses are exact (via the
    base ``cdf`` when available) or sampled otherwise. This is the concrete answer to "if I quantize a
    distribution, what does it become and how" — ``describe(quantize(g))`` shows the gained capabilities.
    """
    import numpy as np

    from pysp.stats.base.categorical import CategoricalDistribution

    n = 1 << int(bits)
    # bracket the bulk of the mass
    if callable(getattr(dist, "density_quantile", None)):
        lo = float(dist.density_quantile(lo_q, seed=seed))
        hi = float(dist.density_quantile(hi_q, seed=seed))
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
    from pysp.stats.combinator.truncated import TruncatedDistribution

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
    from pysp.stats.latent.mixture import MixtureDistribution

    dists = list(dists)
    if w is None:
        w = [1.0 / len(dists)] * len(dists)
    return MixtureDistribution(dists, w)


def transform(dist: Any, f: Any):
    """Change of variables ``y = f(x)`` with a Jacobian-corrected density (``f`` is a ``Transform``)."""
    from pysp.stats.combinator.transform import TransformDistribution

    return TransformDistribution(dist, f)


def tilt(dist: Any, theta: Any):
    """Exponentially tilt ``dist`` by ``theta``. Requires the ``ExponentialFamily`` capability."""
    require(dist, ExponentialFamily, "tilt")
    from pysp.stats.combinator.exponential_tilt import ExponentialTiltedDistribution

    return ExponentialTiltedDistribution(dist, theta)


def project(source: Any, target: Any, *, n_samples: int = 20_000, seed: int = 0, max_its: int = 50):
    """Variationally project ``source`` onto the family of ``target`` (the sample-based M-projection).

    Capability signature: ``Sampleable source x Fittable target -> Distribution``. Draws ``n_samples``
    from ``source`` and fits ``target`` to them by maximum likelihood -- which is exactly the projection
    minimizing the forward divergence ``KL(source || target_family)`` (the M-/moment projection). Works
    for any ``source`` exposing a ``sampler`` (a distribution, mixture, GP, or a trained neural model)
    onto any fittable ``target`` family, so e.g. a neural sequence model can be distilled onto an HMM.

    ``target`` may be a distribution (its :meth:`estimator` supplies the family) or an estimator directly.
    The returned model is a member of the target family; ``describe(project(...))`` shows its capabilities.
    """
    import numpy as np

    from pysp.inference.estimation import fit  # the EM driver -- import from its module, not the package
    # (`from pysp.inference import fit` is ambiguous: the `pysp.inference.fit` gradient-fitting submodule
    # shadows the re-exported function once that submodule is imported anywhere in the process).

    if not callable(getattr(source, "sampler", None)):
        raise CapabilityError("project requires a sampleable source (a `.sampler(seed)` method).")
    estimator = target.estimator() if callable(getattr(target, "estimator", None)) else target
    data = list(source.sampler(seed).sample(int(n_samples)))
    return fit(data, estimator, max_its=max_its, rng=np.random.RandomState(seed), out=None)
