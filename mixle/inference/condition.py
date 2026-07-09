"""``condition()`` / ``do()`` -- generic conditioning and causal intervention over any fitted
mixle model, regardless of how it is composed (composite / mixture / HMM / dependency-tree /
Bayesian network / conditional / sequence / optional).

See ``notes/designs/M0.md`` for the full design: the recursive rule per combinator, the
self-normalized-importance-sampling (SIR) fallback and its ESS receipt, and ``do()``'s
graph-surgery semantics. In one line: ``condition`` composes each family's own closed-form
conditioning surface where one already exists (``MultivariateGaussianDistribution.condition``,
``MixtureDistribution.conditional``, ``HiddenMarkovModelDistribution``'s forward-backward) and
falls back to likelihood-weighted ancestral sampling -- reusing each combinator's own
``log_density``/``sampler`` -- everywhere else; ``do`` severs the incoming edges of the assigned
fields (graph surgery) rather than reweighting via Bayes.

Neither this module nor its callers modify any family's internals -- only their existing public
surfaces are composed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork
from mixle.inference.causal import do as _bn_do
from mixle.inference.structure import DependencyTreeDistribution
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.conditional import ConditionalDistribution
from mixle.stats.combinator.optional import OptionalDistribution
from mixle.stats.combinator.sequence import SequenceDistribution
from mixle.stats.compute.posterior import MarkovChainLatentPosterior
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.discrete.point_mass import PointMassDistribution

__all__ = ["FieldPath", "ConditionReceipt", "Posterior", "condition", "do"]

FieldPath = tuple[int, ...]


class _NoExactRule(Exception):
    """Internal: raised when a combinator has no closed-form conditioning rule -- triggers SIR."""


def _norm_path(key: Any) -> FieldPath:
    if isinstance(key, (int, np.integer)):
        return (int(key),)
    return tuple(int(i) for i in key)


def _norm_evidence(evidence: dict[Any, Any]) -> dict[FieldPath, Any]:
    if not evidence:
        raise ValueError("condition()/do() require at least one evidence/assignment field.")
    return {_norm_path(k): v for k, v in evidence.items()}


def _split(evidence: dict[FieldPath, Any]) -> tuple[dict[int, Any], dict[int, dict[FieldPath, Any]]]:
    """Split evidence keyed by FieldPath into this level's direct fields and residual sub-paths."""
    top: dict[int, Any] = {}
    nested: dict[int, dict[FieldPath, Any]] = {}
    for path, v in evidence.items():
        i, rest = path[0], path[1:]
        if rest:
            nested.setdefault(i, {})[rest] = v
        else:
            top[i] = v
    return top, nested


def _safe_log_density(dist: Any, value: Any) -> float:
    """A field's log-density under an evidence value, with out-of-support -> ``-inf`` (not a crash)."""
    try:
        ld = float(dist.log_density(value))
    except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError):
        return float("-inf")
    return ld if not np.isnan(ld) else float("-inf")


def _rng_seed(rng: RandomState) -> int:
    return int(rng.randint(0, 2**31 - 1))


def _is_gaussian_like(model: Any) -> bool:
    """Duck-types a leaf exposing the ``MultivariateGaussianDistribution``-style closed-form API."""
    return (
        callable(getattr(model, "condition", None))
        and callable(getattr(model, "marginal", None))
        and hasattr(model, "mu")
        and hasattr(model, "dim")
    )


def _analytic_mean(dist: Any, j: int | None = None) -> float:
    """The analytic (not Monte-Carlo) mean of a fitted leaf/composite family, or raise."""
    if hasattr(dist, "mu"):
        mu = np.atleast_1d(np.asarray(dist.mu, dtype=float))
        return float(mu[0]) if j is None else float(mu[j])
    mean_fn = getattr(dist, "mean", None)
    if callable(mean_fn):
        m = mean_fn()
        if j is None:
            return float(m)
        return float(np.atleast_1d(np.asarray(m, dtype=float))[j])
    raise NotImplementedError(f"no analytic mean available for {type(dist).__name__}; use SIR + Posterior.sample.")


@dataclass
class ConditionReceipt:
    """What ``condition()`` actually did: the method used and (for SIR) the importance-sampling health."""

    method: str  # "exact" | "sir"
    ess: float | None = None
    ess_ratio: float | None = None
    n_particles: int | None = None
    warnings: list[str] = field(default_factory=list)


class Posterior:
    """A ``condition()`` result: scoreable/samplable over the unobserved fields.

    Distinct from :class:`mixle.stats.compute.posterior.Posterior` (that hierarchy is for
    parameter/latent/predictive posteriors keyed by ``sample(rng)``); this one is evidence
    conditioning within a fitted joint model, with the signature the M0 card specifies:
    ``sample(n)`` / ``log_density(partial_row)`` / ``mean(field)`` / ``.receipt``.
    """

    def __init__(
        self,
        *,
        sample_fn: Callable[[int, int | None], Any],
        log_density_fn: Callable[[Any], float] | None,
        mean_fn: Callable[[FieldPath], Any],
        receipt: ConditionReceipt,
        model: Any = None,
    ) -> None:
        self._sample_fn = sample_fn
        self._log_density_fn = log_density_fn
        self._mean_fn = mean_fn
        self.receipt = receipt
        # The underlying conditioned distribution when the exact path produced one -- lets a caller
        # splice a sub-posterior back into a bigger composite (see CompositeDistribution recursion
        # below). None for SIR posteriors: there is no closed-form distribution object to hand back.
        self.model = model

    def sample(self, n: int = 1, *, seed: int | None = None) -> Any:
        """``n`` draws over the unobserved fields (a list/array in original field order)."""
        return self._sample_fn(int(n), seed)

    def log_density(self, partial_row: Any) -> float:
        """Log-density of an assignment to the unobserved fields under the posterior."""
        if self._log_density_fn is None:
            raise NotImplementedError(f"{type(self).__name__} does not support log_density for this posterior.")
        return self._log_density_fn(partial_row)

    def mean(self, field: FieldPath | int) -> Any:
        """Posterior mean of one unobserved field (same ``FieldPath``/``int`` used in ``evidence``)."""
        return self._mean_fn(_norm_path(field))


# --------------------------------------------------------------------------------------------- #
# condition() -- exact dispatch
# --------------------------------------------------------------------------------------------- #


def condition(
    model: Any,
    evidence: dict[FieldPath | int, Any],
    *,
    method: str = "auto",
    n_particles: int = 4096,
    seed: int | None = None,
) -> Posterior:
    """The posterior over ``model``'s unobserved fields given ``evidence`` (see ``notes/designs/M0.md``)."""
    if method not in ("auto", "exact", "sir"):
        raise ValueError(f"unknown method {method!r}; expected 'auto', 'exact', or 'sir'.")
    ev = _norm_evidence(evidence)
    if method in ("auto", "exact"):
        try:
            return _condition_exact(model, ev, seed=seed)
        except _NoExactRule:
            if method == "exact":
                raise
    return _condition_sir(model, ev, n_particles=int(n_particles), seed=seed)


def _condition_exact(model: Any, ev: dict[FieldPath, Any], *, seed: int | None) -> Posterior:
    if _is_gaussian_like(model):
        return _condition_gaussian_like(model, ev)
    if isinstance(model, CompositeDistribution):
        return _condition_composite(model, ev, seed=seed)
    if isinstance(model, MixtureDistribution):
        return _condition_mixture(model, ev)
    if isinstance(model, HiddenMarkovModelDistribution):
        return _condition_hmm(model, ev)
    raise _NoExactRule(type(model).__name__)


def _condition_gaussian_like(model: Any, ev: dict[FieldPath, Any]) -> Posterior:
    if any(len(p) != 1 for p in ev):
        raise _NoExactRule("nested evidence is not supported for a Gaussian-like leaf")
    observed = {p[0]: v for p, v in ev.items()}
    cond = model.condition(observed)
    unobs = [i for i in range(int(model.dim)) if i not in observed]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        return cond.sampler(seed=s).sample(n)

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> float:
        return _analytic_mean(cond, pos[path[0]])

    receipt = ConditionReceipt(method="exact")
    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=cond)


def _condition_composite(model: CompositeDistribution, ev: dict[FieldPath, Any], *, seed: int | None) -> Posterior:
    top, nested = _split(ev)
    working = list(model.dists)
    sub_posts: dict[int, Posterior] = {}
    for i, sub_ev in nested.items():
        sp = _condition_exact(working[i], sub_ev, seed=seed)
        if sp.model is None:
            raise _NoExactRule("nested composite field has no closed-form posterior to splice back in")
        sub_posts[i] = sp
        working[i] = sp.model
    working_composite = CompositeDistribution(working)
    cond = working_composite.condition(top)
    unobs = [i for i in range(model.count) if i not in top]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        return cond.sampler(seed=s).sample(n)

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> Any:
        i = path[0]
        if len(path) > 1:
            if i in sub_posts:
                return sub_posts[i].mean(path[1:])
            raise NotImplementedError(f"no nested posterior recorded for field {path}")
        return _analytic_mean(cond.dists[pos[i]])

    receipt = ConditionReceipt(method="exact")
    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=cond)


def _condition_mixture(model: MixtureDistribution, ev: dict[FieldPath, Any]) -> Posterior:
    if any(len(p) != 1 for p in ev):
        raise _NoExactRule("nested evidence is not supported by the mixture exact handler")
    observed = {p[0]: v for p, v in ev.items()}
    for c in model.components:
        if not (callable(getattr(c, "marginal", None)) and callable(getattr(c, "condition", None))):
            raise _NoExactRule("a mixture component lacks marginal()/condition()")
    dim = getattr(model.components[0], "dim", None)
    if dim is None:
        raise _NoExactRule("mixture components have no dim attribute")
    cond = model.conditional(observed)
    unobs = [i for i in range(int(dim)) if i not in observed]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        return cond.sampler(seed=s).sample(n)

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> float:
        j = pos[path[0]]
        means = np.array([_analytic_mean(c, j) for c in cond.components], dtype=np.float64)
        return float(np.sum(cond.w * means))

    receipt = ConditionReceipt(method="exact")
    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=cond)


def _condition_hmm(model: HiddenMarkovModelDistribution, ev: dict[FieldPath, Any]) -> Posterior:
    if any(len(p) != 1 for p in ev):
        raise _NoExactRule("nested evidence is not supported by the HMM exact handler")
    observed = {p[0]: v for p, v in ev.items()}
    if not observed:
        raise _NoExactRule("no evidence")
    t_max = max(observed)
    n_states = model.n_states
    log_b = np.zeros((t_max + 1, n_states), dtype=np.float64)
    for t in range(t_max + 1):
        if t in observed:
            for k in range(n_states):
                log_b[t, k] = _safe_log_density(model.topics[k], observed[t])
    q = MarkovChainLatentPosterior(model.log_w, model.log_transitions, log_b)
    marginals = q.marginals()  # (T, K) smoothed state responsibilities

    def sample_fn(n: int, s: int | None) -> Any:
        rng = RandomState(s)
        out = []
        for _ in range(n):
            z = q.sample(rng)
            row: dict[int, Any] = {}
            for t in range(t_max + 1):
                if t in observed:
                    row[t] = observed[t]
                else:
                    row[t] = model.topics[int(z[t])].sampler(seed=_rng_seed(rng)).sample()
            out.append(row)
        return out

    def log_density_fn(partial_row: dict[int, Any]) -> float:
        total = 0.0
        for t, val in partial_row.items():
            w = marginals[t]
            comp_ld = np.array([_safe_log_density(model.topics[k], val) for k in range(n_states)])
            total += float(logsumexp(comp_ld + np.log(w + 1e-300)))
        return total

    def mean_fn(path: FieldPath) -> float:
        t = path[0]
        w = marginals[t]
        means = np.array([_analytic_mean(model.topics[k]) for k in range(n_states)], dtype=np.float64)
        return float(np.sum(w * means))

    receipt = ConditionReceipt(method="exact")
    post = Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=None)
    post.state_marginals = marginals  # convenience for callers wanting q(z_t | evidence) directly
    return post


# --------------------------------------------------------------------------------------------- #
# SIR fallback -- self-normalized likelihood-weighted ancestral sampling
# --------------------------------------------------------------------------------------------- #


def _generate_weighted(model: Any, ev: dict[FieldPath, Any], rng: RandomState) -> tuple[Any, float]:
    """One ``(record, log_weight)`` particle from ``model``'s own generative order, evidence clamped."""
    top, nested = _split(ev)

    if isinstance(model, CompositeDistribution):
        vals: list[Any] = [None] * model.count
        lw = 0.0
        for i in range(model.count):
            child = model.dists[i]
            if i in top:
                vals[i] = top[i]
                lw += _safe_log_density(child, vals[i])
            elif i in nested:
                vals[i], sub_lw = _generate_weighted(child, nested[i], rng)
                lw += sub_lw
            else:
                vals[i] = child.sampler(seed=_rng_seed(rng)).sample()
        return tuple(vals), lw

    if isinstance(model, MixtureDistribution):
        k = int(rng.choice(model.num_components, p=model.w))
        sub_ev = {(i,): v for i, v in top.items()}
        sub_ev.update({(i, *rest): v for i, sub in nested.items() for rest, v in sub.items()})
        return _generate_weighted(model.components[k], sub_ev, rng)

    if isinstance(model, HiddenMarkovModelDistribution):
        if not top:
            raise ValueError("no evidence for HMM SIR fallback: at least one time index must be evidenced")
        t_max = max(top)
        vals = [None] * (t_max + 1)
        lw = 0.0
        state = int(rng.choice(model.n_states, p=np.exp(model.log_w)))
        trans = np.exp(model.log_transitions)
        for t in range(t_max + 1):
            if t > 0:
                state = int(rng.choice(model.n_states, p=trans[state]))
            if t in top:
                vals[t] = top[t]
                lw += _safe_log_density(model.topics[state], vals[t])
            else:
                vals[t] = model.topics[state].sampler(seed=_rng_seed(rng)).sample()
        return vals, lw

    if isinstance(model, DependencyTreeDistribution):
        vals = [None] * len(model.parents)
        lw = 0.0
        for i in model.order:
            parent = model.parents[i]
            fac = model.factors[i]
            if i in top:
                vals[i] = top[i]
                if parent is None:
                    lw += _safe_log_density(fac, vals[i])
                else:
                    lw += _safe_log_density(fac, (model._key(i, vals[parent]), vals[i]))
            else:
                seed = _rng_seed(rng)
                if parent is None:
                    vals[i] = fac.sampler(seed).sample(1)[0]
                else:
                    vals[i] = fac.sampler(seed).sample_given(model._key(i, vals[parent]))
        return tuple(vals), lw

    if isinstance(model, HeterogeneousBayesianNetwork):
        vals = [None] * len(model.factors)
        by_child = {f.child: f for f in model.factors}
        lw = 0.0
        for i in model.order:
            f = by_child[i]
            if i in top:
                vals[i] = top[i]
                lw += _safe_log_density(f, tuple(vals))
            else:
                vals[i] = f.sample(vals, rng)
        return tuple(vals), lw

    if isinstance(model, ConditionalDistribution):
        if 0 in top:
            x0 = top[0]
            lw = _safe_log_density(model.given_dist, x0) if model.has_given else 0.0
        else:
            x0 = model.given_dist.sampler(seed=_rng_seed(rng)).sample() if model.has_given else None
            lw = 0.0
        branch = model.dmap.get(x0, model.default_dist if model.has_default else None)
        if branch is None:
            return (x0, None), float("-inf")
        if 1 in top:
            x1 = top[1]
            lw += _safe_log_density(branch, x1)
        else:
            x1 = branch.sampler(seed=_rng_seed(rng)).sample()
        return (x0, x1), lw

    if isinstance(model, OptionalDistribution):
        if 0 in top:
            v = top[0]
            lw = _safe_log_density(model, v)
        else:
            v = model.sampler(seed=_rng_seed(rng)).sample()
            lw = 0.0
        return v, lw

    if isinstance(model, SequenceDistribution):
        if not top:
            raise ValueError("no evidence for SequenceDistribution SIR fallback: at least one index must be evidenced")
        t_max = max(top)
        vals = []
        lw = 0.0
        for t in range(t_max + 1):
            if t in top:
                v = top[t]
                lw += _safe_log_density(model.dist, v)
            else:
                v = model.dist.sampler(seed=_rng_seed(rng)).sample()
            vals.append(v)
        return vals, lw

    if top or nested:
        raise TypeError(
            f"condition(): {type(model).__name__} has no known field decomposition for SIR conditioning. "
            "Supported combinators: CompositeDistribution, MixtureDistribution, HiddenMarkovModelDistribution, "
            "DependencyTreeDistribution, HeterogeneousBayesianNetwork, ConditionalDistribution, "
            "SequenceDistribution, OptionalDistribution, and Gaussian-like leaves."
        )
    return model.sampler(seed=_rng_seed(rng)).sample(), 0.0


def _extract(record: Any, path: FieldPath) -> Any:
    v = record
    for i in path:
        v = v[i]
    return v


def _condition_sir(model: Any, ev: dict[FieldPath, Any], *, n_particles: int, seed: int | None) -> Posterior:
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    rng = RandomState(seed)
    records: list[Any] = [None] * n_particles
    log_weights = np.empty(n_particles, dtype=np.float64)
    for i in range(n_particles):
        rec, lw = _generate_weighted(model, ev, rng)
        records[i] = rec
        log_weights[i] = lw

    warnings: list[str] = []
    finite = np.isfinite(log_weights)
    if not finite.any():
        w_norm = np.full(n_particles, 1.0 / n_particles)
        ess = 0.0
        warnings.append("all importance weights are zero (evidence has zero density under the prior).")
    else:
        m = log_weights[finite].max()
        w = np.where(finite, np.exp(log_weights - m), 0.0)
        sw = w.sum()
        w_norm = w / sw
        ess = float(1.0 / np.sum(w_norm**2))
    ess_ratio = ess / n_particles
    if ess_ratio < 0.01:
        warnings.append(
            f"ESS ratio {ess_ratio:.4f} < 0.01 threshold -- evidence may be near-impossible under the prior."
        )
    receipt = ConditionReceipt(method="sir", ess=ess, ess_ratio=ess_ratio, n_particles=n_particles, warnings=warnings)

    def sample_fn(n: int, s: int | None) -> Any:
        r = RandomState(s) if s is not None else RandomState(_rng_seed(rng))
        idx = r.choice(n_particles, size=n, replace=True, p=w_norm)
        return [records[j] for j in idx]

    def mean_fn(path: FieldPath) -> float:
        vals = np.array([float(_extract(records[j], path)) for j in range(n_particles)], dtype=np.float64)
        return float(np.sum(w_norm * vals))

    def log_density_fn(partial_row: dict[FieldPath | int, Any]) -> float:
        return _weighted_kde_log_density(records, w_norm, {_norm_path(k): v for k, v in partial_row.items()})

    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=None)


def _weighted_kde_log_density(records: Sequence[Any], w_norm: np.ndarray, partial_row: dict[FieldPath, Any]) -> float:
    """A self-normalized-weight-KDE estimate of the SIR posterior's log-density (Silverman bandwidth).

    This is an ESTIMATE, not exact -- there is no closed-form density for a generic SIR posterior.
    """
    paths = sorted(partial_row)
    x0 = np.array([float(partial_row[p]) for p in paths], dtype=np.float64)
    x = np.array([[float(_extract(r, p)) for p in paths] for r in records], dtype=np.float64)
    n, d = x.shape
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    bw = std * 1.06 * (n ** (-1.0 / (d + 4)))
    diffs = (x - x0) / bw
    log_kernel = -0.5 * np.sum(diffs**2, axis=1) - np.sum(np.log(bw)) - 0.5 * d * np.log(2.0 * np.pi)
    return float(logsumexp(log_kernel + np.log(w_norm + 1e-300)))


# --------------------------------------------------------------------------------------------- #
# do() -- causal intervention (graph surgery)
# --------------------------------------------------------------------------------------------- #


def do(model: Any, assignments: dict[FieldPath | int, Any]) -> Any:
    """Sever the incoming edges of the assigned fields, then clamp them (Pearl's ``do``).

    Returns a model of the same combinator family wherever possible (``DependencyTreeDistribution``,
    ``CompositeDistribution``, ``MixtureDistribution``) so it can be passed back through
    ``condition()``/``do()``; for a ``HeterogeneousBayesianNetwork`` it returns the existing
    :class:`~mixle.inference.causal.InterventionalNetwork` (sample/expectation/distribution).
    """
    ev = _norm_evidence(assignments)
    return _do_dispatch(model, ev)


def _do_dispatch(model: Any, ev: dict[FieldPath, Any]) -> Any:
    if isinstance(model, DependencyTreeDistribution):
        return _do_dependency_tree(model, ev)
    if isinstance(model, HeterogeneousBayesianNetwork):
        top, nested = _split(ev)
        if nested:
            raise NotImplementedError("do() on nested fields of a HeterogeneousBayesianNetwork is not supported.")
        return _bn_do(model, top)
    if isinstance(model, CompositeDistribution):
        return _do_composite(model, ev)
    if isinstance(model, MixtureDistribution):
        return _do_mixture(model, ev)
    raise TypeError(f"do() has no graph-surgery rule for {type(model).__name__}.")


def _do_dependency_tree(model: DependencyTreeDistribution, ev: dict[FieldPath, Any]) -> DependencyTreeDistribution:
    top, nested = _split(ev)
    if nested:
        raise NotImplementedError("do() on nested fields of a DependencyTreeDistribution is not supported.")
    new_parents = list(model.parents)
    new_factors = list(model.factors)
    new_binners = list(model.binners)
    for i, v in top.items():
        new_parents[i] = None  # sever the incoming edge
        new_factors[i] = PointMassDistribution(v)  # clamp
        new_binners[i] = None
    return DependencyTreeDistribution(new_parents, new_factors, new_binners)


def _do_composite(model: CompositeDistribution, ev: dict[FieldPath, Any]) -> CompositeDistribution:
    top, nested = _split(ev)
    new_dists = list(model.dists)
    for i, sub_ev in nested.items():
        new_dists[i] = _do_dispatch(new_dists[i], sub_ev)
    for i, v in top.items():
        new_dists[i] = PointMassDistribution(v)  # a composite has no internal edges to sever
    return CompositeDistribution(new_dists)


def _do_mixture(model: MixtureDistribution, ev: dict[FieldPath, Any]) -> MixtureDistribution:
    # do() severs each component's incoming edges but -- unlike condition() -- keeps the ORIGINAL
    # mixture weights: an intervention carries no Bayesian evidence about which component generated it.
    new_components = [_do_dispatch(c, ev) for c in model.components]
    return MixtureDistribution(new_components, model.w.copy())
