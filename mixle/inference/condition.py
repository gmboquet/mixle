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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

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

__all__ = ["FieldPath", "ConditionReceipt", "ImpossibleEvidenceError", "Posterior", "condition", "do"]

FieldPath = tuple[int, ...]


class _NoExactRule(Exception):
    """Internal: raised when a combinator has no closed-form conditioning rule -- triggers SIR."""


class ImpossibleEvidenceError(ValueError):
    """Raised when an operation requires a posterior for evidence with zero model probability."""


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


def _valid_top_level_indices(model: Any) -> set[int] | None:
    """The set of valid top-level evidence/assignment field indices for ``model``'s own combinator
    convention, or ``None`` when any non-negative index is meaningful (open-ended sequence models:
    :class:`HiddenMarkovModelDistribution` time steps, :class:`SequenceDistribution` steps, or an
    otherwise-unrecognized model -- left to whatever more specific check its own handler makes)."""
    if _is_gaussian_like(model):
        return set(range(int(model.dim)))
    if isinstance(model, CompositeDistribution):
        return set(range(model.count))
    if isinstance(model, MixtureDistribution):
        dim = getattr(model.components[0], "dim", None)
        return set(range(int(dim))) if dim is not None else None
    if isinstance(model, (HiddenMarkovModelDistribution, SequenceDistribution)):
        return None
    if isinstance(model, DependencyTreeDistribution):
        return set(range(len(model.parents)))
    if isinstance(model, HeterogeneousBayesianNetwork):
        return {f.child for f in model.factors}
    if isinstance(model, ConditionalDistribution):
        return {0, 1}
    if isinstance(model, OptionalDistribution):
        return {0}
    return None


def _check_field_indices(model: Any, ev: dict[FieldPath, Any]) -> None:
    """Raise a clear ``ValueError`` if ``ev`` names a top-level field index that does not exist on
    ``model``, instead of each caller's own ad hoc failure mode on a bad index: ``condition()``'s
    exact path silently no-ops (``unobs``/``cond`` never touch the phantom key, so the result is
    indistinguishable from an unconditioned prior), its SIR fallback never once consults it either
    (every particle's importance weight comes out identically 1.0, and the receipt reports a
    deceptively HEALTHY ``ess_ratio == 1.0`` despite zero evidence actually being applied), and
    ``do()`` crashes with a raw, unhelpful ``IndexError`` deep inside a list assignment -- three
    different failure modes for the same mistake, two of them silent. Called at every recursive
    entry point (:func:`_condition_exact`, :func:`_generate_weighted`, :func:`_do_dispatch`), so a
    bad index nested inside a composite/mixture sub-path is caught at the point it is actually
    consumed, not just when it is the top-level evidence dict.
    """
    valid = _valid_top_level_indices(model)
    for path in ev:
        i = path[0]
        if i < 0 or (valid is not None and i not in valid):
            allowed = f"valid indices are {sorted(valid)}" if valid is not None else "index must be >= 0"
            raise ValueError(
                f"evidence/assignment field index {i} does not exist on this {type(model).__name__} ({allowed})."
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
    evidence_status: str = "possible"  # "possible" | "impossible"
    log_evidence: float | None = None
    sample_contract: str = "full_record"
    ess: float | None = None
    ess_ratio: float | None = None
    n_particles: int | None = None
    warnings: list[str] = field(default_factory=list)


class Posterior:
    """A ``condition()`` result with one sampling contract across exact and approximate paths.

    Distinct from :class:`mixle.stats.compute.posterior.Posterior` (that hierarchy is for
    parameter/latent/predictive posteriors keyed by ``sample(rng)``); this one is evidence
    conditioning within a fitted joint model, with the signature the M0 card specifies:
    ``sample(n)`` / ``log_density(partial_row)`` / ``mean(field)`` / ``.receipt``.

    ``sample(n)`` always returns complete records in the base model's native record type, with
    evidence fields present and clamped. ``log_density`` scores assignments to unobserved fields
    only when the selected conditioning implementation has a valid density with respect to their
    base measure. Generic SIR intentionally does not invent one for mixed or categorical records.
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

    @property
    def possible(self) -> bool:
        """Whether the evidence has nonzero probability under the conditioning model."""
        return self.receipt.evidence_status == "possible"

    def _require_possible(self) -> None:
        if not self.possible:
            raise ImpossibleEvidenceError("Cannot use a posterior for evidence with zero model probability.")

    def sample(self, n: int = 1, *, seed: int | None = None) -> Any:
        """Draw ``n`` complete native records, including the clamped evidence fields."""
        self._require_possible()
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        return self._sample_fn(n, seed)

    def log_density(self, partial_row: Any) -> float:
        """Log-density of an assignment to the unobserved fields under the posterior."""
        self._require_possible()
        if self._log_density_fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not have a valid generic log_density for this posterior's base measure."
            )
        return self._log_density_fn(partial_row)

    def mean(self, field: FieldPath | int) -> Any:
        """Posterior mean of one unobserved field (same ``FieldPath``/``int`` used in ``evidence``)."""
        self._require_possible()
        return self._mean_fn(_norm_path(field))


def _impossible_posterior(
    *,
    method: str,
    warning: str,
    n_particles: int | None = None,
) -> Posterior:
    """Return an explicit, inspectable result for zero-probability evidence."""
    receipt = ConditionReceipt(
        method=method,
        evidence_status="impossible",
        log_evidence=float("-inf"),
        ess=0.0 if n_particles is not None else None,
        ess_ratio=0.0 if n_particles is not None else None,
        n_particles=n_particles,
        warnings=[warning],
    )
    return Posterior(
        sample_fn=lambda _n, _seed: None,
        log_density_fn=None,
        mean_fn=lambda _path: None,
        receipt=receipt,
        model=None,
    )


def _restore_observed_records(samples: Any, n: int, dim: int, observed: dict[int, Any]) -> Any:
    """Expand compact conditional draws back into complete base-model-native records."""
    unobserved = [i for i in range(dim) if i not in observed]
    if not isinstance(samples, np.ndarray):
        complete_rows = []
        for compact in samples:
            compact_values = list(compact) if isinstance(compact, (list, tuple, np.ndarray)) else [compact]
            slots = dict(observed)
            slots.update(zip(unobserved, compact_values))
            row = [slots[i] for i in range(dim)]
            complete_rows.append(np.asarray(row) if isinstance(compact, np.ndarray) else tuple(row))
        return complete_rows
    compact = np.asarray(samples)
    if compact.ndim == 1:
        compact = compact.reshape(n, len(unobserved))
    dtype = np.result_type(compact.dtype, np.asarray(list(observed.values())).dtype)
    complete = np.empty((n, dim), dtype=dtype)
    for i, value in observed.items():
        complete[:, i] = value
    for j, i in enumerate(unobserved):
        complete[:, i] = compact[:, j]
    return complete


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
    _check_field_indices(model, ev)
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
    observed_indices = sorted(observed)
    observed_value = np.asarray([observed[i] for i in observed_indices], dtype=np.float64)
    log_evidence = _safe_log_density(model.marginal(observed_indices), observed_value)
    if log_evidence == float("-inf"):
        return _impossible_posterior(
            method="exact",
            warning="evidence has zero density under the Gaussian-like model.",
        )
    cond = model.condition(observed)
    unobs = [i for i in range(int(model.dim)) if i not in observed]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        return _restore_observed_records(cond.sampler(seed=s).sample(n), n, int(model.dim), observed)

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> float:
        return _analytic_mean(cond, pos[path[0]])

    receipt = ConditionReceipt(method="exact", log_evidence=log_evidence)
    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=cond)


def _condition_composite(model: CompositeDistribution, ev: dict[FieldPath, Any], *, seed: int | None) -> Posterior:
    top, nested = _split(ev)
    working = list(model.dists)
    sub_posts: dict[int, Posterior] = {}
    log_evidence = 0.0
    for i, sub_ev in nested.items():
        sp = _condition_exact(working[i], sub_ev, seed=seed)
        if not sp.possible:
            return _impossible_posterior(
                method="exact",
                warning=f"nested evidence for composite field {i} has zero probability.",
            )
        if sp.model is None:
            raise _NoExactRule("nested composite field has no closed-form posterior to splice back in")
        sub_posts[i] = sp
        working[i] = sp.model
        if sp.receipt.log_evidence is not None:
            log_evidence += sp.receipt.log_evidence
    for i, value in top.items():
        field_log_evidence = _safe_log_density(model.dists[i], value)
        if field_log_evidence == float("-inf"):
            return _impossible_posterior(
                method="exact",
                warning=f"evidence for composite field {i} has zero probability.",
            )
        log_evidence += field_log_evidence
    working_composite = CompositeDistribution(working)
    cond = working_composite.condition(top)
    unobs = [i for i in range(model.count) if i not in top]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        rng = RandomState(s)
        columns: list[list[Any] | np.ndarray] = []
        for i, child in enumerate(model.dists):
            if i in top:
                columns.append([top[i]] * n)
            elif i in sub_posts:
                columns.append(sub_posts[i].sample(n, seed=_rng_seed(rng)))
            else:
                columns.append(child.sampler(seed=_rng_seed(rng)).sample(n))
        return list(zip(*columns))

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> Any:
        i = path[0]
        if len(path) > 1:
            if i in sub_posts:
                return sub_posts[i].mean(path[1:])
            raise NotImplementedError(f"no nested posterior recorded for field {path}")
        return _analytic_mean(cond.dists[pos[i]])

    receipt = ConditionReceipt(method="exact", log_evidence=log_evidence)
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
    observed_indices = sorted(observed)
    observed_values = [observed[i] for i in observed_indices]
    try:
        observed_value: Any = np.asarray(observed_values, dtype=np.float64)
    except (TypeError, ValueError):
        observed_value = tuple(observed_values)
    component_log_evidence = np.asarray(
        [
            model.log_w[k] + _safe_log_density(component.marginal(observed_indices), observed_value)
            for k, component in enumerate(model.components)
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(component_log_evidence)
    if not finite.any():
        return _impossible_posterior(
            method="exact",
            warning="evidence has zero probability under every positive-weight mixture component.",
        )
    m = float(component_log_evidence[finite].max())
    log_evidence = float(m + np.log(np.exp(component_log_evidence[finite] - m).sum()))
    cond = model.conditional(observed)
    unobs = [i for i in range(int(dim)) if i not in observed]
    pos = {f: j for j, f in enumerate(unobs)}

    def sample_fn(n: int, s: int | None) -> Any:
        return _restore_observed_records(cond.sampler(seed=s).sample(n), n, int(dim), observed)

    def log_density_fn(row: Any) -> float:
        return float(cond.log_density(row))

    def mean_fn(path: FieldPath) -> float:
        j = pos[path[0]]
        means = np.array([_analytic_mean(c, j) for c in cond.components], dtype=np.float64)
        return float(np.sum(cond.w * means))

    receipt = ConditionReceipt(method="exact", log_evidence=log_evidence)
    return Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=cond)


def _condition_hmm(model: HiddenMarkovModelDistribution, ev: dict[FieldPath, Any]) -> Posterior:
    if any(len(p) != 1 for p in ev):
        raise _NoExactRule("nested evidence is not supported by the HMM exact handler")
    observed = {int(p[0]): v for p, v in ev.items()}
    if not observed:
        raise _NoExactRule("no evidence")
    t_max = max(observed)
    n_states = model.n_states

    def _emission_log_b(fields: dict[int, Any], horizon: int) -> np.ndarray:
        """Per-time emission log-likelihood matrix ``(horizon, K)``; unevidenced times stay 0."""
        log_b = np.zeros((horizon, n_states), dtype=np.float64)
        for t, val in fields.items():
            for k in range(n_states):
                log_b[t, k] = _safe_log_density(model.topics[k], val)
        return log_b

    q = MarkovChainLatentPosterior(model.log_w, model.log_transitions, _emission_log_b(observed, t_max + 1))
    log_z = q.log_likelihood()  # log p(evidence), the forward normalizer
    if not np.isfinite(log_z):
        return _impossible_posterior(
            method="exact",
            warning="HMM evidence has zero probability under every latent-state path.",
        )
    marginals = q.marginals()  # (T, K) smoothed state responsibilities

    def _state_marginals_at(t: int) -> np.ndarray:
        """``p(z_t | evidence)`` at any time, extending past the last evidenced time by prediction."""
        if t < 0:
            raise ValueError(f"HMM field index must be a non-negative time step; got {t}.")
        if t <= t_max:
            return marginals[t]
        # Beyond the last evidenced time the smoothed chain is a pure prediction: evidence only
        # touches times <= t_max, so p(z_t | evidence) = p(z_{t_max} | evidence) A^(t - t_max).
        w = marginals[t_max]
        trans = np.exp(model.log_transitions)
        for _ in range(t - t_max):
            w = w @ trans
        return w

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
        # The EXACT joint conditional the "exact" label promises (matching sample_fn's own joint
        # draws): log p(query | evidence) = log p(query, evidence) - log p(evidence), both by the
        # forward algorithm -- NOT a sum of per-time smoothed-marginal predictives, which is the
        # product of marginals and ignores the latent correlation between query times. Query times
        # past the last evidenced time simply extend the forward chain (unevidenced rows marginalize
        # out), so any non-negative time is scoreable.
        query = {int(t): val for t, val in partial_row.items()}
        if not query:
            return 0.0
        if min(query) < 0:
            raise ValueError(f"HMM field index must be a non-negative time step; got {min(query)}.")
        clashes = sorted(set(query) & set(observed))
        if clashes:
            raise ValueError(
                "log_density scores an assignment to the UNOBSERVED fields, but time step(s) "
                f"{clashes} are already evidence in this posterior."
            )
        horizon = max(max(query), t_max) + 1
        log_b = _emission_log_b(observed, horizon)
        for t, val in query.items():
            for k in range(n_states):
                log_b[t, k] = _safe_log_density(model.topics[k], val)
        joint = MarkovChainLatentPosterior(model.log_w, model.log_transitions, log_b).log_likelihood()
        return float(joint - log_z)

    def mean_fn(path: FieldPath) -> float:
        w = _state_marginals_at(int(path[0]))
        means = np.array([_analytic_mean(model.topics[k]) for k in range(n_states)], dtype=np.float64)
        return float(np.sum(w * means))

    receipt = ConditionReceipt(method="exact", log_evidence=float(log_z))
    post = Posterior(sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=None)
    post.state_marginals = marginals  # convenience for callers wanting q(z_t | evidence) directly
    return post


# --------------------------------------------------------------------------------------------- #
# SIR fallback -- self-normalized likelihood-weighted ancestral sampling
# --------------------------------------------------------------------------------------------- #


def _generate_weighted(model: Any, ev: dict[FieldPath, Any], rng: RandomState) -> tuple[Any, float]:
    """One ``(record, log_weight)`` particle from ``model``'s own generative order, evidence clamped."""
    _check_field_indices(model, ev)
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
        return _impossible_posterior(
            method="sir",
            warning="all importance weights are zero; evidence has zero probability under the prior.",
            n_particles=n_particles,
        )
    m = log_weights[finite].max()
    w = np.where(finite, np.exp(log_weights - m), 0.0)
    sw = w.sum()
    w_norm = w / sw
    ess = float(1.0 / np.sum(w_norm**2))
    log_evidence = float(m + np.log(sw) - np.log(n_particles))
    ess_ratio = ess / n_particles
    if ess_ratio < 0.01:
        warnings.append(
            f"ESS ratio {ess_ratio:.4f} < 0.01 threshold -- evidence may be near-impossible under the prior."
        )
    receipt = ConditionReceipt(
        method="sir",
        log_evidence=log_evidence,
        ess=ess,
        ess_ratio=ess_ratio,
        n_particles=n_particles,
        warnings=warnings,
    )

    def sample_fn(n: int, s: int | None) -> Any:
        r = RandomState(s) if s is not None else RandomState(_rng_seed(rng))
        idx = r.choice(n_particles, size=n, replace=True, p=w_norm)
        return [records[j] for j in idx]

    def mean_fn(path: FieldPath) -> float:
        try:
            vals = np.array([float(_extract(records[j], path)) for j in range(n_particles)], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("Posterior.mean is defined only for numeric fields.") from exc
        return float(np.sum(w_norm * vals))

    return Posterior(sample_fn=sample_fn, log_density_fn=None, mean_fn=mean_fn, receipt=receipt, model=None)


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
    _check_field_indices(model, ev)
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
