"""Epistemic / aleatoric uncertainty decomposition for any predictive.

Current LLMs emit a point estimate, so their "confidence" cannot be split into *what is
irreducibly noisy* (aleatoric) versus *what more data would resolve* (epistemic). A model that
carries a posterior can. This module makes that split a first-class, model-agnostic operation --
generalizing :meth:`mixle.stats.graphs.knowledge_graph.KnowledgeGraphEnsemble.epistemic_tail_uncertainty`
(which did it only for knowledge-graph tails) to any predictive.

Two *exact* decompositions, matched to the two answer types:

* **discrete outcomes -> the entropy (BALD mutual-information) split**::

      total     = H( mean_m p_m )          predictive entropy (all uncertainty)
      aleatoric = mean_m H( p_m )          expected member entropy (genuine ambiguity)
      epistemic = total - aleatoric >= 0   mutual information (disagreement among members)

  The epistemic term is the Bayesian-active-learning-by-disagreement (BALD) score: it is zero
  when every posterior draw agrees, and large where they disagree -- i.e. where more data helps.

* **continuous outcomes -> the law-of-total-variance split**::

      aleatoric = mean_m Var_m             expected member variance (irreducible noise)
      epistemic = Var_m( mean_m )          variance of member means (model uncertainty)
      total     = aleatoric + epistemic    total predictive variance

"Members" are draws from ``q(theta | data)`` (parameter uncertainty, via
:class:`~mixle.inference.posterior.ParameterPosterior`) or an explicit ensemble of fitted models
(as in a deep ensemble / bagged fit). The DECOMPOSITION IDENTITIES are exact given the members;
what the numbers mean depends on where the members came from (audit U-1): only for exact
posterior draws (the conjugate route) is the finite member count M the sole approximation. A
VARIATIONAL ``q`` is mode-seeking and systematically understates the posterior spread (the
epistemic term inherits that understatement at any M); MCMC members carry burn-in and
autocorrelation, so the effective sample size, not M, governs their error. The finite-M
estimators are themselves biased: the entropy split's plug-in mutual information is capped at
``log M`` outright and biased low (audit U-2 -- at M = 2 a true 2.08-nat disagreement reads at
most 0.693), and the variance split's epistemic term uses the unbiased ``ddof=1`` estimator
precisely so its expectation is the population variance at every M (audit U-3 -- the ``ddof=0``
plug-in understated it by exactly ``(M-1)/M``, 50% at M = 2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.utils.special import softmax as _softmax

__all__ = [
    "UncertaintyDecomposition",
    "decompose_entropy",
    "decompose_variance",
    "predictive_distribution",
    "posterior_ensemble",
    "decompose_uncertainty",
    "Clustering",
    "cluster_samples",
    "marginalize_meaning",
    "semantic_entropy",
]


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


def _entropy_last(p: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) over normalized rows, with the ``0 log 0 = 0`` guard.

    Raises:
        ValueError: if any entry of ``p`` is non-finite (NaN/inf) or falls outside ``[0, 1]``
            (with a small floating-point tolerance).
    """
    p = np.asarray(p, dtype=float)
    if p.ndim < 1 or p.shape[-1] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite (no NaN/inf)")
    tol = 1e-9
    if np.any(p < -tol) or np.any(p > 1.0 + tol):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(np.sum(p, axis=-1), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("probability rows must sum to one")
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=-1)


def _owned_readonly(value: Any) -> Any:
    """Return an owned, write-locked copy of an array field; pass scalars through unchanged.

    A decomposition and a clustering are records of a computation that already happened, and each
    carries an invariant across its own fields: ``total == aleatoric + epistemic``, and a ``probs``
    vector that sums to one over exactly the classes ``labels`` indexes. While those fields were
    plain writable arrays, either could be rewritten after construction -- through the array the
    caller passed in, or through the field itself -- and the record then described numbers it no
    longer held, with ``fraction_epistemic`` silently reporting the edited version (MXR-080-1899).

    Scalars (Python and numpy) are returned as they are: they are already immutable, and wrapping
    them in 0-d arrays would change what ``.item()`` and the float-valued single-point path return.
    """
    if isinstance(value, np.ndarray):
        owned = np.array(value, copy=True)
        owned.setflags(write=False)
        return owned
    return value


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """A predictive uncertainty split into ``aleatoric`` + ``epistemic`` (summing to ``total``).

    ``kind`` is ``"entropy"`` (values in nats) or ``"variance"`` (values in the outcome's squared
    units). Each field is a scalar for a single query point, or an array over query points. Array
    fields are owned read-only copies, so the ``total = aleatoric + epistemic`` identity the record
    asserts cannot be broken after the fact (MXR-080-1899).
    """

    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        for name in ("total", "aleatoric", "epistemic"):
            object.__setattr__(self, name, _owned_readonly(getattr(self, name)))

    @property
    def fraction_epistemic(self) -> np.ndarray:
        """Share of the total uncertainty that is epistemic (reducible by more data), in ``[0, 1]``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(self.total > 0.0, self.epistemic / self.total, 0.0)
        return frac

    def item(self) -> UncertaintyDecomposition:
        """Collapse size-1 arrays to Python floats (convenience for single-point decompositions)."""
        if np.size(self.total) != 1:
            raise ValueError("item() only applies to a single-point decomposition")
        return UncertaintyDecomposition(
            float(np.reshape(self.total, -1)[0]),
            float(np.reshape(self.aleatoric, -1)[0]),
            float(np.reshape(self.epistemic, -1)[0]),
            self.kind,
        )


def decompose_entropy(member_probs: Any) -> UncertaintyDecomposition:
    """BALD entropy split of a discrete predictive.

    Args:
        member_probs: array ``(M, ..., K)`` -- ``M`` posterior draws / ensemble members, each a
            categorical predictive over ``K`` outcomes (optionally batched over query points in the
            middle axes). Rows need not be normalized; each is renormalized over the last axis.

    Returns:
        An :class:`UncertaintyDecomposition` with ``kind="entropy"`` (nats). ``epistemic`` is the
        PLUG-IN mutual information ``H(mean) - mean H``, clamped to ``>= 0`` (non-negative in
        exact arithmetic; the clamp removes floating-point negatives, and ``total`` is recomputed
        as ``aleatoric + epistemic`` afterward so the asserted identity holds exactly -- audit
        U-8). ESTIMATOR LIMITS (audit U-2): the plug-in MI over M members is structurally capped
        at ``log M`` whatever the true disagreement -- at the permitted minimum M = 2 a true
        2.08-nat mutual information reads at most 0.693 -- and is biased low below the cap; read
        it as a lower bound at small M, and prefer M large enough that ``log M`` comfortably
        exceeds the disagreement you need to resolve.
    """
    p = np.asarray(member_probs, dtype=float)
    if p.ndim < 2:
        raise ValueError("member_probs must have shape (M, ..., K) with at least a member and outcome axis")
    if p.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("member probabilities must be finite and non-negative")
    totals = p.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("every member probability row must have positive finite mass")
    p = p / totals
    mean = p.mean(axis=0)  # (..., K)
    total = _entropy_last(mean)  # H(mean)  -> (...)
    aleatoric = _entropy_last(p).mean(axis=0)  # mean_m H(p_m) -> (...)
    epistemic = np.maximum(total - aleatoric, 0.0)
    # recompute total AFTER the clamp so total == aleatoric + epistemic exactly (audit U-8: the
    # class docstring asserts that identity, and the clamp could break it by a float ulp)
    total = aleatoric + epistemic
    return UncertaintyDecomposition(total, aleatoric, epistemic, "entropy")


def decompose_variance(member_means: Any, member_vars: Any = None) -> UncertaintyDecomposition:
    """Law-of-total-variance split of a continuous predictive.

    Args:
        member_means: array ``(M, ...)`` -- each member's predictive mean ``E[y | theta_m]``.
        member_vars: array ``(M, ...)`` -- each member's predictive variance ``Var[y | theta_m]``.
            If ``None``, aleatoric noise is taken as zero (members are point predictors) and the
            decomposition reports only the epistemic spread of the means.

    Returns:
        An :class:`UncertaintyDecomposition` with ``kind="variance"`` (or
        ``kind="variance-epistemic-only"`` when ``member_vars`` is ``None`` -- ``total`` then
        contains NO aleatoric noise and is not a predictive variance; audit U-9).
    """
    mu = np.asarray(member_means, dtype=float)
    if mu.ndim < 1 or not np.all(np.isfinite(mu)):
        raise ValueError("member_means must be a finite array with a member axis")
    if mu.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    # ddof=1 (audit U-3): members are DRAWS representing the posterior, so the estimand is the
    # population variance of member means; the ddof=0 plug-in has expectation ((M-1)/M) * Var --
    # a measured 50% understatement of the epistemic term at the permitted minimum M = 2
    epistemic = mu.var(axis=0, ddof=1)  # Var_m E[y|m]
    if member_vars is None:
        aleatoric = np.zeros_like(epistemic)
    else:
        v = np.asarray(member_vars, dtype=float)
        if v.shape != mu.shape:
            raise ValueError(f"member_vars shape {v.shape} must match member_means shape {mu.shape}")
        if not np.all(np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError("member_vars must be finite non-negative variances")
        aleatoric = v.mean(axis=0)  # mean_m Var[y|m]
    total = aleatoric + epistemic
    kind = "variance" if member_vars is not None else "variance-epistemic-only"
    return UncertaintyDecomposition(total, aleatoric, epistemic, kind)


def predictive_distribution(members: Iterable[Any], support: Sequence[Any]) -> np.ndarray:
    """Evaluate an iterable of fitted distributions over a discrete ``support`` -> ``(M, K)`` probs.

    Each member's ``log_density`` is evaluated at every point of ``support`` and softmax-normalized
    over the support, giving one categorical row per member. Feed the result to
    :func:`decompose_entropy`.

    PRECONDITION (audit U-7): the softmax treats every support point as carrying EQUAL measure.
    That is exact for a discrete family evaluated on its full support, and a grid approximation
    for a continuous density ONLY when the grid is uniform -- a non-uniform grid silently
    reweights the density by the missing cell widths, and any downstream entropy inherits the
    distortion. Restrict a discrete member's support at your own risk: the row then represents
    the conditional given that subset.
    """
    support = list(support)
    if len(support) < 2:
        raise ValueError("support must contain at least two outcomes")
    rows = []
    for m in members:
        logs = np.array([float(m.log_density(s)) for s in support], dtype=float)
        if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)) or not np.any(np.isfinite(logs)):
            raise ValueError("each predictive member must assign finite mass to at least one support value")
        row = np.asarray(_softmax(logs), dtype=float)
        if not np.all(np.isfinite(row)) or np.any(row < 0) or not np.isclose(row.sum(), 1.0):
            raise ValueError("predictive member produced an invalid probability row")
        rows.append(row)
    out = np.asarray(rows, dtype=float)
    if out.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    return out


def posterior_ensemble(param_post: Any, build: Callable[[Any], Any], n: int = 200, rng: Any = None) -> list[Any]:
    """Materialize ``n`` models from a parameter posterior -- an ensemble representing ``q(theta|data)``.

    ``build`` maps one parameter draw (whatever :meth:`ParameterPosterior.sample` returns) to a
    fitted distribution, mirroring
    :meth:`~mixle.inference.posterior.PredictivePosterior.from_parameter_posterior`. The returned
    list is the "members" the decomposition integrates over -- so epistemic uncertainty here is
    genuine *parameter* uncertainty, not just ensemble disagreement.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not callable(build):
        raise TypeError("build must be callable")
    r = _as_rng(rng)
    return [build(param_post.sample(r)) for _ in range(int(n))]


@dataclass(frozen=True)
class Clustering:
    """Samples grouped into equivalence classes: ``representatives``, class ``probs``, per-sample ``labels``.

    The three fields are one interlocking statement -- ``probs[c]`` is the mass of the class whose
    member ``representatives[c]`` stands for, and ``labels[i]`` says which class sample ``i`` landed
    in -- so the record owns them: ``representatives`` is a tuple and the arrays are write-locked
    copies (MXR-080-1899). Consumers in this repository only read them (index, iterate, ``zip``,
    ``argmax``, ``tolist``), which is why the sequence can be a tuple without breaking a caller.
    """

    representatives: tuple[Any, ...]
    probs: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "representatives", tuple(self.representatives))
        object.__setattr__(self, "probs", _owned_readonly(np.asarray(self.probs)))
        object.__setattr__(self, "labels", _owned_readonly(np.asarray(self.labels)))


def cluster_samples(samples: Sequence[Any], equivalent: Callable[[Any, Any], bool] | None = None) -> Clustering:
    """Group ``samples`` into equivalence classes under ``equivalent`` (default exact ``==``).

    For discrete draws whose *surface form* varies but *meaning* does not -- e.g. LLM generations
    ("Paris", "It's Paris.", "The capital is Paris") -- pass a semantic ``equivalent`` (embedding
    similarity, an entailment check, normalized match). True single linkage is
    computed as the connected components of all pairwise equivalence edges.
    """
    values = list(samples)
    if not values:
        raise ValueError("cluster_samples needs at least one sample")
    eq = equivalent if equivalent is not None else (lambda a, b: a == b)
    if not callable(eq):
        raise TypeError("equivalent must be callable")
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(values)):
        for j in range(i):
            # audit U-6: a directional ``equivalent`` (an entailment check is the documented
            # example) made the partition depend on the input ORDER -- shuffling the same samples
            # changed K and every downstream entropy. Merging on either direction makes the
            # clustering order-independent; single-linkage chaining across a non-transitive
            # relation remains, as documented.
            if bool(eq(values[i], values[j])) or bool(eq(values[j], values[i])):
                union(i, j)
    class_by_root: dict[int, int] = {}
    reps: list[Any] = []
    labels: list[int] = []
    for index, value in enumerate(values):
        root = find(index)
        if root not in class_by_root:
            class_by_root[root] = len(reps)
            reps.append(value)
        labels.append(class_by_root[root])
    counts = np.bincount(labels, minlength=len(reps)).astype(float)
    return Clustering(reps, counts / counts.sum(), np.asarray(labels))


def marginalize_meaning(
    items: Sequence[Any],
    equivalent: Callable[[Any, Any], bool] | None = None,
    *,
    log_probs: Any = None,
    weights: Any = None,
) -> Clustering:
    """The distribution over *meanings* = the string distribution marginalized over each meaning class.

    A generative model puts probability on *strings*; a meaning is an equivalence class of strings
    (``equivalent`` decides sameness). The probability of a meaning ``c`` is the pushforward under the
    quotient -- you **sum the string probabilities over the class**: ``P(c) = sum_{s in c} P(s)``.
    This returns that marginal (as a :class:`Clustering`, ``probs`` = ``P(c)``).

    How the per-string probability enters:

    * ``log_probs`` -- the model's sequence log-probabilities ``log P(s)`` for each DISTINCT item;
      classes are combined by ``logsumexp`` and renormalized over the items you passed, so the
      result is the meaning marginal CONDITIONAL ON the observed strings, not the model's full
      marginal (audit U-5) -- unobserved tail mass is excluded, and repeated strings would each
      add their own ``P(s)`` (estimating ``sum P(s)^2``, a different quantity), so duplicates are
      refused in this branch. Use this when the items are *distinct* strings whose probabilities
      you know -- it corrects the counting form's hidden "every string in a class is equiprobable"
      assumption.
    * ``weights`` -- explicit non-negative masses per item (summed within class).
    * neither -- uniform mass, i.e. counting: ``P(c) = count_c / N``. For i.i.d. samples from the
      model this is the unbiased Monte-Carlo estimate of the same marginal.
    """
    c = cluster_samples(items, equivalent)
    if log_probs is not None and weights is not None:
        raise ValueError("pass either log_probs or weights, not both")
    if log_probs is None and weights is None:
        return c
    labels = c.labels
    k = len(c.representatives)
    if log_probs is not None:
        lp = np.asarray(log_probs, dtype=float).reshape(-1)
        if lp.shape[0] != labels.shape[0]:
            raise ValueError("log_probs must have one entry per item")
        if np.any(np.isnan(lp)) or np.any(np.isposinf(lp)) or not np.any(np.isfinite(lp)):
            raise ValueError("log_probs must contain at least one finite value and no NaN or +inf")
        # audit U-5: repeated strings would each contribute their own P(s), estimating
        # sum P(s)^2 per class instead of the marginal sum P(s) -- refuse rather than distort
        seen: dict = {}
        for item in items:
            key = repr(item)
            seen[key] = seen.get(key, 0) + 1
        duplicated = [key for key, count in seen.items() if count > 1]
        if duplicated:
            raise ValueError(
                "log_probs marginalization requires DISTINCT items (each string's probability "
                f"enters once); got repeats of {duplicated[:3]} -- deduplicate first, or drop "
                "log_probs to marginalize by sample counting"
            )
        logmass = np.full(k, -np.inf)
        for i, lab in enumerate(labels):
            logmass[lab] = np.logaddexp(logmass[lab], lp[i])
        probs = np.exp(logmass - logsumexp(logmass))
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != labels.shape[0]:
            raise ValueError("weights must have one entry per item")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        mass = np.zeros(k)
        for i, lab in enumerate(labels):
            mass[lab] += w[i]
        total = mass.sum()
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        probs = mass / total
    return Clustering(c.representatives, probs, labels)


def semantic_entropy(
    samples: Sequence[Any],
    equivalent: Callable[[Any, Any], bool] | None = None,
    *,
    log_probs: Any = None,
    weights: Any = None,
) -> float:
    """PLUG-IN entropy (nats) over the *meaning* classes of ``samples``.

    Sample a stochastic generator (an LLM at temperature) ``n`` times, marginalize the string
    distribution over meaning classes (:func:`marginalize_meaning`), and take the entropy of that
    marginal. High semantic entropy means the model disagrees with itself about *what* the answer is
    (a hallucination signal), as opposed to merely phrasing one answer many ways (which collapses to
    low entropy).

    APPROXIMATION, NOT A MEASUREMENT OF THE GENERATOR (STAT-RR17-14): with ``n`` counted samples
    this is the plug-in estimator, biased LOW by about ``(K - 1) / (2n)`` nats (Miller-Madow,
    ``K`` = observed meaning classes) -- at the eight-draw default an abstention loop actually
    uses, a uniform ten-class generator's true 2.303 nats measured 1.646 on average (bias
    -0.656): the bias direction makes a HALLUCINATING model look confident, the exact failure
    the signal exists to catch. Use :func:`semantic_entropy_receipt` for the estimate WITH its
    sample size, observed class count, bias estimate, and the bias-corrected value; abstention
    thresholds should use the corrected value (it errs toward abstaining). Pass ``log_probs``
    (the sequence log-likelihoods) to marginalize with the actual string probabilities rather
    than by sample counting. Feed the clusters' per-member distributions to
    :func:`decompose_entropy` for an epistemic/aleatoric split.
    """
    probs = marginalize_meaning(samples, equivalent, log_probs=log_probs, weights=weights).probs
    return float(_entropy_last(probs))


def semantic_entropy_receipt(
    samples: Sequence[Any],
    equivalent: Callable[[Any, Any], bool] | None = None,
    *,
    log_probs: Any = None,
    weights: Any = None,
) -> dict[str, Any]:
    """:func:`semantic_entropy` with its approximation receipt (STAT-RR17-14).

    Returns ``entropy_plugin`` (the biased-low plug-in), ``entropy_miller_madow`` (bias-corrected
    by ``(K - 1)/(2n)``; still an estimate), ``n_samples``, ``k_observed``,
    ``bias_estimate_nats`` (the correction added), and ``method``. The correction applies to the
    counting estimator; with explicit ``log_probs``/``weights`` the count-based bias formula does
    not apply and the receipt says so with ``bias_estimate_nats = None``.
    """
    # Materialize ONCE: an iterator/generator input was consumed by marginalize_meaning below,
    # after which len(list(samples)) counted the exhausted remainder -- the receipt then read
    # n_samples=0, bias=None, and an "MM correction available" method string with NO correction
    # applied, silently reporting the uncorrected plug-in as the corrected value (pass 19).
    samples = list(samples)
    clustering = marginalize_meaning(samples, equivalent, log_probs=log_probs, weights=weights)
    plugin = float(_entropy_last(clustering.probs))
    n = len(samples)
    k = int(np.asarray(clustering.probs).size)
    counted = log_probs is None and weights is None
    bias = (k - 1) / (2.0 * n) if (counted and n > 0) else None
    # Parametric-bootstrap standard error of the CORRECTED estimator (STAT-RR21-15): the
    # first-order delta method is sqrt((sum p (log p)^2 - H^2)/n), which DEGENERATES to exactly
    # 0 whenever the observed classes are equiprobable -- at the n=8 serving default a uniform
    # eight-class pattern reported se=0.0 while the exact repeated-sampling SD of the corrected
    # statistic is 0.2534, and a second pattern was 2.5x understated. Resampling counts from the
    # fitted multinomial and recomputing the Miller-Madow value captures the second-order spread
    # (including K-hat varying across resamples) that the delta method cannot see; like the
    # correction itself it applies to the counting estimator only.
    entropy_se = None
    entropy_se_receipt = None
    if counted and n > 0:
        probs = np.asarray(clustering.probs, dtype=float)
        boot_rng = np.random.RandomState(20260809)
        n_boot = 2048
        counts_boot = boot_rng.multinomial(n, probs / probs.sum(), size=n_boot)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = counts_boot / float(n)
            log_frac = np.where(counts_boot > 0, np.log(np.where(counts_boot > 0, frac, 1.0)), 0.0)
        plugin_boot = -np.sum(frac * log_frac, axis=1)
        k_boot = np.sum(counts_boot > 0, axis=1)
        mm_boot = plugin_boot + (k_boot - 1) / (2.0 * n)
        entropy_se = float(mm_boot.std(ddof=1))
        # STAT-RR22-15: the estimate's own method and Monte-Carlo noise, on the receipt -- a
        # bootstrap SE from B replicates carries ~1/sqrt(2(B-1)) relative MC error of its own
        # (~1.6% at B=2048), plus plug-in error from resampling the OBSERVED profile rather
        # than the truth (a [5,2,1] pattern measured 10.6% off the exact parametric SD at
        # B=512; larger B shrinks only the first term).
        entropy_se_receipt = {
            "method": "parametric bootstrap: multinomial(n, observed profile), Miller-Madow recomputed",
            "replicates": int(n_boot),
            "seed": 20260809,
            "se_relative_mc_error": float(1.0 / np.sqrt(2.0 * (n_boot - 1))),
        }
    return {
        "entropy_plugin": plugin,
        "entropy_miller_madow": plugin + bias if bias is not None else plugin,
        "n_samples": n,
        "k_observed": k,
        "bias_estimate_nats": bias,
        # multinomial parametric-bootstrap SE of entropy_miller_madow (None off the counting path)
        "entropy_se_estimate": entropy_se,
        "entropy_se_receipt": entropy_se_receipt,
        "method": "plug-in over sampled meaning classes"
        + (" (Miller-Madow correction available)" if counted else " (probability-weighted; MM correction n/a)"),
    }


def decompose_uncertainty(
    *,
    probs: Any = None,
    means: Any = None,
    variances: Any = None,
) -> UncertaintyDecomposition:
    """Front door: decompose a predictive into aleatoric + epistemic uncertainty.

    Pass exactly one representation of the per-member predictive:

    * ``probs=(M, ..., K)`` -- categorical predictives -> BALD entropy split
      (:func:`decompose_entropy`);
    * ``means=(M, ...)`` (with optional ``variances=(M, ...)``) -- continuous predictives ->
      law-of-total-variance split (:func:`decompose_variance`).

    To decompose a *fitted* model's parameter uncertainty, build the members first with
    :func:`posterior_ensemble` (then :func:`predictive_distribution` for the discrete case).
    """
    if probs is not None and means is not None:
        raise ValueError("pass either probs= (discrete) or means= (continuous), not both")
    if probs is not None:
        return decompose_entropy(probs)
    if means is not None:
        return decompose_variance(means, variances)
    raise ValueError("provide probs= (discrete) or means= (continuous)")
