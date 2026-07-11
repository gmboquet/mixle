"""Node report protocol -- per-subtree diagnostics over a composed distribution tree (workstream D1).

Frame (see the ConditionalJIT track, D1-D6): **the estimator tree is an IR**. Every node -- a leaf
distribution or a combinator subtree (Composite/Mixture/Sequence/Conditional/Optional/...) -- can
report its own residual, its Q-gain (Neal-Hinton free-energy lower-bound improvement), its E/M cost,
its update kind, and cheap health receipts. Later track items (freeze/roll-up caching, a block-EM
scheduler, leaf hot-swapping, backend re-specialization, a learned controller) all read these reports;
none of them may change what a fit computes -- only how it is scheduled -- so this module never runs
its own EM, it only instruments/observes the existing machinery in :mod:`mixle.inference.em` and
:mod:`mixle.inference.estimation`.

Design choices (documented here because later D-track items depend on this interface):

* **residual** -- a Monte-Carlo estimate of this node's own negative log-density,
  ``-mean(log_density(x))`` over samples drawn from the node's *own current fit*
  (``node.sampler().sample(n)``). This is available at EVERY node generically (every
  ``ProbabilityDistribution`` has ``sampler`` and ``log_density``) without needing per-combinator data
  slicing (a leaf under a ``CompositeDistribution`` never sees the top-level tuple directly, so an
  exact real-data residual cannot be computed generically for an arbitrary subtree without
  reimplementing every combinator's data-projection rule -- out of scope for an S-effort generic
  dispatcher). For an exponential-family leaf this MC residual converges to the differential/Shannon
  entropy of the fit; it is a legitimate "how much spread/uncertainty is still in this subtree's fit"
  signal, not a real-data fit residual. Callers who want a real-data top-level residual can pass
  ``data``/``enc_data`` and read the *root* row, whose residual/Q-gain instead comes from the actual
  EM objective (see :func:`root_em_report`).
* **Q-gain** -- ``residual_before - residual_after`` for the SAME field path across two
  :func:`flat_report_table` calls that bracket one EM update (pass the earlier table as
  ``prev_table``). This is a genuine before/after delta of the tracked residual, not invented. The
  Neal-Hinton guarantee (coordinate ascent on one computable free energy F) is proven only for the
  actual tracked EM objective at the TOP level (see :mod:`mixle.inference.em`'s ``run_em`` /
  :mod:`mixle.inference.estimation`'s ``optimize``, whose ``delta``-gated loop never accepts a
  decreasing step); per-node Q-gain is a diagnostic decomposition, not individually guaranteed
  non-negative (a mixture EM step can, e.g., grow one component's spread while improving the joint
  objective). :func:`root_em_report` verifies the real, provably-monotone quantity.
* **E/M cost** -- a parameter-count x dataset-size proxy (``nobs`` if supplied, else 1): E-step cost
  ``~ param_count * nobs`` (every observation is scored against every parameter once), M-step cost
  ``~ param_count`` for a closed-form/conjugate update, ``~ param_count * _GRADIENT_STEPS`` for a
  gradient-based estimator, ``0`` for a frozen/no-op node. Proxy, not wall-clock -- documented so D2's
  freeze/roll-up cache and D3's scheduler can decide whether to replace it with a measured cost later.
* **update kind** -- derived from the node's own estimator/capabilities: ``"frozen"`` (the ``Neutral``
  capability -- a Null distribution/accumulator/encoder), ``"em"`` (latent-variable nodes exposing
  ``seq_posterior`` or the ``LatentStructured`` capability -- the same duck-type
  :mod:`mixle.inference.em` itself uses for ``PosteriorTransformEM``), ``"conjugate_closed_form"``
  (the ``ConjugateUpdatable`` capability), ``"gradient"`` (estimator class name signals a
  gradient/neural/torch optimizer), else ``"closed_form"`` (the common one-shot M-step case).
* **health receipts** -- reuses the near-degenerate-variance check already used by
  :mod:`mixle.inference.precision_plan` (``sigma2``/``variance`` below a floor), adds a generic NaN
  sweep over the node's own numeric attributes (``-inf`` is deliberately NOT flagged -- it is a
  legitimate log-space encoding of a zero-probability event, e.g. a categorical's
  ``log_default_value``; only NaN is never a legitimate parameter value), and an ill-conditioning
  check (``cond`` on any square 2D numeric attribute whose name suggests a covariance/precision
  matrix) -- a minimal, honest version where nothing richer already exists in the codebase for a
  given family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import ProbabilityDistribution

_DEFAULT_MC_SAMPLES = 64
_GRADIENT_STEPS = 50.0  # proxy M-step iteration count for gradient-based estimators
_EM_ITERS = 10.0  # proxy outer-iteration count for latent (E/M) nodes
_NEAR_DEGENERATE_VARIANCE = 1e-10
_ILL_CONDITIONED_COND_NUMBER = 1e8


@dataclass
class NodeReport:
    """Per-node diagnostics for one point in a composed distribution tree.

    ``field_path`` follows the existing ``EnumerationError`` path convention used throughout the
    combinators (e.g. ``"MixtureDistribution.components[0] -> CompositeDistribution.dists[1]"``), so
    reports compose with the rest of the codebase's structural-error/debugging vocabulary.
    """

    field_path: str
    node_type: str
    update_kind: str
    residual: float
    q_gain: float | None
    e_step_cost: float
    m_step_cost: float
    param_count: int
    health: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """True iff every boolean health receipt that flags a problem is False."""
        return not any(bool(v) for k, v in self.health.items() if k != "finite_params") and self.health.get(
            "finite_params", True
        )


def _param_count(dist: ProbabilityDistribution) -> int:
    """Return a parameter-count proxy for ``dist``: declared params if known, else numeric-leaf count."""
    from mixle.stats.compute.declarations import declaration_for

    declaration = declaration_for(dist)
    if declaration is not None and declaration.parameters:
        total = 0
        for spec in declaration.parameters:
            value = getattr(dist, spec.name, None)
            if isinstance(value, np.ndarray):
                total += int(value.size)
            elif value is not None:
                total += 1
        return max(total, 1)

    total = 0
    for name, value in vars(dist).items():
        if name.startswith("_"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            total += int(value.size)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total += 1
    return max(total, 1)


def _update_kind(dist: ProbabilityDistribution) -> str:
    """Classify how ``dist`` gets re-estimated: frozen / em / conjugate_closed_form / gradient / closed_form."""
    from mixle.capability import ConjugateUpdatable, LatentStructured, Neutral, supports

    if supports(dist, Neutral):
        return "frozen"
    if callable(getattr(dist, "seq_posterior", None)) or supports(dist, LatentStructured):
        return "em"
    if supports(dist, ConjugateUpdatable):
        return "conjugate_closed_form"

    try:
        estimator = dist.estimator()
    except Exception:  # noqa: BLE001
        estimator = None
    if estimator is not None:
        est_name = type(estimator).__name__.lower()
        if any(tok in est_name for tok in ("grad", "torch", "neural", "sgd", "adam")):
            return "gradient"
    return "closed_form"


def _em_cost(update_kind: str, param_count: int, nobs: float) -> tuple[float, float]:
    """Return ``(e_step_cost, m_step_cost)`` proxy units for ``update_kind`` (see module docstring)."""
    if update_kind == "frozen":
        return 0.0, 0.0
    e_cost = float(param_count) * float(nobs)
    if update_kind == "gradient":
        m_cost = float(param_count) * _GRADIENT_STEPS
    elif update_kind == "em":
        m_cost = float(param_count) * _EM_ITERS
    else:
        m_cost = float(param_count)
    return e_cost, m_cost


def _residual_mc(dist: ProbabilityDistribution, n_mc: int, seed: int) -> float:
    """Monte-Carlo self-NLL: ``-mean(log_density(x))`` over ``n_mc`` self-samples (see module docstring)."""
    try:
        samples = dist.sampler(seed).sample(int(n_mc))
    except Exception:  # noqa: BLE001
        return float("nan")
    if samples is None:
        # A Neutral/no-op node (e.g. NullDistribution) legitimately samples nothing.
        return float("nan")

    total = 0.0
    count = 0
    for x in samples:
        try:
            lp = float(dist.log_density(x))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(lp):
            total += -lp
            count += 1
    if count == 0:
        return float("nan")
    return total / count


def _health_receipts(dist: ProbabilityDistribution) -> dict[str, Any]:
    """Cheap, generic health diagnostics: NaN/Inf params, near-degenerate variance, ill-conditioning.

    Reuses the near-degenerate-variance convention from
    :func:`mixle.inference.precision_plan.recommend_compute_precision` (a ``sigma2``/``variance``
    attribute below a small floor); the NaN/Inf sweep and covariance-conditioning check are the
    minimal honest additions for families with no richer diagnostic already registered.
    """
    receipts: dict[str, Any] = {"finite_params": True, "near_degenerate": False, "ill_conditioned": False}
    for name, value in vars(dist).items():
        if name.startswith("_"):
            continue
        arr: np.ndarray | None = None
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arr = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            arr = np.asarray([value], dtype=np.float64)

        if arr is not None:
            # NaN is never a legitimate parameter value. -inf is excluded from this check on purpose:
            # log-space parameters (e.g. a zero-probability category's log_default_value) legitimately
            # carry -inf, so flagging it here would be a false positive on perfectly healthy models.
            if arr.size and np.any(np.isnan(arr)):
                receipts["finite_params"] = False
            if "sigma2" in name or "variance" in name or "scale2" in name:
                if np.any(arr < _NEAR_DEGENERATE_VARIANCE):
                    receipts["near_degenerate"] = True

        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.shape[0] == value.shape[1]
            and value.shape[0] > 0
            and any(tok in name.lower() for tok in ("cov", "sigma", "precision"))
        ):
            try:
                if np.all(np.isfinite(value)) and np.linalg.cond(value) > _ILL_CONDITIONED_COND_NUMBER:
                    receipts["ill_conditioned"] = True
            except Exception:  # noqa: BLE001
                pass
    return receipts


def node_report(
    dist: ProbabilityDistribution,
    *,
    field_path: str = "root",
    n_mc: int = _DEFAULT_MC_SAMPLES,
    seed: int = 0,
    nobs: float | None = None,
    prev_residual: float | None = None,
) -> NodeReport:
    """Return a :class:`NodeReport` for a single node (leaf or combinator subtree).

    Dispatches generically over the five-piece contract / capability lens -- no per-family branching.
    """
    node_type = type(dist).__name__
    update_kind = _update_kind(dist)
    # A frozen/no-op node (the Neutral capability, e.g. NullDistribution) contributes nothing to be
    # fit and typically cannot even sample -- report an exact 0.0 residual rather than NaN.
    residual = 0.0 if update_kind == "frozen" else _residual_mc(dist, n_mc=n_mc, seed=seed)
    param_count = _param_count(dist)
    e_step_cost, m_step_cost = _em_cost(update_kind, param_count, 1.0 if nobs is None else float(nobs))
    health = _health_receipts(dist)

    q_gain: float | None = None
    if prev_residual is not None and np.isfinite(prev_residual) and np.isfinite(residual):
        q_gain = float(prev_residual - residual)

    return NodeReport(
        field_path=field_path,
        node_type=node_type,
        update_kind=update_kind,
        residual=residual,
        q_gain=q_gain,
        e_step_cost=e_step_cost,
        m_step_cost=m_step_cost,
        param_count=param_count,
        health=health,
    )


def _child_distributions(dist: ProbabilityDistribution) -> list[tuple[str, ProbabilityDistribution]]:
    """Generic duck-typed child discovery: any attribute holding a distribution (or a list/tuple/dict
    of distributions) is a structural child. Mirrors the ``ClassName.attr[idx]`` path convention
    already used by ``EnumerationError``/``child_enumerator`` in :mod:`mixle.stats.compute.pdist`, and
    the list/tuple/dict child-walk already used by ``_collect_estimator_keys`` in the same module --
    no per-combinator-family special-casing.
    """
    node_type = type(dist).__name__
    out: list[tuple[str, ProbabilityDistribution]] = []
    for name, value in sorted(vars(dist).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, ProbabilityDistribution):
            out.append(("%s.%s" % (node_type, name), value))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                if isinstance(child, ProbabilityDistribution):
                    out.append(("%s.%s[%d]" % (node_type, name, i), child))
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, ProbabilityDistribution):
                    out.append(("%s.%s[%r]" % (node_type, name, key), child))
    return out


def walk_tree(dist: ProbabilityDistribution, *, path: str = "root") -> list[tuple[str, ProbabilityDistribution]]:
    """Return ``(field_path, node)`` for every node in the tree, pre-order, deduplicated by identity."""
    out: list[tuple[str, ProbabilityDistribution]] = []
    visited: set[int] = set()

    def _visit(node: ProbabilityDistribution, node_path: str) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        out.append((node_path, node))
        for child_label, child in _child_distributions(node):
            child_path = "%s -> %s" % (node_path, child_label)
            _visit(child, child_path)

    _visit(dist, path)
    return out


def flat_report_table(
    dist: ProbabilityDistribution,
    *,
    nobs: float | None = None,
    prev_table: list[NodeReport] | None = None,
    n_mc: int = _DEFAULT_MC_SAMPLES,
    seed: int = 0,
) -> list[NodeReport]:
    """Walk a composed tree and return one :class:`NodeReport` per node, in traversal order.

    Satisfies "composed tree -> flat table": every leaf and every combinator subtree gets exactly one
    row, keyed by its ``field_path``. Pass the previous call's return value as ``prev_table`` to
    populate each row's ``q_gain`` as the residual improvement across an intervening EM update.
    """
    prev_by_path = {r.field_path: r.residual for r in prev_table} if prev_table else {}
    rows: list[NodeReport] = []
    for i, (node_path, node) in enumerate(walk_tree(dist)):
        rows.append(
            node_report(
                node,
                field_path=node_path,
                n_mc=n_mc,
                seed=seed + i,
                nobs=nobs,
                prev_residual=prev_by_path.get(node_path),
            )
        )
    return rows


def root_em_report(
    enc_data: Any,
    estimator: Any,
    model: ProbabilityDistribution,
    *,
    engine: Any | None = None,
    max_its: int = 1,
) -> tuple[ProbabilityDistribution, float, float]:
    """Run ``max_its`` real EM steps (via :func:`mixle.inference.em.run_em`) and return
    ``(new_model, objective_before, objective_after)`` on the ACTUAL tracked Neal-Hinton objective
    (observed log-likelihood / MAP / VB, whichever ``run_em`` resolves) -- the one quantity this
    module does not approximate. ``objective_after >= objective_before`` is the real, provably
    monotone Q-gain guarantee; :class:`NodeReport`'s per-node ``q_gain`` is a diagnostic decomposition,
    not individually guaranteed non-negative (see module docstring).
    """
    from mixle.inference.em import _resolve_run_em_objective, run_em

    objective = _resolve_run_em_objective(None, enc_data, estimator, model, engine)
    before = float(objective(model))
    new_model = run_em(enc_data, estimator, model, max_its=max_its, delta=None, engine=engine)
    after = float(objective(new_model))
    return new_model, before, after
