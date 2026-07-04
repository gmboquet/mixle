"""Estimation planning + certificates -- the keystone that makes "right method, provably" auditable.

mixle already dispatches the right estimation method per block: exponential-family leaves get
closed-form MLE, conditional-linear-Gaussian factors get least squares, GLM factors get IRLS,
categorical tables get closed-form counts, and only genuinely non-convex pieces (mixtures, neural
leaves) fall back to EM or gradient descent. What was MISSING is making that visible and CHECKABLE:
a certificate saying, per block, which method ran and how strong its guarantee is, and an audit of
exactly where -- if anywhere -- gradient descent was unavoidable.

This module walks a fitted model (or a distribution prototype) and returns an
:class:`EstimationCertificate`: an ordered guarantee ladder per block plus the aggregate. The
aggregate is the MINIMUM over blocks -- a fit is only as strong as its weakest link -- so a fully
observed exponential-family graph certifies ``GLOBAL_UNIQUE`` while a mixture certifies
``STATIONARY`` even though every one of its M-steps is closed form (the certificate reports that
inner win explicitly: "why not ADAM" is answerable, block by block).

The guarantee ladder (ascending strength):

  HEURISTIC                 gradient descent (SGD/Adam) -- a local optimum, no global claim
  STATIONARY                EM / coordinate ascent -- a fixed point, possibly local
  STATIONARY_ESCAPE_TESTED  EM with saddle-escape restarts (Model.fit(restarts=...))
  GLOBAL                    convex objective (IRLS/least squares) -- the global optimum
  GLOBAL_UNIQUE             closed form with a provably unique global optimum (exp-family MLE, CLG)

Placement (``local`` / ``pool_eligible``) is advisory in v1: gradient blocks are marked
pool-eligible so a later pool executor (workstream H) can offload them; everything else stays local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "Guarantee",
    "BlockPlan",
    "EstimationCertificate",
    "certify",
    "plan_estimation",
]


class Guarantee(IntEnum):
    """How strong the solution to an estimation block is, as an ordered ladder (higher = stronger)."""

    HEURISTIC = 1
    STATIONARY = 2
    STATIONARY_ESCAPE_TESTED = 3
    GLOBAL = 4
    GLOBAL_UNIQUE = 5

    @property
    def label(self) -> str:
        return self.name


@dataclass
class BlockPlan:
    """The estimation plan for one block of a model: which method ran and how strong its guarantee is."""

    name: str  # dotted path within the model, e.g. "field[2]" or "component[0].mean"
    kind: str  # the block's distribution/factor type name
    method: str  # 'closed_form_mle' | 'conjugate' | 'least_squares' | 'convex_irls' | 'em' | 'gradient' | ...
    guarantee: Guarantee
    gradient: bool  # did this block require gradient descent (the ADAM question)
    placement: str  # 'local' | 'pool_eligible'
    reason: str  # human-readable justification

    def __str__(self) -> str:
        tag = " [GRADIENT]" if self.gradient else ""
        return f"{self.name}: {self.method} -> {self.guarantee.label}{tag}  ({self.reason})"


@dataclass
class EstimationCertificate:
    """The auditable proof of how a model was (or would be) estimated: per-block plans + the aggregate.

    ``guarantee`` is the minimum over blocks -- the fit's honest overall strength. ``why_not_adam``
    answers the standing question: gradient descent was used for exactly these blocks, for these
    reasons, and nowhere else.
    """

    guarantee: Guarantee
    blocks: list[BlockPlan] = field(default_factory=list)
    escape_tested: bool = False  # a latent fit that ran saddle-escape restarts (upgrades EM blocks)

    @property
    def gradient_blocks(self) -> list[BlockPlan]:
        return [b for b in self.blocks if b.gradient]

    @property
    def closed_form_blocks(self) -> list[BlockPlan]:
        return [b for b in self.blocks if b.guarantee >= Guarantee.GLOBAL]

    def as_dict(self) -> dict[str, Any]:
        return {
            "guarantee": self.guarantee.label,
            "escape_tested": self.escape_tested,
            "n_blocks": len(self.blocks),
            "n_gradient_blocks": len(self.gradient_blocks),
            "blocks": [
                {
                    "name": b.name,
                    "kind": b.kind,
                    "method": b.method,
                    "guarantee": b.guarantee.label,
                    "gradient": b.gradient,
                    "placement": b.placement,
                    "reason": b.reason,
                }
                for b in self.blocks
            ],
        }

    def why_not_adam(self) -> str:
        """The audit: which blocks needed gradient descent and why -- everything else got a stronger method."""
        grad = self.gradient_blocks
        total = len(self.blocks)
        if not grad:
            return (
                f"No gradient descent was used: all {total} block(s) solved by closed-form / convex / "
                f"EM methods (aggregate guarantee {self.guarantee.label})."
            )
        lines = [
            f"{len(grad)} of {total} block(s) required gradient descent; the other "
            f"{total - len(grad)} got a stronger method:"
        ]
        for b in grad:
            lines.append(f"  - {b.name} ({b.kind}): {b.reason}  [placement: {b.placement}]")
        return "\n".join(lines)

    def table(self) -> str:
        head = (
            f"EstimationCertificate: aggregate={self.guarantee.label}"
            f"{' (escape-tested)' if self.escape_tested else ''}, "
            f"{len(self.blocks)} block(s), {len(self.gradient_blocks)} gradient"
        )
        return "\n".join([head] + [f"  {b}" for b in self.blocks])

    def __str__(self) -> str:
        return self.table()


# --------------------------------------------------------------------------------------------------
# classification: map a fitted distribution (or factor) to its estimation method + guarantee
# --------------------------------------------------------------------------------------------------


def _is_neural(obj: Any) -> bool:
    """A leaf whose fit is gradient descent over a torch module."""
    if hasattr(obj, "module") or hasattr(obj, "_forward") or hasattr(obj, "net"):
        return True
    return type(obj).__name__.startswith(("Neural", "Transformer", "LanguageModel"))


def _is_exp_family(obj: Any) -> bool:
    try:
        from mixle.capability import ExponentialFamily, supports

        return bool(supports(obj, ExponentialFamily))
    except Exception:
        return False


def _has_exact_density(obj: Any) -> bool:
    try:
        from mixle.capability import ExactDensity, supports

        return bool(supports(obj, ExactDensity))
    except Exception:
        return False


def _classify_leaf(obj: Any, name: str) -> BlockPlan:
    """One non-composite block -> its estimation method and guarantee (honest, capability-driven)."""
    kind = type(obj).__name__
    if _is_neural(obj):
        return BlockPlan(
            name,
            kind,
            "gradient",
            Guarantee.HEURISTIC,
            gradient=True,
            placement="pool_eligible",
            reason="torch module fit by gradient descent -- no global optimum guarantee",
        )
    if _is_exp_family(obj):
        return BlockPlan(
            name,
            kind,
            "closed_form_mle",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="exponential family -- MLE is strictly concave in the natural parameters (unique global)",
        )
    # a GLM-shaped edge/factor: convex objective, global but not necessarily a closed form
    if hasattr(obj, "family") and hasattr(obj, "beta"):
        return BlockPlan(
            name,
            kind,
            "convex_irls",
            Guarantee.GLOBAL,
            gradient=False,
            placement="local",
            reason="generalized linear model -- convex log-likelihood solved by IRLS (global optimum)",
        )
    if _has_exact_density(obj):
        return BlockPlan(
            name,
            kind,
            "closed_form",
            Guarantee.GLOBAL,
            gradient=False,
            placement="local",
            reason="exact-density family with a deterministic closed-form estimate (global for its objective)",
        )
    return BlockPlan(
        name,
        kind,
        "iterative",
        Guarantee.STATIONARY,
        gradient=False,
        placement="local",
        reason="iterative estimator with no declared global guarantee (conservative classification)",
    )


def _classify_bn_factor(fac: Any, name: str) -> BlockPlan:
    """A HeterogeneousBayesianNetwork factor -> its method (marginal leaf / CLG / GLM / discrete table)."""
    kind = type(fac).__name__
    if kind == "_MarginalFactor":
        return _classify_leaf(fac.dist, name)
    if kind == "_LinearGaussianFactor":
        return BlockPlan(
            name,
            "CLG",
            "least_squares",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="conditional-linear-Gaussian factor -- least squares has a unique closed-form solution",
        )
    if kind == "_GLMFactor":
        return BlockPlan(
            name,
            "GLM",
            "convex_irls",
            Guarantee.GLOBAL,
            gradient=False,
            placement="local",
            reason="GLM factor (logistic/Poisson/softmax) -- convex objective, global optimum via IRLS/L-BFGS",
        )
    if kind == "_DiscreteConditionalFactor":
        return BlockPlan(
            name,
            "table",
            "closed_form_counts",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="discrete conditional table -- closed-form per-configuration counts (unique)",
        )
    if kind == "_VectorMarginalFactor":
        return BlockPlan(
            name,
            "MVN",
            "closed_form_mle",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="multivariate-Gaussian marginal -- closed-form mean + covariance (unique)",
        )
    if kind == "_VectorCLGFactor":
        return BlockPlan(
            name,
            "vector_CLG",
            "least_squares",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="multivariate conditional-linear-Gaussian -- multivariate least squares (unique closed form)",
        )
    return _classify_leaf(fac, name)


def _walk(obj: Any, name: str, blocks: list[BlockPlan], escape_tested: bool) -> None:
    """Recurse the fitted-model tree, appending a BlockPlan per leaf/factor block."""
    kind = type(obj).__name__

    # composite: independent per-field blocks -- the whole thing factorizes, each child is its own fit
    if isinstance(getattr(obj, "dists", None), (list, tuple)):
        for i, child in enumerate(obj.dists):
            _walk(child, f"{name}field[{i}]" if name else f"field[{i}]", blocks, escape_tested)
        return

    # heterogeneous Bayesian network: a DAG of parametric factors, each solved independently
    if hasattr(obj, "factors") and hasattr(obj, "edges"):
        for fac in obj.factors:
            blocks.append(_classify_bn_factor(fac, f"{name}node[{fac.child}]" if name else f"node[{fac.child}]"))
        return

    # mixture / latent: the latent structure makes the joint non-convex -> EM (STATIONARY). We STILL
    # recurse the components to REPORT that their M-steps are closed form (the "no ADAM inside EM" win),
    # but the EM block itself caps the guarantee.
    if isinstance(getattr(obj, "components", None), (list, tuple)):
        g = Guarantee.STATIONARY_ESCAPE_TESTED if escape_tested else Guarantee.STATIONARY
        reason = (
            "latent mixture fit by EM with saddle-escape restarts -- a fixed point, escape-tested"
            if escape_tested
            else "latent mixture fit by EM -- a fixed point that may be local (restarts='auto' escape-tests it)"
        )
        blocks.append(
            BlockPlan(
                f"{name}mixture" if name else "mixture",
                kind,
                "em",
                g,
                gradient=False,
                placement="local",
                reason=reason,
            )
        )
        for i, comp in enumerate(obj.components):
            _walk(comp, f"{name}component[{i}]." if name else f"component[{i}].", blocks, escape_tested)
        return

    # a single leaf
    blocks.append(_classify_leaf(obj, name.rstrip(".") if name else kind))


def certify(model: Any, *, escape_tested: bool = False) -> EstimationCertificate:
    """Return the :class:`EstimationCertificate` for a fitted model (or distribution prototype).

    Walks the model's block structure and classifies each block's estimation method + guarantee from
    its capability signals -- no fitting is done here, only inspection. Pass ``escape_tested=True``
    when the fit ran saddle-escape restarts (:meth:`mixle.Model.fit` sets this automatically), which
    upgrades EM blocks from ``STATIONARY`` to ``STATIONARY_ESCAPE_TESTED``.
    """
    blocks: list[BlockPlan] = []
    _walk(model, "", blocks, escape_tested)
    if not blocks:  # nothing structural detected -> treat the whole object as one leaf
        blocks.append(_classify_leaf(model, type(model).__name__))
    aggregate = min((b.guarantee for b in blocks), default=Guarantee.HEURISTIC)
    return EstimationCertificate(guarantee=aggregate, blocks=blocks, escape_tested=escape_tested)


def plan_estimation(model: Any, *, escape_tested: bool = False) -> EstimationCertificate:
    """Alias for :func:`certify` -- the pre-fit planning view over a distribution prototype."""
    return certify(model, escape_tested=escape_tested)
