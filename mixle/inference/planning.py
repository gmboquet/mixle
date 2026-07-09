"""Estimation planning and certificates.

Mixle chooses estimation methods from model structure: exponential-family
leaves use closed-form MLE when available, conditional-linear-Gaussian factors
use least squares, GLM factors use IRLS, categorical tables use count updates,
and non-convex pieces such as mixtures or neural leaves use EM or gradient
optimization.

This module makes those choices inspectable. It walks a fitted model or
distribution prototype and returns an :class:`EstimationCertificate` containing
per-block methods, guarantees, and placement hints. The aggregate guarantee is
the minimum over all blocks, so a fully observed exponential-family graph can
certify as ``GLOBAL_UNIQUE`` while a mixture certifies as ``STATIONARY`` even
when each M-step is closed form.

The guarantee ladder, in ascending strength, is ``HEURISTIC``,
``STATIONARY``, ``STATIONARY_ESCAPE_TESTED``, ``GLOBAL``, and
``GLOBAL_UNIQUE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "Guarantee",
    "BlockPlan",
    "EstimationCertificate",
    "EstimationSchedule",
    "SchedulePass",
    "certify",
    "plan_estimation",
    "schedule",
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
        """Return the enum name used in reports."""
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

    ``guarantee`` is the minimum over blocks. ``why_not_adam`` summarizes where
    gradient optimization was used and why those blocks could not use a stronger
    closed-form or convex route.
    """

    guarantee: Guarantee
    blocks: list[BlockPlan] = field(default_factory=list)
    escape_tested: bool = False  # a latent fit that ran saddle-escape restarts (upgrades EM blocks)

    @property
    def gradient_blocks(self) -> list[BlockPlan]:
        """Return blocks that require gradient optimization."""
        return [b for b in self.blocks if b.gradient]

    @property
    def closed_form_blocks(self) -> list[BlockPlan]:
        """Return blocks with global or stronger guarantees."""
        return [b for b in self.blocks if b.guarantee >= Guarantee.GLOBAL]

    def as_dict(self) -> dict[str, Any]:
        """Return the certificate as JSON-compatible data."""
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
        """Render a human-readable block-level estimation table."""
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


def _classify_process(obj: Any, name: str) -> BlockPlan | None:
    """Point-process / temporal families whose estimator is known -> a specific guarantee.

    These do not advertise ExponentialFamily, so without this they fall through to the conservative
    STATIONARY default -- which UNDER-states the ones with a genuine closed-form MLE and leaves the
    genuinely non-convex ones unlabeled. Each verdict is grounded in the family's actual estimator."""
    kind = type(obj).__name__
    if kind == "InhomogeneousPoissonProcessDistribution":
        # piecewise-constant intensity: rate[b] = count[b] / (width[b] * n_realizations). The Poisson
        # log-likelihood is strictly concave in each per-bin rate, so this closed form is the unique MLE.
        return BlockPlan(
            name,
            kind,
            "closed_form_counts",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="inhomogeneous Poisson -- closed-form per-bin rate MLE (Poisson-concave, unique global)",
        )
    if kind == "ContinuousTimeMarkovChainDistribution":
        # q_ij = n_ij / T_i: each off-diagonal rate is an independent closed-form Poisson-rate MLE
        # (strictly concave), so the generator estimate is the unique global optimum.
        return BlockPlan(
            name,
            kind,
            "closed_form_counts",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="CTMC generator -- closed-form q_ij = n_ij / T_i (independent Poisson rates, unique global)",
        )
    if kind == "BirthDeathSamplingDistribution":
        # each rate = (event count of that type) / integral_n: a closed-form Poisson-rate MLE per type,
        # strictly concave, hence the unique global for its objective.
        return BlockPlan(
            name,
            kind,
            "closed_form_counts",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="birth-death -- closed-form per-type rate MLE (count / exposure; Poisson-concave, unique)",
        )
    if kind in ("HawkesProcessDistribution", "MultivariateHawkesProcessDistribution", "PowerLawHawkesDistribution"):
        # Veen-Schoenberg / Lewis-Mohler branching EM (ML for the power-law kernel): the self-excitation
        # makes the likelihood non-convex, so it converges to a stationary point, not a certified global.
        return BlockPlan(
            name,
            kind,
            "em_branching",
            Guarantee.STATIONARY,
            gradient=False,
            placement="local",
            reason="self-exciting Hawkes -- branching EM/ML; non-convex likelihood (stationary point)",
        )
    if kind == "RenewalProcessDistribution":
        # the M-step feeds the inter-arrival gaps to the inter-arrival family's own estimator (the standard
        # renewal MLE); the censored boundary term is O(1/n_events) and not in the M-step. So the renewal
        # guarantee IS the inter-arrival family's guarantee -- delegate to it honestly.
        inner = getattr(obj, "interarrival", None)
        inner_plan = _classify_leaf(inner, name) if inner is not None else None
        guarantee = inner_plan.guarantee if inner_plan is not None else Guarantee.STATIONARY
        inner_kind = type(inner).__name__ if inner is not None else "unknown"
        return BlockPlan(
            name,
            kind,
            f"renewal_mle[{inner_kind}]",
            guarantee,
            gradient=inner_plan.gradient if inner_plan is not None else False,
            placement="local",
            reason=(
                f"renewal process -- M-step is the inter-arrival ({inner_kind}) MLE; boundary term O(1/n); "
                f"inherits its guarantee"
            ),
        )
    return None


def _classify_leaf(obj: Any, name: str) -> BlockPlan:
    """One non-composite block -> its estimation method and capability-driven guarantee."""
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
    process = _classify_process(obj, name)
    if process is not None:
        return process
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


def certify(model: Any, *, escape_tested: bool = False, penalized: str | bool = False) -> EstimationCertificate:
    """Return the :class:`EstimationCertificate` for a fitted model (or distribution prototype).

    Walks the model's block structure and classifies each block's estimation method + guarantee from
    its capability signals -- no fitting is done here, only inspection. Pass ``escape_tested=True``
    when the fit ran saddle-escape restarts (:meth:`mixle.Model.fit` sets this automatically), which
    upgrades EM blocks from ``STATIONARY`` to ``STATIONARY_ESCAPE_TESTED``.

    Pass ``penalized`` (a reason string, or True) when the fit optimized a penalized objective -- soft
    constraints, conservation/PINN residual factors, potentials (E2). The optimum is then of the
    penalized surrogate, not the likelihood, so no block may claim more than STATIONARY however clean
    its own solver is: every stronger block is downgraded with the penalty named in its reason.
    """
    blocks: list[BlockPlan] = []
    _walk(model, "", blocks, escape_tested)
    if not blocks:  # nothing structural detected -> treat the whole object as one leaf
        blocks.append(_classify_leaf(model, type(model).__name__))
    if penalized:
        why = penalized if isinstance(penalized, str) else "soft-constraint / residual penalty"
        for b in blocks:
            if b.guarantee > Guarantee.STATIONARY:
                b.guarantee = Guarantee.STATIONARY
                b.reason += (
                    f" [DOWNGRADED: penalized objective ({why}) -- optimum of the surrogate, not the likelihood]"
                )
    aggregate = min((b.guarantee for b in blocks), default=Guarantee.HEURISTIC)
    return EstimationCertificate(guarantee=aggregate, blocks=blocks, escape_tested=escape_tested)


def plan_estimation(model: Any, *, escape_tested: bool = False) -> EstimationCertificate:
    """Alias for :func:`certify` -- the pre-fit planning view over a distribution prototype."""
    return certify(model, escape_tested=escape_tested)


@dataclass
class SchedulePass:
    """One pass of the estimation schedule: what runs, on which block, how, where, and how often."""

    order: int
    kind: str  # 'estep' | 'mstep' | 'independent' | 'gradient'
    block: str
    method: str
    placement: str  # 'local' | 'pool_eligible'
    repeat: str  # 'per_round' (inside the EM loop) | 'once'


@dataclass
class EstimationSchedule:
    """The block-coordinate schedule planner v2 produces (A3): ordered passes + the loop structure.

    A fully-factorized model schedules one independent pass per block (no loop). A latent model
    schedules the EM loop explicitly: an E-step over the latent, then one M-step pass PER BLOCK per
    round -- each M-step named with its own method, so the schedule shows exactly where the closed
    forms live inside the iteration (and which pass, if any, is the gradient block a pool would take).
    """

    passes: list[SchedulePass] = field(default_factory=list)
    latent: bool = False  # whether the schedule is an EM loop (vs one-shot independent passes)

    @property
    def per_round(self) -> list[SchedulePass]:
        """Return schedule passes repeated inside each latent-variable round."""
        return [p for p in self.passes if p.repeat == "per_round"]

    @property
    def gradient_passes(self) -> list[SchedulePass]:
        """Return schedule passes assigned to gradient optimization."""
        return [p for p in self.passes if p.kind == "gradient"]

    def describe(self) -> str:
        """Return a compact prose description of the estimation schedule."""
        if not self.latent:
            steps = "; ".join(f"{p.block}: {p.method}" for p in self.passes)
            return f"one-shot ({len(self.passes)} independent block(s)): {steps}"
        msteps = [p for p in self.per_round if p.kind in ("mstep", "gradient")]
        inner = "; ".join(f"{p.block}: {p.method}" + (" [pool]" if p.placement != "local" else "") for p in msteps)
        return f"EM loop until converged -- each round: E-step, then {len(msteps)} M-step(s): {inner}"


def schedule(model: Any, *, escape_tested: bool = False) -> EstimationSchedule:
    """Plan the block-coordinate estimation schedule for ``model`` (planner v2, A3).

    Built from the same block classification as :func:`certify`: EM blocks make the schedule a loop
    (E-step + per-block M-steps, repeated until convergence); without a latent block every block is one
    independent pass. Gradient blocks appear as explicit ``gradient`` passes with their pool placement,
    so the schedule is also the offload plan for the hybrid case."""
    cert = certify(model, escape_tested=escape_tested)
    em_blocks = [b for b in cert.blocks if b.method == "em"]
    param_blocks = [b for b in cert.blocks if b.method != "em"]

    passes: list[SchedulePass] = []
    if em_blocks:
        order = 0
        for em in em_blocks:
            passes.append(
                SchedulePass(order, "estep", em.name, "posterior_responsibilities", em.placement, "per_round")
            )
            order += 1
        for b in param_blocks:
            kind = "gradient" if b.gradient else "mstep"
            passes.append(SchedulePass(order, kind, b.name, b.method, b.placement, "per_round"))
            order += 1
        return EstimationSchedule(passes=passes, latent=True)

    for i, b in enumerate(param_blocks):
        kind = "gradient" if b.gradient else "independent"
        passes.append(SchedulePass(i, kind, b.name, b.method, b.placement, "once"))
    return EstimationSchedule(passes=passes, latent=False)
