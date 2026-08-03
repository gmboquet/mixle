"""Estimation planning and evidence-backed certificates.

Planning and certification are deliberately different. Model structure can
suggest an estimator and an *upper bound* on what it could prove, but a class
name or capability marker cannot prove that a particular fit is identified,
converged, or globally optimal. A certificate therefore reports both the
candidate guarantee and the guarantee established by checked conditions or
explicit verification receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from mixle.utils.exact import require_exact_bool

__all__ = [
    "Guarantee",
    "ProofObligation",
    "VerificationReceipt",
    "BlockPlan",
    "EstimationCertificate",
    "EstimationSchedule",
    "SchedulePass",
    "certify",
    "receipt_subject",
    "verify_estimation_conditions",
    "plan_estimation",
    "schedule",
]


class Guarantee(IntEnum):
    """How strong the solution to an estimation block is, as an ordered ladder (higher = stronger)."""

    UNVERIFIED = 0
    HEURISTIC = 1
    STATIONARY = 2
    STATIONARY_ESCAPE_TESTED = 3
    GLOBAL = 4
    GLOBAL_UNIQUE = 5

    @property
    def label(self) -> str:
        """Return the enum name used in reports."""
        return self.name


@dataclass(frozen=True)
class ProofObligation:
    """A condition that must be evidenced before a block may claim a guarantee."""

    check: str
    required_for: Guarantee
    description: str


def _frozen_evidence(value: Any, path: str) -> Any:
    """Return an immutable, JSON-expressible copy of one receipt evidence value, recursively."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"verification receipt {path} must be finite to serialize, got {value!r}")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"verification receipt {path} keys must be strings, got {key!r}")
            frozen[key] = _frozen_evidence(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_evidence(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Real):  # numpy scalar
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"verification receipt {path} must be finite to serialize, got {value!r}")
        return int(value) if isinstance(value, Integral) else numeric
    raise TypeError(
        f"verification receipt {path} holds {type(value).__name__}, which is neither immutable nor "
        "JSON-expressible; a receipt an audit cannot read or serialize is not evidence"
    )


def _plain_evidence(value: Any) -> Any:
    """Undo :func:`_frozen_evidence`'s containers for JSON-compatible output."""
    if isinstance(value, Mapping):
        return {key: _plain_evidence(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_evidence(item) for item in value]
    return value


@dataclass(frozen=True)
class VerificationReceipt:
    """Evidence that named checks were performed for one estimation block.

    Receipts are explicit audit inputs, not Boolean assertions. ``evidence``
    must contain the measurements or result metadata supporting the checks.
    """

    receipt_id: str
    block: str
    guarantee: Guarantee
    checks: tuple[str, ...]
    source: str
    evidence: dict[str, Any]
    # Which object this receipt verified, as mixle.data.hashing.model_hash of the model. Empty means
    # the receipt names no subject, and :func:`certify` will not raise a guarantee on it: naming the
    # right check strings was previously the whole test, so a receipt asserting
    # ("finite_supported_data", "solver_matches_objective", "identified_parameters") upgraded an
    # arbitrary GaussianDistribution(999, 1) from UNVERIFIED to GLOBAL_UNIQUE with evidence reading
    # {"trust": "me"} (MXR-080-1877). A hash cannot stop a determined forger, but it stops a receipt
    # being minted without the model, reused across models, or outliving the parameters it describes.
    subject_hash: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.block or not self.source:
            raise ValueError("verification receipts require non-empty receipt_id, block, and source")
        if not isinstance(self.guarantee, Guarantee):
            raise TypeError(
                f"verification receipt guarantee must be a Guarantee, got {type(self.guarantee).__name__} "
                f"({self.guarantee!r}); an int compares against the ladder but has no .label"
            )
        if self.guarantee <= Guarantee.UNVERIFIED:
            raise ValueError("verification receipt guarantee must be stronger than UNVERIFIED")
        if isinstance(self.checks, (str, bytes)) or not isinstance(self.checks, (tuple, list)):
            raise TypeError(
                f"verification receipt checks must be a sequence of names, got {type(self.checks).__name__}: "
                "a string would iterate as its characters"
            )
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.checks or any(not isinstance(check, str) or not check.strip() for check in self.checks):
            raise ValueError("verification receipts require one or more named checks")
        if not isinstance(self.evidence, dict) or not self.evidence:
            raise ValueError("verification receipts require non-empty structured evidence")
        # ``evidence`` is the audit input the whole class exists to carry, and it was checked only for
        # being a non-empty dict (MXR-080-1874). It could hold an open socket or a live model object,
        # which no audit can read and ``as_dict()`` cannot serialize, and it was stored by reference
        # on a frozen dataclass so the caller could rewrite the measurements after the guarantee they
        # justify had been recorded.
        object.__setattr__(self, "evidence", _frozen_evidence(self.evidence, f"{self.receipt_id}.evidence"))

    def evidence_as_dict(self) -> dict[str, Any]:
        """Return the evidence as plain JSON-compatible data."""
        return _plain_evidence(self.evidence)


@dataclass
class BlockPlan:
    """One estimation block and the distinction between planned and verified strength."""

    name: str  # dotted path within the model, e.g. "field[2]" or "component[0].mean"
    kind: str  # the block's distribution/factor type name
    method: str  # 'closed_form_mle' | 'conjugate' | 'least_squares' | 'convex_irls' | 'em' | 'gradient' | ...
    guarantee: Guarantee
    gradient: bool  # did this block require gradient descent (the ADAM question)
    placement: str  # 'local' | 'pool_eligible'
    reason: str  # human-readable justification
    candidate_guarantee: Guarantee | None = None
    proof_obligations: tuple[ProofObligation, ...] = ()
    verified_checks: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    # Who established this block's guarantee. A certificate previously read the same whether the
    # library had checked the conditions itself or a caller had asserted them, which is exactly the
    # distinction a reader of an audit artifact needs (MXR-080-1877).
    verified_by: str = ""
    # Receipts that named the right checks but not this model, so they were refused. Recorded rather
    # than dropped, so "no evidence offered" is distinguishable from "evidence offered and refused".
    unbound_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_guarantee is None:
            self.candidate_guarantee = self.guarantee

    def __str__(self) -> str:
        tag = " [GRADIENT]" if self.gradient else ""
        candidate = (
            f", candidate={self.candidate_guarantee.label}" if self.candidate_guarantee != self.guarantee else ""
        )
        return f"{self.name}: {self.method} -> {self.guarantee.label}{candidate}{tag}  ({self.reason})"


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
                    "candidate_guarantee": b.candidate_guarantee.label,
                    "gradient": b.gradient,
                    "placement": b.placement,
                    "reason": b.reason,
                    "proof_obligations": [
                        {
                            "check": obligation.check,
                            "required_for": obligation.required_for.label,
                            "description": obligation.description,
                        }
                        for obligation in b.proof_obligations
                    ],
                    "verified_checks": list(b.verified_checks),
                    "receipt_ids": list(b.receipt_ids),
                    "verified_by": b.verified_by,
                    "unbound_receipt_ids": list(b.unbound_receipt_ids),
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
                f"No gradient descent is planned for {total} block(s); the planned methods are closed-form, "
                f"convex, or iterative (verified aggregate {self.guarantee.label})."
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
    except Exception:  # noqa: BLE001
        return False


def _has_exact_density(obj: Any) -> bool:
    try:
        from mixle.capability import ExactDensity, supports

        return bool(supports(obj, ExactDensity))
    except Exception:  # noqa: BLE001
        return False


def _classify_process(obj: Any, name: str) -> BlockPlan | None:
    """Plan a process estimator and its best possible, not yet verified, guarantee."""
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
            reason="inhomogeneous Poisson count/exposure update; uniqueness requires positive checked exposure",
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
            reason="CTMC count/dwell update; uniqueness requires checked positive exposure for every fitted row",
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
            reason="birth-death count/exposure update; uniqueness requires checked positive population exposure",
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
            reason=(
                "self-exciting Hawkes branching EM has a non-convex objective; stationarity requires "
                "an optimizer-convergence receipt"
            ),
        )
    if kind == "RenewalProcessDistribution":
        # the M-step feeds the inter-arrival gaps to the inter-arrival family's own estimator (the standard
        # renewal MLE); the censored boundary term is O(1/n_events) and not in the M-step. So the renewal
        # guarantee IS the inter-arrival family's guarantee -- delegate to it honestly.
        inner = getattr(obj, "interarrival", None)
        inner_plan = _classify_leaf(inner, name) if inner is not None else None
        guarantee = inner_plan.candidate_guarantee if inner_plan is not None else Guarantee.STATIONARY
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
                "inherits only its candidate guarantee; censored-boundary and identification checks remain due"
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
            reason="exponential-family capability suggests an analytic MLE; support and identification remain due",
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
            reason="GLM-shaped block suggests IRLS; rank, separation, convergence, and objective checks remain due",
        )
    if _has_exact_density(obj):
        return BlockPlan(
            name,
            kind,
            "unspecified_exact_density_estimator",
            Guarantee.UNVERIFIED,
            gradient=False,
            placement="local",
            reason="exact density supports scoring, not an estimation or optimality guarantee",
        )
    return BlockPlan(
        name,
        kind,
        "iterative",
        Guarantee.STATIONARY,
        gradient=False,
        placement="local",
        reason="iterative estimator; stationarity requires a convergence receipt",
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
            reason="conditional-linear-Gaussian least squares; full-rank design and residual support remain due",
        )
    if kind == "_GLMFactor":
        return BlockPlan(
            name,
            "GLM",
            "convex_irls",
            Guarantee.GLOBAL,
            gradient=False,
            placement="local",
            reason="GLM factor; rank, separation, convergence, and objective checks remain due",
        )
    if kind == "_DiscreteConditionalFactor":
        return BlockPlan(
            name,
            "table",
            "closed_form_counts",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="conditional count table; every parent configuration must have checked exposure",
        )
    if kind == "_VectorMarginalFactor":
        return BlockPlan(
            name,
            "MVN",
            "closed_form_mle",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="multivariate-Gaussian moments; full-rank empirical covariance remains due",
        )
    if kind == "_VectorCLGFactor":
        return BlockPlan(
            name,
            "vector_CLG",
            "least_squares",
            Guarantee.GLOBAL_UNIQUE,
            gradient=False,
            placement="local",
            reason="multivariate least squares; design and residual covariance ranks remain due",
        )
    return _classify_leaf(fac, name)


def _walk(obj: Any, name: str, blocks: list[BlockPlan]) -> None:
    """Recurse the fitted-model tree, appending a BlockPlan per leaf/factor block."""
    kind = type(obj).__name__

    # composite: independent per-field blocks -- the whole thing factorizes, each child is its own fit
    if isinstance(getattr(obj, "dists", None), (list, tuple)):
        for i, child in enumerate(obj.dists):
            _walk(child, f"{name}field[{i}]" if name else f"field[{i}]", blocks)
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
        blocks.append(
            BlockPlan(
                f"{name}mixture" if name else "mixture",
                kind,
                "em",
                Guarantee.STATIONARY_ESCAPE_TESTED,
                gradient=False,
                placement="local",
                reason=("latent mixture EM; stationarity and any saddle-escape claim require optimizer receipts"),
            )
        )
        for i, comp in enumerate(obj.components):
            _walk(comp, f"{name}component[{i}]." if name else f"component[{i}].", blocks)
        return

    # a single leaf
    blocks.append(_classify_leaf(obj, name.rstrip(".") if name else kind))


def _proof_obligations(block: BlockPlan) -> tuple[ProofObligation, ...]:
    """Return the evidence required to attain ``block.candidate_guarantee``."""
    candidate = block.candidate_guarantee
    if candidate <= Guarantee.HEURISTIC:
        return ()
    if candidate == Guarantee.STATIONARY:
        return (
            ProofObligation(
                "optimizer_converged",
                Guarantee.STATIONARY,
                "the actual optimizer terminated at a checked stationary point",
            ),
        )
    if candidate == Guarantee.STATIONARY_ESCAPE_TESTED:
        return (
            ProofObligation(
                "optimizer_converged",
                Guarantee.STATIONARY,
                "the actual optimizer terminated at a checked stationary point",
            ),
            ProofObligation(
                "saddle_escape_compared",
                Guarantee.STATIONARY_ESCAPE_TESTED,
                "independent or symmetry-broken starts were evaluated on the same fitting objective",
            ),
        )
    if candidate == Guarantee.GLOBAL:
        if block.method == "convex_irls":
            return (
                ProofObligation(
                    "finite_supported_data",
                    Guarantee.GLOBAL,
                    "responses, weights, offsets, and design values are finite and in support",
                ),
                ProofObligation(
                    "identified_finite_optimum",
                    Guarantee.GLOBAL,
                    "rank and separation checks establish that a finite optimum exists",
                ),
                ProofObligation(
                    "objective_globally_convex",
                    Guarantee.GLOBAL,
                    "the fitted objective, including any penalty, is globally convex/concave",
                ),
                ProofObligation(
                    "solver_matches_objective",
                    Guarantee.GLOBAL,
                    "a convergence report establishes optimality for that exact objective",
                ),
            )
        return (
            ProofObligation(
                "objective_globally_convex",
                Guarantee.GLOBAL,
                "the fitted objective is globally convex/concave on the stated parameter domain",
            ),
            ProofObligation(
                "solver_matches_objective",
                Guarantee.GLOBAL,
                "the reported parameters solve that exact objective under its constraints",
            ),
        )
    return (
        ProofObligation(
            "finite_supported_data",
            Guarantee.GLOBAL,
            "all observations and weights are finite and lie in the model support",
        ),
        ProofObligation(
            "solver_matches_objective",
            Guarantee.GLOBAL,
            "the reported closed-form parameters solve the stated fitting objective",
        ),
        ProofObligation(
            "identified_parameters",
            Guarantee.GLOBAL_UNIQUE,
            "rank, exposure, support, and covariance checks establish parameter identification",
        ),
    )


def receipt_subject(model: Any) -> str:
    """Return the subject identifier a :class:`VerificationReceipt` must carry to be about ``model``.

    A caller supplying optimizer evidence the library cannot produce itself -- multi-start escape
    testing, a solver's own convergence report -- names the model it verified::

        VerificationReceipt(..., subject_hash=receipt_subject(model))

    Requiring this is the point (MXR-080-1877): before it, naming the right check strings was the
    whole test, so a receipt whose evidence read ``{"trust": "me"}`` upgraded an arbitrary
    ``GaussianDistribution(999, 1)`` to ``GLOBAL_UNIQUE``. Binding does not make forgery impossible --
    anyone holding the model can compute this -- but it means a receipt cannot be written without the
    model, cannot be reused against a different one, and stops applying the moment the parameters
    change. What a reader gains is in the certificate: ``BlockPlan.verified_by`` names who established
    each guarantee, so library-checked conditions no longer read identically to a caller's assertion.
    """
    return _subject_hash(model)


def _subject_hash(model: Any) -> str:
    """Identify the exact object a receipt is about (MXR-080-1877).

    A model whose parameters cannot be serialized cannot be identified, and evidence cannot be bound
    to a subject nobody can name. Rather than fall back to a value a caller could also produce, this
    returns an unguessable per-call token: receipts minted inside the same :func:`certify` call still
    match, and a caller-supplied receipt cannot, so an unidentifiable model simply cannot have its
    guarantee raised from outside.
    """
    from mixle.data.hashing import model_hash

    try:
        return f"model:{model_hash(model)}"
    except Exception:  # noqa: BLE001 - an unserializable model is unidentifiable, not a crash here
        return f"unidentifiable:{secrets.token_hex(16)}"


def _receipt_for(
    block: BlockPlan, guarantee: Guarantee, evidence: dict[str, Any], subject_hash: str = ""
) -> VerificationReceipt:
    checks = tuple(obligation.check for obligation in block.proof_obligations if obligation.required_for <= guarantee)
    payload = {
        "block": block.name,
        "guarantee": guarantee.label,
        "checks": checks,
        "evidence": evidence,
        "subject": subject_hash,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode()).hexdigest()[:20]
    return VerificationReceipt(
        receipt_id=f"estimation-{digest}",
        block=block.name,
        guarantee=guarantee,
        checks=checks,
        subject_hash=subject_hash,
        source="mixle.inference.planning.verify_estimation_conditions",
        evidence=evidence,
    )


def _leaf_evidence(obj: Any, data: Sequence[Any]) -> dict[str, Any] | None:
    """Check identification conditions for the small set of analytic MLEs we can prove locally."""
    import numpy as np

    kind = type(obj).__name__
    n = len(data)
    if n == 0:
        return None
    if kind in {"GaussianDistribution", "ExponentialDistribution", "PoissonDistribution", "BernoulliDistribution"}:
        try:
            values = np.asarray(data, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size != n or not np.all(np.isfinite(values)):
            return None
        if kind == "GaussianDistribution":
            variance = float(np.var(values))
            mean = float(np.mean(values))
            if (
                n < 2
                or not variance > 0.0
                or not np.isclose(float(obj.mu), mean, rtol=1e-7, atol=1e-10)
                or not np.isclose(float(obj.sigma2), variance, rtol=1e-7, atol=1e-10)
            ):
                return None
            return {
                "n": n,
                "sample_mean": mean,
                "sample_variance": variance,
                "parameters_match_mle": True,
                "objective": "Gaussian log-likelihood",
            }
        if kind == "ExponentialDistribution":
            expected = float(np.mean(values))
            if (
                np.any(values < 0.0)
                or not float(np.sum(values)) > 0.0
                or not np.isclose(float(obj.beta), expected, rtol=1e-7, atol=1e-10)
            ):
                return None
            return {
                "n": n,
                "sum": float(np.sum(values)),
                "parameters_match_mle": True,
                "objective": "exponential log-likelihood",
            }
        if kind == "PoissonDistribution":
            expected = float(np.mean(values))
            if (
                np.any(values < 0.0)
                or np.any(values != np.floor(values))
                or not float(np.sum(values)) > 0.0
                or not np.isclose(float(obj.lam), expected, rtol=1e-7, atol=1e-10)
            ):
                return None
            return {
                "n": n,
                "event_count": int(np.sum(values)),
                "parameters_match_mle": True,
                "objective": "Poisson log-likelihood",
            }
        levels = set(values.tolist())
        expected = float(np.mean(values))
        if (
            not levels <= {0.0, 1.0}
            or levels != {0.0, 1.0}
            or not np.isclose(float(obj.p), expected, rtol=1e-7, atol=1e-10)
        ):
            return None
        return {
            "n": n,
            "counts": [int(np.sum(values == 0.0)), int(np.sum(values == 1.0))],
            "parameters_match_mle": True,
        }

    if kind == "CategoricalDistribution":
        try:
            observed = set(data)
            fitted = set(obj.pmap)
        except (AttributeError, TypeError):
            return None
        if not fitted or observed != fitted or float(getattr(obj, "default_value", 0.0)) != 0.0:
            return None
        counts = {value: sum(item == value for item in data) for value in observed}
        expected = {value: count / n for value, count in counts.items()}
        if any(not np.isclose(float(obj.pmap[value]), probability) for value, probability in expected.items()):
            return None
        return {
            "n": n,
            "observed_levels": sorted(repr(value) for value in observed),
            "parameters_match_mle": True,
        }

    if kind in {"DiagonalGaussianDistribution", "MultivariateGaussianDistribution"}:
        try:
            values = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if values.ndim != 2 or values.shape[0] != n or not np.all(np.isfinite(values)):
            return None
        dim = values.shape[1]
        if (kind == "DiagonalGaussianDistribution" and n < 2) or (
            kind == "MultivariateGaussianDistribution" and n <= dim
        ):
            return None
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=0))
        if kind == "DiagonalGaussianDistribution":
            minimum = float(np.min(np.diag(covariance)))
            if (
                not minimum > 0.0
                or not np.allclose(np.asarray(obj.mu), np.mean(values, axis=0), rtol=1e-7, atol=1e-10)
                or not np.allclose(np.asarray(obj.covar), np.diag(covariance), rtol=1e-7, atol=1e-10)
            ):
                return None
            return {
                "n": n,
                "dimension": dim,
                "minimum_empirical_variance": minimum,
                "parameters_match_mle": True,
            }
        # Full-covariance fits are regularized by the estimator. Without a declared
        # regularized objective they cannot be certified as the exact Gaussian MLE.
        return None

    if kind == "InhomogeneousPoissonProcessDistribution":
        try:
            counts = np.zeros(obj.num_bins, dtype=np.int64)
            for realization in data:
                times = np.asarray(realization, dtype=np.float64)
                if times.ndim != 1 or not np.all(np.isfinite(times)):
                    return None
                if np.any(times < obj.edges[0]) or np.any(times > obj.edges[-1]):
                    return None
                counts += np.histogram(times, bins=obj.edges)[0]
            exposure = len(data) * np.asarray(obj.widths, dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            return None
        expected = counts / exposure
        if not np.all(exposure > 0.0) or not np.allclose(obj.rates, expected, rtol=1e-7, atol=1e-10):
            return None
        return {
            "n_realizations": n,
            "minimum_bin_exposure": float(np.min(exposure)),
            "bin_counts": counts.tolist(),
            "parameters_match_mle": True,
        }

    if kind == "ContinuousTimeMarkovChainDistribution":
        try:
            from mixle.stats.processes.ctmc import _trajectory_stats

            dwell = np.zeros(obj.num_states, dtype=np.float64)
            transitions = np.zeros((obj.num_states, obj.num_states), dtype=np.float64)
            for trajectory in data:
                count, exposure = _trajectory_stats(trajectory, obj.num_states)
                transitions += count
                dwell += exposure
        except (AttributeError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(dwell)) or not np.all(dwell > 0.0):
            return None
        expected = transitions / dwell[:, None]
        np.fill_diagonal(expected, 0.0)
        if not np.allclose(obj.rates, expected, rtol=1e-7, atol=1e-10):
            return None
        return {
            "n_trajectories": n,
            "minimum_state_exposure": float(np.min(dwell)),
            "transition_count": int(np.sum(transitions)),
            "parameters_match_mle": True,
        }

    if kind == "BirthDeathSamplingDistribution":
        try:
            from mixle.stats.processes.birth_death import _trajectory_stats

            statistics = [_trajectory_stats(trajectory) for trajectory in data]
            exposure = float(sum(item[3] for item in statistics))
            counts = np.sum(np.asarray([item[:3] for item in statistics], dtype=np.float64), axis=0)
        except (TypeError, ValueError):
            return None
        fitted = np.asarray([obj.birth_rate, obj.death_rate, obj.sampling_rate], dtype=np.float64)
        if (
            not np.isfinite(exposure)
            or not exposure > 0.0
            or not np.allclose(fitted, counts / exposure, rtol=1e-7, atol=1e-10)
        ):
            return None
        return {"n_trajectories": n, "population_exposure": exposure, "parameters_match_mle": True}
    return None


def _linear_evidence(factor: Any, columns: list[list[Any]]) -> dict[str, Any] | None:
    import numpy as np

    try:
        design = np.asarray(factor._design(columns), dtype=np.float64)
        response = np.asarray(columns[factor.child], dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        design.ndim != 2
        or design.shape[0] <= design.shape[1]
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(response))
        or np.linalg.matrix_rank(design) != design.shape[1]
    ):
        return None
    expected_coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    if not np.allclose(np.asarray(factor.coef), expected_coef, rtol=1e-7, atol=1e-10):
        return None
    residual = response - design @ expected_coef
    residual_measure = float(np.var(residual))
    if not residual_measure > 0.0 or not np.isclose(float(factor.sigma) ** 2, residual_measure, rtol=1e-7, atol=1e-10):
        return None
    return {
        "n": int(design.shape[0]),
        "design_columns": int(design.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "minimum_residual_variance": residual_measure,
    }


def verify_estimation_conditions(model: Any, data: Iterable[Any]) -> list[VerificationReceipt]:
    """Check data-dependent proof obligations for supported analytic estimators.

    Unsupported or iterative estimators produce no receipt rather than a
    guessed guarantee. The returned receipts can be passed to :func:`certify`.
    """
    rows = list(data)
    if not rows:
        return []
    subject = _subject_hash(model)  # binds every receipt below to THIS model (MXR-080-1877)
    blocks: list[BlockPlan] = []
    _walk(model, "", blocks)
    by_name = {block.name: block for block in blocks}
    for block in blocks:
        block.candidate_guarantee = block.guarantee
        block.proof_obligations = _proof_obligations(block)
    receipts: list[VerificationReceipt] = []

    def add_leaf(obj: Any, values: Sequence[Any], name: str) -> None:
        block = by_name.get(name)
        if block is None or block.candidate_guarantee < Guarantee.GLOBAL:
            return
        evidence = _leaf_evidence(obj, values)
        if evidence is not None:
            receipts.append(_receipt_for(block, block.candidate_guarantee, evidence, subject))

    if isinstance(getattr(model, "dists", None), (list, tuple)):
        for index, child in enumerate(model.dists):
            try:
                values = [row[index] for row in rows]
            except (IndexError, TypeError):
                continue
            add_leaf(child, values, f"field[{index}]")
        return receipts

    if hasattr(model, "factors") and hasattr(model, "edges"):
        try:
            columns = [[row[index] for row in rows] for index in range(len(rows[0]))]
        except (IndexError, TypeError):
            return receipts
        for factor in model.factors:
            name = f"node[{factor.child}]"
            block = by_name[name]
            kind = type(factor).__name__
            if kind == "_MarginalFactor":
                add_leaf(factor.dist, columns[factor.child], name)
            elif kind == "_LinearGaussianFactor":
                evidence = _linear_evidence(factor, columns)
                if evidence:
                    receipts.append(_receipt_for(block, block.candidate_guarantee, evidence, subject))
            # Vector fits add an undeclared covariance ridge; conditional tables
            # cannot establish unobserved parent cells from the fitted object;
            # GLMs require convergence/separation evidence. All remain unverified.
        return receipts

    # Responsibilities or latent assignments are required to verify mixture component M-steps.
    if isinstance(getattr(model, "components", None), (list, tuple)):
        return receipts

    name = blocks[0].name if blocks else type(model).__name__
    add_leaf(model, rows, name)
    return receipts


def _prepare_blocks(model: Any) -> list[BlockPlan]:
    blocks: list[BlockPlan] = []
    _walk(model, "", blocks)
    if not blocks:
        blocks.append(_classify_leaf(model, type(model).__name__))
    for block in blocks:
        block.candidate_guarantee = block.guarantee
        block.proof_obligations = _proof_obligations(block)
        if block.guarantee > Guarantee.HEURISTIC:
            block.guarantee = Guarantee.UNVERIFIED
    return blocks


def certify(
    model: Any,
    *,
    data: Iterable[Any] | None = None,
    receipts: Iterable[VerificationReceipt] = (),
    escape_tested: bool = False,
    penalized: str | bool = False,
) -> EstimationCertificate:
    """Return the :class:`EstimationCertificate` for a fitted model (or distribution prototype).

    Model structure supplies only a candidate guarantee. ``data`` is checked by
    :func:`verify_estimation_conditions`; additional optimizer evidence may be
    supplied as structured ``receipts``. A bare ``escape_tested=True`` is
    rejected because a caller assertion is not verification evidence.

    Pass ``penalized`` (a reason string, or True) when the fit optimized a penalized objective -- soft
    constraints, conservation/PINN residual factors, potentials (E2). The optimum is then of the
    penalized surrogate, not the likelihood, so no block may claim more than STATIONARY however clean
    its own solver is: every stronger block is downgraded with the penalty named in its reason.
    """
    # Exact, not truthy (MXR-080-1905). `if escape_tested:` meant escape_tested="false" -- the string
    # a flag arrives as from configuration -- raised, while escape_tested=0 was a silent no-op. A flag
    # whose whole purpose is to be refused must at least refuse the right values.
    if require_exact_bool(escape_tested, "certify escape_tested"):
        raise ValueError("escape_tested=True is not evidence; supply a VerificationReceipt")
    blocks = _prepare_blocks(model)
    if penalized:
        why = penalized if isinstance(penalized, str) else "soft-constraint / residual penalty"
        for b in blocks:
            if b.candidate_guarantee > Guarantee.STATIONARY:
                b.candidate_guarantee = Guarantee.STATIONARY
                b.proof_obligations = _proof_obligations(b)
                b.reason += f" [CANDIDATE CAPPED: penalized objective ({why}) -- optimum concerns the surrogate]"
    supplied = list(receipts)
    if data is not None:
        supplied.extend(verify_estimation_conditions(model, data))
    # The subject every receipt must name to raise a guarantee about THIS model (MXR-080-1877).
    # Naming the right check strings used to be the entire test, so a caller-authored receipt whose
    # evidence read {"trust": "me"} moved an arbitrary GaussianDistribution(999, 1) from UNVERIFIED to
    # GLOBAL_UNIQUE. A receipt that names no subject, or names a different one, is no longer evidence
    # about this model -- it is ignored for the upgrade and recorded as ignored, rather than silently
    # dropped, so a reader can tell "no evidence was offered" from "evidence was offered and refused".
    subject = _subject_hash(model) if data is None else None
    by_name = {block.name: block for block in blocks}
    for receipt in supplied:
        if not isinstance(receipt, VerificationReceipt):
            raise TypeError("receipts must contain VerificationReceipt instances")
        block = by_name.get(receipt.block)
        if block is None:
            raise ValueError(f"verification receipt names unknown block {receipt.block!r}")
        established = min(receipt.guarantee, block.candidate_guarantee)
        required = {
            obligation.check for obligation in block.proof_obligations if obligation.required_for <= established
        }
        if not required.issubset(receipt.checks):
            continue
        if subject is None:
            subject = _subject_hash(model)
        if receipt.subject_hash != subject:
            block.unbound_receipt_ids = (*block.unbound_receipt_ids, receipt.receipt_id)
            continue
        if established > block.guarantee:
            block.guarantee = established
            block.verified_checks = tuple(sorted(required))
            block.receipt_ids = (receipt.receipt_id,)
            block.verified_by = receipt.source
    aggregate = min((b.guarantee for b in blocks), default=Guarantee.UNVERIFIED)
    mixture_blocks = [block for block in blocks if block.method == "em"]
    escape_verified = bool(mixture_blocks) and all(
        block.guarantee >= Guarantee.STATIONARY_ESCAPE_TESTED for block in mixture_blocks
    )
    return EstimationCertificate(guarantee=aggregate, blocks=blocks, escape_tested=escape_verified)


def plan_estimation(model: Any) -> EstimationCertificate:
    """Return a pre-fit plan whose guarantees remain explicitly unverified."""
    return certify(model)


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


def schedule(
    model: Any, *, escape_tested: bool = False, receipts: Iterable[VerificationReceipt] = ()
) -> EstimationSchedule:
    """Plan the block-coordinate estimation schedule for ``model`` (planner v2, A3).

    Built from the same block classification as :func:`certify`: EM blocks make the schedule a loop
    (E-step + per-block M-steps, repeated until convergence); without a latent block every block is one
    independent pass. Gradient blocks appear as explicit ``gradient`` passes with their pool placement,
    so the schedule is also the offload plan for the hybrid case.

    ``receipts`` exists because ``escape_tested`` was otherwise unusable in both directions
    (MXR-080-1905). It forwards to :func:`certify`, which refuses ``escape_tested=True`` with
    "supply a VerificationReceipt" -- and this function had no parameter through which to supply one,
    so the only non-default value raised an error naming a remedy the caller could not reach. Pass
    escape evidence here instead, bound to the model via
    :func:`receipt_subject` (MXR-080-1877).
    """
    cert = certify(model, escape_tested=escape_tested, receipts=receipts)
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
