"""Structural interventions on a heterogeneous Bayesian network.

``do(net, ...)`` performs graph surgery on the supplied directed factorization. That operation is
useful for simulation, but observational structure learning does not establish that an arrow is a
causal direction. Consequently an unaccompanied ``do`` result is explicitly an *unidentified
structural scenario*, not an estimate of an intervention in the data-generating world.

Callers that have a domain-asserted causal graph or an identified design can attach a
:class:`CausalIdentification` receipt. APIs whose names make a causal claim, such as
:func:`average_causal_effect` and :func:`counterfactual`, require that receipt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CausalIdentification:
    """Auditable assumptions under which a structural contrast is interpreted causally.

    This record does not prove its assertions. It makes the assertions reviewable and prevents a
    learned observational arrow from silently becoming a causal result. ``graph_source`` must say
    how the direction was established, and ``evidence`` should identify the design, protocol, or
    domain artifact supporting the claim.
    """

    graph_source: str
    estimand: str
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    exchangeability: bool
    positivity: bool
    consistency: bool
    no_interference: bool
    structural_counterfactuals: bool = False

    def __post_init__(self) -> None:
        allowed_sources = {"domain_asserted", "randomized_design", "identified_natural_experiment"}
        if self.graph_source not in allowed_sources:
            raise ValueError(
                "graph_source must be domain_asserted, randomized_design, or "
                "identified_natural_experiment; a learned observational graph is not causal evidence"
            )
        if not isinstance(self.estimand, str) or not self.estimand.strip():
            raise ValueError("estimand must be a non-empty description")
        if not self.evidence or any(not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise ValueError("evidence must contain at least one non-empty design or domain reference")
        if not self.assumptions or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must contain at least one non-empty identification assumption")

    @property
    def identified(self) -> bool:
        """Whether all common treatment-effect identification assumptions were declared."""
        return bool(
            self.exchangeability and self.positivity and self.consistency and self.no_interference
        )

    @classmethod
    def domain_asserted(
        cls,
        evidence: str,
        *,
        estimand: str = "average treatment effect",
        assumptions: tuple[str, ...] = ("causal DAG supplied independently of fitted associations",),
        structural_counterfactuals: bool = False,
    ) -> CausalIdentification:
        """Construct a receipt for a domain-asserted causal DAG.

        This convenience constructor records that the caller accepts the standard identification
        assumptions. Applications needing qualified assumptions should instantiate the dataclass
        directly rather than use this shorthand.
        """
        return cls(
            graph_source="domain_asserted",
            estimand=estimand,
            evidence=(evidence,),
            assumptions=assumptions,
            exchangeability=True,
            positivity=True,
            consistency=True,
            no_interference=True,
            structural_counterfactuals=structural_counterfactuals,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the receipt."""
        return asdict(self)


class InterventionalNetwork:
    """A graph-surgery scenario with an explicit causal-identification status."""

    def __init__(
        self,
        net: Any,
        interventions: dict[int, Any],
        identification: CausalIdentification | None = None,
    ) -> None:
        self.net = net
        self.interventions = dict(interventions)
        self.identification = identification
        self.identified = identification is not None and identification.identified
        self.interpretation = (
            "identified intervention under the attached assumptions"
            if self.identified
            else "unidentified structural scenario; not a causal estimate"
        )
        # Validate by *membership* against the network's actual (int) node ids -- NOT by coercing
        # each key through int(k), which would silently accept e.g. "0" (a str) as if it were the
        # node id 0. dict/`in` lookups elsewhere in this class (e.g. `sample`'s `i in
        # self.interventions`) use the real int field ids from `net.order`, so a str key that merely
        # *looks* like a valid id would never match there and the intervention would be silently
        # dropped instead of applied -- this must be a loud error, not a no-op.
        valid_fields = set(net.order)
        for k in self.interventions:
            if k not in valid_fields:
                raise ValueError(
                    f"intervened field {k!r} is not a valid node id of this network "
                    f"(expected an int in {sorted(valid_fields)})"
                )

    def sample(self, size: int = 1, *, seed: int | None = None) -> list[tuple]:
        """Ancestral sampling with the intervened fields clamped (their factors are never consulted)."""
        rng = np.random.RandomState(seed)
        by_child = {f.child: f for f in self.net.factors}
        rows: list[tuple] = []
        for _ in range(int(size)):
            vals: list[Any] = [None] * len(self.net.factors)
            for i in self.net.order:
                vals[i] = self.interventions[i] if i in self.interventions else by_child[i].sample(vals, rng)
            rows.append(tuple(vals))
        return rows

    def expectation(self, field: int, *, n: int = 4000, seed: int = 0) -> float:
        """Monte-Carlo expectation in this structural scenario."""
        draws = [row[field] for row in self.sample(n, seed=seed)]
        return float(np.mean(np.asarray(draws, dtype=np.float64)))

    def distribution(self, field: int, *, n: int = 4000, seed: int = 0) -> dict[Any, float]:
        """Structural-scenario marginal of a discrete field as ``{value: probability}``."""
        draws = [row[field] for row in self.sample(n, seed=seed)]
        counts = Counter(draws)
        return {k: v / len(draws) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def do(
    net: Any,
    interventions: dict[int, Any],
    *,
    identification: CausalIdentification | None = None,
) -> InterventionalNetwork:
    """Apply graph surgery, labeling the result unidentified unless a valid receipt is supplied."""
    if not hasattr(net, "factors") or not hasattr(net, "order"):
        raise TypeError("do() expects a learned HeterogeneousBayesianNetwork")
    if identification is not None:
        _require_identification(identification)
    return InterventionalNetwork(net, interventions, identification)


def average_causal_effect(
    net: Any,
    treatment: int,
    a: Any,
    b: Any,
    outcome: int,
    *,
    identification: CausalIdentification | None = None,
    n: int = 4000,
    seed: int = 0,
) -> float:
    """Return an identified average causal contrast under an explicit assumption receipt."""
    receipt = _require_identification(identification)
    ea = do(net, {treatment: a}, identification=receipt).expectation(outcome, n=n, seed=seed)
    eb = do(net, {treatment: b}, identification=receipt).expectation(outcome, n=n, seed=seed)
    return float(ea - eb)


def counterfactual(
    net: Any,
    observed: tuple,
    interventions: dict[int, Any],
    *,
    identification: CausalIdentification | None = None,
) -> tuple:
    """What this observed record would have been under the intervention (abduction-action-prediction).

    Per Pearl's three steps, walked in topological order:

      * **abduction** -- a linear-Gaussian field's exogenous noise is point-identified from the row:
        its residual ``eps = observed - coef @ parents_observed``;
      * **action** -- intervened fields take their ``do`` values;
      * **prediction** -- the same residual replays through the counterfactual parents:
        ``cf = coef @ parents_cf + eps``.

    An identification receipt with ``structural_counterfactuals=True`` is required. Boundaries:
    (1) a field that is not linear-Gaussian keeps its observed value only while its
    parents are unchanged under the intervention (that much IS identified); if its parents change, its
    exogenous noise cannot be recovered from one observation and this raises — use
    :func:`average_causal_effect` for the population answer instead of a guessed individual one.
    (2) The counterfactual is relative to the network's DAG **as given**: purely observational structure
    learning cannot orient Markov-equivalent edges (x -> y and y -> x fit equally well), so if the
    causal direction matters, assert it from domain knowledge rather than trusting the learned arrow.
    """
    from mixle.inference.bayesian_network import _LinearGaussianFactor

    receipt = _require_identification(identification)
    if not receipt.structural_counterfactuals:
        raise ValueError(
            "counterfactual() requires identification.structural_counterfactuals=True to declare "
            "that the structural noise model supports individual counterfactuals"
        )
    if not hasattr(net, "factors") or not hasattr(net, "order"):
        raise TypeError("counterfactual() expects a learned HeterogeneousBayesianNetwork")
    observed = tuple(observed)
    if len(observed) != len(net.factors):
        raise ValueError(f"observed record has {len(observed)} fields; the network has {len(net.factors)}")
    fixed = {int(k): v for k, v in interventions.items()}
    by_child = {f.child: f for f in net.factors}
    cf: list[Any] = [None] * len(net.factors)
    for i in net.order:
        if i in fixed:
            cf[i] = fixed[i]
            continue
        f = by_child[i]
        if isinstance(f, _LinearGaussianFactor):
            mu_obs = float(f._row([observed[p] for p in f.parents]) @ f.coef)
            eps = float(observed[f.child]) - mu_obs  # abduction
            mu_cf = float(f._row([cf[p] for p in f.parents]) @ f.coef)
            cf[i] = mu_cf + eps  # action + prediction
            continue
        if any(not _same_value(cf[p], observed[p]) for p in getattr(f, "parents", [])):
            raise ValueError(
                f"counterfactual for field {i} is not point-identified: it is not linear-Gaussian and its "
                f"parents changed under the intervention; use average_causal_effect for the population answer."
            )
        cf[i] = observed[i]
    return tuple(cf)


def _require_identification(
    identification: CausalIdentification | None,
) -> CausalIdentification:
    if not isinstance(identification, CausalIdentification):
        raise ValueError(
            "a CausalIdentification receipt is required for a causal claim; use do() without a "
            "receipt only as an unidentified structural scenario"
        )
    if not identification.identified:
        raise ValueError(
            "the causal identification receipt must affirm exchangeability, positivity, "
            "consistency, and no interference"
        )
    return identification


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return a == b
