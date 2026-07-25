"""PAC-Bayes certificates over an explicit finite hypothesis space.

The posterior and prior accepted here are probability distributions over fixed predictors/hypotheses—not
observation-density mixture components. The prior must be selected independently of the certified sample;
the posterior may depend on that sample. Callers provide one bounded loss per hypothesis and sample, and
an explicit :class:`PACBayesAssumptions` receipt. The certificate uses the finite-space
``KL(Q || P)`` and the McAllester bounded-loss inequality.

``gaussian_kl`` remains available only as a mathematical primitive for genuine Gaussian distributions over
parameters. It is not used by :func:`certify_generalization` and does not convert fitted observation
distributions into a hypothesis posterior.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

__all__ = [
    "PACBayesAssumptions",
    "GeneralizationCertificate",
    "gaussian_kl",
    "categorical_kl",
    "mcallester_bound",
    "certify_generalization",
]

_THEOREM = (
    "McAllester bounded-loss PAC-Bayes: with probability at least 1-delta over an IID sample, "
    "R(Q) <= Rhat(Q) + sqrt((KL(Q||P)+ln(2*sqrt(n)/delta))/(2*n))."
)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _probability_vector(value: Any, name: str, *, strictly_positive: bool) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector.")
    if strictly_positive and np.any(result <= 0):
        raise ValueError(f"{name} must assign strictly positive mass to every hypothesis.")
    if not strictly_positive and np.any(result < 0):
        raise ValueError(f"{name} must be non-negative.")
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} must have positive finite total mass.")
    result = result / total
    result.setflags(write=False)
    return result


def gaussian_kl(mu_q: float, s2_q: float, mu_p: float, s2_p: float) -> float:
    """``KL(N(mu_q, s2_q) || N(mu_p, s2_p))`` for genuine parameter laws."""
    mu_q = _finite_real(mu_q, "mu_q")
    mu_p = _finite_real(mu_p, "mu_p")
    s2_q = _finite_real(s2_q, "s2_q")
    s2_p = _finite_real(s2_p, "s2_p")
    if s2_q <= 0 or s2_p <= 0:
        raise ValueError("Gaussian variances must be positive.")
    result = 0.5 * (math.log(s2_p / s2_q) + (s2_q + (mu_q - mu_p) ** 2) / s2_p - 1.0)
    if result < -1e-12:
        raise RuntimeError("computed Gaussian KL is unexpectedly negative.")
    return max(float(result), 0.0)


def categorical_kl(posterior: Any, prior: Any) -> tuple[float, tuple[float, ...]]:
    """Finite-hypothesis ``KL(Q || P)`` and its exact per-hypothesis summands."""
    q = _probability_vector(posterior, "posterior", strictly_positive=False)
    p = _probability_vector(prior, "prior", strictly_positive=True)
    if q.shape != p.shape:
        raise ValueError("posterior and prior must index the same hypotheses.")
    terms = np.zeros_like(q)
    positive = q > 0
    terms[positive] = q[positive] * np.log(q[positive] / p[positive])
    total = float(terms.sum())
    if total < -1e-12 or not math.isfinite(total):
        raise RuntimeError("computed categorical KL is invalid.")
    return max(total, 0.0), tuple(float(term) for term in terms)


def mcallester_bound(empirical_loss: float, kl: float, n: int, *, delta: float = 0.05) -> float:
    """Return the raw McAllester upper bound for a loss in ``[0, 1]``."""
    empirical_loss = _finite_real(empirical_loss, "empirical_loss")
    kl = _finite_real(kl, "kl")
    delta = _finite_real(delta, "delta")
    if not 0 <= empirical_loss <= 1:
        raise ValueError("empirical_loss must lie in [0, 1].")
    if kl < 0:
        raise ValueError("kl must be non-negative.")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or int(n) <= 0:
        raise ValueError("n must be a positive exact integer.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between 0 and 1.")
    n = int(n)
    complexity = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
    return empirical_loss + math.sqrt(complexity)


@dataclass(frozen=True)
class PACBayesAssumptions:
    """Caller-attested premises required by the theorem."""

    hypothesis_space_id: str
    prior_id: str
    sample_iid: bool
    hypotheses_fixed_before_sample: bool
    prior_fixed_before_sample: bool
    loss_fixed_before_sample: bool
    loss_lower: float = 0.0
    loss_upper: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.hypothesis_space_id, str) or not self.hypothesis_space_id.strip():
            raise ValueError("hypothesis_space_id must be a non-empty string.")
        if not isinstance(self.prior_id, str) or not self.prior_id.strip():
            raise ValueError("prior_id must be a non-empty string.")
        premises = {
            "sample_iid": self.sample_iid,
            "hypotheses_fixed_before_sample": self.hypotheses_fixed_before_sample,
            "prior_fixed_before_sample": self.prior_fixed_before_sample,
            "loss_fixed_before_sample": self.loss_fixed_before_sample,
        }
        for name, value in premises.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean.")
            if not bool(value):
                raise ValueError(f"PAC-Bayes premise is not satisfied: {name}.")
        lower = _finite_real(self.loss_lower, "loss_lower")
        upper = _finite_real(self.loss_upper, "loss_upper")
        if lower != 0.0 or upper != 1.0:
            raise ValueError("this certificate currently requires loss bounds exactly [0, 1].")


@dataclass(frozen=True)
class GeneralizationCertificate:
    """A theorem-matched finite-hypothesis PAC-Bayes certificate."""

    empirical_loss: float
    empirical_loss_by_hypothesis: tuple[float, ...]
    kl: float
    hypothesis_kl_terms: tuple[float, ...]
    n: int
    delta: float
    raw_bound: float
    bound: float
    vacuous: bool
    posterior: tuple[float, ...]
    prior: tuple[float, ...]
    theorem: str
    assumptions: PACBayesAssumptions

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assumptions"] = asdict(self.assumptions)
        return result

    def largest_kl_term(self) -> int:
        """Index of the largest signed summand in the exact finite-space KL decomposition."""
        return int(np.argmax(self.hypothesis_kl_terms))


def certify_generalization(
    losses: Any,
    posterior: Any,
    prior: Any,
    *,
    assumptions: PACBayesAssumptions,
    delta: float = 0.05,
) -> GeneralizationCertificate:
    """Certify the Gibbs risk of ``posterior`` over a fixed finite hypothesis class.

    Args:
        losses: Matrix with shape ``(n_hypotheses, n_samples)`` and values in ``[0, 1]``.
        posterior: Sample-dependent probability vector ``Q`` over the hypothesis rows.
        prior: Data-independent probability vector ``P`` over the same rows.
        assumptions: Explicit theorem-premise receipt; any false premise fails closed.
        delta: Failure probability in ``(0, 1)``.
    """
    if not isinstance(assumptions, PACBayesAssumptions):
        raise TypeError("assumptions must be a PACBayesAssumptions receipt.")
    assumptions.validate()
    loss_matrix = np.asarray(losses, dtype=float)
    if loss_matrix.ndim != 2 or loss_matrix.shape[0] == 0 or loss_matrix.shape[1] == 0:
        raise ValueError("losses must have non-empty (n_hypotheses, n_samples) shape.")
    if not np.all(np.isfinite(loss_matrix)) or np.any(loss_matrix < 0) or np.any(loss_matrix > 1):
        raise ValueError("losses must contain only finite values in [0, 1].")
    q = _probability_vector(posterior, "posterior", strictly_positive=False)
    p = _probability_vector(prior, "prior", strictly_positive=True)
    if q.shape != (loss_matrix.shape[0],) or p.shape != q.shape:
        raise ValueError("posterior and prior must have one entry per hypothesis row.")

    empirical_by_hypothesis = loss_matrix.mean(axis=1)
    empirical_loss = float(q @ empirical_by_hypothesis)
    kl, terms = categorical_kl(q, p)
    delta = _finite_real(delta, "delta")
    raw_bound = mcallester_bound(empirical_loss, kl, loss_matrix.shape[1], delta=delta)
    return GeneralizationCertificate(
        empirical_loss=empirical_loss,
        empirical_loss_by_hypothesis=tuple(float(value) for value in empirical_by_hypothesis),
        kl=kl,
        hypothesis_kl_terms=terms,
        n=int(loss_matrix.shape[1]),
        delta=delta,
        raw_bound=raw_bound,
        bound=min(raw_bound, 1.0),
        vacuous=raw_bound >= 1.0,
        posterior=tuple(float(value) for value in q),
        prior=tuple(float(value) for value in p),
        theorem=_THEOREM,
        assumptions=assumptions,
    )
