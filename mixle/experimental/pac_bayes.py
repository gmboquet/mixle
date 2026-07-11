"""P10 (experimental) -- compositional PAC-Bayes generalization certificates.

A PAC-Bayes bound turns a fit into a *certificate*: with probability at least ``1 - delta`` over
the training sample, the true (held-out) risk of the fitted model is at most its empirical risk
plus a complexity term driven by ``KL(posterior || prior)``. For exponential-family leaves the KL
is closed form, and -- the point of doing this in mixle -- it **composes along the estimator
tree**: the total KL is the sum of per-node KLs, so a loose bound can be blamed on the subtree
that contributed the most complexity.

This module implements the McAllester bound over a bounded likelihood loss for Gaussian mixtures:

* :func:`gaussian_kl` -- closed-form KL between two Gaussians;
* :func:`per_component_kl` / :func:`total_kl` -- the per-node decomposition and its sum;
* :func:`mcallester_bound` -- the PAC-Bayes upper bound;
* :func:`certify_generalization` -- fit -> :class:`GeneralizationCertificate` (bound + blame).

Honest scope (P10 kill criterion): the bound is only useful if it is non-vacuous; the test
measures the bound-vs-held-out gap across sample sizes and the empirical ``1 - delta`` coverage,
rather than assuming tightness.

Exploratory ``mixle.experimental`` code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def gaussian_kl(mu_q: float, s2_q: float, mu_p: float, s2_p: float) -> float:
    """``KL( N(mu_q, s2_q) || N(mu_p, s2_p) )`` in nats (closed form)."""
    return float(0.5 * (np.log(s2_p / s2_q) + (s2_q + (mu_q - mu_p) ** 2) / s2_p - 1.0))


def _categorical_kl(q: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(np.asarray(q, dtype=float), 1e-12, None)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))


def per_component_kl(model: Any, prior: Any) -> list[float]:
    """Per-component Gaussian KL between the fitted mixture and the prior mixture.

    Both must be Gaussian mixtures with the same number of components (the prior's components are
    the per-node priors -- typically broad). This is the per-node blame vector.
    """
    qc, pc = list(model.components), list(prior.components)
    if len(qc) != len(pc):
        raise ValueError("per_component_kl requires model and prior with equal component counts")
    return [gaussian_kl(q.mu, q.sigma2, p.mu, p.sigma2) for q, p in zip(qc, pc)]


def total_kl(model: Any, prior: Any) -> float:
    """Total ``KL(Q||P)`` = sum of per-component Gaussian KL + the mixing-weight KL (it composes)."""
    node = sum(per_component_kl(model, prior))
    weights = _categorical_kl(np.asarray(model.w, dtype=float), np.asarray(prior.w, dtype=float))
    return float(node + weights)


def _max_logdensity_ub(model: Any) -> float:
    """A data-free upper bound on the mixture's max log-density (sum of component peaks)."""
    peaks = [w / np.sqrt(2.0 * np.pi * c.sigma2) for w, c in zip(model.w, model.components)]
    return float(np.log(np.sum(peaks)))


def bounded_losses(model: Any, data: Any) -> np.ndarray:
    """The bounded likelihood loss ``l(x) = 1 - exp(logdensity(x) - c) in [0, 1)`` per observation.

    ``c`` is the data-free max-log-density upper bound, so ``exp(logdensity - c) in (0, 1]`` and the
    loss is a valid ``[0, 1]``-bounded PAC-Bayes loss.
    """
    c = _max_logdensity_ub(model)
    ld = np.asarray([model.log_density(float(x)) for x in data], dtype=float)
    return 1.0 - np.exp(np.minimum(ld - c, 0.0))


def mcallester_bound(empirical_loss: float, kl: float, n: int, *, delta: float = 0.05) -> float:
    """McAllester PAC-Bayes bound: ``emp + sqrt((KL + ln(2 sqrt(n) / delta)) / (2n))``."""
    complexity = (kl + np.log(2.0 * np.sqrt(n) / delta)) / (2.0 * n)
    return float(empirical_loss + np.sqrt(max(complexity, 0.0)))


@dataclass
class GeneralizationCertificate:
    """A PAC-Bayes certificate attached to a fit."""

    empirical_loss: float
    kl: float
    n: int
    delta: float
    bound: float
    per_component_kl: list[float]
    vacuous: bool  # True if the bound is >= 1 (says nothing for a [0,1] loss)

    def as_dict(self) -> dict[str, Any]:
        return {
            "empirical_loss": self.empirical_loss,
            "kl": self.kl,
            "n": self.n,
            "delta": self.delta,
            "bound": self.bound,
            "per_component_kl": list(self.per_component_kl),
            "vacuous": self.vacuous,
        }

    def worst_subtree(self) -> int:
        """Index of the component contributing the most KL (where a loose bound is blamed)."""
        return int(np.argmax(self.per_component_kl))


def certify_generalization(
    model: Any, prior: Any, train_data: Any, *, delta: float = 0.05
) -> GeneralizationCertificate:
    """Build a PAC-Bayes certificate for ``model`` fitted on ``train_data`` under ``prior``."""
    data = list(train_data)
    n = len(data)
    emp = float(np.mean(bounded_losses(model, data)))
    components = per_component_kl(model, prior)
    kl = total_kl(model, prior)
    bound = mcallester_bound(emp, kl, n, delta=delta)
    return GeneralizationCertificate(
        empirical_loss=emp,
        kl=kl,
        n=n,
        delta=float(delta),
        bound=bound,
        per_component_kl=components,
        vacuous=bound >= 1.0,
    )
