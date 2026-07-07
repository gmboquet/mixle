"""Cycle-consistency as the cross-modal calibration and abstention signal (workstream F5).

A forward transport's own reported confidence can be blind to a real failure mode: an observation
function that COLLAPSES several distinct latents onto the same observed value is, by construction,
just as "confident" (the noise model is unchanged) whether or not the collapse actually happened for
this particular input -- marginal confidence has no way to see it. Round-trip closure does: draw
several independent posterior samples of the latent given the observation, and check whether they
AGREE WITH EACH OTHER (never against ground truth, which is unavailable at serving time) -- low
self-agreement is exactly the signature of a collapsed, irrecoverable region. This is the
"A -> B -> A on invariant content" test from the plan, made concrete and self-supervised: no oracle
needed, only repeated draws from the transport already fit for :mod:`mixle.models.mixture_density`.

Built on the exact fitting/sampling contract CARD TRANSPORT-a (workstream F0) already proved usable and
calibrated (:class:`~mixle.models.mixture_density.NeuralConditionalDensity` / ``build_mdn``, fit via
:func:`mixle.inference.optimize`) -- this module adds no new transport family, only the diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.inference import optimize


def fit_conditional_transport(
    given: np.ndarray,
    target: np.ndarray,
    *,
    k: int = 3,
    hidden: int = 32,
    layers: int = 2,
    max_its: int = 30,
    m_steps: int = 80,
    lr: float = 3e-3,
    seed: int = 0,
    delta: float | None = 1.0e-9,
    reuse_estep_ll: bool = True,
) -> Any:
    """Fit ``p(target | given)`` via a mixture density network (the same family/fitting path CARD
    TRANSPORT-a already proved calibrated) and return the fitted distribution.

    ``given``/``target`` are ``(n, d)`` arrays of paired observations. ``delta``/``reuse_estep_ll``
    default to :func:`~mixle.inference.optimize`'s own early-stopping; pass ``delta=None,
    reuse_estep_ll=False`` for a harder, more multimodal target (mirrors TRANSPORT-a's nonlinear case).
    """
    from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

    given = np.atleast_2d(np.asarray(given, dtype=np.float64))
    target = np.atleast_2d(np.asarray(target, dtype=np.float64))
    module = build_mdn(x_dim=given.shape[1], y_dim=target.shape[1], k=k, hidden=hidden, layers=layers)
    leaf = NeuralConditionalDensity(module, m_steps=m_steps, lr=lr)
    data = [(given[i], target[i]) for i in range(len(given))]
    return optimize(
        data,
        leaf.estimator(),
        max_its=max_its,
        delta=delta,
        reuse_estep_ll=reuse_estep_ll,
        out=None,
        rng=np.random.RandomState(seed),
    )


def cycle_inconsistency(
    sampler: Any,
    given_value: np.ndarray,
    *,
    n_draws: int = 20,
    forward: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Self-supervised reliability signal for one ``given_value``: disagreement among ``n_draws``
    independent posterior samples of the target.

    A well-determined (sharp) posterior yields draws that agree closely (low inconsistency); an
    observation that collapsed several distinct targets onto this same ``given_value`` yields draws
    that disagree (high inconsistency) -- computable with no access to the true target. If ``forward``
    (the known A->B observation function) is supplied, agreement is checked in OBSERVATION space
    (the literal round trip target -> forward(target)) instead of raw target space.
    """
    draws = np.asarray([sampler.sample_given(given_value) for _ in range(n_draws)], dtype=np.float64)
    if forward is not None:
        draws = np.asarray([forward(d) for d in draws], dtype=np.float64)
    return float(np.mean(np.var(draws, axis=0)))


def posterior_mean_estimate(sampler: Any, given_value: np.ndarray, *, n_draws: int = 20) -> np.ndarray:
    """The point estimate a downstream consumer would actually use: the mean of ``n_draws`` posterior
    samples of the target given ``given_value``."""
    draws = np.asarray([sampler.sample_given(given_value) for _ in range(n_draws)], dtype=np.float64)
    return draws.mean(axis=0)


def selective_error(errors: Sequence[float], abstain_scores: Sequence[float], keep_frac: float) -> float:
    """Mean error among the ``keep_frac`` fraction of examples with the LOWEST ``abstain_scores`` --
    the examples a policy would actually answer (rather than escalate) at that coverage budget.
    Lower is better: a good abstention signal keeps the examples it can actually get right.
    """
    errors = np.asarray(errors, dtype=np.float64)
    abstain_scores = np.asarray(abstain_scores, dtype=np.float64)
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    n_keep = max(1, int(round(keep_frac * len(errors))))
    order = np.argsort(abstain_scores)
    return float(np.mean(errors[order[:n_keep]]))
