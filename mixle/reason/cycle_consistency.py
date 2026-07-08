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
from mixle.reason.cross_modal import CrossModalJoint
from mixle.stats.latent.mixture import MixtureDistribution


def _as_paired_batch(a: np.ndarray) -> np.ndarray:
    """``(n,)`` -> ``(n, 1)`` (n scalar observations, NOT one n-dimensional row -- unlike
    ``np.atleast_2d``, which assumes the opposite and would silently misinterpret a batch of scalar
    samples as a single high-dimensional one); ``(n, d)`` passes through unchanged."""
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def fit_cycle_transport(
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
    import torch

    from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

    given = _as_paired_batch(given)
    target = _as_paired_batch(target)
    if len(given) != len(target):
        raise ValueError(
            f"given and target must have the same number of paired observations, got {len(given)} vs {len(target)}"
        )
    torch.manual_seed(seed)  # optimize()'s rng seeds data order only; module init needs its own seed
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
    x_batch = np.repeat(np.atleast_2d(np.asarray(given_value, dtype=np.float64)), n_draws, axis=0)
    draws = np.asarray(sampler.sample_given_batch(x_batch), dtype=np.float64)
    if forward is not None:
        draws = np.asarray([forward(d) for d in draws], dtype=np.float64)
    return float(np.mean(np.var(draws, axis=0)))


def posterior_mean_estimate(sampler: Any, given_value: np.ndarray, *, n_draws: int = 20) -> np.ndarray:
    """The point estimate a downstream consumer would actually use: the mean of ``n_draws`` posterior
    samples of the target given ``given_value``."""
    x_batch = np.repeat(np.atleast_2d(np.asarray(given_value, dtype=np.float64)), n_draws, axis=0)
    draws = np.asarray(sampler.sample_given_batch(x_batch), dtype=np.float64)
    return draws.mean(axis=0)


def joint_cycle_consistency_receipt(
    joint: CrossModalJoint,
    source: str,
    target: str,
    *,
    backward_joint: CrossModalJoint | None = None,
    n_round_trip: int = 300,
    n_kl_samples: int = 500,
    seed: int = 0,
) -> float:
    """Cross-modal generalization (workstream L2) of this module's round-trip closure signal.

    ``cycle_inconsistency`` above measures round-trip closure (A -> B -> A) for a NEURAL transport,
    where the true target is unknown at serving time and self-AGREEMENT among repeated draws is the
    only available proxy. A :class:`~mixle.reason.cross_modal.CrossModalJoint` is a typed grammar
    object, not an opaque transport: its true marginal ``p(source)`` is available in closed form
    (:meth:`CrossModalJoint.infer` with no observations), so the round-trip receipt here compares the
    round-trip estimate DIRECTLY against that true marginal, rather than against itself.

    Two ways to arrive at a belief about ``source`` through the joint: (1) directly, its own marginal
    ``p(source)``; (2) via a round trip, ``p(source) -> infer p(target | source) -> infer p(source |
    target) back``, averaged over many draws into one aggregate "round-trip" belief. This receipt is a
    Monte-Carlo KL-divergence estimate between (2) and (1); a well-specified joint recovers its own
    marginal on a round trip (the receipt is ~0 up to Monte-Carlo noise), while a deliberately
    mis-specified backward projection (``backward_joint`` -- e.g. a joint whose ``target``-given-regime
    distributions have been shuffled relative to ``joint``'s, standing in for a broken/incompatible
    A<-B projection) breaks that identity and the receipt becomes clearly, measurably elevated.
    """
    backward = joint if backward_joint is None else backward_joint
    rng = np.random.RandomState(seed)

    true_marginal = joint.infer({}, [source])
    forward_sampler = true_marginal.sampler(seed=int(rng.randint(0, 2**31 - 1)))

    round_trip_components = []
    round_trip_weights = []
    for _ in range(n_round_trip):
        a_value = forward_sampler.sample()[0]
        post_target = joint.infer({source: a_value}, [target])
        b_sampler = post_target.sampler(seed=int(rng.randint(0, 2**31 - 1)))
        b_value = b_sampler.sample()[0]
        post_source = backward.infer({target: b_value}, [source])
        for component, weight in zip(post_source.components, post_source.w):
            round_trip_components.append(component)
            round_trip_weights.append(weight / n_round_trip)

    round_trip = MixtureDistribution(round_trip_components, w=np.asarray(round_trip_weights, dtype=np.float64))

    kl_sampler = round_trip.sampler(seed=int(rng.randint(0, 2**31 - 1)))
    kl_terms = [
        round_trip.log_density(x) - true_marginal.log_density(x)
        for x in (kl_sampler.sample() for _ in range(n_kl_samples))
    ]
    # KL divergence is non-negative in theory; clamp away small Monte-Carlo undershoot at (near-)zero.
    return float(max(float(np.mean(kl_terms)), 0.0))


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
