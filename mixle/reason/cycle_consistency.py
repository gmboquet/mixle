"""Cycle-consistency diagnostics for cross-modal calibration and abstention.

A forward transport's reported confidence can miss an observation function that
maps several latent states to the same observed value. Round-trip closure adds
a self-supervised check: draw independent posterior samples of the latent given
the observation, project them through the invariant content, and measure
self-agreement. Low self-agreement indicates a region where the transport
should abstain or escalate even if its marginal confidence is high.

The diagnostic uses :class:`~mixle.models.mixture_density.NeuralConditionalDensity`
and ``build_mdn`` fitted via :func:`mixle.inference.optimize`; it does not add a
new transport family.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference import optimize
from mixle.reason.cross_modal import CrossModalJoint
from mixle.stats.latent.mixture import MixtureDistribution


def _as_paired_batch(a: np.ndarray) -> np.ndarray:
    """Return scalar batches as ``(n, 1)`` and matrix batches unchanged."""
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
    """Fit ``p(target | given)`` via a mixture density network.

    ``given``/``target`` are ``(n, d)`` arrays of paired observations. ``delta``/``reuse_estep_ll``
    default to :func:`~mixle.inference.optimize`'s own early-stopping; pass ``delta=None,
    reuse_estep_ll=False`` for a harder, more multimodal target.
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
    forward: Callable[[np.ndarray], np.ndarray],
    scale: float | np.ndarray,
) -> float:
    """Return scaled A -> B -> A round-trip error for one observation.

    ``sampler`` draws B conditioned on the originating A value and ``forward``
    maps each B draw back into A space. ``scale`` supplies the positive,
    unit-aware residual scale for A. A constant but wrong forward map therefore
    receives a nonzero error instead of a falsely perfect dispersion score.
    """
    given = _validated_vector(given_value, "given_value")
    count = _positive_count(n_draws, "n_draws")
    if not callable(forward):
        raise TypeError("forward must map a target draw back into the originating observation space.")
    residual_scale = np.asarray(scale, dtype=np.float64)
    try:
        residual_scale = np.broadcast_to(residual_scale, given.shape)
    except ValueError as exc:
        raise ValueError(f"scale must be scalar or broadcast to shape {given.shape}.") from exc
    if not np.isfinite(residual_scale).all() or np.any(residual_scale <= 0.0):
        raise ValueError("scale must contain only finite positive values.")

    x_batch = np.repeat(given.reshape(1, -1), count, axis=0)
    draws = _validated_draws(sampler.sample_given_batch(x_batch), count)
    round_trip = np.asarray([_validated_vector(forward(draw), "forward(draw)") for draw in draws])
    if round_trip.shape != (count, len(given)):
        raise ValueError(
            f"forward must return one vector with shape {given.shape} per draw; got {round_trip.shape}."
        )
    standardized = (round_trip - given) / residual_scale
    return float(np.mean(np.square(standardized)))


def posterior_mean_estimate(sampler: Any, given_value: np.ndarray, *, n_draws: int = 20) -> np.ndarray:
    """Return the posterior-sample mean of the target given ``given_value``."""
    given = _validated_vector(given_value, "given_value")
    count = _positive_count(n_draws, "n_draws")
    x_batch = np.repeat(given.reshape(1, -1), count, axis=0)
    draws = _validated_draws(sampler.sample_given_batch(x_batch), count)
    return draws.mean(axis=0)


@dataclass(frozen=True)
class CycleKLReceipt:
    """Monte Carlo evidence for a probabilistic round-trip comparison."""

    source: str
    target: str
    raw_estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    sample_count: int
    nonnegative_estimate: float
    clipped_to_nonnegative: bool


def joint_cycle_consistency_receipt(
    joint: CrossModalJoint,
    source: str,
    target: str,
    *,
    backward_joint: CrossModalJoint | None = None,
    n_round_trip: int = 300,
    n_kl_samples: int = 500,
    seed: int = 0,
) -> CycleKLReceipt:
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
    if source == target:
        raise ValueError("source and target must name distinct modalities.")
    if source not in joint.names or target not in joint.names:
        raise ValueError(f"source and target must be present in joint.names={joint.names!r}.")
    backward = joint if backward_joint is None else backward_joint
    if backward.names != joint.names:
        raise ValueError("backward_joint must have the same ordered modality schema as joint.")
    round_trip_count = _positive_count(n_round_trip, "n_round_trip")
    kl_count = _positive_count(n_kl_samples, "n_kl_samples")
    if kl_count < 2:
        raise ValueError("n_kl_samples must be at least two to estimate Monte Carlo uncertainty.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    rng = np.random.RandomState(int(seed))

    true_marginal = joint.infer({}, [source])
    forward_sampler = true_marginal.sampler(seed=int(rng.randint(0, 2**31 - 1)))

    round_trip_components = []
    round_trip_weights = []
    for _ in range(round_trip_count):
        a_value = forward_sampler.sample()[0]
        post_target = joint.infer({source: a_value}, [target])
        b_sampler = post_target.sampler(seed=int(rng.randint(0, 2**31 - 1)))
        b_value = b_sampler.sample()[0]
        post_source = backward.infer({target: b_value}, [source])
        for component, weight in zip(post_source.components, post_source.w):
            round_trip_components.append(component)
            round_trip_weights.append(weight / round_trip_count)

    round_trip = MixtureDistribution(round_trip_components, w=np.asarray(round_trip_weights, dtype=np.float64))

    kl_sampler = round_trip.sampler(seed=int(rng.randint(0, 2**31 - 1)))
    kl_terms = [
        round_trip.log_density(x) - true_marginal.log_density(x)
        for x in (kl_sampler.sample() for _ in range(kl_count))
    ]
    kl_values = np.asarray(kl_terms, dtype=np.float64)
    if kl_values.shape != (kl_count,) or not np.isfinite(kl_values).all():
        raise ValueError("round-trip KL evaluation produced non-finite log-density ratios.")
    raw = float(np.mean(kl_values))
    standard_error = float(np.std(kl_values, ddof=1) / np.sqrt(kl_count))
    margin = 1.96 * standard_error
    return CycleKLReceipt(
        source=source,
        target=target,
        raw_estimate=raw,
        standard_error=standard_error,
        confidence_interval=(raw - margin, raw + margin),
        sample_count=kl_count,
        nonnegative_estimate=max(raw, 0.0),
        clipped_to_nonnegative=raw < 0.0,
    )


def selective_error(errors: Sequence[float], abstain_scores: Sequence[float], keep_frac: float) -> float:
    """Return mean error on examples kept by the lowest abstention scores.

    Lower is better: a useful abstention signal keeps examples the policy can
    answer and escalates examples with higher expected error.
    """
    errors = np.asarray(errors, dtype=np.float64)
    abstain_scores = np.asarray(abstain_scores, dtype=np.float64)
    if errors.ndim != 1 or abstain_scores.ndim != 1 or not len(errors) or len(errors) != len(abstain_scores):
        raise ValueError("errors and abstain_scores must be non-empty aligned one-dimensional arrays.")
    if not np.isfinite(errors).all() or not np.isfinite(abstain_scores).all():
        raise ValueError("errors and abstain_scores must contain only finite values.")
    if not np.isfinite(keep_frac) or not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    n_keep = max(1, int(np.ceil(keep_frac * len(errors))))
    order = np.argsort(abstain_scores, kind="stable")
    return float(np.mean(errors[order[:n_keep]]))


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validated_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return vector


def _validated_draws(value: Any, expected_rows: int) -> np.ndarray:
    draws = np.asarray(value, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] != expected_rows or draws.shape[1] <= 0 or not np.isfinite(draws).all():
        raise ValueError(f"sampler must return a finite two-dimensional table with {expected_rows} rows.")
    return draws
