"""Multi-hop inference programs over composed M0/L2 conditioning queries (roadmap M5).

L2's :class:`~mixle.reason.cross_modal.CrossModalJoint` answers one conditioning query exactly:
condition on any subset of a joint's named modalities, infer any other subset, all within a SINGLE
shared latent regime. Real cross-modal reasoning needs to chain THROUGH modalities that are not all
tied by one joint -- the card's own example is (image field -> shared latent -> predicted field ->
text field), where the image<->latent relationship and the latent<->text relationship are two
separately-fit joints. This module adds exactly that composition and nothing else: a small, explicit,
LINEAR chain of :meth:`~mixle.reason.cross_modal.CrossModalJoint.infer` calls (a path DAG -- no
free-form planning, no branching/merging in v1; see ``notes/designs/M5.md`` part (a)).

The one real design problem a chain introduces that a single hop does not have: hop *i*'s posterior
over the field hop *i+1* needs to condition on is a full DISTRIBUTION, not a single value, but
``CrossModalJoint.infer`` demands a concrete observed value. Two receipted ways to bridge that gap are
implemented (``notes/designs/M5.md`` part (b)):

* ``propagation="sampled"`` (default) -- Monte-Carlo particles carried hop to hop, exactly the
  "one particle, one draw, carry the weight" pattern already proven out by
  :func:`mixle.reason.cycle_consistency.joint_cycle_consistency_receipt`'s round trip. Unbiased;
  converges to the exact marginal as ``n_samples`` grows.
* ``propagation="moment"`` -- collapse each non-final hop's posterior to its own point estimate
  (analytic mean for a Gaussian-like leaf, arg-max of the component-weighted ``pmap`` for a
  categorical leaf) and carry that one point forward. Cheap (one ``infer`` per hop instead of
  ``n_samples``), and HONESTLY the wrong choice whenever a downstream hop is sensitive to the
  intermediate's uncertainty, not just its central tendency -- see
  ``mixle/tests/inference_program_test.py::InferenceProgramTwoHopVsNaiveTest`` for a fixture where
  this mode measurably diverges from the closed-form answer that ``"sampled"`` recovers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.reason.cross_modal import CrossModalJoint
from mixle.stats.latent.mixture import MixtureDistribution

__all__ = ["InferenceHop", "ProgramReceipt", "ProgramPosterior", "run_inference_program"]

_PROPAGATIONS = ("sampled", "moment")


@dataclass(frozen=True)
class InferenceHop:
    """One ``CrossModalJoint.infer`` call in a chain.

    ``target`` is the tuple of modality names this hop infers. ``carry`` renames a value produced by
    the PREVIOUS hop's ``target`` into this hop's own joint's modality-name space (``{prior_target_name:
    this_hop_evidence_name}``) -- the two joints need not share a naming convention. Ignored (and must
    be empty) for the first hop in a program, which conditions on the program's external ``evidence``
    instead. ``extra_evidence`` is fixed evidence local to this hop (observed independently of anything
    carried down the chain).
    """

    joint: CrossModalJoint
    target: tuple[str, ...]
    carry: dict[str, str] = field(default_factory=dict)
    extra_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramReceipt:
    """What :func:`run_inference_program` actually did -- the M5 analogue of M0's ``ConditionReceipt``."""

    propagation: str
    n_hops: int
    hop_targets: list[tuple[str, ...]]
    n_particles: int


class ProgramPosterior:
    """A completed inference program's result: the same ``sample``/``log_density``/``mean``-shaped
    contract M0's own :class:`~mixle.inference.condition.Posterior` exposes, over the final hop's
    ``target`` fields, so a program's output composes with the same downstream code (e.g. the
    language<->belief bridge) that already consumes a single-hop posterior.
    """

    def __init__(self, mixture: MixtureDistribution, target: tuple[str, ...], receipt: ProgramReceipt) -> None:
        self.mixture = mixture
        self.target = target
        self.receipt = receipt

    def _pos(self, field_name: str) -> int:
        try:
            return self.target.index(field_name)
        except ValueError:
            raise KeyError(f"unknown field {field_name!r}; this program's final target is {self.target!r}") from None

    def sample(self, n: int = 1, *, seed: int | None = None) -> Any:
        return self.mixture.sampler(seed=seed).sample(n)

    def log_density(self, value: Any) -> float:
        return float(self.mixture.log_density(value))

    def density(self, value: Any) -> float:
        return float(self.mixture.density(value))

    def mean(self, field_name: str) -> Any:
        """Analytic (component-weighted) mean of one target field -- Gaussian-like leaves only."""
        j = self._pos(field_name)
        return _field_mean(self.mixture.components, self.mixture.w, j)


def _field_mean(components: Sequence[Any], w: np.ndarray, j: int) -> float:
    means = np.array([_leaf_mean(c.dists[j]) for c in components], dtype=np.float64)
    return float(np.sum(np.asarray(w, dtype=np.float64) * means))


def _leaf_mean(dist: Any) -> float:
    if hasattr(dist, "mu"):
        return float(dist.mu)
    mean_fn = getattr(dist, "mean", None)
    if callable(mean_fn):
        return float(mean_fn())
    raise NotImplementedError(f"no analytic mean available for {type(dist).__name__}")


def _leaf_point_estimate(components: Sequence[Any], w: np.ndarray, j: int) -> Any:
    """The point M5 carries forward under ``propagation='moment'`` for one target field: the analytic
    mean for a Gaussian-like leaf, or the arg-max of the component-weighted ``pmap`` for a categorical
    leaf (any other leaf type raises rather than guessing)."""
    first = components[0].dists[j]
    if hasattr(first, "mu"):
        return _field_mean(components, w, j)
    if hasattr(first, "pmap"):
        mixed: dict[Any, float] = {}
        for c, cw in zip(components, w):
            for key, p in c.dists[j].pmap.items():
                mixed[key] = mixed.get(key, 0.0) + float(cw) * float(p)
        return max(mixed, key=mixed.get)
    raise NotImplementedError(f"propagation='moment' has no point-estimate rule for leaf type {type(first).__name__}")


def run_inference_program(
    evidence: dict[str, Any],
    hops: Sequence[InferenceHop],
    *,
    propagation: str = "sampled",
    n_samples: int = 500,
    seed: int = 0,
) -> ProgramPosterior:
    """Run a linear chain of :class:`InferenceHop` conditioning queries, propagating uncertainty (or
    not -- see module docstring) between hops. ``evidence`` conditions the FIRST hop only; later hops
    condition on ``extra_evidence`` plus whatever their ``carry`` mapping pulls from the previous hop's
    posterior over ``target``.
    """
    if propagation not in _PROPAGATIONS:
        raise ValueError(f"propagation must be one of {_PROPAGATIONS}, got {propagation!r}")
    hops = list(hops)
    if not hops:
        raise ValueError("run_inference_program needs at least one hop")
    if hops[0].carry:
        raise ValueError("the first hop conditions on the program's external `evidence`; it cannot `carry`")
    rng = RandomState(seed)

    # particles: list of (carried_values_dict keyed by the PRODUCING hop's own target names, weight)
    particles: list[tuple[dict[str, Any], float]] = [({}, 1.0)]
    final_mixture: MixtureDistribution | None = None
    hop_targets: list[tuple[str, ...]] = []

    for hop_idx, hop in enumerate(hops):
        is_last = hop_idx == len(hops) - 1
        hop_targets.append(hop.target)
        next_components: list[Any] = []
        next_weights: list[float] = []
        next_particles: list[tuple[dict[str, Any], float]] = []

        for carried, w in particles:
            obs = dict(hop.extra_evidence)
            if hop_idx == 0:
                obs.update(evidence)
            else:
                for prior_name, this_name in hop.carry.items():
                    obs[this_name] = carried[prior_name]
            post = hop.joint.infer(obs, list(hop.target))

            if is_last:
                for comp, cw in zip(post.components, post.w):
                    next_components.append(comp)
                    next_weights.append(w * float(cw))
                continue

            if propagation == "moment":
                point = {name: _leaf_point_estimate(post.components, post.w, j) for j, name in enumerate(hop.target)}
                next_particles.append((point, w))
            else:  # "sampled"
                if hop_idx == 0:
                    draws = post.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample(n_samples)
                    per_weight = w / n_samples
                    for draw in draws:
                        next_particles.append((dict(zip(hop.target, draw)), per_weight))
                else:
                    draw = post.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample()
                    next_particles.append((dict(zip(hop.target, draw)), w))

        if is_last:
            total = float(sum(next_weights))
            final_mixture = MixtureDistribution(next_components, w=np.asarray(next_weights, dtype=np.float64) / total)
        else:
            particles = next_particles

    assert final_mixture is not None  # last hop always populates it
    receipt = ProgramReceipt(
        propagation=propagation,
        n_hops=len(hops),
        hop_targets=hop_targets,
        n_particles=len(particles) if propagation == "sampled" else 1,
    )
    return ProgramPosterior(mixture=final_mixture, target=hops[-1].target, receipt=receipt)
