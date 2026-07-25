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

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle import __version__
from mixle.reason.cross_modal import CrossModalJoint
from mixle.stats.latent.mixture import MixtureDistribution

__all__ = ["HopExecutionReceipt", "InferenceHop", "ProgramReceipt", "ProgramPosterior", "run_inference_program"]

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
    carry: Mapping[str, str] = field(default_factory=dict)
    extra_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = tuple(self.target)
        if not target or any(not isinstance(name, str) or not name.strip() for name in target):
            raise ValueError("hop target must contain non-empty modality names.")
        if len(set(target)) != len(target):
            raise ValueError("hop target modalities must be unique.")
        carry = dict(self.carry)
        extra = dict(self.extra_evidence)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in carry.items()):
            raise TypeError("carry must map modality-name strings to modality-name strings.")
        if any(not isinstance(key, str) or not key.strip() for key in extra):
            raise ValueError("extra_evidence keys must be non-empty modality names.")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "carry", MappingProxyType(carry))
        object.__setattr__(self, "extra_evidence", MappingProxyType(extra))


@dataclass(frozen=True)
class HopExecutionReceipt:
    """Content-bound execution statistics for one hop."""

    hop_index: int
    joint_digest: str
    target: tuple[str, ...]
    carry: tuple[tuple[str, str], ...]
    extra_evidence_digest: str
    input_particles: int
    output_particles: int
    posterior_components: int
    output_mass: float
    sampling_failures: int


@dataclass(frozen=True)
class ProgramReceipt:
    """Content-addressed identity and execution record for an inference chain."""

    schema_version: str
    receipt_digest: str
    program_digest: str
    evidence_digest: str
    propagation: str
    seed: int
    n_hops: int
    hop_targets: tuple[tuple[str, ...], ...]
    n_particles: int
    software_version: str
    hops: tuple[HopExecutionReceipt, ...]


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
    sample_count = _positive_count(n_samples, "n_samples")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    evidence = dict(evidence)
    _validate_program(evidence, hops)
    rng = RandomState(int(seed))
    evidence_digest = _digest(evidence)
    program_identity = {
        "propagation": propagation,
        "n_samples": sample_count,
        "seed": int(seed),
        "hops": [
            {
                "joint": _joint_identity(hop.joint),
                "target": hop.target,
                "carry": dict(hop.carry),
                "extra_evidence": dict(hop.extra_evidence),
            }
            for hop in hops
        ],
    }
    program_digest = _digest(program_identity)

    # particles: list of (carried_values_dict keyed by the PRODUCING hop's own target names, weight)
    particles: list[tuple[dict[str, Any], float]] = [({}, 1.0)]
    final_mixture: MixtureDistribution | None = None
    hop_targets: list[tuple[str, ...]] = []
    hop_receipts: list[HopExecutionReceipt] = []

    for hop_idx, hop in enumerate(hops):
        is_last = hop_idx == len(hops) - 1
        hop_targets.append(hop.target)
        next_components: list[Any] = []
        next_weights: list[float] = []
        next_particles: list[tuple[dict[str, Any], float]] = []
        posterior_components = 0
        output_mass = 0.0

        for carried, w in particles:
            if not np.isfinite(w) or w < 0.0:
                raise ValueError(f"hop {hop_idx} received invalid particle mass {w!r}.")
            obs = dict(hop.extra_evidence)
            if hop_idx == 0:
                obs.update(evidence)
            else:
                for prior_name, this_name in hop.carry.items():
                    obs[this_name] = carried[prior_name]
            post = hop.joint.infer(obs, list(hop.target))
            post_weights = np.asarray(post.w, dtype=np.float64)
            if (
                post_weights.shape != (len(post.components),)
                or not len(post_weights)
                or not np.isfinite(post_weights).all()
                or np.any(post_weights < 0.0)
                or not np.isclose(float(post_weights.sum()), 1.0, atol=1e-10)
            ):
                raise ValueError(f"hop {hop_idx} produced invalid posterior component mass.")
            posterior_components += len(post.components)

            if is_last:
                for comp, cw in zip(post.components, post_weights):
                    next_components.append(comp)
                    next_weights.append(w * float(cw))
                continue

            if propagation == "moment":
                point = {name: _leaf_point_estimate(post.components, post.w, j) for j, name in enumerate(hop.target)}
                next_particles.append((point, w))
            else:  # "sampled"
                if hop_idx == 0:
                    draws = post.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample(sample_count)
                    draws = _validated_draws(draws, sample_count, len(hop.target), hop_idx)
                    per_weight = w / sample_count
                    for draw in draws:
                        next_particles.append((dict(zip(hop.target, draw, strict=True)), per_weight))
                else:
                    draw = post.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample()
                    draw = _validated_draw(draw, len(hop.target), hop_idx)
                    next_particles.append((dict(zip(hop.target, draw, strict=True)), w))

        if is_last:
            weights = np.asarray(next_weights, dtype=np.float64)
            total = float(weights.sum())
            if not next_components or weights.shape != (len(next_components),):
                raise RuntimeError(f"final hop {hop_idx} did not produce an aligned posterior mixture.")
            if not np.isfinite(weights).all() or np.any(weights < 0.0) or not np.isfinite(total) or total <= 0.0:
                raise ValueError(f"final hop {hop_idx} produced invalid or zero posterior mass.")
            final_mixture = MixtureDistribution(next_components, w=np.asarray(next_weights, dtype=np.float64) / total)
            output_mass = total
        else:
            if not next_particles:
                raise RuntimeError(f"hop {hop_idx} produced no particles for the next hop.")
            output_mass = float(sum(weight for _, weight in next_particles))
            if not np.isfinite(output_mass) or output_mass <= 0.0:
                raise ValueError(f"hop {hop_idx} produced invalid or zero particle mass.")
            particles = next_particles
        hop_receipts.append(
            HopExecutionReceipt(
                hop_index=hop_idx,
                joint_digest=_digest(_joint_identity(hop.joint)),
                target=hop.target,
                carry=tuple(sorted(hop.carry.items())),
                extra_evidence_digest=_digest(dict(hop.extra_evidence)),
                input_particles=1 if hop_idx == 0 else hop_receipts[-1].output_particles,
                output_particles=len(next_components) if is_last else len(next_particles),
                posterior_components=posterior_components,
                output_mass=output_mass,
                sampling_failures=0,
            )
        )

    if final_mixture is None:
        raise RuntimeError("inference program ended without a final posterior.")
    receipt_fields = {
        "schema_version": "1.0.0",
        "program_digest": program_digest,
        "evidence_digest": evidence_digest,
        "propagation": propagation,
        "seed": int(seed),
        "n_hops": len(hops),
        "hop_targets": tuple(hop_targets),
        "n_particles": sample_count if propagation == "sampled" and len(hops) > 1 else 1,
        "software_version": __version__,
        "hops": tuple(hop_receipts),
    }
    receipt = ProgramReceipt(
        receipt_digest=_digest(receipt_fields),
        **receipt_fields,
    )
    return ProgramPosterior(mixture=final_mixture, target=hops[-1].target, receipt=receipt)


def _validate_program(evidence: Mapping[str, Any], hops: Sequence[InferenceHop]) -> None:
    if any(not isinstance(key, str) or not key.strip() for key in evidence):
        raise ValueError("external evidence keys must be non-empty modality names.")
    first = hops[0]
    if first.carry:
        raise ValueError("the first hop conditions on external evidence and cannot carry prior fields.")
    collision = set(evidence) & set(first.extra_evidence)
    if collision:
        raise ValueError(f"first-hop external and extra evidence collide on {sorted(collision)!r}.")
    _validate_observation_fields(
        0,
        first,
        set(evidence) | set(first.extra_evidence),
        "external/extra evidence",
    )
    previous_target = set(first.target)
    for index, hop in enumerate(hops[1:], start=1):
        if set(hop.carry) != previous_target:
            missing = previous_target - set(hop.carry)
            unknown = set(hop.carry) - previous_target
            raise ValueError(f"hop {index} carry must consume every prior target exactly; missing={missing}, unknown={unknown}.")
        destinations = list(hop.carry.values())
        if len(set(destinations)) != len(destinations):
            raise ValueError(f"hop {index} carry destinations must be unique.")
        collision = set(destinations) & set(hop.extra_evidence)
        if collision:
            raise ValueError(f"hop {index} carried and extra evidence collide on {sorted(collision)!r}.")
        _validate_observation_fields(index, hop, set(destinations) | set(hop.extra_evidence), "carried/extra evidence")
        previous_target = set(hop.target)


def _validate_observation_fields(index: int, hop: InferenceHop, observed: set[str], label: str) -> None:
    known = set(hop.joint.names)
    target = set(hop.target)
    if not target <= known:
        raise ValueError(f"hop {index} target contains unknown modalities {target - known}.")
    if not observed <= known:
        raise ValueError(f"hop {index} {label} contains unknown modalities {observed - known}.")
    collision = observed & target
    if collision:
        raise ValueError(f"hop {index} observes and targets the same modalities {sorted(collision)!r}.")


def _validated_draws(value: Any, rows: int, width: int, hop_index: int) -> list[tuple[Any, ...]]:
    draws = list(value)
    if len(draws) != rows:
        raise ValueError(f"hop {hop_index} sampler returned {len(draws)} draws; expected {rows}.")
    return [_validated_draw(draw, width, hop_index) for draw in draws]


def _validated_draw(value: Any, width: int, hop_index: int) -> tuple[Any, ...]:
    try:
        draw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"hop {hop_index} sampler draw is not a field sequence.") from exc
    if len(draw) != width:
        raise ValueError(f"hop {hop_index} sampler draw has {len(draw)} fields; expected {width}.")
    return draw


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _joint_identity(joint: CrossModalJoint) -> dict[str, Any]:
    return {
        "type": f"{type(joint).__module__}.{type(joint).__qualname__}",
        "names": joint.names,
        "weights": np.asarray(joint.joint.w, dtype=float),
        "components": [
            [_object_identity(distribution) for distribution in component.dists]
            for component in joint.joint.components
        ],
    }


def _object_identity(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": {key: _object_identity(item) for key, item in sorted(vars(value).items()) if not callable(item)},
        }
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"$array": value.tolist(), "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        entries = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True, separators=(",", ":")))
        return {"$mapping": entries}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if dataclass_fields := getattr(value, "__dataclass_fields__", None):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "$fields": {name: _canonical(getattr(value, name)) for name in sorted(dataclass_fields)},
        }
    raise TypeError(f"cannot content-address value of type {type(value).__name__}.")
