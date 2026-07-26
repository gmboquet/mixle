"""Hierarchical Dirichlet process mixture (truncated) for grouped data,
adapted onto the mixle.stats base-class protocol.

Observations arrive in groups (a datum is a sequence of observations). All
groups share K global atoms; each group mixes them with its own weights. This
implements the finite "direct-assignment" truncation of the HDP
(Teh et al. 2006):

    beta             ~ Dirichlet(gamma/K, ..., gamma/K)    (global weights)
    pi_j | beta      ~ Dirichlet(alpha * beta)             (group j weights)
    z_ji | pi_j      ~ pi_j
    x_ji | z_ji = k  ~ components[k]

Estimation alternates:
  - E-step at point estimates: responsibilities phi_jik from the group's
    current weights and the atom densities,
  - posterior-mean update for each group's weights under
    Dirichlet(alpha*beta + expected_counts), deliberately using the mean
    rather than the boundary-degenerate MAP when alpha*beta_k < 1, together
    with the atoms' estimator updates,
  - global-weight update via the standard expected-table-count approximation
    m_jk = alpha*beta_k*(psi(alpha*beta_k + n_jk) - psi(alpha*beta_k)), with
    beta set to the Dirichlet(gamma/K + m_.k) posterior mean. Applying this
    table-count formula to fractional responsibility counts is a deterministic
    approximation, not an exact collapsed-HDP CAVI step.

``seq_local_elbo`` scores training groups with their fitted weights (this is
what the fit driver maximizes); ``seq_log_density`` scores a (possibly new)
group with the global weights beta, i.e. the expected weights of an unseen
group. For multi-observation new groups this is a beta plug-in score, not the
integrated finite-HDP predictive density obtained by integrating over a new
group row pi ~ Dirichlet(alpha*beta).

Group sizes are exogenous unless len_dist is supplied (used for sampling and
added to the per-group score). The length model uses the mixle.stats
NullDistribution/NullEstimator/NullAccumulator family.

This is a port of ``mixle.bstats.hdpm`` onto the ``mixle.stats`` protocol. The
object should be read as the finite direct-assignment approximation described
above, with posterior-mean rows and an expected-table global-row heuristic.
"""

import copy
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.inference.integrity import canonical_digest
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.bayes.dirichlet_process_mixture import (
    _compatible_component_encoder,
    _component_structure_fingerprint,
    _validated_component_sequence,
    _validated_observation_weight,
    _validated_observation_weights,
)
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.mixture_evidence import normalize_mixture_log_scores
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_SIMPLEX_RTOL = 1.0e-10
_SIMPLEX_ATOL = 1.0e-12
_GROUP_ID_PREFIX = "hdp:"


@dataclass(frozen=True)
class HDPGroup:
    """A grouped observation with a stable user-supplied identity."""

    group_id: Any
    values: Sequence[Any]


def _positive_scalar(value: Any, name: str) -> float:
    if np.ndim(value) != 0:
        raise ValueError("HDP %s must be a finite positive scalar." % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("HDP %s must be a finite positive scalar." % name) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("HDP %s must be a finite positive scalar." % name)
    return result


def _simplex_vector(value: Any, width: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("HDP %s must be numeric." % name) from exc
    if result.shape != (width,):
        raise ValueError("HDP %s must have exact shape (%d,)." % (name, width))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("HDP %s must be finite and non-negative." % name)
    if not np.isclose(
        float(result.sum()),
        1.0,
        rtol=_SIMPLEX_RTOL,
        atol=_SIMPLEX_ATOL,
    ):
        raise ValueError("HDP %s must sum to one." % name)
    return result.copy()


def _canonical_group_id(value: Any, *, content: bool = False) -> str:
    if isinstance(value, str) and value.startswith(_GROUP_ID_PREFIX):
        return value
    try:
        digest = canonical_digest(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("HDP group IDs and unkeyed group contents must have stable identity.") from exc
    kind = "content" if content else "user"
    return "%s%s:%s" % (_GROUP_ID_PREFIX, kind, digest)


def _group_identity_and_values(value: Any) -> tuple[str, list[Any]]:
    if isinstance(value, HDPGroup):
        group_id = _canonical_group_id(value.group_id)
        group_values = value.values
    else:
        group_values = value
        group_id = _canonical_group_id(value, content=True)
    if isinstance(group_values, (str, bytes)) or not isinstance(group_values, Sequence):
        raise TypeError("HDP group values must be a sequence.")
    return group_id, list(group_values)


def _validated_group_state(
    group_weights: Any,
    group_ids: Any,
    component_count: int,
    beta: np.ndarray,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if group_weights is None:
        if group_ids is not None and (
            isinstance(group_ids, (str, bytes))
            or not isinstance(group_ids, Sequence)
            or len(group_ids) != 0
        ):
            raise ValueError("HDP group_ids require group_weights.")
        return None, ()
    try:
        weights = np.asarray(group_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("HDP group_weights must be numeric.") from exc
    if weights.ndim != 2 or weights.shape[1] != component_count:
        raise ValueError(
            "HDP group_weights must have exact shape (groups, %d)." % component_count
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("HDP group_weights must be finite and non-negative.")
    if not np.allclose(
        weights.sum(axis=1),
        1.0,
        rtol=_SIMPLEX_RTOL,
        atol=_SIMPLEX_ATOL,
    ):
        raise ValueError("Every HDP group_weights row must sum to one.")
    if np.any(weights[:, beta == 0.0] != 0.0):
        raise ValueError("HDP groups cannot assign mass to globally impossible atoms.")
    if isinstance(group_ids, (str, bytes)) or not isinstance(group_ids, Sequence):
        raise TypeError("HDP fitted group_weights require a sequence of group_ids.")
    canonical_ids = tuple(_canonical_group_id(group_id) for group_id in group_ids)
    if len(canonical_ids) != weights.shape[0]:
        raise ValueError("HDP group_ids must contain exactly one ID per fitted weight row.")
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("HDP group_ids must be unique.")
    return weights.copy(), canonical_ids


def _weighted_scores(component_scores: Any, weights: np.ndarray) -> np.ndarray:
    scores = np.asarray(component_scores, dtype=np.float64).copy()
    positive = weights > 0.0
    scores[..., positive] += np.log(weights[positive])
    scores[..., ~positive] = -np.inf
    return scores


def _validated_encoded_groups(
    value: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, Any, Any]:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError("HDP encoded data must be a five-item tuple.")
    raw_ids, raw_lengths, raw_offsets, flat_enc, len_enc = value
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise TypeError("HDP encoded group IDs must be a sequence.")
    group_ids = tuple(_canonical_group_id(group_id) for group_id in raw_ids)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("HDP encoded group IDs must be unique.")
    try:
        lengths = np.asarray(raw_lengths)
        offsets = np.asarray(raw_offsets)
    except (TypeError, ValueError) as exc:
        raise ValueError("HDP encoded lengths and offsets must be numeric.") from exc
    if (
        lengths.ndim != 1
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths < 0)
    ):
        raise ValueError("HDP encoded group lengths must be non-negative integers.")
    if (
        offsets.ndim != 1
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets.shape != (len(lengths) + 1,)
    ):
        raise ValueError("HDP encoded offsets must have one integer boundary per group.")
    expected_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("HDP encoded offsets must exactly match group lengths.")
    if len(group_ids) != len(lengths):
        raise ValueError("HDP encoded group IDs and lengths must align.")
    return (
        group_ids,
        lengths.astype(np.int64, copy=True),
        offsets.astype(np.int64, copy=True),
        flat_enc,
        len_enc,
    )


def _validated_group_counts(value: Any, component_count: int) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise TypeError("HDP group counts must be a mapping keyed by stable group ID.")
    result: dict[str, np.ndarray] = {}
    for raw_group_id, raw_counts in value.items():
        group_id = _canonical_group_id(raw_group_id)
        if group_id in result:
            raise ValueError("HDP group count IDs must be unique.")
        try:
            counts = np.asarray(raw_counts, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("HDP group counts must be numeric vectors.") from exc
        if counts.shape != (component_count,):
            raise ValueError(
                "Every HDP group count must have exact shape (%d,)." % component_count
            )
        if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
            raise ValueError("HDP group counts must be finite and non-negative.")
        result[group_id] = counts.copy()
    return result


def _validated_hdp_statistics(
    value: Any,
    component_count: int,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray | None,
    float | None,
    str | None,
    tuple[Any, ...],
    Any,
]:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError("HDP sufficient statistics must be a six-item tuple.")
    group_counts = _validated_group_counts(value[0], component_count)
    prev_beta, prev_alpha, model_version = value[1:4]
    metadata_present = (
        prev_beta is not None,
        prev_alpha is not None,
        model_version is not None,
    )
    if any(metadata_present) and not all(metadata_present):
        raise ValueError("HDP previous-model metadata must be wholly present or wholly absent.")
    if all(metadata_present):
        checked_beta = _simplex_vector(
            prev_beta,
            component_count,
            "previous beta",
        )
        checked_alpha = _positive_scalar(prev_alpha, "previous alpha")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("HDP model-version metadata must be a non-empty string.")
        checked_version = model_version
    else:
        checked_beta = None
        checked_alpha = None
        checked_version = None
    if isinstance(value[4], (str, bytes)) or not isinstance(value[4], Sequence):
        raise TypeError("HDP atom statistics must be a sequence.")
    atom_stats = tuple(value[4])
    if len(atom_stats) != component_count:
        raise ValueError("HDP atom statistics must match the component count.")
    return (
        group_counts,
        checked_beta,
        checked_alpha,
        checked_version,
        atom_stats,
        value[5],
    )


class HierarchicalDirichletProcessMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Truncated hierarchical DP mixture over K shared atoms with global weights
    beta and (optionally) fitted per-group weights."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        beta: np.ndarray | list[float],
        alpha: float,
        gamma: float,
        group_weights: np.ndarray | None = None,
        name: str | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        group_ids: Sequence[Any] | None = None,
    ) -> None:
        """Create a finite hierarchical Dirichlet-process mixture approximation.

        Args:
            components: List of K shared atom distributions (each carrying its
                own prior).
            beta: Length-K global weight vector.
            alpha (float): Group-level concentration of Dirichlet(alpha*beta).
            gamma (float): Global concentration of Dirichlet(gamma/K).
            group_weights (Optional[np.ndarray]): (J, K) fitted weights of the
                training groups (used by seq_local_elbo); None scores all groups
                with beta.
            name (Optional[str]): Optional distribution name.
            len_dist (Optional): Distribution of group sizes; a NullDistribution
                (the default) treats sizes as exogenous.

        """
        checked_components = _validated_component_sequence(components, "components")
        component_count = len(checked_components)
        checked_beta = _simplex_vector(beta, component_count, "beta")
        checked_alpha = _positive_scalar(alpha, "alpha")
        checked_gamma = _positive_scalar(gamma, "gamma")
        _compatible_component_encoder(checked_components)
        checked_len_dist = len_dist if len_dist is not None else NullDistribution()
        if not isinstance(checked_len_dist, SequenceEncodableProbabilityDistribution):
            raise TypeError("HDP len_dist must be a probability distribution.")
        checked_group_weights, checked_group_ids = _validated_group_state(
            group_weights,
            group_ids,
            component_count,
            checked_beta,
        )
        fingerprint = _component_structure_fingerprint(checked_components)

        self.name = name
        self.components = checked_components
        self.num_components = component_count
        self.alpha = checked_alpha
        self.gamma = checked_gamma
        self.len_dist = checked_len_dist
        self.beta = checked_beta
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)
        self.group_weights = checked_group_weights
        self.group_ids = checked_group_ids
        self._group_index = {
            group_id: index for index, group_id in enumerate(self.group_ids)
        }
        self._structure_fingerprint = fingerprint
        self._length_structure_fingerprint = self._length_structure()

    def _length_structure(self) -> tuple[str, str, str]:
        """Return a serialization-safe fingerprint of the group-length model."""
        model_type = type(self.len_dist)
        encoder = self.len_dist.dist_to_encoder()
        encoder_type = type(encoder)
        return (
            "%s.%s" % (model_type.__module__, model_type.__qualname__),
            "%s.%s" % (encoder_type.__module__, encoder_type.__qualname__),
            str(encoder),
        )

    def __str__(self) -> str:
        cstr = ",".join(str(u) for u in self.components)
        return (
            "HierarchicalDirichletProcessMixtureDistribution([%s], %s, %s, %s, "
            "group_weights=%s, name=%s, len_dist=%s, group_ids=%s)"
        ) % (
            cstr,
            repr(self.beta.tolist()),
            repr(self.alpha),
            repr(self.gamma),
            "None" if self.group_weights is None else repr(self.group_weights.tolist()),
            repr(self.name),
            str(self.len_dist),
            repr(list(self.group_ids)),
        )

    def get_parameters(self) -> tuple[np.ndarray, float, float, list[Any]]:
        """Return beta, concentrations, and independent component snapshots."""
        return self.beta.copy(), self.alpha, self.gamma, copy.deepcopy(self.components)

    def _assert_structure(self) -> None:
        if len(self.components) != self.num_components:
            raise RuntimeError("HDP component structure changed after construction.")
        if _component_structure_fingerprint(self.components) != self._structure_fingerprint:
            raise RuntimeError("HDP component structure changed after construction.")
        if self._length_structure() != self._length_structure_fingerprint:
            raise RuntimeError("HDP group-length structure changed after construction.")
        checked_beta = _simplex_vector(self.beta, self.num_components, "beta")
        _positive_scalar(self.alpha, "alpha")
        _positive_scalar(self.gamma, "gamma")
        if not np.array_equal(self.log_beta, np.log(checked_beta, where=checked_beta > 0.0, out=np.full_like(checked_beta, -np.inf))):
            raise RuntimeError("HDP cached global log weights are inconsistent.")
        checked_weights, checked_ids = _validated_group_state(
            self.group_weights,
            self.group_ids,
            self.num_components,
            checked_beta,
        )
        if checked_weights is None:
            if self._group_index:
                raise RuntimeError("HDP fitted-group index is inconsistent.")
        elif (
            checked_ids != self.group_ids
            or self._group_index
            != {group_id: index for index, group_id in enumerate(checked_ids)}
        ):
            raise RuntimeError("HDP fitted-group index is inconsistent.")

    def set_parameters(self, params: tuple[np.ndarray, float, float, Sequence[Any]]) -> None:
        """Atomically replace beta, concentrations, and complete component snapshots."""
        if not isinstance(params, (tuple, list)) or len(params) != 4:
            raise ValueError("HDP parameters must be a four-item tuple.")
        self._assert_structure()
        beta, alpha, gamma, components = params
        checked_beta = _simplex_vector(beta, self.num_components, "beta")
        checked_alpha = _positive_scalar(alpha, "alpha")
        checked_gamma = _positive_scalar(gamma, "gamma")
        checked_components = _validated_component_sequence(components, "components")
        if len(checked_components) != self.num_components:
            raise ValueError("HDP replacement components must match the component count.")
        checked_components = copy.deepcopy(checked_components)
        if _component_structure_fingerprint(checked_components) != self._structure_fingerprint:
            raise ValueError("HDP replacement components changed model structure.")
        if self.group_weights is not None and np.any(
            self.group_weights[:, checked_beta == 0.0] != 0.0
        ):
            raise ValueError(
                "HDP beta update would make a fitted group's positive atom mass impossible."
            )

        self.beta = checked_beta
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)
        self.alpha = checked_alpha
        self.gamma = checked_gamma
        self.components = checked_components

    def _model_version(self) -> str:
        """Return the stable identity of every state value used by an E-step."""
        self._assert_structure()
        return canonical_digest(
            {
                "components": [str(component) for component in self.components],
                "beta": self.beta,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "group_ids": self.group_ids,
                "group_weights": self.group_weights,
                "len_dist": str(self.len_dist),
            }
        )

    def group_weight(self, group: Any) -> np.ndarray:
        """Return the fitted row for ``group``, or global beta for an unseen group."""
        self._assert_structure()
        group_id, _ = _group_identity_and_values(group)
        index = self._group_index.get(group_id)
        if index is None or self.group_weights is None:
            return self.beta.copy()
        return self.group_weights[index].copy()

    def _len_term(self, x: Any) -> float:
        if supports(self.len_dist, Neutral) or self.len_dist is None:
            return 0.0
        return self.len_dist.log_density(len(x))

    def _group_log_density(self, log_b: np.ndarray, weights: np.ndarray) -> float:
        """Sum over observations of log sum_k w_k p(x_i | theta_k)."""
        evidence = normalize_mixture_log_scores(
            _weighted_scores(log_b, weights)
        ).log_evidence
        if np.any(np.isneginf(evidence)):
            return -np.inf
        if np.any(np.isposinf(evidence)):
            return np.inf
        return float(evidence.sum())

    def density(self, x: Any) -> float:
        """Density of a group x; see log_density()."""
        return np.exp(self.log_density(x))

    def density_semantics(self):
        """Return density semantics for the expected-weight HDP mixture approximation."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.ESTIMATE  # plug-in with expected global weights (expected-table-count approx.)

    def log_density(self, x: Any) -> float:
        """Score a group with the global weights beta (expected weights of a
        new group)."""
        self._assert_structure()
        _, values = _group_identity_and_values(x)
        if len(values) == 0:
            return self._len_term(values)
        enc = self.components[0].dist_to_encoder().seq_encode(values)
        log_b = np.asarray([c.seq_log_density(enc) for c in self.components]).T
        return self._group_log_density(log_b, self.beta) + self._len_term(values)

    def seq_encode(self, x: Sequence[Sequence]) -> Any:
        """Encode groups into a flat component encoding with offsets.

        Args:
            x (Sequence[Sequence]): Iterable of groups (sequences of observations).

        Returns:
            Tuple ``(group_ids, lengths, offsets, flat_enc, len_enc)`` consumed
            by vectorized ``seq_*`` methods.

        """
        self._assert_structure()
        unpacked = [_group_identity_and_values(group) for group in x]
        group_ids = tuple(group_id for group_id, _ in unpacked)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(
                "HDP group identities must be unique; wrap duplicate-content groups in HDPGroup."
            )
        groups = [values for _, values in unpacked]
        lengths = np.asarray([len(group) for group in groups], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat: list[Any] = []
        for group in groups:
            flat.extend(group)
        flat_enc = self.components[0].dist_to_encoder().seq_encode(flat)

        if supports(self.len_dist, Neutral) or self.len_dist is None:
            len_enc = None
        else:
            len_enc = self.len_dist.dist_to_encoder().seq_encode(lengths)

        return group_ids, lengths, offsets, flat_enc, len_enc

    def _emission_log_densities(self, flat_enc: Any) -> np.ndarray:
        self._assert_structure()
        result = np.asarray(
            [c.seq_log_density(flat_enc) for c in self.components],
            dtype=np.float64,
        ).T
        if result.ndim != 2 or result.shape[1] != self.num_components:
            raise ValueError("HDP atom score matrix has invalid component geometry.")
        return result

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized log_density() at sequence-encoded input x (each group
        scored with the global weights beta)."""
        self._assert_structure()
        group_ids, lengths, offsets, flat_enc, len_enc = _validated_encoded_groups(x)
        log_b_all = self._emission_log_densities(flat_enc)

        rv = np.zeros(len(lengths))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            rv[j] = self._group_log_density(
                log_b_all[offsets[j] : offsets[j + 1], :],
                self.beta,
            )

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def seq_local_elbo(self, x: Any) -> np.ndarray:
        """Per-group data term of the penalized objective: training groups are
        scored with their own fitted weights. Falls back to beta when the group
        count does not match the fit."""
        self._assert_structure()
        group_ids, lengths, offsets, flat_enc, len_enc = _validated_encoded_groups(x)

        log_b_all = self._emission_log_densities(flat_enc)

        rv = np.zeros(len(lengths))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            group_index = self._group_index.get(group_ids[j])
            weights = (
                self.beta
                if group_index is None or self.group_weights is None
                else self.group_weights[group_index]
            )
            rv[j] = self._group_log_density(
                log_b_all[offsets[j] : offsets[j + 1], :],
                weights,
            )

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def group_posteriors(self, x: Sequence[Sequence]) -> np.ndarray:
        """Posterior atom-usage (mean responsibility) per group.

        Groups are scored with their fitted weights when available (else with the
        global weights beta).
        """
        group_ids, lengths, offsets, flat_enc, len_enc = self.seq_encode(x)
        log_b_all = self._emission_log_densities(flat_enc)

        rv = np.zeros((len(lengths), self.num_components))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            group_index = self._group_index.get(group_ids[j])
            weights = (
                self.beta
                if group_index is None or self.group_weights is None
                else self.group_weights[group_index]
            )
            normalized = normalize_mixture_log_scores(
                _weighted_scores(
                    log_b_all[offsets[j] : offsets[j + 1], :],
                    weights,
                )
            )
            if np.any(normalized.impossible):
                rv[j, :] = 0.0
            else:
                rv[j, :] = normalized.responsibilities.mean(axis=0)
        return rv

    def sampler(self, seed: int | None = None) -> "HierarchicalDirichletProcessMixtureSampler":
        """Create a HierarchicalDirichletProcessMixtureSampler for this distribution."""
        return HierarchicalDirichletProcessMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HierarchicalDirichletProcessMixtureEstimator":
        """Create a HierarchicalDirichletProcessMixtureEstimator from this
        distribution's components, concentrations, and length estimator."""
        if pseudo_count is not None:
            raise ValueError("HDP pseudo-count regularization is not implemented.")
        len_est = NullEstimator() if supports(self.len_dist, Neutral) else self.len_dist.estimator()
        return HierarchicalDirichletProcessMixtureEstimator(
            [u.estimator() for u in self.components],
            gamma=self.gamma,
            alpha=self.alpha,
            name=self.name,
            len_estimator=len_est,
        )

    def dist_to_encoder(self) -> "HierarchicalDirichletProcessMixtureDataEncoder":
        """Returns a HierarchicalDirichletProcessMixtureDataEncoder for this distribution."""
        self._assert_structure()
        comp_encoder = self.components[0].dist_to_encoder()
        len_encoder = None if supports(self.len_dist, Neutral) else self.len_dist.dist_to_encoder()
        return HierarchicalDirichletProcessMixtureDataEncoder(comp_encoder, len_encoder)


class HierarchicalDirichletProcessMixtureSampler(DistributionSampler):
    """Draws groups from a HierarchicalDirichletProcessMixtureDistribution
    (per-group weights drawn from Dirichlet(alpha*beta))."""

    def __init__(self, dist: HierarchicalDirichletProcessMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for the finite HDP-mixture approximation."""
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [u.sampler(seed=rng.randint(0, maxrandint)) for u in dist.components]
        if supports(dist.len_dist, Neutral) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, maxrandint))

    def sample_group(self, n: int | None = None) -> list[Any]:
        """Draw a single group of n observations.

        Group weights pi ~ Dirichlet(alpha*beta) are drawn once for the group,
        then each observation draws an atom from pi.
        """
        if n is None:
            if self.len_sampler is None:
                raise ValueError("HDP sampler requires a len_dist (or explicit n) to sample groups.")
            n = self.len_sampler.sample()
        if isinstance(n, (bool, np.bool_)):
            raise TypeError("HDP group size must be a non-negative integer.")
        try:
            n = operator.index(n)
        except TypeError as exc:
            raise TypeError("HDP group size must be a non-negative integer.") from exc
        if n < 0:
            raise ValueError("HDP group size must be non-negative.")
        self.dist._assert_structure()
        active = self.dist.beta > 0.0
        pi = np.zeros(self.dist.num_components)
        if active.sum() == 1:
            pi[active] = 1.0
        else:
            pi[active] = self.rng.dirichlet(self.dist.alpha * self.dist.beta[active])
        states = self.rng.choice(self.dist.num_components, size=n, p=pi)
        return [self.comp_samplers[k].sample() for k in states]

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw size groups (a single group when size is None)."""
        if size is None:
            return self.sample_group()
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("HDP sample size must be a non-negative integer.")
        try:
            size = operator.index(size)
        except TypeError as exc:
            raise TypeError("HDP sample size must be a non-negative integer.") from exc
        if size < 0:
            raise ValueError("HDP sample size must be non-negative.")
        return [self.sample_group() for _ in range(size)]


class HierarchicalDirichletProcessMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates HDP mixture sufficient statistics: per-group expected atom
    counts keyed by stable identity plus each atom's weighted statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for HDP-mixture sufficient statistics."""
        self.accumulators = list(accumulators)
        if not self.accumulators:
            raise ValueError("HDP accumulators require at least one atom.")
        self.num_components = len(accumulators)
        self.name = name
        self.keys = keys
        self.group_counts: dict[str, np.ndarray] = {}
        self.prev_beta: np.ndarray | None = None
        self.prev_alpha: float | None = None
        self.model_version: str | None = None
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize with random Dirichlet assignments for group x."""
        checked_weight = _validated_observation_weight(weight)
        if not isinstance(rng, RandomState):
            raise TypeError("HDP initialization requires numpy.random.RandomState.")
        group_id, values = _group_identity_and_values(x)
        if group_id in self.group_counts:
            raise ValueError(
                "Duplicate HDP group identity; use distinct HDPGroup IDs for duplicate content."
            )
        counts = np.zeros(self.num_components)
        new_accumulators = copy.deepcopy(self.accumulators)
        for value in values:
            p = rng.dirichlet(np.ones(self.num_components))
            counts += p * checked_weight
            for k in range(self.num_components):
                new_accumulators[k].initialize(value, p[k] * checked_weight, rng)

        new_len_accumulator = copy.deepcopy(self.len_accumulator)
        if not supports(self.len_accumulator, Neutral):
            new_len_accumulator.update(len(values), checked_weight, None)
        self.accumulators = new_accumulators
        self.len_accumulator = new_len_accumulator
        self.group_counts[group_id] = counts

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialize() with random Dirichlet assignments."""
        if not isinstance(rng, RandomState):
            raise TypeError("HDP initialization requires numpy.random.RandomState.")
        group_ids, lengths, offsets, flat_enc, len_enc = _validated_encoded_groups(x)
        duplicate = set(group_ids).intersection(self.group_counts)
        if duplicate:
            raise ValueError("Duplicate HDP group identity during initialization.")
        checked_weights = _validated_observation_weights(weights, len(lengths))
        tot = int(lengths.sum())

        phi = rng.dirichlet(np.ones(self.num_components), size=tot)
        seq_w = np.repeat(checked_weights, lengths)
        new_group_counts: dict[str, np.ndarray] = {}

        for j in range(len(lengths)):
            sl = slice(offsets[j], offsets[j + 1])
            new_group_counts[group_ids[j]] = (
                np.dot(phi[sl, :].T, np.repeat(checked_weights[j], lengths[j]))
                if lengths[j] > 0
                else np.zeros(self.num_components)
            )

        new_accumulators = copy.deepcopy(self.accumulators)
        for k in range(self.num_components):
            new_accumulators[k].seq_initialize(flat_enc, phi[:, k] * seq_w, rng)

        new_len_accumulator = copy.deepcopy(self.len_accumulator)
        if len_enc is not None and not supports(self.len_accumulator, Neutral):
            new_len_accumulator.seq_initialize(len_enc, checked_weights, rng)
        self.accumulators = new_accumulators
        self.len_accumulator = new_len_accumulator
        self.group_counts.update(new_group_counts)

    def update(self, x: Any, weight: float, estimate: HierarchicalDirichletProcessMixtureDistribution) -> None:
        """Accumulate the E-step statistics for one group (delegates to seq_update
        on a singleton encoding)."""
        enc = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc, np.asarray([weight]), estimate)

    def seq_update(
        self, x: Any, weights: np.ndarray, estimate: HierarchicalDirichletProcessMixtureDistribution
    ) -> None:
        """E-step on sequence-encoded data at the current point estimates.

        Computes responsibilities phi from each group's current weights (the
        fitted group weights, or beta for new/unmatched groups) and the atom
        densities, recording per-group expected counts and pushing phi-weighted
        updates into the atom accumulators. Also records the estimate's beta and
        alpha for the estimator's global-weight update.
        """
        if not isinstance(estimate, HierarchicalDirichletProcessMixtureDistribution):
            raise TypeError("HDP accumulation requires a matching model estimate.")
        estimate._assert_structure()
        if estimate.num_components != self.num_components:
            raise ValueError("HDP accumulator and estimate atom counts differ.")
        group_ids, lengths, offsets, flat_enc, len_enc = _validated_encoded_groups(x)
        duplicate = set(group_ids).intersection(self.group_counts)
        if duplicate:
            raise ValueError("Duplicate HDP group identity during accumulation.")
        checked_weights = _validated_observation_weights(weights, len(lengths))
        model_version = estimate._model_version()
        if self.model_version is not None and self.model_version != model_version:
            raise ValueError("Cannot combine HDP updates computed from different model versions.")

        log_b_all = estimate._emission_log_densities(flat_enc)
        if log_b_all.shape != (int(lengths.sum()), self.num_components):
            raise ValueError("HDP atom score matrix does not match encoded group geometry.")

        phi_all = np.zeros_like(log_b_all)
        new_group_counts: dict[str, np.ndarray] = {}
        for j in range(len(lengths)):
            sl = slice(offsets[j], offsets[j + 1])
            counts = np.zeros(self.num_components)
            if lengths[j] > 0:
                group_index = estimate._group_index.get(group_ids[j])
                group_weights = (
                    estimate.beta
                    if group_index is None or estimate.group_weights is None
                    else estimate.group_weights[group_index]
                )
                normalized = normalize_mixture_log_scores(
                    _weighted_scores(log_b_all[sl, :], group_weights)
                )
                weighted_phi = (
                    normalized.responsibilities * checked_weights[j]
                )
                phi_all[sl, :] = weighted_phi
                counts = weighted_phi.sum(axis=0)
            new_group_counts[group_ids[j]] = counts

        new_accumulators = copy.deepcopy(self.accumulators)
        for k in range(self.num_components):
            new_accumulators[k].seq_update(
                flat_enc,
                phi_all[:, k],
                estimate.components[k],
            )

        new_len_accumulator = copy.deepcopy(self.len_accumulator)
        if len_enc is not None and not supports(self.len_accumulator, Neutral):
            new_len_accumulator.seq_update(len_enc, checked_weights, None)
        self.accumulators = new_accumulators
        self.len_accumulator = new_len_accumulator
        self.group_counts.update(new_group_counts)
        self.prev_beta = estimate.beta.copy()
        self.prev_alpha = estimate.alpha
        self.model_version = model_version

    def combine(self, suff_stat: tuple) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Add another accumulator's sufficient-statistic value into this one."""
        (
            group_counts,
            prev_beta,
            prev_alpha,
            model_version,
            atom_stats,
            len_value,
        ) = _validated_hdp_statistics(suff_stat, self.num_components)
        duplicate = set(group_counts).intersection(self.group_counts)
        if duplicate:
            raise ValueError("Cannot merge duplicate HDP group identities.")
        if self.model_version is not None and model_version != self.model_version:
            raise ValueError("Cannot merge HDP statistics from different model versions.")
        if self.model_version is None and self.group_counts and model_version is not None:
            raise ValueError("Cannot merge initialized and model-conditioned HDP statistics.")
        if self.model_version is not None and model_version is None and group_counts:
            raise ValueError("Cannot merge model-conditioned and initialized HDP statistics.")
        new_accumulators = copy.deepcopy(self.accumulators)
        for k in range(self.num_components):
            new_accumulators[k].combine(atom_stats[k])
        new_len_accumulator = copy.deepcopy(self.len_accumulator)
        if len_value is not None and not supports(self.len_accumulator, Neutral):
            new_len_accumulator.combine(len_value)
        self.accumulators = new_accumulators
        self.len_accumulator = new_len_accumulator
        self.group_counts.update(group_counts)
        if model_version is not None:
            self.prev_beta = prev_beta
            self.prev_alpha = prev_alpha
            self.model_version = model_version
        return self

    def scale(self, c: float) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Scale linear HDP mixture sufficient statistics while preserving metadata."""
        # Scale only the linear count statistics (per-group counts, atom accumulators, len accumulator).
        # ``prev_beta`` and ``prev_alpha`` are non-linear scalar/vector metadata carried for the
        # estimator's global-weight update; the inherited default would multiply and corrupt them.
        checked_scale = _validated_observation_weight(c)
        self.group_counts = {
            group_id: counts * checked_scale
            for group_id, counts in self.group_counts.items()
        }
        for u in self.accumulators:
            u.scale(checked_scale)
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.scale(checked_scale)
        return self

    def value(self) -> tuple:
        """Return keyed counts, model metadata, atom values, and length value."""
        len_val = None if supports(self.len_accumulator, Neutral) else self.len_accumulator.value()
        return (
            {group_id: counts.copy() for group_id, counts in self.group_counts.items()},
            None if self.prev_beta is None else self.prev_beta.copy(),
            self.prev_alpha,
            self.model_version,
            tuple(u.value() for u in self.accumulators),
            len_val,
        )

    def from_value(self, x: tuple) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Set the sufficient statistics from a value() tuple."""
        (
            group_counts,
            prev_beta,
            prev_alpha,
            model_version,
            atom_stats,
            len_value,
        ) = _validated_hdp_statistics(x, self.num_components)
        new_accumulators = copy.deepcopy(self.accumulators)
        for k in range(self.num_components):
            new_accumulators[k].from_value(atom_stats[k])
        new_len_accumulator = copy.deepcopy(self.len_accumulator)
        if len_value is not None and not supports(self.len_accumulator, Neutral):
            new_len_accumulator.from_value(len_value)
        self.group_counts = group_counts
        self.prev_beta = prev_beta
        self.prev_alpha = prev_alpha
        self.model_version = model_version
        self.accumulators = new_accumulators
        self.len_accumulator = new_len_accumulator
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator's keyed statistics into a shared dict."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self
        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the pooled keyed values."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())
        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HierarchicalDirichletProcessMixtureDataEncoder":
        """Returns a HierarchicalDirichletProcessMixtureDataEncoder for this accumulator."""
        comp_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = None if supports(self.len_accumulator, Neutral) else self.len_accumulator.acc_to_encoder()
        return HierarchicalDirichletProcessMixtureDataEncoder(comp_encoder, len_encoder)


class HierarchicalDirichletProcessMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for HDP-mixture sufficient-statistic accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory | None,
        name: str | None,
        keys: str | None,
    ) -> None:
        """Create an HDP-mixture accumulator factory."""
        self.factories = list(factories)
        if not self.factories:
            raise ValueError("HDP accumulator factories require at least one atom.")
        if len_factory is not None and not isinstance(len_factory, StatisticAccumulatorFactory):
            raise TypeError("HDP len_factory must satisfy StatisticAccumulatorFactory.")
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Returns a new HierarchicalDirichletProcessMixtureAccumulator."""
        len_acc = NullAccumulator() if self.len_factory is None else self.len_factory.make()
        return HierarchicalDirichletProcessMixtureAccumulator(
            [f.make() for f in self.factories], len_accumulator=len_acc, name=self.name, keys=self.keys
        )


class HierarchicalDirichletProcessMixtureEstimator(ParameterEstimator):
    """Estimates a HierarchicalDirichletProcessMixtureDistribution from
    accumulated group counts via the direct-assignment truncation updates."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        gamma: float = 1.0,
        alpha: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
        len_estimator: ParameterEstimator | None = None,
    ) -> None:
        """Create an estimator for the finite HDP-mixture approximation."""
        if isinstance(estimators, (str, bytes)) or not isinstance(estimators, Sequence):
            raise TypeError("HDP estimators must be a sequence.")
        self.estimators = list(estimators)
        if not self.estimators:
            raise ValueError("HDP estimators require at least one atom.")
        self.num_components = len(self.estimators)
        self.gamma = _positive_scalar(gamma, "gamma")
        self.alpha = _positive_scalar(alpha, "alpha")
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        if not isinstance(self.len_estimator, ParameterEstimator):
            raise TypeError("HDP len_estimator must satisfy ParameterEstimator.")

    def accumulator_factory(self) -> "HierarchicalDirichletProcessMixtureAccumulatorFactory":
        """Returns a HierarchicalDirichletProcessMixtureAccumulatorFactory for this estimator."""
        len_factory = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        )
        return HierarchicalDirichletProcessMixtureAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators], len_factory, self.name, self.keys
        )

    def model_log_density(self, model: HierarchicalDirichletProcessMixtureDistribution) -> float:
        """Log-density of the model parameters under the HDP priors.

        Sums the Dirichlet(gamma/K) log-density of the global weights beta, the
        Dirichlet(alpha*beta) log-density of each fitted group's weights on the
        exact active global support, and each atom estimator's
        model_log_density of its atom. Together with seq_local_elbo this forms
        the penalized objective maximized by the fit driver.
        """
        if not isinstance(model, HierarchicalDirichletProcessMixtureDistribution):
            raise TypeError("HDP model objective requires a matching model.")
        model._assert_structure()
        k = self.num_components
        if model.num_components != k:
            raise ValueError("HDP estimator and model atom counts differ.")

        beta_prior = DirichletDistribution(np.ones(k) * self.gamma / k)
        rv = float(beta_prior.log_density(model.beta))

        if model.group_weights is not None:
            active = model.beta > 0.0
            group_prior = DirichletDistribution(self.alpha * model.beta[active])
            for j in range(len(model.group_weights)):
                rv += float(group_prior.log_density(model.group_weights[j, active]))

        for index in range(k):
            est = self.estimators[index]
            comp = model.components[index]
            fn = getattr(est, "model_log_density", None)
            if fn is not None:
                term = fn(comp)
                if term is not None:
                    rv += float(term)

        if np.isnan(rv):
            raise ValueError("HDP model objective produced NaN.")
        return rv

    def estimate(self, nobs: float | None, suff_stat: tuple) -> HierarchicalDirichletProcessMixtureDistribution:
        """Estimate a HierarchicalDirichletProcessMixtureDistribution.

        Re-estimates each atom (whose conjugate update carries its posterior
        forward as its prior), updates the global weights beta via the
        expected-table-count approximation followed by the Dirichlet(gamma/K +
        m_.k) posterior mean, and sets each group's weights to the
        Dirichlet(alpha*beta) posterior mean (deliberately the mean, not the MAP,
        which degenerates when alpha*beta_k < 1).

        Args:
            nobs (Optional[float]): Not used. Kept for the stats
                ``ParameterEstimator.estimate(nobs, suff_stat)`` signature.
            suff_stat: Tuple (group_counts, prev_beta, prev_alpha,
                model_version, atom stats, len_value) as returned by
                ``HierarchicalDirichletProcessMixtureAccumulator.value()``.

        Returns:
            Fitted hierarchical Dirichlet-process mixture approximation.

        """
        k = self.num_components
        (
            group_count_map,
            prev_beta,
            prev_alpha,
            model_version,
            comp_stats,
            len_val,
        ) = _validated_hdp_statistics(suff_stat, k)
        group_ids = tuple(sorted(group_count_map))
        counts = (
            np.asarray([group_count_map[group_id] for group_id in group_ids])
            if group_ids
            else np.zeros((0, k))
        )

        atom_counts = counts.sum(axis=0) if len(counts) else np.zeros(k)
        components = [
            self.estimators[i].estimate(atom_counts[i], comp_stats[i])
            for i in range(k)
        ]

        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            len_dist: SequenceEncodableProbabilityDistribution = NullDistribution()
        else:
            len_dist = self.len_estimator.estimate(None, len_val)

        alpha = self.alpha if prev_alpha is None else prev_alpha
        beta0 = np.ones(k) / k if prev_beta is None else prev_beta

        # global weights via the expected-table-count approximation:
        # m_jk = alpha*beta_k * (psi(alpha*beta_k + n_jk) - psi(alpha*beta_k))
        ab = alpha * beta0
        if counts.shape[0] > 0:
            m_mat = np.zeros_like(counts)
            active = ab > 0.0
            m_mat[:, active] = ab[active] * (
                digamma(ab[active] + counts[:, active]) - digamma(ab[active])
            )
            if np.any(~np.isfinite(m_mat)) or np.any(m_mat < 0.0):
                raise ValueError("HDP expected table counts must be finite and non-negative.")
            m_k = m_mat.sum(axis=0)
        else:
            m_k = np.zeros(k)

        denominator = float(m_k.sum() + self.gamma)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("HDP global-weight update has an invalid normalizer.")
        beta = _simplex_vector(
            (m_k + self.gamma / k) / denominator,
            k,
            "estimated beta",
        )

        # per-group posterior-mean weights under the Dirichlet(alpha*beta) prior.
        # The mean (not the MAP) is used deliberately: with alpha*beta_k < 1 the
        # Dirichlet density is unbounded on the simplex boundary, so MAP weights
        # degenerate to spikes; the mean is strictly interior and keeps the
        # penalized objective well-defined.
        ab_new = alpha * beta
        row_denominators = counts.sum(axis=1) + alpha
        if np.any(~np.isfinite(row_denominators)) or np.any(row_denominators <= 0.0):
            raise ValueError("HDP group posterior rows have invalid normalizers.")
        group_weights = (
            (counts + ab_new) / row_denominators[:, None]
            if len(counts)
            else np.zeros((0, k))
        )
        if len(group_weights) and (
            np.any(~np.isfinite(group_weights))
            or np.any(group_weights < 0.0)
            or not np.allclose(
                group_weights.sum(axis=1),
                1.0,
                rtol=_SIMPLEX_RTOL,
                atol=_SIMPLEX_ATOL,
            )
        ):
            raise ValueError("HDP group posterior weights are not row-stochastic.")

        result = HierarchicalDirichletProcessMixtureDistribution(
            components,
            beta,
            alpha,
            self.gamma,
            group_weights=group_weights,
            name=self.name,
            len_dist=len_dist,
            group_ids=group_ids,
        )
        result.fit_metadata = {
            "converged": True,
            "repairs": (),
            "group_ids": group_ids,
            "source_model_version": model_version,
            "table_count_approximation": True,
        }
        return result


class HierarchicalDirichletProcessMixtureDataEncoder(DataSequenceEncoder):
    """Encodes groups into a flat component encoding with per-group offsets."""

    def __init__(self, encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder | None = None) -> None:
        """Data encoder for grouped HDP-mixture observations.

        Args:
            encoder (DataSequenceEncoder): Encoder for the atom (component)
                distributions.
            len_encoder (Optional[DataSequenceEncoder]): Encoder for the group
                sizes; None treats sizes as exogenous.

        """
        if not isinstance(encoder, DataSequenceEncoder):
            raise TypeError("HDP data encoder requires a DataSequenceEncoder atom encoder.")
        if len_encoder is not None and not isinstance(len_encoder, DataSequenceEncoder):
            raise TypeError("HDP length encoder must satisfy DataSequenceEncoder.")
        self.encoder = encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        return "HierarchicalDirichletProcessMixtureDataEncoder(%s, %s)" % (str(self.encoder), str(self.len_encoder))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchicalDirichletProcessMixtureDataEncoder):
            return False
        return self.encoder == other.encoder and self.len_encoder == other.len_encoder

    def seq_encode(self, x: Sequence[Sequence]) -> Any:
        """Encode groups into stable IDs, lengths, offsets, atom data, and length data."""
        unpacked = [_group_identity_and_values(group) for group in x]
        group_ids = tuple(group_id for group_id, _ in unpacked)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(
                "HDP group identities must be unique; wrap duplicate-content groups in HDPGroup."
            )
        groups = [values for _, values in unpacked]
        lengths = np.asarray([len(group) for group in groups], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat: list[Any] = []
        for group in groups:
            flat.extend(group)
        flat_enc = self.encoder.seq_encode(flat)

        len_enc = None if self.len_encoder is None else self.len_encoder.seq_encode(lengths)

        return group_ids, lengths, offsets, flat_enc, len_enc

    def row_count(self, x: Any) -> int:
        """Return the number of encoded groups after validating grouped geometry."""
        group_ids, lengths, _offsets, _flat_enc, _len_enc = _validated_encoded_groups(x)
        if len(group_ids) != len(lengths):
            raise ValueError("HDP encoded group identities and lengths differ.")
        return len(group_ids)
