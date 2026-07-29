"""Integer Chow-Liu tree distributions for fixed-length integer vectors.

Mixle supports Chow & Liu trees [1] through the IntegerChowLiuTree (Integer Chow Liu Tree) class of objects. IntegerChowLiuTrees model
non-Markov conditional dependence for fixed-length sequences of integers with the likelihood functions of the form

    P(x_1, x_2,..,x_n) = P(x_i1) P(x_{i_2}|x_{j_2})*...*P(x_{i_n}|x_{j_n}),

where j_k < i_k for all k = 1,2,3,..N.

Data type: Union[Sequence[int], np.ndarray] .

"""

import itertools
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

# Tolerance for the "does this table exponentiate to a valid probability distribution" check -- numpy's
# own np.isclose default (rtol=1e-05, atol=1e-08), matching the simplex-sum tolerance used elsewhere for
# this same kind of check (MixtureDistribution.w, SymmetricDirichletDistribution, DictDirichletDistribution
# all use this exact rtol/atol pair). Tight enough to catch a genuinely wrong table (e.g. raw un-normalized
# counts or weights passed in place of a proper conditional) while tolerating ordinary float rounding in a
# fitted table.
_SIMPLEX_SUM_RTOL = 1.0e-5
_SIMPLEX_SUM_ATOL = 1.0e-8
_COUNT_ATOL = 1.0e-8
_DEFAULT_ENUMERATION_ITEM_BUDGET = 100_000


class IntegerChowLiuStatistics(NamedTuple):
    """Versioned, tuple-compatible integer Chow–Liu statistics."""

    num_features: int
    num_states: int
    counts: np.ndarray
    marginal_counts: np.ndarray

    @property
    def schema_version(self) -> int:
        """Return the serialized statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _positive_integer_or_none(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    result = _exact_integer(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _validated_integer_rows(
    value: Any,
    *,
    num_features: int | None,
    domain_sizes: Sequence[int] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape == (0,) and num_features is not None:
        raw = raw.reshape((0, num_features))
    if raw.ndim != 2:
        raise ValueError("Integer Chow-Liu batches must be two-dimensional")
    if num_features is not None and raw.shape[1] != num_features:
        raise ValueError("Integer Chow-Liu observation width does not match feature count")
    if raw.dtype == np.bool_:
        raise TypeError("Integer Chow-Liu states must be exact integers")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError("Integer Chow-Liu states exceed integer range")
        result = raw.astype(np.int64, copy=False)
    elif np.issubdtype(raw.dtype, np.floating):
        if (
            np.any(~np.isfinite(raw))
            or np.any(raw != np.floor(raw))
            or np.any(raw < 0.0)
            or np.any(raw >= float(2**63))
        ):
            raise ValueError("Integer Chow-Liu states must be finite exact integers")
        result = raw.astype(np.int64)
    else:
        raise TypeError("Integer Chow-Liu states must be exact integers")
    if np.any(result < 0):
        raise ValueError("Integer Chow-Liu states must be non-negative")
    if domain_sizes is not None:
        if len(domain_sizes) != result.shape[1]:
            raise ValueError("Integer Chow-Liu domain metadata does not match feature count")
        for feature, size in enumerate(domain_sizes):
            if np.any(result[:, feature] >= size):
                raise ValueError(f"Integer Chow-Liu state for feature {feature} is outside [0, {size})")
    return result


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Integer Chow-Liu weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"Integer Chow-Liu weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Integer Chow-Liu weights must be finite and non-negative")
    return weights


def _validated_statistics(value: Any) -> IntegerChowLiuStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("Integer Chow-Liu sufficient statistics must contain four fields")
    num_features = _positive_integer_or_none(
        value[0],
        label="Integer Chow-Liu statistic feature count",
    )
    num_states = _positive_integer_or_none(
        value[1],
        label="Integer Chow-Liu statistic state count",
    )
    if num_features is None or num_states is None:
        raise ValueError("Integer Chow-Liu sufficient statistics require fixed dimensions")
    try:
        counts = np.asarray(value[2], dtype=np.float64)
        marginal_counts = np.asarray(value[3], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Integer Chow-Liu counts must be numeric arrays") from exc
    expected_counts = (num_features, num_features, num_states, num_states)
    expected_marginals = (num_features, num_states)
    if counts.shape != expected_counts:
        raise ValueError(f"Integer Chow-Liu pair counts must have shape {expected_counts}")
    if marginal_counts.shape != expected_marginals:
        raise ValueError(f"Integer Chow-Liu marginal counts must have shape {expected_marginals}")
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(~np.isfinite(marginal_counts))
        or np.any(marginal_counts < 0.0)
    ):
        raise ValueError("Integer Chow-Liu counts must be finite and non-negative")
    for feature in range(num_features):
        if np.any(counts[feature, : feature + 1] != 0.0):
            raise ValueError("Integer Chow-Liu counts must use the canonical upper-triangle layout")
    total_weight = float(marginal_counts[0].sum())
    tolerance = _COUNT_ATOL * max(1.0, total_weight)
    for feature in range(num_features):
        if not np.isclose(
            marginal_counts[feature].sum(),
            total_weight,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Integer Chow-Liu marginal totals must agree across features")
    for first in range(num_features - 1):
        for second in range(first + 1, num_features):
            pair = counts[first, second]
            if not np.allclose(
                pair.sum(axis=1),
                marginal_counts[first],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("Integer Chow-Liu pair row sums contradict marginal counts")
            if not np.allclose(
                pair.sum(axis=0),
                marginal_counts[second],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("Integer Chow-Liu pair column sums contradict marginal counts")
    return IntegerChowLiuStatistics(
        num_features,
        num_states,
        counts.copy(),
        marginal_counts.copy(),
    )


def _validate_conditional_log_density_table(table: np.ndarray, feature: int, parent: int | None) -> None:
    """Validate one feature's log-probability table against the tree's factorization semantics.

    ``table`` is read the same way log_density()/seq_log_density()/the sampler read it: a 1-d root
    marginal (log P(x_feature)) when ``parent`` is None, or a 2-d conditional table indexed
    ``table[parent_val, child_val]`` (log P(x_feature | x_parent = parent_val)) otherwise. Nothing about
    the constructor's signature guarantees ``table`` is an actual probability table -- e.g. raw
    (non-normalized) counts or weights can be passed in its place, in which case log_density() and
    seq_log_density() still return finite (just meaningless) scores, while the sampler crashes far from
    the actual mistake, deep inside np.random.choice, with "probabilities do not sum to 1". Catch that
    here instead, at construction.

    Conditional rows may sum to zero only provisionally. The distribution
    constructor subsequently proves that every such parent state is globally
    unreachable before accepting the complete tree.

    Args:
        table (np.ndarray): The feature's conditional_log_densities entry.
        feature (int): Feature id the table belongs to (used only for the error message).
        parent (Optional[int]): Parent feature id, or None if ``feature`` is the tree root.

    Raises:
        ValueError: If ``table`` has the wrong number of dimensions, contains NaN or +inf entries, or
            (once exponentiated) does not sum to ~1 (or ~0, for a conditional row) along the axis its
            semantics require.

    """
    finite_or_neg_inf = np.isfinite(table) | np.isneginf(table)
    if parent is None:
        if table.ndim != 1:
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires the root table for feature %d to be 1-d, got "
                "shape %r." % (feature, table.shape)
            )
        if not np.all(finite_or_neg_inf):
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires finite log-density entries (NaN/+inf not "
                "allowed) in the root table for feature %d." % feature
            )
        total = float(np.exp(table).sum())
        if not np.isclose(total, 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL):
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires the root table for feature %d to sum to 1.0 "
                "once exponentiated (a valid marginal distribution), got sum=%r." % (feature, total)
            )
    else:
        if table.ndim != 2:
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires the conditional table for feature %d given "
                "parent %d to be 2-d, got shape %r." % (feature, parent, table.shape)
            )
        if not np.all(finite_or_neg_inf):
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires finite log-density entries (NaN/+inf not "
                "allowed) in the conditional table for feature %d given parent %d." % (feature, parent)
            )
        row_sums = np.exp(table).sum(axis=1)
        valid_row = np.isclose(row_sums, 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL) | np.isclose(
            row_sums, 0.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL
        )
        if not np.all(valid_row):
            bad_rows = np.nonzero(~valid_row)[0]
            raise ValueError(
                "IntegerChowLiuTreeDistribution requires each row of the conditional table for feature "
                "%d given parent %d to sum to 1.0 once exponentiated (a valid P(feature | parent) row; "
                "0.0 is also accepted for a parent state that never occurs); row(s) %s sum to %s."
                % (feature, parent, bad_rows.tolist(), row_sums[bad_rows].tolist())
            )


class IntegerChowLiuTreeDistribution(SequenceEncodableProbabilityDistribution):
    """Integer Chow-Liu tree distribution factorizing a joint over fixed-length integer vectors along a tree.

    Data type: Union[Sequence[int], np.ndarray] (fixed-length vector of non-negative integers).
    """

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for integer Chow-Liu generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the integer Chow-Liu tree."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="integer_chow_liu_tree",
            distribution_type=cls,
            parameters=(
                ParameterSpec("conditional_log_densities", constraint="log_probability_tables", differentiable=False),
            ),
            statistics=(
                StatisticSpec("num_features", kind="metadata", additive=False, scales=False),
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("counts", kind="pairwise_count_tensor"),
                StatisticSpec("marginal_counts", kind="count_tensor"),
            ),
            support="fixed_integer_tuple_tree",
            differentiable=False,
        )

    def __init__(
        self,
        dependency_list: list[int | None],
        conditional_log_densities: Sequence[float] | np.ndarray,
        feature_order: Sequence[int] | None = None,
        name: str | None = None,
        keys: str | None = None,
        pseudo_count: float | None = None,
        enumeration_item_budget: int | None = _DEFAULT_ENUMERATION_ITEM_BUDGET,
    ) -> None:
        """Create an integer Chow-Liu tree distribution.

        Args:
            dependency_list (List[Optional[int]]): Parent feature id for each feature in feature_order, or None
                for the (exactly one) root feature with no parent.
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            feature_order (Optional[Sequence[int]]): Ordering of features. If None, ordering is assumed as entered.
            name (Optional[str]): Optional distribution name.

        Attributes:
            feature_order (Sequence[int]): Ordering of features. If None, ordering is assumed as entered.
            dependency_list (List[ Tuple[int, Optional[int]]]): List of Tuples containing each feature's
                order id and its parent id (or None for the root).
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            conditional_densities (np.ndarray): Conditional densities as numpy array.
            num_features (int): Total number of features.
            name (Optional[str]): Optional distribution name.

        """
        self.num_features = len(dependency_list)
        if self.num_features == 0:
            raise ValueError("IntegerChowLiuTreeDistribution requires at least one feature")
        if len(conditional_log_densities) != self.num_features:
            raise ValueError("Integer Chow-Liu table count must match dependency count")
        raw_order = range(self.num_features) if feature_order is None else feature_order
        checked_order = [_exact_integer(value, label="Integer Chow-Liu feature index") for value in raw_order]
        if sorted(checked_order) != list(range(self.num_features)):
            raise ValueError("Integer Chow-Liu feature_order must be a feature permutation")
        self.feature_order = (
            range(self.num_features) if checked_order == list(range(self.num_features)) else tuple(checked_order)
        )
        parents = [
            None
            if parent is None
            else _exact_integer(
                parent,
                label="Integer Chow-Liu parent index",
            )
            for parent in dependency_list
        ]
        if sum(parent is None for parent in parents) != 1:
            raise ValueError("Integer Chow-Liu dependency list must contain exactly one root")
        self.dependency_list = tuple(zip(self.feature_order, parents))
        seen: set[int] = set()
        for position, (feature, parent) in enumerate(self.dependency_list):
            if parent is None:
                if position != 0:
                    raise ValueError("Integer Chow-Liu feature_order must begin with the root")
            elif parent not in seen:
                raise ValueError("Integer Chow-Liu feature_order must place every parent before its child")
            seen.add(feature)

        self.conditional_log_densities = [
            np.array(table, dtype=np.float64, copy=True) for table in conditional_log_densities
        ]
        for i, (feature, parent) in enumerate(self.dependency_list):
            _validate_conditional_log_density_table(self.conditional_log_densities[i], feature, parent)
        self.domain_sizes = self._validate_domain_geometry_and_reachability()
        for table in self.conditional_log_densities:
            table.setflags(write=False)
        self.conditional_log_densities = tuple(self.conditional_log_densities)
        self.conditional_densities = tuple(np.exp(table).copy() for table in self.conditional_log_densities)
        for table in self.conditional_densities:
            table.setflags(write=False)
        self.name = name
        self.keys = keys
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _nonnegative_scalar(
                pseudo_count,
                label="Integer Chow-Liu pseudo-count",
            )
        )
        if enumeration_item_budget is None:
            self.enumeration_item_budget = None
        else:
            self.enumeration_item_budget = _exact_integer(
                enumeration_item_budget,
                label="Integer Chow-Liu enumeration item budget",
            )
            if self.enumeration_item_budget < 0:
                raise ValueError("Integer Chow-Liu enumeration item budget must be non-negative")

    def _validate_domain_geometry_and_reachability(self) -> tuple[int, ...]:
        domain_sizes: list[int | None] = [None] * self.num_features
        reachable: list[np.ndarray | None] = [None] * self.num_features
        for index, (feature, parent) in enumerate(self.dependency_list):
            table = self.conditional_log_densities[index]
            if parent is None:
                child_size = table.shape[0]
                if child_size <= 0:
                    raise ValueError("Integer Chow-Liu root domain must be non-empty")
                domain_sizes[feature] = child_size
                reachable[feature] = np.exp(table) > 0.0
                continue

            parent_size, child_size = table.shape
            if parent_size <= 0 or child_size <= 0:
                raise ValueError("Integer Chow-Liu conditional domains must be non-empty")
            known_parent_size = domain_sizes[parent]
            if known_parent_size != parent_size:
                raise ValueError(
                    f"Integer Chow-Liu parent domain size for feature {feature} contradicts its parent table"
                )
            if domain_sizes[feature] is not None and domain_sizes[feature] != child_size:
                raise ValueError("Integer Chow-Liu child domain sizes are inconsistent")
            parent_reachable = reachable[parent]
            if parent_reachable is None:
                raise ValueError("Integer Chow-Liu reachability requires parent-first order")
            probabilities = np.exp(table)
            row_sums = probabilities.sum(axis=1)
            missing = parent_reachable & np.isclose(
                row_sums,
                0.0,
                rtol=_SIMPLEX_SUM_RTOL,
                atol=_SIMPLEX_SUM_ATOL,
            )
            if np.any(missing):
                rows = np.nonzero(missing)[0].tolist()
                raise ValueError(
                    f"Integer Chow-Liu conditional for feature {feature} has "
                    f"zero mass for reachable parent state(s) {rows}"
                )
            domain_sizes[feature] = child_size
            reachable[feature] = np.any(
                probabilities[parent_reachable] > 0.0,
                axis=0,
            )
        if any(size is None for size in domain_sizes):
            raise ValueError("Integer Chow-Liu could not infer every feature domain")
        return tuple(int(size) for size in domain_sizes)

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return the constructor-owned state used by the safe JSON codec."""
        return {
            "dependency_list": [parent for _, parent in self.dependency_list],
            "conditional_log_densities": [table.copy() for table in self.conditional_log_densities],
            "feature_order": self.feature_order,
            "name": self.name,
            "keys": self.keys,
            "pseudo_count": self.pseudo_count,
            "enumeration_item_budget": self.enumeration_item_budget,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Validate and reconstruct current and legacy serialized tree state."""
        required = {
            "dependency_list",
            "conditional_log_densities",
            "feature_order",
            "name",
        }
        optional = {
            "keys",
            "pseudo_count",
            "enumeration_item_budget",
        }
        legacy_derived = {"conditional_densities", "num_features"}
        fields = set(state)
        if not required.issubset(fields) or not fields.issubset(required | optional | legacy_derived):
            raise ValueError(
                "invalid IntegerChowLiuTreeDistribution state fields: "
                "required %r, got %r" % (sorted(required), sorted(fields))
            )
        if bool(fields & legacy_derived) and not legacy_derived.issubset(fields):
            raise ValueError("legacy Integer Chow-Liu derived state must contain both fields")
        dependencies = state["dependency_list"]
        if dependencies and all(isinstance(entry, (list, tuple)) and len(entry) == 2 for entry in dependencies):
            dependencies = [entry[1] for entry in dependencies]
        self.__init__(
            dependencies,
            state["conditional_log_densities"],
            feature_order=state["feature_order"],
            name=state["name"],
            keys=state.get("keys"),
            pseudo_count=state.get("pseudo_count"),
            enumeration_item_budget=state.get(
                "enumeration_item_budget",
                _DEFAULT_ENUMERATION_ITEM_BUDGET,
            ),
        )
        if legacy_derived.issubset(fields):
            if (
                _exact_integer(
                    state["num_features"],
                    label="serialized Integer Chow-Liu feature count",
                )
                != self.num_features
            ):
                raise ValueError("serialized num_features is inconsistent with dependency_list")
            expected = state["conditional_densities"]
            if len(expected) != len(self.conditional_densities) or any(
                not np.array_equal(actual, saved) for actual, saved in zip(self.conditional_densities, expected)
            ):
                raise ValueError("serialized conditional_densities are inconsistent with log densities")

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""

        def _fmt(u: np.ndarray) -> str:
            # Recursively render as a bracketed literal that preserves shape (1-d root tables vs.
            # 2-d parent/child tables), so eval(str(dist)) reconstructs the same table shapes.
            if u.ndim <= 1:
                return "[" + ",".join(map(str, u)) + "]"
            return "[" + ",".join(_fmt(row) for row in u) + "]"

        f1 = ",".join([str(u[1]) for u in self.dependency_list])
        f3 = ",".join([str(u[0]) for u in self.dependency_list])
        f2 = ",".join(_fmt(u) for u in self.conditional_log_densities)
        f4 = repr(self.name)
        return (
            "IntegerChowLiuTreeDistribution([%s], [%s], feature_order=[%s], "
            "name=%s, keys=%r, pseudo_count=%r, enumeration_item_budget=%r)"
            % (
                f1,
                f2,
                f3,
                f4,
                self.keys,
                self.pseudo_count,
                self.enumeration_item_budget,
            )
        )

    def density(self, x: Sequence[int] | np.ndarray) -> float:
        """Density of integer Chow-Liu tree distribution at observation x.

        See log_density() for details.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[int] | np.ndarray) -> float:
        """Log-density of integer Chow-Liu tree distribution at observation x.

        Sums the conditional log-densities of each feature given its parent in the dependency tree
        (the root feature contributes its marginal log-density).

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.

        Returns:
            Log-density at observation x.

        """
        row = _validated_integer_rows(
            (x,),
            num_features=self.num_features,
            domain_sizes=self.domain_sizes,
        )[0]
        rv = 0.0
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][row[j]]
            else:
                rv += self.conditional_log_densities[i][row[k], row[j]]

        return float(rv)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        xx = _validated_integer_rows(
            x,
            num_features=self.num_features,
            domain_sizes=self.domain_sizes,
        )
        rv = np.zeros(xx.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][xx[:, j]]
            else:
                rv += self.conditional_log_densities[i][xx[:, k], xx[:, j]]

        return rv

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized table lookup for fixed integer tree factors."""
        checked = _validated_integer_rows(
            x,
            num_features=self.num_features,
            domain_sizes=self.domain_sizes,
        )
        xx = engine.asarray(checked)
        rv = engine.zeros(checked.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            table = engine.asarray(np.array(self.conditional_log_densities[i], copy=True))
            if k is None:
                rv = rv + table[xx[:, j]]
            else:
                rv = rv + table[xx[:, k], xx[:, j]]
        return rv

    def sampler(self, seed: int | None = None) -> "IntegerChowLiuTreeSampler":
        """Create a sampler for this integer Chow-Liu tree distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerChowLiuTreeSampler: Sampler bound to this distribution.

        """
        return IntegerChowLiuTreeSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerChowLiuTreeEstimator":
        """Create an estimator initialized from this integer Chow-Liu tree distribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            IntegerChowLiuTreeEstimator: Estimator configured with the same feature count and state count.

        """
        effective_pseudo = self.pseudo_count if pseudo_count is None else pseudo_count
        num_states = max(self.domain_sizes)
        return IntegerChowLiuTreeEstimator(
            num_features=self.num_features,
            num_states=num_states,
            pseudo_count=effective_pseudo,
            keys=self.keys,
            name=self.name,
            enumeration_item_budget=self.enumeration_item_budget,
        )

    def dist_to_encoder(self) -> "IntegerChowLiuTreeDataEncoder":
        """Return a data encoder for integer Chow-Liu tree observations."""
        return IntegerChowLiuTreeDataEncoder(
            self.num_features,
            self.domain_sizes,
        )

    def enumerator(
        self,
        max_items: int | None = None,
    ) -> "IntegerChowLiuTreeEnumerator":
        """Returns IntegerChowLiuTreeEnumerator iterating fixed-length integer vectors in descending probability order."""
        return IntegerChowLiuTreeEnumerator(
            self,
            max_items=(self.enumeration_item_budget if max_items is None else max_items),
        )

    def support_size(self) -> int:
        """Return the finite Cartesian event-space size."""
        return int(np.prod(self.domain_sizes, dtype=object))


class IntegerChowLiuTreeEnumerator(DistributionEnumerator):
    """Enumerates the finite support of an integer Chow-Liu tree."""

    def __init__(
        self,
        dist: IntegerChowLiuTreeDistribution,
        max_items: int | None = _DEFAULT_ENUMERATION_ITEM_BUDGET,
    ) -> None:
        """Create an enumerator for integer Chow-Liu tree observations.

        The support is the Cartesian product of each feature's finite state range, inferred
        from the root marginal and conditional probability tables.

        Args:
            dist (IntegerChowLiuTreeDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if max_items is None:
            self.max_items = None
        else:
            self.max_items = _exact_integer(
                max_items,
                label="Integer Chow-Liu enumeration item budget",
            )
            if self.max_items < 0:
                raise ValueError("Integer Chow-Liu enumeration item budget must be non-negative")
        self._support_size = int(np.prod(dist.domain_sizes, dtype=object))
        self._entries: list[tuple[list[int], float]] | None = None
        self._pos = 0
        self.items_yielded = 0
        self.truncated = False
        self.termination_reason: str | None = None

    def _materialize_within_budget(self) -> None:
        if self._entries is not None:
            return
        if self.max_items is not None and self._support_size > self.max_items:
            self.truncated = True
            self.termination_reason = "item_budget_exhausted"
            raise EnumerationError(
                self.dist,
                reason=(f"support size {self._support_size} exceeds enumeration item budget {self.max_items}"),
            )
        entries = []
        ranges = [range(size) for size in self.dist.domain_sizes]
        for value in itertools.product(*ranges):
            log_probability = float(self.dist.log_density(value))
            if log_probability > -np.inf:
                entries.append((list(value), log_probability))
        entries.sort(key=lambda item: -item[1])
        self._entries = entries

    def __next__(self) -> tuple[list[int], float]:
        self._materialize_within_budget()
        if self._entries is None:
            raise RuntimeError("Integer Chow-Liu enumeration was not initialized")
        if self._pos >= len(self._entries):
            self.termination_reason = "support_exhausted"
            raise StopIteration
        item = self._entries[self._pos]
        self._pos += 1
        self.items_yielded += 1
        return item

    @property
    def receipt(self) -> dict[str, object]:
        """Return enumeration budget and progress metadata."""
        return {
            "items_yielded": self.items_yielded,
            "support_size": self._support_size,
            "item_budget": self.max_items,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
        }


class IntegerChowLiuTreeSampler(DistributionSampler):
    """Sampler for the IntegerChowLiuTreeDistribution. Samples each feature given its sampled parent value."""

    def __init__(self, dist: IntegerChowLiuTreeDistribution, seed: int | None = None) -> None:
        """Create a sampler for integer Chow-Liu tree observations.

        Args:
            dist (IntegerChowLiuTreeDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int | None] | Sequence[list[int | None]]:
        """Draw iid integer vectors from the integer Chow-Liu tree distribution.

        Features are drawn in dependency order: the root from its marginal, each remaining
        feature from its conditional given the sampled parent value.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single vector is returned.

        Returns:
            A single integer vector (List[int]) if size is None, else a list of size vectors.

        """

        if size is None:
            rv = [None] * self.dist.num_features

            for i, (j, k) in enumerate(self.dist.dependency_list):
                if k is None:
                    pmat = self.dist.conditional_densities[i]
                else:
                    pmat = self.dist.conditional_densities[i][rv[k], :]

                rv[j] = self.rng.choice(len(pmat), p=pmat)

            return rv
        checked_size = _exact_integer(
            size,
            label="Integer Chow-Liu sample size",
        )
        if checked_size < 0:
            raise ValueError("Integer Chow-Liu sample size must be non-negative")
        return [self.sample() for _ in range(checked_size)]


class IntegerChowLiuTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the IntegerChowLiuTreeDistribution. Tracks pairwise joint and marginal feature-state counts."""

    def __init__(self, num_features: int, num_states: int, keys: str | None = None, name: str | None = None):
        """Create an accumulator for integer Chow-Liu tree sufficient statistics.

        Args:
            num_features (int): Number of features (length of observed integer vectors).
            num_states (int): Number of states (distinct integer values) per feature.
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        Attributes:
            num_states (int): Number of states per feature.
            num_features (int): Number of features.
            counts (Optional[np.ndarray]): Pairwise joint counts with shape
                (num_features, num_features, num_states, num_states). None until dimensions are known.
            marginal_counts (Optional[np.ndarray]): Marginal counts with shape (num_features, num_states).
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.num_states = _positive_integer_or_none(
            num_states,
            label="Integer Chow-Liu accumulator state count",
        )
        self.num_features = _positive_integer_or_none(
            num_features,
            label="Integer Chow-Liu accumulator feature count",
        )
        self._fixed_num_states = self.num_states is not None
        self._fixed_num_features = self.num_features is not None

        if self.num_states is not None and self.num_features is not None:
            self.counts = np.zeros(
                (
                    self.num_features,
                    self.num_features,
                    self.num_states,
                    self.num_states,
                )
            )
            self.marginal_counts = np.zeros((self.num_features, self.num_states))
        else:
            self.counts = None
            self.marginal_counts = None

        self.keys = keys
        self.name = name

    def _expand_states(self, num_states: int, num_features: int):
        """Allocate or grow the count arrays to hold num_states states for num_features features.

        Args:
            num_states (int): New number of states per feature.
            num_features (int): Number of features.

        """
        checked_features = _positive_integer_or_none(
            num_features,
            label="Integer Chow-Liu feature count",
        )
        checked_states = _positive_integer_or_none(
            num_states,
            label="Integer Chow-Liu state count",
        )
        if checked_features is None or checked_states is None:
            raise ValueError("Integer Chow-Liu count allocation requires fixed dimensions")
        if self.num_features is not None and checked_features != self.num_features:
            raise ValueError("Integer Chow-Liu feature count cannot change during accumulation")
        if self._fixed_num_states and self.num_states != checked_states:
            raise ValueError("Integer Chow-Liu observation exceeds fixed state count")
        if self.counts is None:
            self.num_features = checked_features
            self.num_states = checked_states
            self.counts = np.zeros(
                (
                    checked_features,
                    checked_features,
                    checked_states,
                    checked_states,
                )
            )
            self.marginal_counts = np.zeros((checked_features, checked_states))

        elif checked_states > self.num_states:
            old_num_states = self.num_states
            new_counts = np.zeros(
                (
                    checked_features,
                    checked_features,
                    checked_states,
                    checked_states,
                )
            )
            new_marginal = np.zeros((checked_features, checked_states))
            new_counts[:, :, :old_num_states, :old_num_states] = self.counts
            new_marginal[:, :old_num_states] = self.marginal_counts
            self.num_features = checked_features
            self.num_states = checked_states
            self.counts = new_counts
            self.marginal_counts = new_marginal

    def update(
        self, x: Sequence[int] | np.ndarray, weight: float, estimate: IntegerChowLiuTreeDistribution | None
    ) -> None:
        """Update pairwise joint and marginal counts with a weighted observation.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.
            weight (float): Weight for observation.
            estimate (Optional[IntegerChowLiuTreeDistribution]): Previous estimate (unused).

        """
        xx = _validated_integer_rows(
            (x,),
            num_features=self.num_features,
        )[0]
        if len(xx) == 0:
            raise ValueError("Integer Chow-Liu observations require at least one feature")
        checked_weight = _nonnegative_scalar(
            weight,
            label="Integer Chow-Liu weight",
        )
        if self._fixed_num_states and np.any(xx >= self.num_states):
            raise ValueError("Integer Chow-Liu observation exceeds fixed state count")
        required_states = self.num_states if self._fixed_num_states else int(np.max(xx)) + 1
        if self.counts is None or required_states > self.num_states:
            self._expand_states(required_states, len(xx))
        ff = np.arange(self.num_features)

        self.marginal_counts[ff, xx] += checked_weight
        for first in range(self.num_features - 1):
            for second in range(first + 1, self.num_features):
                self.counts[
                    first,
                    second,
                    xx[first],
                    xx[second],
                ] += checked_weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: IntegerChowLiuTreeDistribution | None) -> None:
        """Vectorized update of pairwise joint and marginal counts from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[IntegerChowLiuTreeDistribution]): Previous estimate (unused).

        """
        xx = _validated_integer_rows(
            x,
            num_features=self.num_features,
        )
        if xx.shape[1] == 0:
            raise ValueError("Integer Chow-Liu observations require at least one feature")
        checked_weights = _validated_weights(weights, xx.shape[0])
        if self._fixed_num_states and np.any(xx >= self.num_states):
            raise ValueError("Integer Chow-Liu observation exceeds fixed state count")
        if xx.shape[0] == 0:
            if self.num_features is None:
                self.num_features = xx.shape[1]
            return
        required_states = self.num_states if self._fixed_num_states else int(np.max(xx)) + 1
        if self.counts is None or required_states > self.num_states:
            self._expand_states(required_states, xx.shape[1])

        for i in range(self.num_features):
            self.marginal_counts[i, :] += np.bincount(
                xx[:, i],
                weights=checked_weights,
                minlength=self.num_states,
            )

            for j in range(i + 1, self.num_features):
                joint_idx = xx[:, i] * self.num_states + xx[:, j]
                joint_cnt = np.bincount(
                    joint_idx,
                    weights=checked_weights,
                    minlength=self.num_states * self.num_states,
                )
                joint_cnt = np.reshape(
                    joint_cnt,
                    (self.num_states, self.num_states),
                )

                self.counts[i, j, :, :] += joint_cnt

    def initialize(self, x: Sequence[int] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics with a weighted observation.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.
            weights (np.ndarray): Weights for each of the N observations.
            rng (Optional[RandomState]): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[int, int, np.ndarray, np.ndarray]) -> "IntegerChowLiuTreeAccumulator":
        """Combine sufficient statistics from another accumulator into this one.

        Count arrays are expanded if the incoming statistics track more states.

        Args:
            suff_stat (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of
                states, pairwise joint counts, and marginal counts.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        if (
            isinstance(suff_stat, (tuple, list))
            and len(suff_stat) == 4
            and suff_stat[2] is None
            and suff_stat[3] is None
        ):
            incoming_features = _positive_integer_or_none(
                suff_stat[0],
                label="Integer Chow-Liu statistic feature count",
            )
            incoming_states = _positive_integer_or_none(
                suff_stat[1],
                label="Integer Chow-Liu statistic state count",
            )
            if (
                incoming_features is not None
                and self.num_features is not None
                and incoming_features != self.num_features
            ):
                raise ValueError("Integer Chow-Liu feature counts do not match for combine")
            if incoming_states is not None and self._fixed_num_states and incoming_states != self.num_states:
                raise ValueError("Integer Chow-Liu state counts do not match fixed configuration")
            return self
        checked = _validated_statistics(suff_stat)
        num_features, num_states, counts, marginal_counts = checked
        if self.num_features is not None and self.num_features != num_features:
            raise ValueError("Integer Chow-Liu feature counts do not match for combine")
        if self._fixed_num_states and self.num_states != num_states:
            raise ValueError("Integer Chow-Liu state counts do not match fixed configuration")
        if self.counts is None:
            self.num_features = num_features
            self.num_states = num_states
            self.counts = counts.copy()
            self.marginal_counts = marginal_counts.copy()
            return self
        if self.num_states < num_states:
            self._expand_states(num_states, num_features)
        self.counts[:, :, :num_states, :num_states] += counts
        self.marginal_counts[:, :num_states] += marginal_counts

        return self

    def value(self) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Returns sufficient statistics as a Tuple of number of features, number of states, pairwise
        joint counts, and marginal counts."""
        if self.counts is None or self.marginal_counts is None:
            return self.num_features, self.num_states, None, None
        return IntegerChowLiuStatistics(
            self.num_features,
            self.num_states,
            self.counts.copy(),
            self.marginal_counts.copy(),
        )

    def from_value(self, x: tuple[int, int, np.ndarray, np.ndarray]) -> "IntegerChowLiuTreeAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of states,
                pairwise joint counts, and marginal counts.

        Returns:
            Self, with sufficient statistics set to x.

        """
        checked = _validated_statistics(x)
        num_features, num_states, counts, marginal_counts = checked
        if self._fixed_num_features and self.num_features != num_features:
            raise ValueError("Integer Chow-Liu statistic feature count contradicts accumulator")
        if self._fixed_num_states and self.num_states != num_states:
            raise ValueError("Integer Chow-Liu statistic state count contradicts accumulator")
        self.num_features = num_features
        self.num_states = num_states
        self.counts = counts.copy()
        self.marginal_counts = marginal_counts.copy()

        return self

    def scale(self, c: float) -> "IntegerChowLiuTreeAccumulator":
        """Scale all accumulated Chow-Liu sufficient statistics in place."""
        checked_scale = _nonnegative_scalar(
            c,
            label="Integer Chow-Liu scale",
        )
        if self.counts is not None:
            self.counts *= checked_scale
        if self.marginal_counts is not None:
            self.marginal_counts *= checked_scale
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """No-op kept for interface consistency (keyed merging is not supported for IntegerChowLiuTreeAccumulator).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        if self.keys is None:
            return
        if self.keys in stats_dict:
            stats_dict[self.keys].combine(self.value())
        else:
            stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """No-op kept for interface consistency (keyed merging is not supported for IntegerChowLiuTreeAccumulator).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "IntegerChowLiuTreeDataEncoder":
        """Return a data encoder for accumulated integer Chow-Liu tree observations."""
        return IntegerChowLiuTreeDataEncoder(
            self.num_features,
            (None if self.num_states is None or self.num_features is None else [self.num_states] * self.num_features),
        )


class IntegerChowLiuTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer Chow-Liu tree accumulators."""

    def __init__(
        self,
        num_features: int | None = None,
        num_states: int | None = None,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a factory for integer Chow-Liu tree accumulators.

        Args:
            num_features (Optional[int]): Number of features. If None, set from data on first update.
            num_states (Optional[int]): Number of states per feature. If None, set from data.
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.num_features = num_features
        self.num_states = num_states
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerChowLiuTreeAccumulator":
        """Return a new integer Chow-Liu tree accumulator."""
        return IntegerChowLiuTreeAccumulator(
            self.num_features,
            self.num_states,
            self.keys,
            self.name,
        )


class IntegerChowLiuTreeEstimator(ParameterEstimator):
    """Estimator for the IntegerChowLiuTreeDistribution. Learns the dependency tree with the Chow-Liu algorithm."""

    def __init__(
        self,
        num_features: int | None = None,
        num_states: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: Any | None = None,
        keys: str | None = None,
        name: str | None = None,
        enumeration_item_budget: int | None = _DEFAULT_ENUMERATION_ITEM_BUDGET,
    ):
        """Create an estimator for integer Chow-Liu tree distributions.

        Args:
            num_features (Optional[int]): Number of features. If None, set from data.
            num_states (Optional[int]): Number of states per feature. If None, set from data.
            pseudo_count (Optional[float]): Smoothing count spread over the marginal and joint counts.
            suff_stat (Optional[Any]): Kept for interface consistency (unused).
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional name assigned to estimated distributions.

        """
        self.num_features = _positive_integer_or_none(
            num_features,
            label="Integer Chow-Liu estimator feature count",
        )
        self.num_states = _positive_integer_or_none(
            num_states,
            label="Integer Chow-Liu estimator state count",
        )
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _nonnegative_scalar(
                pseudo_count,
                label="Integer Chow-Liu pseudo-count",
            )
        )
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        if enumeration_item_budget is None:
            self.enumeration_item_budget = None
        else:
            self.enumeration_item_budget = _exact_integer(
                enumeration_item_budget,
                label="Integer Chow-Liu enumeration item budget",
            )
            if self.enumeration_item_budget < 0:
                raise ValueError("Integer Chow-Liu enumeration item budget must be non-negative")

    def accumulator_factory(self):
        """Return an accumulator factory configured from this estimator."""
        return IntegerChowLiuTreeAccumulatorFactory(
            self.num_features,
            self.num_states,
            self.keys,
            self.name,
        )

    def estimate(self, nobs, suff_stat):
        """Estimate an IntegerChowLiuTreeDistribution from sufficient statistics via the Chow-Liu algorithm.

        Pairwise mutual information is computed from the (optionally smoothed) joint and marginal
        counts, a maximum mutual information spanning tree is extracted, and conditional densities
        are computed along the tree rooted at feature 0.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of
                states, pairwise joint counts, and marginal counts.

        Returns:
            IntegerChowLiuTreeDistribution object.

        """
        checked = _validated_statistics(suff_stat)
        num_features, num_states, counts, marginal_counts = checked
        if self.num_features is not None and num_features != self.num_features:
            raise ValueError("Integer Chow-Liu statistic feature count contradicts estimator")
        if self.num_states is not None and num_states != self.num_states:
            raise ValueError("Integer Chow-Liu statistic state count contradicts estimator")
        total_weight = float(marginal_counts[0].sum())
        if total_weight <= 0.0:
            raise ValueError("Cannot estimate Integer Chow-Liu without positive weight")
        if nobs is not None:
            _nonnegative_scalar(
                nobs,
                label="Integer Chow-Liu observation count",
            )

        mi_mat = np.zeros((num_features, num_features))

        pseudo_count = self.pseudo_count if self.pseudo_count is not None else 0.0
        pseudo_count_adj0 = pseudo_count / num_states
        pseudo_count_adj1 = pseudo_count / (num_states * num_states)

        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                if pseudo_count > 0:
                    n_ij = counts[i, j, :, :].sum()
                    joint_ij = (counts[i, j, :, :] + pseudo_count_adj1) / (n_ij + pseudo_count)
                    marg_i = (marginal_counts[i, :] + pseudo_count_adj0) / (n_ij + pseudo_count)
                    marg_j = (marginal_counts[j, :] + pseudo_count_adj0) / (n_ij + pseudo_count)
                    indep_ij = np.outer(marg_i, marg_j)
                else:
                    joint_ij = counts[i, j, :, :].copy()
                    indep_ij = np.outer(marginal_counts[i, :], marginal_counts[j, :])

                    joint_ij_sum = joint_ij.sum()
                    indep_ij_sum = indep_ij.sum()

                    if joint_ij_sum > 0:
                        joint_ij /= joint_ij_sum
                    if indep_ij_sum > 0:
                        indep_ij /= indep_ij_sum

                good = np.bitwise_and(joint_ij > 0, indep_ij > 0)

                if good.sum() > 0:
                    mi_val = (joint_ij[good] * (np.log(joint_ij[good]) - np.log(indep_ij[good]))).sum()
                    mi_mat[i, j] = 1.0 + mi_val

                else:
                    mi_mat[i, j] = 1.0

        cost_mat = np.abs(mi_mat.max() - mi_mat)
        cost_mat[mi_mat > 0] += 1.0
        cost_mat[mi_mat == 0] = 0

        span_tree = minimum_spanning_tree(cost_mat)

        root_node = 0
        feature_order, deps = breadth_first_order(span_tree, root_node, directed=False, return_predecessors=True)

        deps = [deps[i] for i in feature_order]
        tmats = [None] * num_features

        with np.errstate(divide="ignore"):
            root_marginal = marginal_counts[root_node, :] + pseudo_count_adj0
            tmats[0] = np.log(root_marginal / (root_marginal.sum()))
            deps[0] = None

            for i in range(1, num_features):
                n = feature_order[i]
                p = deps[i]

                if p < n:
                    tmat = counts[p, n, :, :]
                else:
                    tmat = counts[n, p, :, :].T

                tmat = tmat + pseudo_count_adj1
                tmat_sum = np.sum(tmat, axis=1, keepdims=True)
                tmat_sum[tmat_sum == 0] = 1.0
                tmat /= tmat_sum

                tmats[i] = np.log(tmat)

        return IntegerChowLiuTreeDistribution(
            deps,
            tmats,
            feature_order=feature_order,
            name=self.name,
            keys=self.keys,
            pseudo_count=self.pseudo_count,
            enumeration_item_budget=self.enumeration_item_budget,
        )


class IntegerChowLiuTreeDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of fixed-length integer vector observations."""

    def __init__(
        self,
        num_features: int | None = None,
        domain_sizes: Sequence[int] | None = None,
    ) -> None:
        self.num_features = _positive_integer_or_none(
            num_features,
            label="Integer Chow-Liu encoder feature count",
        )
        if domain_sizes is None:
            self.domain_sizes = None
        else:
            self.domain_sizes = tuple(
                _positive_integer_or_none(
                    size,
                    label=f"Integer Chow-Liu feature {feature} domain size",
                )
                for feature, size in enumerate(domain_sizes)
            )
            if any(size is None for size in self.domain_sizes):
                raise ValueError("Integer Chow-Liu encoder domain sizes must be fixed")
            if self.num_features is not None and len(self.domain_sizes) != self.num_features:
                raise ValueError("Integer Chow-Liu encoder domain sizes must match feature count")
            if self.num_features is None:
                self.num_features = len(self.domain_sizes)

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerChowLiuTreeDataEncoder(num_features=%r, domain_sizes=%r)" % (
            self.num_features,
            self.domain_sizes,
        )

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an integer Chow-Liu tree data encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is an IntegerChowLiuTreeDataEncoder instance, else False.

        """
        return (
            isinstance(other, IntegerChowLiuTreeDataEncoder)
            and self.num_features == other.num_features
            and self.domain_sizes == other.domain_sizes
        )

    def seq_encode(self, x: list[int] | np.ndarray) -> np.ndarray:
        """Encode a sequence of N integer vectors for vectorized functions.

        Args:
            x (Union[List[int], np.ndarray]): Sequence of N fixed-length integer vectors.

        Returns:
            2-d numpy array of ints with N rows and num_features columns.

        """
        return _validated_integer_rows(
            x,
            num_features=self.num_features,
            domain_sizes=self.domain_sizes,
        )

    def row_count(self, x: list[int] | np.ndarray) -> int:
        """Return the validated number of encoded observations."""
        return int(self.seq_encode(x).shape[0])
