"""Integer-categorical distributions over consecutive bounded supports.

The observation type is ``int``. A distribution is parameterized by ``min_val``
and a probability vector ``p_vec`` whose entries correspond to values
``min_val, min_val + 1, ..., min_val + len(p_vec) - 1``. Values outside that
range have zero probability.
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.enumeration.algorithms import QuantizedCrossIndex, QuantizedEnumerationIndex
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.discrete.categorical import CategoricalFisherView
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma


class IntegerCategoricalFisherView(CategoricalFisherView):
    """Fisher view for bounded integer-categorical one-hot statistics."""

    # Marker for fisher._structured_values_matrix's int-categorical fast path (decoupled from import).
    _fisher_integer_categorical = True

    def __init__(self, dist: Any) -> None:
        min_val = int(getattr(dist, "min_val", getattr(dist, "min_index", 0)))
        max_val = int(getattr(dist, "max_val", getattr(dist, "max_index", min_val)))
        probs = np.asarray(dist.p_vec if hasattr(dist, "p_vec") else dist.prob_vec, dtype=np.float64)
        keys = list(range(min_val, max_val + 1))
        super().__init__(dist, keys, probs)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        x = np.asarray(enc_data, dtype=np.int64)
        min_val = int(getattr(self.dist, "min_val", getattr(self.dist, "min_index", 0)))
        cols = x - min_val
        mat = np.zeros((len(x), len(self.keys)), dtype=np.float64)
        rows = np.arange(len(x), dtype=np.int64)
        good = (cols >= 0) & (cols < len(self.keys))
        mat[rows[good], cols[good]] = 1.0
        return mat


class IntegerCategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """Categorical distribution over a bounded integer range."""

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for integer-categorical scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the symbolic declaration for bounded integer-categorical probabilities."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="integer_categorical",
            distribution_type=cls,
            parameters=(
                ParameterSpec("min_val", constraint="integer", differentiable=False),
                ParameterSpec("p_vec", constraint="simplex_vector"),
            ),
            statistics=(
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
                StatisticSpec("count_vec", kind="count_vector"),
            ),
            support="bounded_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                sufficient_statistics_from_params=cls.exp_family_sufficient_statistics_from_params,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure_from_params=cls.exp_family_base_measure_from_params,
                # T(x) is the one-hot category indicator and eta = log(p_vec); A = 0, h(x) = 1 on
                # the support [min_val, min_val+K). The base mask depends on the per-component
                # min_val/K, so fixed_base=False. eta has -inf entries when any p_i = 0, which makes
                # the generic <eta, T> dot form NaN via 0*-inf for OTHER categories; runtime_scoring
                # is therefore False so scoring keeps the safe indexing backend path while
                # to_exponential_family still exposes the canonical map (valid where p > 0).
                fixed_base=False,
                runtime_scoring=False,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return raw values; category-aware one-hot statistics come from ``..._from_params``."""
        return (engine.asarray(x),)

    @staticmethod
    def exp_family_sufficient_statistics_from_params(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the one-hot category indicator ``T(x)`` of shape ``(n, K)`` (zeros off support)."""
        vals = np.asarray(engine.to_numpy(engine.asarray(x))).reshape(-1)
        min_val = int(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        idx = np.rint(vals - min_val).astype(np.int64)
        in_support = (idx >= 0) & (idx < k)
        onehot = np.zeros((vals.shape[0], k), dtype=np.float64)
        rows = np.nonzero(in_support)[0]
        onehot[rows, idx[in_support]] = 1.0
        return (engine.asarray(onehot),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the natural parameter ``eta = log(p_vec)`` (one entry per category)."""
        return (engine.log(engine.asarray(params["p_vec"])),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the log partition ``A = 0`` (normalization is carried by ``eta = log p``)."""
        return engine.asarray(0.0)

    @staticmethod
    def exp_family_base_measure_from_params(x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return ``log h(x) = 0`` on the support ``[min_val, min_val+K)`` and ``-inf`` outside it."""
        vals = engine.asarray(x)
        min_val = engine.asarray(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        v = vals - min_val
        good = (v >= 0) & (v < k)
        return engine.where(good, engine.asarray(0.0), engine.asarray(-np.inf))

    def __init__(
        self,
        min_val: int,
        p_vec: list[float] | np.ndarray = MISSING,
        name: str | None = None,
        prob_vec: list[float] | np.ndarray = MISSING,
        prior: Optional["SequenceEncodableProbabilityDistribution"] = None,
    ) -> None:
        """Create a categorical distribution over consecutive integers.

        Args:
            min_val: Minimum value of the integer categorical support.
            p_vec: Probability vector for values ``min_val`` through
                ``min_val + len(p_vec) - 1``.
            name: Optional distribution name.
            prior (Optional): Conjugate parameter prior over the probability vector. A
                :class:`~mixle.stats.bayes.dirichlet.DirichletDistribution` or
                :class:`~mixle.stats.bayes.symmetric_dirichlet.SymmetricDirichletDistribution` enables the Bayesian /
                variational machinery (``expected_log_density`` and the conjugate posterior update);
                ``None`` (default) is a plain point model.

        Attributes:
            p_vec: Probability vector, normalized by the constructor.
            min_val: Minimum supported integer value.
            max_val: Maximum supported integer value.
            log_p_vec: Elementwise log probabilities.
            num_vals: Number of integer values in the support.
        """
        p_vec = coalesce_alias("p_vec", p_vec, "prob_vec", prob_vec, default=MISSING)
        with np.errstate(divide="ignore"):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.name = name
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-style representation of the integer categorical distribution."""
        s1 = str(self.min_val)
        s2 = repr(list(self.p_vec))
        s3 = repr(self.name)

        return "IntegerCategoricalDistribution(%s, %s, name=%s)" % (s1, s2, s3)

    def get_parameters(self) -> np.ndarray:
        """Return the probability vector p_vec (lets it be scored by a Dirichlet conjugate prior)."""
        return self.p_vec

    def get_prior(self) -> Optional["SequenceEncodableProbabilityDistribution"]:
        """Return the conjugate parameter prior over the probability vector (or None)."""
        return self.prior

    def set_prior(self, prior: Optional["SequenceEncodableProbabilityDistribution"]) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a Dirichlet(alpha) (or SymmetricDirichlet(alpha)) prior over the probability vector this
        caches the variational expected log-probabilities
        E[log p_k] = digamma(alpha_k) - digamma(sum_k alpha_k) so that
        ``expected_log_density(x) = E[log p_{x - min_val}] - log(1 + default_value)``. Any other prior
        (including ``None``) leaves the distribution a plain point model.
        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        self.prior = prior

        if isinstance(prior, DirichletDistribution):
            cpp = prior.get_parameters()
            if np.ndim(cpp) == 0:
                cpp = np.ones(self.num_vals) * cpp
            else:
                cpp = np.asarray(cpp, dtype=float)
            self.conj_prior_params = cpp
            self.expected_nparams = digamma(cpp) - digamma(np.sum(cpp))
            self.has_conj_prior = True
        elif isinstance(prior, SymmetricDirichletDistribution):
            cpp = np.ones(self.num_vals) * prior.get_parameters()
            self.conj_prior_params = cpp
            self.expected_nparams = digamma(cpp) - digamma(np.sum(cpp))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x: int) -> float:
        """Variational expectation E_q[log p(x)] under the (symmetric) Dirichlet prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.expected_nparams is None:
            return self.log_density(x)

        if (x < self.min_val) or (x > self.max_val):
            return -np.inf

        idx = int(x - self.min_val)
        return float(self.expected_nparams[idx])

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.expected_nparams is None:
            return self.seq_log_density(x)

        v = x - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(-np.inf)
        rv[u] = self.expected_nparams[v[u]]
        return rv

    def density(self, x: int) -> float:
        """Evaluate the density of the integer categorical at observation x.

        p_mat(x_mat=x) = p_vec[x] if x in support [min_val, max_val], else 0.0.

        Args:
            x (int): Integer value.

        Returns:
            Density at x.

        """
        return zero if x < self.min_val or x > self.max_val else self.p_vec[x - self.min_val]

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the integer categorical at observation x.

        log_p(x_mat=x) = log_p_vec[x] if x in support [min_val, max_val], else -np.inf.

        Args:
            x (int): Integer value.

        Returns:
            Log-density at x.

        """
        xi = int(x)
        # Accept integer-valued floats (e.g. a count total summed as float64); reject non-integers
        # and out-of-support values. Indexing log_p_vec with a raw float raises, hence the cast.
        if xi != x or xi < self.min_val or xi > self.max_val:
            return -inf
        return self.log_p_vec[xi - self.min_val]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of IntegerCategorical log_density() for sequence encoded iid observations x.

        Args:
            x (np.ndarray[int]): Sequence encoded iid observation of integer categorical distribution.

        Returns:
            Numpy array of floats containing log_density() evaluated at each observation in x.

        """
        x = np.asarray(x)
        rv = np.full(x.shape[0], -np.inf)
        if np.issubdtype(x.dtype, np.integer):
            v = x - self.min_val
            u = np.bitwise_and(v >= 0, v < self.num_vals)
            rv[u] = self.log_p_vec[v[u]]
            return rv
        # Float input (e.g. count totals summed as float64): accept integer-valued entries by casting
        # the index to int, and reject non-integer values (indexing log_p_vec with a float raises).
        xi = np.rint(x).astype(np.int64)
        v = xi - self.min_val
        u = (v >= 0) & (v < self.num_vals) & (np.abs(x - xi) < 1.0e-9)
        rv[u] = self.log_p_vec[v[u]]
        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, min_val: int, log_p_vec: Any, engine: Any) -> Any:
        """Engine-neutral integer-categorical log-density from explicit parameters."""
        v = x - engine.asarray(min_val)
        good = (v >= 0) & (v < len(log_p_vec))
        safe_v = engine.clip(v, 0, len(log_p_vec) - 1)
        return engine.where(good, log_p_vec[safe_v], engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        return self.backend_log_density_from_params(xx, self.min_val, engine.asarray(self.log_p_vec), engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerCategoricalDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer-categorical parameters for a homogeneous mixture kernel."""
        min_val = dists[0].min_val
        num_vals = dists[0].num_vals
        if any(d.min_val != min_val or d.num_vals != num_vals for d in dists):
            raise ValueError("Stacked IntegerCategoricalDistribution components require shared support.")
        return {
            "__pysp_component_axis__": {"log_p": 1},
            "min_val": min_val,
            "num_vals": num_vals,
            "log_p": engine.asarray(np.stack([d.log_p_vec for d in dists], axis=1)),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer-categorical log densities."""
        xx = engine.asarray(x)
        v = xx - engine.asarray(params["min_val"])
        good = (v >= 0) & (v < params["num_vals"])
        safe_v = engine.clip(v, 0, params["num_vals"] - 1)
        rv = params["log_p"][safe_v, :]
        return engine.where(good[:, None], rv, engine.asarray(-np.inf))

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(min_val, count_vec)`` statistics."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        rel = xx - engine.asarray(params["min_val"])
        rows = []
        for i in range(int(params["num_vals"])):
            mask = rel == engine.asarray(i)
            rows.append(engine.sum(ww * mask[:, None], axis=0))
        if rows:
            count_mat = engine.stack(rows, axis=1)
        else:
            count_mat = engine.zeros((tuple(getattr(ww, "shape", (0, 0)))[1], 0))
        min_vals = engine.asarray(np.full(int(tuple(getattr(ww, "shape", (0, 0)))[1]), int(params["min_val"])))
        return min_vals, count_mat

    def support_size(self) -> int:
        """Number of integer values in the range."""
        return int(self.num_vals)

    def to_fisher(self, **kwargs):
        """Return the integer-categorical one-hot Fisher view."""
        if hasattr(self, "p_vec") or hasattr(self, "prob_vec"):
            return IntegerCategoricalFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "IntegerCategoricalSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional random seed.
        """
        return IntegerCategoricalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerCategoricalEstimator":
        """Return an estimator initialized from this distribution.

        When ``pseudo_count`` is provided, the distribution's probabilities are
        used as prior sufficient statistics during estimation.

        Args:
            pseudo_count: Weight assigned to the distribution's current
                probability vector during estimation.
        """
        if pseudo_count is None:
            return IntegerCategoricalEstimator(name=self.name, prior=self.prior)

        else:
            return IntegerCategoricalEstimator(
                pseudo_count=pseudo_count, suff_stat=(self.min_val, self.p_vec), name=self.name, prior=self.prior
            )

    def dist_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Return the encoder for iid integer categorical observations."""
        return IntegerCategoricalDataEncoder()

    def enumerator(self) -> "IntegerCategoricalEnumerator":
        """Return IntegerCategoricalEnumerator iterating the support in descending probability order."""
        return IntegerCategoricalEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the finite integer support."""
        items = [(self.min_val + i, float(lp)) for i, lp in enumerate(self.log_p_vec)]
        return QuantizedEnumerationIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over integer categorical ranges."""
        dists = [self] + list(others)
        if any(not isinstance(dist, IntegerCategoricalDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)

        lo = min(dist.min_val for dist in dists)
        hi = max(dist.max_val for dist in dists)
        items = []
        for value in range(lo, hi + 1):
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two integer categorical ranges."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class IntegerCategoricalEnumerator(DistributionEnumerator):
    """Enumerator over bounded integer support in descending probability order."""

    def __init__(self, dist: IntegerCategoricalDistribution) -> None:
        """Enumerates the support [min_val, max_val] in descending probability order.

        Zero-probability entries of p_vec are skipped.

        Args:
            dist (IntegerCategoricalDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        order = np.argsort(-dist.log_p_vec, kind="stable")
        self._order = [int(i) for i in order if dist.p_vec[i] > 0.0]
        self._pos = 0

    def __next__(self) -> tuple[int, float]:
        if self._pos >= len(self._order):
            raise StopIteration
        i = self._order[self._pos]
        self._pos += 1
        return (self.dist.min_val + i, float(self.dist.log_p_vec[i]))


class IntegerCategoricalSampler(DistributionSampler):
    """Sampler for bounded integer-categorical values."""

    def __init__(self, dist: "IntegerCategoricalDistribution", seed: int | None = None) -> None:
        """Create a sampler for an integer-categorical distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            dist: Distribution sampled by this object.
            rng: Random state used for reproducible draws.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | list[int]:
        """Draw iid samples from the integer-categorical distribution.

        Args:
            size: Number of samples. ``None`` returns one integer; a positive
                value returns a list of that length.
        """
        if size is None:
            return self.rng.choice(range(self.dist.min_val, self.dist.max_val + 1), p=self.dist.p_vec)

        else:
            return list(self.rng.choice(range(self.dist.min_val, self.dist.max_val + 1), p=self.dist.p_vec, size=size))


class IntegerCategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for weighted counts over a bounded integer support."""

    def __init__(self, min_val: int | None = None, max_val: int | None = None, keys: str | None = None) -> None:
        """Create an accumulator for weighted integer-category counts.

        If ``min_val`` and ``max_val`` are not provided, the observed data define
        the support as accumulation proceeds.

        Args:
            min_val: Optional minimum support value.
            max_val: Optional maximum support value.
            keys: Optional merge key for sufficient-statistic aggregation.

        Attributes:
            min_val: Minimum support value seen or configured.
            max_val: Maximum support value seen or configured.
            count_vec: Weighted counts aligned to ``[min_val, max_val]``.
            keys: Optional merge key.
        """
        self.min_val = min_val
        self.max_val = max_val

        if min_val is not None and max_val is not None:
            self.count_vec = vec.zeros(max_val - min_val + 1)

        else:
            self.count_vec = None

        self.keys = keys

    def update(self, x: int, weight: float, estimate: Optional["IntegerCategoricalDistribution"]) -> None:
        """Update sufficient statistics with one weighted observation.

        If the observed value falls outside the current support, the count
        vector is expanded and existing counts are realigned.

        Args:
            x: Integer observation.
            weight: Observation weight.
            estimate: Accepted for accumulator API consistency.
        """

        if self.count_vec is None:
            self.min_val = x
            self.max_val = x
            self.count_vec = np.asarray([weight])

        elif self.max_val < x:
            temp_vec = self.count_vec
            self.max_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[: len(temp_vec)] = temp_vec
            self.count_vec[x - self.min_val] += weight

        elif self.min_val > x:
            temp_vec = self.count_vec
            temp_diff = self.min_val - x
            self.min_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[temp_diff:] = temp_vec
            self.count_vec[x - self.min_val] += weight

        else:
            self.count_vec[x - self.min_val] += weight

    def initialize(self, x: int, weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with one weighted observation.

        Args:
            x: Integer observation.
            weight: Observation weight.
            rng: Accepted for accumulator API consistency.

        Returns:
            None.

        """
        return self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of IntegerCategoricalAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Args:
            x (np.ndarray[int]): Sequence encoded iid observations of integer categorical distribution.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        return self.seq_update(x, weights, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: Optional["IntegerCategoricalDistribution"]
    ) -> None:
        """Vectorized update of IntegerCategoricalAccumulator sufficient statistics with sequence encoded iid
            observations x.

        Note: Determines the range (support) of integer categorical from the sequence encoded data.

        Args:
            x (np.ndarray[int]): Sequence encoded iid observations of integer categorical distribution.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[IntegerCategoricalDistribution]): Previous estimate of IntegerCategoricalDistribution.

        Returns:
            None.

        """
        min_x = x.min()
        max_x = x.max()

        loc_cnt = np.bincount(x - min_x, weights=weights)

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def seq_update_engine(
        self, x: np.ndarray, weights: Any, estimate: Optional["IntegerCategoricalDistribution"], engine: Any
    ) -> None:
        """Engine-resident accumulation: the weighted value histogram is reduced on the active
        engine (numpy or torch); the dynamic support range is host bookkeeping. Matches seq_update.
        """
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        xv = np.asarray(x)
        min_x = int(xv.min())
        max_x = int(xv.max())

        idx = engine.asarray((xv - min_x).astype(np.int64))
        loc_cnt = np.asarray(
            engine.to_numpy(engine.bincount(idx, weights=engine.asarray(weights_np), minlength=max_x - min_x + 1)),
            dtype=np.float64,
        )

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def combine(self, suff_stat: tuple[int | None, np.ndarray | None]) -> "IntegerCategoricalAccumulator":
        """Merge another ``(min_val, count_vec)`` sufficient statistic.

        Supports are aligned before counts are added, so accumulators built on
        different observed ranges can be combined safely.
        """
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> tuple[int, np.ndarray]:
        """Return ``(min_val, count_vec)`` sufficient statistics."""
        return self.min_val, self.count_vec

    def from_value(self, x: tuple[int, np.ndarray]) -> "IntegerCategoricalAccumulator":
        """Replace accumulator state from a ``(min_val, count_vec)`` statistic."""
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def scale(self, c: float) -> "IntegerCategoricalAccumulator":
        """Scale count vector while preserving integer support metadata."""
        if self.count_vec is not None:
            self.count_vec *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Aggregate member sufficient statistics with sufficient statistics of objects with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to corresponding sufficient stats.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())

            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set member sufficient statistics to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to corresponding sufficient stats.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "IntegerCategoricalDataEncoder":
        """Return the encoder associated with this accumulator."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer-categorical count accumulators."""

    def __init__(self, min_val: int | None = None, max_val: int | None = None, keys: str | None = None) -> None:
        """Factory for integer-categorical accumulators.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            min_val (Optional[int]): Minimum value of integer categorical, if None estimated from data.
            max_val (Optional[int]): Maximum value of integer categorical, if None estimated from data.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys

    def make(self) -> "IntegerCategoricalAccumulator":
        """Return a fresh integer categorical accumulator with this factory's bounds and keys."""
        return IntegerCategoricalAccumulator(self.min_val, self.max_val, self.keys)


class IntegerCategoricalEstimator(ParameterEstimator):
    """Estimator for bounded integer-categorical probability vectors."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[int, np.ndarray] | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for bounded integer-categorical distributions.

        Provide either explicit ``min_val``/``max_val`` bounds or prior
        sufficient statistics. Prior sufficient statistics are ``(min_val,
        prob_vec)``, where ``prob_vec`` is aligned to consecutive integer values
        beginning at ``min_val``.

        Args:
            min_val: Optional minimum support value.
            max_val: Optional maximum support value.
            pseudo_count: Optional weight for prior sufficient statistics.
            suff_stat: Optional prior statistic ``(min_val, prob_vec)``.
            name: Optional estimator and fitted-distribution name.
            keys: Optional merge key for accumulator statistics.

        Attributes:
            min_val: Minimum support value, when fixed.
            max_val: Maximum support value, when fixed.
            pseudo_count: Weight for prior sufficient statistics.
            suff_stat: Optional prior statistic.
            name: Optional estimator name.
            keys: Optional merge key.
        """
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.prior = prior
        self._set_has_conj_prior(prior)

    def _set_has_conj_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        self.has_conj_prior = isinstance(prior, (DirichletDistribution, SymmetricDirichletDistribution))

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the conjugate parameter prior over the probability vector (or None)."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the conjugate parameter prior over the probability vector."""
        self.prior = prior
        self._set_has_conj_prior(prior)

    def model_log_density(self, model: "IntegerCategoricalDistribution") -> float:
        """Log-density of the model probability vector under the (symmetric) Dirichlet prior."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p_vec))
        return 0.0

    def accumulator_factory(self) -> "IntegerCategoricalAccumulatorFactory":
        """Return an accumulator factory configured from this estimator's support and keys.

        Note: If min_val and max_val are BOTH not None, these values are passed to IntegerCategoricalAccumulatorFactory.
        Else, they are obtained from member variable suff_stat. One of these conditions must be satisfied.

        Returns:

        """
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        return IntegerCategoricalAccumulatorFactory(min_val, max_val, self.keys)

    def _estimate_conjugate(self, suff_stat: tuple[int, np.ndarray]) -> "IntegerCategoricalDistribution":
        """Dirichlet MAP estimate (counts + alpha - 1, clamped at the simplex boundary, posterior mean
        when degenerate) carrying the posterior Dirichlet forward as the new prior."""
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        min_val, count_vec = suff_stat
        alpha0 = self.prior.get_parameters()
        if np.ndim(alpha0) == 0:
            alpha0 = np.ones(len(count_vec)) * alpha0
        else:
            alpha0 = np.asarray(alpha0, dtype=float)

        posterior_params = count_vec + alpha0

        # Dirichlet MAP sits on the boundary when alpha_k + n_k < 1
        num = np.maximum(count_vec + (alpha0 - 1), 0.0)
        norm_const = np.sum(num)

        if norm_const > 0:
            prob_vec = num / norm_const
        else:
            # fall back to the posterior mean when the MAP is degenerate
            prob_vec = posterior_params / np.sum(posterior_params)

        hyper_posterior = DirichletDistribution(posterior_params)

        return IntegerCategoricalDistribution(min_val, prob_vec, name=self.name, prior=hyper_posterior)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[int, np.ndarray] | None
    ) -> "IntegerCategoricalDistribution":
        """Estimate an integer-categorical distribution from weighted counts.

        ``nobs`` is accepted for estimator API consistency but is not used.
        ``suff_stat`` is ``(min_val, count_vec)``. When ``pseudo_count`` and a
        prior statistic are present, the estimate combines observed counts with
        the weighted prior probability vector.
        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(
                suff_stat[0], (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs, name=self.name
            )

        elif self.pseudo_count is not None and self.min_val is not None and self.max_val is not None:
            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(
                min_val, (count_vec + pseudo_count_per_level) / adjusted_nobs, name=self.name
            )

        elif self.pseudo_count is not None and self.suff_stat is not None:
            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            return IntegerCategoricalDistribution(min_val, count_vec / (count_vec.sum()), name=self.name)

        else:
            return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name)


class IntegerCategoricalDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer-categorical observations."""

    def __str__(self) -> str:
        """Return the integer categorical encoder's display name."""
        return "IntegerCategoricalDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerCategoricalDataEncoder, False is else."""
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(self, x: list[int] | np.ndarray) -> np.ndarray:
        """Sequence encode iid integer categorical observations for "seq_" functions.

        Args:
            x (Union[List[int], np.ndarray]): Assumed int observations of integer categorical.

        Returns:
            Numpy array of integers.

        """
        return np.asarray(x, dtype=int)
