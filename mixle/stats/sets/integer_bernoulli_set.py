"""Integer Bernoulli-set distributions over finite integer supports.

Let ``S = {0, 1, ..., N-1}`` be a finite integer support and let ``X`` be a random subset of ``S``.
The Bernoulli-set distribution gives each integer an independent inclusion probability ``p_k``.
The probability of an observed subset ``x`` is

    p(x) = prod_{k in x} p_k * prod_{k not in x} (1 - p_k).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import MISSING, coalesce_alias


class IntegerBernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution over finite sets of integer-valued Bernoulli outcomes."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the generic table-kernel capabilities for integer-set likelihoods."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Bernoulli-set statistics."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="integer_bernoulli_set",
            distribution_type=cls,
            parameters=(
                ParameterSpec("log_pvec", constraint="log_unit_interval_vector"),
                ParameterSpec("log_nvec", constraint="optional_log_unit_interval_vector", differentiable=False),
            ),
            statistics=(
                StatisticSpec("inclusion_counts"),
                StatisticSpec("total_weight"),
            ),
            support="finite_integer_set",
            differentiable=False,
        )

    def __init__(
        self,
        log_pvec: Sequence[float] | np.ndarray,
        log_nvec: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a Bernoulli set distribution on integer support ``[0, len(pvec))``.

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        Attributes:
            name (Optional[str]): Optional distribution name.
            log_pvec (np.ndarray): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            log_dvec (np.ndarray): Normalized probability for each integer value.
            log_nsum (float): Sum of normalized probabilities used for easily adding unobserved (missing) integer
                values in an observation.
            key (Optional[str]): Key for sharing sufficient statistics.

        """

        num_vals = len(log_pvec)
        self.name = name
        self.num_vals = num_vals
        self.log_pvec = np.asarray(log_pvec, dtype=np.float64).copy()
        self.keys = keys

        with np.errstate(divide="ignore"):
            if log_nvec is None:
                log_nvec = np.log1p(-np.exp(self.log_pvec))
                self.log_nvec = None
                self.log_dvec = self.log_pvec - log_nvec
                self.log_nsum = np.sum(log_nvec[np.isfinite(log_nvec)])
            else:
                self.log_nvec = np.asarray(log_nvec, dtype=np.float64)
                self.log_dvec = self.log_pvec - self.log_nvec
                self.log_nsum = np.sum(self.log_nvec[np.isfinite(self.log_nvec)])

        # An element with p_k = 1 is *required*: its log_dvec entry is +inf (log_nvec = -inf,
        # excluded from log_nsum). Treat it as forced membership (mirrors BernoulliSetDistribution):
        # zero contribution when present, -inf when an observation omits it -- never +inf.
        self.required = np.where(~np.isfinite(self.log_dvec) & (self.log_dvec > 0))[0]
        self.num_required = int(self.required.shape[0])
        if self.num_required:
            self.log_dvec = self.log_dvec.copy()
            self.log_dvec[self.required] = 0.0

    def __str__(self) -> str:
        s1 = repr(list(self.log_pvec))
        s2 = repr(None if self.log_nvec is None else list(self.log_nvec))
        s3 = repr(self.name)
        return "IntegerBernoulliSetDistribution(%s, log_nvec=%s, name=%s)" % (s1, s2, s3)

    def density(self, x: Sequence[int] | np.ndarray) -> float:
        """Return the probability density or mass at a single observation."""
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[int] | np.ndarray) -> float:
        """Return the log-density or log-mass at a single observation."""
        xx = np.asarray(x, dtype=int)
        if self.num_required and not np.all(np.isin(self.required, xx)):
            return -np.inf
        return np.sum(self.log_dvec[xx]) + self.log_nsum

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        sz, idx, xs = x
        rv = np.zeros(sz, dtype=np.float64)
        rv += np.bincount(idx, weights=self.log_dvec[xs], minlength=sz)
        rv += self.log_nsum
        if self.num_required:
            req_loc = np.isin(xs, self.required)
            req_cnt = np.bincount(idx[req_loc], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf
        return rv

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral log-density for encoded integer Bernoulli-set observations."""
        sz, idx, xs = x
        rv = engine.zeros(sz) + engine.asarray(self.log_nsum)
        if len(xs):
            log_dvec = engine.asarray(self.log_dvec)
            rv = rv + engine.bincount(engine.asarray(idx), weights=log_dvec[engine.asarray(xs)], minlength=sz)
        if self.num_required:
            req_cnt = engine.zeros(sz)
            if len(xs):
                required_loc = np.isin(np.asarray(xs), self.required).astype(np.float64)
                req_cnt = engine.bincount(engine.asarray(idx), weights=engine.asarray(required_loc), minlength=sz)
            rv = engine.where(req_cnt != float(self.num_required), engine.asarray(np.full(sz, -np.inf)), rv)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerBernoulliSetDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer Bernoulli-set parameters for shared support size."""
        num_vals = int(dists[0].num_vals)
        if any(int(dist.num_vals) != num_vals for dist in dists):
            raise ValueError("Stacked IntegerBernoulliSetDistribution components require shared support size.")
        required = np.stack([np.isin(np.arange(num_vals), dist.required).astype(np.float64) for dist in dists], axis=1)
        return {
            "__pysp_component_axis__": {"log_dvec": 1, "log_nsum": 0, "required": 1, "num_required": 0},
            "num_vals": num_vals,
            "log_dvec": engine.asarray(np.stack([dist.log_dvec for dist in dists], axis=1)),
            "log_nsum": engine.asarray(np.asarray([dist.log_nsum for dist in dists], dtype=np.float64)),
            "required": engine.asarray(required),
            "num_required": engine.asarray(np.asarray([dist.num_required for dist in dists], dtype=np.float64)),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[int, np.ndarray, np.ndarray], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of integer Bernoulli-set log densities."""
        sz, idx, xs = x
        rv = engine.zeros((sz, int(params["num_components"]))) + params["log_nsum"][None, :]
        if len(xs):
            contrib = params["log_dvec"][engine.asarray(xs), :]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)
        if "num_required" in params and np.any(np.asarray(engine.to_numpy(params["num_required"])) != 0):
            req_cnt = engine.zeros((sz, int(params["num_components"])))
            if len(xs):
                req_loc = params["required"][engine.asarray(xs), :]
                req_cnt = engine.index_add(req_cnt, engine.asarray(idx), req_loc)
            rv = engine.where(req_cnt != params["num_required"][None, :], engine.asarray(-np.inf), rv)
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[int, np.ndarray, np.ndarray], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(inclusion_counts, total_weight)`` statistics."""
        sz, idx, xs = x
        ww = engine.asarray(weights)
        num_vals = int(params["num_vals"])
        if len(xs):
            row_weights = ww[engine.asarray(idx)]
            zero_rows = row_weights * engine.asarray(0.0)
            rows = []
            rel = engine.asarray(xs)
            for value_index in range(num_vals):
                mask = rel == engine.asarray(value_index)
                rows.append(engine.sum(engine.where(mask[:, None], row_weights, zero_rows), axis=0))
            pcnt = engine.stack(rows, axis=1)
        else:
            pcnt = engine.zeros((int(params["num_components"]), num_vals))
        return pcnt, engine.sum(ww, axis=0)

    def sampler(self, seed: int | None = None) -> "IntegerBernoulliSetSampler":
        """Return a sampler for drawing observations from this distribution."""
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerBernoulliSetEstimator":
        """Return an estimator for fitting this distribution from data."""
        return IntegerBernoulliSetEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return IntegerBernoulliSetDataEncoder()

    def enumerator(self) -> "IntegerBernoulliSetEnumerator":
        """Returns IntegerBernoulliSetEnumerator iterating subsets in descending probability order."""
        return IntegerBernoulliSetEnumerator(self)


class IntegerBernoulliSetEnumerator(DistributionEnumerator):
    """Enumerate integer subsets in descending probability order."""

    def __init__(self, dist: IntegerBernoulliSetDistribution) -> None:
        """Enumerates subsets of {0,...,num_vals-1} in descending probability order.

        Membership is independent per integer: including k contributes log_dvec[k] to the
        log-density and excluding it contributes 0 (relative to the log_nsum offset). Each
        integer therefore yields a sorted two-choice stream, and subsets are enumerated with
        a best-first product search. Integers with p_k = 0 are exclude-only; required integers
        (p_k = 1) are include-only and contribute 0 to the log-density.

        Args:
            dist (IntegerBernoulliSetDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        required = {int(k) for k in dist.required}
        streams = []
        for k in range(dist.num_vals):
            d = dist.log_dvec[k]
            if k in required:
                choices = [(True, 0.0)]
            elif d == -np.inf:
                choices = [(False, 0.0)]
            elif d > 0.0:
                choices = [(True, float(d)), (False, 0.0)]
            else:
                choices = [(False, 0.0), (True, float(d))]
            streams.append(BufferedStream(iter(choices)))

        def combine(flags: tuple[bool, ...]) -> list[int]:
            return [k for k, f in enumerate(flags) if f]

        self._product = ProductEnumerator(streams, combine=combine, offset=float(dist.log_nsum))

    def __next__(self) -> tuple[list[int], float]:
        return next(self._product)


class IntegerBernoulliSetSampler(DistributionSampler):
    """Sample finite integer subsets by independent Bernoulli inclusion draws."""

    def __init__(self, dist: IntegerBernoulliSetDistribution, seed: int | None = None) -> None:
        """Create a sampler for an integer Bernoulli-set distribution.

        Args:
            dist (IntegerBernoulliSetDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (IntegerBernoulliSetDistribution): Distribution to sample from.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> list[Sequence[int]] | Sequence[int]:
        """Draw one subset or ``size`` iid subsets."""
        if size is None:
            log_u = np.log(self.rng.rand(self.dist.num_vals))
            return list(np.flatnonzero(log_u <= self.dist.log_pvec))
        else:
            rv = []
            for i in range(size):
                log_u = np.log(self.rng.rand(self.dist.num_vals))
                rv.append(list(np.flatnonzero(log_u <= self.dist.log_pvec)))
            return rv


class IntegerBernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate per-integer inclusion counts and total observation weight."""

    def __init__(self, num_vals: int, keys: str | None = None) -> None:
        """Create an accumulator for integer Bernoulli-set sufficient statistics.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            pcnt (np.ndarray): Used for aggregating weighted counts of integers.
            key (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.
            tot_sum (float): Sum of weights for observations.

        """
        self.pcnt = np.zeros(num_vals, dtype=np.float64)
        self.keys = keys
        self.num_vals = num_vals
        self.tot_sum = 0.0

    def update(
        self, x: Sequence[int] | np.ndarray, weight: float, estimate: IntegerBernoulliSetDistribution | None
    ) -> None:
        """Update inclusion counts from one weighted subset."""
        xx = np.asarray(x, dtype=int)
        self.pcnt[xx] += weight
        self.tot_sum += weight

    def initialize(self, x: Sequence[int] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize inclusion counts from one weighted subset."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[int, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: IntegerBernoulliSetDistribution | None,
    ) -> None:
        """Update inclusion counts from encoded subsets and observation weights."""
        sz, idx, xs = x
        agg_cnt = np.bincount(xs, weights=weights[idx])
        n = len(agg_cnt)
        self.pcnt[:n] += agg_cnt
        self.tot_sum += weights.sum()

    def seq_update_engine(
        self,
        x: tuple[int, np.ndarray, np.ndarray],
        weights: Any,
        estimate: IntegerBernoulliSetDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident accumulation of per-integer inclusion counts (numpy or torch).

        The weighted integer histogram is reduced on the active engine; the fixed-size count
        vector is host bookkeeping. Matches seq_update.
        """
        sz, idx, xs = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w_eng = engine.asarray(weights_np)

        xsv = np.asarray(xs)
        minlen = int(xsv.max()) + 1 if xsv.size > 0 else 0
        if xsv.size > 0:
            agg_cnt = np.asarray(
                engine.to_numpy(
                    engine.bincount(
                        engine.asarray(xsv.astype(np.int64)),
                        weights=w_eng[np.asarray(idx, dtype=np.int64)],
                        minlength=minlen,
                    )
                ),
                dtype=np.float64,
            )
            n = len(agg_cnt)
            self.pcnt[:n] += agg_cnt

        self.tot_sum += float(engine.to_numpy(engine.sum(w_eng)))

    def seq_initialize(
        self, x: tuple[int, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize inclusion counts from encoded subsets."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "IntegerBernoulliSetAccumulator":
        """Merge inclusion counts and total observation weight."""
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return inclusion counts and total observation weight."""
        return self.pcnt, self.tot_sum

    def from_value(self, x: tuple[np.ndarray, float]) -> "IntegerBernoulliSetAccumulator":
        """Restore inclusion counts and total observation weight."""
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def scale(self, c: float) -> "IntegerBernoulliSetAccumulator":
        """Scale inclusion counts and total observation weight by a constant."""
        self.pcnt *= c
        self.tot_sum *= c
        return self

    def acc_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Return the encoder compatible with Bernoulli-set sufficient statistics."""
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for integer Bernoulli-set sufficient statistics."""

    def __init__(self, num_vals: int, keys: str | None = None) -> None:
        """IntegerBernoulliSetAccumulatorFactory for creating IntegerBernoulliSetAccumulator objects.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        """
        self.keys = keys
        self.num_vals = num_vals

    def make(self) -> "IntegerBernoulliSetAccumulator":
        """Create an empty integer Bernoulli-set accumulator."""
        return IntegerBernoulliSetAccumulator(self.num_vals, keys=self.keys)


class IntegerBernoulliSetEstimator(ParameterEstimator):
    """Estimate per-integer Bernoulli inclusion probabilities from aggregate counts."""

    def __init__(
        self,
        num_vals: int = MISSING,
        min_prob: float = 1.0e-128,
        pseudo_count: float | None = None,
        suff_stat: np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        num_values: int = MISSING,
    ) -> None:
        """Estimate integer Bernoulli set distributions from aggregated sufficient statistics.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in range of set dist.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.

        Attributes:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Optional name assigned to estimated distributions.
            min_prob (float): Minimum probability for an integer in range of set dist.

        """
        self.num_vals = coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING)
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> "IntegerBernoulliSetAccumulatorFactory":
        """Return a factory for integer Bernoulli-set sufficient-statistic accumulators."""
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(self, nobs: float | None, suff_stat: np.ndarray | None = None) -> "IntegerBernoulliSetDistribution":
        """Estimate an integer Bernoulli-set distribution from inclusion-count statistics."""
        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = np.multiply(self.suff_stat, self.pseudo_count)
            p1 = np.multiply(np.subtract(1.0, self.suff_stat), self.pseudo_count)
            tsum = np.log(suff_stat[1] + self.pseudo_count)
            log_pvec = np.log(suff_stat[0] + p0) - tsum
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + p1) - tsum

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            log_c = np.log(suff_stat[1] + p)
            log_pvec = np.log(suff_stat[0] + (p / 2.0)) - log_c
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + (p / 2.0)) - log_c

        else:
            if suff_stat[1] == 0:
                # no observations: fall back to p = 0.5 per element (these are log-probabilities)
                log_pvec = np.zeros(self.num_vals, dtype=np.float64) + np.log(0.5)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64) + np.log(0.5)

            elif self.min_prob > 0:
                log_pvec = np.log(np.maximum(suff_stat[0] / suff_stat[1], self.min_prob))
                log_nvec = np.log(np.maximum((suff_stat[1] - suff_stat[0]) / suff_stat[1], self.min_prob))

            else:
                pvec = suff_stat[0] / suff_stat[1]
                nvec = (suff_stat[1] - suff_stat[0]) / suff_stat[1]

                is_zero = pvec == 0
                is_one = nvec == 0

                log_pvec = np.zeros(self.num_vals, dtype=np.float64)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64)

                log_pvec[~is_zero] = np.log(pvec[~is_zero])
                log_pvec[is_zero] = -np.inf
                log_nvec[~is_one] = np.log(nvec[~is_one])
                log_nvec[is_one] = -np.inf

        return IntegerBernoulliSetDistribution(log_pvec, log_nvec, name=self.name, keys=self.keys)


class IntegerBernoulliSetDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer Bernoulli-set observations."""

    def __str__(self) -> str:
        return "IntegerBernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntegerBernoulliSetDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> tuple[int, np.ndarray, np.ndarray]:
        """Encode sequences of iid observations for vectorized calculations.

        Returns 'rv':
            rv[0] (int): Total number of observations.
            rv[1] (np.ndarray): Index for flattened values of observations.
            rv[2] (np.ndarray): Flattened numpy array of integer values.

        Args:
            x (Sequence[Sequence[int]]): Sequence of integer set observations.

        Returns:
            See above for details.

        """
        idx = []
        xs = []
        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        return len(x), idx, xs
