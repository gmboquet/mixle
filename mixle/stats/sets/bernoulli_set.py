"""Bernoulli set distributions over finite hashable supports.

Data type: Sequence[Any]: An observation is a set (any iterable of distinct hashable values) drawn from a
finite support S = {s_1,s_2,....,s_N}. Let x be a random subset of S. Each element s_k is included in x
independently with probability

    p_k = P(s_k is in x) , k = 1,2,...,N,

so the density of an observed set x is

    p(x) = prod_{s_k in x} p_k * prod_{s_k not in x} (1-p_k).

A comment on estimation: Note that probability of an element s_k belonging to the set is 0 if we do not encounter any
elements an observation sequence. For this reason, we need not state the support of the state-space in estimation.

"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
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
from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma


class BernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli set distribution: each support element is included in an observed set independently."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for Bernoulli-set generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Bernoulli set."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="bernoulli_set",
            distribution_type=cls,
            parameters=(
                ParameterSpec("pmap", constraint="simplex_map"),
                ParameterSpec("min_prob", constraint="unit_interval", differentiable=False),
            ),
            statistics=(
                StatisticSpec("inclusion_counts", kind="count_map"),
                StatisticSpec("total_weight"),
            ),
            support="finite_hashable_set",
            differentiable=False,
        )

    def __init__(
        self,
        pmap: dict[Any, float] = MISSING,
        min_prob: float = 1.0e-128,
        name: str | None = None,
        keys: str | None = None,
        prob_map: dict[Any, float] = MISSING,
        prior: SequenceEncodableProbabilityDistribution | None = None,
        posteriors: dict[Any, tuple[float, float]] | None = None,
    ) -> None:
        """Create a Bernoulli set distribution.

        Args:
            pmap (Dict[Any, float]): Maps values to probabilities.
            min_prob (float): Minimum probability for numerical stability in log prob calculations.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        Attributes:
            key (Optional[str]): Key for sharing sufficient statistics.
            name (Optional[str]): Optional distribution name.
            pmap (Dict[Any, float]): Maps elements in support to probabilities.
            required (Set): An observation must contain this subset of elements. Else, return probability 0.0.
            nlog_sum (float): Normalizing term for computing numerically stable likelihood.
            log_dmap (Dict[Any, float]):Map from elements to their corrected log probability of inclusion in the set.
            min_prob (float): Minimum probability for elements. Corrects for prob = 0.
            num_required (int): Number of required elements in a subset. Corrected if min_prob was non-zero.

        """
        pmap = coalesce_alias("pmap", pmap, "prob_map", prob_map, default=MISSING)
        self.keys = keys
        self.name = name
        self.pmap = pmap
        self.required = set()
        self.nlog_sum = 0.0
        self.log_dmap = dict()

        if min_prob == 0:
            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = 0.0
                    self.required.add(k)
                elif v == 0.0:
                    self.log_dmap[k] = -np.inf
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv
            self.min_prob = 0.0
            self.num_required = len(self.required)

        else:
            min_pv = np.log(min_prob)
            min_nv = np.log1p(-min_prob)

            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = min_nv - min_pv
                    self.nlog_sum += min_pv
                elif v == 0.0:
                    self.log_dmap[k] = min_pv - min_nv
                    self.nlog_sum += min_nv
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv

            self.min_prob = min_prob
            self.num_required = 0

        self.set_prior(prior, posteriors)

    def set_prior(
        self,
        prior: SequenceEncodableProbabilityDistribution | None,
        posteriors: dict[Any, tuple[float, float]] | None = None,
    ) -> None:
        """Attach a (per-element) Beta prior and precompute conjugate-prior expectations.

        With a shared Beta(a, b) prior on each element's inclusion probability ``p_k`` this
        caches the digamma expectations so that ``expected_log_density`` evaluates the
        variational Bayes term ``E_q[log p(x | p)]`` via ``E[log p_k] = digamma(a_k) -
        digamma(a_k + b_k)`` and ``E[log(1 - p_k)] = digamma(b_k) - digamma(a_k + b_k)``.
        When ``posteriors`` (element -> (a_k, b_k)) is supplied, those per-element posterior
        Beta parameters are used; otherwise the shared prior parameters are broadcast over
        the support in ``pmap``. Any other prior (including ``None``) leaves the distribution
        a plain point model.

        Args:
            prior: A shared ``BetaDistribution`` prior, or ``None``.
            posteriors: Optional per-element posterior Beta parameters carried forward from a
                conjugate update.
        """
        self.prior = prior
        self.posteriors = posteriors
        if isinstance(prior, BetaDistribution):
            self.has_conj_prior = True
            a0, b0 = prior.get_parameters()
            self._elp = dict()
            self._elnp = dict()
            for k in self.pmap.keys():
                if posteriors is not None and k in posteriors:
                    a, b = posteriors[k]
                else:
                    a, b = a0, b0
                dab = digamma(a + b)
                self._elp[k] = digamma(a) - dab
                self._elnp[k] = digamma(b) - dab
            self._default_elp = digamma(a0) - digamma(a0 + b0)
            self._default_elnp = digamma(b0) - digamma(a0 + b0)
            self._nelnp_sum = sum(self._elnp.values())
        else:
            self.has_conj_prior = False
            self._elp = None
            self._elnp = None
            self._default_elp = None
            self._default_elnp = None
            self._nelnp_sum = 0.0

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the shared Beta prior (or ``None``)."""
        return self.prior

    def get_posteriors(self) -> dict[Any, tuple[float, float]] | None:
        """Return the per-element posterior Beta parameters (or ``None``)."""
        return self.posteriors

    def expected_log_density(self, x: Sequence[Any]) -> float:
        """Variational expectation ``E_q[log p(x | p)]`` under the per-element Beta prior.

        Sums ``E[log p_k]`` over elements present in x plus ``E[log(1 - p_k)]`` over the
        remaining support. Falls back to the plug-in ``log_density(x)`` when no conjugate
        prior is attached.
        """
        if not self.has_conj_prior:
            return self.log_density(x)
        rv = self._nelnp_sum
        for u in x:
            elp = self._elp.get(u, self._default_elp)
            elnp = self._elnp.get(u, self._default_elnp)
            rv += elp - elnp
        return rv

    def seq_expected_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        sz, idx, val_map_inv, xs = x
        diff_loc = np.asarray(
            [self._elp.get(u, self._default_elp) - self._elnp.get(u, self._default_elnp) for u in val_map_inv],
            dtype=np.float64,
        )
        rv = np.bincount(idx, weights=diff_loc[xs], minlength=sz)
        rv += self._nelnp_sum
        return rv

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(sorted(self.pmap.items(), key=lambda t: t[0]))
        s2 = repr(self.min_prob)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return "BernoulliSetDistribution(dict(%s), min_prob=%s, name=%s, keys=%s)" % (s1, s2, s3, s4)

    def density(self, x: Sequence[Any]) -> float:
        """Density of the Bernoulli set distribution at observed set x.

        See log_density() for details.

        Args:
            x (Sequence[Any]): Observed set of distinct elements from the support of pmap.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Any]) -> float:
        """Log-density of the Bernoulli set distribution at observed set x.

        Sums log(p_k / (1-p_k)) over the elements present in x, plus the constant
        sum_k log(1-p_k). Returns -inf if x is missing a required element (an element
        with p_k = 1 when min_prob is 0).

        Args:
            x (Sequence[Any]): Observed set of distinct elements from the support of pmap.

        Returns:
            Log-density at observation x.

        """
        if not self.required.issubset(x):
            return -np.inf
        rv = 0.0
        for v in x:
            rv += self.log_dmap[v]
        return self.nlog_sum + rv

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        sz, idx, val_map_inv, xs = x

        dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)

        rv = np.bincount(idx, weights=dlog_loc[xs], minlength=sz)
        rv += self.nlog_sum

        if self.num_required != 0:
            required_loc = np.isin(val_map_inv, list(self.required))
            req_cnt = np.bincount(idx, weights=required_loc[xs], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf

        return rv

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded object-valued sets."""
        sz, idx, val_map_inv, xs = x
        rv = engine.zeros(sz) + float(self.nlog_sum)

        if len(xs) > 0:
            dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)
            rv = engine.index_add(rv, engine.asarray(idx), engine.asarray(dlog_loc)[engine.asarray(xs)])

        if self.num_required != 0:
            req_cnt = engine.zeros(sz)
            if len(xs) > 0:
                required_loc = np.isin(val_map_inv, list(self.required))
                req_cnt = engine.index_add(
                    req_cnt, engine.asarray(idx), engine.asarray(np.asarray(required_loc[xs], dtype=np.float64))
                )
            rv = engine.where(req_cnt != float(self.num_required), engine.asarray(np.full(sz, -np.inf)), rv)

        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["BernoulliSetDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Bernoulli-set parameters for shared label support."""
        labels = tuple(dists[0].pmap.keys())
        min_prob = float(dists[0].min_prob)
        if any(tuple(dist.pmap.keys()) != labels or float(dist.min_prob) != min_prob for dist in dists):
            raise ValueError("Stacked BernoulliSetDistribution components require shared support/min_prob.")
        log_d = np.asarray([[dist.log_dmap[label] for dist in dists] for label in labels], dtype=np.float64)
        required = np.asarray([[label in dist.required for dist in dists] for label in labels], dtype=np.float64)
        num_required = np.asarray([dist.num_required for dist in dists], dtype=np.float64)
        return {
            "__pysp_component_axis__": {"log_d": 1, "nlog_sum": 0, "required": 1, "num_required": 0},
            "labels": labels,
            "log_d": engine.asarray(log_d),
            "nlog_sum": engine.asarray(np.asarray([dist.nlog_sum for dist in dists], dtype=np.float64)),
            "required": engine.asarray(required),
            "num_required": engine.asarray(num_required),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[int, np.ndarray, np.ndarray, np.ndarray], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of Bernoulli-set log densities."""
        sz, idx, val_map_inv, xs = x
        label_to_idx = {label: i for i, label in enumerate(params["labels"])}
        mapped = np.asarray([label_to_idx.get(label, -1) for label in val_map_inv], dtype=np.int64)
        good = mapped >= 0
        safe = np.clip(mapped, 0, max(0, len(params["labels"]) - 1))
        rv = engine.zeros((sz, int(params["num_components"]))) + params["nlog_sum"][None, :]

        if len(xs) > 0:
            log_dloc = params["log_d"][engine.asarray(safe), :]
            log_dloc = engine.where(engine.asarray(good)[:, None], log_dloc, engine.asarray(-np.inf))
            rv = engine.index_add(rv, engine.asarray(idx), log_dloc[engine.asarray(xs), :])

        if np.any(np.asarray(engine.to_numpy(params["num_required"])) != 0):
            req_cnt = engine.zeros((sz, int(params["num_components"])))
            if len(xs) > 0:
                required_loc = params["required"][engine.asarray(safe), :]
                required_loc = engine.where(engine.asarray(good)[:, None], required_loc, engine.asarray(0.0))
                req_cnt = engine.index_add(req_cnt, engine.asarray(idx), required_loc[engine.asarray(xs), :])
            rv = engine.where(req_cnt != params["num_required"][None, :], engine.asarray(-np.inf), rv)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[int, np.ndarray, np.ndarray, np.ndarray], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[tuple[dict[Any, float], float], ...]:
        """Return per-component legacy ``(count_map, total_weight)`` statistics."""
        sz, idx, val_map_inv, xs = x
        xx = engine.asarray(xs)
        ww = engine.asarray(weights)
        count_rows = []
        if len(xs) > 0:
            row_weights = ww[engine.asarray(idx)]
            zero_rows = row_weights * engine.asarray(0.0)
            for value_index in range(len(val_map_inv)):
                mask = xx == engine.asarray(value_index)
                count_rows.append(engine.sum(engine.where(mask[:, None], row_weights, zero_rows), axis=0))
            counts = np.asarray(engine.to_numpy(engine.stack(count_rows, axis=0)), dtype=np.float64)
        else:
            counts = np.zeros((0, int(params["num_components"])), dtype=np.float64)
        totals = np.asarray(engine.to_numpy(engine.sum(ww, axis=0)), dtype=np.float64)
        return tuple(
            ({val_map_inv[j]: float(counts[j, component]) for j in range(len(val_map_inv))}, float(totals[component]))
            for component in range(int(params["num_components"]))
        )

    def sampler(self, seed: int | None = None) -> "BernoulliSetSampler":
        """Create a sampler for this Bernoulli set distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            BernoulliSetSampler: Sampler bound to this distribution.

        """
        return BernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BernoulliSetEstimator":
        """Create a BernoulliSetEstimator, passing pmap as suff_stat if pseudo_count is given.

        Args:
            pseudo_count (Optional[float]): Used to re-weight the distribution's pmap in estimation.

        Returns:
            BernoulliSetEstimator: Estimator configured with this distribution's prior settings.

        """
        if pseudo_count is None:
            return BernoulliSetEstimator(min_prob=self.min_prob, name=self.name, prior=self.prior)
        else:
            return BernoulliSetEstimator(
                min_prob=self.min_prob,
                pseudo_count=pseudo_count,
                suff_stat=self.pmap,
                name=self.name,
                prior=self.prior,
            )

    def dist_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Return a data encoder for Bernoulli set observations."""
        return BernoulliSetDataEncoder()

    def enumerator(self) -> "BernoulliSetEnumerator":
        """Returns BernoulliSetEnumerator iterating subsets of the support in descending probability order."""
        return BernoulliSetEnumerator(self)


class BernoulliSetEnumerator(DistributionEnumerator):
    """Enumerates subsets of the pmap support in descending probability order."""

    def __init__(self, dist: BernoulliSetDistribution) -> None:
        """Enumerates subsets of dist.pmap's keys in descending probability order.

        Membership is independent per element: including element k contributes log_dmap[k]
        to the log-density and excluding it contributes 0 (relative to the nlog_sum offset).
        Each element therefore yields a sorted two-choice stream, and subsets are enumerated
        with a best-first product search. Elements with p_k = 0 are exclude-only; required
        elements (p_k = 1 with min_prob = 0) are include-only. Each subset corresponds to a
        unique inclusion-flag tuple, so deduplication is exact. Raises EnumerationError when
        a membership probability lies outside [0, 1], which breaks the independent-inclusion
        form of the log-density.

        Args:
            dist (BernoulliSetDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        vals = list(dist.pmap.keys())
        log_d = np.asarray([dist.log_dmap[v] for v in vals], dtype=np.float64)
        if np.any(np.isnan(log_d)) or np.any(np.isposinf(log_d)) or not np.isfinite(dist.nlog_sum):
            raise EnumerationError(
                dist,
                reason="membership probabilities must lie in [0, 1] for the "
                "independent-inclusion log-density to be well-defined",
            )
        streams = []
        for v, d in zip(vals, log_d):
            if v in dist.required:
                choices = [(True, 0.0)]
            elif d == -np.inf:
                choices = [(False, 0.0)]
            elif d > 0.0:
                choices = [(True, float(d)), (False, 0.0)]
            else:
                choices = [(False, 0.0), (True, float(d))]
            streams.append(BufferedStream(iter(choices)))

        def combine(flags: tuple[bool, ...]) -> list[Any]:
            return [v for v, f in zip(vals, flags) if f]

        self._product = ProductEnumerator(streams, combine=combine, offset=float(dist.nlog_sum))

    def __next__(self) -> tuple[list[Any], float]:
        return next(self._product)


class BernoulliSetSampler(DistributionSampler):
    """Draw random sets from a BernoulliSetDistribution."""

    def __init__(self, dist: BernoulliSetDistribution, seed: int | None = None) -> None:
        """Create a sampler for a Bernoulli set distribution.

        Args:
            dist (BernoulliSetDistribution): Distribution to sample from.
            seed (Optional[int]): Set seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (BernoulliSetDistribution): Distribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> Sequence[Any] | list[Sequence[Any]]:
        """Draw iid set observations from the BernoulliSetDistribution instance.

        Args:
            size (Optional[int]): Number of sets to draw. If None, a single set is returned.

        Returns:
            A list of included elements if size is None, else a list of such lists of length size.

        """
        if size is not None:
            retval = [[] for i in range(size)]
            for k, v in self.dist.pmap.items():
                for i in np.flatnonzero(self.rng.rand(size) <= v):
                    retval[i].append(k)
            return retval

        else:
            retval = []
            for k, v in self.dist.pmap.items():
                if self.rng.rand() <= v:
                    retval.append(k)
            return retval


class BernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for per-element inclusion counts from observed sets."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator for Bernoulli-set sufficient statistics.

        Args:
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
            pmap (Dict[Any, float]): Dictionary mapping values to set-inclusion probabilities.
            tot_sum (float): Weighted observation count.
            key (Optional[str]): Key for merging sufficient statistics.
        """
        self.pmap = defaultdict(float)
        self.tot_sum = 0.0
        self.keys = keys

    def update(self, x: Sequence[Any], weight: float, estimate: BernoulliSetDistribution | None) -> None:
        """Add weight to the inclusion count of each element of the observed set x.

        Args:
            x (Sequence[Any]): Observed set of distinct elements.
            weight (float): Weight for the observation.
            estimate (Optional[BernoulliSetDistribution]): Unused (kept for protocol consistency).

        """
        for u in x:
            self.pmap[u] += weight
        self.tot_sum += weight

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState | None) -> None:
        """Initialize the accumulator with a weighted observation. Calls update().

        Args:
            x (Sequence[Any]): Observed set of distinct elements.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Unused (kept for protocol consistency).

        """
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[int, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: BernoulliSetDistribution | None,
    ) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (Optional[BernoulliSetDistribution]): Unused (kept for protocol consistency).

        """
        sz, idx, val_map_inv, xs = x
        agg_cnt = np.bincount(xs, weights[idx])

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += weights.sum()

    def seq_update_engine(
        self,
        x: tuple[int, np.ndarray, np.ndarray, np.ndarray],
        weights: Any,
        estimate: BernoulliSetDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident accumulation of per-element inclusion counts (numpy or torch).

        The weighted element histogram is reduced on the active engine; the object-keyed count
        dict is host bookkeeping. Matches seq_update.
        """
        sz, idx, val_map_inv, xs = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w_eng = engine.asarray(weights_np)

        if len(xs) > 0:
            agg_cnt = np.asarray(
                engine.to_numpy(
                    engine.bincount(
                        engine.asarray(np.asarray(xs, dtype=np.int64)),
                        weights=w_eng[np.asarray(idx, dtype=np.int64)],
                        minlength=len(val_map_inv),
                    )
                ),
                dtype=np.float64,
            )
        else:
            agg_cnt = np.zeros(len(val_map_inv), dtype=np.float64)

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += float(engine.to_numpy(engine.sum(w_eng)))

    def seq_initialize(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray, rng: np.random.RandomState
    ) -> None:
        """Vectorized initialization of sufficient statistics. Calls seq_update().

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (np.random.RandomState): Unused (kept for protocol consistency).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[dict[Any, float], float]) -> "BernoulliSetAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            This BernoulliSetAccumulator.

        """
        for k, v in suff_stat[0].items():
            self.pmap[k] += v
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> tuple[dict[Any, float], float]:
        """Returns the sufficient statistics: (inclusion counts by element, total weight)."""
        return dict(self.pmap), self.tot_sum

    def from_value(self, x: tuple[dict[Any, float], float]) -> "BernoulliSetAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            This BernoulliSetAccumulator.

        """
        self.pmap = x[0]
        self.tot_sum = x[1]
        return self

    def acc_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Return a data encoder for Bernoulli set observations."""
        return BernoulliSetDataEncoder()


class BernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Bernoulli set accumulators."""

    def __init__(self, keys: str | None = None) -> None:
        """Create a factory for Bernoulli set accumulators.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics.

        """
        self.keys = keys

    def make(self) -> "BernoulliSetAccumulator":
        """Return a new Bernoulli set accumulator."""
        return BernoulliSetAccumulator(self.keys)


class BernoulliSetEstimator(ParameterEstimator):
    """Estimate Bernoulli set distributions from aggregated sufficient statistics."""

    def __init__(
        self,
        min_prob: float = 1.0e-128,
        pseudo_count: float | None = None,
        suff_stat: dict[Any, float] | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for Bernoulli set distributions.

        Args:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.min_prob = min_prob
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def accumulator_factory(self) -> "BernoulliSetAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        return BernoulliSetAccumulatorFactory(self.keys)

    def model_log_density(self, model: "BernoulliSetDistribution") -> float:
        """Log-density of the model's per-element probabilities under the shared Beta prior.

        This is the global ELBO term: with a shared Beta(a, b) prior the contribution is the
        sum over support elements of ``log Beta(p_k; a, b)``. Returns 0.0 when no conjugate
        prior is attached.
        """
        if self.has_conj_prior:
            return float(sum(self.prior.log_density(p) for p in model.pmap.values()))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[dict[Any, float], float]) -> "BernoulliSetDistribution":
        """Closed-form per-element Beta conjugate update returning the posterior-mode estimate.

        With a shared Beta(a, b) prior and per-element weighted inclusion count ``v`` out of
        ``tot`` total weighted sets, the per-element posterior is Beta(a + v, b + tot - v) and
        the returned probability is the corresponding posterior mode. The per-element
        posteriors are carried forward as ``posteriors`` on the fitted model.
        """
        obs_cnt, tot_cnt = suff_stat
        a0, b0 = self.prior.get_parameters()
        pmap = dict()
        posteriors = dict()
        for k, v in obs_cnt.items():
            post_a = a0 + v
            post_b = b0 + (tot_cnt - v)
            posteriors[k] = (post_a, post_b)
            pmap[k] = _beta_posterior_mode(a0, b0, v, tot_cnt)
        return BernoulliSetDistribution(
            pmap, min_prob=self.min_prob, name=self.name, prior=self.prior, posteriors=posteriors
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[dict[Any, float], float]) -> "BernoulliSetDistribution":
        """Estimate a BernoulliSetDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Unused (kept for protocol consistency).
            suff_stat (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            BernoulliSetDistribution object.

        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        if self.pseudo_count is not None and self.suff_stat is not None:
            keys = set(suff_stat[0].keys())
            keys.update(self.suff_stat.keys())

            pmap = {
                k: (self.suff_stat.get(k, 0.0) * self.pseudo_count + suff_stat[0].get(k, 0.0))
                / (self.pseudo_count + suff_stat[1])
                for k in keys
            }

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            cnt = float(p + suff_stat[1])
            pmap = {k: (v + (p / 2.0)) / cnt for k, v in suff_stat[0].items()}

        else:
            if suff_stat[1] != 0:
                pmap = {k: v / suff_stat[1] for k, v in suff_stat[0].items()}
            else:
                pmap = {k: 0.5 for k in suff_stat[0].keys()}

        return BernoulliSetDistribution(pmap, min_prob=self.min_prob, name=self.name, keys=self.keys)


class BernoulliSetDataEncoder(DataSequenceEncoder):
    """BernoulliSetDataEncoder for encoding sequences of iid observations."""

    def __str__(self) -> str:
        return "BernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BernoulliSetDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Encode iid Bernoulli-set observations for vectorized ``seq_*`` methods.

        The returned tuple contains:
            rv[0] (int): Number of observed sets.
            rv[1] (np.ndarray): Numpy array of integer indices for flattened array of values.
            rv[2] (np.ndarray): Numpy array of unique values. (dtype is object).
            rv[3] (np.ndarray): Numpy array of val_map (rv[2]) integer indices for flattened array of values.

        Args:
            x (Sequence[Sequence[Any]]): A sequence of iid Bernoulli set observations.

        Returns:
            See 'rv' above.

        """
        idx = []
        xs = []

        for i in range(len(x)):
            idx.extend([i] * len(x[i]))
            xs.extend(x[i])

        val_map, xs = np.unique(xs, return_inverse=True)

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        return len(x), idx, val_map, xs


def _beta_posterior_mode(beta_a: float, beta_b: float, obs_cnt: float, tot_cnt: float) -> float:
    """Per-element Beta posterior-mode inclusion probability in plain ``[0, 1]`` form.

    Mirrors the branches of ``mixle.bstats.setdist.bernoulli_beta_posterior_mode`` exactly,
    but returns the probability directly (the bstats routine encodes probabilities above one
    half as ``p - 1``). With prior Beta(beta_a, beta_b) and weighted inclusion count
    ``obs_cnt`` out of ``tot_cnt`` total weighted sets, let ``a = (beta_a - 1) + obs_cnt`` and
    ``b = (beta_b - 1) - obs_cnt + tot_cnt`` be the (mode-shifted) posterior counts.

    Args:
        beta_a (float): Prior Beta ``a`` parameter.
        beta_b (float): Prior Beta ``b`` parameter.
        obs_cnt (float): Weighted inclusion count for the element.
        tot_cnt (float): Total weighted set count.

    Returns:
        Posterior-mode inclusion probability in ``[0, 1]``.
    """
    a = (beta_a - 1) + obs_cnt
    b = (beta_b - 1) - obs_cnt + tot_cnt

    if a > 0 and b > 0:
        # Beta posterior mode for a>0, b>0 is a / (a + b).
        return a / (a + b)
    elif a == 0 and b == 0:
        return 0.5
    elif b > a:
        return 0.0
    else:
        return 1.0
