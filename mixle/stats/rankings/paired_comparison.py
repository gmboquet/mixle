"""Paired-comparison models beyond plain Bradley-Terry: Gaussian pairwise and ties.

Three canonical models that complete the pairwise-comparison family:

* :class:`ThurstoneMostellerDistribution` -- the Gaussian (probit) pairwise model
  ``P(i beats j) = Phi((mu_i - mu_j) / sqrt(2))``; the pairwise peer of :class:`ThurstoneDistribution`
  and the Gaussian counterpart of :class:`BradleyTerryDistribution`. Observation: ``(winner, loser)``.
* :class:`DavidsonDistribution` -- Bradley-Terry with **ties** (Davidson 1970): a draw has probability
  ``nu * sqrt(w_i w_j) / (w_i + w_j + nu sqrt(w_i w_j))``. Observation: ``(i, j, outcome)`` with
  ``outcome`` in ``{0: i wins, 1: j wins, 2: tie}``.
* :class:`RaoKupperDistribution` -- Bradley-Terry with ties via a threshold ``nu >= 1`` (Rao-Kupper
  1967): ``P(i beats j) = w_i / (w_i + nu w_j)``. Same ``(i, j, outcome)`` observation.

All three treat the compared pair as a uniform draw over the ``C(K, 2)`` unordered pairs, making them
proper distributions over (canonicalized) comparison outcomes. Worths/utilities are identified up to a
shift and stored mean-zero. Fitting maximizes the comparison log-likelihood from the win/tie count
matrices (the sufficient statistic).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy import optimize
from scipy.special import ndtr, ndtri

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_SQRT2 = math.sqrt(2.0)


# =================================================================================================
# Thurstone-Mosteller: Gaussian (probit) pairwise comparisons, no ties
# =================================================================================================
class ThurstoneMostellerDistribution(SequenceEncodableProbabilityDistribution):
    """Gaussian/probit paired-comparison model ``P(i beats j) = Phi((mu_i - mu_j) / sqrt(2))``."""

    @classmethod
    def compute_capabilities(cls):
        """Return backend capabilities for the probit pairwise likelihood."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Probit pairwise likelihood is numpy-native.",
        )

    def __init__(self, mu: Sequence[float] | np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        m = np.asarray(mu, dtype=float)
        if m.ndim != 1 or m.size < 2 or not np.all(np.isfinite(m)):
            raise ValueError("mu must be a finite length-K vector with K >= 2.")
        self.mu = m - m.mean()
        self.dim = m.size
        self.log_pairs = math.log(self.dim * (self.dim - 1) / 2.0)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ThurstoneMostellerDistribution(%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.mu]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: tuple[int, int]) -> float:
        """Return the probability mass of one winner-loser pair."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[int, int]) -> float:
        """Return the log probability of one winner-loser pair."""
        return float(self.seq_log_density(np.asarray([x], dtype=np.int64))[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Score encoded winner-loser pairs."""
        z = (self.mu[x[:, 0]] - self.mu[x[:, 1]]) / _SQRT2
        return np.log(np.clip(ndtr(z), 1e-300, 1.0)) - self.log_pairs

    def sampler(self, seed: int | None = None) -> ThurstoneMostellerSampler:
        """Return a sampler for winner-loser comparisons."""
        return ThurstoneMostellerSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ThurstoneMostellerEstimator:
        """Return the least-squares probit paired-comparison estimator."""
        return ThurstoneMostellerEstimator(dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> PairDataEncoder:
        """Return the encoder for winner-loser pair data."""
        return PairDataEncoder(dim=self.dim)


class ThurstoneMostellerSampler(DistributionSampler):
    """Sampler for Thurstone-Mosteller winner-loser comparisons."""

    def __init__(self, dist: ThurstoneMostellerDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> tuple[int, int]:
        k = self.dist.dim
        i, j = self.rng.choice(k, size=2, replace=False)
        p = 0.5 * (1.0 + math.erf((self.dist.mu[i] - self.dist.mu[j]) / 2.0))  # Phi((mu_i-mu_j)/sqrt2)
        return (int(i), int(j)) if self.rng.rand() < p else (int(j), int(i))

    def sample(self, size: int | None = None, *, batched: bool = True) -> tuple[int, int] | list[tuple[int, int]]:
        """Draw one comparison or a list of comparisons."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class PairWinAccumulator(SequenceEncodableStatisticAccumulator):
    """Win-count matrix ``wins[i, j]`` for (winner, loser) pair data."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.wins = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: tuple[int, int], weight: float, estimate: Any) -> None:
        """Accumulate one weighted winner-loser observation."""
        self.wins[int(x[0]), int(x[1])] += weight
        self.count += weight

    def initialize(self, x: tuple[int, int], weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted pair."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted winner-loser pairs from an encoded batch."""
        np.add.at(self.wins, (x[:, 0], x[:, 1]), weights)
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from an encoded weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> PairWinAccumulator:
        """Merge serialized win-count sufficient statistics."""
        self.count += suff_stat[0]
        self.wins += suff_stat[1]
        return self

    def value(self):
        """Return serialized win-count sufficient statistics."""
        return self.count, self.wins

    def from_value(self, x) -> PairWinAccumulator:
        """Restore accumulator state from serialized win counts."""
        self.count, self.wins = x[0], np.asarray(x[1])
        self.dim = self.wins.shape[0]
        return self

    def acc_to_encoder(self) -> PairDataEncoder:
        """Return the encoder associated with this accumulator."""
        return PairDataEncoder(dim=self.dim)


class PairWinAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for paired win-count accumulators."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim, self.keys = dim, keys

    def make(self) -> PairWinAccumulator:
        """Create a fresh paired-win accumulator."""
        return PairWinAccumulator(dim=self.dim, keys=self.keys)


class ThurstoneMostellerEstimator(ParameterEstimator):
    """``mu_i - mu_j = sqrt(2) Phi^{-1}(P(i beats j))`` from the win-count matrix (least squares)."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        if dim is None or dim < 2:
            raise ValueError("ThurstoneMostellerEstimator requires dim >= 2.")
        self.dim, self.name, self.keys = int(dim), name, keys

    def accumulator_factory(self) -> PairWinAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return PairWinAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> ThurstoneMostellerDistribution:
        """Estimate Thurstone-Mosteller utilities from win-count statistics."""
        count, wins = suff_stat
        n = self.dim
        if count <= 0.0:
            return ThurstoneMostellerDistribution(np.zeros(n), name=self.name, keys=self.keys)
        tot = wins + wins.T
        p = np.where(tot > 0, wins / np.maximum(tot, 1e-12), 0.5)
        np.fill_diagonal(p, 0.5)
        d = _SQRT2 * ndtri(np.clip(p, 1e-4, 1.0 - 1e-4))
        mu = d.mean(axis=1)
        return ThurstoneMostellerDistribution(mu - mu.mean(), name=self.name, keys=self.keys)


class PairDataEncoder(DataSequenceEncoder):
    """Encode ``(winner, loser)`` pairs into an ``(N, 2)`` integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "PairDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PairDataEncoder)

    def seq_encode(self, x: Sequence[tuple[int, int]]) -> np.ndarray:
        """Encode winner-loser pairs as an integer ``(N, 2)`` array."""
        rv = np.asarray([list(p) for p in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[1] != 2 or rv.shape[0] == 0:
            raise ValueError("requires a non-empty sequence of (winner, loser) pairs.")
        if np.any(rv[:, 0] == rv[:, 1]):
            raise ValueError("a comparison must have winner != loser.")
        return rv


# =================================================================================================
# Bradley-Terry with ties: Davidson and Rao-Kupper (observation (i, j, outcome))
# =================================================================================================
class _TieEncoder(DataSequenceEncoder):
    """Encode ``(i, j, outcome)`` triples canonically as ``(lo, hi, o)`` with ``o`` in {0:lo,1:hi,2:tie}."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "TieEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TieEncoder)

    def seq_encode(self, x: Sequence[tuple[int, int, int]]) -> np.ndarray:
        """Encode and canonicalize tie-model comparison triples."""
        rv = np.asarray([list(t) for t in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] == 0:
            raise ValueError("requires a non-empty sequence of (i, j, outcome) triples.")
        if np.any(rv[:, 0] == rv[:, 1]) or np.any((rv[:, 2] < 0) | (rv[:, 2] > 2)):
            raise ValueError("a tie comparison needs i != j and outcome in {0,1,2}.")
        out = rv.copy()
        swap = rv[:, 0] > rv[:, 1]  # canonicalize to lo < hi, remap win outcome 0<->1
        out[swap, 0], out[swap, 1] = rv[swap, 1], rv[swap, 0]
        flip = swap & (rv[:, 2] != 2)
        out[flip, 2] = 1 - rv[flip, 2]
        return out


class _TieAccumulator(SequenceEncodableStatisticAccumulator):
    """``wins[i, j]`` (i beat j) + ``ties[i, j]`` (i<j) from canonical ``(lo, hi, o)`` triples."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.wins = np.zeros((dim, dim))
        self.ties = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x, weight: float, estimate: Any) -> None:
        """Accumulate one weighted comparison triple."""
        self.seq_update(np.asarray([x], dtype=np.int64), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted comparison."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted tie-model comparisons from an encoded batch."""
        lo, hi, o = x[:, 0], x[:, 1], x[:, 2]
        m0, m1, m2 = o == 0, o == 1, o == 2
        np.add.at(self.wins, (lo[m0], hi[m0]), weights[m0])
        np.add.at(self.wins, (hi[m1], lo[m1]), weights[m1])
        np.add.at(self.ties, (lo[m2], hi[m2]), weights[m2])
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from an encoded weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> _TieAccumulator:
        """Merge serialized win-and-tie sufficient statistics."""
        self.count += suff_stat[0]
        self.wins += suff_stat[1]
        self.ties += suff_stat[2]
        return self

    def value(self):
        """Return serialized win-and-tie sufficient statistics."""
        return self.count, self.wins, self.ties

    def from_value(self, x) -> _TieAccumulator:
        """Restore accumulator state from serialized win-and-tie statistics."""
        self.count, self.wins, self.ties = x[0], np.asarray(x[1]), np.asarray(x[2])
        self.dim = self.wins.shape[0]
        return self

    def acc_to_encoder(self) -> _TieEncoder:
        """Return the encoder associated with this accumulator."""
        return _TieEncoder(dim=self.dim)


class _TieAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim, self.keys = dim, keys

    def make(self) -> _TieAccumulator:
        """Create a fresh tie-model accumulator."""
        return _TieAccumulator(dim=self.dim, keys=self.keys)


class _BaseTieDistribution(SequenceEncodableProbabilityDistribution):
    """Shared machinery for the tie paired-comparison models (log_w worths + tie parameter ``nu``)."""

    @classmethod
    def compute_capabilities(cls):
        """Return backend capabilities for paired-comparison tie likelihoods."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Paired-comparison-with-ties likelihood is numpy-native.",
        )

    def __init__(self, log_w, nu, name, keys) -> None:
        lw = np.asarray(log_w, dtype=float)
        if lw.ndim != 1 or lw.size < 2 or not np.all(np.isfinite(lw)):
            raise ValueError("log_w must be a finite length-K vector with K >= 2.")
        self.log_w = lw - lw.mean()
        self.nu = float(nu)
        self.dim = lw.size
        self.log_pairs = math.log(self.dim * (self.dim - 1) / 2.0)
        self.name = name
        self.keys = keys

    def _outcome_logp(self, lo: np.ndarray, hi: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Return outcome log probabilities for canonicalized comparison triples."""
        raise NotImplementedError

    def density(self, x) -> float:
        """Return the probability mass of one comparison triple."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x) -> float:
        """Return the log probability of one comparison triple."""
        i, j, o = int(x[0]), int(x[1]), int(x[2])
        if i > j:
            i, j, o = j, i, (o if o == 2 else 1 - o)
        return float(self._outcome_logp(np.array([i]), np.array([j]), np.array([o]))[0]) - self.log_pairs

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Score encoded canonicalized comparison triples."""
        return self._outcome_logp(x[:, 0], x[:, 1], x[:, 2]) - self.log_pairs

    def _sample_outcome(self, i: int, j: int, rng: RandomState) -> int:
        """Sample an outcome code for the canonical pair ``i < j``."""
        raise NotImplementedError

    def sampler(self, seed: int | None = None):
        """Return a sampler for tie-model comparison triples."""
        return _TieSampler(self, seed)

    def dist_to_encoder(self) -> _TieEncoder:
        """Return the encoder for tie-model comparison triples."""
        return _TieEncoder(dim=self.dim)


class _TieSampler(DistributionSampler):
    def __init__(self, dist: _BaseTieDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self):
        k = self.dist.dim
        i, j = sorted(self.rng.choice(k, size=2, replace=False))
        return (int(i), int(j), int(self.dist._sample_outcome(int(i), int(j), self.rng)))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one comparison triple or a list of triples."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class DavidsonDistribution(_BaseTieDistribution):
    """Bradley-Terry with ties (Davidson 1970); tie mass ``nu sqrt(w_i w_j)``."""

    def __init__(self, log_w, nu: float = 1.0, name: str | None = None, keys: str | None = None) -> None:
        if nu < 0.0:
            raise ValueError("Davidson nu must be >= 0.")
        super().__init__(log_w, nu, name, keys)

    def __str__(self) -> str:
        return "DavidsonDistribution(%s, nu=%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.log_w]),
            repr(self.nu),
            repr(self.name),
            repr(self.keys),
        )

    def _outcome_logp(self, lo, hi, o):
        """Return Davidson log probabilities for canonicalized outcomes."""
        wi, wj = np.exp(self.log_w[lo]), np.exp(self.log_w[hi])
        g = np.sqrt(wi * wj)
        denom = wi + wj + self.nu * g
        num = np.where(o == 0, wi, np.where(o == 1, wj, self.nu * g))
        return np.log(num) - np.log(denom)

    def _sample_outcome(self, i, j, rng):
        """Sample a Davidson outcome code for a canonical pair."""
        wi, wj = math.exp(self.log_w[i]), math.exp(self.log_w[j])
        g = math.sqrt(wi * wj)
        denom = wi + wj + self.nu * g
        return int(rng.choice(3, p=[wi / denom, wj / denom, self.nu * g / denom]))

    def estimator(self, pseudo_count: float | None = None):
        """Return the maximum-likelihood Davidson estimator."""
        return DavidsonEstimator(dim=self.dim, name=self.name, keys=self.keys)


class RaoKupperDistribution(_BaseTieDistribution):
    """Bradley-Terry with ties via a threshold ``nu >= 1`` (Rao-Kupper 1967)."""

    def __init__(self, log_w, nu: float = 1.5, name: str | None = None, keys: str | None = None) -> None:
        if nu < 1.0:
            raise ValueError("Rao-Kupper nu must be >= 1.")
        super().__init__(log_w, nu, name, keys)

    def __str__(self) -> str:
        return "RaoKupperDistribution(%s, nu=%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.log_w]),
            repr(self.nu),
            repr(self.name),
            repr(self.keys),
        )

    def _outcome_logp(self, lo, hi, o):
        """Return Rao-Kupper log probabilities for canonicalized outcomes."""
        wi, wj, nu = np.exp(self.log_w[lo]), np.exp(self.log_w[hi]), self.nu
        p_i = wi / (wi + nu * wj)
        p_j = wj / (nu * wi + wj)
        p_tie = np.clip(1.0 - p_i - p_j, 1e-300, 1.0)
        return np.log(np.where(o == 0, p_i, np.where(o == 1, p_j, p_tie)))

    def _sample_outcome(self, i, j, rng):
        """Sample a Rao-Kupper outcome code for a canonical pair."""
        wi, wj, nu = math.exp(self.log_w[i]), math.exp(self.log_w[j]), self.nu
        p_i = wi / (wi + nu * wj)
        p_j = wj / (nu * wi + wj)
        return int(rng.choice(3, p=[p_i, p_j, max(1.0 - p_i - p_j, 0.0)]))

    def estimator(self, pseudo_count: float | None = None):
        """Return the maximum-likelihood Rao-Kupper estimator."""
        return RaoKupperEstimator(dim=self.dim, name=self.name, keys=self.keys)


def _fit_tie_model(make_dist, dim, wins, ties, nu0, nu_bounds):
    """Maximize the comparison log-likelihood over (log_w centered, nu) with L-BFGS-B."""
    iu = np.triu_indices(dim, 1)
    a = wins[iu]  # lo beat hi
    b = wins.T[iu]  # hi beat lo
    t = ties[iu] + ties.T[iu]
    li, lj = iu

    def neg_ll(params):
        theta = np.zeros(dim)
        theta[1:] = params[:-1]
        nu = params[-1]
        d = make_dist(theta - theta.mean(), nu)
        lp = d._outcome_logp(li, lj, np.zeros_like(li))
        lp1 = d._outcome_logp(li, lj, np.ones_like(li))
        lp2 = d._outcome_logp(li, lj, np.full_like(li, 2))
        return -float(a @ lp + b @ lp1 + t @ lp2)

    x0 = np.zeros(dim)  # theta_1..theta_{K-1} (theta_0=0) + nu
    x0[-1] = nu0
    bounds = [(-20, 20)] * (dim - 1) + [nu_bounds]
    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    theta = np.zeros(dim)
    theta[1:] = res.x[:-1]
    return theta - theta.mean(), float(res.x[-1])


class DavidsonEstimator(ParameterEstimator):
    """Maximum-likelihood Davidson worths and tie parameter (L-BFGS on the count matrices)."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        if dim is None or dim < 2:
            raise ValueError("DavidsonEstimator requires dim >= 2.")
        self.dim, self.name, self.keys = int(dim), name, keys

    def accumulator_factory(self) -> _TieAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return _TieAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> DavidsonDistribution:
        """Estimate Davidson worths and tie parameter from sufficient statistics."""
        count, wins, ties = suff_stat
        if count <= 0.0:
            return DavidsonDistribution(np.zeros(self.dim), 1.0, name=self.name, keys=self.keys)
        log_w, nu = _fit_tie_model(
            lambda lw, nu: DavidsonDistribution(lw, nu), self.dim, wins, ties, 1.0, (1e-6, 100.0)
        )
        return DavidsonDistribution(log_w, nu, name=self.name, keys=self.keys)


class RaoKupperEstimator(ParameterEstimator):
    """Maximum-likelihood Rao-Kupper worths and threshold (L-BFGS on the count matrices)."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        if dim is None or dim < 2:
            raise ValueError("RaoKupperEstimator requires dim >= 2.")
        self.dim, self.name, self.keys = int(dim), name, keys

    def accumulator_factory(self) -> _TieAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return _TieAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> RaoKupperDistribution:
        """Estimate Rao-Kupper worths and threshold from sufficient statistics."""
        count, wins, ties = suff_stat
        if count <= 0.0:
            return RaoKupperDistribution(np.zeros(self.dim), 1.5, name=self.name, keys=self.keys)
        log_w, nu = _fit_tie_model(
            lambda lw, nu: RaoKupperDistribution(lw, nu), self.dim, wins, ties, 1.5, (1.0, 100.0)
        )
        return RaoKupperDistribution(log_w, nu, name=self.name, keys=self.keys)


__all__ = [
    "ThurstoneMostellerDistribution",
    "ThurstoneMostellerSampler",
    "ThurstoneMostellerEstimator",
    "PairWinAccumulator",
    "PairWinAccumulatorFactory",
    "PairDataEncoder",
    "DavidsonDistribution",
    "DavidsonEstimator",
    "RaoKupperDistribution",
    "RaoKupperEstimator",
]
