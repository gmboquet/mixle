"""Fixed invertible-transform wrapper for sequence-encodable distributions."""
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState

from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)


def _uses_density_correction(dist: SequenceEncodableProbabilityDistribution,
                             density_correction: Optional[bool]) -> bool:
    if density_correction is not None:
        return bool(density_correction)
    try:
        dist.enumerator()
        return False
    except EnumerationError:
        return True


class IdentityTransform(object):
    """Identity transform y = x."""

    def __str__(self) -> str:
        return 'IdentityTransform()'

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityTransform)

    def forward(self, x: Any) -> Any:
        return x

    def inverse(self, y: Any) -> Any:
        return y

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        return 0.0

    def invalid_inverse_value(self) -> float:
        return 0.0


class AffineTransform(object):
    """Affine transform y = loc + scale * x."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if scale == 0.0 or not np.isfinite(scale):
            raise ValueError('AffineTransform requires finite non-zero scale.')
        self.loc = float(loc)
        self.scale = float(scale)
        self._log_abs_inv = -math.log(abs(self.scale))

    def __str__(self) -> str:
        return 'AffineTransform(loc=%s, scale=%s)' % (repr(self.loc), repr(self.scale))

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AffineTransform) and self.loc == other.loc and self.scale == other.scale

    def forward(self, x: Any) -> Any:
        return self.loc + self.scale * x

    def inverse(self, y: Any) -> Any:
        return (y - self.loc) / self.scale

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        return self._log_abs_inv

    def invalid_inverse_value(self) -> float:
        return 0.0


class ExpTransform(object):
    """Exponential transform y = exp(x), mapping real x to positive y."""

    def __str__(self) -> str:
        return 'ExpTransform()'

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExpTransform)

    def forward(self, x: Any) -> Any:
        return np.exp(x)

    def inverse(self, y: Any) -> Any:
        if y <= 0.0:
            raise ValueError('ExpTransform inverse requires y > 0.')
        return math.log(y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        if y <= 0.0:
            raise ValueError('ExpTransform inverse requires y > 0.')
        return -math.log(y)

    def invalid_inverse_value(self) -> float:
        return 0.0


class LogTransform(object):
    """Log transform y = log(x), mapping positive x to real y."""

    def __str__(self) -> str:
        return 'LogTransform()'

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogTransform)

    def forward(self, x: Any) -> Any:
        if x <= 0.0:
            raise ValueError('LogTransform forward requires x > 0.')
        return math.log(x)

    def inverse(self, y: Any) -> Any:
        return math.exp(y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        return float(y)

    def invalid_inverse_value(self) -> float:
        return 1.0


class LogitTransform(object):
    """Logistic transform y = 1 / (1 + exp(-x)), mapping real x to (0, 1)."""

    def __str__(self) -> str:
        return 'LogitTransform()'

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogitTransform)

    def forward(self, x: Any) -> Any:
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def inverse(self, y: Any) -> Any:
        if y <= 0.0 or y >= 1.0:
            raise ValueError('LogitTransform inverse requires 0 < y < 1.')
        return math.log(y) - math.log1p(-y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        if y <= 0.0 or y >= 1.0:
            raise ValueError('LogitTransform inverse requires 0 < y < 1.')
        return -math.log(y) - math.log1p(-y)

    def invalid_inverse_value(self) -> float:
        return 0.0


class TransformDistribution(SequenceEncodableProbabilityDistribution):
    """Push a child distribution through a fixed invertible transform.

    Observations live in transformed space. For fixed continuous transforms,
    log-density uses the inverse transform and adds the inverse-Jacobian term.
    The transform is not learned; estimation inverse-transforms observations
    and delegates sufficient statistics to the child estimator.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution,
                 transform: Optional[Any] = None, density_correction: Optional[bool] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.dist = dist
        self.transform = transform if transform is not None else IdentityTransform()
        self.density_correction = _uses_density_correction(dist, density_correction)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'TransformDistribution(%s, transform=%s, density_correction=%s, name=%s, keys=%s)' % (
            str(self.dist), repr(self.transform), repr(self.density_correction),
            repr(self.name), repr(self.keys))

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        try:
            inv = self.transform.inverse(x)
            rv = self.dist.log_density(inv)
            if self.density_correction:
                rv += self.transform.log_abs_det_inverse_jacobian(x)
            return rv
        except Exception:
            return -np.inf

    def seq_log_density(self, x: Tuple[Any, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        child_enc, log_jac, valid = x
        rv = self.dist.seq_log_density(child_enc)
        if self.density_correction:
            rv = rv + log_jac
        return np.where(valid, rv, -np.inf)

    def sampler(self, seed: Optional[int] = None) -> 'TransformSampler':
        """Return a sampler for drawing observations from this distribution."""
        return TransformSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'TransformEstimator':
        """Return an estimator for fitting this distribution from data."""
        return TransformEstimator(self.dist.estimator(pseudo_count=pseudo_count),
                                  self.transform, density_correction=self.density_correction,
                                  name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'TransformDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return TransformDataEncoder(self.dist.dist_to_encoder(), self.transform,
                                    density_correction=self.density_correction)

    def enumerator(self) -> 'TransformEnumerator':
        """Return an enumerator over the distribution support when available."""
        return TransformEnumerator(self)


class TransformEnumerator(DistributionEnumerator):
    """Enumerate transformed child support for discrete child distributions."""

    def __init__(self, dist: TransformDistribution) -> None:
        super().__init__(dist)
        self.child_iter = child_enumerator(dist.dist, 'TransformDistribution.dist')

    def __next__(self) -> Tuple[Any, float]:
        v, lp = next(self.child_iter)
        return self.dist.transform.forward(v), lp


class TransformSampler(DistributionSampler):
    """Sampler that transforms draws from the child distribution."""

    def __init__(self, dist: TransformDistribution, seed: Optional[int] = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.child_sampler = dist.dist.sampler(seed=self.new_seed())

    def sample(self, size: Optional[int] = None):
        x = self.child_sampler.sample(size=size)
        if size is None:
            return self.dist.transform.forward(x)
        return [self.dist.transform.forward(v) for v in x]


class TransformAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that delegates inverse-transformed observations to the child."""

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator,
                 transform: Any, density_correction: Optional[bool] = None,
                 name: Optional[str] = None) -> None:
        self.accumulator = accumulator
        self.transform = transform
        self.density_correction = density_correction
        self.name = name

    def update(self, x: Any, weight: float, estimate: Optional[TransformDistribution]) -> None:
        try:
            inv = self.transform.inverse(x)
        except Exception:
            return
        self.accumulator.update(inv, weight, None if estimate is None else estimate.dist)

    def seq_update(self, x: Tuple[Any, np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[TransformDistribution]) -> None:
        child_enc, _, valid = x
        self.accumulator.seq_update(child_enc, weights * valid.astype(float),
                                    None if estimate is None else estimate.dist)

    def initialize(self, x: Any, weight: float, rng: Optional[RandomState]) -> None:
        try:
            inv = self.transform.inverse(x)
        except Exception:
            return
        self.accumulator.initialize(inv, weight, rng)

    def seq_initialize(self, x: Tuple[Any, np.ndarray, np.ndarray],
                       weights: np.ndarray, rng: Optional[RandomState]) -> None:
        child_enc, _, valid = x
        self.accumulator.seq_initialize(child_enc, weights * valid.astype(float), rng)

    def combine(self, suff_stat: Any) -> 'TransformAccumulator':
        self.accumulator.combine(suff_stat)
        return self

    def value(self) -> Any:
        return self.accumulator.value()

    def from_value(self, x: Any) -> 'TransformAccumulator':
        self.accumulator.from_value(x)
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'TransformDataEncoder':
        return TransformDataEncoder(self.accumulator.acc_to_encoder(), self.transform,
                                    density_correction=self.density_correction)


class TransformAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for TransformAccumulator."""

    def __init__(self, factory: StatisticAccumulatorFactory, transform: Any,
                 density_correction: Optional[bool] = None, name: Optional[str] = None) -> None:
        self.factory = factory
        self.transform = transform
        self.density_correction = density_correction
        self.name = name

    def make(self) -> TransformAccumulator:
        return TransformAccumulator(self.factory.make(), self.transform,
                                    density_correction=self.density_correction, name=self.name)


class TransformEstimator(ParameterEstimator):
    """Estimator for fixed-transform distributions."""

    def __init__(self, estimator: ParameterEstimator, transform: Optional[Any] = None,
                 density_correction: Optional[bool] = None, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.estimator = estimator
        self.transform = transform if transform is not None else IdentityTransform()
        self.density_correction = density_correction
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> TransformAccumulatorFactory:
        return TransformAccumulatorFactory(self.estimator.accumulator_factory(),
                                           self.transform, density_correction=self.density_correction,
                                           name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Any) -> TransformDistribution:
        return TransformDistribution(
            self.estimator.estimate(nobs, suff_stat), transform=self.transform,
            density_correction=self.density_correction, name=self.name, keys=self.keys)


class TransformDataEncoder(DataSequenceEncoder):
    """Encode transformed observations as inverse child data plus Jacobian terms."""

    def __init__(self, encoder: DataSequenceEncoder, transform: Any,
                 density_correction: Optional[bool] = True) -> None:
        self.encoder = encoder
        self.transform = transform
        self.density_correction = density_correction is not False

    def __str__(self) -> str:
        return 'TransformDataEncoder(encoder=%s, transform=%s, density_correction=%s)' % (
            repr(self.encoder), repr(self.transform), repr(self.density_correction))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TransformDataEncoder) and other.encoder == self.encoder \
            and other.transform == self.transform and other.density_correction == self.density_correction

    def seq_encode(self, x: Sequence[Any]) -> Tuple[Any, np.ndarray, np.ndarray]:
        inv_values = []
        valid = np.ones(len(x), dtype=bool)
        log_jac = np.zeros(len(x), dtype=np.float64)
        fill = self.transform.invalid_inverse_value()

        for i, y in enumerate(x):
            try:
                inv_values.append(self.transform.inverse(y))
                if self.density_correction:
                    log_jac[i] = self.transform.log_abs_det_inverse_jacobian(y)
                    if not np.isfinite(log_jac[i]):
                        valid[i] = False
            except Exception:
                inv_values.append(fill)
                log_jac[i] = -np.inf
                valid[i] = False

        return self.encoder.seq_encode(inv_values), log_jac, valid
