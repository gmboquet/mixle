"""Multi-armed bandits: pick-an-arm / observe-a-reward loops with posterior policies.

Where this sits in mixle: :mod:`mixle.doe`'s Bayesian-optimization acquisitions (UCB and
Thompson draws over a surrogate posterior) are the continuous-design cousins; this module is the
small DISCRETE-arm loop for serving-time decisions -- which teacher to query, which prompt
variant, which data source, which model tier answers when calibrated confidence
(:class:`mixle.task.router.Router`) is unavailable or costs drift and the choice must be LEARNED
from observed outcomes instead.

Policies (shared surface: ``select() -> arm``, ``update(arm, reward)``,
``batch_update(arms, rewards)``, ``pulls``, ``means``):

- :class:`ThompsonBernoulli` -- Beta-Bernoulli conjugate Thompson sampling; rewards in [0, 1]
  (clicks, agreement flags, pass/fail, or any fractional credit).
- :class:`ThompsonGaussian` -- Normal-Inverse-Gamma conjugate Thompson sampling; unbounded
  real rewards with unknown variance (latencies, margins, log-likelihood gains).
- :class:`UCB1` -- the deterministic optimism baseline (Auer et al.); no randomness at all,
  useful when reproducible selection matters more than Bayesian credit assignment.
- :class:`EstimatorBandit` -- Thompson sampling with ARBITRARY mixle reward models via the
  online bootstrap (Poisson(1) replicate weights, Eckles & Kaptein): any
  estimator/accumulator pair -- Gamma service times, categorical outcomes scored by a utility,
  mixtures for multi-modal rewards -- becomes an arm with NO conjugate math, riding the same
  accumulator machinery the rest of mixle estimates with.

Every policy owns a seeded ``numpy.random.RandomState`` and is deterministic given it.
``batch_update`` preserves the supplied pair order and is exactly equivalent to calling ``update``
in that order from the same state. Bernoulli, Gaussian, and UCB sufficient-statistic updates commute.
EstimatorBandit's do not: its seeded Poisson bootstrap weights are assigned in observation order, so
reordering delayed feedback is a different (though reproducible) bootstrap replay.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

__all__ = ["EstimatorBandit", "ThompsonBernoulli", "ThompsonGaussian", "UCB1"]


def _exact_int(value, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _seed(value: int | None) -> int | None:
    if value is None:
        return None
    result = _exact_int(value, "seed", minimum=0)
    if result > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    return result


class _BanditBase:
    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = _exact_int(n_arms, "n_arms", minimum=2)
        self.rng = np.random.RandomState(_seed(seed))
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)

    def _check_arm(self, arm: int) -> int:
        arm = _exact_int(arm, "arm", minimum=0)
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm {arm} out of range for {self.n_arms} arms.")
        return arm

    def batch_update(self, arms, rewards) -> None:
        """Apply delayed/batched feedback: exactly the sequential replay of the pairs."""
        arms, rewards = list(arms), list(rewards)
        if len(arms) != len(rewards):
            raise ValueError("arms and rewards must have the same length.")
        for arm, reward in zip(arms, rewards):
            self.update(arm, reward)

    def update(self, arm: int, reward: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class ThompsonBernoulli(_BanditBase):
    """Beta-Bernoulli Thompson sampling. Rewards live in [0, 1]; fractional rewards contribute
    fractional pseudo-counts (the standard Bernoulli-moment update)."""

    def __init__(self, n_arms: int, *, alpha: float = 1.0, beta: float = 1.0, seed: int | None = None) -> None:
        super().__init__(n_arms, seed=seed)
        alpha = _finite_real(alpha, "alpha", positive=True)
        beta = _finite_real(beta, "beta", positive=True)
        self.alpha = np.full(self.n_arms, alpha)
        self.beta = np.full(self.n_arms, beta)

    @property
    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        reward = _finite_real(reward, "reward")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("ThompsonBernoulli rewards must lie in [0, 1].")
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
        self.pulls[arm] += 1


class ThompsonGaussian(_BanditBase):
    """Normal-Inverse-Gamma Thompson sampling: unknown mean AND variance per arm, so early
    optimism comes from honest posterior width rather than a tuned exploration constant."""

    def __init__(
        self,
        n_arms: int,
        *,
        mu0: float = 0.0,
        kappa0: float = 1.0e-2,
        alpha0: float = 0.5,
        beta0: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, seed=seed)
        mu0 = _finite_real(mu0, "mu0")
        kappa0 = _finite_real(kappa0, "kappa0", positive=True)
        alpha0 = _finite_real(alpha0, "alpha0", positive=True)
        beta0 = _finite_real(beta0, "beta0", positive=True)
        self.m = np.full(self.n_arms, mu0)
        self.kappa = np.full(self.n_arms, kappa0)
        self.a = np.full(self.n_arms, alpha0)
        self.b = np.full(self.n_arms, beta0)

    @property
    def means(self) -> np.ndarray:
        return self.m.copy()

    def select(self) -> int:
        sigma2 = self.b / self.rng.gamma(self.a)  # inverse-gamma draw per arm
        mu = self.rng.normal(self.m, np.sqrt(sigma2 / self.kappa))
        return int(np.argmax(mu))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        x = _finite_real(reward, "reward")
        kappa1 = self.kappa[arm] + 1.0
        self.b[arm] += 0.5 * self.kappa[arm] * (x - self.m[arm]) ** 2 / kappa1
        self.m[arm] = (self.kappa[arm] * self.m[arm] + x) / kappa1
        self.kappa[arm] = kappa1
        self.a[arm] += 0.5
        self.pulls[arm] += 1


class UCB1(_BanditBase):
    """The deterministic optimism baseline: play each arm once, then
    ``argmax mean_k + c * sqrt(2 ln t / n_k)``. Ties break to the lowest index; with no
    randomness anywhere, two UCB1 runs on the same reward sequence are identical."""

    def __init__(self, n_arms: int, *, c: float = 1.0, seed: int | None = None) -> None:
        super().__init__(n_arms, seed=seed)
        self.c = _finite_real(c, "c", positive=True)
        self.sums = np.zeros(self.n_arms)

    @property
    def means(self) -> np.ndarray:
        return self.sums / np.maximum(self.pulls, 1)

    def select(self) -> int:
        unplayed = np.flatnonzero(self.pulls == 0)
        if len(unplayed):
            return int(unplayed[0])
        t = float(self.pulls.sum())
        bonus = self.c * np.sqrt(2.0 * np.log(t) / self.pulls)
        return int(np.argmax(self.means + bonus))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        self.sums[arm] += _finite_real(reward, "reward")
        self.pulls[arm] += 1


class EstimatorBandit(_BanditBase):
    """Thompson sampling for ARBITRARY mixle reward models, via the online bootstrap.

    Each arm keeps ``n_boot`` accumulator replicates of its estimator; ``update`` adds the reward
    to every replicate with an independent Poisson(1) weight (Eckles & Kaptein's online bootstrap),
    so the replicate ensemble approximates the sampling distribution of the fitted reward model
    with no conjugate structure required. ``select`` plays each arm once, then draws one non-empty
    replicate per arm, fits it (``estimator.estimate``), scores it with ``mean_fn`` (default:
    Monte-Carlo mean of ``estimate.sampler(...).sample(mc_draws)``), and plays the argmax --
    posterior-sample-then-maximize, exactly Thompson's rule with a bootstrap posterior. Bootstrap
    assignment is deliberately order-sensitive: preserve event order when replaying delayed feedback.

    ``estimators`` is one mixle ParameterEstimator per arm (Gamma for waiting times, Gaussian for
    margins, a mixture for multi-modal rewards -- anything with the accumulator contract).
    """

    def __init__(
        self,
        estimators,
        *,
        n_boot: int = 32,
        mean_fn=None,
        mc_draws: int = 64,
        seed: int | None = None,
    ) -> None:
        estimators = list(estimators)
        super().__init__(len(estimators), seed=seed)
        n_boot = _exact_int(n_boot, "n_boot", minimum=2)
        mc_draws = _exact_int(mc_draws, "mc_draws", minimum=1)
        if mean_fn is not None and not callable(mean_fn):
            raise TypeError("mean_fn must be callable or None")
        for i, estimator in enumerate(estimators):
            if not callable(getattr(estimator, "accumulator_factory", None)) or not callable(
                getattr(estimator, "estimate", None)
            ):
                raise TypeError(f"estimator {i} does not implement the estimator contract")
        self.estimators = estimators
        self.n_boot = n_boot
        self.mc_draws = mc_draws
        self.mean_fn = mean_fn
        self._replicates = [[est.accumulator_factory().make() for _ in range(n_boot)] for est in estimators]
        self._replicate_weight = np.zeros((self.n_arms, self.n_boot))

    def _score(self, arm: int) -> float:
        weights = self._replicate_weight[arm]
        candidates = np.flatnonzero(weights > 0)
        b = int(candidates[self.rng.randint(len(candidates))])
        fitted = self.estimators[arm].estimate(float(weights[b]), self._replicates[arm][b].value())
        if self.mean_fn is not None:
            return _finite_real(self.mean_fn(fitted), "mean_fn score")
        draws = np.asarray(fitted.sampler(seed=int(self.rng.randint(2**31 - 1))).sample(size=self.mc_draws))
        if draws.size != self.mc_draws or draws.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"estimator sampler must return exactly {self.mc_draws} real draws")
        if not np.all(np.isfinite(draws)):
            raise ValueError("estimator sampler returned non-finite draws")
        return _finite_real(np.mean(draws), "estimator score")

    def select(self) -> int:
        unplayed = np.flatnonzero(self.pulls == 0)
        if len(unplayed):
            return int(unplayed[0])
        return int(np.argmax([self._score(arm) for arm in range(self.n_arms)]))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        reward = _finite_real(reward, "reward")
        weights = self.rng.poisson(1.0, self.n_boot).astype(np.float64)
        if not np.any(weights > 0):  # keep every observation represented somewhere
            weights[self.rng.randint(self.n_boot)] = 1.0
        for b in np.flatnonzero(weights > 0):
            self._replicates[arm][b].update(reward, float(weights[b]), None)
        self._replicate_weight[arm] += weights
        self.pulls[arm] += 1
