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

Every policy owns a seeded ``numpy.random.RandomState`` and is deterministic given it. Conjugate
and UCB1 updates commute, so ``batch_update`` (delayed/batched feedback, the streaming pattern
used across mixle) is exactly the sequential replay of its pairs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EstimatorBandit", "ThompsonBernoulli", "ThompsonGaussian", "UCB1"]


class _BanditBase:
    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        if n_arms < 2:
            raise ValueError("a bandit needs at least two arms.")
        self.n_arms = int(n_arms)
        self.rng = np.random.RandomState(seed)
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)

    def _check_arm(self, arm: int) -> int:
        arm = int(arm)
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
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta priors must be positive.")
        self.alpha = np.full(self.n_arms, float(alpha))
        self.beta = np.full(self.n_arms, float(beta))

    @property
    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        reward = float(reward)
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
        if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("kappa0, alpha0, and beta0 must be positive.")
        self.m = np.full(self.n_arms, float(mu0))
        self.kappa = np.full(self.n_arms, float(kappa0))
        self.a = np.full(self.n_arms, float(alpha0))
        self.b = np.full(self.n_arms, float(beta0))

    @property
    def means(self) -> np.ndarray:
        return self.m.copy()

    def select(self) -> int:
        sigma2 = self.b / self.rng.gamma(self.a)  # inverse-gamma draw per arm
        mu = self.rng.normal(self.m, np.sqrt(sigma2 / self.kappa))
        return int(np.argmax(mu))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        x = float(reward)
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
        if c <= 0:
            raise ValueError("the exploration scale c must be positive.")
        self.c = float(c)
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
        self.sums[arm] += float(reward)
        self.pulls[arm] += 1


class EstimatorBandit(_BanditBase):
    """Thompson sampling for ARBITRARY mixle reward models, via the online bootstrap.

    Each arm keeps ``n_boot`` accumulator replicates of its estimator; ``update`` adds the reward
    to every replicate with an independent Poisson(1) weight (Eckles & Kaptein's online bootstrap),
    so the replicate ensemble approximates the sampling distribution of the fitted reward model
    with no conjugate structure required. ``select`` plays each arm once, then draws one non-empty
    replicate per arm, fits it (``estimator.estimate``), scores it with ``mean_fn`` (default:
    Monte-Carlo mean of ``estimate.sampler(...).sample(mc_draws)``), and plays the argmax --
    posterior-sample-then-maximize, exactly Thompson's rule with a bootstrap posterior.

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
        if n_boot < 2:
            raise ValueError("the bootstrap needs at least two replicates.")
        self.estimators = estimators
        self.n_boot = int(n_boot)
        self.mc_draws = int(mc_draws)
        self.mean_fn = mean_fn
        self._replicates = [[est.accumulator_factory().make() for _ in range(n_boot)] for est in estimators]
        self._replicate_weight = np.zeros((self.n_arms, self.n_boot))

    def _score(self, arm: int) -> float:
        weights = self._replicate_weight[arm]
        candidates = np.flatnonzero(weights > 0)
        b = int(candidates[self.rng.randint(len(candidates))])
        fitted = self.estimators[arm].estimate(float(weights[b]), self._replicates[arm][b].value())
        if self.mean_fn is not None:
            return float(self.mean_fn(fitted))
        draws = fitted.sampler(seed=int(self.rng.randint(2**31 - 1))).sample(size=self.mc_draws)
        return float(np.mean(draws))

    def select(self) -> int:
        unplayed = np.flatnonzero(self.pulls == 0)
        if len(unplayed):
            return int(unplayed[0])
        return int(np.argmax([self._score(arm) for arm in range(self.n_arms)]))

    def update(self, arm: int, reward: float) -> None:
        arm = self._check_arm(arm)
        weights = self.rng.poisson(1.0, self.n_boot).astype(np.float64)
        if not np.any(weights > 0):  # keep every observation represented somewhere
            weights[self.rng.randint(self.n_boot)] = 1.0
        for b in np.flatnonzero(weights > 0):
            self._replicates[arm][b].update(reward, float(weights[b]), None)
        self._replicate_weight[arm] += weights
        self.pulls[arm] += 1
