"""K3 DoD -- dose-response / health-risk models (notes/exec/workstream-K.md).

`DoseResponse.probability` must turn an exposure `Posterior` (IC-1) into an outcome-probability
`DerivedQuantity` whose credible interval is *calibrated*: a nominal 90% interval built from one draw
of the pushforward should cover the true dose-response probability -- computed from independent
fresh draws of the same exposure distribution -- close to 90% of the time, and that interval should
widen as the exposure posterior's own variance grows.

Named with the ``test_*.py`` prefix (rather than this repo's own ``*_test.py`` `python_files`
convention -- see ``pyproject.toml``) because this exact path + node id is the frozen DoD command in
``notes/exec/workstream-K.md``; explicit pytest node ids are collected regardless of the
``python_files`` glob, so this does not conflict with the repo's discovery config.
"""

from __future__ import annotations

import numpy as np

from mixle.analysis.health_risk import DoseResponse
from mixle.reason.posterior_protocol import Posterior


def _lognormal_exposure_posterior(mu: float, sigma: float) -> Posterior:
    """A minimal IC-1 `Posterior` over a single-receptor exposure: dose ~ LogNormal(mu, sigma)."""

    class _ExposurePosterior:
        def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
            return np.exp(mu + sigma * rng.standard_normal(n))

        @property
        def mean(self) -> np.ndarray:
            return np.array([np.exp(mu + sigma**2 / 2.0)])

        @property
        def cov(self) -> np.ndarray:
            var = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)
            return np.array([[var]])

        def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
            s = self.samples(200_000, np.random.default_rng(999))
            a = (1.0 - level) / 2.0
            return np.array([np.quantile(s, a)]), np.array([np.quantile(s, 1.0 - a)])

        def derived_quantity(self, fn, n: int, rng: np.random.Generator):
            draws = self.samples(n, rng)
            pushed = fn(draws)

            class _DQ:
                samples = pushed
                prior_dominated = False

                def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
                    a = (1.0 - level) / 2.0
                    return np.quantile(self.samples, a), np.quantile(self.samples, 1.0 - a)

            return _DQ()

    return _ExposurePosterior()


def test_dose_response_calibrated():
    mu, sigma = np.log(15.0), 0.4
    posterior = _lognormal_exposure_posterior(mu, sigma)
    assert isinstance(posterior, Posterior)

    dr = DoseResponse(model="loglinear", params={"beta": 0.05})
    dq = dr.probability(posterior, n=5000, rng=np.random.default_rng(1))
    assert dq.prior_dominated is False
    lo, hi = dq.credible_interval(0.9)

    # Empirical coverage: fresh, independent draws from the *same* generative exposure distribution,
    # pushed through the same response curve, should fall inside the nominal 90% interval close to
    # 90% of the time (loglinear is monotone in dose, so quantiles commute with the pushforward).
    check_rng = np.random.default_rng(42)
    true_doses = np.exp(mu + sigma * check_rng.standard_normal(5000))
    true_probs = dr.response_fn()(true_doses)
    coverage = float(np.mean((true_probs >= lo) & (true_probs <= hi)))
    assert coverage >= 0.88

    # The interval must widen as the exposure posterior's variance grows.
    wide_posterior = _lognormal_exposure_posterior(mu, sigma * 2.0)
    dq_wide = dr.probability(wide_posterior, n=5000, rng=np.random.default_rng(2))
    lo_wide, hi_wide = dq_wide.credible_interval(0.9)
    assert (hi_wide - lo_wide) > (hi - lo)
