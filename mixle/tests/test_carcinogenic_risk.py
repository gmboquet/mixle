"""K7 -- carcinogenic-risk models: linear no-threshold slope-factor / unit-risk (workstream-K.md).

Definition of Done: a benchmark arsenic LADD reproduces the EPA-IRIS reference excess lifetime cancer
risk (LADD * oral_csf) within 1% relative tolerance; the credible interval widens monotonically as
exposure variance grows; the result is an IC-8-style ``DerivedQuantity`` (samples + CI +
``prior_dominated``), with ``prior_dominated`` propagating from the exposure posterior.
"""

import numpy as np
import pytest

from mixle.analysis.carcinogenic_risk import (
    RiskQuantity,
    SlopeFactor,
    excess_lifetime_cancer_risk,
    radon_wlm_risk,
)
from mixle.reason.posterior_protocol import DerivedQuantity


class _ExposureDerivedQuantity:
    """Minimal IC-1 ``DerivedQuantity`` conforming object returned by ``_ExposurePosterior``."""

    def __init__(self, samples: np.ndarray, prior_dominated: bool):
        self.samples = np.asarray(samples, dtype=float)
        self.prior_dominated = prior_dominated

    def credible_interval(self, level: float) -> tuple[float, float]:
        alpha = (1.0 - level) / 2.0
        return float(np.quantile(self.samples, alpha)), float(np.quantile(self.samples, 1.0 - alpha))


class _ExposurePosterior:
    """Minimal IC-1 ``Posterior`` conforming object: a Gaussian lifetime-average-dose posterior."""

    def __init__(self, mean: float, sigma: float, prior_dominated: bool = False):
        self._mean = float(mean)
        self._sigma = float(sigma)
        self._prior_dominated = prior_dominated

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        draws = rng.normal(loc=self._mean, scale=self._sigma, size=n)
        return np.clip(draws, 1e-12, None).reshape(n, 1)

    @property
    def mean(self) -> np.ndarray:
        return np.array([self._mean])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[self._sigma**2]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = {0.9: 1.645, 0.95: 1.96}.get(level, 1.645)
        return np.array([self._mean - z * self._sigma]), np.array([self._mean + z * self._sigma])

    def derived_quantity(self, fn, n: int, rng: np.random.Generator) -> _ExposureDerivedQuantity:
        return _ExposureDerivedQuantity(fn(self.samples(n, rng)), self._prior_dominated)


def test_arsenic_matches_epa_iris():
    ladd = 1e-4  # mg/kg-day, benchmark arsenic lifetime average daily dose
    sf = SlopeFactor(oral_csf=1.5, source="EPA-IRIS")  # (mg/kg-day)^-1, EPA IRIS arsenic oral CSF

    # 1) point estimate matches the reference linear no-threshold risk within 1% relative tolerance.
    result = excess_lifetime_cancer_risk(ladd, sf, route="oral")
    reference = ladd * 1.5
    assert abs(result.mean - reference) / reference < 0.01

    # 2) the result is an IC-8-style DerivedQuantity carrying prior_dominated.
    assert isinstance(result, RiskQuantity)
    assert isinstance(result, DerivedQuantity)
    assert hasattr(result, "prior_dominated")
    assert result.prior_dominated is False

    # 3) the credible interval widens monotonically as exposure variance grows.
    sigmas = [2e-6, 6e-6, 1.8e-5, 4e-5]
    widths = []
    for sigma in sigmas:
        exposure = _ExposurePosterior(mean=ladd, sigma=sigma)
        dq = excess_lifetime_cancer_risk(exposure, sf, route="oral", n=4000, rng=np.random.default_rng(1))
        lo, hi = dq.credible_interval(0.9)
        widths.append(hi - lo)
    assert all(w2 >= w1 for w1, w2 in zip(widths, widths[1:])), widths
    assert widths[-1] > widths[0]

    # prior_dominated propagates untouched from the exposure posterior's own derived_quantity.
    prior_dominated_exposure = _ExposurePosterior(mean=ladd, sigma=4e-5, prior_dominated=True)
    dq_prior = excess_lifetime_cancer_risk(
        prior_dominated_exposure, sf, route="oral", n=2000, rng=np.random.default_rng(2)
    )
    assert dq_prior.prior_dominated is True


def test_inhalation_route_uses_unit_risk():
    sf = SlopeFactor(inhalation_iur=4.3e-3)  # (ug/m3)^-1
    conc = 2.0  # ug/m3
    result = excess_lifetime_cancer_risk(conc, sf, route="inhalation")
    assert result.mean == pytest.approx(conc * 4.3e-3, rel=1e-9)


def test_missing_route_coefficient_raises():
    sf = SlopeFactor(oral_csf=1.5)
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(1.0, sf, route="inhalation")


def test_invalid_route_raises():
    sf = SlopeFactor(oral_csf=1.5)
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(1.0, sf, route="dermal")


def test_high_dose_uses_exact_lnt_form():
    # once dose * csf exceeds ~0.01, EPA guidance uses 1 - exp(-dose*csf), not the raw linear product.
    sf = SlopeFactor(oral_csf=50.0)
    dose = 1.0
    result = excess_lifetime_cancer_risk(dose, sf, route="oral")
    assert result.mean == pytest.approx(1.0 - np.exp(-50.0))


def test_slope_factor_log_normal_band_widens_ci():
    sf_fixed = SlopeFactor(oral_csf=1.5, sigma_log=0.0)
    sf_uncertain = SlopeFactor(oral_csf=1.5, sigma_log=0.3)
    fixed = excess_lifetime_cancer_risk(1e-4, sf_fixed, n=4000, rng=np.random.default_rng(3))
    uncertain = excess_lifetime_cancer_risk(1e-4, sf_uncertain, n=4000, rng=np.random.default_rng(3))
    lo_f, hi_f = fixed.credible_interval(0.9)
    lo_u, hi_u = uncertain.credible_interval(0.9)
    assert (hi_u - lo_u) > (hi_f - lo_f)


def test_array_exposure_samples():
    samples = np.clip(np.random.default_rng(4).normal(loc=1e-4, scale=1e-5, size=5000), 1e-12, None)
    sf = SlopeFactor(oral_csf=1.5)
    result = excess_lifetime_cancer_risk(samples, sf, route="oral")
    assert result.mean == pytest.approx(float(np.mean(samples)) * 1.5, rel=0.05)


def test_radon_wlm_risk_matches_beir_vi_coefficient():
    scalar_result = radon_wlm_risk(4.0)
    assert scalar_result.mean == pytest.approx(4.0 * 5.38e-4)

    array_result = radon_wlm_risk(np.array([1.0, 2.0, 3.0]))
    assert array_result.mean == pytest.approx(float(np.mean([1.0, 2.0, 3.0])) * 5.38e-4)

    custom_coefficient = radon_wlm_risk(4.0, risk_per_wlm=1e-3)
    assert custom_coefficient.mean == pytest.approx(4.0 * 1e-3)


def test_radon_wlm_risk_saturates_instead_of_exceeding_one():
    # A cumulative exposure large enough that the bare linear form would exceed 1 (a probability
    # cannot). risk_per_wlm chosen so wlm * risk_per_wlm = 5.38, far past where the linear
    # approximation is valid; the LNT exp form must cap it below 1.
    huge = radon_wlm_risk(10_000.0, risk_per_wlm=5.38e-4)
    assert 0.0 <= float(huge.mean) < 1.0
    assert float(huge.mean) == pytest.approx(1.0 - np.exp(-10_000.0 * 5.38e-4), rel=1e-9)


def test_radon_wlm_risk_rejects_negative_or_non_finite_inputs():
    with pytest.raises(ValueError):
        radon_wlm_risk(-1.0)
    with pytest.raises(ValueError):
        radon_wlm_risk(4.0, risk_per_wlm=-5.38e-4)
    with pytest.raises(ValueError):
        radon_wlm_risk(np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ValueError):
        radon_wlm_risk(float("nan"))
    with pytest.raises(ValueError):
        radon_wlm_risk(float("inf"))


def test_excess_lifetime_cancer_risk_rejects_negative_or_non_finite_inputs():
    sf = SlopeFactor(oral_csf=1.5)
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(-1e-4, sf, route="oral")
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(np.array([1e-4, -1e-4]), sf, route="oral")
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(1e-4, SlopeFactor(oral_csf=-1.5), route="oral")
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(float("nan"), sf, route="oral")
