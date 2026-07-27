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


class _RawDrawPosterior:
    """A minimal IC-1 ``Posterior`` whose ``samples()`` returns caller-supplied draws verbatim.

    Used to check that ``excess_lifetime_cancer_risk``'s pushforward validates a ``Posterior``'s own
    draws exactly like a plain array (MXR-080-0074), by handing it draws a real posterior should
    never produce: more than one value per sample, or negative/non-finite values.
    """

    def __init__(self, draws: np.ndarray):
        self._draws = np.asarray(draws, dtype=float)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._draws

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self._draws, axis=0)

    @property
    def cov(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self._draws, rowvar=False))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        alpha = (1.0 - level) / 2.0
        return np.quantile(self._draws, alpha, axis=0), np.quantile(self._draws, 1.0 - alpha, axis=0)

    def derived_quantity(self, fn, n: int, rng: np.random.Generator) -> _ExposureDerivedQuantity:
        return _ExposureDerivedQuantity(fn(self.samples(n, rng)), False)


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


def test_multidimensional_exposure_array_rejected_not_truncated():
    """MXR-080-0074: a multi-dimensional exposure draw must be rejected, not silently truncated to
    column 0 (which previously discarded every other route/chemical/time point/receptor).

    Column 0 is a tiny dose; columns 1/2 are huge. Under the old column-0-only truncation this call
    would have silently succeeded with the tiny-dose answer, hiding the huge exposure entirely.
    """
    sf = SlopeFactor(oral_csf=1.5)
    multidim = np.tile([1e-4, 10.0, 10.0], (5, 1))
    assert multidim.shape == (5, 3)
    with pytest.raises(ValueError, match=r"single value per sample"):
        excess_lifetime_cancer_risk(multidim, sf, route="oral")


def test_posterior_multidimensional_draws_rejected():
    """MXR-080-0074: a ``Posterior`` handing back more than one value per draw is rejected the same
    way a plain multi-dimensional array is -- not silently truncated to its first column."""
    sf = SlopeFactor(oral_csf=1.5)
    draws = np.tile([1e-4, 10.0, 10.0], (9, 1))
    with pytest.raises(ValueError, match=r"single value per sample"):
        excess_lifetime_cancer_risk(_RawDrawPosterior(draws), sf, route="oral", n=9)


def test_posterior_draws_validated_same_as_plain_array():
    """MXR-080-0074: a ``Posterior``'s own draws are now validated exactly like a plain array's --
    previously they were explicitly exempted from the finite/non-negative check, so a mis-specified
    exposure posterior with mass below zero (or a NaN draw) could silently yield an invalid "risk"
    sample. Both must now raise."""
    sf = SlopeFactor(oral_csf=1.5)
    negative_draws = np.array([[-5.0], [1e-4], [2e-4]])
    with pytest.raises(ValueError, match=r"exposure"):
        excess_lifetime_cancer_risk(_RawDrawPosterior(negative_draws), sf, route="oral", n=3)

    nan_draws = np.array([[np.nan], [1e-4], [2e-4]])
    with pytest.raises(ValueError, match=r"exposure"):
        excess_lifetime_cancer_risk(_RawDrawPosterior(nan_draws), sf, route="oral", n=3)


def test_posterior_with_legitimate_single_column_draws_still_works():
    """Negative control for MXR-080-0074: the IC-1 ``Posterior.samples`` contract is always shape
    ``(n, d)``; ``d == 1`` (a single quantity per sample, wrapped per-protocol) is legitimate and must
    keep working -- not be caught by the new multi-dimensional rejection."""
    sf = SlopeFactor(oral_csf=1.5)
    draws = np.full((6, 1), 1e-4)
    result = excess_lifetime_cancer_risk(_RawDrawPosterior(draws), sf, route="oral", n=6)
    assert isinstance(result, RiskQuantity)
    assert result.mean == pytest.approx(1e-4 * 1.5, rel=1e-9)


def test_slope_factor_rejects_negative_sigma_log():
    """MXR-080-0075: a negative ``sigma_log`` has no valid meaning as a standard deviation. It
    previously failed the ``> 0`` branch in ``_apply`` and was silently treated as zero (fixed)
    uncertainty; it must now raise at construction instead."""
    with pytest.raises(ValueError):
        SlopeFactor(oral_csf=1.5, sigma_log=-0.3)


def test_slope_factor_rejects_nan_sigma_log():
    """MXR-080-0075: NaN fails the old ``> 0`` check the same way a negative value does (any
    comparison with NaN is False), so it was silently treated as zero uncertainty too; must now raise
    at construction."""
    with pytest.raises(ValueError):
        SlopeFactor(oral_csf=1.5, sigma_log=float("nan"))


def test_slope_factor_is_immutable_after_validation():
    sf = SlopeFactor(oral_csf=1.5, sigma_log=0.1)
    with pytest.raises((AttributeError, TypeError)):
        sf.sigma_log = -0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"oral_csf": np.array([1.0, 2.0])},
        {"inhalation_iur": np.array([1.0, 2.0])},
        {"sigma_log": np.array([0.1, 0.2])},
        {"oral_csf": True},
        {"sigma_log": False},
    ],
)
def test_slope_factor_fields_require_real_scalars(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SlopeFactor(**kwargs)


def test_excess_lifetime_cancer_risk_rejects_non_positive_n():
    """MXR-080-0075: ``n=0`` previously produced an empty risk-sample array whose ``mean``/credible
    interval are invalid (NaN with a RuntimeWarning, or an outright crash on an empty quantile);
    ``n<0`` is equally nonsensical as a draw count. Both must now raise clearly, regardless of whether
    ``sf.sigma_log`` would have made ``n`` actually matter downstream."""
    sf_fixed = SlopeFactor(oral_csf=1.5)  # sigma_log=0.0: n was previously silently ignored here too
    sf_uncertain = SlopeFactor(oral_csf=1.5, sigma_log=0.2)
    for sf in (sf_fixed, sf_uncertain):
        with pytest.raises(ValueError):
            excess_lifetime_cancer_risk(1e-4, sf, route="oral", n=0)
        with pytest.raises(ValueError):
            excess_lifetime_cancer_risk(1e-4, sf, route="oral", n=-5)


def test_excess_lifetime_cancer_risk_rejects_fractional_n():
    """MXR-080-0075: ``n`` must be an exact integer draw count."""
    sf = SlopeFactor(oral_csf=1.5, sigma_log=0.2)
    with pytest.raises(ValueError):
        excess_lifetime_cancer_risk(1e-4, sf, route="oral", n=2.5)


def test_risk_quantity_rejects_empty_samples():
    """MXR-080-0075: ``RiskQuantity`` must reject an empty sample array at construction -- defense in
    depth so invalid state can never flow downstream even if an upstream call site's own validation
    is skipped or buggy."""
    with pytest.raises(ValueError):
        RiskQuantity(samples=np.array([]))


def test_risk_quantity_rejects_non_finite_samples():
    """MXR-080-0075: NaN/inf samples must be rejected at construction."""
    with pytest.raises(ValueError):
        RiskQuantity(samples=np.array([0.1, np.nan, 0.2]))
    with pytest.raises(ValueError):
        RiskQuantity(samples=np.array([0.1, np.inf, 0.2]))


def test_risk_quantity_rejects_out_of_range_samples():
    """MXR-080-0075: a risk quantity is probability-like -- samples must be in [0, 1]."""
    with pytest.raises(ValueError):
        RiskQuantity(samples=np.array([0.1, -0.5, 0.2]))
    with pytest.raises(ValueError):
        RiskQuantity(samples=np.array([0.1, 1.5, 0.2]))


def test_risk_quantity_requires_one_immutable_owned_sample_axis():
    source = np.array([0.1, 0.2, 0.3])
    quantity = RiskQuantity(samples=source)
    source[0] = 0.9
    assert quantity.samples[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        quantity.samples[0] = 0.9
    with pytest.raises((AttributeError, TypeError)):
        quantity.samples = np.array([0.2])
    with pytest.raises(ValueError, match="one-dimensional"):
        RiskQuantity(samples=np.ones((2, 2)))
    with pytest.raises(TypeError, match="Boolean"):
        RiskQuantity(samples=np.array([0.1]), prior_dominated="no")


def test_risk_quantity_accepts_valid_samples():
    """Negative control for MXR-080-0075: a legitimate, in-range sample array (including the boundary
    values 0.0 and 1.0) still constructs a working RiskQuantity."""
    rq = RiskQuantity(samples=np.array([0.0, 0.25, 0.5, 1.0]))
    assert rq.mean == pytest.approx(0.4375)
    lo, hi = rq.credible_interval(0.9)
    assert 0.0 <= lo <= hi <= 1.0


def test_excess_lifetime_cancer_risk_valid_sigma_log_and_n_produce_sensible_riskquantity():
    """Negative control for MXR-080-0075: legitimate scalar exposure with a valid (positive, finite)
    ``sigma_log`` and a valid positive-integer ``n`` still produce a well-formed ``RiskQuantity``
    end-to-end."""
    sf = SlopeFactor(oral_csf=1.5, sigma_log=0.2)
    result = excess_lifetime_cancer_risk(1e-4, sf, route="oral", n=500, rng=np.random.default_rng(7))
    assert isinstance(result, RiskQuantity)
    assert result.samples.shape == (500,)
    assert np.all(np.isfinite(result.samples))
    assert np.all((result.samples >= 0) & (result.samples <= 1))
    lo, hi = result.credible_interval(0.9)
    assert 0.0 <= lo <= hi <= 1.0
