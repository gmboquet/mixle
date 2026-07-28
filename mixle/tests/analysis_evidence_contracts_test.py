"""Evidence contracts across ``mixle.analysis``: identifiability, immutability, exact flags, levels.

Five findings from the 0.8.0 exhaustive review, all of the same shape: an analysis surface produced a
confident-looking answer where the evidence did not support one, or let a validated invariant lapse
after construction.

- MXR-080-1575/1576 (coverage): the least informative sampling designs -- one sampling unit, or an
  all-singleton sample -- reported richness *exactly equal to the observed count* with zero standard
  error, because the estimators' correction terms degenerate to zero rather than blowing up.
- MXR-080-1585 (extreme): one NaN silently truncated a record series instead of raising.
- MXR-080-1588 (health/habitat): ``bool("false")`` is ``True``, so a policy or honesty flag read from
  serialized text could invert the safety decision it names.
- MXR-080-1595 (kriging): ``Variogram`` validated its parameters at construction, then let a caller
  overwrite them; every solver reads the fields without re-checking.
- MXR-080-1430 (max_stable): an empty location matrix built a sampler that failed much later inside
  a NumPy reduction.
- MXR-080-1580 (analysis-wide): the IC-1 carriers disagreed about what a valid interval level is.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mixle.analysis._interval import validated_level
from mixle.analysis.carcinogenic_risk import RiskQuantity
from mixle.analysis.coverage import CoverageInsufficientDataError, ace, chao2, ice
from mixle.analysis.developmental_risk import _SampleDerivedQuantity as _DevQuantity
from mixle.analysis.extreme import n_records, record_times
from mixle.analysis.health_risk import (
    ExceedanceReport,
    _DeterministicRisk,
    _SampleDerivedQuantity,
    exposure_constraints,
)
from mixle.analysis.kriging import Variogram
from mixle.analysis.max_stable import SmithMaxStable, SmithMaxStableSampler


class ReplicationRequiredForUnseenRichnessTest:
    """MXR-080-1575: Chao2/ICE estimate unseen richness *from* replication across sampling units."""

    def test_single_site_incidence_is_rejected_not_reported_as_exact(self):
        one_site = np.array([[1], [1]])  # two species, one unit
        with pytest.raises(CoverageInsufficientDataError, match="independent sampling units"):
            chao2(one_site)
        with pytest.raises(CoverageInsufficientDataError, match="independent sampling units"):
            ice(one_site)

    def test_two_sites_still_estimate(self):
        two_sites = np.array([[1, 0], [1, 1], [0, 1]])
        result = chao2(two_sites)
        assert result["sites"] == 2.0
        assert result["estimate"] >= result["observed"]


class ZeroEstimatedCoverageUnidentifiableTest:
    """MXR-080-1576: at zero estimated coverage the correction is undefined, not zero."""

    def test_all_singleton_abundance_sample_is_rejected(self):
        # c_ace == 0: every rare individual is a singleton. Returning `estimate == observed` gave the
        # least complete sample imaginable no unseen-species correction at all.
        with pytest.raises(CoverageInsufficientDataError, match="coverage is zero"):
            ace(np.array([1, 1, 1, 1, 1]))

    def test_identity_incidence_matrix_is_rejected(self):
        with pytest.raises(CoverageInsufficientDataError, match="coverage is zero"):
            ice(np.eye(4, dtype=int))

    def test_no_rare_species_still_reports_the_observed_count(self):
        # Negative control: with no rare group at all there is genuinely nothing to correct, so
        # `estimate == observed` is the right answer and must not be swept into the raise above.
        result = ace(np.array([50, 60, 70]), rare_threshold=10)
        assert result["s_rare"] == 0.0
        assert result["estimate"] == result["observed"] == 3.0


class RecordSeriesRejectsNonFiniteTest:
    """MXR-080-1585: NaN poisons `maximum.accumulate`, silently deleting every later record."""

    def test_nan_raises_instead_of_truncating(self):
        with pytest.raises(ValueError, match="finite"):
            record_times(np.array([1.0, np.nan, 2.0, 3.0]))
        with pytest.raises(ValueError, match="finite"):
            n_records(np.array([1.0, np.inf, 2.0]))

    def test_finite_series_still_finds_every_record(self):
        assert record_times(np.array([1.0, 0.5, 2.0, 3.0])).tolist() == [0, 2, 3]
        assert record_times(np.array([])).size == 0

    def test_multidimensional_series_is_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            record_times(np.array([[1.0, 2.0], [3.0, 4.0]]))


class PolicyFlagsRequireActualBooleansTest:
    """MXR-080-1588: `bool("false")` is True -- a serialized flag could invert a safety decision."""

    def test_safety_screen_rejects_a_string_policy_flag(self):
        options = [{"dust": 1.0}]
        limits = {"dust": 2.0, "noise": 85.0}  # "noise" unmodeled -> status "unknown"
        with pytest.raises(TypeError, match="treat_unmodeled_as_safe"):
            exposure_constraints(options, limits, treat_unmodeled_as_safe="false")

        # Fails closed by default, and the real Boolean override still works.
        assert exposure_constraints(options, limits)[0]["feasible"] is False
        assert exposure_constraints(options, limits, treat_unmodeled_as_safe=True)[0]["feasible"] is True

    def test_exceedance_report_calibration_flag_must_be_boolean(self):
        with pytest.raises(TypeError, match="calibrated"):
            ExceedanceReport(
                alerts=np.array([True]),
                prob_exceed=np.array([0.9]),
                false_alarm_target=0.05,
                calibrated="false",
            )

    def test_risk_quantity_honesty_flag_must_be_boolean(self):
        with pytest.raises(TypeError, match="Boolean"):
            RiskQuantity(samples=np.array([0.1]), prior_dominated="false")


class VariogramInvariantsSurviveConstructionTest:
    """MXR-080-1595: every kriging solver trusts these fields without revalidating them."""

    def test_fitted_variogram_is_frozen(self):
        variogram = Variogram(model="spherical", nugget=0.1, psill=1.0, rng=2.0)
        assert dataclasses.is_dataclass(Variogram) and Variogram.__dataclass_params__.frozen
        for field, invalid in (("rng", 0.0), ("psill", -1.0), ("model", "not_a_model"), ("nugget", -0.5)):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(variogram, field, invalid)
        assert variogram.rng == 2.0

    def test_anisotropy_is_normalized_to_an_immutable_pair(self):
        variogram = Variogram(model="exponential", nugget=0.0, psill=1.0, rng=1.0, anisotropy=[0.5, 2.0])
        assert variogram.anisotropy == (0.5, 2.0)
        assert isinstance(variogram.anisotropy, tuple)


class MaxStableSamplerRequiresLocationsTest:
    """MXR-080-1430: an empty (0, d) matrix built a sampler that failed later inside NumPy."""

    def test_empty_location_matrix_is_rejected_at_construction(self):
        process = SmithMaxStable(np.eye(2))
        with pytest.raises(ValueError, match="non-empty"):
            SmithMaxStableSampler(process, np.zeros((0, 2)))

    def test_location_dimension_must_match_the_process(self):
        process = SmithMaxStable(np.eye(2))
        with pytest.raises(ValueError, match="coordinate"):
            SmithMaxStableSampler(process, np.zeros((3, 5)))
        with pytest.raises(ValueError, match="finite"):
            SmithMaxStableSampler(process, np.array([[0.0, np.nan]]))

    def test_valid_locations_still_construct(self):
        process = SmithMaxStable(np.eye(2))
        sampler = SmithMaxStableSampler(process, np.array([[0.0, 0.0], [1.0, 1.0]]), seed=0)
        assert sampler.loc.shape == (2, 2)


class OneSharedIntervalLevelContractTest:
    """MXR-080-1580: sibling IC-1 carriers disagreed about what a valid level is."""

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 5.0, float("nan"), float("inf")])
    def test_out_of_range_levels_are_rejected(self, bad):
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            validated_level(bad)

    @pytest.mark.parametrize("bad", [True, np.bool_(False), np.array([0.9])])
    def test_boolean_and_array_levels_are_rejected(self, bad):
        with pytest.raises(TypeError, match="real scalar probability"):
            validated_level(bad)

    def test_deterministic_carrier_validates_the_level_it_does_not_use(self):
        # The point-interval carrier ignored `level` entirely, so `level=5.0` succeeded here while the
        # sibling sample-based carriers raised on it -- exactly how a bad level went unnoticed.
        risk = _DeterministicRisk(samples=np.array([[0.0, 1.0]]), grid_shape=(2,))
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            risk.credible_interval(5.0)
        lo, hi = risk.credible_interval(0.9)
        assert np.array_equal(lo, hi)

    @pytest.mark.parametrize(
        "carrier",
        [
            lambda: _SampleDerivedQuantity(samples=np.array([1.0, 2.0, 3.0])),
            lambda: _DevQuantity(samples=np.array([1.0, 2.0, 3.0])),
            lambda: RiskQuantity(samples=np.array([0.1, 0.2, 0.3])),
        ],
        ids=["health", "developmental", "cancer"],
    )
    def test_sample_carriers_share_the_same_boundary(self, carrier):
        quantity = carrier()
        with pytest.raises((ValueError, TypeError)):
            quantity.credible_interval(float("nan"))
        with pytest.raises((ValueError, TypeError)):
            quantity.credible_interval(1.5)
        lo, hi = quantity.credible_interval(0.9)
        assert np.all(np.asarray(lo) <= np.asarray(hi))
