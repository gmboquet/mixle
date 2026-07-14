"""Applied statistical methods that are not probability-distribution families.

These are estimators / analysis routines (extreme-value, kernel density, species coverage, geostatistics,
rank aggregation, spatial mixtures, max-stable processes, covariance shrinkage, mine economics) that
operate on data but are not ``SequenceEncodableProbabilityDistribution`` families and are not used by any
distribution. They were previously scattered under ``mixle.stats``; collected here so the distributions
package stays focused on distribution families.
"""

from __future__ import annotations

from mixle.analysis.biodiversity import habitat_offset_liability, no_net_loss_constraint
from mixle.analysis.covariance_shrinkage import LedoitWolfEstimator
from mixle.analysis.coverage import (
    ace,
    chao1,
    chao2,
    good_turing,
    hill_numbers,
    ice,
    rarefaction_curve,
    turing_coverage,
)
from mixle.analysis.emissions import EmissionFactors, Footprint, emissions_footprint
from mixle.analysis.extreme import (
    GPDFit,
    endpoint_estimator,
    gpd_fit,
    hill_estimator,
    mean_residual_life,
    moment_estimator,
    n_records,
    peaks_over_threshold,
    record_times,
    return_level,
)
from mixle.analysis.health_risk import (
    DOSE_RESPONSE_MODELS,
    DoseResponse,
    cumulative_exposure,
    exposure_constraints,
    health_liability,
    population_risk,
)
from mixle.analysis.kde import (
    KDE,
    intensity,
    kde,
    kde_mode,
    scott_bandwidth,
    silverman_bandwidth,
)
from mixle.analysis.kriging import (
    Variogram,
    calibrate_variance,
    empirical_variogram,
    fit_variogram,
    ordinary_kriging,
    universal_kriging,
)
from mixle.analysis.max_stable import SmithMaxStable, SmithMaxStableSampler, fit_smith_maxstable
from mixle.analysis.objective import hard_constraints, priced_liabilities
from mixle.analysis.rank_aggregation import (
    borda_count,
    cayley_distance,
    copeland,
    kemeny_consensus,
    kendall_distance,
    mallows_fit,
    spearman_footrule,
)
from mixle.analysis.real_options import OptionValue, real_option_value, voi_dollars
from mixle.analysis.sdm import HabitatModel, SpeciesObservation, fit_sdm
from mixle.analysis.spatial_mixture import SpatialMixture
from mixle.analysis.valuation import NPVDistribution, capex_opex, cost_curve, monte_carlo_npv

__all__ = [
    # extreme-value & boundary estimation (GPD/POT, tail index, endpoints, records)
    "GPDFit",
    "gpd_fit",
    "peaks_over_threshold",
    "return_level",
    "hill_estimator",
    "moment_estimator",
    "mean_residual_life",
    "endpoint_estimator",
    "n_records",
    "record_times",
    # dose-response / health-risk models (K3)
    "DOSE_RESPONSE_MODELS",
    "DoseResponse",
    "cumulative_exposure",
    "population_risk",
    # health/safety liability + exposure-limit hard constraints into J6/H4 (K6)
    "health_liability",
    "exposure_constraints",
    # kernel density estimation
    "KDE",
    "kde",
    "kde_mode",
    "intensity",
    "silverman_bandwidth",
    "scott_bandwidth",
    # species / coverage estimation
    "turing_coverage",
    "good_turing",
    "chao1",
    "chao2",
    "ace",
    "ice",
    "hill_numbers",
    "rarefaction_curve",
    # geostatistics: variograms and kriging
    "Variogram",
    "empirical_variogram",
    "fit_variogram",
    "ordinary_kriging",
    "universal_kriging",
    "calibrate_variance",
    # rank aggregation
    "borda_count",
    "copeland",
    "kemeny_consensus",
    "mallows_fit",
    "kendall_distance",
    "spearman_footrule",
    "cayley_distance",
    # spatial mixture / max-stable processes / covariance shrinkage
    "SpatialMixture",
    "SmithMaxStable",
    "SmithMaxStableSampler",
    "fit_smith_maxstable",
    "LedoitWolfEstimator",
    # mine economics: cost curves, capex/opex roll-up, and Monte-Carlo DCF valuation
    "cost_curve",
    "capex_opex",
    "NPVDistribution",
    "monte_carlo_npv",
    # emissions / carbon accounting (Scope 1/2/3 GHG footprint)
    "EmissionFactors",
    "Footprint",
    "emissions_footprint",
    # reclamation ecology / biodiversity offsets (priced habitat-offset liability + no-net-loss constraint)
    "habitat_offset_liability",
    "no_net_loss_constraint",
    # species-distribution / habitat-suitability modelling (IC-12)
    "SpeciesObservation",
    "HabitatModel",
    "fit_sdm",
    # economic objective integration (J6): priced liabilities + hard constraints for
    # mixle.stochastic_opt.risk_adjusted_plan
    "priced_liabilities",
    "hard_constraints",
    # real options & decision-under-uncertainty
    "OptionValue",
    "real_option_value",
    "voi_dollars",
]
