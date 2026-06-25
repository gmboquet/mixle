"""Applied statistical methods that are not probability-distribution families.

These are estimators / analysis routines (extreme-value, kernel density, species coverage, geostatistics,
rank aggregation, spatial mixtures, max-stable processes, covariance shrinkage) that operate on data but
are not ``SequenceEncodableProbabilityDistribution`` families and are not used by any distribution. They
were previously scattered under ``pysp.stats``; collected here so the distributions package stays focused
on distribution families.
"""

from __future__ import annotations

from pysp.analysis.covariance_shrinkage import LedoitWolfEstimator
from pysp.analysis.coverage import (
    ace,
    chao1,
    chao2,
    good_turing,
    hill_numbers,
    ice,
    rarefaction_curve,
    turing_coverage,
)
from pysp.analysis.extreme import (
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
from pysp.analysis.kde import (
    KDE,
    intensity,
    kde,
    kde_mode,
    scott_bandwidth,
    silverman_bandwidth,
)
from pysp.analysis.kriging import (
    Variogram,
    calibrate_variance,
    empirical_variogram,
    fit_variogram,
    ordinary_kriging,
    universal_kriging,
)
from pysp.analysis.max_stable import SmithMaxStable, SmithMaxStableSampler, fit_smith_maxstable
from pysp.analysis.rank_aggregation import (
    borda_count,
    cayley_distance,
    copeland,
    kemeny_consensus,
    kendall_distance,
    mallows_fit,
    spearman_footrule,
)
from pysp.analysis.spatial_mixture import SpatialMixture

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
]
