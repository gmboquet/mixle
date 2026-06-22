"""pysp.process — temporal / point-process families.

The objects that model event-time data: self-exciting Hawkes processes, inhomogeneous Poisson, the
birth-death process, and the Chinese-restaurant process. The univariate event-time families share the
:class:`~pysp.capability.TemporalPointProcess` capability (``intensity`` / ``expected_count``). A
re-export namespace pulling these out of ``stats/leaf`` so the temporal models are findable in one
place (``docs/ARCHITECTURE.md``).
"""

from __future__ import annotations

from pysp.stats.base.birth_death import BirthDeathSamplingDistribution
from pysp.stats.base.chinese_restaurant_process import ChineseRestaurantProcessDistribution
from pysp.stats.base.hawkes_process import HawkesProcessDistribution
from pysp.stats.base.inhomogeneous_poisson import InhomogeneousPoissonProcessDistribution
from pysp.stats.base.multivariate_hawkes import MultivariateHawkesProcessDistribution
from pysp.stats.base.power_law_hawkes import PowerLawHawkesDistribution
from pysp.stats.base.renewal_process import RenewalProcessDistribution

__all__ = [
    "HawkesProcessDistribution",
    "PowerLawHawkesDistribution",
    "MultivariateHawkesProcessDistribution",
    "InhomogeneousPoissonProcessDistribution",
    "RenewalProcessDistribution",
    "BirthDeathSamplingDistribution",
    "ChineseRestaurantProcessDistribution",
]
