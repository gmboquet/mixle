"""mixle.process — temporal / point-process families.

The objects that model event-time data: self-exciting Hawkes processes, inhomogeneous Poisson, the
birth-death process, and the Chinese-restaurant process. The univariate event-time families share the
:class:`~mixle.capability.TemporalPointProcess` capability (``intensity`` / ``expected_count``). A
re-export namespace pulling these out of ``stats/leaf`` so the temporal models are findable in one
place (``docs/ARCHITECTURE.md``).
"""

from __future__ import annotations

from mixle.stats.processes.birth_death import BirthDeathSamplingDistribution
from mixle.stats.processes.chinese_restaurant_process import ChineseRestaurantProcessDistribution
from mixle.stats.processes.hawkes_process import HawkesProcessDistribution
from mixle.stats.processes.inhomogeneous_poisson import InhomogeneousPoissonProcessDistribution
from mixle.stats.processes.multivariate_hawkes import MultivariateHawkesProcessDistribution
from mixle.stats.processes.power_law_hawkes import PowerLawHawkesDistribution
from mixle.stats.processes.renewal_process import RenewalProcessDistribution

__all__ = [
    "HawkesProcessDistribution",
    "PowerLawHawkesDistribution",
    "MultivariateHawkesProcessDistribution",
    "InhomogeneousPoissonProcessDistribution",
    "RenewalProcessDistribution",
    "BirthDeathSamplingDistribution",
    "ChineseRestaurantProcessDistribution",
]
