"""mixle.stats.univariate -- univariate distributions, split into continuous/ and discrete/.

Both subpackages re-export every class they implement, and this package re-exports both, so the
namespace exposes the API it actually contains (MXR-080-1231). Previously each level advertised
only its submodule names, so ``from mixle.stats.univariate import GaussianDistribution`` failed
while ``from mixle.stats import GaussianDistribution`` worked.

``mixle.stats`` remains the consolidated public API and re-exports these same objects.
"""

from mixle.stats.univariate import continuous, discrete
from mixle.stats.univariate.continuous import *  # noqa: F403
from mixle.stats.univariate.discrete import *  # noqa: F403

__all__ = ["continuous", "discrete"] + list(continuous.__all__) + list(discrete.__all__)
