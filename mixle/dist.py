"""mixle.dist — the distribution families.

The objects: the leaf distributions, multivariate families, combinators, latent-variable models and
priors. A friendly namespace alias of :mod:`mixle.stats` during the concern-oriented reorg
(``docs/ARCHITECTURE.md``) — a re-export, so every ``mixle.stats`` import keeps working. The
cross-cutting *concerns* live in :mod:`mixle.enumeration`, :mod:`mixle.inference`,
and :mod:`mixle.ops` (sampling is intrinsic behavior, not a concern: ``mixle.stats.sample``).
"""

from __future__ import annotations

from mixle.stats import *  # noqa: F401,F403  (re-export the distribution family surface)
from mixle.stats import __all__ as __all__  # noqa: PLC0414
