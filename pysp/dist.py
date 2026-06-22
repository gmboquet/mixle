"""pysp.dist — the distribution families.

The objects: the leaf distributions, multivariate families, combinators, latent-variable models and
priors. A friendly namespace alias of :mod:`pysp.stats` during the concern-oriented reorg
(``docs/ARCHITECTURE.md``) — a re-export, so every ``pysp.stats`` import keeps working. The
cross-cutting *concerns* live in :mod:`pysp.enumeration`, :mod:`pysp.sampling`,
:mod:`pysp.inference`, and :mod:`pysp.ops`.
"""

from __future__ import annotations

from pysp.stats import *  # noqa: F401,F403  (re-export the distribution family surface)
from pysp.stats import __all__ as __all__  # noqa: PLC0414
