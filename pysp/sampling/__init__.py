"""Deprecated shim: sampling is no longer a standalone concern.

Drawing from a model is intrinsic behavior (every model exposes ``.sampler()``), not a concern with
its own input precondition, so the package was dissolved and its parts re-filed by function:

* the ``Posterior`` / ``LatentPosterior`` hierarchy now lives in :mod:`pysp.stats.compute.posterior`
  (the compute layer, beside the sampler contracts);
* the ``sample()`` facade is :func:`pysp.stats.sample`;
* the parameter / predictive posteriors and the ``posterior(model, ...)`` factory live in
  :mod:`pysp.inference.posterior` (inference *produces* posteriors).

This module re-exports the former names for backward compatibility; import from the homes above.
"""

from __future__ import annotations

from pysp.capability import PosteriorPredictive
from pysp.stats.compute.pdist import ConditionalSampler, DistributionSampler
from pysp.stats.compute.posterior import (
    CategoricalLatentPosterior,
    LatentPosterior,
    MarkovChainLatentPosterior,
    MeanFieldLDAPosterior,
    Posterior,
)
from pysp.stats.sampling_api import sample

__all__ = [
    "sample",
    "DistributionSampler",
    "ConditionalSampler",
    "PosteriorPredictive",
    "Posterior",
    "LatentPosterior",
    "CategoricalLatentPosterior",
    "MarkovChainLatentPosterior",
    "MeanFieldLDAPosterior",
]
