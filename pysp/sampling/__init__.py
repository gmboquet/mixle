"""The sampling concern — one home for drawing from a model and inferring its latent state.

Like :mod:`pysp.enumeration`, sampling is a concern shared across objects: every distribution and
relation can be sampled, latent-variable models expose a q(z|x) latent posterior, and fitted models
can draw posterior-predictive data. This module gathers the contract (:class:`DistributionSampler` /
:class:`ConditionalSampler`), the unified :func:`sample` entry point, and the
:class:`LatentPosterior` spine — a re-export shim today (nothing moves), the exemplar layout in
``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from pysp.capability import PosteriorPredictive
from pysp.sampling.latent_posterior import (
    CategoricalLatentPosterior,
    LatentPosterior,
    MarkovChainLatentPosterior,
    MeanFieldLDAPosterior,
)
from pysp.sampling.sampling_api import sample
from pysp.stats.compute.pdist import ConditionalSampler, DistributionSampler

__all__ = [
    "sample",
    "DistributionSampler",
    "ConditionalSampler",
    "PosteriorPredictive",
    "LatentPosterior",
    "CategoricalLatentPosterior",
    "MarkovChainLatentPosterior",
    "MeanFieldLDAPosterior",
]
