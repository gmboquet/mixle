"""Small MCMC utilities over mixle log-density objects.

The low-level functions here deliberately operate on user-supplied log-target
callables and proposal objects.  That keeps the transition machinery orthogonal
to the distribution, estimator, and compute-engine protocols while still making
ordinary ``dist.log_density(x)`` models easy to sample from.

On top of that machinery this module also provides a high-level parameter
posterior API (:func:`sample_parameter_posterior`,
:func:`sample_conjugate_posterior`).  Given a prototype distribution that fixes
the model family, a dataset, and a prior over parameters, it samples
``p(theta | data) proportional to exp(sum_i log p(x_i | theta)) * prior(theta)``
by running Metropolis-Hastings or Hamiltonian Monte Carlo in an unconstrained
reparameterization (log for positive scales, stick-breaking for probability
simplices) and mapping the retained samples back to parameter space (or to
rebuilt distribution objects).
"""

from __future__ import annotations

from .conjugate import sample_conjugate_posterior
from .gradients import torch_available, torch_gradient, value_and_torch_gradient
from .parameter_bridge import (
    ParameterBridge,
    build_parameter_bridge,
    sample_parameter_posterior,
)
from .proposals import (
    AdaptiveCovarianceProposal,
    AdaptiveRandomWalkProposal,
    BlockProposal,
    IndependentProposal,
    LangevinProposal,
    MixtureProposal,
    Proposal,
    RandomWalkProposal,
)
from .samplers import (
    MCMCResult,
    affine_invariant_ensemble,
    dense_mass_hmc,
    distribution_log_target,
    gelman_rubin,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    metropolis_within_gibbs,
    nuts,
    particle_filter,
    posterior_predictive,
    reflective_hmc,
    run_chains,
    sample_distribution,
)

__all__ = [
    "AdaptiveCovarianceProposal",
    "AdaptiveRandomWalkProposal",
    "BlockProposal",
    "IndependentProposal",
    "LangevinProposal",
    "MCMCResult",
    "MixtureProposal",
    "ParameterBridge",
    "Proposal",
    "RandomWalkProposal",
    "affine_invariant_ensemble",
    "build_parameter_bridge",
    "distribution_log_target",
    "gelman_rubin",
    "hamiltonian_monte_carlo",
    "dense_mass_hmc",
    "particle_filter",
    "reflective_hmc",
    "metropolis_hastings",
    "metropolis_within_gibbs",
    "nuts",
    "posterior_predictive",
    "run_chains",
    "sample_conjugate_posterior",
    "sample_distribution",
    "sample_parameter_posterior",
    "torch_available",
    "torch_gradient",
    "value_and_torch_gradient",
]
