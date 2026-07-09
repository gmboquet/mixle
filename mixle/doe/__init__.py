"""Design and analysis of computer experiments for mixle.

This package covers the full loop of reasoning about an expensive black-box model ``f(x)`` over a
bounded input space:

* **Designs** -- space-filling (Latin hypercube, maximin, Sobol'/Halton, MaxPro maximum-projection),
  classical factorial / fractional-factorial / Plackett-Burman / response-surface / mixture, and
  optimal designs (D/A/I/G/E/c criteria).
* **Bayesian optimization** -- the single-point acquisitions (EI/PI/UCB/Thompson/knowledge-gradient),
  Monte-Carlo q-EI and local-penalization *batch* design, Max-value Entropy Search,
  trust-region BO (TuRBO) for high dimensions, constrained and
  multi-objective BO, and cost-aware *multi-fidelity* BO.
* **Active learning & optimal design** -- sequential design to learn a surrogate (ALM / ALC-IMSE) or
  model parameters (expected information gain, closed-form and nested-Monte-Carlo).
* **Analysis** -- global sensitivity (Sobol'/Morris/FAST/DGSM), forward uncertainty propagation, and
  Kennedy-O'Hagan calibration to field data. (These were the standalone ``mixle.uq`` package; folded in
  here they share this package's quasi-Monte-Carlo sampling, GP surrogate, and kernels.)

The space-filling and classical design generators all return a plain ``(n, d)`` numpy matrix of
input points scaled into the supplied per-dimension bounds:

    >>> from mixle.doe import latin_hypercube
    >>> x = latin_hypercube([(0.0, 1.0), (-2.0, 2.0)], n=8, seed=0)
    >>> x.shape
    (8, 2)
"""

from __future__ import annotations

from mixle.doe.active import (
    active_learning_design,
    alc_scores,
    alm_scores,
    expected_information_gain_linear,
    expected_information_gain_nmc,
    propose_active_learning,
)
from mixle.doe.amplify import AmplificationRound, AmplifyReport, StudentTeacher, amplify_and_capture, fit_student
from mixle.doe.analysis import (
    FactorialEffects,
    ResponseSurface,
    design_diagnostics,
    factorial_effects,
    response_surface,
)
from mixle.doe.batch import (
    monte_carlo_qei,
    propose_local_penalization,
    propose_qei_batch,
)
from mixle.doe.bayesopt import (
    BayesOptResult,
    OptimizationResult,
    available_acquisitions,
    expected_improvement,
    knowledge_gradient,
    log_expected_improvement,
    minimize,
    probability_of_improvement,
    propose_batch,
    propose_knowledge_gradient,
    propose_next,
    register_acquisition,
    thompson_sampling,
    upper_confidence_bound,
)

# analysis half (folded in from the former mixle.uq package)
from mixle.doe.calibrate import KOCalibration, calibrate
from mixle.doe.constrained import (
    ConstrainedBayesOptResult,
    constrained_minimize,
    probability_of_feasibility,
    propose_next_constrained,
)
from mixle.doe.designs import (
    Bounds,
    full_factorial,
    halton_design,
    latin_hypercube,
    maximin_latin_hypercube,
    maxpro_design,
    random_design,
    sobol_design,
)
from mixle.doe.distillation import (
    DistillationDesign,
    cross_modal_distillation_design,
    distillation_design,
    multitask_distillation_design,
)
from mixle.doe.entropy import (
    max_value_entropy_search,
    propose_mes,
    sample_max_values,
)
from mixle.doe.factorial import (
    box_behnken,
    central_composite,
    fractional_factorial,
    plackett_burman,
)
from mixle.doe.mixture import simplex_centroid, simplex_lattice, to_pseudocomponents
from mixle.doe.multifidelity import multi_fidelity_minimize
from mixle.doe.multiobjective import (
    MultiObjectiveResult,
    multi_minimize,
    pareto_mask,
)
from mixle.doe.optimal import (
    a_criterion,
    available_criteria,
    c_criterion,
    d_criterion,
    e_criterion,
    g_criterion,
    i_criterion,
    optimal_design,
    polynomial_features,
    register_criterion,
)
from mixle.doe.optimizer import BayesianOptimizer
from mixle.doe.oracle import (
    VERIFIABILITY_TIERS,
    DesignCandidate,
    DesignRun,
    OracleResult,
    VerifiableOracle,
    optimize_under_oracle,
)
from mixle.doe.propagate import propagate, register_propagator, unscented_transform
from mixle.doe.robust import IncumbentResult, noisy_minimize, posterior_incumbent
from mixle.doe.sensitivity import dgsm, fast_indices, morris_screening, sobol_indices
from mixle.doe.trust_region import TrustRegion, turbo_minimize

__all__ = [
    "Bounds",
    "full_factorial",
    "halton_design",
    "latin_hypercube",
    "maximin_latin_hypercube",
    "maxpro_design",
    "random_design",
    "sobol_design",
    "DistillationDesign",
    "distillation_design",
    "multitask_distillation_design",
    "cross_modal_distillation_design",
    "fractional_factorial",
    "plackett_burman",
    "central_composite",
    "box_behnken",
    "simplex_lattice",
    "simplex_centroid",
    "to_pseudocomponents",
    "factorial_effects",
    "FactorialEffects",
    "response_surface",
    "ResponseSurface",
    "design_diagnostics",
    "OptimizationResult",
    "BayesOptResult",
    "expected_improvement",
    "knowledge_gradient",
    "propose_knowledge_gradient",
    "log_expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "thompson_sampling",
    "register_acquisition",
    "available_acquisitions",
    "minimize",
    "propose_next",
    "propose_batch",
    "VerifiableOracle",
    "OracleResult",
    "DesignCandidate",
    "DesignRun",
    "optimize_under_oracle",
    "VERIFIABILITY_TIERS",
    "amplify_and_capture",
    "AmplifyReport",
    "AmplificationRound",
    "StudentTeacher",
    "fit_student",
    "optimal_design",
    "polynomial_features",
    "d_criterion",
    "a_criterion",
    "i_criterion",
    "g_criterion",
    "e_criterion",
    "c_criterion",
    "register_criterion",
    "available_criteria",
    "ConstrainedBayesOptResult",
    "probability_of_feasibility",
    "propose_next_constrained",
    "constrained_minimize",
    "MultiObjectiveResult",
    "pareto_mask",
    "multi_minimize",
    "BayesianOptimizer",
    "monte_carlo_qei",
    "propose_qei_batch",
    "propose_local_penalization",
    "max_value_entropy_search",
    "sample_max_values",
    "propose_mes",
    "turbo_minimize",
    "TrustRegion",
    "alm_scores",
    "alc_scores",
    "propose_active_learning",
    "active_learning_design",
    "expected_information_gain_linear",
    "expected_information_gain_nmc",
    "multi_fidelity_minimize",
    # analysis half (sensitivity / propagation / calibration)
    "sobol_indices",
    "morris_screening",
    "fast_indices",
    "dgsm",
    "propagate",
    "register_propagator",
    "unscented_transform",
    "noisy_minimize",
    "posterior_incumbent",
    "IncumbentResult",
    "calibrate",
    "KOCalibration",
]
