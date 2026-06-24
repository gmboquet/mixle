"""Design and analysis of computer experiments for pysparkplug.

This package covers the full loop of reasoning about an expensive black-box model ``f(x)`` over a
bounded input space: space-filling / classical designs, sequential Bayesian-optimization loops on top
of the existing GP and regression machinery, and the *analysis* half -- global sensitivity (which
inputs drive the output variance), forward uncertainty propagation, and Kennedy-O'Hagan calibration
to field data. (The analysis tools were previously the standalone ``pysp.uq`` package; folded in here
they share this package's quasi-Monte-Carlo sampling, GP surrogate, and kernels instead of
re-implementing them.)

The space-filling and classical design generators all return a plain ``(n, d)`` numpy matrix of
input points scaled into the supplied per-dimension bounds:

    >>> from pysp.doe import latin_hypercube
    >>> x = latin_hypercube([(0.0, 1.0), (-2.0, 2.0)], n=8, seed=0)
    >>> x.shape
    (8, 2)
"""

from __future__ import annotations

from pysp.doe.analysis import (
    FactorialEffects,
    ResponseSurface,
    design_diagnostics,
    factorial_effects,
    response_surface,
)
from pysp.doe.batch import (
    monte_carlo_qei,
    propose_local_penalization,
    propose_qei_batch,
)
from pysp.doe.bayesopt import (
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

# analysis half (folded in from the former pysp.uq package)
from pysp.doe.calibrate import KOCalibration, calibrate
from pysp.doe.constrained import (
    ConstrainedBayesOptResult,
    constrained_minimize,
    probability_of_feasibility,
    propose_next_constrained,
)
from pysp.doe.designs import (
    Bounds,
    full_factorial,
    halton_design,
    latin_hypercube,
    maximin_latin_hypercube,
    random_design,
    sobol_design,
)
from pysp.doe.factorial import (
    box_behnken,
    central_composite,
    fractional_factorial,
    plackett_burman,
)
from pysp.doe.mixture import simplex_centroid, simplex_lattice, to_pseudocomponents
from pysp.doe.multiobjective import (
    MultiObjectiveResult,
    multi_minimize,
    pareto_mask,
)
from pysp.doe.optimal import (
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
from pysp.doe.optimizer import BayesianOptimizer
from pysp.doe.propagate import propagate, register_propagator, unscented_transform
from pysp.doe.sensitivity import dgsm, fast_indices, morris_screening, sobol_indices

__all__ = [
    "Bounds",
    "full_factorial",
    "halton_design",
    "latin_hypercube",
    "maximin_latin_hypercube",
    "random_design",
    "sobol_design",
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
    # analysis half (sensitivity / propagation / calibration)
    "sobol_indices",
    "morris_screening",
    "fast_indices",
    "dgsm",
    "propagate",
    "register_propagator",
    "unscented_transform",
    "calibrate",
    "KOCalibration",
]
