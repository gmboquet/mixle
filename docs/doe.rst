Design of Experiments
=====================

``mixle.doe`` covers the design and analysis loop around expensive black-box
functions. It is useful when you cannot evaluate every input, labels are
expensive, simulation is slow, or you need to quantify how inputs drive model
outputs.

The package is organized around four phases:

1. generate an initial design;
2. fit or query a surrogate;
3. choose follow-up points by acquisition, information gain, or active learning;
4. analyze sensitivity, uncertainty propagation, or calibration.

Design Generators
-----------------

Design generators return NumPy arrays scaled to supplied bounds.

.. code-block:: python

   from mixle.doe import latin_hypercube, sobol_design

   bounds = [(0.0, 1.0), (-2.0, 2.0)]

   x_lhs = latin_hypercube(bounds, n=16, seed=0)
   x_sobol = sobol_design(bounds, n=32)

Available generators include:

* random designs;
* Latin hypercube and maximin Latin hypercube;
* Sobol and Halton sequences;
* MaxPro designs;
* full factorial, fractional factorial, and Plackett-Burman designs;
* central composite and Box-Behnken response-surface designs;
* simplex lattice and simplex centroid mixture designs;
* optimal designs under D, A, I, G, E, and c criteria.

Design Diagnostics
------------------

Use diagnostics before spending an expensive run budget:

.. code-block:: python

   from mixle.doe import design_diagnostics

   report = design_diagnostics(x_lhs)
   print(report)

Diagnostics help compare coverage, spacing, and projection behavior across
candidate designs.

Bayesian Optimization
---------------------

The Bayesian optimization layer provides standard single-point acquisitions and
batch/high-dimensional variants.

.. code-block:: python

   from mixle.doe import minimize

   def objective(x):
       return expensive_simulator(x)

   result = minimize(
       objective,
       bounds=[(0.0, 1.0), (-2.0, 2.0)],
       n_init=8,
       n_iter=24,
       seed=0,
   )

   print(result.x_best, result.y_best)

Acquisition functions include expected improvement, log expected improvement,
probability of improvement, upper confidence bound, Thompson sampling, and
knowledge gradient. Advanced routes include Monte-Carlo q-EI, local
penalization, max-value entropy search, trust-region BO, constrained BO,
multi-objective BO, and multi-fidelity BO.

Active Learning
---------------

Active learning chooses points that most improve a surrogate or parameter
estimate.

.. code-block:: python

   from mixle.doe import active_learning_design

   design = active_learning_design(
       initial_x,
       initial_y,
       bounds=[(0.0, 1.0), (-2.0, 2.0)],
       n_iter=10,
   )

Lower-level scoring functions include ``alm_scores``, ``alc_scores``,
``expected_information_gain_linear``, and ``expected_information_gain_nmc``.

Sensitivity Analysis
--------------------

Sensitivity tools quantify which inputs matter.

.. code-block:: python

   from mixle.doe import morris_screening, sobol_indices

   morris = morris_screening(model_fn, bounds, n_trajectories=20, seed=0)
   sobol = sobol_indices(model_fn, bounds, n=1024, seed=0)

Use Morris screening for a cheaper qualitative pass and Sobol indices when you
need variance-based main and interaction effects.

Uncertainty Propagation and Calibration
---------------------------------------

``propagate`` and ``unscented_transform`` push input uncertainty through a
model. ``calibrate`` implements Kennedy-O'Hagan style calibration to field data.

.. code-block:: python

   from mixle.doe import propagate, unscented_transform

   propagated = propagate(model_fn, input_distribution, n=1000, seed=0)
   approx = unscented_transform(model_fn, mean, cov)

How DOE Connects to Task Distillation
-------------------------------------

``mixle.task`` uses the same design philosophy for label acquisition. Active
distillation treats teacher calls as an expensive experiment and spends the
label budget on informative examples. Use :doc:`task-distillation` for the
task-facing workflow.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Area
     - Key imports
   * - Space-filling designs
     - ``latin_hypercube``, ``sobol_design``, ``halton_design``, ``maxpro_design``
   * - Classical designs
     - ``full_factorial``, ``fractional_factorial``, ``plackett_burman``
   * - Response-surface designs
     - ``central_composite``, ``box_behnken``, ``response_surface``
   * - Mixture designs
     - ``simplex_lattice``, ``simplex_centroid``, ``to_pseudocomponents``
   * - Optimal design
     - ``optimal_design``, ``available_criteria``, ``d_criterion``
   * - Bayesian optimization
     - ``minimize``, ``propose_next``, ``propose_batch``, ``BayesianOptimizer``
   * - Advanced BO
     - ``turbo_minimize``, ``constrained_minimize``, ``multi_minimize``
   * - Active learning
     - ``active_learning_design``, ``propose_active_learning``
   * - Analysis
     - ``sobol_indices``, ``morris_screening``, ``propagate``, ``calibrate``

Detailed API Inventory
----------------------

.. list-table::
   :header-rows: 1

   * - Area
     - Imports
   * - Bounds and random designs
     - ``Bounds``, ``random_design``, ``maximin_latin_hypercube``
   * - Factorial analysis
     - ``factorial_effects``, ``FactorialEffects``, ``polynomial_features``
   * - Response surfaces
     - ``ResponseSurface``, ``OptimizationResult``
   * - Acquisition functions
     - ``expected_improvement``, ``knowledge_gradient``,
       ``propose_knowledge_gradient``, ``log_expected_improvement``,
       ``probability_of_improvement``, ``upper_confidence_bound``,
       ``thompson_sampling``
   * - Acquisition registry
     - ``register_acquisition``, ``available_acquisitions``
   * - Optimal-design criteria
     - ``a_criterion``, ``i_criterion``, ``g_criterion``, ``e_criterion``,
       ``c_criterion``, ``register_criterion``
   * - Constrained BO
     - ``ConstrainedBayesOptResult``, ``probability_of_feasibility``,
       ``propose_next_constrained``
   * - Multi-objective and batch BO
     - ``MultiObjectiveResult``, ``pareto_mask``, ``monte_carlo_qei``,
       ``propose_qei_batch``, ``propose_local_penalization``
   * - Entropy and trust-region BO
     - ``BayesOptResult``, ``max_value_entropy_search``,
       ``sample_max_values``, ``propose_mes``, ``TrustRegion``
   * - Multi-fidelity and sensitivity
     - ``multi_fidelity_minimize``, ``fast_indices``, ``dgsm``
   * - Propagation and calibration
     - ``register_propagator``, ``KOCalibration``
