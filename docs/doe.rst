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
   from mixle.doe.optimal import polynomial_features

   report = design_diagnostics(x_lhs, polynomial_features(degree=1))
   print(report)

Diagnostics help compare coverage, spacing, and projection behavior across
candidate designs.

Run diagnostics before evaluating the expensive function. Once budget has been
spent, a poor design cannot be repaired without changing the evidence trail.

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

Record the acquisition, surrogate, seed, bounds, and evaluated points. Bayesian
optimization evidence is a path, not only the final best value.

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

Active-learning batches should be evaluated on held-out or downstream task
metrics. A high acquisition score means the point is informative under the
surrogate, not that the resulting model improved.

Distillation and Cross-Modal Training Designs
---------------------------------------------

``mixle.doe.distillation`` treats teacher calls, human labels, and paired
cross-modal records as expensive experiments. The selectors work from a fixed
candidate pool and return the indices worth spending budget on.

Candidate-Pool Contract
~~~~~~~~~~~~~~~~~~~~~~~

A distillation design is only meaningful relative to the pool it selected
from. Keep the following fields with the design artifact:

* a stable pool identifier or data fingerprint;
* the row order used by the selector;
* exclusion rules applied before selection;
* task labels, modality labels, costs, preferences, and uncertainty vectors;
* random seed and scoring weights; and
* the selected indices plus candidate and selected scores.

Do not treat selected indices as portable row identifiers unless the pool
fingerprint and filtering rules travel with them. If a later run changes the
pool, re-run the selector rather than reusing indices from a different
candidate universe.

Use ``distillation_design`` for one pool:

.. code-block:: python

   import numpy as np

   from mixle.doe import distillation_design

   embeddings = np.asarray(pool_embeddings)
   design = distillation_design(
       embeddings,
       n=32,
       task_labels=task_names,
       modalities=modality_tags,
       uncertainty=student_uncertainty,
       cost=teacher_cost,
       task_coverage_weight=2.0,
       modality_coverage_weight=1.5,
       seed=0,
   )

   selected_examples = [pool[i] for i in design.indices]

The score balances uncertainty, diversity in feature space, task coverage,
modality coverage, preference, and cost. ``task_labels`` and ``modalities`` may
be one tag per row or several tags per row.

Keep candidate-pool identity and exclusion rules with the design. A selector
can only choose from the pool it was given, so missing or filtered candidates
are part of the evidence.

Use ``multitask_distillation_design`` when the call site should read as a
multi-task selector:

.. code-block:: python

   from mixle.doe import multitask_distillation_design

   batch = multitask_distillation_design(
       embeddings,
       24,
       task_labels=[["caption"], ["retrieval", "caption"], ["qa"]],
       uncertainty=per_task_uncertainty,
       task_weights={"caption": 2.0, "retrieval": 1.0, "qa": 1.0},
       seed=1,
   )

The returned ``DistillationDesign`` includes ``task_counts``, candidate scores,
selected scores, and metadata describing the target coverage.

Use ``cross_modal_distillation_design`` when rows may have several modality
views and the training batch should contain paired or aligned examples:

.. code-block:: python

   from mixle.doe import cross_modal_distillation_design

   paired = cross_modal_distillation_design(
       {
           "text": text_embeddings,
           "image": image_embeddings,
           "signal": signal_features,
       },
       n=20,
       task_labels=task_names,
       required_modalities=["text", "image"],
       min_modalities=2,
       alignment_weight=2.0,
       seed=2,
   )

Rows with non-finite coordinates in one modality are treated as missing for
that modality. Eligibility then follows ``min_modalities`` and
``required_modalities``. Non-finite uncertainty, preference, cost, or coverage
weights are rejected because those vectors define the ranking objective.

Record missing-modality counts and rejected ranking vectors. Cross-modal design
quality depends on which rows were eligible, not only on which rows were
selected.

Task and Modality Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Task and modality coverage terms are batch-shaping constraints, not guarantees
that a teacher is correct or a student will improve. Use them to make expensive
labels more representative, then measure the downstream student on held-out
task metrics.

For multi-task distillation:

* define task tags before looking at the selector output;
* keep task weights in the artifact when some tasks are deliberately more
  important;
* inspect ``task_counts`` for rare-task starvation; and
* separate teacher uncertainty from task coverage in the run report.

For cross-modal training:

* record which modalities were required and which were optional;
* distinguish missing modality views from non-finite ranking vectors;
* validate pair alignment before selection; and
* report held-out performance by modality availability pattern.

Interpreting Distillation Designs
---------------------------------

All distillation selectors return a ``DistillationDesign``. Treat it as part
of the experiment record, not just as a list of row numbers:

``indices``
    The candidate-pool indices selected for labeling, teacher calls, or paired
    training. These indices only make sense with the exact pool and filtering
    rules used for the run.

``scores``
    The final sequential merit values for the selected rows. Use them for
    audit and debugging, not as calibrated probabilities.

``candidate_scores``
    The base uncertainty/preference contribution before diversity, coverage,
    and cost are applied. This helps separate "the student was uncertain" from
    "the batch needed a task or modality."

``task_counts`` and ``modality_counts``
    The realized coverage of the selected batch.

``metadata``
    Target coverage, eligible row count, and the scoring weights used by the
    selector.

For ordinary ``distillation_design``, non-finite feature coordinates are
filled by a column mean only inside the standardized design geometry so that a
partly missing embedding dimension does not crash the selector. The original
candidate pool is not modified. For ``cross_modal_distillation_design``,
non-finite coordinates mark that modality as missing for that row; eligibility
then follows the required-modality and minimum-modality settings. Ranking
vectors such as ``uncertainty``, ``preference``, and ``cost`` must be finite
because they directly define the optimization objective.

Choosing a Selector
-------------------

Use the selector that matches the budget question:

* use ``distillation_design`` when every row is one candidate and task or
  modality tags are optional coverage metadata;
* use ``multitask_distillation_design`` when the call site should make the
  multi-task intent explicit, especially with per-task uncertainty or
  ``task_weights``;
* use ``cross_modal_distillation_design`` when each row can contain several
  aligned views and the batch must satisfy modality-presence constraints.

If a pool includes both high-risk and low-risk tasks, keep the task tags and
weights in the run artifact. Without those fields, a later reader cannot tell
whether the design under-sampled a rare task or whether that task was never in
scope.

Sensitivity Analysis
--------------------

Sensitivity tools quantify which inputs matter.

.. code-block:: python

   from mixle.doe import morris_screening, sobol_indices

   morris = morris_screening(model_fn, bounds, n_trajectories=20, seed=0)
   sobol = sobol_indices(model_fn, bounds, n=1024, seed=0)

Use Morris screening for a cheaper qualitative pass and Sobol indices when you
need variance-based main and interaction effects.

Sensitivity analysis should name the input distribution and parameter bounds
being analyzed. Indices are not transferable when the operating region changes.

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
``mixle.task`` owns the teacher/student artifact lifecycle. ``mixle.doe`` owns
pool selection. A common loop is:

1. embed or featurize an unlabeled pool;
2. use ``distillation_design`` or ``cross_modal_distillation_design`` to choose
   the next teacher batch;
3. call the teacher only for the selected rows;
4. train, calibrate, and profile the student with :doc:`task-distillation`;
5. feed escalations, disagreement cases, failed capability probes, or
   under-covered tasks back as ``reference_features`` or higher
   ``uncertainty`` in the next design.

This keeps distillation budgets auditable. The design result records which
coverage targets were requested, how many tasks and modalities were selected,
and what score drove each picked row.

Release Evidence
----------------

For DOE workflows, preserve:

* bounds, constraints, random seed, and initial design;
* design diagnostics before expensive evaluations;
* surrogate and acquisition settings for sequential design;
* evaluated points, failed evaluations, and timeout policy;
* candidate-pool identity, missing-modality counts, and rejected rows for
  distillation selectors;
* sensitivity or propagation assumptions; and
* the downstream metric that consumed the design.

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
