Evolution And Search
====================

``mixle.evolve`` is the self-improvement layer: measure, propose, verify, and
promote. It is designed for model iteration where a candidate must earn its
way into production through a proper objective and an anti-regression gate.

The package adds orchestration. It does not replace the modeling stack. It
uses existing Mixle scoring, calibration, estimation, automatic model
selection, and decision utilities, then organizes them into repeatable
improvement loops.

The Loop
--------

The core loop has four phases:

1. Measure a champion model with an ``Objective``.
2. Propose challengers with ``ImprovementOperator`` objects.
3. Verify challenger performance on held-out data.
4. Promote only if the ``Verdict`` passes the gate.

.. code-block:: python

   from mixle.evolve import improve, nll_objective

   result = improve(
       champion,
       data,
       objective=nll_objective(),
       holdout=0.25,
       alpha=0.05,
       min_effect=0.01,
   )

   model = result.model

If ``result.verified`` is true, the returned model beat the champion under the
specified gate. If not, the champion is retained.

Objectives
----------

Objective builders include:

* ``nll_objective``;
* ``log_score_objective``;
* ``crps_objective``;
* ``interval_objective``;
* ``calibration_objective``;
* ``decision_regret_objective``.

Use likelihood objectives when the model is generative and the probability
assignment itself matters. Use calibration and interval objectives when
uncertainty quality matters. Use decision regret when the model ultimately
drives an action.

Verification
------------

``challenger_beats_champion`` compares two fitted models on the same held-out
data. The verification gate can include:

* paired objective comparison;
* a practical minimum effect size;
* calibration no-regression checks;
* non-nested model comparison for family swaps;
* multiplicity adjustment when several challengers are tried;
* optional LOO or WAIC pointwise arrays when available.

.. code-block:: python

   from mixle.evolve import challenger_beats_champion, log_score_objective

   verdict = challenger_beats_champion(
       champion,
       challenger,
       heldout,
       objective=log_score_objective(),
       nonnested=True,
   )

   if verdict.promote:
       champion = challenger

The verification step is the difference between automatic improvement and
automatic churn.

Improvement Operators
---------------------

Built-in operators include:

* ``Refit`` for fitting the same family on fresh data;
* ``OnlineUpdate`` for streaming-compatible updates;
* ``AutoSelect`` for automatic family selection;
* ``Recalibrate`` for calibration repair;
* ``Recompose`` and ``Mutate`` for structural moves, registered but expensive
  and off by default in conservative loops.

Operators advertise applicability and a cost hint. ``improve`` can use a
budget so cheap candidates are tried before expensive candidates.

Ledgers
-------

``EvolutionLedger`` records attempts, operators, deltas, costs, verdicts, and
metadata. Use it whenever an improvement loop affects a model that another
person or process will rely on.

.. code-block:: python

   from mixle.evolve import EvolutionLedger

   ledger = EvolutionLedger()
   result = improve(champion, data, objective=nll_objective(), ledger=ledger)

A ledger makes it possible to answer the important operational questions:
which candidates were tried, why were they rejected, and what evidence justified
promotion?

Automatic Selection
-------------------

``auto_select`` infers and fits a model from raw data. With ``criterion="bic"``
it delegates to automatic in-sample selection. With a proper-score objective,
it can add a held-out verification gate.

.. code-block:: python

   from mixle.evolve import auto_select, nll_objective

   result = auto_select(data, criterion=nll_objective(), verify=True)

For user-facing model design and LLM-proposed specifications, see
:doc:`automatic-inference`. ``evolve.auto_select`` is the promotion-oriented
version: it is concerned with whether the selected model should be trusted
under a gate.

Typed Search Spaces
-------------------

``Space`` describes a typed search space over ``Real``, ``Integer``, and
``Categorical`` dimensions.

.. code-block:: python

   from mixle.evolve import Categorical, Integer, Real, Space

   space = Space({
       "components": Integer(1, 6),
       "alpha": Real(0.1, 5.0, log=True),
       "family": Categorical(["gaussian", "student_t"]),
   })

The search surface is model-agnostic. You provide a ``build_fn`` that maps a
configuration dictionary to a fitted model.

.. code-block:: python

   from mixle.evolve import search, nll_objective

   result = search(
       space,
       data,
       objective=nll_objective(),
       build_fn=fit_from_config,
       method="evolutionary",
       n_iter=30,
   )

   best_model = result.best_model

Search methods include:

* ``"bo"`` for Bayesian optimization over the encoded numeric box;
* ``"evolutionary"`` for population search over samples and neighbors;
* ``"bandit"`` for an operator policy that learns which moves help.

Structure Search
----------------

``model_signature``, ``tree_edit_distance``, and ``structural_distance`` expose
distance between compositional model trees. ``Recompose`` and ``Mutate`` use
that structure to propose model changes.

This is intentionally conservative. Structural search can be powerful, but it
has high variance and a larger blast radius than recalibration or refitting.
Use it with held-out gates, ledgers, and clear budgets.

Production Standard
-------------------

Use ``mixle.evolve`` when model changes should be auditable. A mature loop
should state:

* the champion model and lineage hash;
* the objective being optimized;
* the held-out split or verification data;
* every operator tried;
* the statistical and practical promotion thresholds;
* the calibration and decision no-regression checks;
* the final verdict and ledger entry.

That standard is the path from automatic inference to automatic improvement:
models can become more capable over time without making silent regressions easy
to hide.

API Inventory
-------------

.. list-table::
   :header-rows: 1

   * - Area
     - Imports
   * - Improvement results
     - ``ImprovementResult``, ``Verdict``
   * - Operator registry
     - ``register_operator``, ``unregister_operator``, ``registered_operators``,
       ``default_operators``
   * - Search results
     - ``SearchResult``, ``Population``, ``OperatorBandit``
