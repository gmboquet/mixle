Experimental Program API
========================

``mixle.experimental.program`` contains a move-based optimization-program API.
It is kept for research workflows that do not fit the stable declarative
surfaces yet. It is not the recommended first path for ordinary neural,
probabilistic, or task modeling.

For new code, prefer:

* ``mixle.stats`` and ``mixle.inference`` for explicit probabilistic models;
* ``mixle.ppl`` for symbolic model expressions;
* ``mixle.models`` for incubating neural leaves and applied model helpers;
* ``mixle.task`` for task replacement, distillation, cascades, tool calling,
  and planning.

The compatibility module ``mixle.program`` re-exports the experimental program
API so older imports continue to work.

When to Use It
--------------

Use the program surface when the optimization procedure is the object being
studied: alternating objectives, replay buffers, continual-learning penalties,
or adaptation routines that do not yet have a stable Mixle estimator or task
abstraction.

Do not use it merely because a model has trainable parameters. If the goal is
ordinary fitting, scoring, calibration, serving, or artifact management, the
stable estimator, PPL, model, and task APIs provide clearer contracts and
better documentation.

Core Idea
---------

The program API describes optimization as a set of moves:

``minimize(objective, over=params)``
    Minimize a closure over parameters.

``maximize(objective, over=params)``
    Maximize a closure over parameters.

``em(estimator, data, init)``
    Treat EM as a move.

``alternate(move_a, move_b, ...)``
    Alternate between moves.

``weighted([(loss_a, weight_a), ...], over=params)``
    Combine weighted objectives.

``constrain(g, bound=0.0, kind="<=")``
    Add a constraint to an optimization move.

``reinforce(sample_and_reward)``
    Construct a policy-gradient style objective.

``fit(program, ...)``
    Execute a move or program.

The surface can express useful research patterns, but it depends heavily on
closures. That is why stable Mixle workflows favor declarative estimators, PPL
expressions, and task solvers when possible.

The closure-oriented design also means reproducibility is the caller's
responsibility. Record the data iterator, optimizer settings, random seeds,
parameter selection policy, and any external module state needed to rerun the
program.

Parameter Helpers
-----------------

The module includes helpers for selecting and adapting trainable parameters:

``trainable(module)``
    Return trainable parameters from a module-like object.

``freeze(module)``
    Freeze a module's parameters.

``subset(module, *name_substrings)``
    Select parameter subsets by name.

``lora(module, rank=8, alpha=16.0)``
    Add low-rank adaptation parameters to compatible linear modules.

These helpers are useful when experimenting with neural leaves or adaptation
methods that are not yet expressed as stable Mixle model objects.

Check parameter selections before training. Substring-based selection is
convenient for experiments, but a renamed layer can silently change what is
updated. For release-like work, print or persist the selected parameter names
with the experiment record.

Examples
--------

Minimize a Torch loss:

.. code-block:: python

   from mixle.experimental.program import fit, minimize, trainable

   move = minimize(lambda: loss_fn(batch), over=trainable(net), lr=1.0e-3)
   fit(move, steps=200)

Alternate EM with a neural move:

.. code-block:: python

   from mixle.experimental.program import alternate, em, fit, minimize, trainable

   program = alternate(
       em(estimator, rows, init=model0),
       minimize(lambda: neural_loss(rows), over=trainable(net)),
   )

   fit(program, steps=20)

Adapt a module with LoRA-style parameters:

.. code-block:: python

   from mixle.experimental.program import fit, lora, minimize

   params = lora(module, rank=4)
   fit(minimize(lambda: adaptation_loss(batch), over=params), steps=100)

Advanced Moves
--------------

The module also contains experimental support for:

``Stream`` and ``ReplayBuffer``
    Streaming data and replay workflows.

``snapshot`` / ``replay`` / ``distill`` / ``ewc`` / ``fisher_diagonal``
    Continual-learning and distillation-style objectives.

``bilevel``
    Bilevel optimization experiments.

``pareto``
    Multi-objective optimization with Pareto-style moves.

``streaming_em``
    EM over a stream abstraction.

``gail`` and ``maxent_irl``
    Imitation-learning and inverse-reinforcement-learning experiments.

Operational Cautions
--------------------

Program-based workflows can be powerful, but they have fewer structural
guarantees than estimator-based workflows. Before depending on a program result:

* run a deterministic smoke test with a fixed seed;
* record optimizer settings and parameter selections;
* evaluate on data that was not used to tune the move schedule;
* compare against the closest stable Mixle workflow when one exists;
* keep program artifacts separate from production artifacts unless a promotion
  review accepts the experimental dependency.

Status and Stability
--------------------

This API is explicitly experimental:

* names and call signatures may change;
* behavior may move into ``mixle.ppl``, ``mixle.task``, or a more mature
  ``mixle.models`` surface;
* examples should be treated as exploratory workflows, not deployment guidance;
* production artifacts should prefer stable estimator and inference interfaces
  whenever possible.

Use this surface when the optimization problem is genuinely program-shaped.
Use the stable APIs when the goal is to fit, score, calibrate, explain, or
operate a model.

API Reference
-------------

* :doc:`api/mixle.experimental.program`
* :doc:`api/mixle.program`
