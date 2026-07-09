Evolution and Analysis
======================

This tutorial connects analysis diagnostics to an auditable improvement loop.
It is for cases where you already have a working model and want controlled
change, not blind hyperparameter search.

Start With a Champion
---------------------

Assume a champion model already exists. The improvement loop needs data, an
objective, and a verification standard.

.. code-block:: python

   from mixle.evolve import EvolutionLedger, improve, nll_objective

   ledger = EvolutionLedger()

   result = improve(
       champion,
       data,
       objective=nll_objective(),
       holdout=0.25,
       alpha=0.05,
       min_effect=0.01,
       ledger=ledger,
   )

   champion = result.model

The returned model is only a verified challenger if ``result.verified`` is
true. Otherwise the original champion remains in place.

Record both outcomes. A rejected challenger is useful evidence: it tells future
searches which direction did not clear the gate.

Keep the champion fixed while a challenger is evaluated. Search state,
diagnostics, and validation results should be attached to the challenger record
instead of mutating the deployed model in place. That distinction is what makes
the ledger useful after the run is over.

Add a Tail Diagnostic
---------------------

Likelihood can improve while tail behavior gets worse. Use analysis utilities
to inspect the residuals or losses that matter operationally.

.. code-block:: python

   import numpy as np
   from mixle.analysis import peaks_over_threshold, return_level

   residuals = np.asarray([abs(y - champion.predict(x)) for x, y in validation])
   tail = peaks_over_threshold(residuals, threshold=np.quantile(residuals, 0.95))
   level = return_level(tail, period=100)

You can track this diagnostic in the ledger metadata or use it to define a
custom objective.

The reason to keep this separate from the primary likelihood objective is
governance. A model can improve average log score while becoming worse exactly
where the application is most sensitive.

Use the same validation slice for the tail diagnostic that you use for the
promotion decision, or record why a different slice is appropriate. Tail
estimates can be noisy; for high-impact applications, report uncertainty or
repeat the diagnostic across time windows before treating it as a hard gate.

Define a Promotion Gate
-----------------------

A promotion decision should combine the objective result and the diagnostics
that matter for the application.

.. code-block:: python

   passed = (
       result.verified
       and level < champion_tail_limit
       and result.delta >= 0.01
   )

   if passed:
       champion = result.model
       ledger.record(
           operator="promotion_gate",
           delta=result.delta,
           verdict={"promote": True},
           cost=0.0,
           parent_hash=result.parent_hash,
           meta={"tail_level": level},
       )
   else:
       ledger.record(
           operator="promotion_gate",
           delta=result.delta,
           verdict={"promote": False},
           cost=0.0,
           parent_hash=result.parent_hash,
           meta={"tail_level": level},
       )

The exact fields depend on the ledger object and your application, but the
principle is stable: the gate should be explicit enough to audit later.

Promotion gates should be written before reading the final challenger result.
Changing the threshold after seeing a favorable run turns verification into
retrofitted justification. If a threshold needs to change, record that as a new
experiment with its own rationale.

Search a Typed Space
--------------------

For a larger model-design question, define a typed search space and a builder.

.. code-block:: python

   from mixle.evolve import Categorical, Integer, Real, Space, search

   space = Space({
       "components": Integer(1, 5),
       "alpha": Real(0.1, 4.0, log=True),
       "family": Categorical(["gaussian", "student_t"]),
   })

   def build_fn(config):
       return fit_candidate(data, config)

   found = search(
       space,
       data,
       objective=nll_objective(),
       build_fn=build_fn,
       method="bo",
       n_iter=25,
   )

   challenger = found.best_model

Search proposes candidates. Verification still decides promotion.

Typed spaces are preferable to loose dictionaries because the search algorithm
knows which dimensions are categorical, integer, continuous, or log-scaled.

Constrain the search space to values that can be served, monitored, and
explained. A candidate that only appears better because it exceeds latency, memory, or
operational limits is not a viable challenger.

Promote Deliberately
--------------------

Before replacing a model, ask:

* Did the challenger improve the primary objective?
* Did it preserve calibration?
* Did it avoid worse tail behavior or decision regret?
* Is the evaluation split representative of production traffic?
* Are the rejected candidates and reasons recorded?

Also check the negative evidence. A professional release note should say which
reasonable challengers failed, what gate they failed, and whether the failure
was numerical, statistical, operational, or data-related.

That discipline is what makes automatic improvement compatible with
professional model governance.

Read :doc:`/analysis` for diagnostics and :doc:`/evolution` for search spaces,
objectives, verification, and ledgers.
