Troubleshooting
===============

This page collects the problems that usually mean the model shape, dependency
set, or capability expectation is mismatched.

Torch Import or Neural Leaf Failure
-----------------------------------

Install the Torch extra:

.. code-block:: sh

   pip install "mixle[torch]"

Then verify:

.. code-block:: sh

   python - <<'PY'
   from mixle.models import TransformerLMEstimator
   print(TransformerLMEstimator)
   PY

Use ``device="cpu"`` first. Move to ``device="cuda"`` after the shape works.

Estimator Does Not Match Data
-----------------------------

Symptom: an encoder, unpacking, or shape error during ``optimize``.

Check one observation and the estimator side by side:

.. code-block:: python

   row = data[0]
   print(row)
   print(estimator)

For tuples, use ``CompositeEstimator``. For dictionaries or named records, use
record-shaped estimators. For variable-length lists, use ``SequenceEstimator``.
For next-token neural leaves, the field should look like ``(context, target)``.

Mixture Results Change Across Runs
----------------------------------

Mixtures and HMMs have local optima. Use multiple starts:

.. code-block:: python

   import numpy as np
   from mixle.inference import best_of

   score, model = best_of(
       train,
       valid,
       estimator,
       trials=8,
       max_its=100,
       init_p=0.1,
       delta=1e-8,
       rng=np.random.RandomState(0),
       out=None,
   )

Set ``rng=`` when you need reproducibility.

For release evidence, compare the selected mixture against a simpler baseline
on held-out data and keep the rejected restart scores. A single improved run
does not prove that the latent structure is stable.

Unexpected ``NaN`` or ``-inf`` Scores
-------------------------------------

``NaN`` inputs, impossible observations, and missing fields should not be
quietly rewritten by troubleshooting code. First identify which case you have:

* ``NaN`` as source data means the data pipeline or model-specific missing-data
  policy must decide what it represents;
* ``-inf`` usually means the observation is impossible under the fitted
  support;
* ``NaN`` produced during fitting or scoring usually means a numerical or
  validation bug that should be reproduced with a minimal row; and
* masked, optional, or marginalized fields should be represented with the
  explicit missing-data surface documented for the model.

Do not replace non-finite values with zeros to keep a workflow running.
That changes the statistical meaning of the observation and can hide the issue
that the model needs to report.

A Capability Is Missing
-----------------------

Not every model can enumerate, condition, marginalize, or expose latent
posteriors. Ask the model:

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))

If a workflow needs enumeration, choose a family with enumerable support. If it
needs latent paths, choose a latent-structured family that exposes posterior or
decoding methods.

Automatic Recommendation Looks Weak
-----------------------------------

``recommend_model`` reports low-confidence fields when the best family does not
beat the runner-up by much.

.. code-block:: python

   rec = recommend_model(data)
   print(rec.low_confidence_fields())

Treat those as data collection or modeling decisions. Add more data, constrain
the family explicitly, or compare the recommended model with a hand-built
alternative.

LLM-Designed Model Falls Back
-----------------------------

``design_model`` falls back when the LLM returns invalid JSON, uses a
non-allowlisted family, builds an incompatible estimator, or fails
fit-validation.

.. code-block:: python

   designed = design_model(data, llm)
   print(designed.source)
   print(designed.note)

Fallback is intentional. The LLM proposes; mixle validates.

Conformal Cascade Escalates Too Often
-------------------------------------

High escalation can mean:

* the local student is too weak;
* calibration data are small or harder than training data;
* ``alpha`` is too strict for the business tradeoff;
* the density gate is marking traffic as OOD;
* live traffic differs from training traffic.

Inspect conformal sets, OOD flags, and harvested examples. Retrain with the
harvested escalation labels before loosening calibration.

For ``solve_regression``, a high escalation rate usually means the calibrated
interval width ``qhat`` is larger than the requested ``tol``. Check whether
``tol`` is actually the application tolerance, whether the examples span the
live input range, and whether the target function is noisy or discontinuous.
Do not increase ``tol`` solely to force local answers.

For ``solve_multilabel``, one ambiguous label escalates the whole request.
Inspect labels with too few positive or negative calibration examples first:
under-calibrated labels are deliberately treated as ambiguous. Add examples for
rare labels or split the task if one hard label is preventing otherwise stable
tags from being served locally.

For ``solve_structured``, every output field must be locally decided before the
dictionary is returned. Check the per-field report: categorical fields fail for
the same reasons as ``solve``; numeric fields fail for the same ``qhat`` versus
``tol`` reason as ``solve_regression``. A missing tolerance for a numeric field
is a setup error rather than a calibration result.

Spark or Distributed Backend Fails
----------------------------------

Start with ``backend="local"`` and the same estimator. Then try
``backend="mp"``. Move to Spark/Dask/MPI only after the local shape works.

For Spark, make sure the driver and workers use the same Python environment and
that Java is installed.

Documentation Build Fails
-------------------------

Use the same strict verification command:

.. code-block:: sh

   make -C docs html SPHINXOPTS="-W --keep-going"

Warnings are treated as errors. Common causes are stale ``:doc:`` links,
missing optional dependencies during autodoc, or generated API files that need
to be refreshed with ``make -C docs apidoc``. If the working tree build passes
but a clean checkout fails, check for untracked ``docs/api/*.rst`` pages that
were generated locally but never included in the documentation PR.
