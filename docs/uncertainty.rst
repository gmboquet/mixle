Uncertainty
===========

``mixle`` has several uncertainty surfaces, all with the same design bias:
uncertainty should change behavior. It should decide whether to answer,
escalate, collect more data, or report which evidence source mattered.

This page covers:

* uncertainty over LLM answers;
* claim-level reliability for generated text;
* epistemic versus aleatoric decomposition;
* the ``uq`` dispatcher for fitted models, point predictors, ensembles, and
  LLM-like callables;
* conformal answer-or-abstain behavior;
* cross-modal latent evidence fusion;
* calibrated task cascades.

Unified ``uq`` Dispatcher
-------------------------

``mixle.inference.uq`` provides one front door when the caller owns a
heterogeneous predictor and wants Mixle to choose an uncertainty route from the
object's capabilities.

.. code-block:: python

   from mixle.inference import uq

   fitted_uq = uq(model, training_rows)
   interval_uq = uq(point_predictor, (x_cal, y_cal), alpha=0.1)
   llm_uq = uq(generate, example_prompts, alpha=0.1)

``UQResult`` exposes method-specific accessors:

* ``sample_models`` and ``credible_interval`` for fitted Mixle models through a
  Laplace parameter posterior;
* ``interval`` and ``epistemic_std`` for point predictors or ensembles through
  split conformal calibration;
* ``semantic_entropy`` and ``confident`` for LLM-style generation callables.

Use the specialized APIs below when you already know the exact uncertainty
method. Use ``uq`` when the application boundary should accept several kinds
of predictor behind one call.

LLMUncertainty
--------------

``LLMUncertainty`` wraps any stochastic callable:

.. code-block:: text

   generate(prompt) -> answer

It samples multiple answers, clusters them by meaning, and returns the majority
meaning plus a confidence and semantic entropy.

.. code-block:: python

   from mixle.reason import LLMUncertainty

   def equivalent(a, b):
       return str(a).strip().lower() == str(b).strip().lower()

   llm_uq = LLMUncertainty(generate, equivalent=equivalent, n=20)
   assessment = llm_uq.assess("Which city is the Eiffel Tower in?")

   print(assessment.answer)
   print(assessment.confidence)
   print(assessment.semantic_entropy)
   print(assessment.clusters)

High confidence means most samples fell into the same meaning cluster. High
semantic entropy means the model is disagreeing with itself about the answer,
not just rephrasing it.

These quantities are decision signals, not truth guarantees. Use retrieval,
tools, labels, or human review when factual correctness matters.

Equivalence Matters
-------------------

The default equivalence relation is exact equality. That is fine for labels or
normalized short answers. For prose, pass a domain-specific relation:

.. code-block:: python

   import re

   def normalize_city(text):
       return re.sub(r"[^a-z]", "", text.lower())

   llm_uq = LLMUncertainty(
       generate,
       equivalent=lambda a, b: normalize_city(a) == normalize_city(b),
       n=20,
   )

In production this relation might use canonicalization, embeddings, entailment,
or a task-specific parser.

Version the equivalence relation with the artifact. Changing it changes the
clusters, confidence, entropy, and abstention threshold.

Epistemic and Aleatoric Split
-----------------------------

Use several prompts as ensemble members, for example paraphrases of the same
question. ``decompose`` separates disagreement across members from spread
within each member.

.. code-block:: python

   dec = llm_uq.decompose(
       [
           "Who discovered penicillin?",
           "Name the scientist credited with discovering penicillin.",
           "Penicillin was discovered by whom?",
       ],
       n=10,
   )

   print(dec.epistemic)
   print(dec.aleatoric)
   print(dec.total)

Epistemic uncertainty is reducible uncertainty: model or prompt sensitivity.
Aleatoric uncertainty is within-member ambiguity: the question or output space
itself remains variable.

Calibrated Answer-or-Abstain
----------------------------

Sampling confidence is still only a signal until calibrated. ``calibrate`` uses
labeled examples to choose the lowest confidence threshold whose answered set
has empirical error at most ``alpha``.

.. code-block:: python

   examples = [
       ("Capital of France?", "Paris"),
       ("2 + 2?", "4"),
   ]

   llm_uq.calibrate(examples, alpha=0.1)

   answer = llm_uq.answer("Capital of Japan?")
   if answer is None:
       escalate_to_human_or_frontier_model()
   else:
       print(answer.answer)

After calibration, ``answer`` returns ``None`` below the threshold. This is the
important behavioral change: the LLM can abstain instead of hallucinating.

Calibration examples should match the served prompt distribution. A threshold
learned on short factual questions should not be reused for extraction,
planning, or legal/scientific summaries without new evidence.

Claim-Level Reliability
-----------------------

A response can have a stable main answer and still contain one fabricated
detail. ``assess_claims`` takes one sampled response, extracts claims, and
checks whether independent samples corroborate each claim.

.. code-block:: python

   info = llm_uq.assess_claims(
       "Summarize the contract renewal and include dates and parties.",
       threshold=0.6,
   )

   print(info.reliability)
   for claim in info.fabricated:
       print(claim.claim, claim.support)

Defaults:

* claim extraction is sentence-like splitting;
* corroboration uses information-weighted content overlap across samples.

For serious text, pass your own extractor or entailment-based corroborator:

.. code-block:: python

   info = llm_uq.assess_claims(
       prompt,
       extract=my_claim_extractor,
       corroborates=my_entailment_check,
   )

Uncertainty Helpers
-------------------

The underlying decomposition functions are available from
``mixle.inference.uncertainty``:

.. code-block:: python

   from mixle.inference.uncertainty import (
       Clustering,
       UncertaintyDecomposition,
       cluster_samples,
       decompose_entropy,
       decompose_uncertainty,
       decompose_variance,
       marginalize_meaning,
       posterior_ensemble,
       predictive_distribution,
       semantic_entropy,
   )

Use them when you already have samples, probability vectors, or prediction
ensembles and do not need the LLM wrapper.

``UncertaintyDecomposition`` is the shared result object for epistemic,
aleatoric, and total uncertainty summaries. ``predictive_distribution`` and
``posterior_ensemble`` build ensemble-style predictive objects from fitted or
posterior models, ``decompose_variance`` performs the variance analogue of the
entropy split, and ``marginalize_meaning`` aggregates probabilities over
semantic clusters represented by ``Clustering``.

Cross-Modal Reasoning
---------------------

``mixle.reason.reason`` fuses evidence into a shared latent belief. Each
evidence source is a linear-Gaussian observation:

.. code-block:: text

   y = H z + noise,  noise ~ N(0, R)

Example:

.. code-block:: python

   import numpy as np
   from mixle.reason import Evidence, Latent, reason

   prior = Latent.vector(2, mean=0.0, var=10.0)
   evidence = [
       Evidence(np.array([[1.0, 0.0]]), np.array([2.0]), 0.2, name="sensor-a"),
       Evidence(np.array([[0.0, 1.0]]), np.array([-1.0]), 0.5, name="sensor-b"),
   ]

   ans = reason(prior, evidence)

   print(ans.mean)
   print(ans.interval(level=0.9))
   print(ans.information_gain())
   print(ans.attribution(normalize=True))

``ReasonedAnswer`` exposes:

* posterior mean and covariance;
* credible intervals;
* total information gain;
* per-modality attribution;
* prediction-level epistemic/aleatoric variance split.

Mechanistic Latents
-------------------

``Latent.mechanistic`` builds a Gaussian prior over a trajectory constrained by
a linear dynamical law. Evidence at one time step updates all time steps through
the dynamics.

.. code-block:: python

   A = np.array([[1.0, 0.1], [0.0, 1.0]])
   prior = Latent.mechanistic(A, steps=20, process_cov=0.01 * np.eye(2))

Use ``block_selector`` from ``mixle.reason.core`` to observe a specific time
block of the stacked trajectory.

For mechanistic latents, record the dynamical law, process covariance,
observation selector, and evidence times. Those assumptions define how one
observation updates unobserved trajectory steps.

Task Calibration
----------------

For local task models, uncertainty becomes an answer/escalate decision through
``CalibratedTaskModel`` and ``Cascade``:

.. code-block:: python

   from mixle.task import CalibratedTaskModel, Cascade

   model = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_x, cal_y)
   cascade = Cascade(model, teacher)
   y = cascade("new request")

See :doc:`task-distillation` for the full serving workflow.

Track answered accuracy and escalation rate together. A cascade that answers
too often can be unsafe; a cascade that escalates everything may be correct but
not useful.

Related Reasoning Workflows
---------------------------

This page focuses on uncertainty behavior. See :doc:`reasoning-systems` for the
broader ``mixle.reason`` stack: finite-hypothesis reasoning, cross-modal
retrieval with raw-data fallback, graph-producing LLMs, evidence acquisition,
amortized modality encoders, and trainable cross-modal latent models.

Choosing the Right Tool
-----------------------

.. list-table::
   :header-rows: 1

   * - Need
     - Use
   * - Does the LLM know the answer?
     - ``LLMUncertainty.assess`` and semantic entropy
   * - I have a fitted model or predictor and want Mixle to pick a UQ route
     - ``mixle.inference.uq``
   * - Should the LLM answer or abstain?
     - ``LLMUncertainty.calibrate`` and ``answer``
   * - Which claim in this answer is suspect?
     - ``LLMUncertainty.assess_claims``
   * - Is uncertainty due to prompt/model sensitivity?
     - ``LLMUncertainty.decompose``
   * - How do multiple modalities update a latent?
     - ``reason``, ``Evidence``, ``Latent``
   * - Should a local task model escalate?
     - ``CalibratedTaskModel`` and ``Cascade``

Release Evidence
----------------

For uncertainty workflows, preserve:

* sample count, prompt template, model identifier, and sampling settings;
* equivalence, claim extraction, and corroboration functions;
* calibration examples, alpha, learned threshold, and abstention policy;
* false-answer and unnecessary-abstention rates;
* evidence model assumptions for cross-modal or mechanistic latents; and
* cascade scorecards separating answered quality from escalation behavior.
