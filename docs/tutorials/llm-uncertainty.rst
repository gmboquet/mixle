LLM Uncertainty
===============

This tutorial wraps an arbitrary LLM-like callable with
:class:`mixle.reason.LLMUncertainty`. The goal is not to make generation
reliable by assertion. The goal is to turn repeated samples into behavior:
answer, abstain, inspect disagreement, or escalate.

The wrapper needs only one callable:

.. code-block:: text

   generate(prompt) -> answer

For a real system, ``generate`` might call a hosted model, a local model, or an
internal agent. For a test, it can be a deterministic fixture.

1. Define Equivalence
---------------------

Semantic uncertainty depends on what counts as the same answer. Exact string
matching is fine for normalized labels, but most LLM outputs need a domain
relation.

.. code-block:: python

   import re
   from mixle.reason import LLMUncertainty

   def normalize(text):
       return re.sub(r"[^a-z0-9]+", "", str(text).lower())

   def equivalent(a, b):
       return normalize(a) == normalize(b)

   uq = LLMUncertainty(generate, equivalent=equivalent, n=20)

The equivalence function is part of the model. Record it with the application
because changing it changes the uncertainty numbers.

Test equivalence on examples where surface form differs but meaning is the
same, and on examples where wording is similar but the answer is materially
different. A weak equivalence relation can make disagreement disappear; an
overly strict one can turn harmless paraphrases into false uncertainty.

2. Assess One Prompt
--------------------

.. code-block:: python

   assessment = uq.assess("Who discovered penicillin?")

   print(assessment.answer)
   print(assessment.confidence)
   print(assessment.semantic_entropy)
   print(assessment.clusters)

``confidence`` is the mass of the majority meaning cluster. ``semantic_entropy``
is high when samples disagree about meaning, not merely wording.

These values are decision signals, not proof of truth. A model can agree with
itself and still be wrong, especially when the prompt lacks evidence or asks
for a common misconception. Use retrieval, tools, human review, or a trusted
label set when factual correctness matters.

3. Decompose Prompt Sensitivity
-------------------------------

Use paraphrases as ensemble members when you want to separate within-prompt
ambiguity from prompt sensitivity.

.. code-block:: python

   dec = uq.decompose(
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

Epistemic uncertainty points to reducible sensitivity: prompt wording, model
choice, retrieval context, or missing evidence. Aleatoric uncertainty points to
ambiguity that remains inside each prompt.

Use this decomposition to decide what to fix. High epistemic uncertainty often
calls for clearer prompts, better context, or a different model. High
aleatoric uncertainty often calls for a refusal, a clarifying question, or a
workflow that returns multiple admissible answers.

4. Calibrate Abstention
-----------------------

Raw sample agreement is a signal, not a guarantee. Calibrate it on labeled
examples before using it to decide whether to answer.

.. code-block:: python

   examples = [
       ("Capital of France?", "Paris"),
       ("2 + 2?", "4"),
       ("Who discovered penicillin?", "Alexander Fleming"),
   ]

   uq.calibrate(examples, correct=equivalent, alpha=0.1)
   answer = uq.answer("Capital of Japan?")

   if answer is None:
       escalate_to_human_or_frontier_model()
   else:
       print(answer.answer)

After calibration, ``answer`` returns ``None`` below the learned confidence
threshold. This is the operational payoff: the model can decline rather than
fabricate.

Choose the calibration set to match the prompts that will be served. A
threshold learned from short factual questions should not be reused for legal
summaries, scientific extraction, or multi-step planning without new evidence.
Track both false answers and unnecessary abstentions so the system is not tuned
to optimize only one side of the tradeoff.

5. Inspect Claim Reliability
----------------------------

A response can have a stable main answer and still contain unsupported
details. Claim assessment extracts claims from one response and checks whether
independent samples corroborate them.

.. code-block:: python

   info = uq.assess_claims(
       "Summarize the contract renewal and include dates and parties.",
       threshold=0.6,
   )

   print(info.reliability)
   for claim in info.fabricated:
       print(claim.claim, claim.support)

For serious use, pass a task-specific claim extractor and an entailment-style
``corroborates`` function. The defaults are intentionally lightweight.

Claim reliability is most useful when the downstream workflow needs a partial
answer. A response can be routed for review because one claim is weak while
still preserving the supported claims for later use.

Cost and Reproducibility
------------------------

Repeated sampling increases latency and spend. Pick ``n`` from the operational
budget and validate the stability of the decision, not just the smoothness of
the entropy estimate. When a hosted model is used, record the model identifier,
sampling settings, prompt template, equivalence function, and calibration
threshold with the serving artifact.

Validation Checklist
--------------------

Before serving an LLM uncertainty wrapper:

* define and version the equivalence relation;
* calibrate on examples from the same prompt distribution;
* monitor abstention rate and false-answer rate together;
* inspect high-entropy prompts instead of averaging them away;
* route abstentions to a human, retrieval step, frontier model, or safer
  fallback.

Read :doc:`/uncertainty` for the full API and :doc:`/reasoning-systems` for
claim reliability, graph-producing LLMs, and cross-modal evidence.
