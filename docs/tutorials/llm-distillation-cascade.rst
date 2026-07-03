LLM Distillation Cascade
========================

This tutorial turns a repeated LLM classification call into a local model with
an escalation path. The intended production behavior is simple:

* answer locally when calibrated confidence is high;
* ask the teacher when the local model is uncertain;
* measure accuracy, escalation, and cost as one system.

The same pattern applies to frontier-model labelers, human review queues, slow
rules engines, and expensive internal services.

1. Wrap The Teacher
-------------------

``llm_labeler`` turns an LLM-like object into a batched labeling callable.

.. code-block:: python

   from mixle.task import CallableLLM, llm_labeler

   teacher = llm_labeler(
       CallableLLM(generate),
       ["spam", "ham"],
       instruction="Classify the email as spam or ham.",
   )

``generate`` can be a hosted LLM call, a local model, or a deterministic test
fixture. The important contract is that ``teacher(texts)`` returns labels from
the declared label set.

2. Spend Labels Actively
------------------------

Active distillation labels an initial seed set, trains a local student, then
spends the remaining budget on examples that should improve the decision
boundary.

.. code-block:: python

   from mixle.task import active_distill

   active = active_distill(
       teacher,
       unlabeled_pool,
       budget=60,
       seed_size=20,
       rounds=4,
       acquisition="margin",
       labels=["spam", "ham"],
   )

   student = active.model
   print(active.labels_used)

Use margin acquisition when the goal is classification accuracy near the
student boundary. Use diversity-aware acquisition when the pool has obvious
clusters and the seed set may miss some of them.

3. Calibrate The Local Student
------------------------------

Training accuracy is not enough. Calibrate on held-out examples labeled by the
same teacher or by a trusted review process.

.. code-block:: python

   from mixle.task import CalibratedTaskModel

   cal_y = teacher(calibration_texts)
   calibrated = CalibratedTaskModel(student, alpha=0.1).calibrate(
       calibration_texts,
       cal_y,
   )

``alpha`` is the target error rate for answered cases. Calibration may reduce
coverage if the local model is not reliable enough.

4. Serve Through A Cascade
--------------------------

The cascade owns the answer-or-escalate decision.

.. code-block:: python

   from mixle.task import Cascade, CostModel

   cascade = Cascade(
       calibrated,
       teacher,
       cost=CostModel(c_frontier=0.01, c_local=0.00001),
   )

   outputs = cascade.serve(requests)
   print(cascade.report())

The report should be read as a system metric: local coverage, escalation rate,
teacher spend, and agreement all matter.

5. Score The Replacement
------------------------

Before promotion, score the cascade against a held-out teacher or human-labeled
set.

.. code-block:: python

   from mixle.task import scorecard

   card = scorecard(
       cascade,
       teacher,
       test_texts,
       student_cost=0.00001,
       teacher_cost=0.01,
   )

   print(card.table())

A good cascade is not necessarily the one with the highest local coverage. It
is the one that meets the quality target at the lowest acceptable cost and
latency.

Run The Examples
----------------

The repository includes runnable examples for the same workflow:

.. code-block:: sh

   python examples/task_llm_active_example.py
   python examples/task_cascade_economics_example.py

Promotion Checklist
-------------------

Before replacing a live teacher:

* freeze the label set and prompt/instruction used by the teacher;
* keep a held-out test set that was not used for active selection;
* calibrate on recent traffic;
* report answered accuracy separately from escalation rate;
* log teacher fallbacks so future retraining can target them.

Read :doc:`/task-distillation` for the full distillation workflow and
:doc:`/task-serving` for numeric, multi-label, structured-output, edge, and
tool-calling replacements.
