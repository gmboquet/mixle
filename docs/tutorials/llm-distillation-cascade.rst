LLM Distillation Cascade
========================

This tutorial turns a repeated LLM classification call into a local model with
an escalation path. The production behavior is simple:

* answer locally when calibrated confidence is high;
* ask the teacher when the local model is uncertain;
* measure accuracy, escalation, and cost as one system.

The same pattern applies to frontier-model labelers, human review queues, slow
rules engines, and expensive internal services.

1. Wrap the Teacher
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

Validate the teacher contract before spending an active-learning budget. A
small smoke set should include ordinary examples, boundary cases, and examples
that should be refused or escalated by policy. If the teacher prompt,
allowlisted labels, or output parser changes, treat the resulting labels as a
new training source rather than silently appending them to an older dataset.

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

Active selection makes the training set useful, but it also makes it biased
toward uncertain regions. Keep a held-out evaluation set that was not selected
by the acquisition function. That split is the evidence for promotion; the
actively selected examples are the evidence for learning.

3. Calibrate the Local Student
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

Calibration should be rerun when the label set, teacher prompt, feature
pipeline, or traffic distribution changes. Store the learned threshold and the
calibration report with the model artifact so serving behavior can be
reproduced outside the notebook that trained it.

4. Serve Through a Cascade
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

Do not optimize coverage by hiding escalations. The cascade is valuable because
it makes uncertainty visible and routes difficult cases to the expensive path.
Log the escalated examples, teacher responses, and local confidence so future
distillation rounds can focus on the actual production misses.

5. Score the Replacement
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

Read the scorecard by segment as well as in aggregate. A cascade can look
healthy overall while failing rare labels, newly launched workflows, or
high-value customer traffic. When those segments matter, make them explicit in
the test set and promotion report.

Run the Examples
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
* inspect rare labels and high-cost segments separately from the aggregate;
* log teacher fallbacks so future retraining can target them.

Read :doc:`/task-distillation` for the full distillation workflow and
:doc:`/task-serving` for numeric, multi-label, structured-output, edge, and
tool-calling replacements.
