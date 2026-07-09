Bring Your Own Model
=====================

You trained a model outside mixle entirely -- a HuggingFace checkpoint, a
TRL-fine-tuned chat model, an arbitrary ``predict(x)`` function from another
framework -- and you do not want to rewrite it in mixle's terms. This page is
the on-ramp: wrap it as a **callable teacher**, then compose it with mixle's
distillation, calibration, and routing machinery unchanged.

The shape of the whole page is one pipeline:

.. code-block:: text

   externally-trained model -> callable teacher -> distill -> calibrate -> route

Each arrow is one mixle call. Nothing here requires the external model's
weights, framework, or training loop to know anything about mixle.

This is a different interop pattern than :doc:`neural-llm`'s "drop a real
torch module into ``GradLeaf``" story -- see :ref:`byom-vs-gradleaf` below for
when to reach for which.

1. Wrap It As A Callable Teacher
---------------------------------

``mixle.task.llm.CallableLLM`` wraps any ``fn(prompt) -> str`` (or
``fn(prompt, system) -> str``) as an ``LLM``. The function can be a thin
shim around ``model.generate()`` for an HF checkpoint, a TRL-trained policy,
a llama.cpp binding, or a test double -- mixle only ever calls
``complete(prompt)``.

.. code-block:: python

   from mixle.task import CallableLLM, llm_labeler

   # Stand-in for `tokenizer(prompt) -> model.generate(...) -> tokenizer.decode(...)`
   # against a checkpoint trained outside mixle (HF Trainer, TRL SFT/DPO, ...).
   def generate(prompt: str, system: str | None = None) -> str:
       # e.g. hf_pipeline(prompt)[0]["generated_text"], or a TRL-trained model's .generate()
       return "spam" if "free prize" in prompt.lower() else "ham"

   teacher_llm = CallableLLM(generate)

   # Constrain free-text replies to a fixed label set -- the batched-teacher
   # shape (`texts -> [label]`) every mixle.task distillation entry point expects.
   teacher = llm_labeler(
       teacher_llm,
       ["spam", "ham"],
       instruction="Classify the email as spam or ham.",
   )

If the checkpoint is served behind an OpenAI-compatible endpoint (vLLM, TGI,
Ollama, a hosted API) instead of loaded in-process, use ``OpenAICompatLLM``
in place of ``CallableLLM`` -- same downstream contract, no local inference
code at all.

For a vision-language checkpoint, the sibling is ``mixle.task.vlm``:
``CallableVLM`` wraps ``fn(image, prefix) -> [(token, log_prob), ...]``, and
``OpenAICompatVLM`` talks to a logprob-serving OpenAI-compatible endpoint.
Both plug into ``mixle.enumeration.best_first_decode`` the same way an LLM
teacher plugs into distillation.

At this point, ``teacher`` is indistinguishable to the rest of ``mixle.task``
from a frontier API model -- everything below is unaware the labels came
from an HF checkpoint.

2. Distill: Train A Small mixle-Native Student
------------------------------------------------

``distill`` exercises the teacher once over an unlabeled corpus and fits a
small local classifier that reproduces its labels:

.. code-block:: python

   from mixle.task import distill

   student = distill(
       teacher,
       unlabeled_emails,
       labels=["spam", "ham"],
       task="spam classifier distilled from an HF/TRL teacher",
   )

   student.meta["train_agreement"]   # how faithfully the student mimics the teacher
   student.save("spam_student.mixle")

If labeling budget is the constraint rather than compute, ``active_distill``
spends a fixed label ``budget`` on the most informative pool examples
(by margin/entropy acquisition) instead of labeling everything:

.. code-block:: python

   from mixle.task import active_distill

   result = active_distill(teacher, unlabeled_pool, budget=200, rounds=5)
   student = result.model

Either way the teacher is opaque: ``distill``/``active_distill`` only ever
call it as ``teacher(list_of_texts) -> list_of_labels``. Nothing about the
student's training loop touches the checkpoint, its framework, or its
weights.

3. Calibrate: An Honest Answer-Or-Escalate Guarantee
-------------------------------------------------------

A distilled student's softmax is not a probability guarantee on its own.
``CalibratedTaskModel`` wraps the student in a conformal threshold so
"answer locally" vs. "escalate" is backed by coverage, not vibes:

.. code-block:: python

   from mixle.task import CalibratedTaskModel

   calibrated = CalibratedTaskModel(student, alpha=0.1)
   calibrated.calibrate(held_out_texts, held_out_labels)

   calibrated.decide(new_email)          # a label, or ESCALATE (None) if unsure
   calibrated.escalation_rate(val_texts) # the empirical p(escalate)

``distill_for_routing`` (and ``distill_records_for_routing`` for structured
records) does steps 2 and 3 in one call -- it holds out a calibration slice,
fits the student, and returns an already-``decide()``-able
``CalibratedTaskModel``:

.. code-block:: python

   from mixle.task import distill_for_routing

   calibrated = distill_for_routing(teacher, unlabeled_emails, labels=["spam", "ham"])

For *generation* rather than classification -- e.g. serving the external
checkpoint's own free-text output under a coverage guarantee, or picking
among several candidate continuations -- ``CalibratedGenerator`` is the
generation-side sibling: draw ``k`` candidates from any generator (a
``CallableLLM`` sampled ``k`` times, a beam, the teacher itself) and
calibrate a conformal accept-or-abstain threshold over them.

.. note::

   ``CalibratedGenerator`` (``mixle.task.calibrated_generator``) is not yet
   re-exported from ``mixle.task``'s public surface -- import it directly
   from its module until it is.

4. Route: Cheap mixle-Native Model First, Expensive External Model Only When Needed
---------------------------------------------------------------------------------------

``Cascade`` serves traffic with the calibrated local model, escalating to the
external teacher only when the local model is unsure:

.. code-block:: python

   from mixle.task import Cascade, CostModel

   cascade = Cascade(calibrated, teacher, cost=CostModel(c_frontier=0.03, c_local=0.0001))
   label = cascade("free prize inside!!!")
   cascade.report()        # realized $/request saved vs. teacher-only
   cascade.harvested()     # (texts, labels) the teacher answered -- feed back into distill()

``Router`` generalizes this to several calibrated tiers (cheapest first,
... -> the external checkpoint as the final, always-answering fallback).
Build it from ``(name, model, cost)`` tuples -- any tier before the last
just needs a ``decide(x)`` method, exactly what ``calibrated`` above
already has:

.. code-block:: python

   from mixle.task import Router

   router = Router([
       ("local-checkpoint", calibrated, 0.0001),  # the calibrated student from step 3, above
       ("external-checkpoint", teacher, 0.03),    # the wrapped external checkpoint, last and most expensive
   ])

   router("free prize inside!!!")
   router.report()      # per-tier traffic and realized cost
   router.harvested()   # requests only the external model answered -- re-distill to shrink escalation

A deeper stack (tiny -> small -> ... -> external) is the same shape with
more tuples; :func:`~mixle.task.router.Router.from_solutions` is a
convenience constructor for the case where each tier is a full
:class:`~mixle.task.solve.Solution` (from :func:`~mixle.task.solve.solve`)
rather than a bare calibrated model -- it reads each ``Solution``'s
``.cascade.model`` for you.

In both cases the external, HF/TRL-trained model never needs to change: it
is the fallback tier, called exactly the same way it was called in step 1.

.. _byom-vs-gradleaf:

This vs. The ``GradLeaf`` Pattern
------------------------------------

There are two, deliberately different, ways an externally-trained model
enters mixle:

.. list-table::
   :header-rows: 1

   * - Pattern
     - What happens to the model
     - Use when
   * - **Bring your own model** (this page)
     - stays external; called through ``complete()``/``next_logprobs()`` as
       a black-box teacher
     - you want to distill a cheap local student from it, calibrate an
       abstention guarantee, and route around it -- without retraining it
   * - ``GradLeaf`` bridge (see :doc:`neural-llm` and
       ``examples/peft_lora_grad_leaf.py``)
     - loaded *into* mixle as a real ``torch.nn.Module`` and fine-tuned
       in-process (e.g. a ``peft``-wrapped HF checkpoint's LoRA adapters
       trained via ``mixle.inference.estimation.optimize``)
     - you want to keep training the checkpoint itself, inside a mixle
       model/estimation loop, not just query it

``examples/peft_lora_grad_leaf.py`` is the concrete receipt for the second pattern: a real
``hf-internal-testing/tiny-random-gpt2`` checkpoint, LoRA-wrapped with
``peft.get_peft_model``, dropped into ``GradLeaf`` unchanged and fine-tuned
via mixle's ordinary M-step. That is genuinely different work from this
page's pattern -- no fine-tuning happens here, the checkpoint is called,
not trained.

If you are unsure which you want: if you plan to keep training the external
checkpoint's weights, use the ``GradLeaf`` bridge. If you only want to call
it and cheapen/guarantee/route around those calls, use the callable-teacher
pattern above.

Where Each Receipt Lives
----------------------------

Every claim on this page is pinned by an executable test. A skeptical
reader should run these directly rather than trust the prose above:

.. list-table::
   :header-rows: 1

   * - Step
     - Claim
     - Test file
   * - 1. Callable teacher
     - ``CallableLLM``/``OpenAICompatLLM`` wrap an arbitrary function/endpoint
       as an ``LLM``; ``llm_labeler`` constrains it to a label set
     - ``mixle/tests/task_llm_test.py``
   * - 1. VLM sibling
     - ``CallableVLM``/``OpenAICompatVLM`` wrap an image-conditioned scorer
     - ``mixle/tests/task_vlm_test.py``
   * - 2. Distill
     - ``distill``/``distill_from_labels`` reproduce teacher labels in a
       small local student; agreement is recorded
     - ``mixle/tests/task_distill_test.py``,
       ``mixle/tests/distill_methods_test.py``
   * - 2. Active distill
     - ``active_distill`` spends a fixed label budget via acquisition
     - ``mixle/tests/task_active_test.py``,
       ``mixle/tests/doe_active_test.py``
   * - 2+3. One-call routing distill
     - ``distill_for_routing``/``distill_records_for_routing`` return an
       already-calibrated model
     - ``mixle/tests/task_distill_routing_test.py``
   * - 3. Calibrate
     - ``CalibratedTaskModel`` gives ``1 - alpha`` conformal coverage and an
       honest ``ESCALATE`` on ambiguous/OOD input
     - ``mixle/tests/task_calibrate_test.py``
   * - 3. Calibrated generation
     - ``CalibratedGenerator`` gives conformal accept-or-abstain coverage
       over sampled candidates
     - ``mixle/tests/task_calibrated_generator_test.py``
   * - 4. Cascade
     - one calibrated tier plus teacher fallback; realized cost and harvest
     - ``mixle/tests/task_cascade_test.py``
   * - 4. Router
     - N calibrated tiers cheapest-first, teacher as final fallback
     - ``mixle/tests/task_router_test.py``
   * - GradLeaf contrast
     - a real peft-wrapped HF checkpoint fine-tunes unchanged inside
       ``GradLeaf`` (PR #129, open)
     - ``mixle/tests/grad_control_test.py``
       (``AdapterThroughTheBridgeTest``), ``examples/peft_lora_grad_leaf.py``

See Also
------------

* :doc:`task-distillation` -- the full teacher/student/calibration/escalation
  model, including numeric, multi-label, and structured teachers.
* :doc:`task-serving` -- ``Cascade``/``Router`` in more depth, plus edge
  deployment and quantization once the student is trained.
* :doc:`neural-llm` -- the ``GradLeaf`` bridge for training a real torch
  module (including a peft-wrapped HF checkpoint) inside mixle.
