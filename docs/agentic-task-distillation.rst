Agentic Task Distillation
=========================

``mixle.task`` has two task-distillation tracks:

* ordinary task replacement: ``text -> label``, ``record -> label``, or
  ``text -> fields``;
* agentic replacement: ``request -> tool call`` or
  ``request -> verified plan``.

The agentic track keeps the same verification rule as the classifier track: local
models only emit actions when the action is structurally valid and calibrated
enough to trust. Uncertain, malformed, missing-argument, or failed plans
escalate to the teacher and become training traces for the next round.

Tool Specs
----------

Tool-calling and planning start with ``ToolSpec``:

.. code-block:: python

   from mixle.task import ToolSpec

   tools = [
       ToolSpec("lookup_order", ["order_id"]),
       ToolSpec("refund", ["order_id", "amount"], required=["order_id", "amount"]),
       ToolSpec("notify", ["customer_id", "message"]),
   ]

``args`` is the set of argument fields the tool accepts. ``required`` defaults
to all arguments. A local planner may not emit a call unless every required
argument is present.

Tool specs are part of the safety boundary. Version them with the artifact and
record any argument constraints, extractive-field requirements, and tools that
are deliberately unavailable to the local planner.

Single-Step Tool Calling
------------------------

``distill_tool_caller`` trains a selector for the tool name plus one extractor
per tool for arguments:

.. code-block:: python

   from mixle.task import distill_tool_caller

   caller = distill_tool_caller(teacher_call, requests, tools, seed=0)
   result = caller("refund order A-102 for 19.95")

The teacher returns:

.. code-block:: python

   {"tool": "refund", "args": {"order_id": "A-102", "amount": "19.95"}}

The returned ``ToolCaller`` emits a local call only when tool selection and
required argument extraction both succeed. Otherwise it calls the teacher,
returns the teacher call with ``"escalate": True``, and stores the trace in
``harvested``.

Measure local calls and escalations separately. A high local-call rate is not a
success if the emitted calls are malformed, unauthorized, or wrong under the
tool execution contract.

Stepwise Planning
-----------------

``distill_planner`` decomposes a plan into a sequence of next-step decisions.
The teacher returns a list of tool calls:

.. code-block:: python

   [
       {"tool": "lookup_order", "args": {"order_id": "A-102"}},
       {"tool": "refund", "args": {"order_id": "A-102", "amount": "19.95"}},
       {"tool": "notify", "args": {"customer_id": "C-7", "message": "refund issued"}},
   ]

Training flattens each trace into "request plus plan so far" contexts. At
runtime the planner predicts the next tool, extracts its arguments, optionally
executes it, and repeats until ``STOP``.

.. code-block:: python

   from mixle.task import distill_planner

   planner = distill_planner(teacher_plan, requests, tools, seed=0)
   out = planner("refund order A-102 and notify the customer")

Use this planner when plan templates are regular enough that next-step
classification is the right abstraction. It is usually lower-cost and easier to
verify than a generative planner.

Stepwise planners should be evaluated at both step and plan level. A plan can
have accurate early steps and still fail because a later argument is missing or
because execution state changed after a lookup.

Trace-SFT Planning
------------------

``sft_planner`` is the generative rung. It trains one small causal LM to write
the whole serialized plan:

.. code-block:: text

   request
   => lookup_order(order_id=A-102) | refund(order_id=A-102; amount=19.95)

The returned ``GenerativePlanner`` parses the generated text under a strict
plan grammar, validates the tool names and required arguments, checks that
generated argument values are extractive from the request where required, and
escalates when validation fails.

.. code-block:: python

   from mixle.task import sft_planner

   gen = sft_planner(
       teacher_plan,
       requests,
       tools,
       seed=0,
       d_model=96,
       n_layer=3,
       n_head=4,
       constrained=True,
   )

   out = gen("refund order A-102 and notify the customer")
   print(gen.report())

When ``constrained=True``, decoding is grammar-constrained through
``mixle.task.constrained``. The grammar removes many malformed outputs before
they can be returned, while the validation step remains as a backstop. A
calibrated confidence floor can also force low-confidence generations to
escalate.

Use ``sft_planner`` when a single generated plan is the natural object:
variable-length plans, shared argument syntax, or many tools where a separate
next-step classifier would be cumbersome. Do not use it as a license to trust
free-form text. The parser gate is the contract.

Keep parser failures, grammar rejections, and low-confidence generations in
the report. They are evidence that the escalation path is doing useful work,
not noise to hide from the scorecard.

Harvesting Existing Agent Traces
--------------------------------

``harvest_agent_traces`` turns stored agent conversations into deterministic
teachers for tool calling and planning:

.. code-block:: python

   from mixle.task import harvest_agent_traces, distill_tool_caller, sft_planner

   traces = harvest_agent_traces()
   tools = traces.tool_specs()

   caller = distill_tool_caller(
       traces.call_teacher(),
       traces.requests(),
       tools,
       seed=0,
   )

   planner = sft_planner(
       traces.plan_teacher(),
       traces.requests(min_steps=1),
       tools,
       seed=0,
   )

The default location is ``~/.mixle-agent/conversations``. Each stored
conversation is split into ``AgentTrace`` objects:

Stored traces are application data. Review them for secrets, private tool
arguments, and stale tool schemas before using them as teachers.

``parse_conversation`` is the lower-level helper for a single already-loaded
conversation document. Use ``harvest_agent_traces`` for normal directory-level
loading.

``request``
    The user request text.

``plan``
    Ordered tool-use blocks emitted by the assistant before the next user turn.

``reply``
    Final text reply, when present.

``conversation_id``
    Source conversation identifier.

``AgentTraces.tool_specs`` infers argument sets from observed usage. Required
arguments are keys present in every observed call for that tool. The teacher
views are lookup tables over the harvested requests, so distilling from history
does not call a frontier model.

Inferred required arguments should be reviewed before training. Historical
traces can omit rare but mandatory fields if the trace set is narrow.

Serving and Artifacts
---------------------

``GenerativePlanner.save(path)`` writes the LM module, codec, tool specs,
verification gates, and planner metadata. ``GenerativePlanner.load`` restores a
serving planner and requires the teacher fallback:

.. code-block:: python

   path = gen.save("artifacts/refund-planner")
   restored = type(gen).load(path, teacher=teacher_plan)

The teacher remains part of the artifact boundary because escalation is not an
error. It is how the system remains explicit when the local plan is not safe.

Artifact reviews should load the planner, run a held-out trace set, and verify
that escalation still reaches the intended teacher. A saved planner without a
working fallback is not a complete serving artifact.

Choosing the Planner
--------------------

.. list-table::
   :header-rows: 1

   * - Need
     - Use
   * - One tool call
     - ``distill_tool_caller``
   * - Regular plan templates with few steps
     - ``distill_planner``
   * - Variable-length generated plans with strict grammar validation
     - ``sft_planner``
   * - Training from stored agent sessions
     - ``harvest_agent_traces`` plus either planner

For all three, measure held-out agreement and live escalation rate. A low
escalation rate is useful only if the non-escalated plans are correct and
execution-verified.

Release Evidence
----------------

For agentic distillation, preserve:

* tool specs, argument constraints, and unavailable-tool policy;
* held-out tool-call and full-plan agreement;
* parser, grammar, validation, and execution-failure counts;
* escalation rate and teacher fallback identity;
* harvested-trace provenance and secret-review status; and
* artifact reload and fallback smoke tests.

API Reference
-------------

* :doc:`api/mixle.task.toolcall`
* :doc:`api/mixle.task.plan`
* :doc:`api/mixle.task.sft_plan`
* :doc:`api/mixle.task.constrained`
* :doc:`api/mixle.task.traces`
