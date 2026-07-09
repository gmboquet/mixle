Task Serving, Routing, and Edge Deployment
==========================================

The :doc:`task-distillation` guide covers the basic teacher/student workflow:
label with a teacher, train a local student, calibrate answer sets, and serve a
cascade. ``mixle.task`` also contains the production-facing pieces around that
loop: one-call replacement for label, numeric, multi-label, and dict-valued
tasks,
multi-tier routing, edge-device search, post-training quantization, harnesses
for common legacy-code shapes, and scorecards.

One-Call Task Solving
---------------------

``solve`` points Mixle at a function that already performs a task and returns a
deployable ``Solution``.

.. code-block:: python

   from mixle.task import solve

   def route_ticket(ticket):
       if ticket["amount"] > 10_000:
           return "finance-review"
       return "auto-approve"

   solution = solve(route_ticket, historical_tickets, seed=0)
   label = solution({"amount": 425.0, "country": "US"})
   report = solution.report()

The teacher function labels examples. The student is trained and calibrated.
At runtime the solution answers locally only when confident; otherwise it calls
the teacher. ``Solution.improve`` folds harvested escalations into the next
round and promotes only if verification does not regress.

Numeric Task Replacement
------------------------

``solve_regression`` handles routines that return numbers instead of labels:
pricing functions, score calculators, risk estimates, sizing rules, or
scientific surrogate functions.

.. code-block:: python

   from mixle.task import solve_regression

   def price(item):
       base = {"basic": 20.0, "pro": 80.0, "max": 150.0}[item["tier"]]
       return base + 0.5 * item["size"]

   solution = solve_regression(price, historical_items, tol=5.0, alpha=0.1)

   value = solution({"tier": "pro", "size": 42.0})
   yhat, lo, hi = solution.interval({"tier": "pro", "size": 42.0})

The teacher labels the examples exactly as in ``solve``. The student is a
small local regressor. Calibration is split conformal: on held-out calibration
examples, Mixle computes an absolute-residual quantile ``qhat`` so
``[yhat - qhat, yhat + qhat]`` covers the teacher's answer with probability at
least ``1 - alpha`` under the usual exchangeability assumption.

Runtime behavior is intentionally conservative. The local regressor answers
only when ``qhat <= tol``. If the calibrated interval is wider than the
precision your application can tolerate, every request escalates to the teacher
until enough harvested examples support a tighter promoted model.

Use this when the original code returns a scalar and the operational contract
can be expressed as "local answers are acceptable within this tolerance." Keep
``tol`` tied to the business, scientific, or safety requirement rather than to
the model's average error.

When ``qhat`` is infinite because the calibration split is too small or too
difficult, the artifact remains loadable but should behave as non-local until a
better calibration set is available. Do not replace ``inf`` with a finite
stand-in value in reports or manifests.

Multi-Label Task Replacement
----------------------------

``solve_multilabel`` handles routines that return a set of labels: compliance
flags, alert annotations, routing tags, document categories, moderation facets,
or any task where several labels may be true at once.

.. code-block:: python

   from mixle.task import solve_multilabel

   def flags(transaction):
       out = []
       if transaction["amount"] > 400:
           out.append("high-value")
       if transaction["region"] == "eu":
           out.append("eu-rules")
       return out

   solution = solve_multilabel(flags, historical_transactions, alpha=0.1)

   tags = solution({"amount": 525.0, "region": "eu"})
   local_tags = solution.try_local({"amount": 60.0, "region": "us"})

The student uses one shared featurizer with a sigmoid head per label. On the
calibration split, Mixle learns two bars for each label: an absent-label score
bar above which the label is confidently present, and a present-label score
bar below which the label is confidently absent.

Runtime behavior is conservative in the same way as ``solve`` and
``solve_regression``. A label can be present, absent, or ambiguous. The whole
request escalates when any label is ambiguous, and labels with too little
calibration evidence are treated as ambiguous rather than guessed. A locally
returned set is therefore made only of labels the calibrated student decided.

Structured Output Replacement
-----------------------------

``solve_structured`` handles routines that return a dictionary with a stable
schema: enrichers, triagers, quote builders, extracted metadata, or any task
where several named outputs must be produced together.

.. code-block:: python

   from mixle.task import solve_structured

   def enrich(ticket):
       return {
           "route": "finance" if ticket["amount"] > 10_000 else "ops",
           "priority": "high" if ticket["age_hours"] > 24 else "normal",
           "reserve": ticket["amount"] * 0.15,
       }

   solution = solve_structured(
       enrich,
       historical_tickets,
       tol={"reserve": 25.0},
       alpha=0.1,
   )

   output = solution({"amount": 12_500.0, "age_hours": 31})
   maybe_local = solution.try_local({"amount": 500.0, "age_hours": 2})
   report = solution.report()

Mixle calls the teacher once per training example, infers the output fields,
and trains one calibrated sub-solution per field. Categorical fields use the
same conformal singleton rule as ``solve``. Numeric fields use the same
split-conformal interval rule as ``solve_regression`` and require a scalar
``tol`` or a per-field tolerance dictionary.

The composition rule is strict. A structured output is returned locally only
when every field can answer locally. One uncertain categorical label, one
numeric field whose calibrated interval is too wide, or one under-calibrated
field escalates the entire request to the teacher. That keeps the returned
dictionary coherent instead of mixing trusted local fields with guessed ones.

Input Integrity
---------------

Task wrappers do not make missing-data policy decisions for the application.
The teacher still receives the raw request when a local model escalates, and
the local feature adapters use their documented encodings rather than
rewriting records into cleaned business objects. If ``None``, ``NaN``, an empty
string, or a missing field is meaningful to the original task, include those
cases in the training examples, calibration split, and scorecard segments.

For numeric replacement, keep ``qhat=inf`` as evidence that the calibration
split does not support local answers at the requested tolerance. For
structured replacement, give every numeric field its own tolerance and check
that missing or non-finite upstream values route the way the original teacher
contract expects. Do not report a student as locally safe on a segment unless
that segment was represented in calibration or explicitly routed to the
teacher.

Multi-Tier Routing
------------------

``Router`` generalizes ``Cascade`` from one local tier plus teacher to several
calibrated tiers.

.. code-block:: python

   from mixle.task import Router

   router = Router.from_solutions(
       [fast_solution, accurate_solution],
       teacher=frontier_teacher,
       costs=[0.0001, 0.001, 0.03],
       names=["fast", "accurate", "frontier"],
   )

   y = router(request)
   print(router.report())

Each tier must expose ``decide(x)`` and may return ``ESCALATE``. The final tier
is the teacher or frontier model and always answers. Requests answered by the
teacher are harvested as targeted labels for the next solve round.

Use ``route_stack`` when you already have several solutions and per-request
costs and want the tiers sorted by ascending cost.

Tool Calling
------------

``distill_tool_caller`` turns a teacher's function-calling behavior into a
local selector plus per-tool argument extractors.

.. code-block:: python

   from mixle.task import ToolSpec, distill_tool_caller

   tools = [
       ToolSpec("lookup_customer", ["customer_id"]),
       ToolSpec("refund", ["order_id", "amount"]),
   ]

   caller = distill_tool_caller(frontier_tool_teacher, requests, tools, seed=0)
   call = caller("refund order A-102 for 19.95")

The teacher returns ``{"tool": name_or_none, "args": {...}}``. Mixle trains:

* a calibrated selector over the request text;
* one argument extractor per tool;
* an escalation path back to the teacher.

The local caller emits a tool call only when the selector is confident and all
required arguments are present. If the selector abstains, the tool is unknown,
or required arguments are missing, the caller escalates to the teacher and
harvests the trace for later improvement.

This is single-step tool calling. Use planning when a request needs multiple
verified steps. See :doc:`agentic-task-distillation` for the full tool-calling
and planning workflow.

Tool traces should include both the teacher output and the validation result.
That keeps malformed tool names, missing required arguments, and schema
repairs visible instead of folding them into ordinary labels.

Planning
--------

``distill_planner`` trains a sequence of calibrated next-step predictors from
teacher traces. A plan is an autoregressive chain of tool calls ending in
``STOP``.

.. code-block:: python

   from mixle.task import ToolSpec, distill_planner

   tools = [
       ToolSpec("search", ["query"]),
       ToolSpec("summarize", ["document_id"]),
   ]

   planner = distill_planner(frontier_planner, requests, tools, seed=0)
   result = planner("find the latest policy and summarize it")

Each training trace is flattened into contexts of the form "request plus plan
so far" and a next action. At runtime a step is accepted only when:

* the selector confidently chooses the next tool or ``STOP``;
* the required arguments extract successfully;
* optional execution succeeds when an ``execute`` map is provided.

Any uncertainty, malformed step, missing argument, execution failure, or
maximum-step exhaustion escalates the whole request to the teacher. The planner
does not return half-trusted plans as if they were local successes.

Generative Planning and Traces
------------------------------

``sft_planner`` trains a small causal LM to write an entire serialized plan,
then gates the generated text with a strict parser, tool-spec validation,
copy-fidelity checks, and optional grammar-constrained decoding. Use it when a
single generated plan is a better abstraction than a sequence of next-step
classifiers.

``harvest_agent_traces`` can turn stored agent conversations into deterministic
teachers for ``distill_tool_caller``, ``distill_planner``, or ``sft_planner``.
It infers ``ToolSpec`` objects from observed tool usage and exposes
``call_teacher`` and ``plan_teacher`` lookup functions.

See :doc:`agentic-task-distillation` for examples and operational guidance.

Edge Deployment
---------------

Edge deployment is not only hyperparameter tuning. The model family itself may
need to change to fit flash, latency, or runtime constraints.

Key objects:

* ``DeviceSpec`` declares hard constraints such as ``max_bytes``, ``max_ops``,
  and ``torch_free``.
* ``EdgeFootprint`` records measured bytes, operation count, and torch
  dependency.
* ``EdgeSpace`` describes candidate student families and training recipes.
* ``DesignModel`` stores a surrogate over design choices and outcomes.
* ``distill_for_edge`` searches for the best feasible student.
* ``distill_designer`` compresses accumulated design knowledge into a compact
  local model.

.. code-block:: python

   from mixle.task import DeviceSpec, distill_for_edge

   device = DeviceSpec(max_bytes=128_000, max_ops=40_000, torch_free=True)
   result = distill_for_edge(
       teacher,
       examples,
       device=device,
       seed=0,
   )

   student = result.model
   print(result.footprint)

Use ``measure_inference_seconds`` and ``measure_ops_per_second`` on the target
hardware when latency is a hard requirement.

Post-Training Quantization
--------------------------

``quantize_mlp`` converts a trained Torch MLP student into a NumPy-only
``TaskModel`` with int8 or int4 weights.

.. code-block:: python

   from mixle.task import quantize_mlp

   q8 = quantize_mlp(student, bits=8)
   q4 = quantize_mlp(student, bits=4)

Quantized students use ``QuantizedMLP`` and ``QuantizedClassifierIO``. They
store arrays rather than Torch modules and can qualify for ``torch_free``
devices.

The artifact path keeps int4 weights packed in the arrays payload, lets extreme
outlier weights be clipped before quantization with ``clip_percentile``, and
returns correctly shaped probability arrays for empty batches instead of failing
during reshape.

``lns_classifier`` and ``LNSStructuredClassifierIO`` provide integer log-space
execution for structured students where the model is a sum of factor
log-densities.

Harnesses for Existing Code
---------------------------

Harnesses package common replacement patterns:

* ``replace_extractor`` replaces a regex or parser that maps text to fields;
* ``replace_alerter`` replaces a threshold rule over sliding windows;
* ``replace_matcher`` replaces a deduplication or matching rule over pairs.

These are wrappers around the same safety contract: local answer only when
calibrated; otherwise fall back to the original code.

.. code-block:: python

   from mixle.task import replace_matcher

   matcher = replace_matcher(old_match_rule, example_pairs, seed=0)
   result = matcher(record_a, record_b)
   print(matcher.report())

Scorecards
----------

``scorecard`` measures a deployed student or router against the teacher it
replaces.

.. code-block:: python

   from mixle.task import scorecard

   card = scorecard(
       solution,
       teacher,
       heldout_inputs,
       student_cost=0.0001,
       teacher_cost=0.03,
       task="ticket routing",
   )

   print(card.table())

The scorecard reports end-to-end accuracy, local agreement, escalation rate,
latency, artifact size, and blended cost when costs are provided.

Use segment scorecards before changing a route. At minimum, break out rare
labels, high-cost requests, high-risk customer or study groups, and inputs that
previously escalated. A lower average escalation rate is not a release win if
the reduction comes from answering cases that should still be deferred.

Route Evidence
--------------

For a serving change, record the exact route that handled each held-out
request:

* local tier name or teacher fallback;
* conformal set size, regression interval width, or structured field that
  caused escalation;
* density-gate or OOD result when present;
* teacher label used for end-to-end scoring;
* artifact version and calibration ``alpha``.

This evidence makes average metrics interpretable. Without route evidence, an
improved cost number can hide a regression where the wrong tier absorbed a
rare or high-risk segment.

Artifacts
---------

Task artifacts are durable. Relevant helpers include:

* ``TaskModel``;
* ``TaskManifest`` and ``SCHEMA_VERSION``;
* ``save_json`` and ``load_json``;
* ``save_arrays`` and ``load_arrays``;
* ``save_module`` and ``load_module``;
* ``register_builder`` and ``register_arrays_builder``;
* ``read_manifest``.

Use artifact helpers when adding a new student payload type. Ordinary users
usually call ``TaskModel.save`` and ``TaskModel.load``.

Calibrated artifacts preserve non-finite conformal thresholds such as
``qhat=inf`` with a JSON-safe sentinel and restore them as ``float("inf")``.
That keeps small or difficult calibration splits loadable without pretending
the model is locally answerable.

Economics and Route Planning
----------------------------

Economic helpers include:

* ``CostModel``;
* ``RoutePlan``;
* ``cascade_cost_per_request``;
* ``break_even_volume``;
* ``recommend_route``.

These functions make cost assumptions explicit. A cascade or router should
report realized cost from actual traffic, while route planning estimates which
deployment shape is worth trying.

API Reference
-------------

* :doc:`api/mixle.task`
* :doc:`api/mixle.task.solve`
* :doc:`api/mixle.task.regress`
* :doc:`api/mixle.task.multilabel`
* :doc:`api/mixle.task.structured_out`
* :doc:`api/mixle.task.generative_text`
* :doc:`api/mixle.task.router`
* :doc:`api/mixle.task.edge`
* :doc:`api/mixle.task.quantize`
* :doc:`api/mixle.task.toolcall`
* :doc:`api/mixle.task.plan`
* :doc:`api/mixle.task.sft_plan`
* :doc:`api/mixle.task.constrained`
* :doc:`api/mixle.task.traces`
* :doc:`api/mixle.task.scorecard`
* :doc:`api/mixle.task.harness`

Operational Standard
--------------------

For a task-serving deployment, keep:

* the teacher definition or endpoint version;
* the examples used for labeling;
* the student recipe and artifact manifest;
* calibration data and ``alpha``;
* escalation policy and OOD gate settings;
* scorecard results on held-out data;
* harvested escalations and labels;
* edge footprint when deploying outside a server environment.

That record makes it possible to improve a task model without quietly changing
what the original task meant.
