Local Reasoning Ecosystem
=========================

Mixle includes a local reasoning layer around model and task objects. The
purpose is to make evidence acquisition explicit: retrieve what is already
known, run local skills when they are available, simulate or create artifacts
when that is the lowest-cost useful action, and abstain when the system does not
have enough evidence to answer.

This layer is separate from the core probability library. Use
``mixle.stats`` and ``mixle.inference.optimize`` for ordinary model fitting.
Use the ecosystem surfaces when you are building an application that needs
knowledge, provenance, callable capabilities, routing decisions, and local audit
records around fitted models.

Main Surfaces
-------------

.. list-table::
   :header-rows: 1

   * - Surface
     - Role
   * - ``mixle.substrate``
     - typed, scoped, provenanced local store for documents, records,
       artifacts, traces, context packets, and graph facts.
   * - ``mixle.inference.skill``
     - wraps a fitted model or callable as a named reusable capability with
       inherited certificate metadata.
   * - ``mixle.substrate.investigate``
     - orders retrieve, compute, simulate, create, and delegate actions under
       a cost budget; answers only when enough evidence is collected.
   * - ``mixle.substrate.Reasoner``
     - deployable shell around an answerer, a substrate, registered skills,
       and optional custom actions.
   * - ``mixle.pool``
     - job abstraction for work that may run locally or on a configured pool,
       with budget and explicit-confirmation rails.
   * - ``mixle.telemetry``
     - local JSONL event log for fit, placement, route, context, reasoning,
       escalation, pool, and drift decisions.
   * - ``mixle.scientist``
     - optional, heavyweight assembled workflow for local scientific
       reasoning with cached open-weight encoders and local answer generation.

Substrate
---------

``Substrate`` stores typed items. Each item has a ``kind``, a retrievable text
surface, optional structured payload, provenance, scope, tags, links, and an
identifier.

.. code-block:: python

   from mixle.substrate import Substrate, retrieve

   store = Substrate()
   store.add(
       "text",
       text="Refund requests over 5000 USD require finance approval.",
       provenance={"source": "policy"},
       tags=["refund", "finance"],
   )

   hits = retrieve(store, "refund approval", k=3)
   print([item.text for item in hits.items])

Small stores use deterministic lexical matching. Larger text-bearing stores can
build a learned embedding index through ``Substrate.reindex``. The public
contract is the store, item typing, scope filtering, and provenance trail; the
ranker can improve without changing callers.

Treat the substrate as application data. Record ingestion source, scope,
redaction status, and reindex configuration when stored items are used as
evidence for a release-facing answer.

Answering and Investigation
---------------------------

``answer_from_substrate`` is the simple path: retrieve evidence, assemble a
context packet, call an answerer, or abstain if retrieval is too weak.

``investigate`` is the broader path. It accepts named actions:

* ``retrieve_action`` over a substrate;
* ``compute_action`` over a skill or callable;
* ``simulate_action`` over a simulator;
* ``create_action`` over an artifact builder;
* ``delegate_action`` for explicit external escalation.

.. code-block:: python

   from mixle.substrate import Reasoner, Substrate
   from mixle.inference import SkillRegistry, skill

   store = Substrate()
   store.add("text", text="Premium support tickets route to the escalation queue.")

   def answerer(question, evidence):
       return evidence.splitlines()[0]

   registry = SkillRegistry()
   skill("route-ticket", lambda text: "escalation", description="route support tickets", registry=registry)

   reasoner = Reasoner(answerer, substrate=store, skills=registry)
   result = reasoner.ask("Where do premium support tickets route?", verify=True)

   print(result.answer)
   print(result.trace())

The returned ``Investigation`` records the fired actions, evidence fragments,
confidence, spending, and optional factuality receipt. Verification does not
replace the answer; it attaches a receipt so callers can gate on it.

Investigation traces should be kept when answers are reviewed later. A final
answer without the retrieved evidence, skipped actions, and spending record is
not enough to audit the reasoning path.

Trust, Scope, and Governance
----------------------------

The substrate includes operational controls around the knowledge store:

* ``check_factuality`` splits an answer into claims and retrieves supporting
  evidence from the substrate.
* ``verify_lineage`` and ``audit_substrate`` check whether provenance links
  still resolve.
* ``detect_secrets`` and ``redact_secrets`` scan items before they are shared
  or ingested into a broader context.
* ``Space`` and ``publish`` provide team-scoped visibility with an explicit
  sharing action.
* ``Governance`` adds propose/review/approve/reject gates for curated scopes.
* ``Ontology`` and ``OntologyConstrainedKG`` add typed constraints to graph
  facts and knowledge-graph completion.

These tools do not turn a local store into an enterprise governance platform.
They make the application-level contract inspectable: what was stored, who can
see it, what it derives from, and which claims can be cited.

Before sharing or publishing substrate-derived artifacts, run the same data and
secret review used for model examples. Provenance can contain sensitive source
names even when item text has been redacted.

Pool and Placement
------------------

``plan_placement`` in :mod:`mixle.inference` decides which certified estimation
blocks are local and which are pool-eligible. ``mixle.pool`` is the execution
boundary for offloaded work:

.. code-block:: python

   from mixle.pool import PoolJob, submit

   job = PoolJob(
       run=lambda: {"artifact": "done"},
       kind="verb",
       reason="large gradient block",
       est_cost=0.0,
       budget=1.0,
   )
   result = submit(job)
   print(result.ok, result.artifact)

The default backend is local, so the abstraction works without external
infrastructure. Billable backends are expected to require explicit
confirmation and reject jobs above budget.

Pool decisions should be logged with cost estimates, confirmation state, and
resolved backend. A job that falls back to local execution should say so in the
receipt rather than looking like remote capacity was used.

Telemetry and Learned Orchestration
-----------------------------------

``Telemetry`` records decisions as rows of ``(features, choice, outcome)``.
Those rows feed learned placement, action-acquisition, and scheduling policies.

.. code-block:: python

   from mixle.telemetry import Telemetry

   telemetry = Telemetry("mixle-events.jsonl")
   telemetry.record(
       "route",
       features={"kind": "compute", "cost": 1.0},
       choice="local",
       outcome={"value": 1.0},
   )

   rows = telemetry.training_rows("route")

Telemetry events intentionally carry decision features and outcomes, not raw
user content. Treat the JSONL log as application data: rotate it, scope it, and
review it before using it to train routing policy.

Learned orchestration should train only from reviewed telemetry. A policy
trained on stale, redacted, or biased routing logs can make confident placement
decisions for the wrong reason.

Scientist
---------

``mixle.scientist`` is an optional assembled workflow, installed with the
``scientist`` extra. It combines cached open-weight encoders, certified heads
over learned latents, substrate-backed answering, and edge-distillation
receipts. It is useful as a reference application for local scientific
reasoning; it is not required for the core library.

.. code-block:: sh

   pip install "mixle[scientist]"

The module sets offline Hugging Face environment defaults and expects weights
to already be available in the local cache. Use it deliberately when those
assets and dependencies are part of the application.

Scientist workflows should be documented as optional assembled applications.
Record model-weight identity, local-cache status, dependency extras, and
whether answer generation was actually exercised before presenting results as
release evidence.

Validation Evidence
-------------------

For local reasoning ecosystem workflows, preserve:

* substrate ingestion sources, scopes, and redaction status;
* retrieval or investigation traces with cited evidence;
* skill certificates or fallback behavior;
* pool placement decisions, budgets, and backend resolution;
* telemetry retention and review status; and
* optional scientist dependency and model-weight evidence.

API Reference
-------------

* :doc:`api/mixle.substrate`
* :doc:`api/mixle.pool`
* :doc:`api/mixle.telemetry`
* :doc:`api/mixle.scientist`
* :doc:`api/mixle.inference.skill`
* :doc:`api/mixle.inference.orchestration`
