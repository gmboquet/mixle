Reasoning Systems
=================

The :doc:`uncertainty` page introduces LLM uncertainty and linear-Gaussian
evidence fusion. ``mixle.reason`` also includes a broader reasoning system:
finite-hypothesis reasoning, cross-modal retrieval as evidence selection,
knowledge-graph-producing LLMs, typed ontologies, acquisition planning,
amortized encoders, and a trainable cross-modal latent model.

Use this page for probabilistic reasoning and evidence representations. Use
:doc:`reasoning-ecosystem` for the application shell around those ideas:
substrate storage, reasoner actions, skills, pool jobs, telemetry, and the
optional ``Scientist`` workflow.

Discrete Reasoning
------------------

``reason_discrete`` fuses evidence over a finite hypothesis set.

.. code-block:: python

   import numpy as np
   from mixle.reason import reason_discrete

   answer = reason_discrete(
       ["normal", "fault-a", "fault-b"],
       [
           ("sensor", np.array([-0.2, -1.8, -2.1])),
           ("text", np.array([-1.4, -0.3, -2.0])),
       ],
   )

   print(answer.top(2))
   print(answer.summary())

``model_evidence`` turns fitted Mixle models into evidence by scoring the same
observation under one model per hypothesis.

``DiscreteAnswer.decide`` computes the Bayes-optimal action under a loss matrix
and can include an explicit abstain cost.

Record the hypothesis set, priors, evidence sources, and loss matrix when a
discrete answer drives a decision. Changing any one of those inputs changes the
meaning of the posterior and the recommended action.

Cross-Modal Store
-----------------

``CrossModalStore`` treats retrieval as evidence selection. A low-cost
embedding key retrieves candidates; each candidate can then contribute coarse
embedding evidence or fine raw-payload evidence.

.. code-block:: python

   from mixle.reason import CrossModalStore

   store = CrossModalStore(
       keys,
       payloads,
       coarse=payload_to_embedding_evidence,
       fine=payload_to_raw_evidence,
       metric="cosine",
   )

   belief, steps = store.assimilate(prior_belief, query_key, k=8, epsilon=0.05)

Each ``RetrievalStep`` records the item index, fidelity, and information gain.
Use ``next_evidence`` for active retrieval: the next item whose evidence most
reduces query entropy.

Retrieval evidence should remain auditable. Store candidate identifiers,
fidelity choices, and skipped high-cost evidence when the reasoning result is
used outside exploration.

Acquisition Planning
--------------------

``select_evidence_batch`` chooses a budgeted batch of evidence items and
fidelities.

.. code-block:: python

   from mixle.reason import select_evidence_batch

   plan = select_evidence_batch(
       store,
       belief,
       budget=3.0,
       fine_cost=1.0,
       coarse_cost=0.2,
   )

   print(plan.indices)
   print(plan.total_gain)

The planner greedily re-scores candidates after each selected item, so the
batch avoids paying twice for redundant evidence.

Budget settings are part of the result. A lower budget can change not only the
amount of evidence collected but also which modalities or fidelities are
trusted by the final belief.

Graph-Producing LLMs
--------------------

``GraphLLM`` asks a generator to emit structured facts rather than prose.
Parsed generations become canonical graphs, and uncertainty is computed over
graphs rather than strings.

.. code-block:: python

   from mixle.reason import GraphLLM

   graph_llm = GraphLLM(generate, parse_triples, n=20)
   dist = graph_llm.distribution("Extract facts about the contract.")

   print(dist.edge_marginals())
   print(dist.query("contract", "renewal_date"))

``GraphDistribution`` supports:

* graph-level entropy;
* marginalization over graph-derived outcomes;
* edge marginals ``P(triple in graph)``;
* fact probabilities;
* calibrated edge marginals through ``fit_fact_calibrator``.

This is useful when generated text needs fact-level reliability rather than a
single answer confidence.

Graph extraction should keep parse failures and invalid graphs visible. A
graph distribution built only from successfully parsed outputs can overstate
reliability if many generations failed the schema.

Ontologies and Typed Graphs
---------------------------

``Ontology`` provides symbolic constraints over graph facts: classes,
subclass relations, relation signatures, relation axioms, and disjointness.
It can audit triples before they become substrate knowledge or before a graph
completion is accepted.

.. code-block:: python

   from mixle.reason.ontology import Ontology

   ontology = (
       Ontology()
       .add_class("Person")
       .add_class("Organization")
       .add_relation("works_at", "Person", "Organization", "functional")
   )

   problems = ontology.check_triple(
       "ada",
       "works_at",
       "acme",
       {"ada": "Person", "acme": "Organization"},
   )

``OntologyConstrainedKG`` wraps a fitted knowledge-graph distribution and masks
tail completions to range-conforming entities. This makes the schema part of
the probability query rather than an after-the-fact filter.

Ontology checks are validation evidence, not proof that the source facts are
true. Keep type violations, masked completions, and disjointness conflicts with
the graph artifact.

Amortized Encoders
------------------

``AmortizedEncoder`` learns a heteroscedastic Gaussian expert:

.. code-block:: python

   from mixle.reason import AmortizedEncoder

   encoder = AmortizedEncoder(in_dim=32, latent_dim=4).fit(X, Z)
   evidence = encoder.evidence(x, name="spectrum")

The encoder maps raw modality features into a Gaussian belief about a latent.
Predicted variance is input-dependent, so the evidence can down-weight itself
on ambiguous inputs.

Cross-Modal Model
-----------------

``CrossModalModel`` is a trainable product-of-experts latent model. It learns a
shared latent from unlabeled multimodal records and can infer that latent from
any subset of modalities.

.. code-block:: python

   from mixle.reason import CrossModalModel

   model = (
       CrossModalModel(latent_dim=8)
       .add_modality("text", 128)
       .add_modality("sensor", 64)
       .fit({"text": text_features, "sensor": sensor_features})
   )

   belief = model.belief({"text": text_features[0]})
   predicted_sensor = model.predict({"text": text_features[0]}, "sensor")

Use ``calibrate`` and ``predict_interval`` when cross-modal prediction needs
finite-sample coverage for a target modality.

Cross-modal predictions should be evaluated per modality and per missing-view
pattern. A model that works with all modalities present can fail when only text
or only sensor evidence is available.

Relationship to LLM UQ
----------------------

LLM uncertainty in :doc:`uncertainty` asks whether a language model's sampled
answers agree. Reasoning systems ask a broader question: how does evidence from
several sources change a belief or decision?

Use:

* ``LLMUncertainty`` for answer-or-abstain over sampled text answers;
* ``GraphLLM`` when generated information should be represented as facts;
* ``Ontology`` when graph facts need typed constraints before they are stored
  or completed;
* ``reason`` for continuous linear-Gaussian latent fusion;
* ``reason_discrete`` for finite hypotheses;
* ``CrossModalStore`` for retrieval that decides when raw evidence is worth
  fetching;
* ``CrossModalModel`` when the shared latent itself should be learned from
  multimodal data.

Validation Evidence
-------------------

For reasoning workflows, preserve:

* hypothesis sets, priors, evidence source identifiers, and loss matrices;
* retrieval candidates, selected fidelities, and information-gain records;
* graph parser failures, ontology violations, and fact-calibration reports;
* cross-modal calibration by modality and missing-view pattern;
* abstention or escalation thresholds; and
* the action policy that consumes the belief.

API Reference
-------------

Generated reference pages:

* :doc:`api/mixle.reason`;
* :doc:`api/mixle.reason.core`;
* :doc:`api/mixle.reason.discrete`;
* :doc:`api/mixle.reason.graph_llm`;
* :doc:`api/mixle.reason.ontology`;
* :doc:`api/mixle.reason.store`;
* :doc:`api/mixle.reason.encoder`;
* :doc:`api/mixle.reason.model`.

Reasoning API Inventory
-----------------------

.. list-table::
   :header-rows: 1

   * - Import
     - Role
   * - ``LinearGaussianEvidence``
     - Evidence object for linear-Gaussian latent assimilation.
   * - ``NonlinearEvidence``
     - Evidence adapter for nonlinear observation models.
   * - ``AcquisitionPlan``
     - Selected evidence batch and utility metadata.
   * - ``LLMAssessment``, ``ClaimAssessment``, ``InformationAssessment``
     - Structured reports from LLM uncertainty and claim checks.
   * - ``FactualityModel``, ``content_overlap``
     - Claim/factuality helper and overlap scoring.
   * - ``canonical_graph``
     - Normalize graph outputs before graph-level uncertainty or calibration.
   * - ``Ontology``, ``OntologyConstrainedKG``
     - Typed graph constraints and ontology-masked KG completion.
   * - ``ScaledEmbedding``
     - Embedding wrapper used by shared latent and cross-modal models.
