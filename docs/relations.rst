Relations
=========

``mixle.relations`` describes ranked feasible sets over structured spaces. A
relation is not merely an optimization problem. The single best solution is
often useful, but the full ranked set is the real object: assignments in
increasing cost, strings outward from an edit-distance center, k-best Viterbi
paths, shortest paths, spanning trees, feature subsets, stable matchings, graph
cuts, and related combinatorial structures.

The common surface is deliberately small:

.. code-block:: python

   relation.solve()       # best Solution or None
   relation.top(k)        # k best Solutions
   relation.enumerator()  # lazy iterator, best first

Every result is a ``Solution(value, objective)``. ``value`` is the structured
object itself; ``objective`` is the cost or score used for ranking.

Callers should keep the objective definition with the solution. For many
workflows, the same structured value can be ranked by probability, cost,
penalty, reward, or a calibrated score-to-cost transform. The result is only
auditable when that transform is named.

Why Relations Belong in Mixle
-----------------------------

Heterogeneous probabilistic systems frequently need structured decisions:

* assign observations to latent explanations;
* enumerate plausible state paths;
* search edits near a generated answer;
* rank feature subsets under a regression objective;
* enforce matching, flow, or graph constraints after a model produces scores.

Treating those as relations keeps the ranking and feasibility logic explicit.
The probabilistic model can score evidence, while the relation layer can
enumerate the structured objects that satisfy the constraint.

This separation is useful during review. A poor decision can come from the
model scores, from score-to-cost conversion, from the feasible-set definition,
or from the policy that consumes the top alternatives. Relations keep those
failure modes visible instead of hiding them in one opaque decision function.

Assignment
----------

``Assignment`` ranks bipartite assignments by total cost.

.. code-block:: python

   from mixle.relations import Assignment

   relation = Assignment([[1.0, 9.0], [9.0, 1.0]])
   best = relation.solve()

   assert best.value.tolist() == [0, 1]
   assert best.objective == 2.0

   alternatives = relation.top(3)

Use this when a model produces a cost matrix and the downstream decision must
be one-to-one: entity resolution, worker-task assignment, latent component
matching, or alignment across modalities.

Validate matrix orientation and units before solving. A transposed cost matrix
or a probability treated as a cost can produce a perfectly feasible assignment
that is semantically wrong.

Paths and Viterbi
-----------------

``ShortestPath`` and ``ViterbiPath`` expose best-first path enumeration.
``best_first_paths`` is the lower-level engine for custom state graphs.

Use this when you need more than the best path:

* compare the top few hidden-state explanations;
* estimate ambiguity in a decoded sequence;
* present alternative routes or derivations;
* search a trellis lazily until enough probability mass has been covered.

The distinction from a normal sampler is important. A relation enumerator is
ranked by objective. It is the right tool when you need the best alternatives,
not random alternatives.

When path ambiguity matters, store more than the best path. The margin between
the top paths often explains whether a decoded state sequence is stable enough
for interpretation.

Edit-Distance Neighborhoods
---------------------------

``EditDistance`` and ``nearest_first`` enumerate states outward from a center.
This supports workflows like:

* spelling or normalization alternatives;
* local robustness checks around a generated string;
* approximate matching against a structured vocabulary;
* bounded repair of malformed identifiers.

Because the iterator is lazy, callers can request only the first few neighbors
or stop at a distance threshold.

Record the distance metric and stopping rule when local neighborhoods are used
as evidence. Edit neighborhoods are sensitive to the alphabet, tokenization,
and allowed operations.

Spanning Trees and Graph Relations
----------------------------------

``SpanningTree`` ranks spanning trees by edge objective. Additional graph
helpers include:

* ``max_flow`` and ``min_cut``;
* ``min_arborescence``;
* ``graph_coloring``;
* ``max_clique`` and ``max_independent_set``;
* ``tsp_held_karp``.

These are useful when a probabilistic model estimates local scores but the
valid output has a global graph constraint.

Graph constraints should be checked on small fixtures before they are connected
to model scores. Feasibility bugs are easiest to see on hand-verifiable graphs.

Regression and MILP Helpers
---------------------------

``BestSubsetRegression`` ranks feature subsets. The module also includes
bounded and cardinality-constrained optimization helpers:

* ``branch_and_bound_milp``;
* ``cardinality_constrained_milp``;
* ``admm_bounded_least_squares``;
* ``irreducible_infeasible_subset``.

Use these when you need a constrained explanation, sparse decision rule, or
debugging pass for infeasible linear constraints.

Stable Matching
---------------

``stable_matching`` implements proposer-optimal Gale-Shapley matching.
``is_stable_matching`` verifies that a candidate matching has no blocking pair.

This is a decision layer, not a probabilistic estimator. It pairs naturally
with Mixle when preference scores are model outputs but the final assignment
must satisfy a stable matching constraint.

Model-estimated preferences should be calibrated or at least stress-tested
before they are treated as a matching policy. Stability is a property of the
declared preferences, not proof that the preferences themselves are correct.

Relations and Probability
-------------------------

Relations do not replace distributions. They complement them:

* A distribution scores or samples observations.
* A relation enumerates feasible structured decisions.
* An inference loop can use both: model scores become relation costs, and
  relation alternatives become candidate explanations.

For example, an HMM can estimate transition and emission probabilities, while
``ViterbiPath`` enumerates the best latent paths. A record model can score
entity-pair likelihoods, while ``Assignment`` enforces a one-to-one matching.

Operational Guidance
--------------------

Use relations when all three are true:

1. The output is structured.
2. Feasibility matters globally.
3. The top alternatives are more useful than independent samples.

Use distributions or samplers when uncertainty over the full support matters
more than ranked feasibility. Use :doc:`enumeration` when the object is a
probability distribution with enumerable support. Use ``mixle.relations`` when
the object is a constrained structured space with an objective.

Release Evidence
----------------

For relation-backed workflows, preserve:

* the relation type and versioned objective definition;
* input score or cost arrays before any transformation;
* feasibility checks on at least one hand-verifiable case;
* the selected solution and top alternatives when ambiguity matters;
* score gaps or objective margins for near ties; and
* the policy that consumes infeasible, tied, or no-solution outcomes.

That evidence keeps a feasible output from being mistaken for a validated
modeling decision.
