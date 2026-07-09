Structured Statistical Families
===============================

This page covers distribution families whose observations are not scalar:
records, tuples, sequences, sets, rankings, trees, graphs, vectors, matrices,
and directional data. These are the families that make Mixle useful for
heterogeneous data rather than only iid columns.

Combinators
-----------

Combinators turn smaller distributions into distributions over structured
observations.

.. list-table::
   :header-rows: 1

   * - Family
     - Observation shape
     - Use when
   * - ``CompositeDistribution`` / ``CompositeEstimator``
     - tuple
     - fields are ordered by position.
   * - ``RecordDistribution`` / ``RecordEstimator``
     - named record
     - fields are dictionaries or schema-backed records; dictionary shape
       should be explicit.
   * - ``SequenceDistribution`` / ``SequenceEstimator``
     - variable-length sequence
     - elements share a family and length may be modeled.
   * - ``OptionalDistribution`` / ``OptionalEstimator``
     - missing-or-present value
     - a field can be absent without dropping the whole record.
   * - ``SelectDistribution`` / ``SelectEstimator``
     - dispatch by type or field
     - different subfamilies apply to different observed cases.
   * - ``ConditionalDistribution`` / ``ConditionalDistributionEstimator``
     - conditional relation
     - a distribution depends on observed covariates.
   * - ``TransformDistribution`` / ``TransformEstimator``
     - deterministic transformed value
     - modeling scale differs from observation scale.
   * - ``FiniteStochasticTransformDistribution``
     - stochastic finite mapping
     - a latent finite source emits through a finite channel.
   * - ``TruncatedDistribution`` / ``TruncatedEstimator``
     - restricted support
     - legal or observed support is a subset of the base family.
   * - ``CensoredDistribution`` / ``CensoredEstimator``
     - censored observation
     - true value is partially hidden by thresholds or intervals.
   * - ``SurvivalDistribution`` / ``SurvivalEstimator``
     - event or censoring time
     - time-to-event data with censoring.
   * - ``HurdleDistribution`` / ``HurdleEstimator``
     - structural zero plus positive part
     - zero occurrence and positive magnitude are separate processes.
   * - ``ZeroInflatedDistribution`` / ``ZeroInflatedEstimator``
     - extra zeros
     - count data have more zeros than the base family explains.
   * - ``WeightedDistribution`` / ``WeightedEstimator``
     - weighted observation
     - examples carry frequency or importance weights.
   * - ``IgnoredDistribution`` / ``IgnoredEstimator``
     - ignored field
     - a field must pass through shape validation but not affect likelihood.
   * - ``NullDistribution`` / ``NullEstimator``
     - no information
     - neutral branch in a larger composition.

Combinators are how a heterogeneous row becomes one model:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import CategoricalEstimator, GammaEstimator, RecordEstimator, field

   estimator = RecordEstimator(
       {
           field("event"): CategoricalEstimator(),
           field("duration"): GammaEstimator(),
       }
   )

   model = optimize(records, estimator, out=None)

Validate the record contract before fitting. A structured estimator should
describe the intended observation shape; it should not silently repair rows
where a field changes type, disappears, or uses a different unit.

Multivariate and Matrix Families
--------------------------------

.. list-table::
   :header-rows: 1

   * - Family
     - Observation
     - Use when
   * - ``MultivariateGaussianDistribution`` / ``MultivariateGaussianEstimator``
     - real vector
     - full-covariance Gaussian vector model.
   * - ``DiagonalGaussianDistribution`` / ``DiagonalGaussianEstimator``
     - real vector
     - independent Gaussian dimensions or high-dimensional baseline.
   * - ``MultivariateStudentTDistribution`` / ``MultivariateStudentTEstimator``
     - real vector
     - heavy-tailed vector residuals.
   * - ``GaussianCopulaDistribution`` / ``GaussianCopulaEstimator``
     - vector with copula dependence
     - separate marginal behavior from Gaussian dependence.
   * - ``AitchisonNormalDistribution`` / ``AitchisonNormalEstimator``
     - compositional vector
     - parts constrained to a whole, modeled via the Aitchison logratio
       transform.
   * - ``MultinomialDistribution`` / ``MultinomialEstimator``
     - finite-count vector
     - category counts from repeated trials.
   * - ``IntegerMultinomialDistribution`` / ``IntegerMultinomialEstimator``
     - integer-count vector
     - dense integer-coded multinomial counts.
   * - ``DirichletMultinomialDistribution`` / ``DirichletMultinomialEstimator``
     - count vector
     - over-dispersed multinomial counts.
   * - ``MatrixNormalDistribution`` / ``MatrixNormalEstimator``
     - matrix
     - row/column covariance structure.
   * - ``WishartDistribution`` / ``WishartEstimator``
     - positive-definite matrix
     - covariance-like random matrices.
   * - ``InverseWishartDistribution`` / ``InverseWishartEstimator``
     - positive-definite matrix
     - inverse covariance or Bayesian covariance prior.
   * - ``LKJDistribution`` / ``LKJEstimator``
     - correlation matrix
     - correlation priors or fitted correlation structure.

For full-covariance multivariate Gaussian fits, the default numeric path uses
two safeguards. Weighted second moments are accumulated through a BLAS-backed
matrix multiply instead of a naive tensor contraction, which keeps large mixture
fits from spending most of their time in covariance assembly. The Cholesky path
also has a minimal jitter fallback for nearly positive-definite float32
covariance estimates, so GPU or reduced-precision EM runs can recover from
roundoff without changing the ordinary float64 fast path.

For release evidence, compare reduced-precision or GPU-oriented covariance
fits with a float64 baseline on representative data. Jitter recovery should be
visible in diagnostics when it occurs.

Directional Families
--------------------

Directional observations live on circles, spheres, or orientation manifolds.

.. list-table::
   :header-rows: 1

   * - Family
     - Support
     - Use when
   * - ``VonMisesDistribution`` / ``VonMisesEstimator``
     - circle
     - circular angles.
   * - ``WrappedNormalDistribution`` / ``WrappedNormalEstimator``
     - circle
     - wrapped Gaussian-like angular data.
   * - ``WrappedCauchyDistribution`` / ``WrappedCauchyEstimator``
     - circle
     - heavier-tailed circular data.
   * - ``VonMisesFisherDistribution`` / ``VonMisesFisherEstimator``
     - sphere
     - directional vectors around a mean direction.
   * - ``WatsonDistribution`` / ``WatsonEstimator``
     - axial sphere
     - orientations where sign is not meaningful.
   * - ``KentDistribution`` / ``KentEstimator``
     - sphere
     - directional data with elliptical concentration.
   * - ``BinghamDistribution`` / ``BinghamEstimator``
     - antipodal directional support
     - axial orientation and shape.
   * - ``ProjectedNormalDistribution`` / ``ProjectedNormalEstimator``
     - sphere/circle
     - directions induced by projecting Gaussian vectors.

Directional families require support checks that ordinary vector models do not
need. Normalize or validate angular units, antipodal equivalence, and sphere
constraints before interpreting fitted concentration or orientation.

Sequences and Markov Families
-----------------------------

The sequence package covers explicit Markov structure over observed states:

* ``MarkovChainDistribution`` and ``MarkovChainEstimator``;
* ``IntegerMarkovChainDistribution`` and ``IntegerMarkovChainEstimator``;
* ``MarkovTransformDistribution`` and ``MarkovTransformEstimator``;
* ``SparseMarkovAssociationDistribution`` and
  ``SparseMarkovAssociationEstimator``.

Use these when the observation is an observed-state sequence. Use
:doc:`hmms-latent` when the state path is hidden.

Observed-state sequence families should be checked for start-state and
transition support. An unseen state or impossible transition should remain
visible as support evidence rather than being hidden by preprocessing.

Sets
----

Set families model unordered finite collections:

* ``BernoulliSetDistribution`` and ``BernoulliSetEstimator``;
* ``IntegerBernoulliSetDistribution`` and ``IntegerBernoulliSetEstimator``;
* ``IntegerBernoulliEditDistribution`` and ``IntegerBernoulliEditEstimator``;
* ``IntegerStepBernoulliEditDistribution`` and ``IntegerStepBernoulliEditEstimator``.

Use set families when membership matters but order does not. Use edit families
when the likelihood is naturally expressed as set changes from a reference.

Rankings and Pairwise Preferences
---------------------------------

Ranking families model permutations, partial orderings, or pairwise outcomes:

* ``MallowsDistribution`` and ``MallowsEstimator``;
* ``GeneralizedMallowsDistribution`` and ``GeneralizedMallowsEstimator``;
* ``GeneralizedMallowsModelDistribution`` and ``GeneralizedMallowsModelEstimator``;
* ``PlackettLuceDistribution`` and ``PlackettLuceEstimator``;
* ``SpearmanRankingDistribution`` and ``SpearmanRankingEstimator``;
* ``BradleyTerryDistribution`` and ``BradleyTerryEstimator``;
* ``ThurstoneDistribution`` and ``ThurstoneEstimator``;
* ``ThurstoneMostellerDistribution`` and ``ThurstoneMostellerEstimator``;
* ``DavidsonDistribution`` and ``DavidsonEstimator``;
* ``RaoKupperDistribution`` and ``RaoKupperEstimator``;
* ``LowRankPermutationDistribution`` and ``LowRankPermutationEstimator``;
* ``EwensDistribution`` and ``EwensEstimator``;
* ``MatchingDistribution`` and ``MatchingEstimator``.

Use these for preference data, rankings, pairwise comparison, matching, or
ordering uncertainty. For consensus analysis outside a distribution family, see
:doc:`analysis`.

Preference data need explicit treatment of ties, missing comparisons, and
judge identity. A ranking model can fit the observed orderings while still
being inappropriate for a downstream policy if the comparison process is
biased.

Trees and Graphs
----------------

Tree and graph families score structured graph-valued observations:

* ``ChowLiuTreeDistribution`` and ``ChowLiuTreeEstimator``;
* ``IntegerChowLiuTreeDistribution`` and ``IntegerChowLiuTreeEstimator``;
* ``SpanningTreeDistribution`` and ``SpanningTreeEstimator``;
* ``ErdosRenyiGraphDistribution`` and ``ErdosRenyiGraphEstimator``;
* ``StochasticBlockGraphDistribution`` and ``StochasticBlockGraphEstimator``;
* ``RandomDotProductGraphDistribution`` and ``RandomDotProductGraphEstimator``;
* ``KnowledgeGraphDistribution`` and ``KnowledgeGraphEstimator``;
* ``KnowledgeGraphEnsemble`` and ``fit_knowledge_graph_ensemble``;
* temporal graph grammar families including ``TemporalGraphGrammarDistribution``
  and labeled, homophily, churning, latent, attributed, and latent-churning
  variants.

Use graph distributions when the graph itself is the observation. Use
:doc:`relations` when a graph algorithm is the decision layer after a model has
produced edge or node scores.

Graph-valued observations should carry node identity, edge direction, and
schema assumptions with the artifact. Reindexing a graph can change the meaning
of a fitted model even when the adjacency shape is unchanged.

Workflow
--------

The safest workflow for structured families is:

1. Start with the simplest shape-preserving family.
2. Fit locally on a representative sample.
3. Inspect field-level or component-level likelihoods.
4. Add latent structure only when held-out scoring supports it.
5. Add enumeration, relation, or production constraints only after the
   probabilistic model is behaving sensibly.

Release Evidence
----------------

For structured families, keep:

* a schema example for one valid observation;
* field-level preprocessing and missing-data policy;
* support checks for sequences, sets, rankings, graphs, matrices, or
  directional values;
* baseline comparisons before adding latent or high-capacity structure;
* capability checks for enumeration, conditioning, or backend scoring; and
* diagnostics for malformed or impossible observations.

API Reference
-------------

Generated reference pages:

* :doc:`api/mixle.stats.combinator`;
* :doc:`api/mixle.stats.multivariate`;
* :doc:`api/mixle.stats.matrix`;
* :doc:`api/mixle.stats.directional`;
* :doc:`api/mixle.stats.sequences`;
* :doc:`api/mixle.stats.sets`;
* :doc:`api/mixle.stats.rankings`;
* :doc:`api/mixle.stats.trees`;
* :doc:`api/mixle.stats.graphs`.
