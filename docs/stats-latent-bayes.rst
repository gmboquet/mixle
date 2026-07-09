Latent, Bayesian, and Nonparametric Families
============================================

This page covers the statistical families where the model contains unobserved
structure or explicit prior/posterior behavior. These are the families most
closely tied to EM, variational inference, posterior inspection, and automatic
model selection.

Latent Mixtures
---------------

Mixture families introduce hidden component assignments:

* ``MixtureDistribution`` and ``MixtureEstimator``;
* ``GaussianMixtureDistribution`` where available through the latent modules;
* ``HeterogeneousMixtureDistribution`` and ``HeterogeneousMixtureEstimator``;
* ``HierarchicalMixtureDistribution`` and ``HierarchicalMixtureEstimator``;
* ``JointMixtureDistribution`` and ``JointMixtureEstimator``;
* ``SemiSupervisedMixtureDistribution`` and ``SemiSupervisedMixtureEstimator``;
* ``DiracLengthMixtureDistribution`` and ``DiracLengthMixtureEstimator``;
* ``sparse_mixture_score`` for certified top-k tail-bound scoring over an
  existing mixture (a diagnostic utility, not a standalone distribution
  family).

Use mixtures when observations plausibly come from several regimes but the
regime label is not observed. Use :func:`mixle.inference.best_of` or
:doc:`evolution` when local optima matter.

Do not interpret component names before checking stability. A useful mixture
fit should survive the application's restart, validation, and responsibility
inspection protocol; otherwise the components are fitting artifacts rather
than reliable latent structure.

Hidden-State Sequence Models
----------------------------

Hidden-state models introduce a latent path over a sequence:

* ``HiddenMarkovModelDistribution`` and ``HiddenMarkovEstimator``;
* ``QuantizedHiddenMarkovModelDistribution`` and quantized HMM estimators;
* ``SegmentalHiddenMarkovModelDistribution`` and segmental estimators;
* ``LookbackHiddenMarkovModel`` families;
* ``ScheduledHiddenMarkovModelDistribution`` and ``ScheduledHMMEstimator``;
* ``TreeHiddenMarkovModelDistribution`` and tree-HMM estimators;
* ``StructuredHMM`` and ``StructuredHMMEstimator``;
* transition operators such as ``DenseTransition``, ``LowRankTransition``,
  ``BlockDiagonalTransition``, ``KroneckerTransition``, and
  ``SparseTransition``.

Use :doc:`hmms-latent` for the practical HMM workflow, including structured
transitions, decoding, and enumeration.

For HMM-style models, keep decoding evidence separate from likelihood
evidence. A model can score sequences well while producing state paths that are
not meaningful for the downstream explanation or control policy.

Topic, Attention, and Association Models
----------------------------------------

Latent structure is not only clustering. Mixle also includes topic and
attention-like latent families:

* ``LDADistribution`` and ``LDAEstimator``;
* ``IntegerProbabilisticLatentSemanticIndexingDistribution`` and estimator;
* ``LabeledLDA`` families;
* ``HiddenAssociationDistribution`` and ``HiddenAssociationEstimator``;
* ``IntegerHiddenAssociationDistribution`` and estimator;
* ``SparseMarkovAssociationDistribution`` and estimator;
* ``ResponsibilityAttentionDistribution`` and estimator;
* ``VariationalEmbeddingAttentionDistribution`` and estimator;
* ``ChainedAttentionDistribution`` and estimator;
* ``VariationalMultiHopAttentionDistribution`` and estimator.

Use these when the latent object is an assignment, topic, association, or
attention path rather than a simple mixture component.

These models should be reviewed through the latent object they expose. Topic
coherence, association calibration, and attention-path stability are different
checks from aggregate likelihood.

Grammar and Circuit Families
----------------------------

Structured latent families include:

* ``HeterogeneousPCFGDistribution`` and ``HeterogeneousPCFGEstimator``;
* ``InducedHeterogeneousPCFGEstimator``;
* ``ProbabilisticCircuit`` families;
* ``ProbabilisticPCADistribution`` and ``ProbabilisticPCAEstimator``.

Use these when the latent structure is compositional, grammatical, circuit-like,
or low-rank continuous.

For grammar and circuit families, validate both probability behavior and
structural validity. A high-scoring parse or circuit explanation should still
respect the grammar, schema, or low-rank assumption that made the family useful.

Bayesian Families
-----------------

``mixle.stats.bayes`` provides prior and posterior-bearing families:

* ``DirichletDistribution`` and ``DirichletEstimator``;
* ``SymmetricDirichletDistribution``;
* ``DictDirichletDistribution``;
* ``NormalGammaDistribution``;
* ``NormalWishartDistribution``;
* ``MultivariateNormalGammaDistribution``;
* ``ConjugatePosterior`` and ``conjugate_posterior``;
* ``MixtureConjugatePosterior`` and ``mixture_conjugate_posterior``;
* ``mixture_prior`` helpers where available.

Use conjugate families when posterior updates should be closed form, streaming,
or easy to audit.

Conjugacy is a modeling assumption, not only a computational convenience.
Record the prior parameters and the sufficient statistics used for each update
so posterior changes can be reconstructed from the artifact.

Bayesian Nonparametrics
-----------------------

The nonparametric family surface includes:

* ``DirichletProcessMixtureDistribution`` and estimator;
* ``HierarchicalDirichletProcessMixtureDistribution`` and estimator;
* ``PitmanYorProcessDistribution`` and estimator;
* ``IndianBuffetProcessDistribution`` and estimator;
* ``ChineseRestaurantProcessDistribution`` and estimator.

Use these when the number of clusters, features, or partitions should not be
fixed too early. In deployed systems, prefer finite truncations or explicit
promotion gates so the inferred complexity remains inspectable.

When complexity is inferred, report the effective complexity that the fit used:
active clusters, active features, occupied tables, or posterior mass assigned
to unused structure. That evidence is often more useful to reviewers than the
nominal truncation level alone.

Posteriors
----------

Posterior helper objects include:

* ``Posterior``;
* ``LatentPosterior``;
* ``CategoricalLatentPosterior``;
* ``MarkovChainLatentPosterior``;
* ``MeanFieldLDAPosterior``.

These objects are useful for exposing responsibilities, state marginals,
posterior predictive behavior, and attribution from latent variables back to
observed data.

Posterior summaries should carry the conditioning data and fitted model
identity that produced them. Responsibilities or marginals detached from their
model version are easy to misuse as permanent labels.

Fitting Guidance
----------------

Latent and Bayesian models are more sensitive to initialization and objective
choice than simple scalar families.

Use this default discipline:

1. Fit a simpler non-latent baseline.
2. Fit the latent model with several random starts.
3. Compare on held-out log score or task loss.
4. Inspect posterior responsibilities, not only aggregate likelihood.
5. Check calibration if the posterior drives decisions.
6. Record the chosen structure and rejected alternatives in provenance.

Example: Mixture With Restarts
------------------------------

.. code-block:: python

   import numpy as np
   from mixle.inference import best_of
   from mixle.stats import GaussianEstimator, MixtureEstimator

   estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])

   score, model = best_of(
       train,
       valid,
       estimator,
       trials=12,
       max_its=100,
       rng=np.random.RandomState(0),
       out=None,
   )

Use ``model`` only after checking that it improves validation behavior over a
single-family baseline.

Release Evidence
----------------

For latent, Bayesian, or nonparametric models, preserve:

* the non-latent or simpler baseline used for comparison;
* initialization and restart settings;
* held-out score or task metric;
* posterior diagnostics such as responsibilities, state marginals, or active
  complexity;
* calibration checks when posterior probabilities drive decisions;
* prior parameters and sufficient statistics for Bayesian updates; and
* rejected alternatives when they explain why the selected structure was
  promoted.

This evidence keeps latent structure from being treated as ground truth merely
because the fitting routine returned a model object.

Relationship to ``mixle.models``
--------------------------------

``mixle.stats`` families are distribution families that participate directly
in the distribution/estimator contract. ``mixle.models`` contains incubating
higher-level model helpers, neural leaves, fitting utilities, graph/POMDP
helpers, and training search tools. When both namespaces contain related
ideas, prefer this rule:

* use ``mixle.stats`` when you need a distribution family inside a composition;
* use ``mixle.models`` when you deliberately need a specialized model helper,
  external training loop, or fit-result utility and are ready to validate it.

API Reference
-------------

Generated reference pages:

* :doc:`api/mixle.stats.latent`;
* :doc:`api/mixle.stats.bayes`;
* :doc:`api/mixle.stats.processes`;
* :doc:`api/mixle.models.dirichlet_process_mixture`;
* :doc:`api/mixle.models.grammar`.
