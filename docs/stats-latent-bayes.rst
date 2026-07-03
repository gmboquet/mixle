Latent, Bayesian, And Nonparametric Families
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
* ``SparseMixture`` variants;
* ``SemiSupervisedMixtureDistribution`` and ``SemiSupervisedMixtureEstimator``;
* ``DiracLengthMixtureDistribution`` and ``DiracLengthMixtureEstimator``.

Use mixtures when observations plausibly come from several regimes but the
regime label is not observed. Use :func:`mixle.inference.best_of` or
:doc:`evolution` when local optima matter.

Hidden-State Sequence Models
----------------------------

Hidden-state models introduce a latent path over a sequence:

* ``HiddenMarkovModelDistribution`` and ``HiddenMarkovModelEstimator``;
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

Topic, Attention, And Association Models
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

Grammar And Circuit Families
----------------------------

Structured latent families include:

* ``HeterogeneousPCFGDistribution`` and ``HeterogeneousPCFGEstimator``;
* ``InducedHeterogeneousPCFGEstimator``;
* ``ProbabilisticCircuit`` families;
* ``ProbabilisticPCADistribution`` and ``ProbabilisticPCAEstimator``.

Use these when the latent structure is compositional, grammatical, circuit-like,
or low-rank continuous.

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

Relationship To ``mixle.models``
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
