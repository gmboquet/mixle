Distribution Families
=====================

Most model families are available from ``mixle.stats``. The important idea is
that families are not only scalar densities. They include combinators, latent
wrappers, structured supports, graph/ranking models, Bayesian families, and
process models, all using the same distribution and estimator contract.

How to Choose a Family
----------------------

Start from the shape and support of one observation.

.. list-table::
   :header-rows: 1

   * - Observation
     - Start with
     - Consider when needed
   * - real scalar
     - ``GaussianEstimator``
     - ``StudentTEstimator`` for heavy tails, ``SkewNormalEstimator`` for skew
   * - positive duration or magnitude
     - ``GammaEstimator`` or ``ExponentialEstimator``
     - ``WeibullEstimator`` or ``LogGaussianEstimator`` for richer shapes
   * - non-negative count
     - ``PoissonEstimator``
     - ``NegativeBinomialEstimator`` for over-dispersion
   * - label, token, state
     - ``CategoricalEstimator``
     - ``IntegerCategoricalEstimator`` for dense integer labels
   * - vector
     - ``MultivariateGaussianEstimator``
     - ``DiagonalGaussianEstimator`` for simpler covariance
   * - tuple
     - ``CompositeEstimator``
     - nested composites when fields are themselves structured
   * - dictionary or named record
     - ``RecordEstimator``
     - schemas from ``mixle.data`` for external sources
   * - variable-length sequence
     - ``SequenceEstimator``
     - HMMs when latent state drives the sequence
   * - cluster or regime
     - ``MixtureEstimator``
     - ``best_of`` or validation restarts for local optima
   * - state path through time
     - ``Markov`` in PPL or HMM families
     - ``StructuredHMM`` for constrained transitions

The family can be nested. A mixture of records with a sequence field is still
one estimator tree.

Family choice should be justified by support, shape, and validation behavior.
Do not choose a richer family only because it can fit the training data. For
release evidence, compare against the simplest shape-preserving family that
could plausibly explain the data.

Detailed Catalogs
-----------------

This page is the overview. Use the detailed catalogs when you need the full
family map:

* :doc:`stats-univariate` for scalar continuous and discrete distributions.
* :doc:`stats-structured` for combinators, vectors, matrices, directional
  families, sets, rankings, trees, graphs, and structured supports.
* :doc:`stats-latent-bayes` for mixtures, HMM variants, topic models, grammar
  families, Bayesian priors, posteriors, and nonparametric families.

Basic Usage
-----------

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianDistribution, GaussianEstimator

   dist = GaussianDistribution(mu=0.0, sigma2=1.0)
   print(dist.log_density(0.25))

   fitted = optimize([0.1, -0.2, 0.3], GaussianEstimator(), out=None)
   print(fitted.mu, fitted.sigma2)

Distribution classes represent fitted models. Estimator classes represent what
to fit.

Keep that distinction visible in documentation and artifacts. A distribution
object can score, sample, and expose capabilities; an estimator records the
fitting family and sufficient-statistic route used to produce one.

Combinators
-----------

Combinators build distributions over structured observations.

``CompositeDistribution`` / ``CompositeEstimator``
    Tuple-shaped observations, matched position by position.

``RecordDistribution`` / ``RecordEstimator``
    Named fields, usually dictionaries or schema-backed records.

``SequenceDistribution`` / ``SequenceEstimator``
    Variable-length sequences with an element distribution and optional length
    model.

``OptionalDistribution``
    Values that may be missing.

``TransformDistribution``
    Change of variables or deterministic transformations.

``TruncatedDistribution``, ``CensoredDistribution``, ``HurdleDistribution``,
``ZeroInflatedDistribution``
    Support modifications for common data-generation effects.

Example:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import CategoricalEstimator, CompositeEstimator, GammaEstimator

   rows = [("click", 0.4), ("view", 1.2), ("click", 0.7)]
   est = CompositeEstimator((CategoricalEstimator(), GammaEstimator()))
   model = optimize(rows, est, out=None)

Latent Models
-------------

Latent families add hidden variables over otherwise ordinary observations.

* mixtures and Gaussian mixtures;
* sparse, heterogeneous, hierarchical, joint, and semi-supervised mixtures;
* LDA, LLDA, PLSI variants, and topic models;
* probabilistic PCA;
* hidden association models;
* PCFGs and grammar-related models;
* Indian buffet process and nonparametric latent models;
* HMMs, segmental HMMs, lookback HMMs, tree HMMs, scheduled HMMs, structured
  HMMs, and quantized HMMs.

Latent models usually fit by EM or a variational route. Use
:doc:`hmms-latent` for the HMM and structured-state workflow.

Latent structure should not be treated as observed labels. Inspect
responsibilities, state marginals, and restart stability before interpreting
components, topics, or paths.

Univariate Families
-------------------

Continuous families include Gaussian, Student-t, Logistic, LogGaussian,
Laplace, Uniform, Exponential, Gamma, InverseGamma, InverseGaussian,
HalfNormal, Gumbel, Beta, Weibull, Rayleigh, Pareto, generalized extreme-value
families, Tweedie, Nakagami, Rician, and skew-normal variants.

Discrete families include Categorical, IntegerCategorical, Bernoulli, Binomial,
BetaBinomial, Poisson, Geometric, NegativeBinomial, LogSeries, Skellam, and
PointMass.

Multivariate, Matrix, and Directional Families
----------------------------------------------

Vector families include MultivariateGaussian, DiagonalGaussian,
MultivariateStudentT, GaussianCopula, Composition, CategoricalMultinomial,
IntegerMultinomial, and DirichletMultinomial.

Matrix-valued families include Wishart, InverseWishart, MatrixNormal, and LKJ.

Directional families include VonMises, VonMisesFisher, Watson, Kent, Bingham,
WrappedCauchy, WrappedNormal, and ProjectedNormal.

Structured Supports
-------------------

Some families score and enumerate structured objects rather than simple
vectors:

* Markov chains and Markov transforms;
* Bernoulli sets and integer set distributions;
* rankings and permutations such as Mallows, Plackett-Luce, Bradley-Terry,
  Thurstone, Spearman, Ewens, matching, and paired comparison models;
* trees and graphs such as Chow-Liu trees, spanning trees, Erdos-Renyi graphs,
  stochastic block models, random dot-product graphs, knowledge graphs, and
  graph grammars;
* point processes such as Hawkes, multivariate Hawkes, power-law Hawkes,
  inhomogeneous Poisson, renewal, birth-death, and Chinese restaurant
  processes.

Use :doc:`enumeration` when the question is about top-k, rank, or support
traversal.

Bayesian Families
-----------------

``mixle.stats.bayes`` provides conjugate and nonparametric families including:

* Dirichlet and symmetric/dictionary Dirichlet;
* NormalGamma, NormalWishart, and MultivariateNormalGamma;
* Dirichlet-process mixtures and hierarchical DPMs;
* Pitman-Yor process families.

Many estimators accept ``prior=``. With conjugate priors, the same fit surface
can produce posterior-bearing models.

When priors affect a public result, record the prior parameters and the
sufficient statistics that updated them. Otherwise posterior changes are hard
to reproduce from the fitted model alone.

Generated Kernels and Capabilities
----------------------------------

Many distributions expose metadata used by generated kernels, backend scoring,
conjugate updates, enumeration, or symbolic export. Inspect behavior through
the capability layer:

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))

Do not assume a family supports every operation because it supports
``log_density``. Capabilities are the public way to ask.

Release Evidence
----------------

For distribution-family documentation or promoted artifacts, preserve:

* the observation shape and support that motivated the family;
* the estimator or prototype used to fit the model;
* baseline comparisons for richer families such as mixtures, HMMs, or graph
  models;
* non-finite and impossible-observation behavior where relevant;
* capability checks for any operation used downstream; and
* provenance that distinguishes fitted distributions from estimators and
  transformed artifacts.

Related Guides
--------------

* :doc:`concepts` explains the distribution and estimator contract.
* :doc:`stats-univariate`, :doc:`stats-structured`, and
  :doc:`stats-latent-bayes` provide the detailed family catalogs.
* :doc:`inference` explains fit routes and objectives.
* :doc:`ppl` shows the expression layer over these families.
* :doc:`api/modules` contains the generated reference for every module.
