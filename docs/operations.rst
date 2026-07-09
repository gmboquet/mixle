Operations
==========

``mixle.ops`` contains the verbs that transform distributions. If
distributions are the nouns and capabilities describe what those nouns can do,
operations describe how a model moves from one capability set to another.

This matters because many practical workflows are not simply "fit a model".
They are "condition this model on evidence", "quantize a continuous leaf so it
can be enumerated", "project a neural source into a simpler family", or "pool
several experts into one distribution". ``mixle.ops`` gives those moves a
single public home.

Capability Signatures
---------------------

.. list-table::
   :header-rows: 1

   * - Operation
     - Input requirement
     - Output behavior
   * - ``quantize(dist, bits)``
     - Any sampleable or CDF-capable distribution
     - Finite categorical approximation with enumeration support.
   * - ``truncate(dist, allowed=..., forbidden=...)``
     - Distribution
     - Renormalized restricted distribution.
   * - ``condition(dist, observed)``
     - ``Conditionable``
     - Conditional distribution over unobserved coordinates.
   * - ``marginalize(dist, keep)``
     - ``Marginalizable``
     - Marginal distribution over selected coordinates.
   * - ``mixture(dists, w)``
     - Sequence of distributions
     - Latent mixture distribution.
   * - ``transform(dist, f)``
     - Distribution plus invertible transform
     - Change-of-variables distribution.
   * - ``tilt(dist, theta)``
     - ``ExponentialFamily``
     - Exponentially tilted distribution.
   * - ``project(source, target)``
     - Sampleable source, fittable target family
     - Forward-KL M-projection into the target family.
   * - ``product_of_experts(dists)``
     - Tractable shared family
     - Log-linear pooled distribution.

The table is intentionally capability-oriented. You should not need to know
the concrete class name before asking whether a transformation is legal.

Operations should fail loudly when the required capability is missing. A manual
fallback can be useful for exploration, but it should be documented as an
approximation and kept separate from exact operation evidence.

Quantization
------------

``quantize`` turns a continuous distribution into a finite categorical
approximation. The result can be enumerated, ranked, and used by algorithms
that require finite support.

.. code-block:: python

   from mixle.ops import quantize
   from mixle.stats import GaussianDistribution

   dist = GaussianDistribution(0.0, 1.0)
   finite = quantize(dist, bits=8)

   top = finite.enumerator().top_k(5)

Use quantization when you need a bridge from continuous uncertainty to
enumeration, top-k search, discrete decision policies, or compact artifacts.
Keep the number of bits tied to the downstream need: more bins improve fidelity
but make enumeration heavier.

Record the support window, binning policy, and approximation error or coverage
diagnostic when a quantized model is used outside exploration.

Truncation
----------

``truncate`` restricts a distribution to an allowed set or removes a forbidden
set, then renormalizes mass.

.. code-block:: python

   from mixle.ops import truncate

   visible = truncate(categorical_model, forbidden={"unknown", "masked"})

This is useful for policy constraints, legal label subsets, active learning
pools, and diagnostic "what if this option were unavailable?" analysis.

Truncation changes normalization. Keep the allowed or forbidden set with the
artifact so later reviewers can distinguish a policy-restricted distribution
from the original model.

Conditioning and Marginalization
--------------------------------

``condition`` and ``marginalize`` are only available when the input distribution
declares the required capability.

.. code-block:: python

   from mixle.ops import condition, marginalize

   posterior = condition(record_model, {0: "premium"})
   only_amount = marginalize(record_model, keep=[2])

Prefer these operations to manual slicing. The distribution itself knows how
to preserve normalization, sufficient statistics, and any family-specific
closed forms.

Conditioning and marginalization are exact only when the model advertises the
capability. If a workflow uses sampling or projection instead, name that route
and keep its error checks separate.

Mixtures
--------

``mixture`` constructs a latent mixture from component distributions and
weights.

.. code-block:: python

   from mixle.ops import mixture
   from mixle.stats import GaussianDistribution

   model = mixture(
       [GaussianDistribution(-2.0, 0.4), GaussianDistribution(2.0, 0.8)],
       w=[0.35, 0.65],
   )

Mixtures are the simplest way to express unobserved regimes. For fitted mixture
estimators and EM workflows, see :doc:`distributions`, :doc:`inference`, and
:doc:`hmms-latent`.

When constructing mixtures manually, validate weights and component support
before scoring. A component with incompatible support can turn ordinary inputs
into impossible observations.

Transforms and Tilts
--------------------

``transform`` applies an invertible change of variables with the appropriate
Jacobian correction. ``tilt`` exponentially reweights an exponential-family
distribution.

Use transforms when the natural modeling scale is not the data scale: logs,
positive constraints, calibrated score transforms, or physical unit changes.
Use tilts when you want to encode a moment preference while staying inside the
exponential-family calculus.

For transforms, record the data scale and model scale. For tilts, record the
moment preference and any normalization check. These details determine how the
result should be interpreted.

Projection
----------

``project`` fits a target family to samples drawn from a source model. This is
a practical M-projection: it minimizes the forward divergence from the source
to the target family as estimated by samples.

.. code-block:: python

   from mixle.ops import project
   from mixle.stats import HiddenMarkovModelEstimator

   simpler = project(neural_sequence_model, HiddenMarkovModelEstimator(...))

Projection is useful for distillation, compression, and production fallback:
sample from a rich model, fit a simpler model, then compare the result with
proper scores before promoting it.

Exact Mixture Projection
------------------------

Some projections do not need samples. For Gaussian mixtures,
``mixle.inference`` exposes exact moment-based compression helpers:

.. code-block:: python

   from mixle.inference import collapse_mixture, moment_project, reduce_mixture

   one_gaussian = collapse_mixture(gaussian_mixture)
   four_components = reduce_mixture(gaussian_mixture, n_components=4)
   projected = moment_project(gaussian_mixture)

``collapse_mixture`` returns the single Gaussian with the same overall mean and
covariance as the mixture. ``reduce_mixture`` repeatedly merges Gaussian
components using an analytic merge cost while preserving the mixture's global
first two moments. ``moment_project`` chooses the exact path when possible and
can delegate back to ``mixle.ops.project`` for sampling-based projection onto a
target family.

The distinction is important:

* use ``mixle.ops.project`` for a general source and target family;
* use ``collapse_mixture`` or ``reduce_mixture`` when the source is a Gaussian
  mixture and the exact closed-form route is available;
* record which route was used, because a sampled projection and an exact
  moment projection have different error profiles.

Product of Experts
------------------

``product_of_experts`` pools distributions geometrically:

.. code-block:: python

   from mixle.ops import product_of_experts

   pooled = product_of_experts([language_prior, policy_filter], weights=[1.0, 0.5])

The exact implementation is available for tractable cases:

* categorical distributions over a shared finite support;
* Gaussian distributions, using precision-weighted pooling.

For arbitrary continuous experts, the normalizing constant is generally
intractable. Use sampling, MCMC, or projection when exact pooling is not
available.

Operations As Audit Boundaries
------------------------------

Operations should be visible in model provenance. A production artifact should
make clear whether a distribution was:

* fitted directly from data;
* conditioned on runtime evidence;
* truncated by a policy rule;
* quantized for enumeration;
* projected from a richer source model;
* pooled from multiple experts.

That distinction matters for debugging, calibration, and governance. A
quantized or projected model can be perfectly useful, but it should not be
mistaken for the original source model.

Release Evidence
----------------

For operation-heavy workflows, preserve:

* the source model and operation sequence;
* capability checks before each exact operation;
* parameters such as quantization bits, truncation sets, conditioning evidence,
  transform definitions, projection sample sizes, or expert weights;
* diagnostics showing normalization, score parity, or projection error; and
* the downstream policy that consumes the transformed distribution.

Common Pitfalls
---------------

* Do not call ``condition`` or ``marginalize`` and then silently fall back to a
  manual approximation. If the capability is missing, either choose a model
  that supports the operation or record the approximation explicitly.
* Do not over-quantize early. Keep continuous models continuous until a finite
  support is required.
* Do not pool experts with incompatible supports unless you have decided what
  zero-probability conflicts mean.
* Do not promote a projected model solely because it is lower-cost. Compare it
  against the source model on held-out data and calibration metrics.
