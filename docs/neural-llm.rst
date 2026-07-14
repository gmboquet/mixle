Neural and LLM Models
=====================

Neural models enter Mixle in three different ways. Keeping them separate makes
the API much easier to reason about.

This page covers the optional neural-model surface. The examples are useful for
hybrid models that genuinely need a neural likelihood, but they carry more
dependency, training-state, and reproducibility risk than the core
``mixle.stats`` families.

Install and document neural dependencies explicitly. A model that only works
with Torch, a GPU device, or a registered module builder must record that
requirement in its artifact manifest and in the deployment environment.

.. list-table::
   :header-rows: 1

   * - Surface
     - What it models
     - Use when
   * - ``mixle.models``
     - neural leaves inside generative/composable models
     - a Transformer, neural categorical, DPO model, or neural Gaussian is
       deliberately part of a larger distribution
   * - ``mixle.task``
     - local task models distilled from teachers
     - you want ``text -> label`` or ``text -> fields`` served cost-effectively
   * - ``mixle.reason``
     - uncertainty over LLM answers
     - you already have ``generate(prompt) -> str`` and need confidence,
       abstention, or claim reliability

This page covers the first surface. See :doc:`task-distillation` and
:doc:`uncertainty` for the other two.

Transformer Leaves
------------------

``TransformerLMEstimator`` is the direct way to put a causal Transformer into a
mixle model. Its observations are ``(context, next_token)`` pairs.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.models import TransformerLMEstimator

   pairs = [
       ([0, 1, 2, 3], 4),
       ([1, 2, 3, 4], 5),
   ]
   est = TransformerLMEstimator(vocab=8000, d_model=256, n_layer=4, n_head=4, block=64)
   model = optimize(pairs, est, max_its=20, out=None)

   log_p_next = model.log_density((context_ids, next_id))

The fitted object is a conditional next-token distribution. It does not sample
complete sequences through ``sampler``; feed contexts and use the model's
next-token scoring or prediction behavior.

Shared Language-Model Experts
-----------------------------

A practical reason to use a Transformer leaf inside Mixle is to build several
language-model experts that share part of their parameterization. Each expert
still models the same observation shape, ``(context, next_token)``, while the
mixture learns which expert explains each example.

.. code-block:: python

   from mixle.models import CategoricalEmbedding, TransformerLMEstimator
   from mixle.stats import MixtureEstimator

   embedding = CategoricalEmbedding(num_categories=8000, dim=256, name="word")
   experts = [
       TransformerLMEstimator(8000, d_model=256, n_layer=4, block=64, embedding=embedding)
       for _ in range(3)
   ]
   est = MixtureEstimator(experts)

This is a coherent neural-composition example: the latent mixture separates
token-stream regimes, while ``CategoricalEmbedding`` prevents every expert from
learning an independent vocabulary table.

Analytic and Routed Fitting
---------------------------

``GradLeaf`` does not default to Adam. If a module defines
``mixle_analytic_m_step(*fields, weights=..., batch_size=...)``, Mixle runs that
exact or symbolically generated update and skips autograd. Otherwise it plans
updates per parameter: embeddings and routers use AdaGrad, large matrices use
Muon, stable full-batch blocks use Rprop, and sign-unstable small blocks use
momentum or AdaGrad. The returned leaf's ``fit_receipt`` records the selected
families and reasons.

Pass ``optimizer="adam"`` or another named/callable optimizer only when the
automatic plan is inappropriate. A module-owned analytic M-step has priority
because eliminating an iterative solve is generally more valuable than tuning
the iterative optimizer.

Neural Density Leaves
---------------------

``NeuralDensity`` and ``NeuralConditionalDensity`` adapt Torch modules that
already expose density methods into Mixle leaves. They are useful when the
neural part is a probability model, not just a feature extractor.

.. list-table::
   :header-rows: 1

   * - Surface
     - Models
     - Ready builders
   * - ``NeuralDensity``
     - unconditional ``p(x)`` with ``module.log_density(x)``
     - ``build_coupling_flow``, ``build_maf``, ``build_vae``
   * - ``NeuralConditionalDensity``
     - conditional ``p(y | x)`` with ``module.log_density(x, y)`` and
       ``module.sample_given(x)``
     - ``build_mdn``, ``build_conditional_flow``

``build_mdn`` builds a mixture density network. It is useful when ``p(y | x)``
is multimodal or heteroscedastic, but its components are still diagonal
Gaussians. ``build_conditional_flow`` builds the exact-density counterpart: a
conditional coupling flow whose invertible transform of ``y`` is conditioned on
``x``. Use the flow when the target dimensions have nonlinear within-``y``
dependence and exact log-density matters.

For discrete vectors, ``build_autoregressive_categorical`` provides an exact
unconditional neural density over ``{0, ..., C-1}^d``. Its conditional sibling,
``build_conditional_autoregressive_categorical``, provides exact ``p(y | x)``
for discrete target vectors. Use these builders when a categorical output has
strong coordinate dependence and independent categorical leaves would erase
the structure.

.. code-block:: python

   from mixle.models import NeuralConditionalDensity, build_conditional_flow

   module = build_conditional_flow(x_dim=4, y_dim=2, hidden=64, layers=4)
   leaf = NeuralConditionalDensity(module, m_steps=80, lr=1.0e-3)
   estimator = leaf.estimator()

Observations for a conditional density leaf are ``(x, y)`` pairs. The M-step is
responsibility-weighted negative log-likelihood, so the leaf can sit inside a
mixture or another latent wrapper. Treat the fitted object as a Torch-backed
artifact: record dependencies, training settings, validation data, and reload
behavior.

Constructible Neural Density Families
-------------------------------------

Mixle provides direct distribution classes for common neural density families.
Use them when the model tree should contain a neural density leaf without first
building a Torch module and then wrapping it.

.. code-block:: python

   from mixle.models import Flow, VAE
   from mixle.stats import MixtureDistribution

   prior_shape = MixtureDistribution(
       [Flow(dim=4, hidden=64, layers=4), VAE(dim=4, latent=2)],
       [0.5, 0.5],
   )
   estimator = prior_shape.estimator()

Available constructible families are:

``Flow``
    Exact continuous density via a RealNVP-style coupling flow.

``MAF``
    Exact continuous density via a masked autoregressive flow.

``VAE``
    Latent-variable density with an ELBO-style lower-bound score. Compare it
    against other bounded neural leaves carefully; it is not an exact
    likelihood like a flow.

``DiscreteAR``
    Exact normalized autoregressive density over fixed-length discrete vectors.

These classes still use the same ``NeuralDensityEstimator`` route underneath,
so EM responsibilities and sample weights reach the neural M-step.

Energy Models
-------------

``EnergyModel`` wraps a Torch module whose ``energy(x)`` method returns a
scalar compatibility score. Unlike normalizing flows, an energy model has an
intractable normalizer, so Mixle trains it with noise-contrastive estimation
and reports an approximately normalized ``log_density``.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.models import EnergyModel, build_energy_net

   leaf = EnergyModel(build_energy_net(dim=2, hidden=64, layers=3))
   fitted = optimize(points, leaf.estimator(), max_its=10, out=None)
   score = fitted.log_density([0.2, -0.5])

Use this when compatibility is a better inductive bias than an invertible flow
or an autoregressive factorization. The sampler uses Langevin dynamics, and the
normalization caveat should be recorded when mixing an energy leaf with exact
density leaves.

Streaming Transformer
---------------------

``StreamingTransformer`` is the lower-level adapter when you already have a
Torch module. Its accumulator owns a persistent optimizer and trains on
streamed micro-batches, so it does not materialize the whole corpus as
sufficient statistics.

.. code-block:: python

   from mixle.models import LM, StreamingTransformer

   leaf = StreamingTransformer(
       LM(vocab=K, d_model=96, n_layer=3, n_head=4, block=B).module
   )
   est = leaf.estimator()

Use ``TransformerLMEstimator`` first. Reach for ``StreamingTransformer``
when you need to bring your own module, control streaming behavior, or share a
live module across a larger fitting loop.

Streaming Transformer accumulation preserves sample weights. When the leaf sits
below a mixture or HMM, EM responsibilities are passed into the streaming update
instead of being discarded. That makes the streaming adapter consistent with the
other neural leaves for latent-model M-steps.

Shared Embeddings
-----------------

Mixtures of language-model experts can waste memory if every expert learns its
own token embedding. ``CategoricalEmbedding`` declares one table and hands it
to each expert.

.. code-block:: python

   from mixle.models import CategoricalEmbedding, TransformerLMEstimator
   from mixle.stats import MixtureEstimator

   emb = CategoricalEmbedding(num_categories=8000, dim=256, name="word")
   experts = [
       TransformerLMEstimator(8000, d_model=256, n_layer=4, block=64, embedding=emb)
       for _ in range(3)
   ]
   estimator = MixtureEstimator(experts)

All three experts share token vectors while learning separate expert dynamics.
Run ``examples/shared_embedding_example.py`` to see the parameter accounting.

Direct LM Helper
----------------

``LM`` is a small direct language-model helper:

.. code-block:: python

   from mixle.models import LM

   lm = LM(vocab=5000, d_model=128, n_layer=2, n_head=4, block=64)
   lm.fit(token_ids, epochs=10, batch_size=128, lr=3e-4)
   out = lm.generate(prompt_ids, n=20, temperature=0.8, seed=0)
   nll = lm.nll(token_ids)

Use ``LM`` directly when the language model is the artifact. Use
``TransformerLMEstimator`` when the language model is one leaf in a larger
distribution.

Neural PPL Predictors
---------------------

The PPL can place neural predictors inside distribution parameters:

.. code-block:: python

   from mixle.ppl import Categorical, Transformer

   model = Categorical(
       logits=Transformer(out=vocab, d_model=64, n_layer=2, n_head=4)
   ).fit(
       next_tokens,
       given={"x": contexts},
       epochs=40,
       batch_size=128,
       lr=0.003,
   )

The same contract supports MLP and convolutional predictors. The output is
still a probabilistic model with scoring and diagnostics, not just a neural
network.

Preference Optimization
-----------------------

``DPOModel`` trains from preference triples ``(prompt, chosen, rejected)``:

.. code-block:: python

   from mixle.inference import estimate
   from mixle.models import DPOModel

   leaf = DPOModel(policy, reference_policy, beta=0.1, m_steps=20, lr=1e-4, device="cuda")
   model = estimate(preference_triples, leaf.estimator())

Use it when the learning signal is comparative preference rather than a
categorical label. DPO accumulation preserves per-pair weights, so
responsibilities, streaming decay, or sample weights affect the DPO loss instead
of being dropped.

Serialization and Artifacts
---------------------------

The neural surface now supports more durable round trips:

* ``LM`` objects can serialize trained state and guard edge cases such as empty
  sequences.
* ``StreamingTransformer`` and ``DPOModel`` expose ``to_dict``/``to_json``
  style state.
* ``NeuralGaussian``, ``NeuralCategorical``, ``NeuralDensity``,
  ``NeuralConditionalDensity``, ``EnergyModel``, and related density leaves
  can round-trip through pickle and JSON-style state where the module builder
  is registered.
* The constructible families ``VAE``, ``Flow``, ``MAF``, and ``DiscreteAR`` are
  registered with the serialization layer.

Treat serialization as an artifact boundary, not a quality guarantee. Reload
the model in a fresh process and rerun a small scoring or prediction check
before relying on it in a service.

Device and Dependency Checks
----------------------------

Neural artifacts should not assume the training environment is still present at
serve time. Before promoting a neural or LLM-centered model, verify:

* the artifact loads on the intended CPU or GPU target;
* missing optional dependencies produce a clear error or a documented fallback;
* seeds and training settings are recorded;
* a held-out prediction or log-density matches the pre-save result within the
  expected numerical tolerance;
* the artifact does not claim unconditional sampling when the leaf is purely
  conditional.

These checks are especially important for streaming Transformer leaves, DPO
models, and custom neural-density modules whose builder registration is part of
the artifact contract.

How Neural Leaves Compose with Latents
--------------------------------------

A neural leaf can sit under a latent wrapper. For a mixture of Transformer
experts, EM assigns responsibilities to examples, and each expert's M-step is
gradient fitting under those weights. For an HMM, the neural emission can train
against expected state occupancy. The parent latent model sees an estimator;
the child decides how to do its M-step.

Practical Checklist
-------------------

* Install ``mixle[torch]``.
* Start with ``TransformerLMEstimator`` unless you already own the module.
* Keep one observation as ``(context, target)`` for next-token leaves.
* Use neural leaves when the neural likelihood is the natural model for that
  field or latent expert.
* Use ``CategoricalEmbedding`` to tie embeddings across experts.
* Prefer ``Flow``, ``MAF``, ``VAE``, or ``DiscreteAR`` when a common neural
  density family is the target distribution.
* Treat ``EnergyModel`` scores as approximately normalized; validate them
  before comparing directly with exact-density leaves.
* Verify serialized neural artifacts after load, especially when optional
  Torch or device state is involved.
* Fix seeds, record training settings, and evaluate on held-out data.
* Use ``mixle.describe(model)`` after fitting to see what query capabilities
  the resulting object supports.

The older names ``StreamingTransformerLeaf``, ``NeuralLeaf``,
``SoftmaxNeuralLeaf``, and ``DPOLeaf`` remain as compatibility aliases. Prefer
``StreamingTransformer``, ``NeuralGaussian``, ``NeuralCategorical``, and
``DPOModel`` in new code.

Composition Grid
-----------------

The composition grid makes the advertised behavior checkable. It lists which
latent or structured contexts are validated and where a JSON or pickle round
trip is part of that validation rather than assumed.

.. list-table::
   :header-rows: 1

   * - Leaf
     - Mixture component
     - Composite field
     - HMM emission
     - Serializes in each context
   * - ``NeuralGaussian`` (conditional, ``p(y|x)``)
     - tested
     - tested
     - tested
     - tested in all three
   * - ``NeuralCategorical`` (conditional, ``p(y|x)``)
     - tested
     - tested
     - tested
     - tested in all three
   * - ``NeuralDensity`` / ``Flow`` / ``VAE`` / ``MAF`` / ``DiscreteAR`` (unconditional)
     - tested
     - tested
     - tested
     - tested in all three
   * - ``NeuralConditionalDensity`` (``mixture_density``, conditional)
     - tested
     - tested
     - tested
     - tested in all three
   * - ``EnergyModel`` (unconditional, approximately normalized)
     - tested
     - tested
     - not currently validated
     - tested (Mixture, Composite)

A purely conditional leaf such as ``NeuralCategorical``, ``NeuralGaussian``,
or ``NeuralConditionalDensity`` has no marginal ``p(x)`` to draw from. A
Mixture or HMM whose every component or state is conditional therefore cannot
use the model's own ``sampler()``; ``NeuralCategorical.sample()`` raises by
design. Fitting still works because ``optimize`` needs ``log_density`` and the
accumulator contract, not unconditional sampling, so a conditionally emitting
HMM trains normally on externally supplied ``(x, y)`` sequences.

``EnergyModel`` as an HMM emission is outside the current validation grid.
Do not document that route as supported until it has the same fitting,
scoring, and serialization evidence as the validated neural leaves.

Examples and Validation
-----------------------

Validation for neural and LLM-centered models should cover:

* shared embeddings across language-model mixture experts;
* neural PPL predictors, streaming Transformer leaves, EWC, DPO, and the
  direct ``LM`` surface;
* constructible neural-density families; and
* neural and LM artifact round trips in a fresh process.
