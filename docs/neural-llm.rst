Neural and LLM Models
=====================

Neural models enter mixle in three different ways. Keeping them separate makes
the API much easier to reason about.

This page covers an incubating part of the project. The examples are useful
for experiments and for hybrid models that genuinely need a neural likelihood,
but they carry more dependency, training-state, and reproducibility risk than
the core ``mixle.stats`` families.

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
     - you want ``text -> label`` or ``text -> fields`` served cheaply
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

Hybrid Neural-Classical Models
------------------------------

The healthiest reason to use a neural leaf in mixle is that it is only one
channel of the data. For event streams:

.. code-block:: python

   from mixle.models import TransformerLMEstimator
   from mixle.stats import CompositeEstimator, GammaEstimator

   event_estimator = CompositeEstimator(
       (
           TransformerLMEstimator(vocab=500, d_model=128, n_layer=4, block=64),
           GammaEstimator(),
       )
   )

One row is ``((history, next_event_type), wait_time)``. The joint score is:

.. code-block:: text

   log p(next_event_type | history) + log p(wait_time)

This is the pattern behind ``examples/hybrid_llm_example.py``.

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
mixture or another latent wrapper, but it should be treated as an incubating
Torch-backed surface with the usual training and reproducibility checks.

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

Tests also cover MLP and convolutional predictors. The output is still a
probabilistic model with scoring and diagnostics, not just a neural network.

Preference Optimization
-----------------------

``DPOModel`` trains from preference triples ``(prompt, chosen, rejected)``:

.. code-block:: python

   from mixle.inference import estimate
   from mixle.models import DPOModel

   leaf = DPOModel(policy, reference_policy, beta=0.1, m_steps=20, lr=1e-4, device="cuda")
   model = estimate(preference_triples, leaf.estimator())

Use it when the learning signal is comparative preference rather than a
categorical label.

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
* Use ``CompositeEstimator`` when the neural channel should be scored jointly
  with non-neural fields.
* Use ``CategoricalEmbedding`` to tie embeddings across experts.
* Treat ``EnergyModel`` scores as approximately normalized; validate them
  before comparing directly with exact-density leaves.
* Fix seeds, record training settings, and evaluate on held-out data.
* Use ``mixle.describe(model)`` after fitting to see what query capabilities
  the resulting object supports.

The older names ``StreamingTransformerLeaf``, ``NeuralLeaf``,
``SoftmaxNeuralLeaf``, and ``DPOLeaf`` remain as compatibility aliases. Prefer
``StreamingTransformer``, ``NeuralGaussian``, ``NeuralCategorical``, and
``DPOModel`` in new code.

Examples And Tests
------------------

* ``examples/hybrid_llm_example.py``: Transformer event type plus Gamma timing.
* ``examples/shared_embedding_example.py``: shared embeddings across LM mixture
  experts.
* ``mixle/tests/neural_ppl_test.py``: neural PPL, streaming Transformer leaves,
  EWC, DPO, and the direct ``LM`` surface.
