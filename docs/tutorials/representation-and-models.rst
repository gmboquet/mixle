Representation And Model Families
=================================

This tutorial sketches a heterogeneous representation pipeline and shows how it
connects to model families.

The example record has two modalities:

* text describing an event;
* a numeric signal observed around the same event.

Build A Shared Encoder
----------------------

.. code-block:: python

   from mixle.represent import (
       ByteSegmenter,
       CategoricalEmbedding,
       FeatureEmbedding,
       HeterogeneousEncoder,
       WindowSegmenter,
   )

   encoder = HeterogeneousEncoder(dim=64)
   encoder.register("text", ByteSegmenter(), CategoricalEmbedding(256, 64))
   encoder.register("signal", WindowSegmenter(window=128, hop=64), FeatureEmbedding(128, 64))

   stream, tags = encoder.encode({
       "text": "pump pressure rose quickly",
       "signal": waveform,
   })

The stream can be consumed by a neural sequence model, pooled for a task head,
or summarized before fitting a classical distribution.

Optional Quantization
---------------------

If the downstream model needs discrete ids, learn a codebook in the shared
embedding space.

.. code-block:: python

   from mixle.represent import VectorQuantizer

   vectors, _ = encoder.encode_numpy(record)

   codebook = VectorQuantizer(num_codes=256, dim=64).fit(vectors)
   token_ids = codebook.quantize(vectors)

Quantization is optional. It is most useful for compression, enumeration,
discrete language-model style training, and stable production artifacts.

Choose A Model Family
---------------------

Once the representation exists, choose the model family based on the modeling
question. Prefer stable ``mixle.stats`` families when they fit; use
``mixle.models`` helpers when the applied family is specifically needed and
you are ready to validate it.

.. list-table::
   :header-rows: 1

   * - Question
     - Candidate
   * - What token or event comes next?
     - Incubating ``TransformerLMEstimator`` or ``StreamingTransformer``.
   * - What continuous response follows from features?
     - Applied helpers such as ``GaussianProcessRegressor`` or
       ``RandomForestEstimator``.
   * - Are there unknown clusters?
     - ``fit_truncated_dpm`` or a standard ``MixtureEstimator``.
   * - Is there a latent regime over time?
     - HMMs in :doc:`/hmms-latent`.
   * - Are event times self-exciting?
     - Process models in :doc:`/processes`.
   * - Is the output a structured decision?
     - Relations in :doc:`/relations`.

The representation layer should not decide the modeling story by itself. It
should preserve evidence in a form that lets the right model family use it.

Practical Checks
----------------

Before committing to a representation:

* verify that each modality produces the expected number of units;
* check vector dimensions before registering encoders;
* inspect quantization reconstruction error if a codebook is used;
* hold out data before comparing representation choices;
* keep the representation config with the fitted model artifact.
