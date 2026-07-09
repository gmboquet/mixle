Representation and Model Families
=================================

This tutorial sketches a heterogeneous representation pipeline and shows how it
connects to model families.

The example record has two modalities:

* text describing an event;
* a numeric signal observed around the same event.

Build a Shared Encoder
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

Inspect ``tags`` before moving on. A good representation pipeline should make
it clear which embedding rows came from which modality and segment. If the
alignment metadata is lost here, later model diagnostics will be much harder to
explain.

Keep the encoder configuration with the records used to validate it. Segmenter
settings, tokenization behavior, numeric scaling, and modality keys define the
data contract just as much as the model class does.

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

Quantization also changes the validation question. Before using token ids as a
model input, compare reconstruction or nearest-code error across modalities.
Large errors in one modality can make a downstream model look calibrated on
average while failing on the modality that was compressed poorly.

If quantized tokens are persisted, store the codebook identity and fitting data
window. Reusing an old codebook on shifted inputs can look like a model failure
when the representation has actually drifted.

Choose a Model Family
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

Keep the first model deliberately simple. A pooled Gaussian or categorical
baseline is often a better first diagnostic than a large neural sequence model:
it reveals whether the representation contains useful signal before the model
capacity can hide representation problems.

Move to a larger family only when the simpler baseline has answered its
diagnostic question. The goal is not to prove that the baseline is sufficient;
it is to catch invalid alignment, missing modalities, or empty signal before
training cost obscures the failure.

Practical Checks
----------------

Before committing to a representation:

* verify that each modality produces the expected number of units;
* check vector dimensions before registering encoders;
* inspect quantization reconstruction error if a codebook is used;
* hold out data before comparing representation choices;
* keep the representation config with the fitted model artifact.

Release or production evidence should record:

* the modality keys accepted by the encoder;
* segmenter settings, embedding dimensions, and quantizer settings;
* the baseline model used to check that the representation carries signal;
* held-out metrics for each modality or task, not only the aggregate score; and
* the artifact location where the encoder configuration is stored.

Common Failure Modes
--------------------

Silent shape drift
    A segmenter change alters the number of produced units, but the downstream
    model still accepts the tensor. Track expected unit counts or add explicit
    assertions around the representation boundary.

Modality dominance
    One modality produces many more units or much larger vector norms, so pooled
    representations mostly reflect that modality. Normalize deliberately and
    inspect per-modality contribution.

Unreviewed compression
    A codebook improves storage size but destroys rare-event signal. Treat
    quantization as a modeling choice and validate it against the task that will
    consume the tokens.

Release Evidence
----------------

For release-like representation work, include:

* a small schema example showing accepted modality keys and value shapes;
* segmenter and embedding configuration;
* quantizer configuration and reconstruction or nearest-code diagnostics when
  a codebook is used;
* per-modality validation metrics or qualitative checks;
* the baseline model used to verify signal; and
* the artifact reference that ties the encoder configuration to the fitted
  model.

Without that evidence, a representation pipeline is difficult to reproduce and
hard to debug when the downstream model changes.
