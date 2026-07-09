Representation Layer
====================

``mixle.represent`` is the representation layer for heterogeneous data. Its
job is to turn text, images, signals, sets, graphs, and arbitrary scientific
objects into a shared vector stream without pretending that one fixed tokenizer
is the correct entry point for every modality.

The layer separates three decisions:

* segmentation: how raw objects are cut into units;
* embedding: how each unit maps into a shared ``R^dim`` space;
* optional quantization: how vectors become learned discrete code ids.

This separation keeps representation choices auditable. When possible, choose
the representation under the modeling objective rather than hard-coding it as
an upstream vocabulary choice.

Segmenters
----------

A ``Segmenter`` exposes ``segment(raw) -> np.ndarray`` and declares whether the
units are discrete ids or continuous features.

.. list-table::
   :header-rows: 1

   * - Segmenter
     - Input
     - Output
   * - ``ByteSegmenter``
     - String or bytes
     - Byte ids in ``[0, 256)``.
   * - ``ElementSegmenter``
     - Sequence over a known alphabet
     - Integer ids.
   * - ``PatchSegmenter``
     - Image array
     - Patch feature vectors.
   * - ``WindowSegmenter``
     - One-dimensional signal
     - Sliding-window feature vectors.
   * - ``WholeSegmenter``
     - One feature vector
     - A single unit.
   * - ``SetSegmenter``
     - Set or list of feature vectors
     - One unit per element.
   * - ``LearnedSegmenter``
     - Objective-coupled sequence input
     - Learned segmentation boundaries.

Segmenters intentionally avoid learned vocabulary decisions. They only decide
where the units are.

Embeddings
----------

An embedding exposes a ``dim`` and a ``module()`` method that returns a Torch
module mapping units into ``(n_units, dim)`` vectors.

``CategoricalEmbedding`` handles discrete ids. ``FeatureEmbedding`` handles
continuous units such as patches, windows, node features, or pooled descriptors.

.. code-block:: python

   from mixle.represent import ByteSegmenter, CategoricalEmbedding

   text_segmenter = ByteSegmenter()
   text_embedding = CategoricalEmbedding(256, dim=64, name="bytes")

For continuous units:

.. code-block:: python

   from mixle.represent import FeatureEmbedding, WindowSegmenter

   signal_segmenter = WindowSegmenter(window=128, hop=64)
   signal_embedding = FeatureEmbedding(in_features=128, dim=64, hidden=(128,))

Passing the same embedding instance to multiple encoders ties the
representation. That is how shared vocabularies, shared feature projections,
and cross-modal alignment can be expressed explicitly.

Heterogeneous Encoding
----------------------

``HeterogeneousEncoder`` is a registry of modality-specific encoders that all
land in the same vector space. Each modality has a segmenter and an embedding.
The encoder adds a learned modality tag and concatenates all units into one
stream.

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
   encoder.register("signal", WindowSegmenter(128, 64), FeatureEmbedding(128, 64))

   stream, tags = encoder.encode({
       "text": "pressure spike",
       "signal": waveform,
   })

The result is a single ``(N, dim)`` stream plus modality ids. A downstream
model can be a Transformer, a density leaf, a mixture, a task head, or a custom
module.

Keep modality ids and segment metadata through validation. If alignment
information is dropped at the representation boundary, diagnostics can no
longer assign errors cleanly to a modality, segmenter, or downstream model.

Vector Quantization
-------------------

``VectorQuantizer`` learns a codebook in the shared embedding space. It does
not impose tokens before embedding; it discretizes vectors after the modalities
have already landed in a common space.

.. code-block:: python

   from mixle.represent import VectorQuantizer

   vectors, _ = encoder.encode_numpy(record)

   vq = VectorQuantizer(num_codes=512, dim=64, seed=0).fit(vectors)
   token_ids = vq.quantize(vectors)
   reconstructed = vq.dequantize(token_ids)
   error = vq.reconstruction_error(vectors)

This is useful for compression, discrete sequence modeling, cross-modal
codebooks, and production artifacts that need a stable finite vocabulary.

Codebooks should be versioned with their fitting data window and reconstruction
diagnostics. A shifted input distribution can make old token ids misleading
even when the downstream model still accepts them.

Graph and Structured Inputs
---------------------------

``GraphEmbedding`` and ``GraphEncoder`` support graph-like inputs. They are
part of the same representation contract: graph units become vectors in the
shared space and can be combined with text, scalar metadata, time series, or
other modalities.

For graph-valued probability models, see :doc:`models` for random graph
families and :doc:`relations` for graph-constrained decisions.

Autoencoders and Fitted Embedders
---------------------------------

``fit_autoencoder`` and ``fit_embedder`` are training utilities for learning
representations from data. Use them when the representation itself should be
fit before being passed into a density model, task model, or downstream
estimator.

The clean separation is:

* ``represent`` learns or applies the input representation;
* ``stats`` and ``models`` define the probability or prediction model;
* ``inference`` fits model parameters;
* ``task`` and ``reason`` turn those models into calibrated decisions and LLM
  workflows.

Modality Vectorization
----------------------

Mixle includes deterministic modality helpers for cases where raw images or
signals need to enter a cross-modal graph as fixed-length vector nodes.

.. code-block:: python

   from mixle.represent.modality import vectorize, vectorize_all

   image_vector = vectorize(image_array, "image", dim=16)
   signal_vectors = vectorize_all(signals, "signal", dim=24)

The public helpers are:

``image_features``
    Mean intensity over a grid of image cells, padded or truncated to a fixed
    dimension.

``signal_features``
    Mean, energy, and range over evenly spaced windows of a one-dimensional
    signal.

``vectorize`` / ``vectorize_all``
    One public dispatcher for ``text``, ``record``, ``image``, and ``signal``
    inputs.

These are dependency-free baseline featurizers. They make an image or signal
field usable in structure discovery and heterogeneous Bayesian-network factors.
Use learned encoders when the task depends on semantics that coarse
deterministic features cannot preserve.

Baseline featurizers are useful validation checks because they avoid optional
heavy dependencies. Do not describe them as semantic image or signal
understanding without task evidence.

Posterior Retrieval
-------------------

``PosteriorRetriever`` retrieves records by model posterior affinity rather
than raw-feature cosine similarity. It is useful after fitting a mixture over
heterogeneous records: two records are near each other when the fitted model's
field-restricted latent posteriors agree.

.. code-block:: python

   import mixle
   from mixle.represent.posterior import PosteriorRetriever

   model = mixle.propose(records, fit=True)
   retriever = PosteriorRetriever(model.fitted, records, evidence_cap=1.0)

   neighbors = retriever.retrieve(query_record, k=5)
   batch_neighbors = retriever.retrieve_batch(query_records, k=5)

The retriever expects a fitted mixture-like model with ``components`` and
``log_w``. Internally it uses the balanced model-affinity factors from
``mixle.utils.hvis`` and an evidence cap so one inconsistent field can contribute
negative evidence without dominating every other field.

This is a reranking and analysis tool for moderate corpora, not a large-scale
vector-index replacement. For large corpora, use an embedding or ANN system for
first-stage recall, then rerank a shortlist with posterior affinity.

Representation API Inventory
----------------------------

.. list-table::
   :header-rows: 1

   * - Import
     - Role
   * - ``Embedder`` / ``fit_embedder``
     - Fit and apply a general embedding model.
   * - ``AutoencoderResult`` / ``fit_autoencoder``
     - Train and inspect autoencoder representations.
   * - ``ModalityEncoder``
     - One registered modality inside ``HeterogeneousEncoder``.
   * - ``PosteriorRetriever``
     - Import from ``mixle.represent.posterior`` for model-posterior retrieval.
   * - ``vectorize`` / ``vectorize_all``
     - Import from ``mixle.represent.modality`` for fixed-dimension modality
       descriptors.
   * - ``image_features`` / ``signal_features``
     - Deterministic baseline features for image and one-dimensional signal
       fields.

Design Guidance
---------------

Use the least committed representation that preserves the information needed
by the objective.

* For text, bytes are a robust baseline when no domain tokenizer is known.
* For categorical scientific sequences, ``ElementSegmenter`` with a known
  alphabet is more interpretable.
* For waveforms and dense signals, windows preserve local temporal structure.
* For images, patches preserve spatially local evidence.
* For records or structures already summarized into features, ``WholeSegmenter``
  can be enough.
* For sets, molecules, graph nodes, or unordered elements, preserve one unit per
  element and let the model learn aggregation.

Quantize only when a discrete bottleneck is needed. Otherwise keep the shared
vectors continuous and let the downstream model use the full representation.

Validation Evidence
-------------------

For representation workflows, preserve:

* accepted modality keys and one representative input shape per modality;
* segmenter, embedding, and modality-tag configuration;
* per-modality unit counts and vector dimensions;
* codebook identity and reconstruction diagnostics when quantization is used;
* baseline model evidence showing the representation carries signal;
* drift checks for modality availability, unit counts, and vector norms; and
* artifact links tying the representation config to the fitted model.
