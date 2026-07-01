"""The representation layer -- embed any modality into one shared space; discretize it only if (and how) you want.

The tokenizer question, resolved by *not* fixing a vocabulary. Three separable pieces:

  * :mod:`~mixle.represent.segment` -- a ``Segmenter`` cuts raw data (text, image, signal, sequence, structure)
    into units, committing to a decomposition but no vocabulary;
  * :mod:`~mixle.represent.embed` -- an ``Embedding`` maps each unit into shared ``R^dim`` (a lookup table for a
    discrete unit, a small encoder for a continuous one), sharable/tie-able across models;
  * :mod:`~mixle.represent.quantize` -- an optional, *learned* ``VectorQuantizer`` turns the shared vectors into
    discrete ids **in the embedding space**, so a codebook (a "vocabulary") is fit to the data -- and can be
    shared across modalities -- rather than guessed.

:class:`~mixle.represent.heterogeneous.HeterogeneousEncoder` composes per-modality ``(segmenter, embedding)``
pairs into one ``(N, dim)`` stream a single downstream model consumes -- trainable to a generative or a downstream
objective, so "the right tokenization" is *inferred* under the objective rather than fixed upfront.
"""

from __future__ import annotations

from mixle.represent.embed import CategoricalEmbedding, FeatureEmbedding
from mixle.represent.graph import GraphEmbedding, GraphEncoder
from mixle.represent.heterogeneous import HeterogeneousEncoder, ModalityEncoder
from mixle.represent.quantize import VectorQuantizer
from mixle.represent.segment import (
    ByteSegmenter,
    ElementSegmenter,
    PatchSegmenter,
    Segmenter,
    SetSegmenter,
    WholeSegmenter,
    WindowSegmenter,
)

__all__ = [
    "ByteSegmenter",
    "CategoricalEmbedding",
    "ElementSegmenter",
    "FeatureEmbedding",
    "GraphEmbedding",
    "GraphEncoder",
    "HeterogeneousEncoder",
    "ModalityEncoder",
    "PatchSegmenter",
    "Segmenter",
    "SetSegmenter",
    "VectorQuantizer",
    "WholeSegmenter",
    "WindowSegmenter",
]
