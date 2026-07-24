"""Heterogeneous encoding -- text, images, signals, sequences, and arbitrary scientific structures in ONE space.

A :class:`ModalityEncoder` pairs a segmenter (raw -> units) with an embedding (units -> ``R^dim``): one modality's
path into the shared space. A :class:`HeterogeneousEncoder` is a registry of them plus a learned *modality tag*
(a vector added to each unit so the downstream model knows the source), so a record with several modalities --
an image, a caption, a seismic trace, a molecule -- becomes a single ``(N, dim)`` stream of vectors a single model
(a transformer, a mixture, the dependency-structure learner) consumes. Modalities compose because they land in
the same space; that is the representation-layer form of mixle's "compose heterogeneous things into one model".

Everything is torch modules, so the encoders train end to end -- to a generative objective (wrap the stream in a
language-model / density leaf) or a downstream one (pool + a task head). Shared embeddings (the same
``CategoricalEmbedding``/``FeatureEmbedding`` instance in two modalities) tie their vectors, as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ModalityEncoder:
    """One modality's ``raw -> (n_units, dim)`` path: a segmenter feeding an embedding into the shared space."""

    def __init__(self, segmenter: Any, embedding: Any) -> None:
        self.segmenter = segmenter
        self.embedding = embedding
        self.dim = int(embedding.dim)

    def encode(self, raw: Any) -> Any:
        """Torch tensor ``(n_units, dim)`` -- gradients flow to the embedding (and, via it, train the encoder)."""
        import torch

        units = self.segmenter.segment(raw)
        t = torch.as_tensor(units)
        t = t.long() if self.segmenter.discrete else t.float()
        return self.embedding.module()(t)

    def encode_numpy(self, raw: Any) -> np.ndarray:
        """Detached ``(n_units, dim)`` array -- for eval, quantization, or feeding a non-torch model."""
        import torch

        with torch.no_grad():
            return self.encode(raw).cpu().numpy()


class HeterogeneousEncoder:
    """A registry of per-modality encoders sharing one ``dim`` space, plus a learned modality-type embedding."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.encoders: dict[str, ModalityEncoder] = {}
        self._modality_ids: dict[str, int] = {}
        self._modality_embedding: Any = None  # CategoricalEmbedding over modality ids, built lazily

    def register(self, modality: str, segmenter: Any, embedding: Any) -> HeterogeneousEncoder:
        """Add a modality's ``(segmenter, embedding)``; the embedding's ``dim`` must match the shared space."""
        return self.register_encoder(modality, ModalityEncoder(segmenter, embedding))

    def register_encoder(self, modality: str, encoder: ModalityEncoder) -> HeterogeneousEncoder:
        """Add a pre-built :class:`ModalityEncoder` (e.g. a ``GraphEncoder`` that owns its own segment+embed path).

        Re-registering an already-registered modality (same name -- e.g. swapping its encoder) leaves any
        already-learned modality-tag embedding untouched, since the modality *set* did not change. Registering a
        genuinely new modality grows the tag embedding by one row, preserving already-learned rows for the
        modalities that were already registered.
        """
        if int(encoder.dim) != self.dim:
            raise ValueError(f"modality {modality!r} dim {encoder.dim} != shared dim {self.dim}")
        self.encoders[modality] = encoder
        is_new_modality = modality not in self._modality_ids
        self._modality_ids.setdefault(modality, len(self._modality_ids))
        if is_new_modality and self._modality_embedding is not None:
            self._modality_embedding = self._expand_modality_embedding(self._modality_embedding)
        return self

    def _modality_embed(self) -> Any:
        if self._modality_embedding is None:
            from mixle.models.embedding import CategoricalEmbedding

            self._modality_embedding = CategoricalEmbedding(max(1, len(self._modality_ids)), self.dim, name="modality")
        return self._modality_embedding

    def _expand_modality_embedding(self, old: Any) -> Any:
        """Rebuild the modality-tag embedding with one more row, copying over ``old``'s already-learned rows."""
        import torch

        from mixle.models.embedding import CategoricalEmbedding

        grown = CategoricalEmbedding(len(self._modality_ids), self.dim, name="modality")
        with torch.no_grad():
            grown.module().weight[: old.num_categories] = old.module().weight
        return grown

    def encode(self, record: dict[str, Any]) -> tuple[Any, np.ndarray]:
        """A record ``{modality: raw}`` -> ``(stream, modality_ids)``: one ``(N, dim)`` tensor + each unit's source id.

        Each modality's units are embedded and the modality-tag vector is added, then all are concatenated in
        registration order -- the order modalities were registered, independent of ``record``'s own key order --
        the unified token stream for a downstream model. ``record`` must carry exactly the registered modalities:
        a key naming no registered modality, or a registered modality missing from ``record``, raises ``KeyError``.
        """
        import torch

        extra = [m for m in record if m not in self.encoders]
        if extra:
            raise KeyError(f"no encoder registered for modalit{'y' if len(extra) == 1 else 'ies'} {sorted(extra)}")
        missing = [m for m in self.encoders if m not in record]
        if missing:
            raise KeyError(
                f"record is missing registered modalit{'y' if len(missing) == 1 else 'ies'} {sorted(missing)}"
            )

        parts, tags = [], []
        mod_emb = self._modality_embed().module()
        for modality in self.encoders:  # registration order, independent of record's own key order
            vecs = self.encoders[modality].encode(record[modality])  # (n_units, dim)
            mid = self._modality_ids[modality]
            vecs = vecs + mod_emb(torch.tensor(mid))[None, :]  # add the learned modality tag
            parts.append(vecs)
            tags.extend([mid] * vecs.shape[0])
        stream = torch.cat(parts, dim=0) if parts else torch.zeros((0, self.dim))
        return stream, np.asarray(tags, dtype=np.int64)

    def encode_numpy(self, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Encode a heterogeneous record and return NumPy token stream plus modality ids."""
        import torch

        with torch.no_grad():
            stream, tags = self.encode(record)
            return stream.cpu().numpy(), tags

    def parameters(self) -> list:
        """Every trainable parameter across all modality encoders + the modality-tag embedding (for an optimizer)."""
        seen, out = set(), []
        modules = [e.embedding.module() for e in self.encoders.values()] + [self._modality_embed().module()]
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out
