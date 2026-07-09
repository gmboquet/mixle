"""Structured / graph encoding -- message passing over a structure's elements, so edges actually matter.

A molecule, a knowledge graph, a fossil section's co-occurrence network, a protein contact map: these are
*structures*, not bags. :class:`~mixle.represent.segment.SetSegmenter` gives their elements' features but throws
the edges away. :class:`GraphEmbedding` keeps them -- a small message-passing net (GCN-style) embeds each element
and then mixes it with its neighbours over ``layers`` rounds, so an atom's vector reflects its bonds and a node's
reflects its graph context. The output is one ``(n_elements, dim)`` block in the *same* shared space as every
other modality, so a molecule composes with text/images/signals in one :class:`~mixle.represent.heterogeneous.HeterogeneousEncoder`.

A graph-modality record value is ``(node_features (n, f), adjacency (n, n))``; ``GraphEncoder`` is a
``ModalityEncoder`` you drop into the registry via ``register_encoder``. Trains end to end like any encoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.represent.heterogeneous import ModalityEncoder


class GraphEmbedding:
    """A message-passing embedding: ``(node_features, adjacency) -> (n_nodes, dim)`` with ``layers`` GCN rounds."""

    def __init__(self, in_features: int, dim: int, *, layers: int = 2, name: str | None = None) -> None:
        self.in_features = int(in_features)
        self.dim = int(dim)
        self.layers = int(layers)
        self.name = name
        self._module: Any = None

    def module(self) -> Any:
        """Build or return the Torch graph-embedding module."""
        if self._module is None:
            import torch
            import torch.nn as nn

            in_features, dim, n_layers = self.in_features, self.dim, self.layers

            class GCN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.inp = nn.Linear(in_features, dim)
                    self.self_lin = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
                    self.msg_lin = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

                def forward(self, x: Any, adj: Any) -> Any:
                    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)  # self-loops
                    deg = a.sum(dim=1, keepdim=True).clamp(min=1.0)
                    a = a / deg  # row-normalized mean aggregation
                    h = self.inp(x)
                    for self_lin, msg_lin in zip(self.self_lin, self.msg_lin):
                        h = torch.relu(self_lin(h) + msg_lin(a @ h))  # update = own + aggregated neighbours
                    return h

            self._module = GCN()
        return self._module

    def __repr__(self) -> str:
        tag = f", name={self.name!r}" if self.name else ""
        return f"GraphEmbedding(in_features={self.in_features}, dim={self.dim}, layers={self.layers}{tag})"


class GraphEncoder(ModalityEncoder):
    """A structure modality: ``raw = (node_features (n, f), adjacency (n, n))`` -> per-node vectors in shared ``R^dim``."""

    def __init__(self, embedding: GraphEmbedding) -> None:
        self.segmenter = None  # a graph is not a fixed-decomposition segmentation; the encoder owns both halves
        self.embedding = embedding
        self.dim = int(embedding.dim)

    def encode(self, raw: Any) -> Any:
        """Encode graph nodes and adjacency into Torch embeddings."""
        import torch

        nodes, adj = raw
        x = torch.as_tensor(np.asarray(nodes), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(adj), dtype=torch.float32)
        return self.embedding.module()(x, a)  # (n_nodes, dim)

    def encode_numpy(self, raw: Any) -> np.ndarray:
        """Encode a graph and return NumPy embeddings."""
        import torch

        with torch.no_grad():
            return self.encode(raw).cpu().numpy()
