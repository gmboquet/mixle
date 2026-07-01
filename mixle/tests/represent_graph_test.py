"""Structured/graph encoding (mixle.represent.graph): message passing so edges matter, in the shared space.

A molecule/graph must embed to per-node vectors that depend on the adjacency (not just node features), compose in
the HeterogeneousEncoder like any modality, and train end to end.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.represent import (  # noqa: E402
    ByteSegmenter,
    CategoricalEmbedding,
    GraphEmbedding,
    GraphEncoder,
    HeterogeneousEncoder,
)

DIM = 12


def _molecule(seed=0, n=6, f=4):
    rng = np.random.RandomState(seed)
    nodes = rng.rand(n, f).astype(np.float32)
    adj = (rng.rand(n, n) < 0.4).astype(np.float32)
    adj = np.maximum(adj, adj.T)  # undirected
    np.fill_diagonal(adj, 0.0)
    return nodes, adj


class GraphEncoderTest(unittest.TestCase):
    def test_per_node_embeddings_in_shared_space(self):
        nodes, adj = _molecule()
        enc = GraphEncoder(GraphEmbedding(in_features=4, dim=DIM, layers=2))
        out = enc.encode_numpy((nodes, adj))
        self.assertEqual(out.shape, (nodes.shape[0], DIM))

    def test_edges_actually_matter(self):
        # same node features, different adjacency -> different embeddings (message passing used the edges)
        nodes, adj = _molecule(1)
        enc = GraphEncoder(GraphEmbedding(in_features=4, dim=DIM, layers=2))
        torch.manual_seed(0)
        _ = enc.embedding.module()  # build once so both calls share weights
        with_edges = enc.encode_numpy((nodes, adj))
        no_edges = enc.encode_numpy((nodes, np.zeros_like(adj)))
        self.assertFalse(np.allclose(with_edges, no_edges))  # message passing used the adjacency
        self.assertGreater(float(np.mean(np.linalg.norm(with_edges - no_edges, axis=1))), 1e-3)

    def test_composes_in_heterogeneous_registry(self):
        enc = HeterogeneousEncoder(dim=DIM)
        enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))
        enc.register_encoder("molecule", GraphEncoder(GraphEmbedding(in_features=4, dim=DIM)))
        nodes, adj = _molecule(2)
        stream, tags = enc.encode_numpy({"text": "hi", "molecule": (nodes, adj)})
        self.assertEqual(stream.shape, (2 + nodes.shape[0], DIM))  # 2 bytes + n atoms
        self.assertEqual(len(set(tags.tolist())), 2)

    def test_trains_end_to_end(self):
        enc = HeterogeneousEncoder(dim=DIM)
        enc.register_encoder("molecule", GraphEncoder(GraphEmbedding(in_features=4, dim=DIM)))
        head = torch.nn.Linear(DIM, 2)
        opt = torch.optim.Adam(enc.parameters() + list(head.parameters()), lr=1e-2)
        mols = [_molecule(i) for i in range(10)]
        y = torch.tensor([i % 2 for i in range(10)])
        w0 = next(iter(enc.parameters())).detach().clone()
        loss0 = None
        for step in range(20):
            opt.zero_grad()
            pooled = torch.stack([enc.encode({"molecule": m})[0].mean(dim=0) for m in mols])
            loss = torch.nn.functional.cross_entropy(head(pooled), y)
            if step == 0:
                loss0 = float(loss.detach())
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()), loss0)
        self.assertFalse(torch.allclose(w0, next(iter(enc.parameters()))))  # the GNN trained


if __name__ == "__main__":
    unittest.main()
