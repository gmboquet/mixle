"""Acceptance tests for mixle.task.deploy_family (roadmap F11: deployment of the family).

Builds a real J2 checkpoint family (small trained headline + two rungs, reusing
``checkpoint_family_ladder_test``'s training curriculum at reduced steps) and a real J4 served
edge cascade (reusing ``frontier_to_native_test``'s frontier/student setup), then exercises F11's
own acceptance criterion: an end-to-end serve receipt carrying a real cost/quality frontier across
the family, plus the edge tier's own real cost/quality receipt reported alongside it.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.eval_harness import markov_transition_matrix  # noqa: E402
from mixle.models.transformer import build_causal_lm  # noqa: E402
from mixle.task.checkpoint_family_ladder import RungSpec, build_checkpoint_family  # noqa: E402
from mixle.task.deploy_family import (  # noqa: E402
    ArtifactReceipt,
    deploy_family,
    quantize_family_artifacts,
)
from mixle.task.distill import distill_records_from_labels  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402
from mixle.task.edge import footprint  # noqa: E402
from mixle.task.frontier_to_native import (  # noqa: E402
    build_served_cascade,
    distill_to_lns_student,
    measure_cascade_receipt,
)

pytestmark = pytest.mark.fast


# --- a small real (not random-init) headline causal LM, trained on F10's own four eval axes at
# reduced steps -- enough to be measurably above chance without J2's own full 8000-step budget. -----


def _train_small_headline(seed: int, vocab: int, d_model: int, n_layer: int, n_head: int, block: int, steps: int):
    torch.manual_seed(seed)
    model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)

    trans = markov_transition_matrix(vocab)
    rng = np.random.default_rng(seed + 1000)
    ctx_len = min(block, 16)
    modulus = min(vocab - 2, 10)
    plus_id, eq_id = vocab - 2, vocab - 1
    bit_len = min(block, 5)

    def batch_perplexity(bs):
        seqs = np.empty((bs, ctx_len), dtype=np.int64)
        cur = rng.integers(0, vocab, size=bs)
        seqs[:, 0] = cur
        for t in range(1, ctx_len):
            nxt = np.array([rng.choice(vocab, p=trans[c]) for c in cur])
            seqs[:, t] = nxt
            cur = nxt
        return seqs[:, :-1], seqs[:, -1]

    def batch_arithmetic(bs):
        a = rng.integers(0, modulus, size=bs)
        b = rng.integers(0, modulus, size=bs)
        target = (a + b) % modulus
        seq = np.stack([a, np.full(bs, plus_id), b, np.full(bs, eq_id)], axis=1)
        return seq, target

    def batch_parity(bs):
        bits = rng.integers(0, 2, size=(bs, bit_len))
        target = bits.sum(axis=1) % 2
        return bits, target

    def batch_induction(bs):
        seqs = rng.integers(0, vocab, size=(bs, ctx_len))
        a = rng.integers(0, vocab, size=bs)
        b = rng.integers(0, vocab, size=bs)
        b = np.where(b == a, (b + 1) % vocab, b)
        plant_pos = rng.integers(0, ctx_len - 3, size=bs)
        for i in range(bs):
            p = int(plant_pos[i])
            seqs[i, p] = a[i]
            seqs[i, p + 1] = b[i]
            seqs[i, ctx_len - 1] = a[i]
        return seqs, b

    batchers = (batch_perplexity, batch_arithmetic, batch_parity, batch_induction)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    for step in range(steps):
        x_np, y_np = batchers[step % len(batchers)](64)
        x = torch.as_tensor(x_np.astype(np.int64))
        y = torch.as_tensor(y_np.astype(np.int64))
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


# --- a small J4 frontier/student pair, straight from frontier_to_native_test's own setup. -----------


def _truth(records):
    out = []
    for r in records:
        score = (1.5 if r["region"] == "west" else -0.5) + 0.4 * r["spend"] + 0.3 * r["visits"]
        out.append("churn" if score < 1.0 else "retain")
    return out


def _gen(n, seed):
    rng = np.random.RandomState(seed)
    return [
        {
            "region": rng.choice(["west", "east"]),
            "spend": float(rng.normal(2.0, 1.5)),
            "visits": int(rng.poisson(3)),
        }
        for _ in range(n)
    ]


class _Teacher:
    def __init__(self, frontier):
        self.frontier = frontier

    def __call__(self, records):
        return self.frontier.batch(list(records))


def _build_edge_cascade_receipt():
    records = _gen(1200, 0)
    labels = _truth(records)
    frontier = distill_records_from_labels(
        records, labels, dim=512, hidden=[128], epochs=120, lr=1e-2, seed=0, task="frontier"
    )
    teacher = _Teacher(frontier)
    teacher_bytes = footprint(frontier).bytes

    train = _gen(500, 1)
    lns_student = distill_to_lns_student(teacher, train, min_gain=1.0, seed=0, task="lns_edge_student")

    cal = _gen(150, 2)
    cost = CostModel(c_frontier=1.0, c_local=0.001)
    cascade = build_served_cascade(lns_student, teacher, cal, alpha=0.1, cost=cost)

    test_records = _gen(250, 3)
    truth = _truth(test_records)
    return measure_cascade_receipt(cascade, test_records, truth, teacher_bytes=teacher_bytes)


class QuantizeFamilyArtifactsTest(unittest.TestCase):
    """I1 rolled up over a real torch model: real measured bytes/error, never assumed."""

    def test_artifact_receipt_is_real_and_smaller_than_dense(self):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=23, d_model=16, n_layer=3, n_head=2, block=12)
        artifact = quantize_family_artifacts(model, name="probe", seed=0)

        self.assertIsInstance(artifact, ArtifactReceipt)
        self.assertGreater(artifact.n_tensors, 0)
        self.assertGreater(artifact.dense_bytes, 0)
        self.assertGreater(artifact.quantized_bytes, 0)
        # int8/int4/lns/sorted_profile: every method should beat fp32 dense storage on a real model.
        self.assertLess(artifact.quantized_bytes, artifact.dense_bytes)
        self.assertGreater(artifact.compression_ratio, 1.0)
        self.assertGreaterEqual(artifact.mean_reconstruction_error, 0.0)
        self.assertEqual(sum(artifact.method_counts.values()), artifact.n_tensors)

    def test_empty_model_raises(self):
        class _Empty(torch.nn.Module):
            pass

        with self.assertRaises(ValueError):
            quantize_family_artifacts(_Empty(), name="empty")


class DeployFamilyEndToEndTest(unittest.TestCase):
    """The F11 acceptance criterion: a real end-to-end serve receipt (cost/quality frontier) built
    from a real J2 family, I1-quantized, priced with a real CostModel, next to a real J4 edge-tier
    served-cascade receipt."""

    @classmethod
    def setUpClass(cls):
        vocab, d_model, n_layer, n_head, block = 23, 16, 6, 2, 12
        cls.headline = _train_small_headline(
            seed=7, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, steps=1500
        )

        calib_rng = np.random.RandomState(7)
        calibration_data = torch.as_tensor(calib_rng.randint(0, vocab, size=(150, block)), dtype=torch.long)

        rung_specs = [
            RungSpec(name="rung_mid", real_target="1B-equivalent stand-in", budget=0.4, trust_region=0.4, seed=0),
            RungSpec(name="rung_edge", real_target="edge-equivalent stand-in", budget=1.5, trust_region=1.5, seed=0),
        ]
        cls.family = build_checkpoint_family(
            cls.headline, rung_specs, calibration_data=calibration_data, eval_n_examples=256
        )

        cls.edge_receipt = _build_edge_cascade_receipt()

        cls.serve_receipt = deploy_family(
            cls.family,
            cls.headline,
            edge_cascade_receipt=cls.edge_receipt,
            cost=CostModel(c_frontier=1.0),
            seed=0,
        )

    def test_family_actually_shrank(self):
        # sanity on the fixture itself: the ladder did produce a real, non-increasing size sequence.
        self.assertGreaterEqual(len(self.family.rungs), 1)
        n_params = [self.family.headline_n_params] + [r.n_params for r in self.family.rungs]
        for a, b in zip(n_params, n_params[1:]):
            self.assertLessEqual(b, a)

    def test_serve_receipt_has_one_point_per_family_member(self):
        print("\n" + self.serve_receipt.summary())
        # headline + every attempted rung, each with a real quantized artifact.
        self.assertEqual(len(self.serve_receipt.points), 1 + len(self.family.rungs))
        names = {p.name for p in self.serve_receipt.points}
        self.assertIn("headline", names)
        for rung in self.family.rungs:
            self.assertIn(rung.name, names)

    def test_cost_tracks_real_measured_artifact_bytes(self):
        headline_point = next(p for p in self.serve_receipt.points if p.name == "headline")
        # headline is priced at exactly CostModel.c_frontier (the pricing anchor).
        self.assertAlmostEqual(headline_point.cost_per_request, 1.0, places=9)
        for p in self.serve_receipt.points:
            self.assertGreater(p.cost_per_request, 0.0)
            self.assertGreater(p.artifact.quantized_bytes, 0)
            # the cost/bytes ratio is identical across every point (same linear scaling by construction) --
            # a real, checkable relationship between the priced number and the measured one.
            ratio = p.cost_per_request / p.artifact.quantized_bytes
            headline_ratio = headline_point.cost_per_request / headline_point.artifact.quantized_bytes
            self.assertAlmostEqual(ratio, headline_ratio, places=9)

    def test_quality_scores_are_real_and_bounded(self):
        for p in self.serve_receipt.points:
            self.assertGreaterEqual(p.quality, 0.0)
            self.assertLessEqual(p.quality, 1.0)

    def test_frontier_plot_renders_every_point(self):
        plot = self.serve_receipt.frontier_plot()
        for p in self.serve_receipt.points:
            self.assertIn(p.name, plot)

    def test_edge_cascade_receipt_is_carried_through_unmodified(self):
        self.assertIs(self.serve_receipt.edge_cascade, self.edge_receipt)
        self.assertTrue(self.edge_receipt.earns_its_complexity())
        self.assertIn("served", self.serve_receipt.summary())


if __name__ == "__main__":
    unittest.main()
