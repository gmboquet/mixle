"""int8 post-training quantization of distilled students (mixle.task.quantize): real bytes, real fidelity."""

import tempfile
import unittest

import numpy as np

from mixle.task import TaskModel, distill_from_labels, distill_records_from_labels, footprint, quantize_mlp


def _record_task(n, seed):
    rng = np.random.RandomState(seed)
    recs, labels = [], []
    for _ in range(n):
        x = float(rng.normal())
        tag = "p" if rng.random() < 0.5 else "q"
        recs.append((x, tag))
        labels.append("a" if (tag == "p") == (x > 0) else "b")
    return recs, labels


class QuantizeMLPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.train_y = _record_task(240, 0)
        cls.val, cls.val_y = _record_task(120, 1)
        cls.fp32 = distill_records_from_labels(
            cls.train, cls.train_y, dim=128, hidden=[16], epochs=120, lr=1e-2, seed=0
        )
        cls.q = quantize_mlp(cls.fp32)

    def test_numpy_forward_matches_torch_on_dequantized_weights(self):
        import torch

        feats = self.q.adapter.features(self.val[:32])
        ours = self.q.model.logits(feats)
        # torch reference: run the same dequantized weights through torch linear algebra
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
        for i, (w, s, b) in enumerate(self.q.model.layers):
            x = x @ torch.from_numpy(w.astype(np.float32) * s).T + torch.from_numpy(b)
            if i != len(self.q.model.layers) - 1:
                x = torch.relu(x)
        np.testing.assert_allclose(ours, x.numpy(), rtol=1e-5, atol=1e-4)

    def test_quantization_preserves_fidelity_and_cuts_bytes_4x(self):
        fp32_acc = np.mean([self.fp32(r) == y for r, y in zip(self.val, self.val_y)])
        q_acc = np.mean([self.q(r) == y for r, y in zip(self.val, self.val_y)])
        self.assertGreaterEqual(q_acc, fp32_acc - 0.05)  # int8 costs at most a few points
        fp_fp32 = footprint(self.fp32)
        fp_q = footprint(self.q)
        self.assertLess(fp_q.bytes, 0.30 * fp_fp32.bytes)  # ~4x weight compression (biases stay fp32)
        self.assertEqual(fp_q.ops, fp_fp32.ops)  # same MAC count, cheaper per-MAC
        self.assertTrue(fp_q.torch_free)
        self.assertFalse(fp_fp32.torch_free)

    def test_weights_are_actually_int8(self):
        for w, s, b in self.q.model.layers:
            self.assertEqual(w.dtype, np.int8)
            self.assertGreater(s, 0.0)
            self.assertEqual(b.dtype, np.float32)

    def test_artifact_roundtrip_preserves_predictions_and_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            self.q.save(d)
            loaded = TaskModel.load(d)
        self.assertEqual(loaded.payload, "arrays")
        self.assertEqual(loaded.model.layers[0][0].dtype, np.int8)
        self.assertEqual(loaded.batch(self.val[:40]), self.q.batch(self.val[:40]))
        self.assertEqual(loaded.meta.get("quantized", {}).get("bits"), 8)

    def test_text_student_quantizes_and_roundtrips(self):
        texts = [f"sample number {i} with tone {'pos' if i % 2 else 'neg'}" for i in range(160)]
        labels = ["p" if i % 2 else "n" for i in range(160)]
        fp32 = distill_from_labels(texts, labels, dim=128, hidden=[8], epochs=80, lr=1e-2, seed=0)
        q = quantize_mlp(fp32)
        agree = np.mean([q(t) == y for t, y in zip(texts, labels)])
        self.assertGreater(agree, 0.9)  # the parity-of-index tone is trivially learnable
        with tempfile.TemporaryDirectory() as d:
            q.save(d)
            loaded = TaskModel.load(d)
        self.assertEqual(loaded.batch(texts[:20]), q.batch(texts[:20]))

    def test_guards(self):
        with self.assertRaises(NotImplementedError):
            quantize_mlp(self.fp32, bits=2)  # LNS/sub-4-bit rungs are explicitly not wired
        with self.assertRaises(ValueError):
            quantize_mlp(self.q)  # already-quantized (arrays payload) is not a torch student


class Int4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.train_y = _record_task(240, 0)
        cls.val, cls.val_y = _record_task(120, 1)
        cls.fp32 = distill_records_from_labels(
            cls.train, cls.train_y, dim=128, hidden=[16], epochs=120, lr=1e-2, seed=0
        )
        cls.q4 = quantize_mlp(cls.fp32, bits=4)

    def test_nibble_pack_unpack_is_exact(self):
        from mixle.task.quantize import _pack_nibbles, _unpack_nibbles

        rng = np.random.RandomState(0)
        for shape in [(3, 5), (4, 4), (1, 7)]:  # odd and even element counts
            w = rng.randint(-7, 8, size=shape).astype(np.int8)
            packed = _pack_nibbles(w)
            self.assertEqual(packed.dtype, np.uint8)
            self.assertEqual(packed.size, (w.size + 1) // 2)  # two weights per byte
            np.testing.assert_array_equal(_unpack_nibbles(packed, shape), w)

    def test_int4_weights_in_range_and_bytes_8x_smaller(self):
        for w, s, _b in self.q4.model.layers:
            self.assertLessEqual(int(np.abs(w).max()), 7)
            self.assertGreater(s, 0.0)
        fp_fp32 = footprint(self.fp32)
        fp_q4 = footprint(self.q4)
        self.assertLess(fp_q4.bytes, 0.17 * fp_fp32.bytes)  # ~8x on weights; fp32 biases remain
        self.assertTrue(fp_q4.torch_free)

    def test_int4_fidelity_holds_on_the_rule_task(self):
        fp32_acc = np.mean([self.fp32(r) == y for r, y in zip(self.val, self.val_y)])
        q4_acc = np.mean([self.q4(r) == y for r, y in zip(self.val, self.val_y)])
        self.assertGreaterEqual(q4_acc, fp32_acc - 0.10)  # 4-bit costs more than int8; bounded here

    def test_int4_artifact_is_packed_on_disk_and_roundtrips(self):
        arrays = self.q4.model.to_arrays()
        self.assertEqual(int(arrays["bits"]), 4)
        self.assertEqual(arrays["w0"].dtype, np.uint8)  # nibble-packed storage, not int8
        w0_shape = tuple(int(d) for d in arrays["shape0"])
        self.assertEqual(arrays["w0"].size, (int(np.prod(w0_shape)) + 1) // 2)
        with tempfile.TemporaryDirectory() as d:
            self.q4.save(d)
            loaded = TaskModel.load(d)
        self.assertEqual(loaded.model.bits, 4)
        self.assertEqual(loaded.batch(self.val[:40]), self.q4.batch(self.val[:40]))


if __name__ == "__main__":
    unittest.main()
