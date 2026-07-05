"""learn_structure_embedded: text fields enter structure discovery as embedded cluster codes."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_CHEAP = ["cheap tiny widget", "basic small widget", "budget mini part", "low cost simple piece"]
_PREM = ["premium deluxe gadget", "luxury pro device", "flagship elite gadget", "high end pro unit"]
_EXTRA = ["red", "blue", "green", "matte", "gloss", "new", "old", "big", "small"]


def _rows(n, dependent=True, seed=0):
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        prem = r.rand() < 0.5
        base = _PREM[r.randint(len(_PREM))] if prem else _CHEAP[r.randint(len(_CHEAP))]
        text = base + " " + _EXTRA[r.randint(len(_EXTRA))] + " " + _EXTRA[r.randint(len(_EXTRA))]
        price = float((100.0 if prem else 10.0) + 3.0 * r.randn()) if dependent else float(50.0 + 20.0 * r.randn())
        out.append((text, price))
    return out


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbeddedStructureTest(unittest.TestCase):
    def test_text_price_dependence_is_discovered_and_pays(self):
        from mixle.inference import learn_structure_embedded

        m = learn_structure_embedded(_rows(300, True), n_clusters=4, embed_dim=8, epochs=120, seed=0)
        self.assertTrue(any(0 in e for e in m.edges()))  # the text field participates in the graph
        fresh = _rows(80, True, seed=9)
        lls = m.seq_log_density(fresh)
        self.assertTrue(np.isfinite(lls).all())
        # cluster representatives make the discovered structure readable
        reps = m.describe()["text_fields"][0]
        self.assertEqual(len(reps), 4)
        self.assertTrue(all(isinstance(v, str) and v for v in reps.values()))

    def test_independent_text_finds_no_edge(self):
        from mixle.inference import learn_structure_embedded

        m = learn_structure_embedded(_rows(300, False, seed=1), n_clusters=4, embed_dim=8, epochs=120, seed=0)
        self.assertEqual(m.edges(), [])

    def test_auto_detection_requires_a_real_text_field(self):
        from mixle.inference import learn_structure_embedded

        with self.assertRaises(ValueError):
            learn_structure_embedded([("a", 1.0)] * 60)  # 'a' is a plain categorical, not free text

    def test_text_field_enters_as_a_vector_node(self):
        from mixle.inference import learn_structure_embedded

        m = learn_structure_embedded(_rows(200, True), n_clusters=3, embed_dim=8, epochs=80, seed=0)
        # the text field (0) is embedded to an 8-d vector in the record, driven as a multivariate node
        enc = m.encode_record(_rows(1, True, seed=7)[0])
        self.assertEqual(np.asarray(enc[0]).shape, (8,))  # a vector, not a cluster-code string
        # and the network fitted a vector-valued factor for it (multivariate marginal or CLG)
        vf = [f for f in m.net.factors if f.child == 0]
        self.assertTrue(vf and type(vf[0]).__name__ in ("_VectorMarginalFactor", "_VectorCLGFactor"))

    def test_single_record_encode_matches_batch(self):
        from mixle.inference import learn_structure_embedded

        m = learn_structure_embedded(_rows(120, True), n_clusters=3, embed_dim=8, epochs=80, seed=0)
        rows = _rows(20, True, seed=3)
        batch = m.encode_records(rows)
        for r, enc in zip(rows, batch):
            single = m.encode_record(r)  # record is (text, price): field 0 -> vector, field 1 -> unchanged
            # same embedding up to float32 batch-vs-single matmul noise
            np.testing.assert_allclose(np.asarray(single[0]), np.asarray(enc[0]), atol=1e-5)
            self.assertEqual(single[1], enc[1])  # the scalar price field is unchanged


if __name__ == "__main__":
    unittest.main()
