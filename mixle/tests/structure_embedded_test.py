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

    def test_single_record_encode_matches_batch(self):
        from mixle.inference import learn_structure_embedded

        m = learn_structure_embedded(_rows(120, True), n_clusters=3, embed_dim=8, epochs=80, seed=0)
        rows = _rows(20, True, seed=3)
        batch = m.encode_records(rows)
        for r, enc in zip(rows, batch):
            self.assertEqual(m.encode_record(r), enc)


if __name__ == "__main__":
    unittest.main()
