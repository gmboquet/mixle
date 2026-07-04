"""synthesize() (F2): a dataset factory -- sample, label, keep only what verifies."""

import unittest

from mixle.inference import synthesize
from mixle.inference.synthesize import Dataset


def _draw(rng):
    return int(rng.randint(0, 100))


class RejectionSamplingTest(unittest.TestCase):
    def test_verify_gates_every_row(self):
        ds = synthesize(_draw, verify=lambda x: x % 2 == 0, n=30, seed=1)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 30)
        self.assertTrue(all(x % 2 == 0 for x in ds.inputs))  # only accepted rows survive
        self.assertGreater(ds.n_rejected, 0)  # odds were rejected
        self.assertLess(ds.acceptance_rate, 1.0)

    def test_labels_and_pairs(self):
        ds = synthesize(_draw, label=lambda x: x * x, verify=lambda x, y: y < 2500, n=20, seed=2)
        for x, y in ds.pairs():
            self.assertEqual(y, x * x)
            self.assertLess(y, 2500)

    def test_recheck_audits_the_shipped_verifier(self):
        ds = synthesize(_draw, verify=lambda x: x >= 10, n=15, seed=3)
        self.assertTrue(ds.recheck())  # every shipped row re-verifies

    def test_no_verifier_accepts_everything(self):
        ds = synthesize(_draw, n=25, seed=0)
        self.assertEqual(len(ds), 25)
        self.assertEqual(ds.acceptance_rate, 1.0)
        self.assertEqual(ds.n_rejected, 0)
        self.assertTrue(ds.recheck())  # vacuous, no verifier

    def test_impossible_verifier_stops_at_max_tries(self):
        ds = synthesize(_draw, verify=lambda x: False, n=10, max_tries=40, seed=0)
        self.assertEqual(len(ds), 0)  # nothing passes
        self.assertLessEqual(ds.provenance["tried"], 40)  # bounded, no infinite loop


class SourcesTest(unittest.TestCase):
    def test_callable_source_no_rng_arg(self):
        seq = iter(range(100))
        ds = synthesize(lambda: next(seq), n=5, seed=0)
        self.assertEqual(ds.inputs, [0, 1, 2, 3, 4])

    def test_real_inputs_infer_a_generator(self):
        reals = [(["free", "pro"][i % 2], float(20 + 80 * (i % 2))) for i in range(60)]
        ds = synthesize(reals, n=12, seed=0)
        self.assertEqual(len(ds), 12)
        self.assertTrue(all(len(x) == 2 for x in ds.inputs))  # same record shape

    def test_unlabeled_pairs_raises(self):
        ds = synthesize(_draw, n=5, seed=0)
        with self.assertRaises(ValueError):
            ds.pairs()


if __name__ == "__main__":
    unittest.main()
