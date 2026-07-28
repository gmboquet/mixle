"""synthesize() (F2): a dataset factory -- sample, label, keep only what verifies."""

import unittest

from mixle.inference import synthesize
from mixle.inference.synthesize import Dataset, IncompleteSynthesisError


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
        # MXR-080-1624: bounded (no infinite loop) AND an explicit failure -- an exhausted budget must not
        # come back as an ordinary Dataset that reads like the request was filled.
        with self.assertRaises(IncompleteSynthesisError) as ctx:
            synthesize(_draw, verify=lambda x: False, n=10, max_tries=40, seed=0)
        partial = ctx.exception.dataset
        self.assertEqual(len(partial), 0)  # nothing passes
        self.assertLessEqual(partial.provenance["tried"], 40)  # bounded, no infinite loop
        self.assertEqual(partial.provenance["requested"], 10)


class RequestedCountContractTest(unittest.TestCase):
    """MXR-080-1624: a returned Dataset holds exactly the requested rows, and the request is an exact count."""

    def test_short_sampler_raises_instead_of_returning_a_shortfall(self):
        seq = iter(range(3))

        def short():
            return next(seq, None)

        with self.assertRaises(IncompleteSynthesisError) as ctx:
            synthesize(short, verify=lambda x: x is not None, n=8, max_tries=12, seed=0)
        self.assertEqual(len(ctx.exception.dataset), 3)
        self.assertIn("3 of the 8 requested", str(ctx.exception))

    def test_negative_count_is_rejected_rather_than_recorded_as_the_request(self):
        with self.assertRaises(ValueError):
            synthesize(_draw, n=-2, seed=0)

    def test_fractional_count_is_rejected(self):
        with self.assertRaises(TypeError):
            synthesize(_draw, n=2.9, seed=0)

    def test_boolean_is_not_a_count(self):
        with self.assertRaises(TypeError):
            synthesize(_draw, n=True, seed=0)

    def test_zero_rows_is_an_exact_and_complete_request(self):
        ds = synthesize(_draw, n=0, seed=0)
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.provenance["requested"], 0)


class DatasetAlignmentTest(unittest.TestCase):
    """MXR-080-1625: a labeled dataset carries one label per input, checked before every row walk."""

    def test_construction_rejects_missing_labels(self):
        with self.assertRaisesRegex(ValueError, "3 inputs but 1 labels"):
            Dataset(inputs=[1, 2, 3], labels=[9])

    def test_recheck_cannot_pass_while_rows_are_skipped(self):
        ds = Dataset(inputs=[1, 2, 3], labels=[9, 9, 9], verify=lambda x, y: x == 1)
        ds.labels.pop()  # post-construction edit leaves input 3 with no label
        with self.assertRaises(ValueError):
            ds.recheck()

    def test_pairs_and_iteration_refuse_a_misaligned_dataset(self):
        ds = Dataset(inputs=[1, 2, 3], labels=[9, 9, 9])
        ds.inputs.append(4)
        with self.assertRaises(ValueError):
            ds.pairs()
        with self.assertRaises(ValueError):
            list(ds)

    def test_aligned_dataset_still_iterates_and_pairs(self):
        ds = Dataset(inputs=[1, 2], labels=[9, 8], verify=lambda x, y: y > x)
        self.assertEqual(ds.pairs(), [(1, 9), (2, 8)])
        self.assertEqual(list(ds), [(1, 9), (2, 8)])
        self.assertTrue(ds.recheck())

    def test_unlabeled_dataset_is_vacuously_aligned(self):
        ds = Dataset(inputs=[1, 2, 3])
        self.assertEqual(list(ds), [1, 2, 3])
        self.assertTrue(ds.recheck())


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
