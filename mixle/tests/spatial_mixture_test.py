"""SpatialMixture: a Potts-coupled mixture over a grid with pluggable mixle emission distributions."""

import itertools
import unittest

import numpy as np

from mixle.analysis.spatial_mixture import SpatialMixture
from mixle.stats import GaussianEstimator, MultivariateGaussianDistribution, MultivariateGaussianEstimator
from mixle.stats.compute.pdist import DataSequenceEncoder, SequenceEncodableProbabilityDistribution


class _TaggedEncoder(DataSequenceEncoder):
    """Encodes each observation as ``(raw_value, tag)``; ``tag`` records which encoder instance ran,
    so scoring a component with a DIFFERENT component's encoder is directly observable in a test."""

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TaggedEncoder) and other.tag == self.tag

    def seq_encode(self, x):
        raw = np.asarray(x, dtype=float)
        return np.column_stack([raw, np.full(raw.shape[0], self.tag, dtype=float)])


class _TaggedDistribution(SequenceEncodableProbabilityDistribution):
    """A minimal pluggable "emission" that only scores sanely when the data it is handed was encoded
    by ITS OWN ``dist_to_encoder()``: scoring with a differently-tagged encoding (i.e. some other
    component's encoder) returns a large negative penalty instead of a real log-density, so misuse is
    directly observable rather than merely numerically off."""

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def log_density(self, x):
        raw, tag = x
        return -(raw**2) if tag == self.tag else -1e6

    def seq_log_density(self, x):
        raw, tag = x[:, 0], x[:, 1]
        return np.where(tag == self.tag, -(raw**2), -1e6)

    def dist_to_encoder(self):
        return _TaggedEncoder(self.tag)

    def sampler(self, seed=None):
        raise NotImplementedError

    def estimator(self, pseudo_count=None):
        raise NotImplementedError


class _OkEncoder(DataSequenceEncoder):
    """A trivial pass-through encoder, structurally unremarkable on its own."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OkEncoder)

    def seq_encode(self, x):
        return np.asarray(x, dtype=float)


class _WrongShapeDistribution(SequenceEncodableProbabilityDistribution):
    """A pluggable "emission" whose ``seq_log_density`` deliberately returns the wrong shape, to
    exercise the encoder-contract shape check independent of any encoder mix-up."""

    def __init__(self, out_shape: tuple[int, ...]) -> None:
        self.out_shape = out_shape

    def log_density(self, x):
        return 0.0

    def seq_log_density(self, x):
        return np.zeros(self.out_shape)

    def dist_to_encoder(self):
        return _OkEncoder()

    def sampler(self, seed=None):
        raise NotImplementedError

    def estimator(self, pseudo_count=None):
        raise NotImplementedError


def _layered_field(seed=0):
    rng = np.random.RandomState(seed)
    nx, ny = 40, 40
    true = np.zeros((nx, ny), dtype=int)
    true[14:27] = 1
    true[27:] = 2
    yy, xx = np.mgrid[0:nx, 0:ny]
    true[((xx - 30) ** 2 + (yy - 8) ** 2) < 25] = 2
    means = np.array([[2.0, 2.0], [3.0, 2.6], [4.0, 2.4]])
    obs = [rng.multivariate_normal(means[true[i, j]], 0.45 * np.eye(2)) for i in range(nx) for j in range(ny)]
    return (nx, ny), true, obs


def _best_accuracy(pred, truth, k=3):
    return max(
        np.mean(np.vectorize(dict(zip(range(k), perm)).get)(pred) == truth) for perm in itertools.permutations(range(k))
    )


class SpatialMixtureTest(unittest.TestCase):
    def setUp(self):
        self.shape, self.true, self.obs = _layered_field()

    def test_potts_coupling_beats_an_ordinary_mixture(self):
        spatial = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        plain = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=0.0).fit(self.obs, seed=1)
        self.assertGreater(_best_accuracy(spatial.labels(), self.true), 0.9)
        self.assertGreater(_best_accuracy(spatial.labels(), self.true), _best_accuracy(plain.labels(), self.true))

    def test_components_are_pysp_distributions(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        self.assertIsInstance(sm.component(0), MultivariateGaussianDistribution)
        np.testing.assert_allclose(sorted(sm.component(j).mu[0] for j in range(3)), [2.0, 3.0, 4.0], atol=0.3)

    def test_responsibilities_and_labels(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        q = sm.responsibilities()
        self.assertEqual(q.shape, (self.shape[0] * self.shape[1], 3))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-8)
        self.assertEqual(sm.labels().shape, self.shape)
        self.assertEqual(sm.entropy().shape, self.shape)

    def test_uncertainty_peaks_at_boundaries(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        ent = sm.entropy()
        self.assertLess(ent[2:8].mean(), ent[12:16].mean())  # confident interior, uncertain layer boundary

    def test_composes_with_a_different_emission_family(self):
        rng = np.random.RandomState(3)
        obs1 = [
            float(rng.normal([0.0, 5.0, 10.0][self.true.ravel()[i]], 0.6)) for i in range(self.shape[0] * self.shape[1])
        ]
        sm = SpatialMixture(self.shape, 3, GaussianEstimator(), beta=1.5).fit(obs1, seed=0)
        self.assertGreater(_best_accuracy(sm.labels(), self.true), 0.85)
        self.assertEqual(type(sm.component(0)).__name__, "GaussianDistribution")

    def test_three_dimensional_grid(self):
        rng = np.random.RandomState(2)
        true = np.zeros((8, 8, 8), dtype=int)
        true[4:] = 1
        obs = [np.array([rng.normal([0.0, 5.0][true.ravel()[i]], 0.4)]) for i in range(8**3)]
        sm = SpatialMixture((8, 8, 8), 2, MultivariateGaussianEstimator(), beta=1.5).fit(obs, seed=0)
        self.assertEqual(sm.labels().shape, (8, 8, 8))
        self.assertGreater(_best_accuracy(sm.labels().ravel(), true.ravel(), k=2), 0.9)


class SpatialMixtureInitializationTest(unittest.TestCase):
    """MXR-080-0115: shape/component-count/beta/iteration-control validation at construction and fit
    entry, plus initialization robustness -- an empty component must be repaired before it is ever
    re-estimated, and the repair itself must never cascade emptiness onto another component."""

    def test_rejects_more_components_than_cells(self):
        # pigeonhole: 6 components can never be filled by 4 cells, so at least 2 are permanently
        # empty regardless of the random draw -- this must be rejected at construction, not left to
        # fail unpredictably during fitting.
        with self.assertRaises(ValueError):
            SpatialMixture((2, 2), 6, GaussianEstimator(), beta=0.5)

    def test_rejects_zero_or_negative_components(self):
        for bad_k in (0, -1, -5):
            with self.assertRaises(ValueError):
                SpatialMixture((2, 2), bad_k, GaussianEstimator())

    def test_rejects_non_positive_shape_dimensions(self):
        for bad_shape in ((0, 5), (-1, 3), (3, -2), (0,)):
            with self.assertRaises(ValueError):
                SpatialMixture(bad_shape, 2, GaussianEstimator())

    def test_rejects_fractional_or_boolean_topology_and_component_counts(self):
        for bad_shape in ((1.9, 2), (True, 2), (2.0, 2)):
            with self.subTest(shape=bad_shape):
                with self.assertRaises(ValueError):
                    SpatialMixture(bad_shape, 2, GaussianEstimator())
        for bad_components in (1.9, True, 2.0):
            with self.subTest(n_components=bad_components):
                with self.assertRaises(ValueError):
                    SpatialMixture((2, 2), bad_components, GaussianEstimator())

    def test_rejects_grid_before_product_overflow_or_resource_allocation(self):
        with self.assertRaisesRegex(ValueError, "supported"):
            SpatialMixture((1_000_000, 2), 2, GaussianEstimator())

    def test_rejects_invalid_beta(self):
        for bad_beta in (-1.0, -0.001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                SpatialMixture((2, 2), 2, GaussianEstimator(), beta=bad_beta)

    def test_rejects_non_positive_iteration_controls(self):
        sm = SpatialMixture((2, 2), 2, GaussianEstimator())
        obs = [0.0, 1.0, 2.0, 3.0]
        for bad_max_iter in (0, -1):
            with self.assertRaises(ValueError):
                sm.fit(obs, max_iter=bad_max_iter)
        for bad_mf_iter in (0, -1):
            with self.assertRaises(ValueError):
                sm.fit(obs, mf_iter=bad_mf_iter)

    def test_rejects_fractional_or_boolean_iteration_controls(self):
        sm = SpatialMixture((2, 2), 2, GaussianEstimator())
        obs = [0.0, 1.0, 2.0, 3.0]
        for bad in (1.5, True, 2.0):
            with self.subTest(max_iter=bad):
                with self.assertRaisesRegex(ValueError, "max_iter"):
                    sm.fit(obs, max_iter=bad)
            with self.subTest(mf_iter=bad):
                with self.assertRaisesRegex(ValueError, "mf_iter"):
                    sm.fit(obs, mf_iter=bad)

    def test_fitted_responsibilities_are_valid_and_cannot_mutate_model_state(self):
        sm = SpatialMixture((2, 2), 2, GaussianEstimator(), beta=0.5)
        sm.fit([0.0, 0.1, 5.0, 5.1], seed=0, max_iter=2)
        responsibilities = sm.responsibilities()
        self.assertFalse(responsibilities.flags.writeable)
        with self.assertRaises(ValueError):
            responsibilities[0, 0] = np.nan
        np.testing.assert_allclose(sm.responsibilities().sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(sm.entropy()).all())

    def test_published_summaries_revalidate_internal_fit_state(self):
        sm = SpatialMixture((2, 2), 2, GaussianEstimator(), beta=0.5)
        sm.fit([0.0, 0.1, 5.0, 5.1], seed=0, max_iter=2)
        sm._q = np.full((4, 2), np.nan)
        with self.assertRaisesRegex(ValueError, "responsibilities"):
            sm.responsibilities()
        with self.assertRaisesRegex(ValueError, "responsibilities"):
            sm.labels()
        with self.assertRaisesRegex(ValueError, "responsibilities"):
            sm.entropy()

    def test_initial_partition_is_repaired_before_the_first_reestimate(self):
        # seed=1 at this shape/k is known to draw an initial random partition with an empty
        # component (component 2 gets zero of the 9 cells); the FIRST _reestimate call must already
        # see a repaired (nonempty) partition, never the raw empty one.
        shape, k = (3, 3), 5
        raw = np.random.RandomState(1).randint(k, size=9)
        self.assertTrue((np.bincount(raw, minlength=k) == 0).any())  # sanity: the raw draw IS empty

        calls = []
        sm = SpatialMixture(shape, k, GaussianEstimator(), beta=0.3)
        real_reestimate = sm._reestimate

        def spy(acc_enc, q, current=None):
            calls.append(q.sum(axis=0).copy())
            return real_reestimate(acc_enc, q, current)

        sm._reestimate = spy
        sm.fit([float(i) for i in range(9)], seed=1, max_iter=2)
        self.assertGreaterEqual(len(calls), 1)
        self.assertFalse((calls[0] == 0).any(), "the first _reestimate call saw an empty component")

    def test_repair_does_not_cascade_emptiness_onto_a_singleton(self):
        # component 0 holds 5 of 6 cells, component 1 holds exactly 1 (a singleton, NOT empty),
        # components 2 and 3 are empty -- a naive reseed that ignores donor size can steal component
        # 1's only cell while filling 2/3, creating a new empty component instead of just fixing one.
        sm = SpatialMixture((3, 2), 4, GaussianEstimator(), beta=0.5)
        lab = np.array([0, 0, 0, 0, 1, 0])
        counts_before = np.bincount(lab, minlength=4)
        self.assertEqual(counts_before.tolist(), [5, 1, 0, 0])

        repaired = sm._repair_empty_components(lab)
        counts_after = np.bincount(repaired, minlength=4)
        self.assertTrue((counts_after > 0).all(), f"a component is still empty: {counts_after}")
        self.assertGreaterEqual(counts_after[1], 1, "the pre-existing singleton was emptied by the repair")

    def test_repair_handles_many_simultaneous_empty_components(self):
        sm = SpatialMixture((21,), 10, GaussianEstimator(), beta=0.0)
        lab = np.array([0] * 20 + [1])  # component 1 is a singleton; components 2..9 are all empty
        repaired = sm._repair_empty_components(lab)
        counts = np.bincount(repaired, minlength=10)
        self.assertTrue((counts > 0).all(), f"some component is still empty: {counts}")

    def test_repair_is_deterministic(self):
        sm = SpatialMixture((3, 2), 4, GaussianEstimator(), beta=0.5)
        lab = np.array([0, 0, 0, 0, 1, 0])
        first = sm._repair_empty_components(lab)
        for _ in range(10):
            self.assertTrue(np.array_equal(sm._repair_empty_components(lab), first))

    def test_negative_control_well_posed_fit_still_initializes_and_iterates(self):
        # a normal k <= n_cells configuration must still fit cleanly end to end.
        shape, k = (6, 6), 3
        rng = np.random.RandomState(4)
        true = np.zeros((6, 6), dtype=int)
        true[2:4] = 1
        true[4:] = 2
        obs = [float(rng.normal([0.0, 5.0, 10.0][true.ravel()[i]], 0.3)) for i in range(36)]
        sm = SpatialMixture(shape, k, GaussianEstimator(), beta=1.0).fit(obs, seed=0)
        q = sm.responsibilities()
        self.assertEqual(q.shape, (36, k))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-8)
        counts = np.bincount(sm.labels().ravel(), minlength=k)
        self.assertTrue((counts > 0).all(), "a well-posed fit ended up with an empty component")


class SpatialMixtureContractTest(unittest.TestCase):
    """MXR-080-0116: observation count must match prod(shape), and every component must be scored
    with its OWN fitted encoder -- never a different component's -- while still honoring a shared,
    structurally compatible encoder contract (one log-density value per cell)."""

    def test_rejects_observation_count_mismatch(self):
        # a plain assertRaises(ValueError) would also pass against the pre-fix code for most of
        # these -- a mismatched length already crashed *some* mismatches downstream inside the M-step
        # (an incidental "shapes (5,) and (16,) not aligned" from np.dot deep in the accumulator), just
        # not with a clear boundary error naming the expected count, and not for every mismatch. Assert
        # the actual validation message so this only passes against the real fix.
        sm = SpatialMixture((4, 4), 2, GaussianEstimator(), beta=0.5)  # 16 cells
        for bad_n in (0, 1, 5, 15, 17, 20):
            with self.assertRaisesRegex(ValueError, r"prod\(shape\)=16"):
                sm.fit([float(i) for i in range(bad_n)], seed=0, max_iter=2)

    def test_negative_control_matching_observation_count_fits(self):
        sm = SpatialMixture((4, 4), 2, GaussianEstimator(), beta=0.5)
        sm.fit([float(i % 2) * 10.0 for i in range(16)], seed=0, max_iter=3)
        self.assertEqual(sm.responsibilities().shape, (16, 2))

    def test_each_component_is_scored_with_its_own_encoder(self):
        # tags 0, 1, 2 stand in for three components whose fitted encoders genuinely differ (a
        # legitimate scenario the class's pluggable-emission design explicitly allows) but are
        # structurally compatible (same encoded shape/dtype). Scoring with the WRONG encoder is a
        # huge penalty (see _TaggedDistribution), making misuse of a shared/hoisted encoder observable.
        sm = SpatialMixture((2, 2), 3, GaussianEstimator(), beta=0.0)
        sm.components = [_TaggedDistribution(tag=j) for j in range(3)]
        data = [0.0, 1.0, 2.0, 3.0]

        emis = sm._emission_loglik(data)

        self.assertEqual(emis.shape, (4, 3))
        for j in range(3):
            self.assertTrue(np.all(emis[:, j] > -1e5), f"component {j} was not scored with its own encoder")
            np.testing.assert_allclose(emis[:, j], -(np.arange(4.0) ** 2))

    def test_rejects_structurally_incompatible_component_output(self):
        # component 0 is a normal (correctly-shaped) pluggable emission; component 1 deliberately
        # returns a (4, 2) log-density array instead of (4,) -- the shared responsibilities/neighbor
        # bookkeeping cannot use that, and it must be rejected with a clear error, not an incidental
        # column_stack crash or a silent shape mismatch.
        sm = SpatialMixture((2, 2), 2, GaussianEstimator(), beta=0.0)
        sm.components = [_TaggedDistribution(tag=0), _WrongShapeDistribution(out_shape=(4, 2))]
        with self.assertRaises(ValueError):
            sm._emission_loglik([0.0, 1.0, 2.0, 3.0])

    def test_negative_control_genuinely_compatible_pluggable_emissions_score_correctly(self):
        # real (non-stub) emission family, real fit -- per-component encoder use must stay correct.
        rng = np.random.RandomState(5)
        shape, k = (5, 5), 2
        true = np.zeros((5, 5), dtype=int)
        true[3:] = 1
        obs = [float(rng.normal([0.0, 8.0][true.ravel()[i]], 0.3)) for i in range(25)]
        sm = SpatialMixture(shape, k, GaussianEstimator(), beta=1.0).fit(obs, seed=1)

        emis = sm._emission_loglik(obs)

        self.assertEqual(emis.shape, (25, k))
        self.assertTrue(np.isfinite(emis).all())
        # identify components by fitted mean (EM label order is arbitrary) and confirm the
        # near-zero cell scores higher under the near-zero component than under the near-eight one.
        low_component = int(np.argmin([sm.component(j).mu for j in range(k)]))
        high_component = 1 - low_component
        low_cell_idx = 0  # shape.ravel()[0] has true label 0, i.e. obs drawn near 0.0
        self.assertGreater(emis[low_cell_idx, low_component], emis[low_cell_idx, high_component])


if __name__ == "__main__":
    unittest.main()
