"""Streaming HViS (mixle.utils.hvis.stream): frozen-atlas placement, drift accounting, aligned refresh.

Fixture: a KNOWN two-component 1-D Gaussian mixture (no fitting -- the affinity machinery only needs
a model, and constructing it directly keeps the tests fast and fully deterministic). The load-bearing
claims: placement puts arriving points with their own cluster's landmarks; the atlas never moves
during streaming; drift trips on a genuinely shifted stream and not on an in-distribution one; a
refresh preserves visual continuity measurably (small Procrustes residual), not by assumption.
"""

import unittest

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution
from mixle.utils.hvis import StreamingHvis, place_in_atlas

_MODEL = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(8.0, 1.0)], [0.5, 0.5])


def _draw(n_per_cluster, rng):
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per_cluster), rng.normal(8.0, 1.0, n_per_cluster)])
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)
    order = rng.permutation(len(xs))
    return [float(v) for v in xs[order]], labels[order]


def _make_stream(seed=0, n_landmarks_per_cluster=30, **kwargs):
    rng = np.random.RandomState(seed)
    landmarks, labels = _draw(n_landmarks_per_cluster, rng)
    stream = StreamingHvis(_MODEL, landmarks, perplexity=10.0, seed=seed, max_its=300, **kwargs)
    return stream, labels, rng


class PlacementTest(unittest.TestCase):
    def test_add_places_points_with_their_own_cluster(self):
        stream, landmark_labels, rng = _make_stream(seed=0)
        batch, batch_labels = _draw(20, rng)
        coords = stream.add(batch)
        self.assertEqual(coords.shape, (40, 2))

        # each placed point's nearest LANDMARK must come from the same true cluster
        d2 = np.square(coords[:, None, :] - stream.atlas[None, :, :]).sum(axis=2)
        nearest = d2.argmin(axis=1)
        agreement = float(np.mean(landmark_labels[nearest] == batch_labels))
        self.assertGreater(agreement, 0.9)

    def test_add_never_moves_the_atlas(self):
        stream, _, rng = _make_stream(seed=1)
        before = stream.atlas.copy()
        batch, _ = _draw(15, rng)
        stream.add(batch)
        np.testing.assert_array_equal(stream.atlas, before)  # bit-identical: stability is structural

    def test_determinism_given_seed(self):
        stream_a, _, rng_a = _make_stream(seed=2)
        stream_b, _, rng_b = _make_stream(seed=2)
        batch_a, _ = _draw(10, rng_a)
        batch_b, _ = _draw(10, rng_b)
        np.testing.assert_array_equal(stream_a.add(batch_a), stream_b.add(batch_b))

    def test_empty_batch(self):
        stream, _, _ = _make_stream(seed=3)
        self.assertEqual(stream.add([]).shape, (0, 2))

    def test_place_in_atlas_one_hot_row_converges_onto_that_landmark(self):
        atlas = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        p = np.array([[0.0, 1.0, 0.0]])
        y = place_in_atlas(p, atlas, max_its=500)
        self.assertLess(float(np.linalg.norm(y[0] - atlas[1])), 1.0)


class DriftTest(unittest.TestCase):
    def test_in_distribution_stream_does_not_trip(self):
        stream, _, rng = _make_stream(seed=4)
        batch, _ = _draw(25, rng)
        stream.add(batch)
        self.assertLess(stream.drift_score(), stream.drift_threshold_nats)
        self.assertFalse(stream.drifted)

    def test_shifted_stream_trips(self):
        stream, _, rng = _make_stream(seed=5)
        shifted = [float(v) for v in rng.normal(30.0, 1.0, 30)]  # far outside both components
        stream.add(shifted)
        self.assertTrue(stream.drifted)
        self.assertGreater(stream.drift_score(), stream.drift_threshold_nats)

    def test_refresh_resets_the_drift_accumulator(self):
        stream, _, rng = _make_stream(seed=6)
        stream.add([float(v) for v in rng.normal(30.0, 1.0, 20)])
        self.assertTrue(stream.drifted)
        stream.refresh()
        self.assertFalse(stream.drifted)
        self.assertEqual(stream.drift_score(), 0.0)


class RefreshAndGrowthTest(unittest.TestCase):
    def test_refresh_on_unchanged_data_keeps_continuity_measurably(self):
        stream, _, _ = _make_stream(seed=7)
        # first refresh may legitimately move things: it CONTINUES the optimization of a
        # not-yet-converged atlas (that is real geometry refinement, not misalignment).
        first = stream.refresh()
        self.assertIn("alignment_scale", first)
        # at steady state (converged atlas, unchanged data/model) a refresh must be ~identity up to
        # rigid motion + uniform scale -- continuity is this measured claim, not an assumption.
        second = stream.refresh()
        self.assertLess(second["alignment_residual_rms"], 0.25 * second["atlas_spread"])
        self.assertEqual(second["n_landmarks"], 60)

    def test_extend_landmarks_grows_the_atlas_without_moving_it(self):
        stream, _, rng = _make_stream(seed=8)
        before = stream.atlas.copy()
        batch, _ = _draw(5, rng)
        stream.extend_landmarks(batch)
        self.assertEqual(stream.atlas.shape, (70, 2))
        np.testing.assert_array_equal(stream.atlas[:60], before)
        self.assertEqual(len(stream.landmark_data), 70)

    def test_bad_atlas_shape_raises(self):
        rng = np.random.RandomState(9)
        landmarks, _ = _draw(10, rng)
        with self.assertRaises(ValueError):
            StreamingHvis(_MODEL, landmarks, atlas=np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
