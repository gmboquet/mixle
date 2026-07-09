"""Embedding goals (mixle.utils.hvis.goals) + the internal UMAP engine (mixle.utils.hvis.umap_np).

Same deterministic known-model fixture as hvis_stream_test: a two-component 1-D Gaussian mixture --
the affinity machinery only needs a model, so no fitting. The load-bearing claims: hard anchors are
EXACT (projection, not a strong suggestion); partial labels measurably tighten labeled groups
without dictating unlabeled points; AxisAlign makes a chosen scalar actually run along the chosen
axis; and the internal UMAP produces a usable layout (cluster separation) with zero optional
dependencies, honoring goals that umap-learn structurally cannot.
"""

import sys
import unittest
from unittest import mock

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution
from mixle.utils.hvis import Anchor, AxisAlign, LabelCohesion, htsne, humap

_MODEL = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(8.0, 1.0)], [0.5, 0.5])


def _data(n_per_cluster=25, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per_cluster), rng.normal(8.0, 1.0, n_per_cluster)])
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)
    return [float(v) for v in xs], labels


def _cluster_separation(coords, labels):
    """Mean nearest-neighbor label agreement -- the layout keeps clusters coherent."""
    d2 = np.square(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.mean(labels[d2.argmin(axis=1)] == labels))


class AnchorTest(unittest.TestCase):
    def test_hard_anchors_are_exact_on_the_exact_engine(self):
        data, labels = _data(seed=0)
        pins = np.array([[-5.0, -5.0], [5.0, 5.0]])
        anchor = Anchor([0, 25], pins)  # one point from each cluster
        coords = htsne(data, mix_model=_MODEL, method="exact", seed=0, max_its=300, goals=[anchor], out=None)
        np.testing.assert_array_equal(coords[[0, 25]], pins)  # projection: exact, not approximately
        self.assertGreater(_cluster_separation(coords, labels), 0.9)

    def test_hard_anchors_are_exact_on_the_barnes_hut_engine(self):
        data, _ = _data(seed=1)
        pins = np.array([[-3.0, 0.0], [3.0, 0.0]])
        anchor = Anchor([0, 25], pins)
        coords = htsne(data, mix_model=_MODEL, method="barnes_hut", seed=1, max_its=300, goals=[anchor], out=None)
        np.testing.assert_array_equal(coords[[0, 25]], pins)

    def test_soft_anchor_equilibrium_gap_shrinks_with_the_rate(self):
        # a soft anchor reaches equilibrium where the data force balances the relaxation pull, so
        # (a) the point sits near the pin but NOT exactly on it, and (b) a faster rate sits closer.
        # Gaps are scale-normalized by each layout's own spread: separate t-SNE runs settle at
        # different spreads, so RAW cross-run distances are trajectory noise (comparing them flaked
        # across BLAS builds). The rate contrast is wide (0.05 vs 0.8; a ~50x normalized-gap
        # separation locally) so no build inverts it.
        data, _ = _data(seed=2)
        pin = np.array([[4.0, 4.0]])

        def gap_at(weight):
            coords = htsne(
                data,
                mix_model=_MODEL,
                method="exact",
                seed=2,
                max_its=300,
                goals=[Anchor([0], pin, weight=weight)],
                out=None,
            )
            return float(np.linalg.norm(coords[0] - pin[0]) / coords.std())

        gap_slow = gap_at(0.05)
        gap_fast = gap_at(0.8)
        self.assertGreater(gap_fast, 0.0)  # soft: never an exact projection
        self.assertLess(gap_fast, 0.25)  # but the pull is real: well inside the layout scale
        self.assertLess(gap_fast, 0.25 * gap_slow)  # and monotone in the rate, with cross-build margin

    def test_anchor_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Anchor([0, 1], np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            Anchor([0], np.zeros((1, 2)), weight=-1.0)
        with self.assertRaises(ValueError):
            Anchor([0], np.zeros((1, 2)), weight=2.0)  # rate semantics: weights live in (0, 1]


class LabelCohesionTest(unittest.TestCase):
    def test_partial_labels_tighten_labeled_groups(self):
        data, labels = _data(seed=3)
        # PARTIAL labeling: only 40% of points labeled, the rest None
        rng = np.random.RandomState(3)
        partial = [int(lab) if rng.rand() < 0.4 else None for lab in labels]
        goal = LabelCohesion(partial, weight=0.3)

        coords_free = htsne(data, mix_model=_MODEL, method="exact", seed=3, max_its=300, out=None)
        coords_goal = htsne(data, mix_model=_MODEL, method="exact", seed=3, max_its=300, goals=[goal], out=None)

        def within_spread(coords):
            spread = 0.0
            for lab in (0, 1):
                idx = [i for i, p in enumerate(partial) if p == lab]
                spread += float(np.linalg.norm(coords[idx] - coords[idx].mean(axis=0, keepdims=True), axis=1).mean())
            return spread / (2.0 * float(np.abs(coords).std()))  # scale-normalized: t-SNE spreads differ

        self.assertLess(within_spread(coords_goal), within_spread(coords_free))
        # semi-supervision, not relabeling: unlabeled points still sit with their true cluster
        self.assertGreater(_cluster_separation(coords_goal, labels), 0.9)

    def test_margin_pushes_centroids_apart(self):
        data, labels = _data(seed=4)
        margin = 40.0
        goal = LabelCohesion([int(lab) for lab in labels], weight=0.3, margin=margin)
        coords = htsne(data, mix_model=_MODEL, method="exact", seed=4, max_its=400, goals=[goal], out=None)
        c0 = coords[labels == 0].mean(axis=0)
        c1 = coords[labels == 1].mean(axis=0)
        self.assertGreater(float(np.linalg.norm(c0 - c1)), 0.8 * margin)

    def test_all_unlabeled_raises(self):
        with self.assertRaises(ValueError):
            LabelCohesion([None, None, None])


class AxisAlignTest(unittest.TestCase):
    def test_values_run_along_the_chosen_axis(self):
        data, _ = _data(seed=5)
        goal = AxisAlign(data, axis=0, weight=0.5)  # the raw 1-D value should order embedding axis 0
        coords = htsne(data, mix_model=_MODEL, method="exact", seed=5, max_its=400, goals=[goal], out=None)
        r = float(np.corrcoef(coords[:, 0], np.asarray(data))[0, 1])
        # without the goal this seed lands at r=-0.79 (sign is arbitrary); the goal forces positive
        # alignment at 0.75-0.98 depending on the BLAS/numpy build, so pin the sign with margin
        self.assertGreater(r, 0.6)

    def test_constant_values_raise(self):
        with self.assertRaises(ValueError):
            AxisAlign([1.0, 1.0, 1.0])


class InternalUmapTest(unittest.TestCase):
    def test_internal_engine_separates_clusters(self):
        data, labels = _data(seed=6)
        coords = humap(data, mix_model=_MODEL, engine="internal", seed=6, n_epochs=150, out=None)
        self.assertEqual(coords.shape, (50, 2))
        self.assertGreater(_cluster_separation(coords, labels), 0.9)

    def test_internal_engine_is_deterministic_given_seed(self):
        data, _ = _data(seed=7)
        a = humap(data, mix_model=_MODEL, engine="internal", seed=7, n_epochs=100, out=None)
        b = humap(data, mix_model=_MODEL, engine="internal", seed=7, n_epochs=100, out=None)
        np.testing.assert_array_equal(a, b)

    def test_auto_engine_with_goals_uses_internal_and_honors_hard_anchors(self):
        data, _ = _data(seed=8)
        pins = np.array([[-9.0, 0.0], [9.0, 0.0]])
        coords = humap(
            data, mix_model=_MODEL, engine="auto", seed=8, n_epochs=100, goals=[Anchor([0, 25], pins)], out=None
        )
        # umap-learn structurally cannot pin points, so exact pins prove the internal engine ran
        np.testing.assert_array_equal(coords[[0, 25]], pins)

    def test_umap_learn_engine_with_goals_refuses_rather_than_dropping_them(self):
        data, _ = _data(seed=9)
        with self.assertRaises(ValueError):
            humap(data, mix_model=_MODEL, engine="umap-learn", goals=[Anchor([0], [[0.0, 0.0]])], out=None)

    def test_auto_engine_falls_back_to_internal_when_umap_learn_is_missing(self):
        data, labels = _data(seed=10)
        with mock.patch.dict(sys.modules, {"umap": None}):  # import umap -> ImportError
            coords = humap(data, mix_model=_MODEL, engine="auto", seed=10, n_epochs=100, out=None)
        self.assertEqual(coords.shape, (50, 2))
        self.assertGreater(_cluster_separation(coords, labels), 0.85)

    def test_bad_engine_raises(self):
        data, _ = _data(seed=11)
        with self.assertRaises(ValueError):
            humap(data, mix_model=_MODEL, engine="not-an-engine", out=None)


if __name__ == "__main__":
    unittest.main()
