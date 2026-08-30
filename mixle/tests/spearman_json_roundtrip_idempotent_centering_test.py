"""Spearman: the identical idempotent-centering defect fixed on Thurstone, pinned here too.

``SpearmanRankingDistribution.__init__`` canonicalizes its ``sigma`` argument through
``_validate_location``: ``sigma = raw - raw.mean() + (dim - 1) / 2.0`` removes the common-shift
non-identifiability before checking the permutahedron constraints. ``__pysp_getstate__`` serializes
that already-canonicalized ``sigma`` verbatim, and ``__pysp_setstate__`` used to reconstruct via
``self.__init__(state['sigma'], ...)`` -- feeding the already-canonicalized array back through the
SAME shift. That shift is not exactly idempotent in float64 (see
``thurstone_json_roundtrip_idempotent_centering_test.py`` for the identical mechanism on
``ThurstoneDistribution.mu``), so a restored ``sigma`` could differ from the original by 1-few ULP --
invisible to ``log_density``/sampling, but changing the raw bytes
:func:`mixle.data.hashing.model_hash` fingerprints, which could trip ``Model.load``'s "integrity
note" ``UserWarning`` (mixle/lifecycle.py) on a same-process, zero-tampering round trip.

Unlike Thurstone, this does not currently manifest through ``SpearmanRankingEstimator.estimate()``:
its ``sigma`` there is always ``argsort(argsort(rank_sum))`` -- an EXACT integer permutation of
``0..dim-1`` -- and canonicalizing an exact integer permutation is bit-exact idempotent (every
intermediate value is a small integer or half-integer, all exactly representable in float64, so
``raw - raw.mean() + (dim - 1) / 2.0`` reproduces ``raw`` exactly). ``EstimatorPathNeverManifestsTest``
below pins that this stays true. The class explicitly supports a second construction path where it
does NOT hold, though: the module docstring documents ``sigma`` as "a (possibly fractional) mean
rank vector", and direct construction with one -- e.g. a consensus mean-rank vector averaged from
several judges' rankings, a wholly ordinary way to build a prototype distribution -- is exactly the
case this file's main sweeps exercise. The fix is the same shape as Thurstone's: an internal,
keyword-only ``_sigma_already_centered`` flag on ``__init__``/``_validate_location``, used only by
``__pysp_setstate__``, skips the shift on restore while leaving ordinary construction (and the
permutahedron validity check, which only reads ``sigma``) untouched.

Fixing this is defensive rather than a live false-positive: nothing in the estimated path exercises
it today. It is fixed anyway because it is the identical latent defect Thurstone had, and a future
change to how ``estimate()`` derives ``sigma`` (e.g. a tie-breaking or smoothing scheme that no
longer lands on an exact integer permutation) could make it manifest silently, exactly as it did for
Thurstone this release.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

from mixle.data.hashing import model_hash
from mixle.lifecycle import Model
from mixle.stats.rankings.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator

_DIMS = range(2, 9)
_SEEDS = range(15)
_RHOS = (0.3, 0.8, 2.0)

_DEPLOY_DIMS = range(2, 8)
_DEPLOY_SEEDS = range(8)


def _fractional_mean_rank_sigma(dim: int, seed: int, n_judges: int = 7) -> np.ndarray:
    """A realistic fractional mean-rank vector: the elementwise average rank across several
    independent judges' full rankings -- the direct-construction use case the module docstring
    documents ("a (possibly fractional) mean rank vector"). Guaranteed to lie in the permutahedron
    (no try/except needed here) because it is literally a convex combination of permutations of
    ``0..dim-1``, and the permutahedron is defined as their convex hull.
    """
    rng = np.random.RandomState(seed)
    rank_vectors = [np.argsort(rng.permutation(dim)) for _ in range(n_judges)]
    return np.mean(rank_vectors, axis=0)


class SpearmanPureGetstateSetstateIdempotentCenteringTest(unittest.TestCase):
    """Purest isolation: ``__pysp_getstate__`` -> ``__pysp_setstate__`` in memory, zero JSON/file
    machinery, matching the isolation used for the Thurstone finding."""

    def test_round_trip_is_bit_identical_across_a_seeded_sweep(self):
        checked = 0
        for dim in _DIMS:
            for seed in _SEEDS:
                for rho in _RHOS:
                    with self.subTest(dim=dim, seed=seed, rho=rho):
                        sigma = _fractional_mean_rank_sigma(dim, seed)
                        dist = SpearmanRankingDistribution(sigma, rho=rho)
                        state = dist.__pysp_getstate__()
                        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                        restored.__pysp_setstate__(state)
                        self.assertTrue(
                            np.array_equal(dist.sigma, restored.sigma),
                            f"sigma drifted by {restored.sigma - dist.sigma!r} across a pure "
                            f"getstate/setstate round trip (dim={dim!r} seed={seed!r} rho={rho!r})",
                        )
                        self.assertEqual(model_hash(dist), model_hash(restored))
                        self.assertTrue(np.array_equal(dist.log_weights, restored.log_weights))
                        self.assertEqual(dist.log_const, restored.log_const)
                        checked += 1
        self.assertGreater(checked, 100)

    def test_fresh_construction_still_canonicalizes_on_first_construction(self):
        # The fix must not touch ordinary construction: only __pysp_setstate__'s internal
        # _sigma_already_centered=True call may skip the shift.
        raw = np.array([5.0, 6.0, 7.0, 8.0])  # a shifted permutation: valid, with a real shift to undo
        dim = len(raw)
        dist = SpearmanRankingDistribution(raw, rho=0.5)
        self.assertTrue(np.array_equal(dist.sigma, raw - raw.mean() + (dim - 1) / 2.0))

    def test_sigma_stays_read_only_on_both_construction_paths(self):
        dist = SpearmanRankingDistribution(np.array([0.0, 1.0, 2.0]), rho=0.5)
        with self.assertRaises(ValueError):
            dist._sigma[0] = 0.0
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(dist.__pysp_getstate__())
        with self.assertRaises(ValueError):
            restored._sigma[0] = 0.0


class SpearmanDeployLoadIdempotentCenteringTest(unittest.TestCase):
    """Same defect, exercised through the real artifact path (see mixle/lifecycle.py's
    "integrity note" check, and ``ManifestSwapDisclosureTest`` in campaign2_lifecycle_test.py)."""

    def test_deploy_then_load_preserves_sigma_and_model_hash_across_a_seeded_sweep(self):
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for dim in _DEPLOY_DIMS:
                for seed in _DEPLOY_SEEDS:
                    for rho in _RHOS:
                        with self.subTest(dim=dim, seed=seed, rho=rho):
                            sigma = _fractional_mean_rank_sigma(dim, seed)
                            dist = SpearmanRankingDistribution(sigma, rho=rho)
                            # Same construction Model.load() itself uses to wrap an already-fitted
                            # distribution -- exercises the real deploy()/load() path directly.
                            model = Model(dist)
                            model.fitted = dist
                            out = os.path.join(tmp, f"spearman-{dim}-{seed}-{rho}")
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model.deploy(out)
                                loaded = Model.load(out)
                            self.assertTrue(
                                np.array_equal(dist.sigma, loaded.fitted.sigma),
                                f"sigma drifted across deploy()/load() (dim={dim!r} seed={seed!r} "
                                f"rho={rho!r})",
                            )
                            self.assertEqual(model_hash(dist), model_hash(loaded.fitted))
                            self.assertFalse(
                                any("integrity note" in note for note in loaded.notes), loaded.notes
                            )
                            user_warnings = [
                                str(w.message) for w in caught if issubclass(w.category, UserWarning)
                            ]
                            self.assertEqual(user_warnings, [])
                            checked += 1
        self.assertGreater(checked, 100)


class EstimatorPathNeverManifestsTest(unittest.TestCase):
    """Documents (and pins) why this defect has never been observable through the estimator: pins
    the invariant this file's docstring relies on, so a future change to how ``estimate()`` derives
    ``sigma`` that broke it would be caught here rather than discovered as a live false positive."""

    def test_estimated_sigma_round_trips_exactly_across_a_seeded_sweep(self):
        for dim in range(2, 8):
            for seed in range(10):
                with self.subTest(dim=dim, seed=seed):
                    rng = np.random.RandomState(seed)
                    estimator = SpearmanRankingEstimator(dim=dim)
                    accumulator = estimator.accumulator_factory().make()
                    for _ in range(30):
                        accumulator.update(list(rng.permutation(dim)), 1.0, None)
                    dist = estimator.estimate(None, accumulator.value())
                    restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                    restored.__pysp_setstate__(dist.__pysp_getstate__())
                    self.assertTrue(np.array_equal(dist.sigma, restored.sigma))


if __name__ == "__main__":
    unittest.main()
