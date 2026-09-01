"""Thurstone false-positive tamper warning: centering an already-centered ``mu`` on restore.

``ThurstoneDistribution.__init__`` centers its ``mu`` argument (``self.mu = raw_mu -
raw_mu.mean()``). ``__pysp_getstate__`` serializes that already-centered ``mu`` verbatim.
``__pysp_setstate__`` used to reconstruct the object via ``self.__init__(state['mu'], ...)`` --
feeding the already-centered array back through the SAME unconditional centering step. Centering an
already-centered float64 array is not exactly idempotent: the residual mean left over from the first
centering is generically a tiny nonzero value (e.g. ``-4.44e-17``), not exactly ``0.0``, so
re-centering shifts one or more elements by 1-few ULP.

That shift is invisible to ``log_density``/``sample`` -- the approximation is unaffected at any
scale that matters -- but it flips raw float64 bytes, and :func:`mixle.data.hashing.model_hash`
fingerprints those bytes with SHA-256. A same-process, same-mixle-version, zero-tampering
``deploy()``-then-``load()`` cycle could therefore raise ``Model.load``'s "integrity note"
``UserWarning`` (mixle/lifecycle.py) -- exactly the false positive
``campaign2_lifecycle_test.py``'s ``ManifestSwapDisclosureTest.
test_untampered_artifact_loads_without_integrity_noise`` already pins as forbidden, just never
exercised against Thurstone before (one of three families -- Bernoulli-set, Thurstone, Spearman --
whose JSON codec is new this release; see ``campaign2_codecs_test.py``).

The fix makes ``__pysp_setstate__`` restore ``mu`` as given rather than re-deriving it: an internal,
keyword-only ``_mu_already_centered`` flag on ``__init__`` (used only by ``__pysp_setstate__``) skips
the mean-subtraction step for the restore path while leaving ordinary construction -- and every other
field's validation and the RNG-derived approximation tables -- untouched.

This reproduces on a substantial fraction of realistically-fitted Thurstone models, not all of them
(whether the first centering leaves an exact-zero or a tiny-nonzero residual mean depends on the
specific float64 bit pattern of the fitted ``mu``) -- which is exactly why
``campaign2_codecs_test.py``'s existing single-fit round trip
(``test_thurstone_round_trip_scores_bit_identical``) did not catch it: that one fit's ``mu`` happens
to land on an exact-zero residual. The sweeps below vary dim/seed/pseudo_count (the same axes
``ThurstoneEstimator`` exposes) precisely so a reintroduced regression cannot hide behind a single
lucky case again; on the pre-fix code this sweep methodology reproduced mismatches on roughly a
quarter to a third of combinations.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

from mixle.data.hashing import model_hash
from mixle.lifecycle import Model
from mixle.stats.rankings.thurstone import ThurstoneDistribution, ThurstoneEstimator

# Same three axes ThurstoneEstimator exposes, swept jointly so the fitted mu vectors that come out
# of the pairwise-moment estimate cover a wide range of float64 bit patterns -- some of which land
# on an exact-zero post-centering residual (no bug visible) and some of which do not (bug visible).
_DIMS = (2, 3, 4, 5, 6, 7, 8)
_SEEDS = range(10)
_PSEUDO_COUNTS = (0.0, 0.5, 1.0, 5.0)

# A smaller grid for the deploy()/load() variant: identical methodology, kept narrower only because
# each case now also pays for real JSON encode/decode and a full model_hash() (a registry-population
# pass on first use, then a SHA-256 canonicalization per call) rather than being pure in-memory.
_DEPLOY_DIMS = (2, 3, 4, 5, 6)
_DEPLOY_SEEDS = range(6)


def _thurstone_data(dim: int, seed: int, n_obs: int = 40) -> list[list[int]]:
    """A realistic full-ranking dataset: iid Case-V utility draws around a random ``mu``."""
    rng = np.random.RandomState(seed)
    true_mu = rng.normal(scale=1.5, size=dim)
    true_mu -= true_mu.mean()
    rows = []
    for _ in range(n_obs):
        utilities = true_mu + rng.standard_normal(dim)
        rows.append([int(v) for v in np.argsort(-utilities)])
    return rows


def _fit(dim: int, seed: int, pseudo_count: float) -> ThurstoneDistribution:
    """Fit through the real accumulator/estimator path -- the same one ThurstoneEstimator.estimate()
    uses in production, and the same construction campaign2_codecs_test.py's own ``_fit`` helper
    uses -- not a hand-built mu vector."""
    estimator = ThurstoneEstimator(dim=dim, n_mc=50, seed=seed, pseudo_count=pseudo_count)
    accumulator = estimator.accumulator_factory().make()
    for row in _thurstone_data(dim, seed):
        accumulator.update(row, 1.0, None)
    return estimator.estimate(None, accumulator.value())


class ThurstonePureGetstateSetstateIdempotentCenteringTest(unittest.TestCase):
    """Purest isolation: ``__pysp_getstate__`` -> ``__pysp_setstate__`` in memory, zero JSON/file
    machinery -- exactly where the finding isolated the defect to confirm it has nothing to do with
    JSON's float encoding (Python's json module round-trips float64 exactly) and everything to do
    with the centering arithmetic itself."""

    def test_round_trip_is_bit_identical_across_a_seeded_sweep(self):
        checked = 0
        for dim in _DIMS:
            for seed in _SEEDS:
                for pseudo_count in _PSEUDO_COUNTS:
                    with self.subTest(dim=dim, seed=seed, pseudo_count=pseudo_count):
                        dist = _fit(dim, seed, pseudo_count)
                        state = dist.__pysp_getstate__()
                        restored = ThurstoneDistribution.__new__(ThurstoneDistribution)
                        restored.__pysp_setstate__(state)
                        self.assertTrue(
                            np.array_equal(dist.mu, restored.mu),
                            f"mu drifted by {restored.mu - dist.mu!r} across a pure getstate/setstate "
                            f"round trip (dim={dim!r} seed={seed!r} pseudo_count={pseudo_count!r})",
                        )
                        self.assertEqual(model_hash(dist), model_hash(restored))
                        # The approximation tables are re-derived from mu in __init__, not shipped --
                        # if setstate ever restored a shifted mu again, this would also drift (any
                        # shift, however tiny, feeds into the RNG-seeded utility draws) even though
                        # the ranking probabilities it implies would still be indistinguishable.
                        self.assertTrue(np.array_equal(dist._approximation_draws, restored._approximation_draws))
                        checked += 1
        # Large enough that a reintroduced regression cannot hide behind one lucky exact-zero
        # residual the way campaign2_codecs_test.py's single-fit round trip did.
        self.assertGreater(checked, 100)

    def test_fresh_construction_still_centers_on_first_construction(self):
        # The fix must not touch ordinary construction: only __pysp_setstate__'s internal
        # _mu_already_centered=True call may skip the mean subtraction.
        raw = np.array([5.0, 7.0, 9.0, -3.0])
        dist = ThurstoneDistribution(raw, n_mc=10)
        self.assertTrue(np.array_equal(dist.mu, raw - raw.mean()))
        self.assertAlmostEqual(float(dist.mu.mean()), 0.0, places=12)

    def test_mu_stays_read_only_on_both_construction_paths(self):
        dist = ThurstoneDistribution(np.array([1.0, 2.0, 3.0]), n_mc=10)
        with self.assertRaises(ValueError):
            dist.mu[0] = 0.0
        restored = ThurstoneDistribution.__new__(ThurstoneDistribution)
        restored.__pysp_setstate__(dist.__pysp_getstate__())
        with self.assertRaises(ValueError):
            restored.mu[0] = 0.0


class ThurstoneDeployLoadIdempotentCenteringTest(unittest.TestCase):
    """Same defect, exercised through the real artifact path: ``deploy()`` records
    ``model_hash(fitted)`` in the manifest and ``load()`` recomputes it on the restored object,
    warning an "integrity note" ``UserWarning`` on any mismatch (mixle/lifecycle.py). A clean,
    untampered round trip must never trip that check."""

    def test_deploy_then_load_preserves_mu_and_model_hash_across_a_seeded_sweep(self):
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for dim in _DEPLOY_DIMS:
                for seed in _DEPLOY_SEEDS:
                    for pseudo_count in _PSEUDO_COUNTS:
                        with self.subTest(dim=dim, seed=seed, pseudo_count=pseudo_count):
                            dist = _fit(dim, seed, pseudo_count)
                            # Same construction Model.load() itself uses (mixle/lifecycle.py) to wrap
                            # an already-fitted distribution -- deploy() only needs self.fitted set,
                            # so this exercises the real deploy()/load() path without paying for the
                            # unrelated optimize()/certify() machinery a full Model(...).fit(data)
                            # call would run for what is a closed-form pairwise-moment estimate.
                            model = Model(dist)
                            model.fitted = dist
                            out = os.path.join(tmp, f"thurstone-{dim}-{seed}-{pseudo_count}")
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model.deploy(out)
                                loaded = Model.load(out)
                            self.assertTrue(
                                np.array_equal(dist.mu, loaded.fitted.mu),
                                f"mu drifted across deploy()/load() (dim={dim!r} seed={seed!r} "
                                f"pseudo_count={pseudo_count!r})",
                            )
                            self.assertEqual(model_hash(dist), model_hash(loaded.fitted))
                            self.assertFalse(any("integrity note" in note for note in loaded.notes), loaded.notes)
                            user_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
                            self.assertEqual(user_warnings, [])
                            checked += 1
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
