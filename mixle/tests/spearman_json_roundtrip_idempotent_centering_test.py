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

An adversarial review of the fix above found it introduced a real regression, Spearman-specific
(Thurstone's ``mu`` has no analogous constraint, so it is unaffected): skipping the shift on the
``already_centered=True`` restore path also skipped the ONLY thing that ever enforced ``sigma``'s
total-sum invariant, ``sum(sigma) == dim*(dim-1)/2``. The majorization loop in
``_validate_location`` only lower-bounds ascending prefix sums through ``dim - 1`` terms and never
inspects the full sum, so a ``sigma`` shifted by any constant -- still strictly ascending, so it
clears every one of those bounds -- restored with a silently corrupted total instead of being
rejected. ``SetstateRejectsNonCanonicalTotalTest`` below reproduces that exact scenario and pins the
follow-up fix: a dedicated equality check for the total, scoped to ``already_centered=True``, that
rejects a bad total rather than silently accepting it or silently re-deriving it (re-deriving would
reintroduce the very ULP drift this file's other tests exist to close).

A second adversarial review, of that follow-up fix itself, found a further gap:
``SetstateRejectsNonCanonicalTotalTest``'s new tolerance was ``1.0e-10 * max(1.0, max(abs(sigma)))``
-- sized from the RESTORED, already-centered ``sigma``, which is always small (O(dim)) by
construction. The cancellation error the tolerance actually needs to bound was produced once, by
``__init__``'s ``raw - raw.mean() + (dim - 1) / 2.0`` shift, at the ORIGINAL object's construction
time, and scales with the magnitude of the RAW, pre-centering input -- invisible by the time
``__pysp_setstate__`` runs. Direct construction from a raw location vector whose entries share a
large common offset (the module docstring's "possibly fractional mean rank vector" supports this
directly) is a legitimate, unmodified ``__init__`` call whose one-time centering residual can
exceed that tiny tolerance, and was falsely rejected on restore. Confirmed reproduction: ``dim=7``,
``raw = arange(7) + 994987940.0788243`` constructs validly but used to raise ``ValueError`` on
restore.

Fixed by carrying the raw pre-centering magnitude forward through
``__pysp_getstate__``/``__pysp_setstate__`` (``sigma_raw_scale`` in the serialized state) so restore
sizes its tolerance from the same reference scale ``__init__`` itself used, rather than guessing
from values that no longer carry that information -- capped at
``_TRUSTED_SIGMA_RAW_SCALE_CEILING`` (2**31, ~2.147e9) so a corrupted or spoofed hint cannot
rubber-stamp an arbitrary total and reopen the exact gap ``SetstateRejectsNonCanonicalTotalTest``
closes (see the module comment above ``_TRUSTED_SIGMA_RAW_SCALE_CEILING`` in ``spearman_rho.py``
for the full reasoning). ``SpearmanRestoreTrustedRawScaleTest`` below pins the reproduction, the
empirically-verified "handled" range (common offsets through at least 1e11, across the dims an
adversarial sweep exercised), and the residual "not handled" range beyond the ceiling that remains
a documented, accepted limitation rather than a silent gap.

A THIRD adversarial review, of that raw-scale-hint fix, found a further gap: the hint's parse
(``candidate = float(raw_scale_hint)``) caught only ``(TypeError, ValueError)``, not
``OverflowError`` -- raised instead of ``ValueError`` when ``raw_scale_hint`` is a Python ``int``
too large for ``float()`` to represent. ``json.loads`` decodes an oversized JSON integer literal
(no size limit) into exactly such an ``int``, so a corrupted or adversarial ``sigma_raw_scale``
field reachable through ordinary deserialization crashed a restore whose actual ``sigma`` was
completely legitimate -- directly contradicting this fix's own documented intent ("a corrupted hint
should not itself be why a restore whose actual sigma is fine gets rejected"). Fixed by adding
``OverflowError`` to the caught tuple, so it is treated exactly like every other unusable hint: the
widening is forfeited, not crashed. ``SpearmanRestoreOverflowHintTest`` below pins this and a sweep
of other adversarial hint values.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

from mixle.data.hashing import model_hash
from mixle.lifecycle import Model
from mixle.stats.rankings.spearman_rho import (
    _TRUSTED_SIGMA_RAW_SCALE_CEILING,
    SpearmanRankingDistribution,
    SpearmanRankingEstimator,
    _validate_location,
)

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


class SetstateRejectsNonCanonicalTotalTest(unittest.TestCase):
    """Regression test for a real defect an adversarial review found in 25c18167 itself: skipping
    the shift on the ``already_centered=True`` restore path also skipped the ONLY check that ever
    enforced ``sigma``'s total-sum invariant (``sum(sigma) == dim*(dim-1)/2``). The majorization
    loop in ``_validate_location`` only lower-bounds each ascending prefix sum through ``dim - 1``
    terms and never inspects the full sum -- on the ordinary (non-restore) path that total is
    already exact by construction, because it is exactly what the skipped shift computes. A
    ``sigma`` shifted by a positive constant is still strictly ascending, so it clears every one of
    those prefix-sum lower bounds, but its total is wrong: fed through ``__pysp_setstate__`` on the
    code 25c18167 introduced, such a ``sigma`` silently kept the corrupted total instead of being
    rejected. This pins the fix: a dedicated equality check for that invariant, scoped to the
    ``already_centered=True`` path, that rejects rather than either silently accepting a bad total
    or silently re-deriving one (re-shifting would reintroduce the ULP drift 25c18167 fixed).
    """

    def test_reviewer_scenario_shift_by_100_on_dim_5_is_rejected(self):
        # The reviewer's exact repro: canonical sigma=[0,1,2,3,4] (dim=5, sum=10) shifted by +100
        # to [100,101,102,103,104] (sum=510, still strictly ascending).
        dim = 5
        canonical = np.arange(dim, dtype=np.float64)
        dist = SpearmanRankingDistribution(canonical, rho=0.7, name="judges", keys="k")
        state = dist.__pysp_getstate__()
        self.assertTrue(np.array_equal(state["sigma"], canonical))

        corrupted = state["sigma"] + 100.0
        self.assertTrue(np.all(np.diff(corrupted) > 0), "corrupted sigma must stay strictly ascending")
        self.assertEqual(float(corrupted.sum()), 510.0)

        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        with self.assertRaisesRegex(ValueError, "already_centered"):
            restored.__pysp_setstate__(dict(state, sigma=corrupted))

    def test_corrupted_sigma_clears_the_majorization_loop_on_its_own(self):
        # Sanity check that the reviewer's scenario is really the gap the new check closes, and not
        # something the pre-existing majorization loop already caught for an unrelated reason: every
        # ascending prefix sum clears its lower bound even though the full total (510) is wrong.
        dim = 5
        corrupted = np.arange(dim, dtype=np.float64) + 100.0
        ordered = np.sort(corrupted)
        for count in range(1, dim):
            minimum = count * (count - 1) / 2.0
            self.assertGreaterEqual(float(np.sum(ordered[:count])), minimum)

    def test_a_sweep_of_positive_constant_shifts_and_dims_is_rejected(self):
        checked = 0
        for dim in range(2, 9):
            canonical = np.arange(dim, dtype=np.float64)
            dist = SpearmanRankingDistribution(canonical, rho=1.0)
            state = dist.__pysp_getstate__()
            for shift in (0.01, 0.5, 1.0, 3.5, 100.0):
                with self.subTest(dim=dim, shift=shift):
                    corrupted = state["sigma"] + shift
                    restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                    with self.assertRaisesRegex(ValueError, "already_centered"):
                        restored.__pysp_setstate__(dict(state, sigma=corrupted))
                    checked += 1
        self.assertGreater(checked, 30)

    def test_legitimately_centered_sigma_still_restores_without_error(self):
        # The fix must not become a false positive on the untampered path this file's other tests
        # already sweep: a real, once-canonicalized sigma must keep restoring cleanly.
        dim = 6
        canonical = _fractional_mean_rank_sigma(dim, seed=3)
        dist = SpearmanRankingDistribution(canonical, rho=0.4)
        state = dist.__pysp_getstate__()
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(state)  # must not raise
        self.assertTrue(np.array_equal(dist.sigma, restored.sigma))


class SpearmanRestoreTrustedRawScaleTest(unittest.TestCase):
    """Regression test for a MAJOR gap an adversarial review found in
    ``SetstateRejectsNonCanonicalTotalTest``'s own fix (see this file's module docstring for the
    full history): its tolerance was sized from the restored, already-centered ``sigma`` -- always
    small, O(dim) -- rather than from the RAW, pre-centering magnitude ``__init__`` actually
    centered away, which is invisible by the time ``__pysp_setstate__`` runs. Direct construction
    from a raw location vector whose entries share a large common offset (the module docstring's
    "possibly fractional mean rank vector") is a legitimate, unmodified ``__init__`` call whose
    one-time centering residual can exceed a tolerance sized from the small centered values, and
    was falsely rejected on restore.

    Fixed by carrying the raw pre-centering magnitude forward through
    ``__pysp_getstate__``/``__pysp_setstate__`` (``sigma_raw_scale``) so restore sizes its
    tolerance from the same reference scale ``__init__`` used, capped at
    ``_TRUSTED_SIGMA_RAW_SCALE_CEILING`` (2**31, ~2.147e9) so a corrupted or spoofed hint cannot
    rubber-stamp an arbitrary total -- see ``spearman_rho.py``'s comment above that constant for
    the full reasoning and the arithmetic behind the specific ceiling and safety-factor constants.

    HANDLED, empirically verified (see this file's fix commit for the sweep methodology): OF THE
    common offsets in the 1e9-1e11 range that constructed into a legitimate mean-rank vector in the
    first place (not every random draw at these magnitudes does -- __init__'s own, unmodified
    majorization check on the freshly-centered result can reject some, for reasons unrelated to
    this fix), 100% then round-tripped through restore with zero false rejects, across dims 7,
    10-14, and 18-22 (the dims an adversarial sweep exercised). NOT HANDLED, and not claimed to be:
    offsets at 1e12 and beyond, where the false-reject rate on restore becomes nonzero and grows
    with scale -- a documented, accepted residual limitation (see
    ``SpearmanRankingDistribution``'s class docstring), not silently papered over: the original
    magnitude is genuinely unrecoverable at restore time, and widening the cap further would start
    silently admitting the same shifted-total corruption ``SetstateRejectsNonCanonicalTotalTest``
    exists to catch.
    """

    def test_reviewer_reproduction_dim7_offset_994987940_now_round_trips(self):
        # The confirmed false-reject repro: a valid, unmodified __init__ call that used to raise
        # ValueError on restore.
        dim = 7
        raw = np.arange(dim, dtype=np.float64) + 994987940.0788243
        dist = SpearmanRankingDistribution(raw, rho=0.6)
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(dist.__pysp_getstate__())  # used to raise ValueError
        self.assertTrue(np.array_equal(dist.sigma, restored.sigma))
        self.assertEqual(model_hash(dist), model_hash(restored))

    def test_reviewer_reproduction_round_trips_through_deploy_and_load(self):
        # Same case, through the real artifact path (mirrors
        # SpearmanDeployLoadIdempotentCenteringTest above): before this fix, __pysp_setstate__'s
        # ValueError on Model.load() falls back to the untrusted-pickle path, which requires the
        # caller to pass trust_code=True -- a real behavioral regression for validly-constructed
        # input even though it fails safe. Confirm the safe JSON path now handles it directly, with
        # no fallback and no integrity-note warning.
        dim = 7
        raw = np.arange(dim, dtype=np.float64) + 994987940.0788243
        dist = SpearmanRankingDistribution(raw, rho=0.6)
        model = Model(dist)
        model.fitted = dist
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "spearman-large-common-offset")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.deploy(out)
                loaded = Model.load(out)
            self.assertTrue(np.array_equal(dist.sigma, loaded.fitted.sigma))
            self.assertEqual(model_hash(dist), model_hash(loaded.fitted))
            self.assertFalse(any("integrity note" in note for note in loaded.notes), loaded.notes)
            user_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(user_warnings, [])

    def test_handled_range_offsets_through_1e11_round_trip_across_reviewer_dims(self):
        # HANDLED range: the reviewer's exact dims at the three scale decades (1e9, 1e10, 1e11)
        # empirically confirmed to round-trip with a 0% false-reject rate. Goes through
        # _validate_location directly rather than the full SpearmanRankingDistribution constructor:
        # the class's exact O(dim * 2**dim) partition computation (needed for log_density/sampling,
        # irrelevant to the restore-time tolerance this test exercises) takes tens of seconds at
        # dim=22 and would make this sweep prohibitively slow for what it is checking.
        #
        # A random raw offset at these magnitudes is not always itself a legitimate mean-rank
        # vector once centered -- independent of this fix, __init__'s OWN (unmodified,
        # already_centered=False) majorization check can reject some -- so each cell retries a
        # bounded number of times to collect a handful of legitimately-constructible offsets rather
        # than assuming every draw succeeds.
        dims = (7, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22)
        scales = (1.0e9, 1.0e10, 1.0e11)
        rng = np.random.RandomState(20260830)
        checked = 0
        for scale in scales:
            for dim in dims:
                constructed = 0
                attempts = 0
                while constructed < 5 and attempts < 100:
                    attempts += 1
                    offset = scale * float(rng.uniform(1.0, 2.0)) + float(rng.uniform(0.0, 1.0))
                    raw = np.arange(dim, dtype=np.float64) + offset
                    try:
                        sigma, raw_scale = _validate_location(raw, already_centered=False)
                    except ValueError:
                        continue  # not a legitimate mean-rank vector at this draw; not our concern
                    constructed += 1
                    with self.subTest(dim=dim, scale=scale, offset=offset):
                        restored_sigma, _ = _validate_location(
                            sigma, already_centered=True, raw_scale_hint=raw_scale
                        )
                        self.assertTrue(np.array_equal(sigma, restored_sigma))
                        checked += 1
                self.assertGreater(
                    constructed,
                    0,
                    f"could not legitimately construct any dim={dim} sigma near scale=1e"
                    f"{np.log10(scale):.0f} in {attempts} attempts",
                )
        self.assertGreaterEqual(checked, len(dims) * len(scales) * 5)

    def test_handled_range_round_trips_through_the_real_class_too(self):
        # Same handled range and scales, but through the real class end to end (construct,
        # getstate, setstate, sigma/model_hash equality) rather than _validate_location directly --
        # restricted to dims cheap enough for the exact partition computation (<=14, well under a
        # second each) to keep this fast; dims 18-22 are covered directly above. Retries per cell
        # for the same reason as the sweep above: not every random draw is itself a legitimate
        # mean-rank vector once centered, independent of this fix.
        rng = np.random.RandomState(20260831)
        checked = 0
        for scale in (1.0e9, 1.0e10, 1.0e11):
            for dim in (7, 10, 11, 12, 13, 14):
                constructed = False
                for _ in range(50):
                    offset = scale * float(rng.uniform(1.0, 2.0)) + float(rng.uniform(0.0, 1.0))
                    raw = np.arange(dim, dtype=np.float64) + offset
                    try:
                        dist = SpearmanRankingDistribution(raw, rho=0.5, max_dim=dim)
                    except ValueError:
                        continue
                    constructed = True
                    with self.subTest(dim=dim, scale=scale):
                        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                        restored.__pysp_setstate__(dist.__pysp_getstate__())
                        self.assertTrue(np.array_equal(dist.sigma, restored.sigma))
                        self.assertEqual(model_hash(dist), model_hash(restored))
                        checked += 1
                    break
                self.assertTrue(constructed, f"could not legitimately construct dim={dim} at scale={scale}")
        self.assertGreaterEqual(checked, 18)

    def test_residual_limitation_specific_offsets_beyond_ceiling_still_false_reject(self):
        # NOT HANDLED, documented residual limitation: fixed, deterministic (dim, offset) pairs at
        # ~1.7e15-7.1e15 -- well beyond _TRUSTED_SIGMA_RAW_SCALE_CEILING (2**31, ~2.147e9) -- whose
        # legitimate cancellation residual exceeds even the capped tolerance. Each of these
        # constructs validly (an unmodified, unaffected __init__ call) but still cannot be
        # restored, exactly as documented in SpearmanRankingDistribution's class docstring and the
        # module comment above _TRUSTED_SIGMA_RAW_SCALE_CEILING. This is not a leftover bug this
        # fix failed to catch: it is the intentional, reasoned boundary of how far the tolerance is
        # willing to widen before it would start risking silently admitting genuine corruption
        # (see SetstateRejectsNonCanonicalTotalTest, unmodified above, for what that would reopen).
        cases = (
            (14, 1686774528057326.0),
            (18, 5510083671818978.0),
            (22, 7113069285858470.0),
        )
        for dim, offset in cases:
            with self.subTest(dim=dim, offset=offset):
                raw = np.arange(dim, dtype=np.float64) + offset
                sigma, raw_scale = _validate_location(raw, already_centered=False)  # constructs fine
                with self.assertRaisesRegex(ValueError, "already_centered"):
                    _validate_location(sigma, already_centered=True, raw_scale_hint=raw_scale)

    def test_residual_limitation_case_also_false_rejects_through_the_real_class(self):
        # Same as above, spot-checked through the real class end to end (cheap at dim=14) rather
        # than _validate_location alone, so the documented limitation is pinned on the actual
        # public restore path and not just the internal helper.
        dim = 14
        offset = 1686774528057326.0
        raw = np.arange(dim, dtype=np.float64) + offset
        dist = SpearmanRankingDistribution(raw, rho=0.5, max_dim=dim)  # constructs fine
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        with self.assertRaisesRegex(ValueError, "already_centered"):
            restored.__pysp_setstate__(dist.__pysp_getstate__())

    def test_spoofed_raw_scale_hint_cannot_mask_a_small_shift_corruption(self):
        # Defense-in-depth regression: sigma_raw_scale is untrusted input like every other state
        # field (nothing binds it cryptographically to sigma), so a state that lies about it --
        # claiming the maximum trusted ceiling -- must not be able to buy enough tolerance to admit
        # a corrupted total that has nothing to do with a legitimate large-offset construction. This
        # must keep failing exactly like SetstateRejectsNonCanonicalTotalTest's smallest case (a
        # dim=2, shift=0.01 corruption) even in the worst case where the hint is also fabricated.
        dim = 2
        canonical = np.arange(dim, dtype=np.float64)
        dist = SpearmanRankingDistribution(canonical, rho=1.0)
        state = dist.__pysp_getstate__()
        corrupted = state["sigma"] + 0.01
        spoofed_state = dict(state, sigma=corrupted, sigma_raw_scale=2.0**31)
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        with self.assertRaisesRegex(ValueError, "already_centered"):
            restored.__pysp_setstate__(spoofed_state)

    def test_missing_sigma_raw_scale_falls_back_to_pre_fix_behavior(self):
        # Backward/forward compatibility: sigma_raw_scale is optional, exactly like fit_diagnostics
        # (state serialized before this field existed, or built by hand without it, must not hard
        # fail). For an ordinary small-scale sigma this is a complete no-op --
        # _validate_location's fallback estimate is exactly what it computed before this fix.
        dim = 6
        canonical = np.arange(dim, dtype=np.float64)
        dist = SpearmanRankingDistribution(canonical, rho=0.4)
        state = dict(dist.__pysp_getstate__())
        del state["sigma_raw_scale"]
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(state)  # must not raise
        self.assertTrue(np.array_equal(dist.sigma, restored.sigma))

    def test_missing_sigma_raw_scale_reproduces_the_original_false_reject_for_large_offsets(self):
        # The flip side of the fallback above, stated explicitly rather than left implicit: for a
        # state that predates this field, the reviewer's exact large-offset repro is NOT fixed --
        # there is no information to recover it from. This is expected, not a gap this fix missed:
        # the whole point of sigma_raw_scale is that the raw magnitude cannot be reconstructed from
        # the restored, already-centered sigma alone.
        dim = 7
        raw = np.arange(dim, dtype=np.float64) + 994987940.0788243
        dist = SpearmanRankingDistribution(raw, rho=0.6)
        state = dict(dist.__pysp_getstate__())
        del state["sigma_raw_scale"]
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        with self.assertRaisesRegex(ValueError, "already_centered"):
            restored.__pysp_setstate__(state)

    def test_sigma_raw_scale_is_exact_and_survives_repeated_round_trips_unclamped(self):
        # The carried provenance value is the RAW magnitude itself (not pre-clamped to the trusted
        # ceiling), so it round-trips exactly across any number of hops -- a future release could
        # raise _TRUSTED_SIGMA_RAW_SCALE_CEILING and immediately benefit from state already on disk
        # rather than having lossily discarded the extra precision on the first restore. Uses a
        # fixed offset ABOVE the ceiling (unlike the reviewer's ~9.95e8 repro, which is comfortably
        # below it) specifically so the clamp inside _validate_location's tolerance computation is
        # actually exercised, while the value handed back by __pysp_getstate__ stays the true,
        # uncapped one.
        dim = 7
        offset = 4081618133.2800913  # ~4.08e9, > 2**31; empirically confirmed to round-trip
        raw = np.arange(dim, dtype=np.float64) + offset
        dist = SpearmanRankingDistribution(raw, rho=0.6)
        expected_raw_scale = float(np.max(np.abs(raw)))
        self.assertEqual(dist._sigma_raw_scale, expected_raw_scale)
        self.assertGreater(dist._sigma_raw_scale, _TRUSTED_SIGMA_RAW_SCALE_CEILING)

        current = dist
        for _ in range(3):
            restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
            restored.__pysp_setstate__(current.__pysp_getstate__())
            self.assertEqual(restored._sigma_raw_scale, expected_raw_scale)
            self.assertTrue(np.array_equal(current.sigma, restored.sigma))
            current = restored


class SpearmanRestoreOverflowHintTest(unittest.TestCase):
    """Regression test for a MAJOR gap an adversarial review found in
    ``SpearmanRestoreTrustedRawScaleTest``'s own fix: parsing ``raw_scale_hint`` via
    ``float(raw_scale_hint)`` only caught ``(TypeError, ValueError)``, not the ``OverflowError`` a
    too-large Python ``int`` raises -- reachable through ordinary JSON deserialization of a
    corrupted ``sigma_raw_scale`` field, crashing a restore whose actual ``sigma`` was fine. See
    this file's module docstring for the full history.
    """

    def _tampered_state(self, dim: int = 6, *, sigma_raw_scale) -> dict:
        canonical = np.arange(dim, dtype=np.float64)
        dist = SpearmanRankingDistribution(canonical, rho=0.4)
        state = dict(dist.__pysp_getstate__())
        state["sigma_raw_scale"] = sigma_raw_scale
        return state

    def _assert_restores_like_no_hint(self, *, sigma_raw_scale) -> None:
        dim = 6
        canonical = np.arange(dim, dtype=np.float64)
        expected = SpearmanRankingDistribution(canonical, rho=0.4)
        state = self._tampered_state(dim, sigma_raw_scale=sigma_raw_scale)
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(state)
        self.assertTrue(np.array_equal(expected.sigma, restored.sigma))

    def test_an_oversized_positive_int_hint_does_not_crash_the_restore(self):
        # The exact reviewer repro: sigma itself is untouched, only the hint is corrupted.
        self._assert_restores_like_no_hint(sigma_raw_scale=10**400)

    def test_an_oversized_negative_int_hint_does_not_crash_the_restore(self):
        self._assert_restores_like_no_hint(sigma_raw_scale=-(10**400))

    def test_infinity_hint_is_excluded_by_the_existing_finite_check(self):
        self._assert_restores_like_no_hint(sigma_raw_scale=float("inf"))

    def test_negative_infinity_hint_is_excluded_by_the_existing_finite_check(self):
        self._assert_restores_like_no_hint(sigma_raw_scale=float("-inf"))

    def test_a_nan_valued_string_hint_is_excluded_by_the_existing_finite_check(self):
        self._assert_restores_like_no_hint(sigma_raw_scale="nan")

    def test_an_ordinary_negative_hint_is_excluded_by_the_existing_positivity_check(self):
        self._assert_restores_like_no_hint(sigma_raw_scale=-1e20)

    def test_genuine_corruption_at_realistic_magnitude_is_still_rejected_despite_an_oversized_hint(self):
        # The anti-spoofing property this whole mechanism exists for: an oversized hint must not
        # smuggle a genuinely corrupted, realistic-magnitude sigma past the total-sum check.
        dim = 6
        canonical = np.arange(dim, dtype=np.float64)
        shifted = canonical + 5.0  # still ascending, clears the majorization loop, wrong total
        dist = SpearmanRankingDistribution(shifted, rho=0.4)  # __init__ re-canonicalizes on fresh build
        state = dict(dist.__pysp_getstate__())
        state["sigma"] = shifted  # reintroduce the corrupted (never-canonicalized) total
        state["sigma_raw_scale"] = 10**400
        with self.assertRaises(ValueError):
            SpearmanRankingDistribution.__new__(SpearmanRankingDistribution).__pysp_setstate__(state)


if __name__ == "__main__":
    unittest.main()
