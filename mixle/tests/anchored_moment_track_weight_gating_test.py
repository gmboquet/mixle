"""Regression tests: a non-positive-weight observation must never seed the shift-anchored
moment track's anchor.

CONFIRMED FINDING (release 0.8.0, campaign eight, independently reproduced against the built wheel
by two separate agents): ``AnchoredMomentTrack._anchor_scalar``/``_anchor_chunk`` in
``mixle/stats/univariate/continuous/_observation_contracts.py`` activated the shift-anchoring
mechanism from whichever observation arrived first -- ``update()``'s scalar ``x``, or
``seq_update()``'s ``x[0]`` via ``_anchor_fold_chunk`` -- with NO check on that observation's
weight. A single observation carried at weight 0.0 at an extreme magnitude became the permanent
anchor, and every subsequent fully-weighted, legitimate observation's contribution was computed as
a delta against that poisoned anchor, destroying precision: every other real point differences
against an anchor that sits nowhere near the data's own scale, reintroducing exactly the
large-offset cancellation the anchored track exists to eliminate. Confirmed to corrupt
GeneralizedPareto (scale off by up to 5x, shape sign-flipped or pinned at the wrong boundary, one
variant >43,000x off), Gumbel/StudentT (scale collapses 8 orders of magnitude to the family floor),
and Gaussian (variance 7 billion times too large).

A zero-weight first observation is ORDINARY usage, not misuse: EM's per-point responsibilities
passed through ``seq_update`` are the library's own internal calling convention
(``mixle/stats/compute/stacked.py``: ``acc.seq_update(host_enc, gamma_np[:, i], ...)``,
``mixle/stats/compute/torch_mixture.py``), and a component can legitimately receive a
responsibility of exactly 0.0 for the first point in a batch.

The fix -- mirroring the ``if weight > 0.0`` gate ``GeneralizedParetoAccumulator.update`` already
applies to its max-tracking -- is to activate the anchor only from an observation whose weight is
strictly positive; for the chunk path, that means the first element of the chunk with positive
weight, not positionally ``x[0]``. The pattern was duplicated (not shared) across several
accumulators; this file exercises the four families campaign eight ran directly (GeneralizedPareto,
Gumbel, Gaussian, StudentT) against both the scalar ``update()`` path (``_anchor_scalar``, shared
via the ``AnchoredMomentTrack`` mixin for GeneralizedPareto/Gumbel/StudentT, and an independently
duplicated copy in ``GaussianAccumulator``) and the vectorized ``seq_update()`` path
(``_anchor_chunk`` via ``_anchor_fold_chunk``).

A note on ``weight=0.0`` vs. a merely tiny positive weight (e.g. ``1e-300``): the specified fix's
gate is exactly ``weight > 0.0``, matching the accepted sibling pattern -- not a fuzzy epsilon
threshold. That boundary is exact and provable: with ``weight=0.0``, every term the poisoned
observation contributes to every accumulated quantity (raw sums, anchored sums, the ``chunk_sum``/
``chunk_sum2``/``w_sum`` the conditioning gate itself reads) is a literal ``+0.0`` in IEEE-754
arithmetic, so "fit with the poisoned point" and "fit with it omitted" must produce bit-identical
sufficient statistics whenever the fix correctly keeps that point from being chosen as the anchor --
this file asserts exactly that. A weight of ``1e-300`` is, by contrast, still strictly positive, so
per the specified fix it is legitimately allowed to seed the anchor (exactly like any other
positive-weight leading observation) -- ``test_negligible_but_still_positive_weight_seeds_the_anchor``
below pins that boundary explicitly, rather than treating ``1e-300`` as a second must-be-gated case.

FOLLOW-UP (release 0.8.0, adversarial re-review of e9b43c6b): the same unguarded pattern was found
independently duplicated, and still unfixed, in
``mixle/stats/univariate/continuous/generalized_gaussian.py``'s ``GeneralizedGaussianAccumulator`` --
its ``update()``/``seq_update()`` hand-roll their own anchor bookkeeping rather than mixing in
``AnchoredMomentTrack``, and both reproduced the exact defect (directly executed against this
checkout): a leading weight-0.0 extreme observation drove ``alpha`` off by ~2,000,000x and pinned
``beta`` at the estimator's explicit degenerate-fallback value, and an accumulator that had received
ONLY weight-0.0 calls -- every call ever made -- still incorrectly left its anchor activated instead
of ``None`` (of every family covered by this module plus the vector families in
``anchored_moment_track_weight_gating_multivariate_test.py``, this was the one exception at the time
of this finding). ``GeneralizedGaussianAnchorWeightGatingTest`` below reuses the exact same shared
test bodies -- its ``update``/``seq_update``/``value`` surface matches the mixin-based families
exactly -- plus one extra case pinning the all-zero-weight-calls invariant specifically.
"""

import unittest

import numpy as np

from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator, GaussianDistribution, GaussianEstimator
from mixle.stats.univariate.continuous.generalized_pareto import (
    GeneralizedParetoAccumulator,
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
)
from mixle.stats.univariate.continuous.generalized_gaussian import (
    GeneralizedGaussianAccumulator,
    GeneralizedGaussianDistribution,
    GeneralizedGaussianEstimator,
)
from mixle.stats.univariate.continuous.gumbel import GumbelAccumulator, GumbelDistribution, GumbelEstimator
from mixle.stats.univariate.continuous.student_t import StudentTAccumulator, StudentTDistribution, StudentTEstimator

# A magnitude every family's accumulator accepts unconditionally (finite, and >= the GPD threshold
# used below) and that is wildly different from BULK_OFFSET, so choosing it as the anchor -- instead
# of a real bulk point -- reintroduces large-offset cancellation in a way a tight-tolerance
# comparison will unambiguously catch.
EXTREME_X = 1.0e14

# The location every family's "bulk" (legitimate, always-positive-weight) data is centered at: the
# exact magnitude this codebase's own docstrings use as the canonical case that needs the
# shift-anchored track at all (sd ~1-5 data at this offset loses ~50+ bits to cancellation in the
# raw E[x^2]-E[x]^2 form the M-step would otherwise use). Centering the bulk here -- rather than at
# a small, tame scale -- guarantees ``needs_anchor()`` activates the anchor mechanism on the
# seq_update/chunk path from the bulk data alone (mean^2/spread^2 is ~1e17-1e18, far past
# ANCHOR_CONDITION_RATIO=4e6), so the leading/trailing-observation tests below exercise WHICH point
# gets chosen as the anchor, not merely whether anchoring turns on at all.
BULK_OFFSET = 1.7e9

TOLERANCE = 1.0e-9


def _relative_close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


class _AnchorWeightGatingCases:
    """Shared test bodies, mixed into one ``unittest.TestCase`` per family below.

    Not itself a ``TestCase`` (no ``test_*`` method here runs on its own) so unittest discovery
    does not try to collect this mixin directly; each concrete subclass supplies the family-specific
    plumbing (``make_accumulator``, ``make_estimator``, ``bulk_data``, ``fitted_params``).
    """

    family_name = ""
    n_bulk = 400
    seed = 0

    def make_accumulator(self):
        raise NotImplementedError

    def make_estimator(self):
        raise NotImplementedError

    def bulk_data(self) -> np.ndarray:
        """``n_bulk`` legitimate observations, all centered at ``BULK_OFFSET``."""
        raise NotImplementedError

    def fitted_params(self, dist) -> tuple:
        """The numbers a fit should be compared on (shift-invariant + location, family-specific)."""
        raise NotImplementedError

    # -- shared plumbing ----------------------------------------------------------------------

    def _fit_scalar(self, xs, weights):
        acc = self.make_accumulator()
        for x, w in zip(xs, weights):
            acc.update(float(x), float(w), None)
        return acc, self.make_estimator().estimate(None, acc.value())

    def _fit_seq(self, xs, weights):
        acc = self.make_accumulator()
        acc.seq_update(np.asarray(xs, dtype=np.float64), np.asarray(weights, dtype=np.float64), None)
        return acc, self.make_estimator().estimate(None, acc.value())

    def _assert_params_match(self, dist_a, dist_b, msg):
        pa, pb = self.fitted_params(dist_a), self.fitted_params(dist_b)
        self.assertEqual(len(pa), len(pb))
        for a, b in zip(pa, pb):
            self.assertTrue(
                _relative_close(a, b),
                "%s: fitted params %r vs %r not within tolerance (%s)" % (self.family_name, pa, pb, msg),
            )

    # -- 1. leading weight-0.0 extreme observation must vanish, scalar update() path ----------

    def test_leading_zero_weight_extreme_observation_matches_omission_via_scalar_update(self):
        bulk = self.bulk_data()
        _, without = self._fit_scalar(bulk, np.ones(len(bulk)))
        xs = np.concatenate([[EXTREME_X], bulk])
        weights = np.concatenate([[0.0], np.ones(len(bulk))])
        acc_with, with_poison = self._fit_scalar(xs, weights)
        self._assert_params_match(with_poison, without, "scalar update(), leading weight-0.0 outlier")
        # The anchor actually activated on the first REAL (positive-weight) observation, never on
        # the poisoned one -- pins the mechanism, not just its downstream numerical effect.
        self.assertEqual(acc_with._anchor, float(bulk[0]))

    # -- 2. leading weight-0.0 extreme observation must vanish, seq_update()/chunk path --------

    def test_leading_zero_weight_extreme_observation_matches_omission_via_seq_update(self):
        bulk = self.bulk_data()
        _, without = self._fit_seq(bulk, np.ones(len(bulk)))
        xs = np.concatenate([[EXTREME_X], bulk])
        weights = np.concatenate([[0.0], np.ones(len(bulk))])
        acc_with, with_poison = self._fit_seq(xs, weights)
        self._assert_params_match(with_poison, without, "seq_update(), leading weight-0.0 outlier")
        self.assertIsNotNone(acc_with._anchor)
        self.assertEqual(acc_with._anchor, float(bulk[0]))

    # -- 3. the SAME extreme point at the END of the sequence (seq_update) == omitting it ------

    def test_trailing_zero_weight_extreme_observation_via_seq_update_matches_omission(self):
        bulk = self.bulk_data()
        _, without = self._fit_seq(bulk, np.ones(len(bulk)))
        xs = np.concatenate([bulk, [EXTREME_X]])
        weights = np.concatenate([np.ones(len(bulk)), [0.0]])
        acc_with, with_poison = self._fit_seq(xs, weights)
        self._assert_params_match(with_poison, without, "seq_update(), trailing weight-0.0 outlier")
        self.assertEqual(acc_with._anchor, float(bulk[0]))

    # -- 4. ordinary anchoring is NOT broken: a NORMAL-weight leading outlier still anchors -----

    def test_leading_extreme_observation_with_normal_weight_still_seeds_the_anchor_via_update(self):
        acc = self.make_accumulator()
        acc.update(EXTREME_X, 1.0, None)  # the very first call: unconditional activation, unchanged
        self.assertEqual(acc._anchor, EXTREME_X)

    def test_leading_extreme_observation_with_normal_weight_still_seeds_the_anchor_via_seq_update(self):
        # No synthetic poison here: the bulk data is ALREADY centered at BULK_OFFSET (~1.7e9), so
        # relative to this family's ordinary small-magnitude parameters, bulk[0] with an everyday
        # weight of 1.0 (leading, all weights positive -- the completely ordinary, unpoisoned case)
        # IS the "extreme leading point with a normal weight" scenario: needs_anchor() reliably
        # trips on the offset alone (mean^2/spread^2 ~1e17, far past ANCHOR_CONDITION_RATIO), and the
        # fix must still let this ordinary first real observation seed the anchor. (Mixing in a
        # separately-injected EXTREME_X here instead would not exercise the same thing: with EXTREME_X
        # weighted at 1.0, its own huge deviation inflates the chunk's raw spread enough that
        # needs_anchor's mean^2-vs-spread^2 ratio no longer trips at all, so the anchor would
        # correctly stay unset for a completely different reason than what this test is checking.)
        bulk = self.bulk_data()
        acc, _ = self._fit_seq(bulk, np.ones(len(bulk)))
        self.assertEqual(acc._anchor, float(bulk[0]))

    def test_negligible_but_still_positive_weight_seeds_the_anchor(self):
        # Pins the exact gate boundary documented in the module docstring: the fix is
        # `weight > 0.0`, not an epsilon threshold, so a positive-but-vanishingly-small weight
        # (1e-300, distinct from underflowing all the way to exactly 0.0) is legitimate and must
        # still be allowed to seed the anchor -- only weight <= 0.0 is excluded.
        acc = self.make_accumulator()
        acc.update(EXTREME_X, 1.0e-300, None)
        self.assertEqual(acc._anchor, EXTREME_X)


class GeneralizedParetoAnchorWeightGatingTest(_AnchorWeightGatingCases, unittest.TestCase):
    family_name = "GeneralizedPareto"

    def make_accumulator(self):
        return GeneralizedParetoAccumulator(loc=BULK_OFFSET)

    def make_estimator(self):
        return GeneralizedParetoEstimator(loc=BULK_OFFSET)

    def bulk_data(self):
        true_dist = GeneralizedParetoDistribution(scale=2.0, shape=0.3, loc=BULK_OFFSET)
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.scale, dist.shape)


class GumbelAnchorWeightGatingTest(_AnchorWeightGatingCases, unittest.TestCase):
    family_name = "Gumbel"

    def make_accumulator(self):
        return GumbelAccumulator()

    def make_estimator(self):
        return GumbelEstimator()

    def bulk_data(self):
        true_dist = GumbelDistribution(loc=BULK_OFFSET, scale=2.0)
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.loc, dist.scale)


class GaussianAnchorWeightGatingTest(_AnchorWeightGatingCases, unittest.TestCase):
    family_name = "Gaussian"

    def make_accumulator(self):
        return GaussianAccumulator()

    def make_estimator(self):
        return GaussianEstimator()

    def bulk_data(self):
        true_dist = GaussianDistribution(mu=BULK_OFFSET, sigma2=1.0)
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.mu, dist.sigma2)


class StudentTAnchorWeightGatingTest(_AnchorWeightGatingCases, unittest.TestCase):
    family_name = "StudentT"

    def make_accumulator(self):
        return StudentTAccumulator()

    def make_estimator(self):
        return StudentTEstimator(df=5.0)

    def bulk_data(self):
        true_dist = StudentTDistribution(df=5.0, loc=BULK_OFFSET, scale=1.0)
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.loc, dist.scale)


class GeneralizedGaussianAnchorWeightGatingTest(_AnchorWeightGatingCases, unittest.TestCase):
    family_name = "GeneralizedGaussian"

    def make_accumulator(self):
        return GeneralizedGaussianAccumulator()

    def make_estimator(self):
        return GeneralizedGaussianEstimator()

    def bulk_data(self):
        true_dist = GeneralizedGaussianDistribution(mu=BULK_OFFSET, alpha=2.0, beta=1.5)
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.mu, dist.alpha, dist.beta)

    # GeneralizedGaussianAccumulator hand-rolls its own anchor bookkeeping (it does not mix in
    # AnchoredMomentTrack), and unlike every other family in this module it was found to still
    # incorrectly activate its anchor when EVERY call it ever received carried weight exactly 0.0 --
    # there is no positively-weighted call anywhere in its history to correctly seed from, so the
    # only correct state is an anchor that stays unset (None), left for a later, real observation.
    def test_all_zero_weight_calls_never_activates_the_anchor(self):
        bulk = self.bulk_data()
        acc = self.make_accumulator()
        for x in bulk:
            acc.update(float(x), 0.0, None)
        self.assertIsNone(acc._anchor)
        # Same invariant on the seq_update()/chunk path: an entirely-zero-weight chunk carries
        # w_sum == 0.0, which must never satisfy the conditioning gate either.
        acc_seq = self.make_accumulator()
        acc_seq.seq_update(bulk, np.zeros(len(bulk)), None)
        self.assertIsNone(acc_seq._anchor)


if __name__ == "__main__":
    unittest.main()
