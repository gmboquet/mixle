"""Regression tests: a non-positive-weight observation must never seed the shift-anchored moment
track's anchor -- the VECTOR-VALUED siblings of ``anchored_moment_track_weight_gating_test.py``.

CONFIRMED FINDING (release 0.8.0, adversarial re-review of e9b43c6b): the identical unguarded
pattern e9b43c6b fixed in the univariate accumulators -- ``if self._anchor is None:
self._activate_anchor(...)`` with no check on the seeding observation's weight, and (on the
chunk/``seq_update`` path) seeding positionally at index 0 rather than at the first
POSITIVELY-weighted element -- was independently duplicated, and left unfixed, in three more
accumulators:

* ``mixle/stats/multivariate/multivariate_gaussian.py``: ``MultivariateGaussianAccumulator.update()``
  (``if self._anchor is None: self._activate_anchor(checked)``, no weight check) and
  ``.seq_update()`` (``self._activate_anchor(checked[0])``, positional).
* ``mixle/stats/multivariate/diagonal_gaussian.py``: ``DiagonalGaussianAccumulator``, byte-for-byte
  the same unguarded shape in both methods.
* ``mixle/stats/multivariate/multivariate_student_t.py``: ``MultivariateStudentTAccumulator``'s
  shared chunk-fold helper ``_anchor_rows`` (``self._activate_anchor(rows[0])``, positional), used
  by BOTH ``update()`` and ``seq_update()``.

Directly executed against this checkout (pre-fix), a leading weight-0.0 observation at 1e14 (vs. a
bulk centered at 1.7e9 with unit spread) collapsed ``MultivariateGaussianAccumulator`` and
``DiagonalGaussianAccumulator``'s fitted covariance to the 1e-8 regularization floor in every
coordinate; the identical setup made ``MultivariateStudentTAccumulator.seq_update()`` corrupt the
fitted scale matrix badly enough that constructing the resulting distribution raised
``ValueError: ... requires a positive-definite scale matrix`` outright (a harder failure than a
silent floor-collapse). In every case, moving the SAME poisoned observation to the END of the
sequence instead reproduced the omitted-point fit exactly -- confirming the defect is specifically
about which element seeds the anchor, not whether anchoring engages at all.

MultivariateStudentTAccumulator's scalar ``update()`` was checked directly too and did NOT reproduce
the corruption: it routes through the same ``_anchor_rows`` helper as a degenerate length-1 chunk,
and that helper's own conditioning gate (``needs_vector_anchor``) already requires the chunk's
latent-weighted total to be strictly positive before it will ever call ``_activate_anchor`` -- for a
single observation that total IS that observation's own weight, so a non-positive-weight scalar call
was already correctly refused before this fix touched anything. Only the multi-row ``seq_update()``/
chunk path (where a poisoned row can sit alongside other, real, positively-weighted rows in the same
chunk) was vulnerable, and only that path's fix changes any observable behavior. See
``MultivariateStudentTAnchorWeightGatingTest`` below: its scalar-``update()`` case is still included,
for the same symmetry the univariate suite keeps, but it passes unchanged both before and after the
fix -- the finding is confirmed for the chunk path only.

Fixed the same way as e9b43c6b: anchor activation is gated on the seeding element's weight being
strictly ``> 0.0``; on the chunk path, the anchor seeds from the FIRST element whose weight is
positive rather than positionally at index/row 0, and when no element qualifies the anchor is left
unset for a later call.

Mirrors the structure and the four core properties of ``anchored_moment_track_weight_gating_test.py``
(leading poison matches omission via both update() and seq_update(), trailing poison always matched
omission, an ordinary positive-weight leading outlier still seeds unconditionally, and a positive but
negligible weight of 1e-300 still legitimately seeds), adapted for vector-valued observations: fitted
parameters are mean vectors and covariance/shape matrices compared elementwise, and the poisoned/
control observation is a length-``DIM`` vector rather than a scalar.
"""

import unittest

import numpy as np

from mixle.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianAccumulator,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianAccumulator,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_student_t import (
    MultivariateStudentTAccumulator,
    MultivariateStudentTDistribution,
    MultivariateStudentTEstimator,
)

DIM = 3

# A magnitude every family's accumulator accepts unconditionally (finite) and wildly different from
# BULK_OFFSET, so choosing it as the anchor -- instead of a real bulk point -- reintroduces
# large-offset cancellation a tight-tolerance comparison unambiguously catches.
EXTREME_X = np.full(DIM, 1.0e14)

# The location every family's "bulk" (legitimate, always-positive-weight) data is centered at,
# matching the scalar suite's BULK_OFFSET: mean^2/spread^2 here is far past every family's own
# ANCHOR_CONDITION_RATIO, so needs_(vector_)anchor() activates the anchor mechanism from the bulk
# data alone on the seq_update/chunk path -- the leading/trailing tests below exercise WHICH element
# gets chosen as the anchor, not merely whether anchoring turns on.
BULK_OFFSET = np.full(DIM, 1.7e9)

TOLERANCE = 1.0e-9


def _relative_close(a, b, tol: float = TOLERANCE) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)))


class _VectorAnchorWeightGatingCases:
    """Shared test bodies, mixed into one ``unittest.TestCase`` per family below.

    Not itself a ``TestCase`` (no ``test_*`` method here runs on its own) so unittest discovery does
    not try to collect this mixin directly; each concrete subclass supplies the family-specific
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
        """``n_bulk`` legitimate observations, shape ``(n_bulk, DIM)``, centered at ``BULK_OFFSET``."""
        raise NotImplementedError

    def fitted_params(self, dist) -> tuple:
        """The arrays a fit should be compared on (mean vector, covariance/shape matrix)."""
        raise NotImplementedError

    # -- shared plumbing ----------------------------------------------------------------------

    def _fit_scalar(self, xs, weights):
        acc = self.make_accumulator()
        for x, w in zip(xs, weights):
            acc.update(x, float(w), None)
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
                "%s: fitted params not within tolerance (%s)\n  a=%r\n  b=%r" % (self.family_name, msg, a, b),
            )

    def _assert_anchor_equal(self, anchor, expected_row):
        self.assertIsNotNone(anchor)
        np.testing.assert_array_equal(np.asarray(anchor, dtype=np.float64), np.asarray(expected_row, dtype=np.float64))

    # -- 1. leading weight-0.0 extreme observation must vanish, scalar update() path ----------

    def test_leading_zero_weight_extreme_observation_matches_omission_via_scalar_update(self):
        bulk = self.bulk_data()
        _, without = self._fit_scalar(bulk, np.ones(len(bulk)))
        xs = np.vstack([EXTREME_X[None, :], bulk])
        weights = np.concatenate([[0.0], np.ones(len(bulk))])
        acc_with, with_poison = self._fit_scalar(xs, weights)
        self._assert_params_match(with_poison, without, "scalar update(), leading weight-0.0 outlier")
        # The anchor actually activated on the first REAL (positive-weight) observation, never on
        # the poisoned one -- pins the mechanism, not just its downstream numerical effect.
        self._assert_anchor_equal(acc_with._anchor, bulk[0])

    # -- 2. leading weight-0.0 extreme observation must vanish, seq_update()/chunk path --------

    def test_leading_zero_weight_extreme_observation_matches_omission_via_seq_update(self):
        bulk = self.bulk_data()
        _, without = self._fit_seq(bulk, np.ones(len(bulk)))
        xs = np.vstack([EXTREME_X[None, :], bulk])
        weights = np.concatenate([[0.0], np.ones(len(bulk))])
        acc_with, with_poison = self._fit_seq(xs, weights)
        self._assert_params_match(with_poison, without, "seq_update(), leading weight-0.0 outlier")
        self._assert_anchor_equal(acc_with._anchor, bulk[0])

    # -- 3. the SAME extreme point at the END of the sequence (seq_update) == omitting it ------

    def test_trailing_zero_weight_extreme_observation_via_seq_update_matches_omission(self):
        bulk = self.bulk_data()
        _, without = self._fit_seq(bulk, np.ones(len(bulk)))
        xs = np.vstack([bulk, EXTREME_X[None, :]])
        weights = np.concatenate([np.ones(len(bulk)), [0.0]])
        acc_with, with_poison = self._fit_seq(xs, weights)
        self._assert_params_match(with_poison, without, "seq_update(), trailing weight-0.0 outlier")
        self._assert_anchor_equal(acc_with._anchor, bulk[0])

    # -- 4. ordinary anchoring is NOT broken: a NORMAL-weight leading outlier still anchors -----

    def test_leading_extreme_observation_with_normal_weight_still_seeds_the_anchor_via_update(self):
        acc = self.make_accumulator()
        acc.update(EXTREME_X, 1.0, None)  # the very first call: unconditional activation, unchanged
        self._assert_anchor_equal(acc._anchor, EXTREME_X)

    def test_leading_extreme_observation_with_normal_weight_still_seeds_the_anchor_via_seq_update(self):
        # No synthetic poison here: the bulk data is ALREADY centered at BULK_OFFSET (~1.7e9), so
        # relative to this family's ordinary small-magnitude parameters, bulk[0] with an everyday
        # weight of 1.0 (leading, all weights positive -- the completely ordinary, unpoisoned case)
        # IS the "extreme leading point with a normal weight" scenario: needs_(vector_)anchor()
        # reliably trips on the offset alone, and the fix must still let this ordinary first real
        # observation seed the anchor.
        bulk = self.bulk_data()
        acc, _ = self._fit_seq(bulk, np.ones(len(bulk)))
        self._assert_anchor_equal(acc._anchor, bulk[0])

    def test_negligible_but_still_positive_weight_seeds_the_anchor(self):
        # Pins the exact gate boundary: the fix is `weight > 0.0`, not an epsilon threshold, so a
        # positive-but-vanishingly-small weight (1e-300, distinct from underflowing all the way to
        # exactly 0.0) is legitimate and must still be allowed to seed the anchor.
        acc = self.make_accumulator()
        acc.update(EXTREME_X, 1.0e-300, None)
        self._assert_anchor_equal(acc._anchor, EXTREME_X)


class MultivariateGaussianAnchorWeightGatingTest(_VectorAnchorWeightGatingCases, unittest.TestCase):
    family_name = "MultivariateGaussian"

    def make_accumulator(self):
        return MultivariateGaussianAccumulator(dim=DIM)

    def make_estimator(self):
        return MultivariateGaussianEstimator(dim=DIM)

    def bulk_data(self):
        true_dist = MultivariateGaussianDistribution(mu=BULK_OFFSET, covar=np.eye(DIM))
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.mu, dist.covar)


class DiagonalGaussianAnchorWeightGatingTest(_VectorAnchorWeightGatingCases, unittest.TestCase):
    family_name = "DiagonalGaussian"

    def make_accumulator(self):
        return DiagonalGaussianAccumulator(dim=DIM)

    def make_estimator(self):
        return DiagonalGaussianEstimator(dim=DIM)

    def bulk_data(self):
        true_dist = DiagonalGaussianDistribution(mu=BULK_OFFSET, covar=np.ones(DIM))
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.mu, dist.covar)


class MultivariateStudentTAnchorWeightGatingTest(_VectorAnchorWeightGatingCases, unittest.TestCase):
    family_name = "MultivariateStudentT"
    dof = 5.0

    def make_accumulator(self):
        return MultivariateStudentTAccumulator(dof=self.dof, dim=DIM)

    def make_estimator(self):
        return MultivariateStudentTEstimator(dof=self.dof, dim=DIM)

    def bulk_data(self):
        true_dist = MultivariateStudentTDistribution(dof=self.dof, loc=BULK_OFFSET, shape=np.eye(DIM))
        return np.asarray(true_dist.sampler(seed=self.seed).sample(self.n_bulk), dtype=np.float64)

    def fitted_params(self, dist):
        return (dist.mu, dist.shape)


if __name__ == "__main__":
    unittest.main()
