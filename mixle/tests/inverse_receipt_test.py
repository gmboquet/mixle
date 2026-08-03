"""``mixle.task.inverse`` record integrity: receipts, and the arrays a corrected posterior is built on.

MXR-080-1894. Deliberately torch-free and deliberately NOT registered in ``conftest.FILE_MARKERS``, so
these land in the default ``fast`` gate. ``inverse_test.py`` -- the only other coverage of
:meth:`InverseModel.posterior` -- is registered ``("stochastic", "slow")`` AND opens with
``pytest.importorskip("torch")``, which is how a defect that made *every* ``posterior()`` call raise
went unnoticed: nothing in the default gate ever called it.
"""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from mixle.inference.condition import ConditionReceipt
from mixle.task.inverse import InverseConditionReceipt, InverseModel, InverseReceipts


def _receipts(**overrides):
    kwargs = dict(
        sbc_statistic=0.1,
        sbc_pvalue=0.5,
        sbc_bins=10,
        sbc_replications=50,
        sbc_pass=True,
        coverage={0.9: 0.9},
        coverage_pass={0.9: True},
        prior_predictive={"ok": True},
        rounds_trained=1,
        ess=8.0,
        ess_ratio=0.8,
    )
    kwargs.update(overrides)
    return InverseReceipts(**kwargs)


def _corrected_model(y, particles, weights, receipts=None):
    return InverseModel(
        module=None,
        prior=None,
        simulator=lambda t: t,
        family="maf",
        theta_dim=1,
        y_dim=2,
        receipts=receipts if receipts is not None else _receipts(),
        reweighted_y=y,
        reweighted_particles=particles,
        reweighted_weights=weights,
    )


class PosteriorReceiptTest(unittest.TestCase):
    """MXR-080-1894: ``posterior()`` attached its calibration report by assigning an undeclared
    attribute onto a ``ConditionReceipt``. MXR-080-1876 froze that class, and a frozen dataclass
    refuses assignment to any name -- so every call raised ``FrozenInstanceError`` and the method was
    completely non-functional on both paths."""

    def test_the_amortized_path_returns_a_posterior_carrying_its_inverse_receipts(self):
        receipts = _receipts()
        model = InverseModel(
            module=None,
            prior=None,
            simulator=lambda t: t,
            family="maf",
            theta_dim=1,
            y_dim=2,
            receipts=receipts,
        )
        posterior = model.posterior(np.array([1.0, 2.0]))
        self.assertEqual(posterior.receipt.method, "amortized")
        self.assertIs(posterior.receipt.inverse_receipts, receipts)
        # The subclass must remain a ConditionReceipt: consumers type-test the M0 contract.
        self.assertIsInstance(posterior.receipt, ConditionReceipt)
        self.assertIsInstance(posterior.receipt, InverseConditionReceipt)

    def test_the_target_corrected_path_returns_a_posterior_carrying_its_inverse_receipts(self):
        receipts = _receipts()
        model = _corrected_model(
            np.array([1.0, 2.0]),
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            receipts=receipts,
        )
        posterior = model.posterior(np.array([1.0, 2.0]))
        self.assertEqual(posterior.receipt.method, "sir")
        self.assertIs(posterior.receipt.inverse_receipts, receipts)
        self.assertEqual(posterior.receipt.n_particles, 4)

    def test_the_receipt_is_still_frozen(self):
        # The fix declares the field; it must not re-open the record MXR-080-1876 closed.
        receipt = InverseConditionReceipt(method="amortized")
        with self.assertRaises(FrozenInstanceError):
            receipt.method = "exact"
        with self.assertRaises(FrozenInstanceError):
            receipt.inverse_receipts = _receipts()


class TargetCorrectionArrayOwnershipTest(unittest.TestCase):
    """MXR-080-1894: the constructor validated arrays it did not own.

    ``np.asarray`` does not copy an already-float array and ``reshape`` returns a view, so a caller
    that mutated its own array afterwards invalidated every check at once: a weight vector validated
    as summing to 1.0 summed to 5.75, and particles validated finite carried a NaN -- with nothing
    re-run and the corrupted state feeding ``posterior()``'s weighted sampler and mean.
    """

    def setUp(self):
        self.y = np.array([1.0, 2.0])
        self.particles = np.array([[0.0], [1.0], [2.0], [3.0]])
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.model = _corrected_model(self.y, self.particles, self.weights)

    def test_state_is_unchanged_after_the_caller_mutates_its_own_arrays(self):
        self.weights[0] = 5.0
        self.particles[0, 0] = np.nan
        self.y[0] = 99.0
        self.assertAlmostEqual(float(self.model._reweighted_weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(self.model._reweighted_particles)))
        np.testing.assert_array_equal(self.model._reweighted_y, [1.0, 2.0])

    def test_the_stored_arrays_are_not_writable_through_the_model_either(self):
        for array in (
            self.model._reweighted_y,
            self.model._reweighted_particles,
            self.model._reweighted_weights,
        ):
            with self.subTest(shape=array.shape), self.assertRaises(ValueError):
                array[0] = 0.0

    def test_the_binding_to_y_obs_survives_caller_mutation(self):
        self.y[0] = 99.0
        posterior = self.model.posterior(np.array([1.0, 2.0]))  # the original y still matches
        self.assertEqual(posterior.receipt.method, "sir")
        with self.assertRaises(ValueError):
            self.model.posterior(np.array([99.0, 2.0]))

    def test_the_legitimate_path_still_samples_and_averages(self):
        # Negative control: copying and freezing must not break the read paths that use these arrays.
        posterior = self.model.posterior(np.array([1.0, 2.0]))
        draws = posterior.sample(64, seed=0)
        self.assertEqual(np.asarray(draws).shape, (64, 1))
        self.assertTrue(set(np.asarray(draws).reshape(-1).tolist()) <= {0.0, 1.0, 2.0, 3.0})
        self.assertAlmostEqual(posterior.mean(0), 1.5, places=12)

    def test_cross_field_validation_still_rejects_a_partial_correction(self):
        with self.assertRaises(ValueError):
            InverseModel(
                module=None,
                prior=None,
                simulator=lambda t: t,
                family="maf",
                theta_dim=1,
                y_dim=2,
                receipts=_receipts(),
                reweighted_y=np.array([1.0, 2.0]),  # particles/weights missing
            )

    def test_unnormalized_weights_are_still_rejected(self):
        with self.assertRaises(ValueError):
            _corrected_model(
                np.array([1.0, 2.0]),
                np.array([[0.0], [1.0]]),
                np.array([0.9, 0.9]),
            )


class InverseReceiptsContainerOwnershipTest(unittest.TestCase):
    """MXR-080-1894: a calibration report held the caller's containers by reference."""

    def test_mutating_the_source_containers_does_not_rewrite_the_receipt(self):
        warnings = ["low ESS"]
        coverage = {0.9: 0.9}
        rounds = [{"round": 0, "rows": 100}]
        receipts = _receipts(warnings=warnings, coverage=coverage, round_training=rounds)

        warnings.append("fabricated after the fact")
        coverage[0.9] = 0.0
        rounds[0]["rows"] = 1

        self.assertEqual(receipts.warnings, ["low ESS"])
        self.assertEqual(receipts.coverage, {0.9: 0.9})
        self.assertEqual(receipts.round_training, [{"round": 0, "rows": 100}])

    def test_container_types_are_preserved(self):
        # detach, not freeze: these fields' concrete types are documented and load-bearing.
        receipts = _receipts()
        self.assertIsInstance(receipts.coverage, dict)
        self.assertIsInstance(receipts.warnings, list)
        self.assertIsInstance(receipts.round_training, list)

    def test_defaults_are_still_independent_between_instances(self):
        first, second = _receipts(), _receipts()
        first.warnings.append("only mine")
        self.assertEqual(second.warnings, [])


if __name__ == "__main__":
    unittest.main()
