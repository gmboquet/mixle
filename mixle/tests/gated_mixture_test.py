"""GatedMixtureDistribution (mixle.stats.latent.gated_mixture): a mixture-of-experts whose weights are a
learned gate p(k|z), not constants. Recovers experts + a z-dependent routing, and beats a plain (fixed-
weight) mixture when the mixing genuinely depends on the covariate."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.latent.gated_mixture import (
    GateBufferReceipt,
    GatedMixtureDistribution,
    GatedMixtureEstimator,
    GateOptimizationReceipt,
    SoftmaxGate,
)


def _switching_data(seed, n=1000):
    # z in [-3,3]; sign of z selects which expert generates y (N(-5,1) vs N(+5,1))
    rng = np.random.RandomState(seed)
    z = rng.uniform(-3, 3, n)
    y = np.where(z < 0, rng.normal(-5, 1, n), rng.normal(5, 1, n))
    return list(zip(z.reshape(-1, 1).tolist(), y.tolist())), z, y


def _proto():
    return GatedMixtureDistribution(
        [st.GaussianDistribution(-1.0, 1.0), st.GaussianDistribution(1.0, 1.0)],
        SoftmaxGate.zeros(2, 1),
    )


class GatedMixtureTest(unittest.TestCase):
    def test_recovers_experts_and_a_covariate_dependent_gate(self):
        data, _, _ = _switching_data(0)
        fit = optimize(data, _proto().estimator(), rng=np.random.RandomState(0), max_its=30, out=None)
        means = sorted(c.mu for c in fit.components)
        self.assertAlmostEqual(means[0], -5.0, delta=0.5)
        self.assertAlmostEqual(means[1], 5.0, delta=0.5)
        # the gate routes opposite experts at z=-2 vs z=+2
        lp_neg = np.exp(fit.gate.log_prob_batch(np.array([[-2.0]]))[0])
        lp_pos = np.exp(fit.gate.log_prob_batch(np.array([[2.0]]))[0])
        self.assertNotEqual(int(np.argmax(lp_neg)), int(np.argmax(lp_pos)))
        self.assertGreater(lp_neg.max(), 0.9)
        self.assertGreater(lp_pos.max(), 0.9)

    def test_beats_a_plain_fixed_weight_mixture_on_gated_data(self):
        data, _, _ = _switching_data(1)
        gated = optimize(data, _proto().estimator(), rng=np.random.RandomState(1), max_its=30, out=None)
        # a plain mixture over y alone cannot use z; fit it on the y column for a fair likelihood comparison
        ys = [row[1] for row in data]
        plain = st.MixtureDistribution(
            [st.GaussianDistribution(-1.0, 1.0), st.GaussianDistribution(1.0, 1.0)], [0.5, 0.5]
        )
        plain_fit = optimize(ys, plain.estimator(), rng=np.random.RandomState(1), max_its=30, out=None)

        ll_gated = float(np.sum(gated.seq_log_density(gated.dist_to_encoder().seq_encode(data))))
        ll_plain = float(np.sum(plain_fit.seq_log_density(plain_fit.dist_to_encoder().seq_encode(ys))))
        # the gate contributes log p(k|z) information the fixed weights cannot -- a large, real gap
        self.assertGreater(ll_gated, ll_plain + 100.0)

    def test_seq_and_scalar_log_density_agree(self):
        data, _, _ = _switching_data(2, n=40)
        d = _proto()
        seq = d.seq_log_density(d.dist_to_encoder().seq_encode(data))
        scalar = np.array([d.log_density(x) for x in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sample_given_respects_the_gate(self):
        data, _, _ = _switching_data(3)
        fit = optimize(data, _proto().estimator(), rng=np.random.RandomState(3), max_its=30, out=None)
        s = fit.sampler(0)
        # at strongly-negative z, samples cluster near the negative expert; at positive z, near the positive one
        neg = np.array([s.sample_given([-2.5]) for _ in range(200)])
        pos = np.array([s.sample_given([2.5]) for _ in range(200)])
        self.assertLess(neg.mean(), 0.0)
        self.assertGreater(pos.mean(), 0.0)

    def test_requires_at_least_two_experts(self):
        with self.assertRaises(ValueError):
            GatedMixtureDistribution([st.GaussianDistribution(0.0, 1.0)], SoftmaxGate.zeros(1, 1))

    def test_gate_class_count_must_match_experts(self):
        with self.assertRaises(ValueError):
            GatedMixtureDistribution(
                [st.GaussianDistribution(-1.0, 1.0), st.GaussianDistribution(1.0, 1.0)],
                SoftmaxGate.zeros(3, 1),  # 3 gate classes, 2 experts
            )

    def test_gate_parameters_and_fit_controls_are_validated_and_owned(self):
        weight = np.zeros((2, 1))
        bias = np.zeros(2)
        gate = SoftmaxGate(weight, bias)
        weight[:] = 99.0
        bias[:] = 99.0
        np.testing.assert_array_equal(gate.weight, np.zeros((2, 1)))
        np.testing.assert_array_equal(gate.bias, np.zeros(2))

        invalid_parameters = [
            (np.zeros(2), np.zeros(2)),
            (np.zeros((2, 1)), np.zeros(3)),
            (np.asarray([[np.nan], [0.0]]), np.zeros(2)),
        ]
        for invalid_weight, invalid_bias in invalid_parameters:
            with self.subTest(weight=repr(invalid_weight), bias=repr(invalid_bias)):
                with self.assertRaises(ValueError):
                    SoftmaxGate(invalid_weight, invalid_bias)

        z = np.asarray([[-1.0], [1.0]])
        r = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        for kwargs in (
            {"steps": 0},
            {"steps": 1.5},
            {"lr": 0.0},
            {"lr": np.inf},
            {"tol": -1.0},
        ):
            with self.subTest(kwargs=repr(kwargs)):
                with self.assertRaises((TypeError, ValueError)):
                    gate.fit(z, r, **kwargs)

        for kwargs in (
            {"gate_steps": 0},
            {"gate_steps": 1.5},
            {"gate_lr": 0.0},
            {"gate_lr": np.nan},
            {"gate_tol": -1.0},
            {"max_buffer_rows": 0},
        ):
            with self.subTest(estimator_kwargs=repr(kwargs)):
                with self.assertRaises((TypeError, ValueError)):
                    GatedMixtureEstimator(
                        [st.GaussianEstimator(), st.GaussianEstimator()],
                        gate,
                        **kwargs,
                    )

        fitted, receipt = gate.fit_with_receipt(z, r, steps=200, lr=0.1)
        self.assertIsInstance(receipt, GateOptimizationReceipt)
        self.assertIs(fitted.fit_receipt, receipt)
        self.assertGreater(receipt.steps_completed, 0)
        self.assertLessEqual(receipt.final_loss, receipt.initial_loss)
        self.assertTrue(np.isfinite(fitted.log_prob_batch(z)).all())

    def test_gate_and_encoder_reject_invalid_covariate_geometry(self):
        model = _proto()
        encoder = model.dist_to_encoder()
        empty = encoder.seq_encode([])
        self.assertEqual(empty[0].shape, (0, 1))
        np.testing.assert_array_equal(model.seq_log_density(empty), np.zeros(0))
        for data in (
            [([np.nan], 0.0)],
            [([1.0, 2.0], 0.0)],
            [([1.0], 0.0), ([1.0, 2.0], 1.0)],
        ):
            with self.subTest(data=repr(data)):
                with self.assertRaises((TypeError, ValueError)):
                    encoder.seq_encode(data)

        class InvalidGate:
            n_classes = 2
            n_features = 1

            def log_prob_batch(self, z):
                return np.zeros((len(z), 2))

        with self.assertRaisesRegex(ValueError, "sum to one"):
            GatedMixtureDistribution(
                [st.GaussianDistribution(-1.0, 1.0), st.GaussianDistribution(1.0, 1.0)],
                InvalidGate(),
            )

    def test_impossible_evidence_has_zero_responsibility(self):
        model = GatedMixtureDistribution(
            [
                st.CategoricalDistribution({"a": 1.0}),
                st.CategoricalDistribution({"b": 1.0}),
            ],
            SoftmaxGate.zeros(2, 1),
        )
        observation = ([0.0], "outside")
        np.testing.assert_array_equal(model.posterior(observation), [0.0, 0.0])
        self.assertEqual(model.log_density(observation), -np.inf)

        enc = model.dist_to_encoder().seq_encode([observation])
        accumulator = model.estimator().accumulator_factory().make()
        accumulator.seq_update(enc, np.ones(1), model)
        comp_stats, z, responsibilities, receipt = accumulator.value()
        self.assertEqual(comp_stats, ({}, {}))
        np.testing.assert_array_equal(responsibilities, [[0.0, 0.0]])
        self.assertEqual(receipt.rows_seen, 1)

    def test_gate_training_buffer_is_bounded_mergeable_and_owned(self):
        model = _proto()
        estimator = GatedMixtureEstimator(
            [st.GaussianEstimator(), st.GaussianEstimator()],
            model.gate,
            gate_steps=20,
            max_buffer_rows=3,
        )
        data = [([float(index)], float(index % 2)) for index in range(10)]
        enc = model.dist_to_encoder().seq_encode(data)

        first = estimator.accumulator_factory().make()
        first.seq_update(enc, np.ones(10), model)
        value = first.value()
        self.assertEqual(value[1].shape, (3, 1))
        self.assertEqual(value[2].shape, (3, 2))
        self.assertEqual(value[3], GateBufferReceipt(10, 3, 7, 3))

        value[1][:] = 999.0
        value[2][:] = 999.0
        self.assertFalse(np.any(first.value()[1] == 999.0))
        self.assertFalse(np.any(first.value()[2] == 999.0))

        second = estimator.accumulator_factory().make()
        second.seq_update(enc, np.ones(10), model)
        first.combine(second.value())
        combined = first.value()
        self.assertEqual(combined[1].shape, (3, 1))
        self.assertEqual(combined[3], GateBufferReceipt(20, 3, 17, 3))

        fitted = estimator.estimate(20.0, combined)
        self.assertIsInstance(fitted.gate_fit_receipt, GateOptimizationReceipt)
        self.assertEqual(fitted.gate_buffer_receipt, combined[3])
        self.assertIs(estimator.last_gate_fit_receipt, fitted.gate_fit_receipt)


if __name__ == "__main__":
    unittest.main()
