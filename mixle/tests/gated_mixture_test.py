"""GatedMixtureDistribution (mixle.stats.latent.gated_mixture): a mixture-of-experts whose weights are a
learned gate p(k|z), not constants. Recovers experts + a z-dependent routing, and beats a plain (fixed-
weight) mixture when the mixing genuinely depends on the covariate."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.latent.gated_mixture import GatedMixtureDistribution, SoftmaxGate


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


if __name__ == "__main__":
    unittest.main()
