"""Occasional missing entries by marginalization (MAR), and posteriors/imputation over the full model.

A missing field is integrated out of the likelihood (it contributes log-density 0 and no sufficient
statistics), so EM fits each field from its present rows only; for a mixture over composites, conditioning
on the present fields yields the posterior over the missing ones (imputation).
"""

import importlib.util
import io
import pickle
import unittest

import numpy as np

_HAS_TORCH = importlib.util.find_spec("torch") is not None

from mixle.inference import estimate, optimize
from mixle.stats import (
    MISSING,
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    composite_with_missing,
    marginalized,
)


class SentinelTest(unittest.TestCase):
    def test_missing_is_a_pickle_stable_singleton(self):
        self.assertIs(pickle.loads(pickle.dumps(MISSING)), MISSING)
        self.assertEqual(repr(MISSING), "MISSING")


class MarginalizationTest(unittest.TestCase):
    def setUp(self):
        self.d = composite_with_missing([GaussianDistribution(3.0, 1.0), CategoricalDistribution({"x": 0.7, "y": 0.3})])

    def test_missing_field_is_marginalized_out(self):
        g_only = GaussianDistribution(3.0, 1.0).log_density(3.0)
        self.assertAlmostEqual(self.d.log_density((3.0, MISSING)), g_only, places=12)  # cat marginalized
        self.assertEqual(self.d.log_density((MISSING, MISSING)), 0.0)  # both marginalized
        self.assertAlmostEqual(
            self.d.log_density((3.0, "x")),
            g_only + CategoricalDistribution({"x": 0.7, "y": 0.3}).log_density("x"),
            places=12,
        )

    def test_em_fits_from_present_rows_only(self):
        true = CompositeDistribution((GaussianDistribution(3.0, 1.0), CategoricalDistribution({"x": 0.7, "y": 0.3})))
        rng = np.random.RandomState(1)
        data = []
        for g, c in true.sampler(0).sample(4000):  # 40% MCAR per field
            data.append((MISSING if rng.rand() < 0.4 else g, MISSING if rng.rand() < 0.4 else c))
        est = composite_with_missing([GaussianDistribution(0.0, 1.0), CategoricalDistribution({"x": 0.5, "y": 0.5})])
        m = optimize(data, est.estimator(), max_its=30, rng=np.random.RandomState(2), out=io.StringIO())
        self.assertAlmostEqual(m.dists[0].dist.mu, 3.0, delta=0.15)
        self.assertAlmostEqual(m.dists[0].dist.sigma2, 1.0, delta=0.2)
        self.assertAlmostEqual(np.exp(m.dists[1].dist.log_density("x")), 0.7, delta=0.05)


class CompositeMarginalConditionTest(unittest.TestCase):
    def setUp(self):
        self.c = CompositeDistribution([GaussianDistribution(0, 1), CategoricalDistribution({"x": 0.6, "y": 0.4})])

    def test_marginal_subcomposite(self):
        self.assertEqual([type(d).__name__ for d in self.c.marginal([1]).dists], ["CategoricalDistribution"])
        self.assertEqual([type(d).__name__ for d in self.c.marginal([0, 1]).dists], self.c_names())

    def c_names(self):
        return ["GaussianDistribution", "CategoricalDistribution"]

    def test_condition_drops_observed(self):
        cond = self.c.condition({0: 3.0})  # independence => conditional is the unobserved factor unchanged
        self.assertEqual([type(d).__name__ for d in cond.dists], ["CategoricalDistribution"])


class MixtureImputationTest(unittest.TestCase):
    def setUp(self):
        self.mix = MixtureDistribution(
            [
                CompositeDistribution([GaussianDistribution(-2, 1), CategoricalDistribution({"x": 0.8, "y": 0.2})]),
                CompositeDistribution([GaussianDistribution(2, 1), CategoricalDistribution({"x": 0.2, "y": 0.8})]),
            ],
            [0.5, 0.5],
        )

    def test_numeric_observed_imputes_categorical(self):
        cond = self.mix.conditional({0: 2.0})  # observe the Gaussian field -> posterior over the categorical
        pcat = sum(
            cond.w[k] * np.array([np.exp(cond.components[k].dists[0].log_density(v)) for v in ["x", "y"]])
            for k in range(2)
        )
        self.assertGreater(pcat[1], pcat[0])  # x0=2 favors component 2 (cat y:0.8)
        self.assertAlmostEqual(pcat.sum(), 1.0, places=9)

    def test_heterogeneous_observed_imputes_gaussian(self):
        cond = self.mix.conditional({1: "x"})  # observe the CATEGORICAL field -> posterior over the Gaussian
        np.testing.assert_allclose(cond.w, [0.8, 0.2], atol=1e-9)  # cat=x favors component 1
        e_x0 = sum(w * c.dists[0].mu for w, c in zip(cond.w, cond.components))
        self.assertLess(e_x0, 0.0)  # pulled toward component 1's mean (-2)


class HmmMissingEmissionsTest(unittest.TestCase):
    """A sequence model with occasional missing emissions: wrap the per-state emissions in marginalized()
    and pass MISSING for absent observations. Missing positions become uninformative in forward-backward
    (no HMM code change); EM fits from present observations and latent_posterior smooths over the gaps."""

    def _hmm(self, optional):
        emits = [{"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.1, "b": 0.3, "c": 0.6}]
        topics = [(marginalized(CategoricalDistribution(e)) if optional else CategoricalDistribution(e)) for e in emits]
        return HiddenMarkovModelDistribution(
            topics,
            w=[0.6, 0.4],
            transitions=[[0.8, 0.2], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(4, [1.0]),
        )

    def test_marginalized_emissions_equal_plain_when_nothing_missing(self):
        plain, opt = self._hmm(False), self._hmm(True)
        seqs = [["a", "b", "c", "a"], ["c", "c", "a", "b"], ["b", "a", "a", "c"]]
        for s in seqs:
            self.assertAlmostEqual(opt.log_density(s), plain.log_density(s), places=12)

    def test_partial_sequence_scores_finite_and_smooths(self):
        opt = self._hmm(True)
        self.assertTrue(np.isfinite(opt.log_density(["a", MISSING, "c", MISSING])))
        post = opt.latent_posterior(["a", MISSING, "c", MISSING])  # smoothed state path given present obs
        z = post.sample(np.random.RandomState(0))
        self.assertEqual(len(z), 4)

    def test_em_recovers_from_partial_sequences(self):
        true = self._hmm(False)
        rng = np.random.RandomState(1)
        data = [[MISSING if rng.rand() < 0.3 else e for e in seq] for seq in true.sampler(0).sample(5000)]
        m = self._hmm(True)  # start from truth to isolate the missing mechanism from the HMM init-saddle
        for _ in range(12):
            m = estimate(data, self._hmm(True).estimator(), m)
        em = [np.array([np.exp(t.dist.log_density(k)) for k in ["a", "b", "c"]]) for t in m.topics]
        np.testing.assert_allclose(em[0], [0.7, 0.2, 0.1], atol=0.05)
        np.testing.assert_allclose(em[1], [0.1, 0.3, 0.6], atol=0.05)
        np.testing.assert_allclose(m.transitions, [[0.8, 0.2], [0.3, 0.7]], atol=0.06)


class PplMissingTest(unittest.TestCase):
    """The PPL fits from data with NaNs by marginalizing them out (no imputation): the mode/posterior is
    over the present entries only. fit(..., missing='marginalize') on the EM path."""

    def _data(self):
        rng = np.random.RandomState(0)
        clean = rng.normal(3.0, 2.0, size=4000)
        return [float("nan") if rng.rand() < 0.3 else float(x) for x in clean]

    def test_marginalize_matches_mle_over_present(self):
        from mixle.ppl import Normal, free

        data = self._data()
        m = Normal(free, free).fit(data, missing="marginalize")
        present = np.array([x for x in data if not np.isnan(x)])
        self.assertAlmostEqual(m._dist.mu, present.mean(), places=4)  # fit == MLE over present (no imputation)
        self.assertAlmostEqual(m._dist.sigma2, present.var(), places=3)

    def test_default_rejects_nan(self):
        from mixle.ppl import Normal, free

        with self.assertRaises(Exception):  # noqa: B017 -- the support check rejects NaN with a bare Exception
            Normal(free, free).fit(self._data())

    @unittest.skipUnless(_HAS_TORCH, "full-Bayesian marginalization needs the torch autograd target")
    def test_full_bayesian_posterior_with_missing(self):
        # MAP/VI (and the samplers) marginalize NaNs through the autograd target -> posterior over the
        # present data, no imputation. Weak prior, so the location posterior ~ the present-data mean.
        from mixle.ppl import Normal

        rng = np.random.RandomState(2)
        clean = rng.normal(3.0, 2.0, size=1500)
        data = [float("nan") if rng.rand() < 0.3 else float(x) for x in clean]
        present_mean = np.mean([x for x in data if not np.isnan(x)])
        for how in ("map", "vi"):
            m = Normal(Normal(0.0, 5.0), 2.0).fit(data, how=how, missing="marginalize")
            self.assertAlmostEqual(m._dist.mu, present_mean, delta=0.1)

    def test_unsupported_how_raises(self):
        from mixle.ppl import Normal

        with self.assertRaises(NotImplementedError):  # closed-form conjugate path isn't wired for missing
            Normal(Normal(0.0, 5.0), 2.0).fit(self._data(), how="conjugate", missing="marginalize")


if __name__ == "__main__":
    unittest.main()
