"""Occasional missing entries by marginalization (MAR), and posteriors/imputation over the full model.

A missing field is integrated out of the likelihood (it contributes log-density 0 and no sufficient
statistics), so EM fits each field from its present rows only; for a mixture over composites, conditioning
on the present fields yields the posterior over the missing ones (imputation).
"""

import io
import pickle
import unittest

import numpy as np

from pysp.inference import optimize
from pysp.stats import (
    MISSING,
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
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


if __name__ == "__main__":
    unittest.main()
