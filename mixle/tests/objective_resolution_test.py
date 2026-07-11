"""The prior is the single switch: optimize()/fit()/best_of() auto-select the objective.

With ``objective='auto'`` (the default) a frequentist estimator is fit by maximum likelihood, an
estimator carrying a conjugate prior by MAP (penalized LL), and a variational model (one exposing
``seq_local_elbo``, e.g. a DPM) by the variational ELBO -- regardless of which verb the caller uses.
An explicit ``objective='mle'|'map'|'vb'`` overrides the auto-detection.
"""

import io
import unittest

import numpy as np

from mixle.inference.estimation import _resolve_objective, fit, optimize
from mixle.stats.bayes.dirichlet_process_mixture import DirichletProcessMixtureEstimator
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator


def _gaussian_prior():
    return NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)


class ObjectiveResolutionTestCase(unittest.TestCase):
    def test_resolve_auto(self):
        # no prior -> mle
        self.assertEqual(_resolve_objective("auto", GaussianEstimator(), GaussianDistribution(0.0, 1.0)), "mle")
        # conjugate prior -> map
        gp = GaussianEstimator(prior=_gaussian_prior())
        gm = GaussianDistribution(0.0, 1.0, prior=_gaussian_prior())
        self.assertEqual(_resolve_objective("auto", gp, gm), "map")
        # variational model (has seq_local_elbo) -> vb; build a real DPM via one fit iteration
        dpm_est = DirichletProcessMixtureEstimator([GaussianEstimator(prior=_gaussian_prior()) for _ in range(4)])
        data = list(np.random.RandomState(0).normal(0.0, 1.0, 50))
        m = fit(data, dpm_est, max_its=1, delta=None, rng=np.random.RandomState(0), out=None)
        self.assertTrue(hasattr(m, "seq_local_elbo"))
        self.assertEqual(_resolve_objective("auto", dpm_est, m), "vb")

    def test_explicit_overrides_and_validation(self):
        gp = GaussianEstimator(prior=_gaussian_prior())
        gm = GaussianDistribution(0.0, 1.0)
        for o in ("mle", "map", "vb"):
            self.assertEqual(_resolve_objective(o, gp, gm), o)
        with self.assertRaises(ValueError):
            _resolve_objective("nope", gp, gm)

    def test_optimize_matches_fit_for_conjugate_leaf(self):
        # optimize() (auto -> map) and fit() give the same MAP point estimate for a conjugate Gaussian.
        rng = np.random.RandomState(3)
        data = list(rng.normal(2.0, 1.5, 400))
        prior = _gaussian_prior()
        m_opt = optimize(data, GaussianEstimator(prior=prior), max_its=5, out=None)
        m_fit = fit(data, GaussianEstimator(prior=prior), max_its=5, delta=None, out=None)
        self.assertAlmostEqual(m_opt.mu, m_fit.mu, places=10)
        self.assertAlmostEqual(m_opt.sigma2, m_fit.sigma2, places=10)
        self.assertIsInstance(m_opt.get_prior(), NormalGammaDistribution)

    def test_optimize_auto_uses_elbo_for_dpm(self):
        # The progress line is labeled by the resolved objective; auto on a DPM -> ELBO, override mle -> LL.
        rng = np.random.RandomState(0)
        data = list(np.concatenate([rng.normal(-8, 1, 200), rng.normal(8, 1, 200)]))
        mk = lambda: DirichletProcessMixtureEstimator([GaussianEstimator(prior=_gaussian_prior()) for _ in range(6)])

        b_auto = io.StringIO()
        optimize(data, mk(), max_its=4, delta=None, rng=np.random.RandomState(1), out=b_auto)
        self.assertIn("ELBO=", b_auto.getvalue())

        b_mle = io.StringIO()
        optimize(data, mk(), max_its=4, delta=None, objective="mle", rng=np.random.RandomState(1), out=b_mle)
        self.assertIn("ln[p_mat(Data|Model)]=", b_mle.getvalue())
        self.assertNotIn("ELBO=", b_mle.getvalue())

    def test_mle_path_unchanged_no_prior(self):
        # No prior anywhere: auto resolves to mle, optimize keeps the historical likelihood label.
        rng = np.random.RandomState(2)
        data = list(rng.normal(0.0, 1.0, 200))
        b = io.StringIO()
        optimize(data, GaussianEstimator(), max_its=2, delta=None, out=b)
        self.assertIn("ln[p_mat(Data|Model)]=", b.getvalue())
        self.assertNotIn("ELBO=", b.getvalue())

    def test_strategy_receives_the_same_resolved_map_objective_as_outer_loop(self):
        data = list(np.random.RandomState(4).normal(1.0, 0.5, 30))
        prior = _gaussian_prior()
        estimator = GaussianEstimator(prior=prior)
        initial = GaussianDistribution(0.0, 1.0, prior=prior)

        class CaptureStrategy:
            def __init__(self):
                self.value = None

            def step(self, enc_data, estimator, model, engine=None, objective=None):
                from mixle.inference.em import EMStepResult

                self.value = objective(model)
                return EMStepResult(model, self.value)

        strategy = CaptureStrategy()
        optimize(
            data,
            estimator,
            prev_estimate=initial,
            max_its=1,
            strategy=strategy,
            objective="map",
            out=None,
        )
        enc = initial.dist_to_encoder().seq_encode(data)
        expected = float(initial.seq_log_density(enc).sum()) + estimator.model_log_density(initial)
        self.assertAlmostEqual(strategy.value, expected, places=12)

    def test_optimize_honors_an_explicit_strategy_rejection(self):
        data = [0.0, 0.1, -0.1]
        initial = GaussianDistribution(0.0, 1.0)

        class RejectStrategy:
            def step(self, enc_data, estimator, model, engine=None, objective=None):
                from mixle.inference.em import EMStepResult

                better_but_rejected = GaussianDistribution(0.0, 0.01)
                return EMStepResult(better_but_rejected, objective(better_but_rejected), False)

        fitted = optimize(
            data,
            GaussianEstimator(),
            prev_estimate=initial,
            strategy=RejectStrategy(),
            max_its=1,
            reuse_estep_ll=False,
            out=None,
        )
        self.assertIs(fitted, initial)


if __name__ == "__main__":
    unittest.main()
