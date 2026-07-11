"""Regression tests for the EM driver's non-finite log-likelihood guard (WS-L P5).

A collapsed/singular covariance can make an EM step's data log-likelihood NaN or -inf.
The shared EM loops (:func:`mixle.inference.estimation._em_loop` and ``_fused_em_loop``) must
never *accept* such a step and must not let the non-finite value poison the convergence
reference (which would stall every subsequent iteration on NaN comparisons). These tests
pin that behavior with deterministic stand-in steps, and a high-dimensional diagonal Gaussian
mixture smoke test exercises the bundled ``robust=True`` path end-to-end (the reported
"mixture of Gaussians on digits is unstable" scenario).
"""

import unittest

import numpy as np

from mixle.inference.estimation import _em_loop, _resolve_monotone, _resolve_track_best, optimize
from mixle.stats import DiagonalGaussianEstimator, MixtureEstimator


class _Model:
    """Opaque stand-in model carrying a tag and controlled log-likelihood."""

    def __init__(self, tag: str, ll: float) -> None:
        self.tag = tag
        self.ll = ll


def _ll_fn(_enc, model):
    return (1, model.ll)


def _steps(sequence):
    """Return a ``step_fn`` that yields the given models in order."""
    it = iter(sequence)

    def step_fn(_enc, _est, _model):
        return next(it)

    return step_fn


class EmNonFiniteGuardTest(unittest.TestCase):
    def test_auto_policy_is_strict_for_exact_updates_and_best_seen_for_variational_updates(self):
        exact = _Model("exact", 0.0)
        self.assertTrue(_resolve_monotone(None, None, exact))

        variational = _Model("variational", 0.0)
        variational.seq_local_elbo = lambda _enc: np.array([0.0])
        self.assertFalse(_resolve_monotone(None, None, variational))
        self.assertTrue(_resolve_monotone(True, None, variational))

        class SurrogateEstimator:
            outer_objective_compatible = False

        surrogate = SurrogateEstimator()
        self.assertFalse(_resolve_monotone(None, surrogate, exact))
        self.assertFalse(_resolve_track_best(None, surrogate))
        self.assertTrue(_resolve_track_best(True, surrogate))

    def test_best_seen_policy_can_cross_a_temporary_objective_valley(self):
        init = _Model("init", 1.0)
        valley = _Model("valley", 0.0)
        peak = _Model("peak", 2.0)
        chosen, score = _em_loop(
            None,
            None,
            init,
            _steps([valley, peak]),
            _ll_fn,
            max_its=2,
            delta=None,
            out=None,
            monotone=False,
        )
        self.assertIs(chosen, peak)
        self.assertEqual(score, 2.0)

    def test_delta_none_does_not_disable_monotone_acceptance(self):
        """A fixed iteration budget still rejects a finite objective decrease."""
        init = _Model("init", 1.0)
        bad = _Model("bad", 0.0)
        chosen, score = _em_loop(
            None,
            None,
            init,
            _steps([bad]),
            _ll_fn,
            max_its=1,
            delta=None,
            out=None,
            monotone=True,
            track_best=False,
        )
        self.assertIs(chosen, init)
        self.assertEqual(score, 1.0)

    def test_monotone_rejects_nonfinite_and_keeps_best(self):
        """A NaN step is rejected; the best finite model is returned (monotone path)."""
        init = _Model("init", 0.0)
        good = _Model("good", 1.0)
        bad = _Model("bad", float("nan"))
        chosen, score = _em_loop(
            None, None, init, _steps([good, bad]), _ll_fn, max_its=2, delta=None, out=None, monotone=True
        )
        self.assertIs(chosen, good)
        self.assertTrue(np.isfinite(score))

    def test_nonmonotone_still_refuses_nonfinite_step(self):
        """Even with ``monotone=False`` (which accepts decreases), a non-finite step is refused."""
        init = _Model("init", 0.0)
        good = _Model("good", 1.0)
        bad = _Model("bad", float("-inf"))
        # track_best=False so the return is the final *accepted* model, exposing acceptance behavior.
        chosen, _ = _em_loop(
            None,
            None,
            init,
            _steps([good, bad]),
            _ll_fn,
            max_its=2,
            delta=None,
            out=None,
            monotone=False,
            track_best=False,
        )
        self.assertIs(chosen, good)

    def test_fused_loop_survives_nonfinite_without_stalling(self):
        """The fused loop returns a finite-best model and terminates when a step goes non-finite."""
        init = _Model("init", 0.0)
        m1 = _Model("m1", 1.0)
        m2 = _Model("m2", float("nan"))
        m3 = _Model("m3", 2.0)

        def fused_step_fn(_enc, _est, model):
            # Return (next_model, ll_of_input_model); the input model's ll is the posterior normalizer.
            nxt = {"init": m1, "m1": m2, "m2": m3}[model.tag]
            return nxt, model.ll

        chosen, score = _em_loop(
            None, None, init, None, _ll_fn, max_its=3, delta=1.0e-9, out=None, fused_step_fn=fused_step_fn
        )
        self.assertTrue(np.isfinite(score))
        self.assertIsInstance(chosen, _Model)

    def test_high_dim_diagonal_mixture_robust_fits_without_crash(self):
        """robust=True fits a high-dim, few-sample Gaussian mixture without singular-covariance crashes."""
        rng = np.random.RandomState(0)
        dim, num_comp, n = 32, 6, 150
        centers = rng.normal(0.0, 3.0, (num_comp, dim))
        labels = rng.randint(0, num_comp, n)
        x = centers[labels] + rng.normal(0.0, 1.0, (n, dim))
        data = [x[i] for i in range(n)]

        est = MixtureEstimator([DiagonalGaussianEstimator(dim=dim) for _ in range(num_comp)], robust=True)
        model = optimize(data, est, max_its=15, rng=np.random.RandomState(1), out=None)

        log_density = np.asarray([model.log_density(d) for d in data[:25]], dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(log_density)))


if __name__ == "__main__":
    unittest.main()
