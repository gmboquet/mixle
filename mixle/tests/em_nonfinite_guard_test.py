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
import unittest.mock

import numpy as np

from mixle.inference.em import MonteCarloEM, OnlineEM
from mixle.inference.estimation import _em_loop, _resolve_monotone, _resolve_track_best, optimize
from mixle.stats import DiagonalGaussianEstimator, GaussianEstimator, MixtureEstimator


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

    def test_stochastic_strategies_default_to_best_seen_over_an_otherwise_exact_model(self):
        # _resolve_monotone previously only looked at has_mutable_state/seq_local_elbo/
        # outer_objective_compatible -- none of which flag a genuinely stochastic EM *strategy*
        # (MonteCarloEM/OnlineEM) layered on top of an ordinary, otherwise-"exact" model/estimator.
        # The strict gate then defaulted to True and broke the very first noisy step, silently
        # terminating the whole optimize() run after one iteration regardless of max_its.
        exact = _Model("exact", 0.0)
        self.assertTrue(_resolve_monotone(None, None, exact))  # no strategy: unaffected, still strict

        mc = MonteCarloEM(lambda *args: (1, None), num_samples=1, seed=0)
        self.assertFalse(_resolve_monotone(None, None, exact, mc))
        self.assertTrue(_resolve_monotone(True, None, exact, mc))  # explicit bool still wins

        online = OnlineEM()
        self.assertFalse(_resolve_monotone(None, None, exact, online))

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

    def test_finite_step_replaces_nan_initial_best_state(self):
        """Best-state tracking must not restore a NaN-scoring initializer over a valid fit."""
        init = _Model("init", float("nan"))
        repaired = _Model("repaired", 2.0)
        chosen, score = _em_loop(
            None,
            None,
            init,
            _steps([repaired]),
            _ll_fn,
            max_its=1,
            delta=None,
            out=None,
            monotone=True,
            track_best=True,
        )
        self.assertIs(chosen, repaired)
        self.assertEqual(score, 2.0)

    def test_nan_validation_baseline_is_replaced_by_first_finite_score(self):
        init = _Model("init", 0.0)
        init.vll = float("nan")
        repaired = _Model("repaired", 1.0)
        repaired.vll = 3.0

        def scorer(enc, model):
            return 1, model.vll if enc == "validation" else model.ll

        chosen, score = _em_loop(
            "train",
            None,
            init,
            _steps([repaired]),
            scorer,
            max_its=1,
            delta=None,
            enc_vdata="validation",
            out=None,
            monotone=True,
            track_best=True,
        )
        self.assertIs(chosen, repaired)
        self.assertEqual(score, 3.0)

    def test_no_finite_initial_or_candidate_state_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "did not produce a finite objective"):
            _em_loop(
                None,
                None,
                _Model("init", float("nan")),
                _steps([_Model("bad", float("-inf"))]),
                _ll_fn,
                max_its=1,
                delta=None,
                out=None,
                monotone=False,
                track_best=False,
            )

    def test_fused_loop_replaces_nan_initial_best_state(self):
        init = _Model("init", float("nan"))
        repaired = _Model("repaired", 4.0)

        def fused_step_fn(_enc, _est, model):
            return repaired, model.ll

        chosen, score = _em_loop(
            None,
            None,
            init,
            None,
            _ll_fn,
            max_its=1,
            delta=None,
            out=None,
            track_best=True,
            fused_step_fn=fused_step_fn,
        )
        self.assertIs(chosen, repaired)
        self.assertEqual(score, 4.0)

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

    def test_public_optimizer_rejects_invalid_policies_and_initialization_controls(self):
        data = [0.0, 1.0]
        estimator = GaussianEstimator()
        invalid = (
            {"structure": "guess"},
            {"schedule": "sometimes"},
            {"objective": "likelihood"},
            {"init_p": 0.0},
            {"init_p": 1.1},
            {"init_p": float("nan")},
            {"max_its": 1.5},
            {"delta": -1.0},
            {"print_iter": 0.5},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                optimize(data, estimator, **({"max_its": 1, "out": None} | kwargs))
        for kwargs in ({"reuse_estep_ll": 1}, {"monotone": "yes"}, {"track_best": 0}):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(TypeError):
                optimize(data, estimator, max_its=1, out=None, **kwargs)

    def test_internal_structure_fit_errors_are_not_masked_as_routing_fallbacks(self):
        rows = [("a" if i % 2 else "b", float(i)) for i in range(60)]
        with unittest.mock.patch(
            "mixle.inference.bayesian_network.learn_bayesian_network",
            side_effect=ValueError("internal structure fit failed"),
        ):
            with self.assertRaisesRegex(ValueError, "internal structure fit failed"):
                optimize(rows, out=None)

    def test_nonfinite_candidate_bic_fails_closed(self):
        rows = [("a" if i % 2 else "b", float(i)) for i in range(60)]

        class Network:
            @staticmethod
            def edges():
                return [(0, 1)]

        with (
            unittest.mock.patch(
                "mixle.inference.bayesian_network.learn_bayesian_network",
                return_value=Network(),
            ),
            unittest.mock.patch(
                "mixle.inference.bayesian_network.bayesian_network_bic",
                return_value=float("nan"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite candidate BIC"):
                optimize(rows, out=None, max_its=1)

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
