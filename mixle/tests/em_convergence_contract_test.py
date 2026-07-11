"""The EM convergence contract (worklist Q5.4): trustworthy fit histories, as tests.

Tolerance-aware monotonicity rules, enforced by mixle's outer loop and pinned here:

- CLOSED-FORM trees (mixtures, HMMs, nested composites): every accepted round's observed
  log-likelihood is non-decreasing within the loop's acceptance tolerance (1e-12 of magnitude);
  the objective SEQUENCE therefore converges (it is bounded above by the estimator variance
  floors). ``monotone`` resolves to True automatically for these trees.
- MUTABLE (neural) trees: the loop auto-switches to best-visited selection; a round may decline
  (a stochastic M-step is not a Q-maximizer) but the RETURNED model's objective equals the best
  accepted round's. A failed M-step (non-finite objective, or a diverging module recovered by the
  GradLeaf guard) is never accepted and never poisons the convergence reference.
- ``lr_decay`` places the neural M-step schedule inside the Robbins-Monro window used by
  stochastic-approximation EM analyses (sum of steps infinite, sum of squares finite for
  exponents in (0.5, 1]); the schedule is disclosed per round in the fit receipt.
- Default console output is silent: iteration reporting happens only when ``out`` is supplied.

Weighted-fit trustworthiness (weighted == integer-replicated statistics) is Q5.3's contract and
lives in weighted_estimation_test.py; this file covers the remaining Q5.4 model classes.
"""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
)
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _trace_fit(proto, data, **kwargs):
    trace = []
    fitted = optimize(
        data,
        proto.estimator(),
        prev_estimate=proto,
        print_iter=0,
        on_step=lambda s: trace.append(s.log_density),
        **kwargs,
    )
    return fitted, trace


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _assert_monotone(test, trace):
    for a, b in zip(trace, trace[1:]):
        test.assertGreaterEqual(b, a - 1e-9 * max(1.0, abs(a)))


class ClosedFormMonotonicityTest(unittest.TestCase):
    """Closed-form trees: accepted-round objectives never decline beyond tolerance."""

    def test_mixture_trace_is_monotone_and_quiet_by_default(self):
        rng = np.random.RandomState(0)
        data = np.concatenate([rng.normal(m, 1.0, 250) for m in (-5.0, 0.0, 5.0)])
        proto = MixtureDistribution([GaussianDistribution(float(m), 2.0) for m in (-2.0, 0.5, 2.0)], [1 / 3] * 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            fitted, trace = _trace_fit(proto, data, max_its=25, delta=None)
        self.assertEqual(buf.getvalue(), "")  # no unsolicited iteration output
        self.assertGreaterEqual(len(trace), 5)
        _assert_monotone(self, trace)
        self.assertTrue(np.isfinite(_ll(fitted, data)))

    def test_hmm_trace_is_monotone(self):
        rng = np.random.RandomState(1)
        data = [list(rng.normal(-3.0 if rng.rand() < 0.5 else 3.0, 1.0, rng.randint(3, 7))) for _ in range(150)]
        proto = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(-1.0, 2.0), GaussianDistribution(1.0, 2.0)],
            w=[0.5, 0.5],
            transitions=[[0.8, 0.2], [0.2, 0.8]],
            len_dist=IntegerCategoricalDistribution(3, [0.25, 0.25, 0.25, 0.25]),
        )
        _, trace = _trace_fit(proto, data, max_its=15, delta=None)
        _assert_monotone(self, trace)

    def test_nested_composite_trace_is_monotone(self):
        rng = np.random.RandomState(2)
        values = np.concatenate([rng.normal(-4.0, 1.0, 300), rng.normal(4.0, 1.0, 300)])
        data = [(float(v), "a" if v < 0 else "b") for v in values]
        proto = MixtureDistribution(
            [
                CompositeDistribution(
                    (GaussianDistribution(float(m), 2.0), CategoricalDistribution({"a": 0.5, "b": 0.5}))
                )
                for m in (-1.0, 1.0)
            ],
            [0.5, 0.5],
        )
        _, trace = _trace_fit(proto, data, max_its=15, delta=None)
        _assert_monotone(self, trace)

    def test_delta_convergence_stops_early_with_small_final_gain(self):
        rng = np.random.RandomState(3)
        data = np.concatenate([rng.normal(-4.0, 1.0, 300), rng.normal(4.0, 1.0, 300)])
        proto = MixtureDistribution([GaussianDistribution(-2.0, 2.0), GaussianDistribution(2.0, 2.0)], [0.5, 0.5])
        _, trace = _trace_fit(proto, data, max_its=300, delta=1e-8)
        self.assertLess(len(trace), 300)  # converged, not exhausted
        self.assertLess(trace[-1] - trace[-2], 1e-6)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MutableModeContractTest(unittest.TestCase):
    """Neural trees: best-visited selection, disclosed failures, and the SAEM schedule."""

    class DiagGauss(torch.nn.Module if _HAS_TORCH else object):
        def __init__(self, mu0=0.0):
            super().__init__()
            self.mu = torch.nn.Parameter(torch.tensor([float(mu0)]))
            self.log_sigma = torch.nn.Parameter(torch.zeros(1))

        def log_density(self, x):
            d = torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))
            return d.log_prob(x if x.dim() > 1 else x.unsqueeze(-1)).sum(-1)

    def _bimodal(self, n=200, seed=0):
        rng = np.random.RandomState(seed)
        return [float(v) for v in np.concatenate([rng.normal(-4, 1, n), rng.normal(4, 1, n)])]

    def _proto(self, lr=0.05, lr_decay=None, m_steps=25):
        from mixle.models import GradLeaf

        torch.manual_seed(0)
        return MixtureDistribution(
            [
                GradLeaf(self.DiagGauss(0.5), m_steps=m_steps, lr=lr, lr_decay=lr_decay),
                GaussianDistribution(-1.0, 3.0),
            ],
            [0.5, 0.5],
        )

    def test_returned_model_is_best_visited(self):
        data = self._bimodal()
        fitted, trace = _trace_fit(self._proto(), data, max_its=12, delta=None)
        self.assertGreaterEqual(_ll(fitted, data), max(trace) - 1e-6)

    def test_failed_m_step_is_disclosed_not_accepted(self):
        # lr=25 diverges; the GradLeaf guard recovers and the receipt says so, while the fit's
        # objective stays finite -- a failed M-step is distinguishable from a true decline.
        data = self._bimodal()
        fitted, trace = _trace_fit(self._proto(lr=25.0, m_steps=40), data, max_its=8, delta=None)
        self.assertTrue(all(np.isfinite(v) for v in trace))
        self.assertTrue(np.isfinite(_ll(fitted, data)))
        self.assertGreaterEqual(fitted.components[0].fit_receipt["nonfinite_recoveries_total"], 1)

    def test_monotone_override_audits_the_trajectory(self):
        data = self._bimodal()
        _, trace = _trace_fit(self._proto(), data, max_its=12, delta=None, monotone=True)
        _assert_monotone(self, trace)  # with the strict gate, every accepted round is monotone

    def test_lr_decay_follows_the_declared_schedule_and_is_receipted(self):
        from mixle.models import GradLeaf

        data = self._bimodal()
        torch.manual_seed(0)
        proto = GradLeaf(self.DiagGauss(0.0), m_steps=10, lr=0.08, lr_decay=0.75)
        receipts = []
        optimize(
            data,
            proto.estimator(),
            prev_estimate=proto,
            max_its=6,
            delta=None,
            print_iter=0,
            on_step=lambda s: receipts.append(dict(s.model.fit_receipt)),
        )
        self.assertGreaterEqual(len(receipts), 3)
        for r in receipts:
            t = r["fit_round"]
            self.assertAlmostEqual(r["lr_effective"], 0.08 / (t**0.75), places=12)
            self.assertTrue(r["saem_schedule"])
        rounds = [r["fit_round"] for r in receipts]
        self.assertEqual(rounds, sorted(rounds))

    def test_constant_lr_is_disclosed_as_outside_the_saem_window(self):
        data = self._bimodal()
        fitted, _ = _trace_fit(self._proto(), data, max_its=4, delta=None)
        receipt = fitted.components[0].fit_receipt
        self.assertIsNone(receipt["lr_decay"])
        self.assertFalse(receipt["saem_schedule"])

    def test_lr_decay_validation(self):
        from mixle.models import GradLeaf

        with self.assertRaises(ValueError):
            GradLeaf(self.DiagGauss(), lr_decay=0.0)
        with self.assertRaises(ValueError):
            GradLeaf(self.DiagGauss(), lr_decay=1.5)
        with self.assertRaises(ValueError):
            GradLeaf(self.DiagGauss(), lr_decay=0.75, optimizer=lambda params: None)


if __name__ == "__main__":
    unittest.main()
