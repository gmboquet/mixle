"""Cycle-consistency as the cross-modal calibration/abstention signal (workstream F5, CARD builds on
TRANSPORT-a's F0 gate). Fixture: y = max(x, 0) + noise -- an observation function that is a bijection
for x >= 0 (fully recoverable) and collapses every x < 0 onto the SAME noisy y ~ 0 (irrecoverably
ambiguous). The forward noise is homoscedastic (constant 0.05 everywhere): a forward/marginal
confidence signal has NO way to see the collapse, since the noise model it reports does not depend on
whether this particular input happened to land in the collapsed region. Cycle-consistency (disagreement
among repeated backward-transport draws) does.

Kill criterion (per the card): if inconsistency does not correlate with error on this fixture, or
abstaining on it does not beat marginal-confidence abstention on selective accuracy, this signal is
dropped and the negative result recorded -- printed explicitly here, mirroring TRANSPORT-a's own
go/no-go report.
"""

import unittest

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.reason.cycle_consistency import (  # noqa: E402
    cycle_inconsistency,
    fit_cycle_transport,
    posterior_mean_estimate,
    selective_error,
)

_N_TRAIN = 2500
_N_TEST = 300
_NOISE = 0.05


def _sample(n, rng):
    x = rng.normal(0, 2.0, size=(n, 1))
    y = np.maximum(x, 0) + rng.normal(0, _NOISE, size=(n, 1))
    return x, y


# fit both transports ONCE at import time (mirrors transport_proof_test.py's own pattern -- every test
# class reads these same fits rather than repeating ~10s of training per test).
_BACKWARD = _FORWARD = None
_X_TEST = _Y_TEST = None
_ERRORS = _CYCLE_SCORES = _MARGINAL_CONF = None

if _HAS_TORCH:
    _x_train, _y_train = _sample(_N_TRAIN, np.random.RandomState(0))
    _BACKWARD = fit_cycle_transport(_y_train, _x_train, k=3, seed=0, max_its=45)  # p(x | y): the ambiguous hop
    _FORWARD = fit_cycle_transport(_x_train, _y_train, k=2, seed=0, max_its=30)  # p(y | x): the easy hop
    _back_sampler = _BACKWARD.sampler(seed=1)
    _fwd_sampler = _FORWARD.sampler(seed=2)

    _X_TEST, _Y_TEST = _sample(_N_TEST, np.random.RandomState(1))
    _errors, _cycle_scores, _marginal_conf = [], [], []
    for _i in range(_N_TEST):
        _est = posterior_mean_estimate(_back_sampler, _Y_TEST[_i], n_draws=20)
        _errors.append(abs(float(_est[0]) - float(_X_TEST[_i, 0])))
        _cycle_scores.append(cycle_inconsistency(_back_sampler, _Y_TEST[_i], n_draws=20))
        _fwd_draws = np.asarray([_fwd_sampler.sample_given(_X_TEST[_i]) for _ in range(20)])
        _marginal_conf.append(float(np.var(_fwd_draws)))
    _ERRORS = np.asarray(_errors)
    _CYCLE_SCORES = np.asarray(_cycle_scores)
    _MARGINAL_CONF = np.asarray(_marginal_conf)


@pytest.mark.skipif(_ERRORS is None, reason="torch not installed")
class CycleInconsistencyCorrelatesWithErrorTest(unittest.TestCase):
    def test_correlation_is_strong_and_positive(self):
        corr = float(np.corrcoef(_CYCLE_SCORES, _ERRORS)[0, 1])
        self.assertGreater(corr, 0.4)

    def test_inconsistency_and_error_are_both_higher_in_the_collapsed_region(self):
        collapsed = _X_TEST[:, 0] < 0
        recoverable = ~collapsed
        self.assertGreater(_CYCLE_SCORES[collapsed].mean(), _CYCLE_SCORES[recoverable].mean() * 5)
        self.assertGreater(_ERRORS[collapsed].mean(), _ERRORS[recoverable].mean() * 3)


@pytest.mark.skipif(_ERRORS is None, reason="torch not installed")
class CycleAbstentionBeatsMarginalConfidenceTest(unittest.TestCase):
    def test_marginal_confidence_is_blind_to_the_collapse(self):
        # the forward noise is homoscedastic by construction: the forward model's own reported
        # uncertainty should be close to flat across the collapsed/recoverable regions, and only
        # weakly (if at all) correlated with the actual backward-recovery error.
        corr = float(np.corrcoef(_MARGINAL_CONF, _ERRORS)[0, 1])
        self.assertLess(abs(corr), 0.3)

    def test_cycle_based_selection_beats_marginal_confidence_at_every_budget(self):
        for keep_frac in (0.3, 0.5, 0.7):
            se_cycle = selective_error(_ERRORS, _CYCLE_SCORES, keep_frac)
            se_marginal = selective_error(_ERRORS, _MARGINAL_CONF, keep_frac)
            self.assertLess(se_cycle, se_marginal * 0.9)

    def test_cycle_based_selection_beats_answering_everyone(self):
        se_cycle = selective_error(_ERRORS, _CYCLE_SCORES, 0.5)
        self.assertLess(se_cycle, float(_ERRORS.mean()) * 0.5)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class FitCycleTransportShapeTest(unittest.TestCase):
    """fit_cycle_transport must treat a 1D array as N scalar observations (n, 1), never as one
    n-dimensional row -- np.atleast_2d does the opposite, which silently misinterprets the data
    (this is the natural call shape for a scalar-valued modality reading, e.g. `given = rng.normal(size=n)`)."""

    def test_1d_given_and_target_fit_without_crashing_and_sample_correctly(self):
        rng = np.random.RandomState(0)
        n = 200
        given = rng.normal(size=n)
        target = 2 * given + 0.1 * rng.normal(size=n)

        fit = fit_cycle_transport(given, target, max_its=5, m_steps=5)
        sampler = fit.sampler(seed=0)
        draws = np.asarray(sampler.sample_given_batch(np.array([[0.5]])))
        self.assertEqual(draws.shape, (1, 1))

    def test_mismatched_1d_lengths_raise_a_clear_error_not_an_index_error(self):
        rng = np.random.RandomState(0)
        given = rng.normal(size=200)
        target = rng.normal(size=150)  # deliberately mismatched
        with self.assertRaisesRegex(ValueError, "same number of paired observations"):
            fit_cycle_transport(given, target, max_its=2, m_steps=2)


@pytest.mark.skipif(_ERRORS is None, reason="torch not installed")
class CycleConsistencyGoNoGoTest(unittest.TestCase):
    def test_go_no_go_report(self):
        """The card's required explicit verdict, computed fresh from the module-level fits."""
        corr = float(np.corrcoef(_CYCLE_SCORES, _ERRORS)[0, 1])
        se_cycle = selective_error(_ERRORS, _CYCLE_SCORES, 0.5)
        se_marginal = selective_error(_ERRORS, _MARGINAL_CONF, 0.5)
        correlates = corr > 0.3
        beats_marginal = se_cycle < se_marginal
        verdict = "GO" if (correlates and beats_marginal) else "NO-GO"
        print(
            f"\n[F5 cycle-consistency go/no-go] corr(cycle, error)={corr:.3f}, "
            f"selective_error(cycle@0.5)={se_cycle:.4f}, selective_error(marginal@0.5)={se_marginal:.4f} "
            f"-> {verdict}"
        )
        self.assertTrue(correlates and beats_marginal, f"kill criterion triggered: verdict={verdict}")


if __name__ == "__main__":
    unittest.main()
