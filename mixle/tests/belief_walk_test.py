"""Belief walk across a chain of verified transports (workstream F3, builds on F0/TRANSPORT-a).

Fixture: a linear-Gaussian AR(1) chain x_{t+1} = A*x_t + noise, A=0.8, noise_std=0.5 -- the SAME
family CARD TRANSPORT-a already proved calibrated, so composing it is composing a verified premise,
not an assumed one. The true composed distribution after k hops is analytically known (a standard
AR(1) recursion), so both the "uncertainty compounds honestly" claim and calibration are CHECKED
against ground truth, never assumed.

The direct-vs-composed comparison uses the realistic asymmetry the plan motivates multi-hop reasoning
with: abundant per-hop (A->B) training pairs are cheap, but a long-range direct (A->Z) pair is scarce.
With few direct pairs, a single end-to-end map badly under-calibrates; composing the (abundantly
trained) per-hop transport does not, because it never had to learn the long-range mapping directly.
"""

import unittest

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.reason.belief_walk import HopTransport, belief_walk, coverage_by_hop_count  # noqa: E402
from mixle.reason.cycle_consistency import fit_cycle_transport  # noqa: E402

A_COEF = 0.8
NOISE_STD = 0.5


def _ar1_step(x, rng, n):
    return A_COEF * x + rng.normal(0, NOISE_STD, size=(n, 1))


def _true_std(k: int) -> float:
    return float(np.sqrt(sum(A_COEF ** (2 * (k - i)) * NOISE_STD**2 for i in range(1, k + 1))))


_HOP = _WALK3_STD = None
_X0_TEST = _TRUE_BY_HOP = None
_DIRECT_SAMPLER = None

if _HAS_TORCH:
    _rng = np.random.RandomState(0)
    _x0_train = _rng.normal(0, 1.0, size=(2000, 1))
    _x1_train = _ar1_step(_x0_train, _rng, 2000)
    _HOP_FIT = fit_cycle_transport(_x0_train, _x1_train, k=1, seed=0, max_its=25)
    _HOP = HopTransport("ar1_step", _HOP_FIT, premise_passed=True)

    _rng2 = np.random.RandomState(1)
    _n_test = 200
    _X0_TEST = _rng2.normal(0, 1.0, size=(_n_test, 1))
    _cur = _X0_TEST
    _TRUE_BY_HOP = {}
    for _k in (1, 2, 3):
        _cur = _ar1_step(_cur, _rng2, _n_test)
        _TRUE_BY_HOP[_k] = _cur.copy()

    _WALK3 = belief_walk([_HOP, _HOP, _HOP], 0.0, n_draws=400, seed=5)
    _WALK3_STD = float(_WALK3.std[0])

    # scarce direct (x0 -> x3) training pairs -- the realistic motivation for composing hops at all.
    _rng_direct = np.random.RandomState(0)
    _x0_direct = _rng_direct.normal(0, 1.0, size=(40, 1))
    _cur_d = _x0_direct
    for _ in range(3):
        _cur_d = _ar1_step(_cur_d, _rng_direct, 40)
    _direct_fit = fit_cycle_transport(_x0_direct, _cur_d, k=2, seed=0, max_its=25)
    _DIRECT_SAMPLER = _direct_fit.sampler(seed=2)


@pytest.mark.skipif(_HOP is None, reason="torch not installed")
class UncertaintyCompoundsHonestlyTest(unittest.TestCase):
    def test_walk_posterior_std_tracks_the_analytic_ar1_recursion(self):
        for k in (1, 2, 3):
            walk = belief_walk([_HOP] * k, 0.0, n_draws=400, seed=5)
            self.assertAlmostEqual(float(walk.std[0]), _true_std(k), delta=0.08)

    def test_std_strictly_increases_with_hop_count(self):
        stds = [float(belief_walk([_HOP] * k, 0.0, n_draws=400, seed=5).std[0]) for k in (1, 2, 3)]
        self.assertLess(stds[0], stds[1])
        self.assertLess(stds[1], stds[2])


@pytest.mark.skipif(_HOP is None, reason="torch not installed")
class CalibrationByHopCountTest(unittest.TestCase):
    def test_coverage_is_consistent_with_nominal_at_every_hop_count(self):
        report = coverage_by_hop_count([_HOP, _HOP, _HOP], _X0_TEST, _TRUE_BY_HOP, n_draws=150, seed=0)
        self.assertEqual(set(report), {1, 2, 3})
        for k, entry in report.items():
            self.assertTrue(
                entry["consistent_with_nominal"], f"hop {k} coverage={entry['coverage']} p={entry['p_value']}"
            )

    def test_refuses_to_compose_an_unverified_hop(self):
        bad_hop = HopTransport("unverified", _HOP.fit, premise_passed=False)
        with self.assertRaises(ValueError):
            belief_walk([_HOP, bad_hop], 0.0)


@pytest.mark.skipif(_HOP is None, reason="torch not installed")
class ComposedWalkBeatsDirectEndToEndTest(unittest.TestCase):
    def test_composed_walk_beats_a_direct_map_starved_of_long_range_pairs(self):
        n_test = len(_X0_TEST)
        covered_direct = covered_walk = 0
        for i in range(n_test):
            # batched, not 150 individual sample_given calls -- statistically identical (each row of
            # the batch draws its own independent mixture component + Gaussian noise, see
            # NeuralConditionalDensitySampler.sample_given_batch's docstring) but avoids paying
            # per-call torch dispatch overhead 150*n_test times over.
            x_batch = np.repeat(np.atleast_2d(_X0_TEST[i]), 150, axis=0)
            direct_draws = np.asarray(_DIRECT_SAMPLER.sample_given_batch(x_batch))
            lo, hi = np.quantile(direct_draws, 0.05), np.quantile(direct_draws, 0.95)
            covered_direct += lo <= _TRUE_BY_HOP[3][i, 0] <= hi

            walk = belief_walk([_HOP, _HOP, _HOP], _X0_TEST[i], n_draws=150, seed=10 + i)
            lo, hi = walk.credible_interval(0.1)
            covered_walk += lo[0] <= _TRUE_BY_HOP[3][i, 0] <= hi[0]

        cov_direct = covered_direct / n_test
        cov_walk = covered_walk / n_test
        # the direct map, starved of long-range pairs, badly under-covers; the composed walk (built
        # from the SAME abundantly-trained per-hop transport) does not.
        self.assertLess(cov_direct, 0.7)
        self.assertGreater(cov_walk, 0.8)
        self.assertGreater(cov_walk, cov_direct + 0.15)


if __name__ == "__main__":
    unittest.main()
