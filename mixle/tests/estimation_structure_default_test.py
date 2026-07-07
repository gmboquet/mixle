"""optimize(data)/fit(data) with no estimator discover cross-field structure by default."""

import unittest
import unittest.mock

import numpy as np

from mixle.inference.estimation import fit, optimize


def _dependent(n, seed=0):
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        k = ["a", "b"][r.randint(0, 2)]
        out.append((k, float((5.0 if k == "b" else -5.0) + r.randn())))
    return out


def _independent(n, seed=0):
    r = np.random.RandomState(seed)
    return [(["a", "b"][r.randint(0, 2)], float(r.randn())) for _ in range(n)]


class StructureDefaultTest(unittest.TestCase):
    def test_dependent_records_return_a_discovered_graph(self):
        m = optimize(_dependent(400), out=None)
        self.assertEqual(type(m).__name__, "HeterogeneousBayesianNetwork")
        self.assertEqual(len(m.edges()), 1)
        # and it actually pays on fresh data vs the independence the old default assumed
        indep = optimize(_dependent(400), out=None, structure="off")
        fresh = _dependent(200, seed=9)
        ll_net = float(np.sum(m.seq_log_density(m.dist_to_encoder().seq_encode(fresh))))
        ll_ind = float(np.sum(indep.seq_log_density(indep.dist_to_encoder().seq_encode(fresh))))
        self.assertGreater(ll_net, ll_ind + 20.0)

    def test_independent_records_keep_the_historical_composite(self):
        m = optimize(_independent(400), out=None)
        self.assertEqual(type(m).__name__, "CompositeDistribution")

    def test_structure_off_restores_unconditional_behavior(self):
        m = optimize(_dependent(400), out=None, structure="off")
        self.assertEqual(type(m).__name__, "CompositeDistribution")

    def test_explicit_estimator_is_untouched(self):
        import mixle.stats as st

        m = optimize([float(v) for v in np.random.RandomState(0).randn(100)], st.GaussianEstimator(), out=None)
        self.assertEqual(type(m).__name__, "GaussianDistribution")

    def test_fit_shares_the_front_door(self):
        m = fit(_dependent(400), out=None)
        self.assertEqual(type(m).__name__, "HeterogeneousBayesianNetwork")
        m2 = fit(_dependent(400), out=None, structure="off")
        self.assertEqual(type(m2).__name__, "CompositeDistribution")

    def test_non_record_and_nested_data_fall_back_silently(self):
        m = optimize([float(v) for v in np.random.RandomState(0).randn(120)], out=None)
        self.assertEqual(type(m).__name__, "GaussianDistribution")
        m2 = optimize([("a", [1.0, 2.0])] * 100, out=None)
        self.assertEqual(type(m2).__name__, "CompositeDistribution")

    def test_small_samples_keep_the_composite(self):
        m = optimize(_dependent(30), out=None)  # under the 40-row floor: never engage on scraps
        self.assertEqual(type(m).__name__, "CompositeDistribution")

    def test_structure_search_forwards_the_callers_rng_to_the_composite_candidate(self):
        # Regression: the composite candidate fit inside the structure front door (built only to
        # BIC-compare against the discovered network) used a fresh, unseeded RandomState() instead of
        # the caller's rng -- silently breaking optimize(data, rng=...)'s documented reproducibility
        # contract for that one internal fit. The composite candidate's seq_initialize call must be
        # the exact rng object passed in, not a fresh default.
        import mixle.inference.estimation as est_mod

        calls = []
        original = est_mod.seq_initialize

        def spy(**kwargs):
            calls.append(kwargs.get("rng"))
            return original(**kwargs)

        given_rng = np.random.RandomState(11)
        with unittest.mock.patch.object(est_mod, "seq_initialize", side_effect=spy):
            m = optimize(_dependent(400), out=None, rng=given_rng)

        self.assertEqual(type(m).__name__, "HeterogeneousBayesianNetwork")  # net wins here -> composite
        # candidate's own init is the last (and only rng-relevant) seq_initialize call for this path.
        self.assertGreater(len(calls), 0)
        self.assertIs(calls[-1], given_rng)


if __name__ == "__main__":
    unittest.main()
