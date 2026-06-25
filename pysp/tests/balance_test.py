"""Automatic compute/memory/load-balancing planner (balance.py): grid choice across the model spectrum."""

import unittest

import numpy as np

import pysp.stats as stats
from pysp.inference import optimize
from pysp.utils.parallel.balance import auto_balanced_estimator, balance_plan
from pysp.utils.parallel.model_decomposition import compute_cost, cost_children, subtree_work
from pysp.utils.parallel.planner import DeviceSpec, Resources


def _cluster(p, mem=None, throughput=1.0):
    return Resources(
        devices=tuple(DeviceSpec(name=f"w{i}", kind="cpu", memory_bytes=mem, throughput=throughput) for i in range(p))
    )


def _mixture(k, d=0):
    if d:
        comps = [stats.MultivariateGaussianDistribution([0.0] * d, np.eye(d).tolist()) for _ in range(k)]
    else:
        comps = [stats.GaussianDistribution(float(i), 1.0) for i in range(k)]
    return stats.MixtureDistribution(comps, [1.0 / k] * k)


def _hmm(s, d=0):
    emit = (
        [stats.MultivariateGaussianDistribution([0.0] * d, np.eye(d).tolist()) for _ in range(s)]
        if d
        else [stats.GaussianDistribution(float(i), 1.0) for i in range(s)]
    )
    return stats.HiddenMarkovModelDistribution(emit, [1.0 / s] * s, (np.ones((s, s)) / s).tolist())


class CostModelTest(unittest.TestCase):
    def test_cost_children_finds_all_nested_dists(self):
        # reflective discovery: HMM emission states + len_dist are real compute even though HMM is atomic
        self.assertGreaterEqual(len(cost_children(_hmm(10))), 10)
        self.assertEqual(len(cost_children(stats.GaussianDistribution(0.0, 1.0))), 0)

    def test_compute_cost_recurses_and_scales(self):
        f_small, _ = compute_cost(_hmm(3))
        f_big, _ = compute_cost(_hmm(20))
        self.assertGreater(f_big, 5 * f_small)  # transition ~S^2 + emissions
        # rich emissions dominate a same-state-count HMM
        self.assertGreater(compute_cost(_hmm(20, d=15))[0], compute_cost(_hmm(20))[0])

    def test_subtree_work_counts_nested_nonshardable(self):
        # a mixture component that is itself an HMM costs its full subtree, not just its own params
        comp_hmm = stats.MixtureDistribution([_hmm(8), stats.GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
        works = [subtree_work(c) for c in comp_hmm.components]
        self.assertGreater(works[0], 5 * works[1])  # the HMM component vastly outweighs the lone Gaussian


class SpectrumDecisionTest(unittest.TestCase):
    def test_tiny_model_big_data_is_data_parallel(self):
        plan = balance_plan(_mixture(4), _cluster(8), n_data=10_000)
        self.assertEqual(plan.model_parallel, 1)
        self.assertEqual(plan.data_parallel, 8)

    def test_single_observation_splittable_model_is_model_parallel(self):
        plan = balance_plan(_mixture(32), _cluster(8), n_data=1)
        self.assertEqual(plan.data_parallel, 1)
        self.assertEqual(plan.model_parallel, 8)  # split the model across all workers
        self.assertEqual(plan.workers_used, 8)

    def test_few_observations_uses_data_times_model_grid(self):
        plan = balance_plan(_mixture(16), _cluster(8), n_data=4)
        self.assertEqual(plan.data_parallel, 4)  # one replica per observation
        self.assertEqual(plan.model_parallel, 2)  # split the model to fill the rest
        self.assertEqual(plan.workers_used, 8)

    def test_memory_forces_model_split(self):
        big = _mixture(16, d=100)  # ~2.5 MB of covariances
        plan = balance_plan(big, _cluster(8, mem=400_000), n_data=500)
        self.assertGreater(plan.model_parallel, 1)
        self.assertTrue(plan.fits)
        self.assertLessEqual(plan.model_bytes / plan.model_parallel, 400_000)

    def test_model_too_big_even_split_is_reported(self):
        big = _mixture(4, d=100)  # only 4 units, each ~80 KB
        plan = balance_plan(big, _cluster(8, mem=40_000), n_data=500)
        self.assertFalse(plan.fits)
        self.assertIn("WARNING", plan.rationale)

    def test_single_device_is_data_parallel_degenerate(self):
        plan = balance_plan(_mixture(8), _cluster(1), n_data=100)
        self.assertEqual(plan.workers_used, 1)

    def test_tiny_hmm_big_data_is_data_parallel(self):
        plan = balance_plan(_hmm(3), _cluster(8), n_data=5000)
        self.assertEqual(plan.model_parallel, 1)
        self.assertEqual(plan.data_parallel, 8)

    def test_massive_hmm_single_observation_is_model_parallel(self):
        # the goal case: a massive HMM on ONE observation splits along its state axis across the cluster
        plan = balance_plan(_hmm(50, d=10), _cluster(8), n_data=1)
        self.assertEqual(plan.data_parallel, 1)  # cannot data-parallel a single observation
        self.assertEqual(plan.model_parallel, 8)  # ...so split the 50 states across all 8 workers
        self.assertEqual(plan.workers_used, 8)
        self.assertEqual(
            plan.axis, __import__("pysp.stats.compute.decomposition", fromlist=["DecompAxis"]).DecompAxis.STATE
        )

    def test_truly_atomic_model_single_observation_is_honestly_diagnosed(self):
        # a lone leaf on one observation genuinely has no axis to split -> 1 worker, reported as such
        plan = balance_plan(stats.GaussianDistribution(0.0, 1.0), _cluster(8), n_data=1)
        self.assertEqual(plan.workers_used, 1)
        self.assertIn("no splittable axis", plan.rationale)


class HmmRealizationTest(unittest.TestCase):
    def test_tiny_hmm_big_data_realized_data_parallel_matches_local(self):
        # case from the goal: a tiny HMM on a bunch of data -> data-parallel, balanced, correct
        est = stats.HiddenMarkovEstimator(
            [stats.GaussianEstimator() for _ in range(3)], len_estimator=stats.PoissonEstimator()
        )
        init = stats.HiddenMarkovModelDistribution(
            [stats.GaussianDistribution(float(i) * 3, 1.0) for i in range(3)],
            [1 / 3] * 3,
            (np.ones((3, 3)) / 3).tolist(),
            len_dist=stats.PoissonDistribution(6.0),
        )
        sampler = init.sampler(seed=1)
        data = [sampler.sample() for _ in range(400)]  # 400 short sequences
        base = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="local")
        chosen, plan = auto_balanced_estimator(est, init, _cluster(4), n_data=len(data))
        self.assertFalse(plan.is_model_parallel)  # data-parallel
        fit = optimize(data, chosen, prev_estimate=init, max_its=5, out=None, backend="mp", num_workers=2)

        def ll(m):
            return float(np.sum(m.seq_log_density(m.dist_to_encoder().seq_encode(data))))

        self.assertTrue(np.isclose(ll(base), ll(fit), rtol=1e-6), (ll(base), ll(fit)))

    def test_rich_emission_hmm_state_parallel_is_bit_identical(self):
        from pysp.utils.parallel.model_parallel import ModelParallelEstimator

        # the massive/rich-emission HMM case: per-state emission scoring + accumulation are threaded
        # across states inside Baum-Welch; the forward-backward stays serial; the fit is bit-identical.
        emit = [stats.MultivariateGaussianDistribution((np.zeros(6)).tolist(), np.eye(6).tolist()) for _ in range(8)]
        init = stats.HiddenMarkovModelDistribution(
            emit, [1 / 8] * 8, (np.ones((8, 8)) / 8).tolist(), len_dist=stats.PoissonDistribution(10.0)
        )
        est = stats.HiddenMarkovEstimator(
            [stats.MultivariateGaussianEstimator(dim=6) for _ in range(8)], len_estimator=stats.PoissonEstimator()
        )
        data = [init.sampler(seed=s).sample() for s in range(50)]
        base = optimize(data, est, prev_estimate=init, max_its=4, out=None, backend="local")
        mp = optimize(
            data, ModelParallelEstimator(est, num_workers=4), prev_estimate=init, max_its=4, out=None, backend="local"
        )
        self.assertEqual(str(base), str(mp))


class FlopBalanceTest(unittest.TestCase):
    def test_model_cuts_balance_work_not_count(self):
        # mixture of unequal-cost components; the 2-way split should equalize FLOPs, not component counts
        comps = [stats.MultivariateGaussianDistribution([0.0] * 30, np.eye(30).tolist()) for _ in range(2)] + [
            stats.GaussianDistribution(float(i), 1.0) for i in range(10)
        ]
        model = stats.MixtureDistribution(comps, [1 / 12] * 12)
        plan = balance_plan(model, _cluster(2), n_data=1)
        self.assertEqual(len(plan.model_cuts), 2)
        works = [subtree_work(c) for c in model.components]
        shard_work = [sum(works[c.start : c.stop]) for c in plan.model_cuts]
        ratio = max(shard_work) / min(shard_work)
        self.assertLess(ratio, 2.0)  # work-balanced (a count-split would be ~6x off here)


class DriverCorrectnessTest(unittest.TestCase):
    def test_data_parallel_branch_returns_plain_estimator(self):
        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(4)])
        model = _mixture(4)
        chosen, plan = auto_balanced_estimator(est, model, _cluster(8), n_data=10_000)
        self.assertIs(chosen, est)
        self.assertFalse(plan.is_model_parallel)

    def test_model_parallel_branch_is_bit_identical(self):
        from pysp.utils.parallel.model_parallel import ModelParallelEstimator

        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(8)])
        init = stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 4, 1.0) for i in range(8)], [1 / 8] * 8)
        rng = np.random.RandomState(0)
        data = [float(rng.randn() + 3 * (rng.randint(8) - 4)) for _ in range(6)]  # N=6 < P=8 -> model-parallel
        base = optimize(data, est, prev_estimate=init, max_its=6, out=None, backend="local")
        chosen, plan = auto_balanced_estimator(est, init, _cluster(8), n_data=len(data))
        self.assertIsInstance(chosen, ModelParallelEstimator)
        fit = optimize(data, chosen, prev_estimate=init, max_its=6, out=None, backend="local")
        self.assertEqual(str(base), str(fit))  # realized grid is exactly the serial fit

    def test_heterogeneous_nested_dag_is_bit_identical(self):
        # an unbalanced, deeply heterogeneous nest: composite of [MVG mixture, mixture-of-composites, leaf]
        est = stats.CompositeEstimator(
            (
                stats.MixtureEstimator([stats.MultivariateGaussianEstimator(dim=4) for _ in range(3)]),
                stats.MixtureEstimator(
                    [stats.CompositeEstimator((stats.GaussianEstimator(), stats.PoissonEstimator())) for _ in range(4)]
                ),
                stats.PoissonEstimator(),
            )
        )
        rng = np.random.RandomState(1)
        init = stats.CompositeDistribution(
            (
                stats.MixtureDistribution(
                    [
                        stats.MultivariateGaussianDistribution((rng.randn(4)).tolist(), np.eye(4).tolist())
                        for _ in range(3)
                    ],
                    [1 / 3] * 3,
                ),
                stats.MixtureDistribution(
                    [
                        stats.CompositeDistribution(
                            (stats.GaussianDistribution(float(i), 1.0), stats.PoissonDistribution(float(i) + 1))
                        )
                        for i in range(4)
                    ],
                    [0.25] * 4,
                ),
                stats.PoissonDistribution(2.0),
            )
        )
        data = [
            ((rng.randn(4)).tolist(), (float(rng.randn()), int(rng.poisson(2))), int(rng.poisson(3)))
            for _ in range(200)
        ]
        base = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="local")
        chosen, _ = auto_balanced_estimator(est, init, _cluster(8), n_data=len(data))
        fit = optimize(data, chosen, prev_estimate=init, max_its=5, out=None, backend="local")
        self.assertEqual(str(base), str(fit))


if __name__ == "__main__":
    unittest.main()
