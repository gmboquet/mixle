"""The keyed-pooling protocol: every tied site estimates from the FULL pool, order-independently.

Eight families implemented ``key_merge`` as pull-from-dict-into-self WITHOUT writing the pooled
result back, so the dict kept the first site's statistics and ``key_replace`` handed that truncated
pool to every tied site -- later sites' data silently discarded, with order-dependent wrong fits
(found by the compiler review's keyed-tying probe: a tied-variance Gaussian mixture stamped
component 1's statistics onto both components and crashed the likelihood before "converging" to a
wrong fixed point). The canonical protocol (Poisson, Categorical, and 100 other files) accumulates
the pool IN the dict. These tests pin the pooled-sum and order-invariance properties for every
previously-broken family, plus the Gaussian mixture's analytic pooled fixed point end-to-end.
"""

import unittest

import numpy as np

from mixle.stats import (
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    MixtureDistribution,
    SkellamDistribution,
)


def _pooled_stats(dist, est_kwargs, data_a, data_b, order):
    """Run the merge/replace protocol over two keyed sites and return both sites' values."""
    est = dist.estimator()
    fac = type(est)(keys="k").accumulator_factory() if _ctor_takes_keys(type(est)) else None
    if fac is None:
        raise unittest.SkipTest(f"{type(est).__name__} has no keys ctor")
    acc_a, acc_b = fac.make(), fac.make()
    enc = dist.dist_to_encoder()
    acc_a.seq_update(enc.seq_encode(data_a), np.ones(len(data_a)), dist)
    acc_b.seq_update(enc.seq_encode(data_b), np.ones(len(data_b)), dist)
    stats: dict = {}
    first, second = (acc_a, acc_b) if order == "ab" else (acc_b, acc_a)
    first.key_merge(stats)
    second.key_merge(stats)
    first.key_replace(stats)
    second.key_replace(stats)
    return acc_a.value(), acc_b.value()


def _ctor_takes_keys(est_cls):
    import inspect

    return "keys" in inspect.signature(est_cls.__init__).parameters


def _flat(v):
    out = []

    def walk(u):
        if isinstance(u, (tuple, list)):
            for piece in u:
                walk(piece)
        elif isinstance(u, dict):
            for k in sorted(u, key=repr):
                walk(u[k])
        elif u is not None:
            out.extend(np.asarray(u, dtype=np.float64).ravel().tolist())

    walk(v)
    return np.asarray(out)


class KeyedPoolingProtocolTest(unittest.TestCase):
    FAMILIES = [
        (GaussianDistribution(0.0, 1.0), [0.5, -1.2, 2.0], [3.1, -0.4]),
        (ExponentialDistribution(1.0), [0.5, 1.2, 2.0], [3.1, 0.4]),
        (GammaDistribution(2.0, 1.0), [0.5, 1.2, 2.0], [3.1, 0.4]),
        (GeometricDistribution(0.5), [1, 2, 3], [4, 1]),
        (SkellamDistribution(2.0, 1.0), [1, -2, 3], [0, 2]),
    ]

    def test_both_sites_hold_the_full_pool_and_order_does_not_matter(self):
        for dist, data_a, data_b in self.FAMILIES:
            with self.subTest(family=type(dist).__name__):
                va_ab, vb_ab = _pooled_stats(dist, {}, data_a, data_b, order="ab")
                va_ba, vb_ba = _pooled_stats(dist, {}, data_a, data_b, order="ba")
                # both sites identical (the pool), regardless of merge order
                np.testing.assert_allclose(_flat(va_ab), _flat(vb_ab), rtol=1e-12)
                np.testing.assert_allclose(_flat(va_ab), _flat(va_ba), rtol=1e-12)
                # and the pool equals accumulating ALL the data at one site
                est = type(dist.estimator())(keys="k")
                acc_all = est.accumulator_factory().make()
                enc = dist.dist_to_encoder()
                both = list(data_a) + list(data_b)
                acc_all.seq_update(enc.seq_encode(both), np.ones(len(both)), dist)
                np.testing.assert_allclose(_flat(va_ab), _flat(acc_all.value()), rtol=1e-12)

    def test_ctmc_keys_actually_pool(self):
        # The CTMC accumulator had a THIRD failure shape: neither key_merge nor key_replace ever
        # INSERTED the key, so both were unconditional no-ops and keyed CTMCs silently never tied.
        from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainAccumulator

        acc_a = ContinuousTimeMarkovChainAccumulator(num_states=2, keys="q")
        acc_b = ContinuousTimeMarkovChainAccumulator(num_states=2, keys="q")
        acc_a.counts[:] = [[0.0, 2.0], [1.0, 0.0]]
        acc_a.dwell[:] = [1.5, 0.5]
        acc_b.counts[:] = [[0.0, 4.0], [3.0, 0.0]]
        acc_b.dwell[:] = [2.0, 1.0]
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        for acc in (acc_a, acc_b):
            np.testing.assert_array_equal(acc.counts, [[0.0, 6.0], [4.0, 0.0]])
            np.testing.assert_array_equal(acc.dwell, [3.5, 1.5])

    def test_malformed_tuple_keys_on_scalar_families_refuse_loudly(self):
        # Scalar families declare keys: str|None but silently accepted tuples (the combinator
        # convention), tying as one opaque composite key instead of what the caller meant. The
        # validator now checks the value against each family's OWN ctor annotation.
        from mixle.stats.compute.pdist import validate_estimator_keys
        from mixle.stats.latent.mixture import MixtureEstimator
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator

        bad = MixtureEstimator([GaussianEstimator(keys=(None, "shared_var")), GaussianEstimator()])
        with self.assertRaisesRegex(ValueError, "combinator convention"):
            validate_estimator_keys(bad)
        # tuple keys stay legal exactly where they are declared: combinator estimators
        good = MixtureEstimator([GaussianEstimator(keys="s"), GaussianEstimator(keys="s")], keys=("w", "c"))
        validate_estimator_keys(good)

    def test_tied_gaussian_mixture_reaches_the_analytic_pooled_fixed_point(self):
        from mixle.stats.compute.sequence import seq_estimate
        from mixle.stats.latent.mixture import MixtureEstimator
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator

        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(-3, 1, 2000), rng.normal(3, 2, 2000)])]
        model = MixtureDistribution([GaussianDistribution(-2.0, 1.5), GaussianDistribution(2.0, 1.5)], [0.5, 0.5])
        keyed = MixtureEstimator([GaussianEstimator(keys="shared"), GaussianEstimator(keys="shared")])
        enc_data = [(len(data), model.dist_to_encoder().seq_encode(data))]
        fit = seq_estimate(enc_data, keyed, model)
        for c in fit.components:
            self.assertAlmostEqual(c.mu, float(np.mean(data)), places=10)
            self.assertAlmostEqual(c.sigma2, float(np.var(data)), places=8)


if __name__ == "__main__":
    unittest.main()
