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


class BaseProtocolAdoptionTest(unittest.TestCase):
    """Families normalized onto the base-class canonical protocol (2026-07-14 centralization)."""

    def test_power_law_hawkes_pools_through_the_base_protocol_without_aliasing(self):
        # Previously pooled by ALIASING (stored itself; key_replace assigned the pool's list
        # reference to every site). Now the base protocol + a copying from_value: sites hold
        # independent copies of the pooled realizations.
        from mixle.stats.processes.power_law_hawkes import PowerLawHawkesAccumulator

        acc_a = PowerLawHawkesAccumulator(window=10.0, alpha_fixed=None, keys="h")
        acc_b = PowerLawHawkesAccumulator(window=10.0, alpha_fixed=None, keys="h")
        acc_a.realizations.extend([[0.1, 0.5], [1.0, 2.0]])
        acc_b.realizations.extend([[3.0, 4.0]])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        self.assertEqual(len(acc_a.realizations), 3)
        self.assertEqual(len(acc_b.realizations), 3)
        self.assertIsNot(acc_a.realizations, acc_b.realizations, "tied sites must not share one list")
        acc_a.realizations.append([9.9])  # mutating one site must not leak into the other
        self.assertEqual(len(acc_b.realizations), 3)

    def test_optional_delegates_the_key_pass_to_its_wrapped_child(self):
        # A keyed estimator nested under an Optional never pooled: Optional's key pass did not
        # delegate to the wrapped child accumulator (the dead-keys shape, combinator edition).
        from mixle.stats.combinator.optional import OptionalEstimator
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator

        def site(data):
            est = OptionalEstimator(estimator=GaussianEstimator(keys="inner"))
            acc = est.accumulator_factory().make()
            enc = acc.acc_to_encoder()
            acc.seq_update(enc.seq_encode(data), np.ones(len(data)), None)
            return acc

        acc_a, acc_b = site([1.0, 2.0, 3.0]), site([10.0, 20.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        self.assertIn("inner", stats, "the wrapped child's key must reach the stats dict")
        np.testing.assert_allclose(_flat(acc_a.accumulator.value()), _flat(acc_b.accumulator.value()), rtol=1e-12)
        self.assertAlmostEqual(float(acc_a.accumulator.count), 5.0, places=12)


class ArrayAliasingProtocolTest(unittest.TestCase):
    """A second compiler-review pass, this time over families that pool a raw NumPy array (or a
    tuple of arrays) directly instead of going through the base class's store-self-and-combine()
    protocol. ``key_merge`` stored the first tied site's own live array in the dict and then
    mutated it in place on the next merge (``stats_dict[key] += self.<field>``); ``key_replace``
    then handed that same array object to every tied site with no copy, so any site's later
    local accumulation would silently corrupt every other tied site's counts. Fixed by copying on
    both boundaries: once when a key is first adopted into the dict, and once when a value is
    read back out of the dict into a site.
    """

    @staticmethod
    def _bare(cls):
        """Construct via __new__, bypassing __init__: these tests target only key_merge's and
        key_replace's handling of specific fields, not full accumulator construction."""
        return object.__new__(cls)

    def test_mixture_family_comp_counts_no_aliasing(self):
        from mixle.stats.latent.heterogeneous_mixture import HeterogeneousMixtureAccumulator
        from mixle.stats.latent.mixture import MixtureAccumulator
        from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureEstimatorAccumulator

        for cls in (MixtureAccumulator, HeterogeneousMixtureAccumulator, SemiSupervisedMixtureEstimatorAccumulator):
            with self.subTest(cls=cls.__name__):
                acc_a, acc_b = self._bare(cls), self._bare(cls)
                for acc in (acc_a, acc_b):
                    acc.weight_key = "w"
                    acc.comp_key = None
                    acc.accumulators = []
                acc_a.comp_counts = np.array([1.0, 2.0, 3.0])
                acc_b.comp_counts = np.array([10.0, 20.0, 30.0])
                stats: dict = {}
                acc_a.key_merge(stats)
                acc_b.key_merge(stats)
                acc_a.key_replace(stats)
                acc_b.key_replace(stats)
                np.testing.assert_allclose(acc_a.comp_counts, [11.0, 22.0, 33.0])
                np.testing.assert_allclose(acc_b.comp_counts, [11.0, 22.0, 33.0])
                self.assertIsNot(acc_a.comp_counts, acc_b.comp_counts, "tied sites must not share one array")
                acc_a.comp_counts += 1000.0  # mutating one site must not leak into the other
                np.testing.assert_allclose(acc_b.comp_counts, [11.0, 22.0, 33.0])

    def test_dirac_length_comp_counts_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.dirac_length import DiracLengthMixtureAccumulator

        acc_a, acc_b = self._bare(DiracLengthMixtureAccumulator), self._bare(DiracLengthMixtureAccumulator)
        for acc in (acc_a, acc_b):
            acc.weight_key = "w"
            acc.comp_key = None
            acc.accumulator = NullAccumulator()
        acc_a.comp_counts = np.array([1.0, 2.0])
        acc_b.comp_counts = np.array([10.0, 20.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.comp_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.comp_counts, [11.0, 22.0])
        self.assertIsNot(acc_a.comp_counts, acc_b.comp_counts, "tied sites must not share one array")
        acc_a.comp_counts += 1000.0
        np.testing.assert_allclose(acc_b.comp_counts, [11.0, 22.0])

    def test_hmm_family_init_trans_counts_no_aliasing(self):
        from mixle.stats.latent.hidden_markov import HiddenMarkovAccumulator
        from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovAccumulator

        for cls in (HiddenMarkovAccumulator, TreeHiddenMarkovAccumulator):
            with self.subTest(cls=cls.__name__):
                acc_a, acc_b = self._bare(cls), self._bare(cls)
                for acc in (acc_a, acc_b):
                    acc.init_key = "i"
                    acc.trans_key = "t"
                    acc.state_key = None
                    acc.accumulators = []
                    acc.len_accumulator = None
                acc_a.init_counts = np.array([1.0, 2.0])
                acc_b.init_counts = np.array([10.0, 20.0])
                acc_a.trans_counts = np.array([[1.0, 0.0], [0.0, 1.0]])
                acc_b.trans_counts = np.array([[5.0, 5.0], [5.0, 5.0]])
                stats: dict = {}
                acc_a.key_merge(stats)
                acc_b.key_merge(stats)
                acc_a.key_replace(stats)
                acc_b.key_replace(stats)
                np.testing.assert_allclose(acc_a.init_counts, [11.0, 22.0])
                np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
                np.testing.assert_allclose(acc_a.trans_counts, [[6.0, 5.0], [5.0, 6.0]])
                np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])
                self.assertIsNot(acc_a.init_counts, acc_b.init_counts, "tied sites must not share one array")
                self.assertIsNot(acc_a.trans_counts, acc_b.trans_counts, "tied sites must not share one array")
                acc_a.init_counts += 1000.0
                acc_a.trans_counts += 1000.0
                np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
                np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])

    def test_lookback_hmm_init_trans_counts_no_aliasing(self):
        from mixle.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelEstimatorAccumulator

        cls = LookbackHiddenMarkovModelEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.init_key = "i"
            acc.trans_key = "t"
            acc.state_key = None
            acc.init_accumulators = []
            acc.seq_accumulators = []
            acc.len_accumulator = None
        acc_a.init_counts = np.array([1.0, 2.0])
        acc_b.init_counts = np.array([10.0, 20.0])
        acc_a.trans_counts = np.array([[1.0, 0.0], [0.0, 1.0]])
        acc_b.trans_counts = np.array([[5.0, 5.0], [5.0, 5.0]])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.init_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
        self.assertIsNot(acc_a.init_counts, acc_b.init_counts, "tied sites must not share one array")
        self.assertIsNot(acc_a.trans_counts, acc_b.trans_counts, "tied sites must not share one array")
        acc_a.init_counts += 1000.0
        acc_a.trans_counts += 1000.0
        np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])

    def test_segmental_hmm_init_trans_counts_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.segmental_hidden_markov_model import SegmentalHiddenMarkovAccumulator

        cls = SegmentalHiddenMarkovAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.init_key = "i"
            acc.trans_key = "t"
            acc.state_key = None
            acc.accumulators = []
            acc.len_accumulator = NullAccumulator()
        acc_a.init_counts = np.array([1.0, 2.0])
        acc_b.init_counts = np.array([10.0, 20.0])
        acc_a.trans_counts = np.array([[1.0, 0.0], [0.0, 1.0]])
        acc_b.trans_counts = np.array([[5.0, 5.0], [5.0, 5.0]])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.init_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
        self.assertIsNot(acc_a.init_counts, acc_b.init_counts, "tied sites must not share one array")
        self.assertIsNot(acc_a.trans_counts, acc_b.trans_counts, "tied sites must not share one array")
        acc_a.init_counts += 1000.0
        acc_a.trans_counts += 1000.0
        np.testing.assert_allclose(acc_b.init_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])

    def test_semi_supervised_hmm_trans_counts_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.semi_supervised_hidden_markov_model import (
            SemiSupervisedHiddenMarkovEstimatorAccumulator,
        )

        cls = SemiSupervisedHiddenMarkovEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.trans_key = "t"
            acc.state_key = None
            acc.accumulators = []
            acc.len_accumulator = NullAccumulator()
        acc_a.trans_counts = np.array([[1.0, 0.0], [0.0, 1.0]])
        acc_b.trans_counts = np.array([[5.0, 5.0], [5.0, 5.0]])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.trans_counts, [[6.0, 5.0], [5.0, 6.0]])
        np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])
        self.assertIsNot(acc_a.trans_counts, acc_b.trans_counts, "tied sites must not share one array")
        acc_a.trans_counts += 1000.0
        np.testing.assert_allclose(acc_b.trans_counts, [[6.0, 5.0], [5.0, 6.0]])

    def test_hierarchical_mixture_comp_and_w_counts_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.hierarchical_mixture import HierarchicalMixtureEstimatorAccumulator

        cls = HierarchicalMixtureEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.weight_key = "w"
            acc.comp_key = None
            acc.accumulators = []
            acc.len_accumulator = NullAccumulator()
        acc_a.comp_counts = np.array([[1.0, 2.0], [3.0, 4.0]])
        acc_b.comp_counts = np.array([[10.0, 20.0], [30.0, 40.0]])
        acc_a.w_counts = np.array([1.0, 2.0])
        acc_b.w_counts = np.array([10.0, 20.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.comp_counts, [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(acc_b.comp_counts, [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(acc_a.w_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.w_counts, [11.0, 22.0])
        self.assertIsNot(acc_a.comp_counts, acc_b.comp_counts, "tied sites must not share one array")
        self.assertIsNot(acc_a.w_counts, acc_b.w_counts, "tied sites must not share one array")
        acc_a.comp_counts += 1000.0
        acc_a.w_counts += 1000.0
        np.testing.assert_allclose(acc_b.comp_counts, [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(acc_b.w_counts, [11.0, 22.0])

    def test_joint_mixture_comp_and_joint_counts_no_aliasing(self):
        from mixle.stats.latent.joint_mixture import JointMixtureEstimatorAccumulator

        cls = JointMixtureEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.keys = ("w", None, None)
            acc.accumulators1 = []
            acc.accumulators2 = []
        acc_a.comp_counts1 = np.array([1.0, 2.0])
        acc_b.comp_counts1 = np.array([10.0, 20.0])
        acc_a.comp_counts2 = np.array([1.0, 2.0, 3.0])
        acc_b.comp_counts2 = np.array([10.0, 20.0, 30.0])
        acc_a.joint_counts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        acc_b.joint_counts = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.comp_counts1, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.comp_counts1, [11.0, 22.0])
        self.assertIsNot(acc_a.comp_counts1, acc_b.comp_counts1, "tied sites must not share one array")
        self.assertIsNot(acc_a.comp_counts2, acc_b.comp_counts2, "tied sites must not share one array")
        self.assertIsNot(acc_a.joint_counts, acc_b.joint_counts, "tied sites must not share one array")
        acc_a.comp_counts1 += 1000.0
        acc_a.comp_counts2 += 1000.0
        acc_a.joint_counts += 1000.0
        np.testing.assert_allclose(acc_b.comp_counts1, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.comp_counts2, [11.0, 22.0, 33.0])
        np.testing.assert_allclose(acc_b.joint_counts, [[6.0, 5.0, 5.0], [5.0, 6.0, 5.0]])

    def test_lda_sum_of_logs_and_prev_alpha_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.lda import LDAEstimatorAccumulator

        cls = LDAEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.alpha_key = "a"
            acc.topics_key = None
            acc.accumulators = []
            acc.len_accumulator = NullAccumulator()
        acc_a.sum_of_logs = np.array([1.0, 2.0])
        acc_b.sum_of_logs = np.array([10.0, 20.0])
        acc_a.doc_counts = 3.0
        acc_b.doc_counts = 7.0
        acc_a.prev_alpha = np.array([0.5, 0.5])
        acc_b.prev_alpha = None
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.sum_of_logs, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.sum_of_logs, [11.0, 22.0])
        self.assertEqual(acc_a.doc_counts, 10.0)
        self.assertEqual(acc_b.doc_counts, 10.0)
        np.testing.assert_allclose(acc_a.prev_alpha, [0.5, 0.5])
        np.testing.assert_allclose(acc_b.prev_alpha, [0.5, 0.5])
        self.assertIsNot(acc_a.sum_of_logs, acc_b.sum_of_logs, "tied sites must not share one array")
        self.assertIsNot(acc_a.prev_alpha, acc_b.prev_alpha, "tied sites must not share one array")
        acc_a.sum_of_logs += 1000.0
        acc_a.prev_alpha += 1000.0
        np.testing.assert_allclose(acc_b.sum_of_logs, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.prev_alpha, [0.5, 0.5])

    def test_labeled_lda_set_stats_and_prev_alpha_no_aliasing(self):
        from mixle.stats.latent.labeled_lda import LabeledLDAEstimatorAccumulator, LabeledLDALabelSetStats

        cls = LabeledLDAEstimatorAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.alpha_key = "a"
            acc.topics_key = None
            acc.accumulators = []
        acc_a.set_stats = LabeledLDALabelSetStats({(0,): [3.0, np.array([1.0, 2.0])]})
        acc_b.set_stats = LabeledLDALabelSetStats({(0,): [7.0, np.array([10.0, 20.0])]})
        acc_a.doc_counts = 3.0
        acc_b.doc_counts = 7.0
        acc_a.prev_alpha = np.array([0.5, 0.5])
        acc_b.prev_alpha = None
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.set_stats.stats[(0,)][0], 10.0)
        np.testing.assert_allclose(acc_a.set_stats.stats[(0,)][1], [11.0, 22.0])
        np.testing.assert_allclose(acc_b.set_stats.stats[(0,)][0], 10.0)
        np.testing.assert_allclose(acc_b.set_stats.stats[(0,)][1], [11.0, 22.0])
        self.assertEqual(acc_a.doc_counts, 10.0)
        self.assertEqual(acc_b.doc_counts, 10.0)
        np.testing.assert_allclose(acc_a.prev_alpha, [0.5, 0.5])
        np.testing.assert_allclose(acc_b.prev_alpha, [0.5, 0.5])
        self.assertIsNot(acc_a.set_stats, acc_b.set_stats, "tied sites must not share one object")
        self.assertIsNot(acc_a.prev_alpha, acc_b.prev_alpha, "tied sites must not share one array")
        acc_a.set_stats.stats[(0,)][1] += 1000.0
        acc_a.prev_alpha += 1000.0
        np.testing.assert_allclose(acc_b.set_stats.stats[(0,)][1], [11.0, 22.0])
        np.testing.assert_allclose(acc_b.prev_alpha, [0.5, 0.5])

    def test_heterogeneous_pcfg_terminal_and_binary_counts_no_aliasing(self):
        from mixle.stats.latent.heterogeneous_pcfg import HeterogeneousPCFGAccumulator

        cls = HeterogeneousPCFGAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.rule_key = "r"
            acc.emission_key = None
            acc.emission_accumulators = []
        acc_a.terminal_counts = np.array([1.0, 2.0])
        acc_b.terminal_counts = np.array([10.0, 20.0])
        acc_a.binary_counts = np.array([1.0, 2.0, 3.0])
        acc_b.binary_counts = np.array([10.0, 20.0, 30.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.terminal_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.terminal_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_a.binary_counts, [11.0, 22.0, 33.0])
        np.testing.assert_allclose(acc_b.binary_counts, [11.0, 22.0, 33.0])
        self.assertIsNot(acc_a.terminal_counts, acc_b.terminal_counts, "tied sites must not share one array")
        self.assertIsNot(acc_a.binary_counts, acc_b.binary_counts, "tied sites must not share one array")
        acc_a.terminal_counts += 1000.0
        acc_a.binary_counts += 1000.0
        np.testing.assert_allclose(acc_b.terminal_counts, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.binary_counts, [11.0, 22.0, 33.0])

    def test_integer_plsi_word_comp_doc_counts_no_aliasing(self):
        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.latent.integer_probabilistic_latent_semantic_indexing import (
            IntegerProbabilisticLatentSemanticIndexingAccumulator,
        )

        cls = IntegerProbabilisticLatentSemanticIndexingAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.wc_key = "wc"
            acc.sc_key = "sc"
            acc.dc_key = "dc"
            acc.len_acc = NullAccumulator()
        acc_a.word_count = np.array([[1.0, 2.0], [3.0, 4.0]])
        acc_b.word_count = np.array([[10.0, 20.0], [30.0, 40.0]])
        acc_a.comp_count = np.array([[1.0, 2.0], [3.0, 4.0]])
        acc_b.comp_count = np.array([[10.0, 20.0], [30.0, 40.0]])
        acc_a.doc_count = np.array([1.0, 2.0])
        acc_b.doc_count = np.array([10.0, 20.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        for field in ("word_count", "comp_count"):
            np.testing.assert_allclose(getattr(acc_a, field), [[11.0, 22.0], [33.0, 44.0]])
            np.testing.assert_allclose(getattr(acc_b, field), [[11.0, 22.0], [33.0, 44.0]])
            self.assertIsNot(getattr(acc_a, field), getattr(acc_b, field), "tied sites must not share one array")
        np.testing.assert_allclose(acc_a.doc_count, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.doc_count, [11.0, 22.0])
        self.assertIsNot(acc_a.doc_count, acc_b.doc_count, "tied sites must not share one array")
        acc_a.word_count += 1000.0
        acc_a.comp_count += 1000.0
        acc_a.doc_count += 1000.0
        np.testing.assert_allclose(acc_b.word_count, [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(acc_b.comp_count, [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(acc_b.doc_count, [11.0, 22.0])

    def test_structured_hmm_pi_acc_no_aliasing(self):
        from mixle.stats.latent.structured_hmm import StructuredHMMAccumulator

        cls = StructuredHMMAccumulator
        acc_a, acc_b = self._bare(cls), self._bare(cls)
        for acc in (acc_a, acc_b):
            acc.init_key = "i"
            acc.trans_key = None
            acc.emit = []
        acc_a.pi_acc = np.array([1.0, 2.0])
        acc_b.pi_acc = np.array([10.0, 20.0])
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.pi_acc, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.pi_acc, [11.0, 22.0])
        self.assertIsNot(acc_a.pi_acc, acc_b.pi_acc, "tied sites must not share one array")
        acc_a.pi_acc += 1000.0  # mirrors the in-place += that seq_update/combine/scale perform
        np.testing.assert_allclose(acc_b.pi_acc, [11.0, 22.0])

    def test_init_trans_keyed_accumulator_mixin_no_aliasing(self):
        from scipy.sparse import csc_matrix

        from mixle.stats.combinator.null_dist import NullAccumulator
        from mixle.stats.sequences._keyed_accumulator import InitTransKeyedAccumulator

        class _Host(InitTransKeyedAccumulator):
            def __init__(self, init_count, trans_count, keys):
                self.init_key, self.trans_key = keys
                self.init_count = init_count
                self.trans_count = trans_count
                self.size_accumulator = NullAccumulator()

        # Dense init_count (always ndarray) + sparse trans_count (the MarkovTransformAccumulator
        # shape): .copy() must work polymorphically for both, not just np.asarray(...).copy().
        acc_a = _Host(np.array([1.0, 2.0]), csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])), ("i", "t"))
        acc_b = _Host(np.array([10.0, 20.0]), csc_matrix(np.array([[5.0, 5.0], [5.0, 5.0]])), ("i", "t"))
        stats: dict = {}
        acc_a.key_merge(stats)
        acc_b.key_merge(stats)
        acc_a.key_replace(stats)
        acc_b.key_replace(stats)
        np.testing.assert_allclose(acc_a.init_count, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.init_count, [11.0, 22.0])
        np.testing.assert_allclose(acc_a.trans_count.toarray(), [[6.0, 5.0], [5.0, 6.0]])
        np.testing.assert_allclose(acc_b.trans_count.toarray(), [[6.0, 5.0], [5.0, 6.0]])
        self.assertIsNot(acc_a.init_count, acc_b.init_count, "tied sites must not share one array")
        self.assertIsNot(acc_a.trans_count, acc_b.trans_count, "tied sites must not share one sparse matrix")
        acc_a.init_count += 1000.0
        acc_a.trans_count *= 1000.0  # sparse += a nonzero scalar is unsupported by scipy; use *=
        np.testing.assert_allclose(acc_b.init_count, [11.0, 22.0])
        np.testing.assert_allclose(acc_b.trans_count.toarray(), [[6.0, 5.0], [5.0, 6.0]])
