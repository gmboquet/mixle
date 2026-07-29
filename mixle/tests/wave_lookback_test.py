"""Smoke tests for the lookback hidden Markov module.

Covers mixle.stats.latent.lookback_hidden_markov_model (the typed lookback HMM). Checks that it imports, that a tiny
fixed-seed sample -> seq_initialize -> seq_estimate loop runs, and that the data encoders
round-trip (dist_to_encoder == acc_to_encoder, vectorized log-densities match per-item
log_density).
"""

import unittest

import numpy as np
from numpy.random import RandomState

import mixle.stats.latent.lookback_hidden_markov_model as new_mod
from mixle.inference import seq_estimate, seq_initialize
from mixle.inference.estimation import _resolve_objective, optimize
from mixle.stats import seq_encode, seq_log_density_sum
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.combinator.sequence import SequenceDistribution, SequenceEstimator
from mixle.stats.sequences.integer_markov_chain import IntegerMarkovChainDistribution, IntegerMarkovChainEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)

MODULES = [new_mod]


def make_dist(mod):
    """Small two-state lag-1 lookback HMM over integer sequences (mirrors lookback_hmm_example.py)."""
    d0 = IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2])
    t1 = IntegerMarkovChainDistribution(3, [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    t2 = IntegerMarkovChainDistribution(3, [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    init_dists = [SequenceDistribution(d0, CategoricalDistribution({1: 1.0}))] * 2
    len_dist = CategoricalDistribution({5: 0.5, 6: 0.5})

    return mod.LookbackHiddenMarkovModelDistribution(
        [t1, t2], w=[0.6, 0.4], transitions=[[0.8, 0.2], [0.3, 0.7]], lag=1, init_dist=init_dists, len_dist=len_dist
    )


def make_estimator(mod, with_len=True):
    """Estimator matching make_dist(), lightly regularized so the smoke EM stays numerically safe.

    Set with_len=False to drop the length estimator (the incremental update() path feeds the
    categorical length accumulator batch-by-batch, which trips a pre-existing KeyError bug in
    CategoricalAccumulator.seq_update for values unseen in earlier batches).
    """
    init_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    topic_est = IntegerMarkovChainEstimator(3, lag=1, pseudo_count=0.1)
    len_est = CategoricalEstimator(pseudo_count=0.1) if with_len else None

    return mod.LookbackHiddenMarkovModelEstimator(
        [topic_est] * 2, lag=1, init_estimators=[init_est] * 2, len_estimator=len_est, pseudo_count=(1.0, 1.0)
    )


def make_estimator_with_explicit_chain_prior(mod):
    """Like make_estimator(), but the OUTER chain is regularized by an explicit Dirichlet
    conjugate prior= instead of pseudo_count -- exercises the conjugate-prior parity this module's
    estimator/distribution now share with mixle.stats.latent.hidden_markov's HiddenMarkovEstimator/
    HiddenMarkovModelDistribution."""
    init_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    topic_est = IntegerMarkovChainEstimator(3, lag=1, pseudo_count=0.1)
    len_est = CategoricalEstimator(pseudo_count=0.1)
    prior = (
        DirichletDistribution(np.array([2.0, 2.0])),
        [DirichletDistribution(np.array([2.0, 2.0])), DirichletDistribution(np.array([2.0, 2.0]))],
    )
    return mod.LookbackHiddenMarkovModelEstimator(
        [topic_est] * 2, lag=1, init_estimators=[init_est] * 2, len_estimator=len_est, prior=prior
    )


class LookbackHmmImportTestCase(unittest.TestCase):
    def test_both_modules_define_protocol_classes(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                for cls_name in (
                    "LookbackHiddenMarkovModelDistribution",
                    "LookbackHiddenMarkovModelSampler",
                    "LookbackHiddenMarkovModelEstimatorAccumulator",
                    "LookbackHiddenMarkovModelEstimatorAccumulatorFactory",
                    "LookbackHiddenMarkovModelEstimator",
                    "LookbackHiddenMarkovModelDataEncoder",
                ):
                    self.assertTrue(hasattr(mod, cls_name), "%s missing %s" % (mod.__name__, cls_name))

    def test_protocol_methods_present(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                acc = make_estimator(mod).accumulator_factory().make()
                for name in ("log_density", "seq_log_density", "seq_encode", "dist_to_encoder", "sampler", "estimator"):
                    self.assertTrue(callable(getattr(dist, name)))
                for name in (
                    "update",
                    "initialize",
                    "seq_update",
                    "seq_initialize",
                    "combine",
                    "value",
                    "from_value",
                    "acc_to_encoder",
                ):
                    self.assertTrue(callable(getattr(acc, name)))


class LookbackHmmEncoderTestCase(unittest.TestCase):
    def test_encoder_round_trip(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                data = dist.sampler(seed=7).sample(20)

                enc_d = dist.dist_to_encoder()
                enc_a = make_estimator(mod).accumulator_factory().make().acc_to_encoder()

                self.assertEqual(enc_d, dist.dist_to_encoder())
                self.assertEqual(enc_d, enc_a)
                self.assertTrue(str(enc_d).startswith("LookbackHiddenMarkovModelDataEncoder"))

                ld_enc_d = dist.seq_log_density(enc_d.seq_encode(data))
                ld_enc_a = dist.seq_log_density(enc_a.seq_encode(data))
                ld_item = np.asarray([dist.log_density(u) for u in data])

                self.assertTrue(np.all(np.isfinite(ld_item)))
                self.assertTrue(np.allclose(ld_enc_d, ld_item))
                self.assertTrue(np.allclose(ld_enc_d, ld_enc_a))


class LookbackHmmEstimationTestCase(unittest.TestCase):
    def test_sample_seq_initialize_seq_estimate_smoke(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                data = dist.sampler(seed=11).sample(40)
                est = make_estimator(mod)

                enc_data = seq_encode(data, model=dist)

                init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
                self.assertIsInstance(init_model, mod.LookbackHiddenMarkovModelDistribution)

                model1 = seq_estimate(enc_data, est, init_model)
                model2 = seq_estimate(enc_data, est, model1)
                self.assertIsInstance(model2, mod.LookbackHiddenMarkovModelDistribution)

                _, ll0 = seq_log_density_sum(enc_data, init_model)
                _, ll1 = seq_log_density_sum(enc_data, model1)
                _, ll2 = seq_log_density_sum(enc_data, model2)

                self.assertTrue(np.isfinite(ll0))
                self.assertTrue(np.isfinite(ll1))
                self.assertTrue(np.isfinite(ll2))
                # EM steps (lightly regularized) should not noticeably decrease the likelihood
                self.assertGreaterEqual(ll1, ll0 - 0.5)
                self.assertGreaterEqual(ll2, ll1 - 0.5)

    def test_update_matches_seq_update(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                data = dist.sampler(seed=5).sample(8)
                est = make_estimator(mod, with_len=False)

                acc1 = est.accumulator_factory().make()
                for u in data:
                    acc1.update(u, 1.0, dist)

                acc2 = est.accumulator_factory().make()
                acc2.seq_update(dist.seq_encode(data), np.ones(len(data)), dist)

                v1, v2 = acc1.value(), acc2.value()
                self.assertTrue(np.allclose(v1[2], v2[2]))  # init_counts
                self.assertTrue(np.allclose(v1[3], v2[3]))  # state_counts
                self.assertTrue(np.allclose(v1[4], v2[4]))  # trans_counts


class LookbackHmmConjugatePriorTestCase(unittest.TestCase):
    """LookbackHiddenMarkovModelEstimator/Distribution previously had NO conjugate-prior machinery
    at all (no prior=, set_prior, get_prior, model_log_density, has_conj_prior), unlike the sibling
    HiddenMarkovEstimator/HiddenMarkovModelDistribution. This mirrors that sibling's Dirichlet chain
    prior exactly, so a caller can set an explicit prior= on the outer chain and
    mixle.inference.estimation.optimize() correctly auto-detects the 'map' objective instead of
    silently tracking plain (unpenalized) MLE.
    """

    def test_explicit_chain_prior_is_carried_through_estimate_and_detected_as_map(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                data = dist.sampler(seed=13).sample(40)
                est = make_estimator_with_explicit_chain_prior(mod)

                self.assertIsNotNone(est.get_prior())
                self.assertEqual(_resolve_objective("auto", est, dist), "map")

                enc_data = seq_encode(data, model=dist)
                init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
                fitted = seq_estimate(enc_data, est, init_model)

                self.assertTrue(fitted.has_conj_prior)
                self.assertIsNotNone(fitted.get_prior())
                mld = est.model_log_density(fitted)
                self.assertTrue(np.isfinite(mld))


class LookbackHmmObjectiveResolutionTestCase(unittest.TestCase):
    """``_resolve_objective`` (mixle.inference.estimation) used to treat a compound estimator's OWN
    ``get_prior()`` returning None as conclusive evidence of "definitely mle", even when that None
    exists purely because no EXPLICIT outer conjugate prior was set on the chain -- while a nested
    child estimator (here, each state's IntegerMarkovChainEstimator, regularized only by
    pseudo_count) still carried a genuine, now-exposed implicit prior, making the compound
    estimator's own ``model_log_density`` genuinely non-zero. This silently misdetected 'mle' for
    exactly the ``make_estimator()`` fixture already used throughout this file (pseudo_count
    everywhere, no explicit prior= anywhere) -- the scenario a raw, ungated seq_estimate loop
    (see LookbackHmmEstimationTestCase above) shows a non-monotone likelihood trace under, because
    the tracked quantity was the wrong (unpenalized) one.
    """

    def test_child_only_pseudo_count_regularization_still_resolves_to_map(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                est = make_estimator(mod)

                self.assertIsNone(est.get_prior())  # no EXPLICIT outer chain prior was set
                self.assertEqual(_resolve_objective("auto", est, dist), "map")

    def test_optimize_tracks_a_monotone_objective_under_child_only_regularization(self):
        for mod in MODULES:
            with self.subTest(module=repr(mod.__name__)):
                dist = make_dist(mod)
                est = make_estimator(mod)
                data = dist.sampler(seed=17).sample(60)

                trace = []
                fitted = optimize(
                    data,
                    est,
                    prev_estimate=dist,
                    max_its=30,
                    delta=None,
                    out=None,
                    on_step=lambda s, trace=trace: trace.append(s.log_density),
                )
                self.assertTrue(np.all(np.isfinite(fitted.w)))
                self.assertTrue(np.all(np.isfinite(fitted.transitions)))
                self.assertGreaterEqual(len(trace), 5)
                for a, b in zip(trace, trace[1:]):
                    self.assertGreaterEqual(b, a - 1e-9 * max(1.0, abs(a)))


if __name__ == "__main__":
    unittest.main()
