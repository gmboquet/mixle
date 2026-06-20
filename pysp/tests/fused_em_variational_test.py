"""Fused-EM (optimize(reuse_estep_ll=True)) support for the variational topic-model families.

These families' seq_log_density returns a per-document variational lower bound (ELBO) rather than a
plain logsumexp marginal (IntegerProbabilisticLatentSemanticIndexing's is an exact marginal -- still a per-document data
log-likelihood). For each family this verifies:

  * PARITY: with _track_ll enabled the accumulator's reported batch log-likelihood (_seq_ll) equals
    seq_log_density_sum() for the same data and distribution. The ELBO is recovered inside the E-step
    from the variational quantities it already computes (gammas / responsibilities / per-topic
    densities), without re-running the per-document variational loop.
  * FUSED: EM run with reuse_estep_ll=True reaches the same optimum as the standard loop
    (same init/seed/iterations).

Families covered: LDA, LabeledLDA, IntegerProbabilisticLatentSemanticIndexing.
"""

import io
import unittest

import numpy as np
from numpy.random import RandomState

from pysp.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    IntegerProbabilisticLatentSemanticIndexingDistribution,
    IntegerProbabilisticLatentSemanticIndexingEstimator,
    LDADistribution,
    seq_encode,
    seq_log_density_sum,
)
from pysp.stats.latent.labeled_lda import LabeledLDADistribution, LabeledLDAEstimator
from pysp.stats.latent.lda import LDAEstimator
from pysp.utils.estimation import optimize
from pysp.utils.optsutil import count_by_value


class FusedEMVariationalTestCase(unittest.TestCase):
    def _parity(self, dist, est, data):
        enc = seq_encode(data, model=dist)
        _, ref = seq_log_density_sum(enc, dist)
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), dist)
        self.assertAlmostEqual(acc._seq_ll, ref, delta=1e-4 * max(1.0, abs(ref)))

    def _fused(self, est_factory, data, max_its=15, init_p=0.1):
        std = optimize(
            data,
            est_factory(),
            max_its=max_its,
            delta=None,
            rng=RandomState(1),
            out=io.StringIO(),
            init_p=init_p,
            reuse_estep_ll=False,
        )
        fused = optimize(
            data,
            est_factory(),
            max_its=max_its,
            delta=None,
            rng=RandomState(1),
            out=io.StringIO(),
            init_p=init_p,
            reuse_estep_ll=True,
        )
        _, ls = seq_log_density_sum(seq_encode(data, model=std), std)
        _, lf = seq_log_density_sum(seq_encode(data, model=fused), fused)
        self.assertAlmostEqual(ls, lf, delta=1e-4 * max(1.0, abs(ls)))

    # ------------------------------------------------------------------ LDA
    def _lda(self):
        topics = [
            CategoricalDistribution({0: 0.6, 1: 0.2, 2: 0.1, 3: 0.1}),
            CategoricalDistribution({0: 0.1, 1: 0.1, 2: 0.4, 3: 0.4}),
            CategoricalDistribution({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}),
        ]
        dist = LDADistribution(
            topics,
            alpha=[1.0, 1.0, 1.0],
            len_dist=CategoricalDistribution({4: 0.3, 5: 0.4, 6: 0.3}),
            gamma_threshold=1e-10,
        )
        raw = dist.sampler(seed=3).sample(40)
        data = [sorted(count_by_value(u).items()) for u in raw]
        mk = lambda: LDAEstimator([CategoricalEstimator() for _ in range(3)])
        return dist, mk, data

    def test_lda_parity(self):
        dist, mk, data = self._lda()
        self._parity(dist, mk(), data)

    def test_lda_fused(self):
        _, mk, data = self._lda()
        self._fused(mk, data)

    # ----------------------------------------------------------------- LabeledLDA
    def _llda(self):
        VOCAB = ["w0", "w1", "w2", "w3"]
        PMATS = [[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]
        rng = RandomState(7)
        label_sets = [[0], [1], [0, 2], [1, 2]]
        data = []
        for i in range(24):
            labels = list(label_sets[i % len(label_sets)])
            p = np.asarray(PMATS[labels[0] % 2]) * 0.7 + np.asarray(PMATS[(labels[-1] + 1) % 2]) * 0.3
            words = rng.choice(4, size=9, p=p / p.sum())
            cnts = {}
            for wd in words:
                cnts[VOCAB[wd]] = cnts.get(VOCAB[wd], 0) + 1
            data.append((sorted(cnts.items()), labels))
        dist = LabeledLDADistribution(
            [
                CategoricalDistribution({"w0": 0.4, "w1": 0.4, "w2": 0.1, "w3": 0.1}),
                CategoricalDistribution({"w0": 0.1, "w1": 0.1, "w2": 0.4, "w3": 0.4}),
            ],
            np.asarray([[1.0, 1.0], [1.5, 0.5], [0.5, 1.5]], dtype=float),
            gamma_threshold=1e-10,
        )
        mk = lambda: LabeledLDAEstimator(
            [CategoricalEstimator(), CategoricalEstimator()], num_alphas=3, gamma_threshold=1e-10
        )
        return dist, mk, data

    def test_llda_parity(self):
        dist, mk, data = self._llda()
        self._parity(dist, mk(), data)

    def test_llda_fused(self):
        _, mk, data = self._llda()
        self._fused(mk, data)

    # ------------------------------------------------------------- IntegerProbabilisticLatentSemanticIndexing
    def _int_plsi(self):
        # Dense, low-cardinality config + init_p=0.5: PLSI's numba E-step (fast_seq_update) has a
        # pre-existing division-by-zero instability for sparse/degenerate inits, unrelated to
        # _track_ll. This config keeps the standard EM path stable.
        num_states, num_authors, num_words = 2, 4, 6
        rng = np.random.RandomState(1)
        sw = rng.dirichlet(np.ones(num_words), size=num_states).T
        ds = rng.dirichlet(np.ones(num_states), size=num_authors)
        dv = rng.dirichlet(np.ones(num_authors))
        dist = IntegerProbabilisticLatentSemanticIndexingDistribution(
            state_word_mat=sw, doc_state_mat=ds, doc_vec=dv, len_dist=CategoricalDistribution({8: 1.0})
        )
        data = dist.sampler(seed=10).sample(400)
        mk = lambda: IntegerProbabilisticLatentSemanticIndexingEstimator(
            num_vals=num_words, num_states=num_states, num_docs=num_authors, len_estimator=CategoricalEstimator()
        )
        return dist, mk, data

    def test_int_plsi_parity(self):
        dist, mk, data = self._int_plsi()
        self._parity(dist, mk(), data)

    def test_int_plsi_fused(self):
        _, mk, data = self._int_plsi()
        self._fused(mk, data, init_p=0.5)


if __name__ == "__main__":
    unittest.main()
