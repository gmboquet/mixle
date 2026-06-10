"""Tests for model-based t-SNE (pysp.utils.htsne).

Kept fast by fitting a small mixture once in setUpClass and reusing it; no DPM
fitting is exercised here (htsne accepts a prefit mix_model).
"""
import io
import time
import unittest

import numpy as np

from pysp.stats import (
    CategoricalDistribution, CategoricalEstimator, CompositeDistribution, CompositeEstimator,
    GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator,
    seq_encode, seq_estimate, seq_initialize,
)
from pysp.stats import PoissonDistribution, SequenceDistribution, IntegerCategoricalDistribution
from pysp.utils.htsne import (
    conditional_pmat, dpmsne, get_pmat, htsne, humap, model_knn, model_log_affinity,
    sparse_model_distances, t_kernel, tsne_exact, update_alpha,
)


def make_data_and_model(n, seed=1):
    """Three well-separated heterogeneous clusters and a mixture fit to them."""
    comps = [
        CompositeDistribution((GaussianDistribution(mu, 1.0),
                               CategoricalDistribution(pm)))
        for mu, pm in [(-12.0, {'a': 0.8, 'b': 0.1, 'c': 0.1}),
                       (0.0, {'a': 0.1, 'b': 0.8, 'c': 0.1}),
                       (12.0, {'a': 0.1, 'b': 0.1, 'c': 0.8})]
    ]
    truth = MixtureDistribution(comps, [1.0 / 3] * 3)
    data = truth.sampler(seed=seed).sample(size=n)
    labels = np.argmin(np.abs(np.subtract.outer([x[0] for x in data], [-12.0, 0.0, 12.0])), axis=1)

    est = MixtureEstimator([CompositeEstimator((GaussianEstimator(), CategoricalEstimator()))] * 3)
    enc = seq_encode(data, model=truth)
    model = seq_initialize(enc, est, np.random.RandomState(1), p=1.0)
    for _ in range(25):
        model = seq_estimate(enc, est, model)

    return data, labels, model


def separation_ratio(y, labels):
    """Mean between-centroid distance over mean within-cluster spread."""
    cents = np.stack([y[labels == c].mean(axis=0) for c in np.unique(labels)])
    within = np.mean([np.linalg.norm(y[labels == c] - cents[i], axis=1).mean()
                      for i, c in enumerate(np.unique(labels))])
    between = np.mean([np.linalg.norm(cents[i] - cents[j])
                       for i in range(len(cents)) for j in range(i + 1, len(cents))])
    return between / max(within, 1.0e-12)


class HTSNETestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 240
        cls.data, cls.labels, cls.model = make_data_and_model(cls.n)
        from pysp.utils.htsne import _posteriors_and_loglikes
        cls.z, cls.l = _posteriors_and_loglikes(cls.model, data=cls.data)

    # ---- affinity construction -------------------------------------------------

    def test_log_affinity_shape_and_diag(self):
        for aff in ('coassign', 'bhattacharyya', 'likelihood'):
            log_s = model_log_affinity(self.z, self.l, affinity=aff)
            self.assertEqual(log_s.shape, (self.n, self.n))
            self.assertTrue(np.all(np.isneginf(np.diag(log_s))))
            off = log_s[~np.eye(self.n, dtype=bool)]
            self.assertFalse(np.any(np.isnan(off)))
            self.assertFalse(np.any(np.isposinf(off)))

    def test_coassign_is_posterior_similarity(self):
        # s_ij = P(z_i = z_j | x) = sum_k z_ik z_jk: exact, symmetric, in [0, 1]
        log_s = model_log_affinity(self.z, affinity='coassign')
        s = np.exp(log_s)
        expected = np.dot(self.z, self.z.T)
        expected[np.arange(self.n), np.arange(self.n)] = 0.0
        self.assertTrue(np.allclose(s, expected, atol=1.0e-12))
        self.assertTrue(np.allclose(s, s.T))
        self.assertTrue(np.all((s >= 0) & (s <= 1 + 1.0e-12)))

    def test_bhattacharyya_bounds_coassign(self):
        # BC >= co-assignment probability (Cauchy-Schwarz), both <= 1
        s_co = np.exp(model_log_affinity(self.z, affinity='coassign'))
        s_bc = np.exp(model_log_affinity(self.z, affinity='bhattacharyya'))
        self.assertTrue(np.all(s_bc >= s_co - 1.0e-12))
        self.assertTrue(np.all(s_bc <= 1 + 1.0e-12))

    def test_conditional_rows_normalized(self):
        log_s = model_log_affinity(self.z, self.l)
        for px in (None, 20.0):
            p = conditional_pmat(log_s, perplexity=px)
            self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8))
            self.assertTrue(np.all(np.diag(p) == 0.0))
            self.assertTrue(np.all(p >= 0.0))

    @staticmethod
    def _row_entropies(p):
        with np.errstate(divide='ignore', invalid='ignore'):
            lp = np.where(p > 0, np.log(p), 0.0)
        return -np.sum(p * lp, axis=1)

    def test_perplexity_calibration_hits_target(self):
        # tie-free synthetic affinities: the binary search must hit the target exactly
        rng = np.random.RandomState(7)
        m = 150
        log_aff = rng.randn(m, m) * 3.0
        log_aff[np.arange(m), np.arange(m)] = -np.inf
        for px in (5.0, 20.0, 50.0):
            ent = self._row_entropies(conditional_pmat(log_aff, perplexity=px))
            self.assertTrue(np.allclose(ent, np.log(px), atol=5.0e-3),
                            'max entropy err %g for perplexity %g' % (np.abs(ent - np.log(px)).max(), px))

    def test_perplexity_calibration_saturates_on_ties(self):
        # model affinities from a K-component mixture are low-rank, so rows have
        # large tie groups; calibration must saturate gracefully (no NaN, never
        # sharper than requested by more than tolerance, normalized rows)
        log_s = model_log_affinity(self.z, self.l)
        p = conditional_pmat(log_s, perplexity=5.0)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8))
        self.assertTrue(np.all(self._row_entropies(p) >= np.log(5.0) - 5.0e-3))

    def test_get_pmat_symmetric_and_normalized(self):
        for vlen in (False, True):
            p = get_pmat(self.z, self.l, targ_perplexity=20.0, vlen=vlen)
            self.assertTrue(np.allclose(p, p.T))
            self.assertAlmostEqual(p.sum(), 1.0, places=10)

    def test_row_scale_invariance(self):
        # adding per-row offsets to the log-likelihood matrix (e.g. from
        # variable-length observations) must not change the conditionals
        shift = np.random.RandomState(0).uniform(-50, 50, size=(self.n, 1))
        p0 = get_pmat(self.z, self.l, targ_perplexity=None, affinity='likelihood')
        p1 = get_pmat(self.z, self.l + shift, targ_perplexity=None, affinity='likelihood')
        self.assertTrue(np.allclose(p0, p1, atol=1.0e-12))

        # and with calibration, on tie-free affinities
        rng = np.random.RandomState(3)
        m = 100
        log_aff = rng.randn(m, m) * 3.0
        log_aff[np.arange(m), np.arange(m)] = -np.inf
        row_shift = rng.uniform(-50, 50, size=(m, 1))
        c0 = conditional_pmat(log_aff, perplexity=15.0)
        c1 = conditional_pmat(log_aff + row_shift, perplexity=15.0)
        self.assertTrue(np.allclose(c0, c1, atol=1.0e-8))

    def test_sparse_distances_match_dense(self):
        k = 10
        d_csr = sparse_model_distances(self.z, self.l, k=k, block_size=64)
        self.assertEqual(d_csr.shape, (self.n, self.n))
        self.assertTrue(np.all(d_csr.getnnz(axis=1) == k))
        self.assertTrue(np.all(d_csr.data >= 0.0))

        # neighbor *identities* are arbitrary among model-affinity ties, so
        # compare the selected affinity values against the dense top-k values
        log_s = model_log_affinity(self.z, self.l)
        for i in (0, 17, self.n - 1):
            dense_vals = np.sort(log_s[i])[::-1][:k]
            sparse_vals = np.sort(log_s[i, d_csr[i].indices])[::-1]
            self.assertTrue(np.allclose(dense_vals, sparse_vals, atol=1.0e-9))

    # ---- kernel and alpha ------------------------------------------------------

    def test_kernel_standard_tsne_at_alpha_one(self):
        rng = np.random.RandomState(0)
        y = rng.randn(40, 2)
        q, num, d2 = t_kernel(y, 1.0)
        qt = 1.0 / (1.0 + d2)
        qt[np.arange(40), np.arange(40)] = 0.0
        self.assertTrue(np.allclose(q, qt / qt.sum()))
        self.assertAlmostEqual(q.sum(), 1.0, places=12)

    def test_update_alpha_does_not_increase_kl(self):
        rng = np.random.RandomState(0)
        y = rng.randn(60, 2)
        p = get_pmat(self.z[:60], self.l[:60], targ_perplexity=10.0)
        q0, _, _ = t_kernel(y, 1.0)
        m = (p > 0) & (q0 > 0)
        kl0 = np.dot(p[m], np.log(p[m]) - np.log(q0[m]))
        a1 = update_alpha(p, y, 1.0, 1.0e-6, 1.0e-128, max_its=10)
        q1, _, _ = t_kernel(y, a1)
        kl1 = np.dot(p[m], np.log(p[m]) - np.log(q1[m]))
        self.assertLessEqual(kl1, kl0 + 1.0e-9)

    # ---- end-to-end embeddings -------------------------------------------------

    def test_exact_embedding_converges_and_separates(self):
        t0 = time.time()
        y = htsne(self.data, mix_model=self.model, perplexity=20.0, method='exact',
                  max_its=400, seed=3, out=io.StringIO())
        self.assertLess(time.time() - t0, 30.0)
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 3.0)

    def test_exact_kl_decreases(self):
        p = get_pmat(self.z, self.l, targ_perplexity=20.0)
        buf = io.StringIO()
        tsne_exact(p, max_its=300, seed=3, print_iter=50, out=buf)
        kls = [float(line.rsplit('=', 1)[1]) for line in buf.getvalue().strip().split('\n')]
        self.assertGreater(len(kls), 2)
        self.assertLess(kls[-1], kls[0])
        self.assertLess(kls[-1], 1.0)

    def test_barnes_hut_embedding(self):
        t0 = time.time()
        y = htsne(self.data, mix_model=self.model, perplexity=20.0, method='barnes_hut',
                  max_its=350, seed=3, out=io.StringIO())
        self.assertLess(time.time() - t0, 60.0)
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 3.0)

    def test_optimize_alpha_path(self):
        y = htsne(self.data[:100], mix_model=self.model, perplexity=10.0, method='exact',
                  optimize_alpha=True, max_its=300, seed=3, out=io.StringIO())
        self.assertEqual(y.shape, (100, 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_dpmsne_precomputed(self):
        p = get_pmat(self.z, self.l, targ_perplexity=20.0)
        y = dpmsne(P=p, max_its=300, seed=3, out=io.StringIO())
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 3.0)

    # ---- model kNN and UMAP ------------------------------------------------------

    def test_model_knn_properties(self):
        k = 12
        idx, dist = model_knn(self.z, self.l, k=k, block_size=64)
        self.assertEqual(idx.shape, (self.n, k))
        self.assertTrue(np.all(idx[:, 0] == np.arange(self.n)))
        self.assertTrue(np.all(dist[:, 0] == 0.0))
        self.assertTrue(np.all(np.diff(dist[:, 1:], axis=1) >= -1.0e-12))
        self.assertTrue(np.all(dist >= 0.0))

    def test_humap_embedding(self):
        y = humap(self.data, mix_model=self.model, n_neighbors=15, seed=4, out=io.StringIO())
        self.assertEqual(y.shape, (self.n, 2))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(separation_ratio(y, self.labels), 2.0)

    # ---- variable-length handling ------------------------------------------------

    @classmethod
    def _varlen_data_and_model(cls, n_per=120, seed=2):
        # two topics over integers; lengths vary wildly *within* each topic
        len_probs = np.zeros(60)
        len_probs[2:60] = 1.0
        len_probs /= len_probs.sum()
        len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=len_probs)
        topic_a = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.85, 0.05, 0.05, 0.05]), len_dist=len_dist)
        topic_b = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.05, 0.05, 0.05, 0.85]), len_dist=len_dist)

        data = topic_a.sampler(seed=seed).sample(size=n_per) + \
            topic_b.sampler(seed=seed + 1).sample(size=n_per)
        labels = np.repeat([0, 1], n_per)
        model = MixtureDistribution([topic_a, topic_b], [0.5, 0.5])
        return data, labels, model

    def test_varlen_affinity_is_length_free(self):
        # with co-assignment affinities, the only length effect is posterior
        # certainty: two long observations of the same topic and two short ones
        # of the same topic should both have near-1 affinity
        data, labels, model = self._varlen_data_and_model(60)
        from pysp.utils.htsne import _posteriors_and_loglikes
        z, _ = _posteriors_and_loglikes(model, data=data)
        s = np.dot(z, z.T)
        lengths = np.asarray([len(x) for x in data])
        same_topic = np.equal.outer(labels, labels)
        long_pairs = np.greater.outer(lengths, 20) & np.greater.outer(lengths, 20).T
        m = same_topic & long_pairs & ~np.eye(len(data), dtype=bool)
        if np.any(m):
            self.assertGreater(s[m].mean(), 0.9)

    def test_varlen_embedding_organizes_by_topic(self):
        data, labels, model = self._varlen_data_and_model(120)
        lengths = np.asarray([len(x) for x in data], dtype=float)

        y = htsne(data, mix_model=model, perplexity=20.0, method='exact', max_its=350,
                  seed=3, out=io.StringIO())
        self.assertGreater(separation_ratio(y, labels), 2.0)

        # the embedding should not be organized by observation length: within
        # each topic, short and long observations should overlap
        for c in (0, 1):
            yc, lc = y[labels == c], lengths[labels == c]
            short, long_ = yc[lc <= np.median(lc)], yc[lc > np.median(lc)]
            gap = np.linalg.norm(short.mean(0) - long_.mean(0))
            spread = 0.5 * (np.linalg.norm(short - short.mean(0), axis=1).mean()
                            + np.linalg.norm(long_ - long_.mean(0), axis=1).mean())
            self.assertLess(gap, 2.0 * spread)

    def test_varlen_humap(self):
        data, labels, model = self._varlen_data_and_model(120)
        y = humap(data, mix_model=model, n_neighbors=15, seed=4, out=io.StringIO())
        self.assertEqual(y.shape, (len(data), 2))
        self.assertGreater(separation_ratio(y, labels), 1.5)


    # ---- balanced (mixed-type) affinities -----------------------------------

    def test_balanced_factors_structure(self):
        from pysp.utils.htsne import balanced_factors, model_log_affinity
        factors = balanced_factors(self.model, self.data)
        self.assertEqual(len(factors), 2)  # gaussian + categorical fields
        # log-affinity over factors = sum of per-field Bhattacharyya logs
        la = model_log_affinity(None, None, affinity=factors)
        self.assertEqual(la.shape, (self.n, self.n))
        self.assertTrue(np.all(np.isneginf(np.diag(la))))
        # symmetric: each factor is (sq, sq)
        off = ~np.eye(self.n, dtype=bool)
        self.assertTrue(np.allclose(la[off], la.T[off], atol=1.0e-10))

    def test_balanced_embeddings_run(self):
        y = htsne(self.data, mix_model=self.model, perplexity=20.0, affinity='balanced',
                  seed=3, max_its=300, out=io.StringIO())
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 2.0)
        yu = humap(self.data, mix_model=self.model, n_neighbors=15, affinity='balanced',
                   seed=3, out=io.StringIO())
        self.assertTrue(np.all(np.isfinite(yu)))

    def test_auto_affinity_resolution(self):
        from pysp.utils.htsne import _resolve_affinity
        # composite components + raw data -> balanced factor list (one per leaf field)
        r = _resolve_affinity('auto', self.model, self.data, None)
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), 2)
        # no raw data -> falls back to bhattacharyya
        self.assertEqual(_resolve_affinity('auto', self.model, None, None), 'bhattacharyya')
        # non-composite components decompose too: a plain mixture is one leaf field
        plain = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
                                    [0.5, 0.5])
        r = _resolve_affinity('auto', plain, [0.0, 1.0], None)
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), 1)

    def test_balanced_requires_data(self):
        with self.assertRaises(ValueError):
            htsne(None, mix_model=self.model, enc_data=('x',), affinity='balanced', out=io.StringIO())

    def test_balanced_decomposes_sequences_and_optionals(self):
        # sequence-of-records components flatten into element fields + length;
        # the old implementation refused anything without top-level .dists
        from pysp.utils.htsne import balanced_factors
        data, labels, model = self._varlen_data_and_model(60)
        factors = balanced_factors(model, data)
        self.assertGreater(len(factors), 1)
        n = len(data)
        for g, h in factors:
            self.assertEqual(g.shape[0], n)
            # per-field posteriors: squared factors sum to one per row
            self.assertTrue(np.allclose((g * g).sum(axis=1), 1.0, atol=1.0e-8))
        y = htsne(data, mix_model=model, seed=3, max_its=200, out=io.StringIO())
        self.assertEqual(y.shape, (n, 2))
        self.assertGreater(separation_ratio(y, labels), 1.5)

    def test_evidence_cap_bounds_field_influence(self):
        from pysp.utils.htsne import balanced_factors, model_log_affinity
        factors = balanced_factors(self.model, self.data)
        la_inf = model_log_affinity(None, None, affinity=factors)
        cap = 1.0
        la_cap = model_log_affinity(None, None, affinity=factors, evidence_cap=cap)
        off = ~np.eye(self.n, dtype=bool)
        # capped log-affinity is bounded below by -cap * n_fields and never
        # smaller than the uncapped one
        self.assertTrue(np.all(la_cap[off] >= -cap * len(factors) - 1.0e-12))
        self.assertTrue(np.all(la_cap[off] >= la_inf[off] - 1.0e-12))
        # pairs within every field's cap are unaffected
        mild = la_inf[off] > -0.5
        if mild.any():
            self.assertTrue(np.allclose(la_cap[off][mild], la_inf[off][mild], atol=1.0e-10))
        # single-factor affinities ignore the cap (it could only create ties)
        from pysp.utils.htsne import _posteriors_and_loglikes
        z, _ = _posteriors_and_loglikes(self.model, data=self.data)
        la1 = model_log_affinity(z, affinity='bhattacharyya')
        la1c = model_log_affinity(z, affinity='bhattacharyya', evidence_cap=cap)
        self.assertTrue(np.allclose(np.nan_to_num(la1, neginf=-1e30),
                                    np.nan_to_num(la1c, neginf=-1e30)))

    def test_field_weights_validated_against_flattened_fields(self):
        from pysp.utils.htsne import balanced_factors
        with self.assertRaises(ValueError):
            balanced_factors(self.model, self.data, field_weights=[1.0])


if __name__ == '__main__':
    unittest.main()
