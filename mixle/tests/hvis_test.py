"""Tests for model-based t-SNE (mixle.utils.hvis).

Kept fast by fitting a small mixture once in setUpClass and reusing it; no DPM
fitting is exercised here (htsne accepts a prefit mix_model).
"""

import importlib
import io
import time
import unittest

import numpy as np
import scipy.sparse

from mixle.inference import seq_estimate, seq_initialize
from mixle.inference.fisher import FisherView
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    HeterogeneousPCFGDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    MixtureEstimator,
    OptionalDistribution,
    PoissonDistribution,
    SequenceDistribution,
    seq_encode,
)
from mixle.utils.hvis import (
    _barnes_hut_negative_forces,
    _exact_negative_forces,
    _exact_tsne_gradient,
    _kl,
    _posteriors_and_loglikes,
    _python_barnes_hut_negative_forces,
    _sparse_joint_pmat,
    _sparse_positive_forces_from_edges,
    _sparse_positive_forces_symmetric_from_edges,
    _tsne_barnes_hut_from_p,
    approx_sparse_model_distances,
    balanced_factors,
    conditional_pmat,
    dpmsne,
    fisher_factors,
    get_pmat,
    htsne,
    humap,
    local_factors,
    model_knn,
    model_log_affinity,
    sparse_model_distances,
    t_kernel,
    tsne_exact,
    update_alpha,
    update_embed,
)
from mixle.utils.optional_deps import HAS_NUMBA

HAS_UMAP = importlib.util.find_spec("umap") is not None


def make_data_and_model(n, seed=1):
    """Three well-separated heterogeneous clusters and a mixture fit to them."""
    comps = [
        CompositeDistribution((GaussianDistribution(mu, 1.0), CategoricalDistribution(pm)))
        for mu, pm in [
            (-12.0, {"a": 0.8, "b": 0.1, "c": 0.1}),
            (0.0, {"a": 0.1, "b": 0.8, "c": 0.1}),
            (12.0, {"a": 0.1, "b": 0.1, "c": 0.8}),
        ]
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
    within = np.mean(
        [np.linalg.norm(y[labels == c] - cents[i], axis=1).mean() for i, c in enumerate(np.unique(labels))]
    )
    between = np.mean(
        [np.linalg.norm(cents[i] - cents[j]) for i in range(len(cents)) for j in range(i + 1, len(cents))]
    )
    return between / max(within, 1.0e-12)


def affinity_neighbor_purity(log_s, labels, k=10):
    labels = np.asarray(labels)
    vals = []
    for i in range(len(labels)):
        nbr = np.argsort(log_s[i])[::-1][:k]
        vals.append(np.mean(labels[nbr] == labels[i]))
    return float(np.mean(vals))


def embedding_neighbor_purity(y, labels, k=10):
    labels = np.asarray(labels)
    d2 = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    vals = []
    for i in range(len(labels)):
        nbr = np.argsort(d2[i])[:k]
        vals.append(np.mean(labels[nbr] == labels[i]))
    return float(np.mean(vals))


class _VandermondeAccumulator:
    def __init__(self, degree):
        self.degree = int(degree)
        self.stats = np.zeros(self.degree + 1, dtype=np.float64)

    def update(self, x, weight, estimate):
        self.stats += float(weight) * np.power(float(x), np.arange(self.degree + 1))

    def value(self):
        return self.stats.copy()


class _VandermondeAccumulatorFactory:
    def __init__(self, degree):
        self.degree = int(degree)

    def make(self):
        return _VandermondeAccumulator(self.degree)


class _VandermondeEstimator:
    def __init__(self, degree):
        self.degree = int(degree)

    def accumulator_factory(self):
        return _VandermondeAccumulatorFactory(self.degree)


class _VandermondeMomentModel:
    def __init__(self, degree):
        self.degree = int(degree)

    def estimator(self):
        return _VandermondeEstimator(self.degree)

    def to_fisher(self):
        return FisherView(self)


class HTSNETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 240
        cls.data, cls.labels, cls.model = make_data_and_model(cls.n)
        from mixle.utils.hvis import _posteriors_and_loglikes

        cls.z, cls.l = _posteriors_and_loglikes(cls.model, data=cls.data)

    # ---- affinity construction -------------------------------------------------

    def test_log_affinity_shape_and_diag(self):
        for aff in ("coassign", "bhattacharyya", "likelihood"):
            log_s = model_log_affinity(self.z, self.l, affinity=aff)
            self.assertEqual(log_s.shape, (self.n, self.n))
            self.assertTrue(np.all(np.isneginf(np.diag(log_s))))
            off = log_s[~np.eye(self.n, dtype=bool)]
            self.assertFalse(np.any(np.isnan(off)))
            self.assertFalse(np.any(np.isposinf(off)))

    def test_coassign_is_posterior_similarity(self):
        # s_ij = P(z_i = z_j | x) = sum_k z_ik z_jk: exact, symmetric, in [0, 1]
        log_s = model_log_affinity(self.z, affinity="coassign")
        s = np.exp(log_s)
        expected = np.dot(self.z, self.z.T)
        expected[np.arange(self.n), np.arange(self.n)] = 0.0
        self.assertTrue(np.allclose(s, expected, atol=1.0e-12))
        self.assertTrue(np.allclose(s, s.T))
        self.assertTrue(np.all((s >= 0) & (s <= 1 + 1.0e-12)))

    def test_bhattacharyya_bounds_coassign(self):
        # BC >= co-assignment probability (Cauchy-Schwarz), both <= 1
        s_co = np.exp(model_log_affinity(self.z, affinity="coassign"))
        s_bc = np.exp(model_log_affinity(self.z, affinity="bhattacharyya"))
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
        with np.errstate(divide="ignore", invalid="ignore"):
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
            self.assertTrue(
                np.allclose(ent, np.log(px), atol=5.0e-3),
                "max entropy err %g for perplexity %g" % (np.abs(ent - np.log(px)).max(), px),
            )

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
        p0 = get_pmat(self.z, self.l, targ_perplexity=None, affinity="likelihood")
        p1 = get_pmat(self.z, self.l + shift, targ_perplexity=None, affinity="likelihood")
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
            self.assertTrue(np.allclose(d_csr[i].data, np.maximum(-log_s[i, d_csr[i].indices], 0.0), atol=1.0e-12))

    def test_local_affinity_resolves_within_component_geometry(self):
        data = [-3.0, 0.0, 0.1, 4.0]
        model = MixtureDistribution([GaussianDistribution(0.0, 4.0)], [1.0])
        factors = local_factors(model, data)
        self.assertEqual(len(factors), 1)
        self.assertIsInstance(factors[0], dict)

        log_s = model_log_affinity(None, None, affinity=factors)
        self.assertGreater(log_s[1, 2], log_s[1, 3])

        d_csr = sparse_model_distances(None, None, k=1, affinity=factors)
        self.assertEqual(d_csr[1].indices[0], 2)

    def test_fisher_affinity_resolves_within_component_geometry(self):
        data = [-3.0, 0.0, 0.1, 4.0]
        model = MixtureDistribution([GaussianDistribution(0.0, 4.0)], [1.0])
        factors = fisher_factors(model, data=data, metric="diagonal")
        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0]["kind"], "fisher")

        log_s = model_log_affinity(None, None, affinity=factors)
        self.assertGreater(log_s[1, 2], log_s[1, 3])

        d_csr = sparse_model_distances(None, None, k=1, affinity=factors)
        self.assertEqual(d_csr[1].indices[0], 2)

    def test_fisher_affinity_uses_fisher_vectors(self):
        data = self.data[:35]
        factors = fisher_factors(self.model, data=data, metric="diagonal")
        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0]["kind"], "fisher")
        self.assertEqual(factors[0]["x"].shape[0], len(data))

        log_s = model_log_affinity(None, None, affinity=factors)
        self.assertEqual(log_s.shape, (len(data), len(data)))
        self.assertTrue(np.all(np.isneginf(np.diag(log_s))))
        self.assertTrue(np.all(np.isfinite(log_s[~np.eye(len(data), dtype=bool)])))
        self.assertTrue(np.allclose(log_s, log_s.T))

        p = get_pmat(None, None, targ_perplexity=8.0, affinity=factors)
        self.assertTrue(np.allclose(p, p.T))
        self.assertAlmostEqual(p.sum(), 1.0, places=10)

        d_csr = sparse_model_distances(None, None, k=5, block_size=11, affinity=factors)
        self.assertEqual(d_csr.shape, (len(data), len(data)))
        self.assertTrue(np.all(d_csr.getnnz(axis=1) == 5))
        self.assertTrue(np.all(d_csr.data >= 0.0))

    def test_fisher_affinity_raw_and_encoded_match(self):
        data = self.data[:35]
        enc = self.model.dist_to_encoder().seq_encode(data)
        f_raw = fisher_factors(self.model, data=data, metric="diagonal")
        f_enc = fisher_factors(self.model, enc_data=enc, metric="diagonal")

        np.testing.assert_allclose(f_enc[0]["x"], f_raw[0]["x"], atol=1.0e-10)
        np.testing.assert_allclose(
            model_log_affinity(None, None, affinity=f_enc), model_log_affinity(None, None, affinity=f_raw), atol=1.0e-10
        )

    def test_fisher_affinity_uses_observed_score_covariance(self):
        data = self.data[:40]
        ridge = 1.0e-8
        view = self.model.to_fisher()
        stats = view.expected_statistics_matrix(data=data)
        center = view._model_mean()
        centered = stats - center.reshape((1, -1))
        diag = np.mean(centered * centered, axis=0)
        expected = centered / np.sqrt(diag.reshape((1, -1)) + ridge)

        factors = fisher_factors(self.model, data=data, metric="diagonal", ridge=ridge, information="observed")
        np.testing.assert_allclose(factors[0]["x"], expected, atol=1.0e-10)

        model_factors = fisher_factors(self.model, data=data, metric="diagonal", ridge=ridge, information="model")
        self.assertFalse(np.allclose(model_factors[0]["x"], factors[0]["x"]))

    def test_fisher_affinity_is_gaussian_kernel_on_fisher_vectors(self):
        data = np.linspace(-2.0, 2.0, 7)
        factors = fisher_factors(GaussianDistribution(0.0, 2.0), data=data, metric="diagonal")
        log_s = model_log_affinity(None, None, affinity=factors)

        x = factors[0]["x"]
        d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
        expected = -0.5 * d2
        expected[np.arange(len(data)), np.arange(len(data))] = -np.inf

        finite = np.isfinite(expected)
        np.testing.assert_allclose(log_s[finite], expected[finite], atol=1.0e-12)

        p0 = get_pmat(None, None, targ_perplexity=3.0, affinity=factors)
        p1 = conditional_pmat(expected, perplexity=3.0)
        p1 = (p1 + p1.T) / (2.0 * len(data))
        np.testing.assert_allclose(p0, p1, atol=1.0e-10)

    def test_fisher_affinity_does_not_require_dpm_or_mixture_model(self):
        model = CompositeDistribution((GaussianDistribution(0.0, 1.0), PoissonDistribution(3.0)))
        data = [(-2.0, 1), (-1.8, 1), (-0.1, 3), (0.1, 3), (2.0, 8), (2.2, 8)]
        factors = fisher_factors(model, data=data, metric="diagonal")
        log_s = model_log_affinity(None, None, affinity=factors)

        self.assertGreater(log_s[0, 1], log_s[0, 5])

        y_model = htsne(
            data,
            mix_model=model,
            affinity="fisher",
            method="exact",
            perplexity=2.0,
            max_its=5,
            seed=10,
            out=io.StringIO(),
        )
        self.assertEqual(y_model.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y_model)))

        y_prebuilt = htsne(
            None,
            mix_model=None,
            affinity=factors,
            method="exact",
            perplexity=2.0,
            max_its=5,
            seed=10,
            out=io.StringIO(),
        )
        self.assertEqual(y_prebuilt.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y_prebuilt)))

    def test_fisher_affinity_supports_pcfg_model(self):
        model = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 0.45), ("B", "A", 0.55)]},
            terminal_rules={
                "A": [(CategoricalDistribution({"a": 0.75, "b": 0.25}), 1.0)],
                "B": [(CategoricalDistribution({"x": 0.7, "y": 0.3}), 1.0)],
            },
            start="S",
        )
        data = [["a", "x"], ["a", "y"], ["b", "x"], ["x", "a"], ["y", "a"], ["x", "b"]]
        enc = model.dist_to_encoder().seq_encode(data)

        f_raw = fisher_factors(model, data=data, metric="diagonal")
        f_enc = fisher_factors(model, enc_data=enc, metric="diagonal")
        np.testing.assert_allclose(f_enc[0]["x"], f_raw[0]["x"], atol=1.0e-10)

        log_s = model_log_affinity(None, None, affinity=f_raw)
        self.assertEqual(log_s.shape, (len(data), len(data)))
        self.assertTrue(np.all(np.isfinite(log_s[~np.eye(len(data), dtype=bool)])))

        y = htsne(
            data,
            mix_model=model,
            affinity="fisher",
            method="exact",
            perplexity=3.0,
            max_its=5,
            print_iter=1000,
            seed=8,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_fisher_affinity_supports_hmm_model(self):
        model = HiddenMarkovModelDistribution(
            [
                CategoricalDistribution({"a": 0.85, "b": 0.15}),
                CategoricalDistribution({"a": 0.2, "b": 0.8}),
            ],
            [0.6, 0.4],
            [[0.75, 0.25], [0.2, 0.8]],
            len_dist=IntegerCategoricalDistribution(2, [1.0]),
        )
        data = [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"], ["a", "a"], ["b", "b"]]
        enc = model.dist_to_encoder().seq_encode(data)

        f_raw = fisher_factors(model, data=data, metric="diagonal")
        f_enc = fisher_factors(model, enc_data=enc, metric="diagonal")
        np.testing.assert_allclose(f_enc[0]["x"], f_raw[0]["x"], atol=1.0e-10)

        log_s = model_log_affinity(None, None, affinity=f_raw)
        self.assertEqual(log_s.shape, (len(data), len(data)))
        self.assertTrue(np.all(np.isfinite(log_s[~np.eye(len(data), dtype=bool)])))

        y = htsne(
            data,
            mix_model=model,
            affinity="fisher",
            method="exact",
            perplexity=3.0,
            max_its=5,
            print_iter=1000,
            seed=9,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_fisher_hard_assignments_match_coassign_partition(self):
        data = ["a", "a", "b", "b", "a", "b"]
        labels = np.asarray([0 if x == "a" else 1 for x in data])
        model = MixtureDistribution(
            [
                CategoricalDistribution({"a": 1.0, "b": 0.0}),
                CategoricalDistribution({"a": 0.0, "b": 1.0}),
            ],
            [0.5, 0.5],
        )

        with np.errstate(divide="ignore"):
            z, l = _posteriors_and_loglikes(model, data=data)
            log_co = model_log_affinity(z, l, affinity="coassign")
            log_bh = model_log_affinity(z, l, affinity="bhattacharyya")
            factors = fisher_factors(model, data=data, metric="diagonal")
            log_f = model_log_affinity(None, None, affinity=factors)

        np.testing.assert_allclose(np.exp(log_co), np.exp(log_bh), atol=1.0e-12)

        off = ~np.eye(len(data), dtype=bool)
        same = (labels[:, None] == labels[None, :]) & off
        diff = labels[:, None] != labels[None, :]
        np.testing.assert_allclose(log_f[same], 0.0, atol=1.0e-12)
        self.assertLess(float(log_f[diff].max()), float(log_f[same].min()))

        d_csr = sparse_model_distances(None, None, k=1, affinity=factors)
        for i in range(len(data)):
            self.assertEqual(labels[d_csr[i].indices[0]], labels[i])

    def test_htsne_fisher_affinity_exact_smoke(self):
        data = self.data[:30]
        y = htsne(
            data,
            mix_model=self.model,
            method="exact",
            affinity="fisher",
            perplexity=8.0,
            max_its=5,
            print_iter=1000,
            seed=4,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_htsne_fisher_affinity_encoded_exact_smoke(self):
        data = self.data[:30]
        enc = self.model.dist_to_encoder().seq_encode(data)
        y = htsne(
            None,
            mix_model=self.model,
            enc_data=enc,
            method="exact",
            affinity="fisher",
            perplexity=8.0,
            max_its=5,
            print_iter=1000,
            seed=4,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_fisher_model_knn_matches_dense_affinity(self):
        data = self.data[:40]
        factors = fisher_factors(self.model, data=data, metric="diagonal")
        log_s = model_log_affinity(None, None, affinity=factors)

        idx, dist = model_knn(None, None, k=7, affinity=factors, block_size=13)
        self.assertEqual(idx.shape, (len(data), 7))
        self.assertTrue(np.all(idx[:, 0] == np.arange(len(data))))
        for i in (0, 11, len(data) - 1):
            np.testing.assert_allclose(dist[i, 1:], np.maximum(-log_s[i, idx[i, 1:]], 0.0), atol=1.0e-12)

    def test_fisher_approx_sparse_distances_can_reduce_to_exact(self):
        data = self.data[:45]
        factors = fisher_factors(self.model, data=data, metric="diagonal")
        k = 8
        d_csr = approx_sparse_model_distances(
            None, None, k=k, affinity=factors, n_trees=1, leaf_size=len(data), candidate_multiplier=1, seed=6
        )
        self.assertEqual(d_csr.shape, (len(data), len(data)))
        self.assertTrue(np.all(d_csr.getnnz(axis=1) == k))

        log_s = model_log_affinity(None, None, affinity=factors)
        for i in (0, 13, len(data) - 1):
            dense_vals = np.sort(log_s[i])[::-1][:k]
            sparse_vals = np.sort(log_s[i, d_csr[i].indices])[::-1]
            np.testing.assert_allclose(sparse_vals, dense_vals, atol=1.0e-9)

    def test_fisher_information_mode_is_validated(self):
        with self.assertRaises(ValueError):
            fisher_factors(self.model, data=self.data[:5], information="complete")

    def test_approx_sparse_distances_can_reduce_to_exact(self):
        k = 10
        d_csr = approx_sparse_model_distances(
            self.z, self.l, k=k, n_trees=1, leaf_size=self.n, candidate_multiplier=1, seed=5
        )
        self.assertEqual(d_csr.shape, (self.n, self.n))
        self.assertTrue(np.all(d_csr.getnnz(axis=1) == k))

        log_s = model_log_affinity(self.z, self.l)
        for i in (0, 17, self.n - 1):
            dense_vals = np.sort(log_s[i])[::-1][:k]
            sparse_vals = np.sort(log_s[i, d_csr[i].indices])[::-1]
            self.assertTrue(np.allclose(dense_vals, sparse_vals, atol=1.0e-9))

    def test_sparse_joint_pmat_matches_dense_when_graph_is_complete(self):
        rng = np.random.RandomState(9)
        n = 45
        log_s = -rng.gamma(shape=2.0, scale=1.0, size=(n, n))
        log_s[np.arange(n), np.arange(n)] = -np.inf
        rows, cols = np.where(~np.eye(n, dtype=bool))
        dist = -log_s[rows, cols]
        d_csr = scipy.sparse.csr_matrix((dist, (rows, cols)), shape=(n, n))

        p_sparse = _sparse_joint_pmat(d_csr, perplexity=12.0).toarray()
        p_dense = conditional_pmat(log_s, perplexity=12.0)
        p_dense = (p_dense + p_dense.T) / (2.0 * n)

        self.assertTrue(np.allclose(p_sparse, p_dense, atol=1.0e-10))

    def test_barnes_hut_theta_zero_matches_exact_repulsion(self):
        rng = np.random.RandomState(11)
        y = rng.randn(35, 2)
        f_bh, z_bh = _barnes_hut_negative_forces(y, theta=0.0, leaf_size=1)

        f_exact = np.zeros_like(y)
        z_exact = 0.0
        for i in range(y.shape[0]):
            diff = y[i] - y
            d2 = np.sum(diff * diff, axis=1)
            q = 1.0 / (1.0 + d2)
            q[i] = 0.0
            f_exact[i] = np.sum((q * q)[:, None] * diff, axis=0)
            z_exact += q.sum()

        self.assertTrue(np.allclose(f_bh, f_exact, atol=1.0e-12))
        self.assertAlmostEqual(z_bh, z_exact, places=10)

    def test_vectorized_exact_repulsion_matches_tree_exact(self):
        rng = np.random.RandomState(12)
        y = rng.randn(50, 2)
        f_vec, z_vec = _exact_negative_forces(y)
        f_tree, z_tree = _barnes_hut_negative_forces(y, theta=0.0, leaf_size=1)
        self.assertTrue(np.allclose(f_vec, f_tree, atol=1.0e-12))
        self.assertAlmostEqual(z_vec, z_tree, places=10)

    def test_barnes_hut_exact_mode_matches_dense_leaves(self):
        rng = np.random.RandomState(13)
        y = rng.randn(60, 2)
        f_vec, z_vec = _exact_negative_forces(y)
        f_tree, z_tree = _barnes_hut_negative_forces(y, theta=0.0, leaf_size=9)
        np.testing.assert_allclose(f_tree, f_vec, atol=1.0e-12)
        self.assertAlmostEqual(z_tree, z_vec, places=10)

    def test_barnes_hut_approximation_is_close_to_exact_at_default_theta(self):
        rng = np.random.RandomState(17)
        y = rng.randn(150, 2)
        f_exact, z_exact = _exact_negative_forces(y)
        f_bh, z_bh = _barnes_hut_negative_forces(y, theta=0.5, leaf_size=8)

        rel_z = abs(z_bh - z_exact) / z_exact
        rel_force = np.linalg.norm(f_bh - f_exact) / np.linalg.norm(f_exact)
        self.assertLess(rel_z, 0.01)
        self.assertLess(rel_force, 0.02)

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_barnes_hut_numba_matches_python_fallback(self):
        rng = np.random.RandomState(19)
        y = rng.randn(180, 2)
        f_py, z_py = _python_barnes_hut_negative_forces(y, theta=0.45, leaf_size=10)
        f_nb, z_nb = _barnes_hut_negative_forces(y, theta=0.45, leaf_size=10)

        np.testing.assert_allclose(f_nb, f_py, atol=1.0e-12)
        self.assertAlmostEqual(z_nb, z_py, places=10)

    def test_symmetric_sparse_positive_forces_match_full_edges(self):
        rng = np.random.RandomState(14)
        n = 30
        y = rng.randn(n, 2)
        a = scipy.sparse.random(n, n, density=0.2, random_state=rng, format="csr")
        p_coo = (a + a.T).tocoo()
        keep = p_coo.row != p_coo.col
        p = scipy.sparse.csr_matrix((p_coo.data[keep], (p_coo.row[keep], p_coo.col[keep])), shape=p_coo.shape)
        p.eliminate_zeros()
        p = p / p.sum()
        full = p.tocoo()
        upper = scipy.sparse.triu(p, k=1).tocoo()

        f_full = _sparse_positive_forces_from_edges(full.row, full.col, full.data, y, scale=3.0)
        f_upper = _sparse_positive_forces_symmetric_from_edges(upper.row, upper.col, upper.data, y, scale=3.0)
        np.testing.assert_allclose(f_upper, f_full, atol=1.0e-12)

    def test_barnes_hut_exact_one_step_matches_dense_update_on_complete_graph(self):
        rng = np.random.RandomState(15)
        n = 18
        y = rng.randn(n, 2)
        y -= y.mean(axis=0, keepdims=True)
        a = rng.rand(n, n)
        p = a + a.T
        p[np.arange(n), np.arange(n)] = 0.0
        p /= p.sum()
        p_sparse = scipy.sparse.csr_matrix(p)

        y_dense, _, _, _ = update_embed(
            p, y.copy(), np.zeros_like(y), np.ones_like(y), momentum=0.0, eta=7.0, alpha=1.0, min_gain=0.01
        )
        y_bh = _tsne_barnes_hut_from_p(
            p_sparse,
            Y=y.copy(),
            max_its=1,
            eta=7.0,
            momentum=0.0,
            early_exaggeration=1.0,
            early_its=0,
            min_gain=0.01,
            repulsion_method="exact",
            print_iter=1000,
            check_every=1000,
            out=io.StringIO(),
        )
        np.testing.assert_allclose(y_bh, y_dense, atol=1.0e-10)

    def test_barnes_hut_respects_max_iterations(self):
        mod = importlib.import_module("mixle.utils.hvis")
        rng = np.random.RandomState(16)
        n = 12
        a = rng.rand(n, n)
        p = a + a.T
        p[np.arange(n), np.arange(n)] = 0.0
        p = scipy.sparse.csr_matrix(p / p.sum())
        y = rng.randn(n, 2) * 1.0e-4
        calls = []
        old = mod._negative_forces

        def counted(*args, **kwargs):
            calls.append(1)
            return old(*args, **kwargs)

        try:
            mod._negative_forces = counted
            mod._tsne_barnes_hut_from_p(
                p,
                Y=y,
                max_its=3,
                eta=1.0,
                early_exaggeration=1.0,
                early_its=250,
                print_iter=1000,
                check_every=1000,
                repulsion_method="exact",
                out=io.StringIO(),
            )
        finally:
            mod._negative_forces = old

        self.assertEqual(len(calls), 3)

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

    def test_exact_tsne_gradient_matches_finite_differences(self):
        rng = np.random.RandomState(42)
        y = rng.randn(7, 2)
        y -= y.mean(axis=0, keepdims=True)
        a = rng.rand(7, 7)
        p = a + a.T
        p[np.arange(7), np.arange(7)] = 0.0
        p /= p.sum()
        alpha = 1.7

        grad, _ = _exact_tsne_gradient(p, y, alpha)
        eps = 1.0e-6
        for i, d in ((0, 0), (2, 1), (5, 0)):
            yp = y.copy()
            ym = y.copy()
            yp[i, d] += eps
            ym[i, d] -= eps
            fp = _kl(p, t_kernel(yp, alpha)[0])
            fm = _kl(p, t_kernel(ym, alpha)[0])
            numeric = (fp - fm) / (2.0 * eps)
            self.assertAlmostEqual(float(grad[i, d]), numeric, places=5)

    def test_exact_tsne_known_optimum_when_p_equals_q(self):
        rng = np.random.RandomState(43)
        y = rng.randn(9, 2)
        y -= y.mean(axis=0, keepdims=True)
        p, _, _ = t_kernel(y, alpha=1.0)

        grad, q = _exact_tsne_gradient(p, y, alpha=1.0)
        self.assertAlmostEqual(_kl(p, q), 0.0, places=12)
        np.testing.assert_allclose(grad, 0.0, atol=1.0e-12)

        y2, iy2, gains2, _ = update_embed(
            p, y.copy(), np.zeros_like(y), np.ones_like(y), momentum=0.0, eta=100.0, alpha=1.0, min_gain=0.01
        )
        np.testing.assert_allclose(y2, y, atol=1.0e-10)
        np.testing.assert_allclose(iy2, 0.0, atol=1.0e-10)
        self.assertTrue(np.all(gains2 >= 0.01))

    def test_vandermonde_fisher_features_are_exact_and_full_rank(self):
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        degree = 4
        model = _VandermondeMomentModel(degree)
        expected = np.vander(x, N=degree + 1, increasing=True)

        stats = model.to_fisher().expected_statistics_matrix(data=x)
        np.testing.assert_allclose(stats, expected, atol=1.0e-12)
        self.assertEqual(np.linalg.matrix_rank(stats), degree + 1)

        factors = fisher_factors(model, data=x, metric="identity")
        expected_vectors = expected - expected.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(factors[0]["x"], expected_vectors, atol=1.0e-12)

        log_s = model_log_affinity(None, None, affinity=factors)
        d2 = np.sum((expected_vectors[:, None, :] - expected_vectors[None, :, :]) ** 2, axis=2)
        expected_log_s = -0.5 * d2
        expected_log_s[np.arange(len(x)), np.arange(len(x))] = -np.inf
        finite = np.isfinite(expected_log_s)
        np.testing.assert_allclose(log_s[finite], expected_log_s[finite], atol=1.0e-12)
        self.assertGreater(log_s[2, 1], log_s[2, 0])
        self.assertGreater(log_s[2, 3], log_s[2, 4])

    # ---- end-to-end embeddings -------------------------------------------------

    def test_exact_embedding_converges_and_separates(self):
        t0 = time.time()
        y = htsne(
            self.data,
            mix_model=self.model,
            perplexity=20.0,
            method="exact",
            affinity="balanced",
            max_its=400,
            seed=3,
            out=io.StringIO(),
        )
        self.assertLess(time.time() - t0, 30.0)
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 3.0)

    def test_default_local_embedding_runs(self):
        data = self.data[:75]
        labels = self.labels[:75]
        y = htsne(data, mix_model=self.model, perplexity=10.0, method="exact", max_its=300, seed=3, out=io.StringIO())
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))
        # the old 1.5 bar encoded the categorical field contributing NOTHING within a cluster; the
        # universal typicality coordinates now surface real within-cluster substructure (a cluster's
        # minority-category points genuinely differ), which legitimately widens clusters. The exact
        # ratio is trajectory-volatile even at a fixed seed (1.249 on the CI numpy/BLAS build vs
        # 1.414 locally), so pin decisive separation with cross-build margin rather than a value
        # inside that noise...
        self.assertGreater(separation_ratio(y, labels), 1.15)
        # ...and that substructure is the new claim worth pinning: within a cluster, cross-category
        # pairs sit measurably farther apart than same-category pairs.
        cats = np.array([x[1] for x in data])
        same_d, cross_d = [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            for a_pos, i in enumerate(idx):
                for j in idx[a_pos + 1 :]:
                    d = float(np.linalg.norm(y[i] - y[j]))
                    (same_d if cats[i] == cats[j] else cross_d).append(d)
        self.assertGreater(np.mean(cross_d) / np.mean(same_d), 1.1)

    def test_exact_kl_decreases(self):
        p = get_pmat(self.z, self.l, targ_perplexity=20.0)
        buf = io.StringIO()
        tsne_exact(p, max_its=300, seed=3, print_iter=50, out=buf)
        kls = [float(line.rsplit("=", 1)[1]) for line in buf.getvalue().strip().split("\n")]
        self.assertGreater(len(kls), 2)
        self.assertLess(kls[-1], kls[0])
        self.assertLess(kls[-1], 1.0)

    def test_barnes_hut_embedding(self):
        t0 = time.time()
        y = htsne(
            self.data,
            mix_model=self.model,
            perplexity=20.0,
            method="barnes_hut",
            affinity="balanced",
            max_its=350,
            seed=3,
            out=io.StringIO(),
        )
        self.assertLess(time.time() - t0, 60.0)
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 3.0)

    def test_barnes_hut_approx_neighbor_embedding(self):
        y = htsne(
            self.data[:30],
            mix_model=self.model,
            perplexity=5.0,
            method="barnes_hut",
            affinity="balanced",
            neighbor_method="approx",
            neighbor_trees=2,
            neighbor_leaf_size=20,
            candidate_multiplier=2,
            max_its=300,
            seed=3,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (30, 2))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_optimize_alpha_path(self):
        y = htsne(
            self.data[:100],
            mix_model=self.model,
            perplexity=10.0,
            method="exact",
            optimize_alpha=True,
            max_its=300,
            seed=3,
            out=io.StringIO(),
        )
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
        log_s = model_log_affinity(self.z, self.l)
        for i in (0, 17, self.n - 1):
            self.assertTrue(np.allclose(dist[i, 1:], np.maximum(-log_s[i, idx[i, 1:]], 0.0), atol=1.0e-12))

    @unittest.skipUnless(HAS_UMAP, "umap-learn is not installed")
    def test_humap_embedding(self):
        y = humap(
            self.data, mix_model=self.model, n_neighbors=15, affinity="balanced", n_epochs=50, seed=4, out=io.StringIO()
        )
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
        topic_a = SequenceDistribution(IntegerCategoricalDistribution(0, [0.85, 0.05, 0.05, 0.05]), len_dist=len_dist)
        topic_b = SequenceDistribution(IntegerCategoricalDistribution(0, [0.05, 0.05, 0.05, 0.85]), len_dist=len_dist)

        data = topic_a.sampler(seed=seed).sample(size=n_per) + topic_b.sampler(seed=seed + 1).sample(size=n_per)
        labels = np.repeat([0, 1], n_per)
        model = MixtureDistribution([topic_a, topic_b], [0.5, 0.5])
        return data, labels, model

    def test_varlen_affinity_is_length_free(self):
        # with co-assignment affinities, the only length effect is posterior
        # certainty: two long observations of the same topic and two short ones
        # of the same topic should both have near-1 affinity
        data, labels, model = self._varlen_data_and_model(60)
        from mixle.utils.hvis import _posteriors_and_loglikes

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

        y = htsne(
            data,
            mix_model=model,
            perplexity=20.0,
            method="exact",
            affinity="balanced",
            max_its=350,
            seed=3,
            out=io.StringIO(),
        )
        self.assertGreater(separation_ratio(y, labels), 2.0)

        # the embedding should not be organized by observation length: within
        # each topic, short and long observations should overlap
        for c in (0, 1):
            yc, lc = y[labels == c], lengths[labels == c]
            short, long_ = yc[lc <= np.median(lc)], yc[lc > np.median(lc)]
            gap = np.linalg.norm(short.mean(0) - long_.mean(0))
            spread = 0.5 * (
                np.linalg.norm(short - short.mean(0), axis=1).mean()
                + np.linalg.norm(long_ - long_.mean(0), axis=1).mean()
            )
            self.assertLess(gap, 2.0 * spread)

    def test_sequence_field_log_density_sums_unless_len_normalized(self):
        from mixle.utils.hvis import _field_log_density_features

        data = [[0], [0, 0, 0]]
        len_dist = PoissonDistribution(2.0)
        raw_components = [
            SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.8, 0.2]), len_dist=len_dist, len_normalized=False
            ),
            SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.2, 0.8]), len_dist=len_dist, len_normalized=False
            ),
        ]
        norm_components = [
            SequenceDistribution(IntegerCategoricalDistribution(0, [0.8, 0.2]), len_dist=len_dist, len_normalized=True),
            SequenceDistribution(IntegerCategoricalDistribution(0, [0.2, 0.8]), len_dist=len_dist, len_normalized=True),
        ]

        raw_terms = list(_field_log_density_features(raw_components, data))
        norm_terms = list(_field_log_density_features(norm_components, data))
        raw_element_ll = raw_terms[0][0]
        norm_element_ll = norm_terms[0][0]

        np.testing.assert_allclose(raw_element_ll[1], 3.0 * raw_element_ll[0], atol=1.0e-12)
        np.testing.assert_allclose(norm_element_ll[1], norm_element_ll[0], atol=1.0e-12)

    @unittest.skipUnless(HAS_UMAP, "umap-learn is not installed")
    def test_varlen_humap(self):
        data, labels, model = self._varlen_data_and_model(120)
        y = humap(data, mix_model=model, n_neighbors=15, n_epochs=50, seed=4, out=io.StringIO())
        self.assertEqual(y.shape, (len(data), 2))
        self.assertGreater(separation_ratio(y, labels), 1.5)

    @classmethod
    def _crossed_varlen_data_and_model(cls, n_per=45):
        content = [
            IntegerCategoricalDistribution(0, [0.82, 0.12, 0.03, 0.03]),
            IntegerCategoricalDistribution(0, [0.03, 0.03, 0.12, 0.82]),
        ]
        lengths = [PoissonDistribution(6.0), PoissonDistribution(24.0)]

        comps, comp_meta = [], []
        for topic in range(2):
            for length in range(2):
                comps.append(SequenceDistribution(content[topic], len_dist=lengths[length]))
                comp_meta.append((topic, length))

        data, comp_labels, topic_labels, length_labels = [], [], [], []
        for k, comp in enumerate(comps):
            samples = comp.sampler(seed=100 + k).sample(size=n_per)
            data.extend(samples)
            comp_labels.extend([k] * len(samples))
            topic_labels.extend([comp_meta[k][0]] * len(samples))
            length_labels.extend([comp_meta[k][1]] * len(samples))

        model = MixtureDistribution(comps, [0.25] * 4)
        return data, np.asarray(comp_labels), np.asarray(topic_labels), np.asarray(length_labels), model

    def test_crossed_varlen_sequence_local_affinity_tracks_content_and_length(self):
        data, comp_labels, topic_labels, length_labels, model = self._crossed_varlen_data_and_model()
        factors = local_factors(model, data)
        self.assertEqual(len(factors), 2)  # sequence content + sequence length
        self.assertTrue(any(isinstance(f, dict) and f.get("kind") == "local" for f in factors))

        log_s = model_log_affinity(None, None, affinity=factors, evidence_cap=1.0)
        self.assertGreater(affinity_neighbor_purity(log_s, comp_labels, k=12), 0.90)
        self.assertGreater(affinity_neighbor_purity(log_s, topic_labels, k=12), 0.95)
        self.assertGreater(affinity_neighbor_purity(log_s, length_labels, k=12), 0.90)

    def test_crossed_varlen_sequence_local_embedding_tracks_both_axes(self):
        data, comp_labels, topic_labels, length_labels, model = self._crossed_varlen_data_and_model()
        y = htsne(
            data,
            mix_model=model,
            perplexity=18.0,
            method="exact",
            affinity="local",
            max_its=300,
            seed=8,
            out=io.StringIO(),
        )
        self.assertGreater(embedding_neighbor_purity(y, comp_labels, k=12), 0.90)
        self.assertGreater(embedding_neighbor_purity(y, topic_labels, k=12), 0.95)
        self.assertGreater(embedding_neighbor_purity(y, length_labels, k=12), 0.90)

    @classmethod
    def _complex_heterogeneous_data_and_model(cls, n_per=55):
        cat_probs = [
            {"red": 0.78, "blue": 0.12, "green": 0.10},
            {"red": 0.10, "blue": 0.78, "green": 0.12},
            {"red": 0.12, "blue": 0.10, "green": 0.78},
        ]
        seq_probs = [
            [0.80, 0.12, 0.04, 0.04],
            [0.04, 0.80, 0.12, 0.04],
            [0.04, 0.04, 0.12, 0.80],
        ]
        miss_probs = [0.15, 0.45, 0.70]
        count_rates = [3.0, 9.0, 16.0]
        seq_lens = [5.0, 14.0, 24.0]

        comps = []
        for k in range(3):
            comps.append(
                CompositeDistribution(
                    (
                        GaussianDistribution(-5.0 + 5.0 * k, 1.0 + 0.5 * k),
                        CategoricalDistribution(cat_probs[k]),
                        OptionalDistribution(GaussianDistribution(-3.0 + 3.0 * k, 1.2), p=miss_probs[k]),
                        PoissonDistribution(count_rates[k]),
                        SequenceDistribution(
                            IntegerCategoricalDistribution(0, seq_probs[k]), len_dist=PoissonDistribution(seq_lens[k])
                        ),
                    )
                )
            )

        data, labels = [], []
        for k, comp in enumerate(comps):
            samples = comp.sampler(seed=200 + k).sample(size=n_per)
            data.extend(samples)
            labels.extend([k] * len(samples))

        model = MixtureDistribution(comps, [1.0 / 3.0] * 3)
        return data, np.asarray(labels), model

    def test_complex_heterogeneous_local_affinity_decomposes_all_fields(self):
        data, labels, model = self._complex_heterogeneous_data_and_model()
        factors = local_factors(model, data)
        self.assertEqual(len(factors), 7)
        # every field except gated-inner scores now carries local geometry: native coordinates
        # where the leaf has them, universal typicality coordinates otherwise, and the Optional
        # missingness gate carries its presence indicator (the missing/present pattern is real
        # within-cluster structure). Only the Optional INNER field (scores gated to present rows)
        # remains posterior-only. Earlier pins (3, then 5) documented the fallback gaps as they
        # were successively closed.
        self.assertEqual(sum(isinstance(f, dict) and f.get("kind") == "local" for f in factors), 6)

        log_s = model_log_affinity(None, None, affinity=factors, evidence_cap=1.0)
        self.assertGreater(affinity_neighbor_purity(log_s, labels, k=12), 0.95)

    def test_complex_heterogeneous_local_barnes_hut_embedding(self):
        data, labels, model = self._complex_heterogeneous_data_and_model()
        y = htsne(
            data,
            mix_model=model,
            perplexity=18.0,
            method="barnes_hut",
            affinity="local",
            neighbor_method="exact",
            max_its=300,
            seed=9,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (len(data), 2))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(embedding_neighbor_purity(y, labels, k=12), 0.90)
        self.assertGreater(separation_ratio(y, labels), 2.5)

    # ---- balanced (mixed-type) affinities -----------------------------------

    def test_balanced_factors_structure(self):
        factors = balanced_factors(self.model, self.data)
        self.assertEqual(len(factors), 2)  # gaussian + categorical fields
        # log-affinity over factors = sum of per-field Bhattacharyya logs
        la = model_log_affinity(None, None, affinity=factors)
        self.assertEqual(la.shape, (self.n, self.n))
        self.assertTrue(np.all(np.isneginf(np.diag(la))))
        # symmetric: each factor is (sq, sq)
        off = ~np.eye(self.n, dtype=bool)
        self.assertTrue(np.allclose(la[off], la.T[off], atol=1.0e-10))

    def test_balanced_factors_preserve_crossed_heterogeneous_fields(self):
        comps = []
        for mu in (-3.0, 3.0):
            for cat_probs in ({"x": 0.999, "y": 0.001}, {"x": 0.001, "y": 0.999}):
                comps.append(
                    CompositeDistribution(
                        (
                            GaussianDistribution(mu, 0.08),
                            CategoricalDistribution(cat_probs),
                        )
                    )
                )
        model = MixtureDistribution(comps, [0.25] * 4)
        data = [(-3.0, "x"), (-3.1, "y"), (3.0, "x"), (3.1, "y")]

        factors = balanced_factors(model, data)
        self.assertEqual(len(factors), 2)
        g_cont, h_cont = factors[0]
        g_cat, h_cat = factors[1]
        s_cont = np.dot(g_cont, h_cont.T)
        s_cat = np.dot(g_cat, h_cat.T)

        self.assertGreater(s_cont[0, 1], 0.95)
        self.assertLess(s_cont[0, 2], 0.05)
        self.assertGreater(s_cat[0, 2], 0.95)
        self.assertLess(s_cat[0, 1], 0.10)

        log_cont = model_log_affinity(None, None, affinity=balanced_factors(model, data, field_weights=[1.0, 0.0]))
        log_cat = model_log_affinity(None, None, affinity=balanced_factors(model, data, field_weights=[0.0, 1.0]))
        self.assertGreater(log_cont[0, 1], log_cont[0, 2])
        self.assertGreater(log_cat[0, 2], log_cat[0, 1])

    def test_balanced_embeddings_run(self):
        y = htsne(
            self.data,
            mix_model=self.model,
            perplexity=20.0,
            affinity="balanced",
            method="exact",
            seed=3,
            max_its=300,
            out=io.StringIO(),
        )
        self.assertEqual(y.shape, (self.n, 2))
        self.assertGreater(separation_ratio(y, self.labels), 2.0)

    def test_auto_affinity_resolution(self):
        from mixle.utils.hvis import _resolve_affinity

        # composite components + raw data -> local factor list (one per leaf field)
        r = _resolve_affinity("auto", self.model, self.data, None)
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), 2)
        self.assertTrue(any(isinstance(f, dict) and f.get("kind") == "local" for f in r))
        # no raw data -> falls back to bhattacharyya
        self.assertEqual(_resolve_affinity("auto", self.model, None, None), "bhattacharyya")
        # non-composite components decompose too: a plain mixture is one leaf field
        plain = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        r = _resolve_affinity("auto", plain, [0.0, 1.0], None)
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), 1)

    def test_balanced_requires_data(self):
        with self.assertRaises(ValueError):
            htsne(None, mix_model=self.model, enc_data=("x",), affinity="balanced", out=io.StringIO())

    def test_balanced_decomposes_sequences_and_optionals(self):
        # sequence-of-records components flatten into element fields + length;
        # the old implementation refused anything without top-level .dists
        from mixle.utils.hvis import balanced_factors

        data, labels, model = self._varlen_data_and_model(60)
        factors = balanced_factors(model, data)
        self.assertGreater(len(factors), 1)
        n = len(data)
        for g, h in factors:
            self.assertEqual(g.shape[0], n)
            # per-field posteriors: squared factors sum to one per row
            self.assertTrue(np.allclose((g * g).sum(axis=1), 1.0, atol=1.0e-8))
        y = htsne(data, mix_model=model, seed=3, max_its=200, affinity="balanced", out=io.StringIO())
        self.assertEqual(y.shape, (n, 2))
        self.assertGreater(separation_ratio(y, labels), 1.5)

    def test_optional_missing_inner_field_does_not_create_nan_affinities(self):
        from mixle.utils.hvis import balanced_factors

        model = MixtureDistribution(
            [
                OptionalDistribution(CategoricalDistribution({"a": 1.0}), p=0.5),
                OptionalDistribution(CategoricalDistribution({"b": 1.0}), p=0.5),
            ],
            [0.5, 0.5],
        )
        data = [None, "a", None, "b"] * 8

        for maker in (balanced_factors, local_factors):
            factors = maker(model, data)
            log_s = model_log_affinity(None, None, affinity=factors)
            self.assertFalse(np.isnan(log_s).any())

    def test_evidence_cap_bounds_field_influence(self):
        from mixle.utils.hvis import balanced_factors, model_log_affinity

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
        from mixle.utils.hvis import _posteriors_and_loglikes

        z, _ = _posteriors_and_loglikes(self.model, data=self.data)
        la1 = model_log_affinity(z, affinity="bhattacharyya")
        la1c = model_log_affinity(z, affinity="bhattacharyya", evidence_cap=cap)
        self.assertTrue(np.allclose(np.nan_to_num(la1, neginf=-1e30), np.nan_to_num(la1c, neginf=-1e30)))

    def test_field_weights_apply_to_whole_field_coefficient(self):
        from mixle.utils.hvis import balanced_factors, model_log_affinity

        factors = balanced_factors(self.model, self.data, field_weights=[2.0, 0.0])
        la = model_log_affinity(None, None, affinity=factors, evidence_cap=None)

        g, h, weight = factors[0]
        expected = weight * np.log(np.dot(g, h.T))
        expected[np.arange(self.n), np.arange(self.n)] = -np.inf
        off = ~np.eye(self.n, dtype=bool)
        self.assertTrue(np.allclose(la[off], expected[off], atol=1.0e-12))

    def test_optional_without_missing_probability_has_no_gate_field(self):
        from mixle.utils.hvis import balanced_factors, model_log_affinity

        model = MixtureDistribution(
            [
                OptionalDistribution(GaussianDistribution(-2.0, 1.0), p=None),
                OptionalDistribution(GaussianDistribution(2.0, 1.0), p=None),
            ],
            [0.5, 0.5],
        )
        data = [None, -2.0, -1.5, None, 1.5, 2.0]
        factors = balanced_factors(model, data)
        self.assertEqual(len(factors), 1)
        la = model_log_affinity(None, None, affinity=factors)
        off = ~np.eye(len(data), dtype=bool)
        self.assertTrue(np.all(np.isfinite(la[off])))

    def test_field_weights_validated_against_flattened_fields(self):
        from mixle.utils.hvis import balanced_factors

        with self.assertRaises(ValueError):
            balanced_factors(self.model, self.data, field_weights=[1.0])
        with self.assertRaises(ValueError):
            balanced_factors(self.model, self.data, field_weights=[1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
