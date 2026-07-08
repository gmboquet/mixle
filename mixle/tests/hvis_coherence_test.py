"""The coherent default view (mixle.utils.hvis): the three classic "janky visualization" complaints,
reproduced as fixtures and pinned as fixed.

1. SEQUENCES THROUGH AN HMM -- a mixture of HMMs has no native leaf coordinates, so the old 'local'
   fallback degraded to posterior overlap; sharp posteriors made every same-cluster pair an exact
   tie and clusters rendered as tiny structureless points. Typicality coordinates (per-token
   log-density rate per component + a log-length axis) give every HMM observation real
   within-cluster geometry.
2. VARIABLE-LENGTH FIELDS -- total sequence evidence grows with length, so unnormalized views let
   length masquerade as all the structure. Per-token rates keep content primary; length stays one
   honest axis.
3. MIXED CONTINUOUS + DISCRETE -- a sharp continuous field dominates the joint posterior and the
   discrete field's relationships become invisible under joint-posterior affinities. Per-field
   factors with component-local whitening make the fields commensurate.

Plus affinity_health: the degeneracies above are measurable properties of the AFFINITY, reported in
milliseconds with plain-language diagnosis -- not something to discover after a thousand t-SNE
iterations.
"""

import io
import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    SequenceDistribution,
)
from mixle.utils.hvis import affinity_health, htsne, local_factors, model_log_affinity

# medium lengths: posteriors sharp enough to cluster, soft enough that geometry is graded
_LEN_MED = IntegerCategoricalDistribution(18, [1.0 / 9.0] * 9)  # lengths 18..26
# long lengths: posterior sharpness underflows into EXACT ties -- the hard-collapse regime
_LEN_LONG = IntegerCategoricalDistribution(40, [1.0 / 21.0] * 21)  # lengths 40..60


def _hmm(trans, emit_a, len_dist):
    return HiddenMarkovModelDistribution(
        [CategoricalDistribution({"a": emit_a, "b": 1.0 - emit_a}), CategoricalDistribution({"a": 0.5, "b": 0.5})],
        [0.5, 0.5],
        trans,
        len_dist=len_dist,
    )


def _hmm_mixture_fixture(n_per=35, seed=0, len_dist=_LEN_MED):
    """Two HMM regimes with different dynamics over the SAME alphabet, variable lengths."""
    comps = [_hmm([[0.9, 0.1], [0.1, 0.9]], 0.9, len_dist), _hmm([[0.2, 0.8], [0.8, 0.2]], 0.1, len_dist)]
    model = MixtureDistribution(comps, [0.5, 0.5])
    data, labels = [], []
    for k, comp in enumerate(comps):
        data.extend(comp.sampler(seed=seed + k).sample(size=n_per))
        labels.extend([k] * n_per)
    return data, np.asarray(labels), model


def _embedding_shape_stats(y, labels):
    cents = np.stack([y[labels == c].mean(axis=0) for c in np.unique(labels)])
    within = np.mean(
        [np.linalg.norm(y[labels == c] - cents[i], axis=1).mean() for i, c in enumerate(np.unique(labels))]
    )
    between = np.mean(
        [np.linalg.norm(cents[i] - cents[j]) for i in range(len(cents)) for j in range(i + 1, len(cents))]
    )
    d2 = np.square(y[:, None, :] - y[None, :, :]).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    purity = float(np.mean(labels[d2.argmin(axis=1)] == labels))
    return purity, float(within), float(between)


class HmmSequenceCoherenceTest(unittest.TestCase):
    """Complaint 1: 'sequences through an HMM ... collapse into tiny points'."""

    def test_posterior_only_view_is_measurably_degenerate(self):
        # long sequences: posterior sharpness underflows into exact within-cluster ties -- the
        # affinity literally cannot rank a point's same-regime neighbors.
        data, _, model = _hmm_mixture_fixture(len_dist=_LEN_LONG)
        report = affinity_health(model, data, affinity="bhattacharyya")
        self.assertGreater(report["top_tie_fraction"], 0.05)
        self.assertTrue(report["diagnosis"])
        # ...while the local view on the SAME data stays rankable
        report_auto = affinity_health(model, data, affinity="auto")
        self.assertLess(report_auto["top_tie_fraction"], 0.05)

    def test_auto_view_is_healthy_and_structured(self):
        data, labels, model = _hmm_mixture_fixture()
        report = affinity_health(model, data, affinity="auto")
        self.assertLess(report["top_tie_fraction"], 0.05)
        self.assertTrue(all(f["geometry"] == "local" for f in report["fields"]))

        y = htsne(data, mix_model=model, perplexity=12.0, method="exact", max_its=300, seed=0, out=io.StringIO())
        purity, within, between = _embedding_shape_stats(y, labels)
        self.assertGreater(purity, 0.85)  # regimes separate (short sequences stay honestly ambiguous)...
        self.assertGreater(within, 0.10 * between)  # ...and clusters are clouds, not tiny points

    def test_hmm_leaf_gets_local_geometry_not_posterior_fallback(self):
        data, _, model = _hmm_mixture_fixture(n_per=15)
        factors = local_factors(model, data)
        self.assertTrue(all(isinstance(f, dict) and f.get("kind") == "local" for f in factors))
        # sequence-valued leaf: per-component rate columns + one log-length axis, scored per dof
        self.assertEqual(factors[0]["x"].shape[1], 3)  # K=2 rates + log-length
        self.assertEqual(factors[0]["delta_scale"], 3.0)


class VariableLengthCoherenceTest(unittest.TestCase):
    """Complaint 2: 'stuff with variable length fields' -- length must not masquerade as content."""

    def _fixture(self, n_per=35, seed=3):
        comps = [
            SequenceDistribution(CategoricalDistribution({"a": 0.85, "b": 0.15}), len_dist=_LEN_MED),
            SequenceDistribution(CategoricalDistribution({"a": 0.15, "b": 0.85}), len_dist=_LEN_MED),
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data, labels = [], []
        for k, comp in enumerate(comps):
            data.extend(comp.sampler(seed=seed + k).sample(size=n_per))
            labels.extend([k] * n_per)
        return data, np.asarray(labels), model

    def test_content_organizes_the_embedding_not_length(self):
        data, content, model = self._fixture()

        lengths = np.asarray([len(x) for x in data])
        length_split = (lengths > np.median(lengths)).astype(int)

        y = htsne(data, mix_model=model, perplexity=12.0, method="exact", max_its=300, seed=1, out=io.StringIO())
        content_purity, _, _ = _embedding_shape_stats(y, content)
        length_purity, _, _ = _embedding_shape_stats(y, length_split)
        self.assertGreater(content_purity, 0.85)
        # both regimes share the SAME length distribution: content must organize the layout, with
        # length visible at most as secondary structure, never the dominant axis
        self.assertGreater(content_purity, length_purity)


class MixedFieldCoherenceTest(unittest.TestCase):
    """Complaint 3: 'mixed continuous and discrete data' -- a sharp continuous field must not make
    the discrete field's relationships invisible."""

    def _fixture(self, n_per=35, seed=6):
        # cluster-independent discrete field. NON-uniform on purpose: a model-based view can only
        # show structure the model encodes -- under a uniform categorical every category has the
        # same probability and there is genuinely nothing to visualize.
        cat = {"x": 0.6, "y": 0.3, "z": 0.1}
        comps = [
            CompositeDistribution((GaussianDistribution(-4.0, 0.01), CategoricalDistribution(cat))),
            CompositeDistribution((GaussianDistribution(4.0, 0.01), CategoricalDistribution(cat))),
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data, labels = [], []
        for k, comp in enumerate(comps):
            data.extend(comp.sampler(seed=seed + k).sample(size=n_per))
            labels.extend([k] * n_per)
        return data, np.asarray(labels), model

    def _within_cluster_category_contrast(self, log_s, data, labels):
        """Mean within-cluster affinity gap: same-category pairs minus cross-category pairs."""
        cats = np.array([x[1] for x in data])
        same, cross = [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            for a_pos, i in enumerate(idx):
                for j in idx[a_pos + 1 :]:
                    (same if cats[i] == cats[j] else cross).append(log_s[i, j])
        return float(np.mean(same) - np.mean(cross))

    def test_discrete_relationships_visible_under_local_invisible_under_joint_posterior(self):
        data, labels, model = self._fixture()
        from mixle.utils.hvis import _posteriors_and_loglikes

        z, ll = _posteriors_and_loglikes(model, data=data)
        log_s_joint = model_log_affinity(z, ll, affinity="bhattacharyya")
        factors = local_factors(model, data)
        log_s_local = model_log_affinity(None, None, affinity=factors, evidence_cap=1.0)

        contrast_joint = self._within_cluster_category_contrast(log_s_joint, data, labels)
        contrast_local = self._within_cluster_category_contrast(log_s_local, data, labels)
        # under the joint posterior the razor-sharp Gaussian owns the posterior and the categorical
        # contributes ~nothing within a cluster; the per-field local view makes it visible.
        self.assertLess(abs(contrast_joint), 1.0e-6)
        self.assertGreater(contrast_local, 0.05)


class AffinityHealthTest(unittest.TestCase):
    def test_healthy_local_view_has_empty_diagnosis(self):
        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(0, 1, 30), rng.normal(8, 1, 30)])]
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(8.0, 1.0)], [0.5, 0.5])
        report = affinity_health(model, data, affinity="auto")
        self.assertLess(report["top_tie_fraction"], 0.05)
        self.assertEqual(report["diagnosis"], [])

    def test_subsampling_caps_the_diagnostic_cost(self):
        rng = np.random.RandomState(1)
        data = [float(v) for v in rng.normal(0, 1, 900)]
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(8.0, 1.0)], [0.5, 0.5])
        report = affinity_health(model, data, affinity="auto", max_rows=100)
        self.assertEqual(report["n"], 900)
        self.assertEqual(report["n_sampled"], 100)


if __name__ == "__main__":
    unittest.main()
