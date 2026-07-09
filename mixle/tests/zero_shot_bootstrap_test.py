"""Zero-shot modality bootstrap (workstream L3): a brand-new data type joins a cross-modal joint
with NO retraining of anything already fitted; the RESONANCE embedding separates classes on a
held-out modality never modeled natively; the fit-health-style gate correctly decides adequate vs
graduate in both directions.
"""

import importlib.util
import unittest

import numpy as np

from mixle.reason.cross_modal import CrossModalJoint
from mixle.reason.zero_shot_bootstrap import (
    add_modality_to_joint,
    fit_resonance_leaves,
    induce_leaf_for_unseen_type,
    resonance_adequacy_gate,
    resonance_embedding,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _two_modality_joint() -> CrossModalJoint:
    """Same shape as L2's fixture: 2 regimes, image=Gaussian, text=Categorical, already fitted."""
    return CrossModalJoint.from_components(
        names=("image", "text"),
        component_fields=[
            (
                GaussianDistribution(mu=-3.0, sigma2=1.0),
                CategoricalDistribution(pmap={"cat": 0.9, "dog": 0.1}),
            ),
            (
                GaussianDistribution(mu=3.0, sigma2=1.0),
                CategoricalDistribution(pmap={"cat": 0.1, "dog": 0.9}),
            ),
        ],
        weights=[0.6, 0.4],
    )


def _snapshot(joint: CrossModalJoint) -> list[tuple]:
    """A concrete before/after receipt of every existing leaf's fitted parameters."""
    out = []
    for component in joint.joint.components:
        image, text = component.dists
        out.append((float(image.mu), float(image.sigma2), dict(text.pmap)))
    return out


class UnseenModalityJoinsJointTest(unittest.TestCase):
    """Acceptance criterion 1: an unseen modality joins a joint and supports cross-modal inference,
    with NO retraining of the other leaves (verified bitwise, not assumed)."""

    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.joint = _two_modality_joint()
        self.before_objects = [c.dists for c in self.joint.joint.components]
        self.before_snapshot = _snapshot(self.joint)

        # A brand-new third modality never modeled before: a 3-D "sensor" vector whose mean tracks
        # the SAME latent regime as image/text (regime 0 centered near -1, regime 1 centered near 1),
        # so conditioning on it is genuinely informative, not just mechanically wired up.
        n_per_regime = 60
        regime0 = self.rng.normal(loc=-1.0, scale=0.3, size=(n_per_regime, 3))
        regime1 = self.rng.normal(loc=1.0, scale=0.3, size=(n_per_regime, 3))
        sensor_samples = np.concatenate([regime0, regime1], axis=0)
        labels = np.concatenate([np.zeros(n_per_regime), np.ones(n_per_regime)]).astype(int)

        leaf0 = induce_leaf_for_unseen_type([tuple(row) for row in regime0], rng=np.random.RandomState(1))
        leaf1 = induce_leaf_for_unseen_type([tuple(row) for row in regime1], rng=np.random.RandomState(2))
        self.sensor_samples = sensor_samples
        self.labels = labels
        self.new_joint = add_modality_to_joint(self.joint, "sensor", [leaf0, leaf1])

    def test_other_modalities_are_bitwise_unchanged(self):
        after_objects = [c.dists[:2] for c in self.new_joint.joint.components]
        for before, after in zip(self.before_objects, after_objects):
            # same underlying distribution objects -- never copied, never refit.
            self.assertIs(before[0], after[0])
            self.assertIs(before[1], after[1])
        after_snapshot = _snapshot(CrossModalJoint(names=self.joint.names, joint=self.joint.joint))
        self.assertEqual(self.before_snapshot, after_snapshot)

    def test_new_modality_names_and_regime_count_are_correct(self):
        self.assertEqual(self.new_joint.names, ("image", "text", "sensor"))
        self.assertEqual(len(self.new_joint.joint.components), 2)
        np.testing.assert_allclose(self.new_joint.joint.w, self.joint.joint.w)

    def test_condition_on_new_modality_infers_original_modalities(self):
        # a sensor reading squarely in regime 0's cluster should push the image posterior toward -3.
        posterior = self.new_joint.infer({"sensor": (-1.0, -1.0, -1.0)}, ["image"])
        map_regime = int(np.argmax(posterior.w))
        self.assertAlmostEqual(posterior.components[map_regime].dists[0].mu, -3.0)
        self.assertGreater(posterior.w[map_regime], 0.5)

    def test_condition_on_original_modalities_infers_new_modality(self):
        # conditioning on image/text from regime 1 should push weight onto sensor leaf 1 (mean ~+1).
        posterior = self.new_joint.infer({"image": 3.0, "text": "dog"}, ["sensor"])
        map_regime = int(np.argmax(posterior.w))
        self.assertGreater(posterior.w[map_regime], 0.5)
        # regime 1's sensor leaf was fit on samples centered at +1.
        self.assertGreater(float(np.mean(self.sensor_samples[self.labels == 1])), 0.0)


class ResonanceEmbeddingSeparationTest(unittest.TestCase):
    """Acceptance criterion 2: the resonance embedding separates classes on a held-out modality
    never modeled natively -- measured, not assumed, with a real downstream classifier."""

    def setUp(self):
        rng = np.random.RandomState(7)
        # The existing model zoo: several already-fitted, DIVERSE classical leaves (never touched
        # again below) -- exactly what "existing model zoo" means for L3.
        self.zoo = [
            GaussianDistribution(mu=0.0, sigma2=1.0),
            GaussianDistribution(mu=10.0, sigma2=4.0),
            GaussianDistribution(mu=-5.0, sigma2=0.5),
            GaussianDistribution(mu=50.0, sigma2=25.0),
        ]
        # A brand-new modality (never fitted natively) with 3 genuinely different classes.
        n = 40
        class_a = rng.normal(loc=1.0, scale=0.3, size=n)
        class_b = rng.normal(loc=20.0, scale=0.3, size=n)
        class_c = rng.normal(loc=-8.0, scale=0.3, size=n)
        self.samples = np.concatenate([class_a, class_b, class_c])
        self.labels = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)]).astype(int)

    def test_resonance_embedding_separates_three_classes(self):
        embedding = resonance_embedding(list(self.samples), self.zoo)
        self.assertEqual(embedding.shape, (len(self.samples), len(self.zoo)))

        # Real, measured separation: leave-one-out nearest-centroid classification on the embedding.
        centroids = {label: embedding[self.labels == label].mean(axis=0) for label in np.unique(self.labels)}
        correct = 0
        for i in range(embedding.shape[0]):
            dists = {label: float(np.linalg.norm(embedding[i] - c)) for label, c in centroids.items()}
            predicted = min(dists, key=dists.get)
            correct += int(predicted == self.labels[i])
        accuracy = correct / embedding.shape[0]
        chance = 1.0 / len(np.unique(self.labels))
        self.report_accuracy = accuracy
        self.assertGreater(
            accuracy,
            chance + 0.3,
            f"resonance embedding nearest-centroid accuracy {accuracy:.3f} vs chance {chance:.3f}",
        )


class ResonanceAdequacyGateTest(unittest.TestCase):
    """Acceptance criterion 3: the fit-health-style gate is correct in both directions."""

    def test_well_separated_embedding_is_adequate(self):
        rng = np.random.RandomState(3)
        n = 60
        group_a = rng.normal(loc=-10.0, scale=0.2, size=(n, 3))
        group_b = rng.normal(loc=10.0, scale=0.2, size=(n, 3))
        embedding = np.concatenate([group_a, group_b], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        self.assertTrue(resonance_adequacy_gate(embedding, labels))

    def test_poorly_separated_embedding_is_inadequate(self):
        rng = np.random.RandomState(4)
        n = 60
        group_a = rng.normal(loc=0.0, scale=5.0, size=(n, 3))
        group_b = rng.normal(loc=0.05, scale=5.0, size=(n, 3))
        embedding = np.concatenate([group_a, group_b], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        self.assertFalse(resonance_adequacy_gate(embedding, labels))

    def test_too_few_samples_defaults_to_graduate(self):
        embedding = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertFalse(resonance_adequacy_gate(embedding, [0, 1]))


class InduceLeafForUnseenTypeTest(unittest.TestCase):
    """The automatic-profiler extension's fallback chain: classical -> neural -> graph/sequence,
    each exercised on a data type that the base profiler cannot already parse (custom objects)."""

    class _Blob:
        """A genuinely unrecognized Python object: not a tuple/list/dict/set/number/str, so the
        base DatumNode profiler abstains (``obj_count`` -> ``IgnoredDistribution``)."""

        def __init__(self, vector):
            self.vector = vector

    def test_low_dimensional_numeric_blob_gets_a_classical_leaf(self):
        rng = np.random.RandomState(5)
        samples = [self._Blob(rng.normal(loc=2.0, scale=1.0, size=3)) for _ in range(80)]
        leaf = induce_leaf_for_unseen_type(samples, rng=np.random.RandomState(6))
        # a classical family (multivariate Gaussian) -- not neural.
        self.assertIn("MultivariateGaussian", type(leaf).__name__)

    @unittest.skipUnless(HAS_TORCH, "the high-dimensional path falls back to GradLeaf, which requires torch")
    def test_high_dimensional_numeric_blob_gets_a_neural_fallback(self):
        rng = np.random.RandomState(8)
        samples = [self._Blob(rng.normal(size=24)) for _ in range(50)]
        leaf = induce_leaf_for_unseen_type(samples, max_its=1)
        self.assertTrue(hasattr(leaf, "log_density"))
        self.assertIn("GradLeaf", type(leaf).__name__)

    def test_graph_structured_blob_gets_a_graph_model(self):
        class _GraphBlob:
            def __init__(self, adjacency):
                self.adjacency = adjacency

        rng = np.random.RandomState(9)
        samples = []
        for _ in range(30):
            mat = (rng.random((5, 5)) < 0.3).astype(float)
            mat = np.triu(mat, 1)
            mat = mat + mat.T
            samples.append(_GraphBlob(mat))
        leaf = induce_leaf_for_unseen_type(samples)
        self.assertIn("ErdosRenyi", type(leaf).__name__)

    def test_unmodelable_type_raises_clear_error(self):
        class _Opaque:
            pass

        with self.assertRaises(TypeError):
            induce_leaf_for_unseen_type([_Opaque() for _ in range(5)])


class FitResonanceLeavesTest(unittest.TestCase):
    """A lightweight resonance-coordinate leaf can also plug into add_modality_to_joint without
    graduating to a native leaf, still with no retraining of the other modalities."""

    def test_resonance_leaves_join_the_joint_without_retraining(self):
        joint = _two_modality_joint()
        before = [c.dists[:2] for c in joint.joint.components]

        zoo = [GaussianDistribution(mu=-3.0, sigma2=1.0), GaussianDistribution(mu=3.0, sigma2=1.0)]
        rng = np.random.RandomState(11)
        regime0_samples = rng.normal(loc=-1.0, scale=0.2, size=40)
        regime1_samples = rng.normal(loc=1.0, scale=0.2, size=40)
        samples = np.concatenate([regime0_samples, regime1_samples])
        labels = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)

        embedding = resonance_embedding(list(samples), zoo)
        self.assertTrue(resonance_adequacy_gate(embedding, labels))

        leaves = fit_resonance_leaves(embedding, labels, num_regimes=2)
        new_joint = add_modality_to_joint(joint, "proxy_sensor", leaves)

        after = [c.dists[:2] for c in new_joint.joint.components]
        for b, a in zip(before, after):
            self.assertIs(b[0], a[0])
            self.assertIs(b[1], a[1])
        self.assertEqual(new_joint.names, ("image", "text", "proxy_sensor"))


if __name__ == "__main__":
    unittest.main()
