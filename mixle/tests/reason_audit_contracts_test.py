"""Focused regressions for cross-modal and zero-shot reasoning audit contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mixle.reason.cross_modal import CrossModalJoint
from mixle.reason.zero_shot_bootstrap import (
    _generic_scalar_reduction,
    _try_automatic_profiler,
    add_modality_to_joint,
    fit_resonance_leaves,
    induce_leaf_for_unseen_type,
    resonance_adequacy_gate,
    resonance_embedding,
)
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def _joint() -> CrossModalJoint:
    return CrossModalJoint.from_components(
        ("left", "right"),
        [
            (GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)),
            (GaussianDistribution(1.0, 1.0), GaussianDistribution(-1.0, 1.0)),
        ],
        [0.5, 0.5],
    )


class CrossModalSchemaTest(unittest.TestCase):
    def test_schema_and_component_arity_are_validated_on_direct_construction(self):
        component = CompositeDistribution([GaussianDistribution(0.0, 1.0)])
        mixture = MixtureDistribution([component], [1.0])
        for names in ((), ("",), ("x", "x"), ("x", "y")):
            with self.subTest(names=repr(names)), self.assertRaises((TypeError, ValueError)):
                CrossModalJoint(names, mixture)

    def test_target_must_be_unique_and_nonempty(self):
        joint = _joint()
        with self.assertRaisesRegex(ValueError, "at least one"):
            joint.infer({"left": 0.0, "right": 0.0})
        with self.assertRaisesRegex(ValueError, "unique"):
            joint.infer({"left": 0.0}, ["right", "right"])

    def test_new_modality_cannot_collide_with_schema(self):
        joint = _joint()
        leaves = [GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0)]
        with self.assertRaisesRegex(ValueError, "already exists"):
            add_modality_to_joint(joint, "left", leaves)


class ZeroShotFallbackTest(unittest.TestCase):
    def test_profiler_implementation_failure_propagates(self):
        with patch("mixle.inference.estimation.optimize", side_effect=RuntimeError("optimizer defect")):
            with self.assertRaisesRegex(RuntimeError, "optimizer defect"):
                _try_automatic_profiler([1.0, 2.0], rng=None, max_its=1)

    def test_adjacency_is_detected_before_numeric_flattening(self):
        rows = [np.eye(3), np.eye(3)]
        sentinel = object()
        with (
            patch("mixle.reason.zero_shot_bootstrap._try_automatic_profiler", return_value=None),
            patch("mixle.reason.zero_shot_bootstrap._fit_graph_fallback", return_value=sentinel) as graph_fit,
            patch("mixle.reason.zero_shot_bootstrap._fit_numeric_fallback") as numeric_fit,
        ):
            self.assertIs(induce_leaf_for_unseen_type(rows), sentinel)
        graph_fit.assert_called_once()
        numeric_fit.assert_not_called()

    def test_sequence_rows_are_materialized_once(self):
        def row(values):
            yield from values

        sentinel = object()
        rows = [row([1, 2]), row([3, 4, 5])]
        with (
            patch("mixle.reason.zero_shot_bootstrap._try_automatic_profiler", return_value=None),
            patch("mixle.reason.zero_shot_bootstrap._fit_sequence_fallback", return_value=sentinel) as fit,
        ):
            self.assertIs(induce_leaf_for_unseen_type(rows), sentinel)
        frozen_rows, elements = fit.call_args.args[:2]
        self.assertEqual(frozen_rows, [(1, 2), (3, 4, 5)])
        self.assertEqual(elements, [1, 2, 3, 4, 5])


class ResonanceContractTest(unittest.TestCase):
    def test_opaque_default_repr_is_not_used_as_identity(self):
        class Opaque:
            pass

        with self.assertRaisesRegex(TypeError, "canonical"):
            _generic_scalar_reduction(Opaque())

    def test_structured_samples_require_an_explicit_reduction(self):
        model = GaussianDistribution(0.0, 1.0)
        with self.assertRaisesRegex(TypeError, "reduction"):
            resonance_embedding([[1.0, 2.0]], [model])
        result = resonance_embedding([[1.0, 2.0]], [model], reduction=lambda row: row[0])
        self.assertEqual(result.shape, (1, 1))

    def test_broken_model_capability_is_not_converted_to_neutral_evidence(self):
        class BrokenGaussian(GaussianDistribution):
            def cdf(self, x):
                raise RuntimeError("broken cdf")

        with self.assertRaisesRegex(RuntimeError, "broken cdf"):
            resonance_embedding([1.0], [BrokenGaussian(0.0, 1.0)])

    def test_adequacy_requires_independent_finite_evidence(self):
        embedding = np.asarray([[-3.0], [-2.0], [2.0], [3.0]])
        with self.assertRaisesRegex(ValueError, "labels_or_structure"):
            resonance_adequacy_gate(embedding)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            resonance_adequacy_gate(np.asarray([1.0, 2.0, 3.0, 4.0]), [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "threshold"):
            resonance_adequacy_gate(embedding, [0, 0, 1, 1], threshold=np.nan)

    def test_regime_leaf_fit_requires_complete_valid_assignments(self):
        embedding = np.asarray([[0.0], [1.0], [2.0]])
        invalid = [
            ([0, 0, 0], 2),
            ([0, 1], 2),
            ([0, 1, 2.5], 3),
            ([0, -1, 1], 2),
        ]
        for labels, regimes in invalid:
            with self.subTest(labels=repr(labels), regimes=repr(regimes)), self.assertRaises(ValueError):
                fit_resonance_leaves(embedding, labels, regimes)


if __name__ == "__main__":
    unittest.main()
