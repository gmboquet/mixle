"""Malformed fused geometry is rejected before bounds-unchecked kernel entry."""

import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    LaplaceDistribution,
    MarkovChainDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from mixle.stats.compute import fused_codegen as fc


class _BadParameterDistribution:
    pass


class FusedGeometryValidationTest(unittest.TestCase):
    def test_composite_factors_must_have_equal_row_counts(self):
        model = CompositeDistribution((GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)))
        encoded = (np.asarray([0.0, 1.0]), np.asarray([0.0]))
        with self.assertRaisesRegex(ValueError, "has 1 rows.*has 2"):
            fc.fused_seq_log_density(model, encoded)

    def test_every_array_in_one_factor_must_have_equal_rows(self):
        model = PoissonDistribution(2.0)
        encoded = (np.asarray([0.0, 1.0]), np.asarray([0.0]))
        with self.assertRaisesRegex(ValueError, "data\\[1\\].*1 rows.*has 2"):
            fc.fused_seq_log_density(model, encoded)

    def test_bridge_score_columns_must_align_with_other_factors(self):
        model = CompositeDistribution((GaussianDistribution(0.0, 1.0), LaplaceDistribution(0.0, 1.0)))
        encoded = (np.asarray([0.0, 1.0]), np.asarray([0.0]))
        with self.assertRaisesRegex(ValueError, "bridge factor 1 component 0 has 1 rows.*has 2"):
            fc.fused_seq_log_density(model, encoded)

    def test_categorical_indices_are_checked_against_table_width(self):
        model = CategoricalDistribution({"a": 0.5, "b": 0.5})
        encoded = (np.asarray([0, 2]), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "category indices.*\\[0, 2\\)"):
            fc.fused_seq_log_density(model, encoded)

    def test_a_lengthless_chain_is_refused_rather_than_scattered(self):
        """A chain with no length model reaches no fused path at all, so no geometry is checked.

        dd294198 took the chain leaf template out of service: a Null length model makes the chain a
        conditional likelihood factor rather than a law over finite sequences, and the template also
        emitted the superseded mutable-map statistic layout. ``_markov_chain_matches`` still requires
        a Null length model *and* LAW semantics -- conditions the docstring itself calls mutually
        exclusive -- which is how it stays dormant until a template carrying an explicit length law
        exists. A proper chain law (one with a len_dist) goes through the native bridge instead; that
        parity is fused_chain_test's subject. The scatter range check this test used to exercise is
        retained for the future template, and ``_validate_fused_indices`` itself stays covered by the
        categorical case above.
        """
        model = MarkovChainDistribution(
            {"a": 0.5, "b": 0.5},
            {"a": {"a": 0.5, "b": 0.5}, "b": {"a": 0.5, "b": 0.5}},
        )
        encoded = list(model.dist_to_encoder().seq_encode([["a"], ["b"]]))
        encoded[1] = np.asarray([0, 2])
        with self.assertRaisesRegex(ValueError, "MarkovChainDistribution is not fusible on any path"):
            fc.fused_seq_log_density(model, tuple(encoded))

    def test_template_parameter_component_width_is_checked(self):
        template = fc.LeafTemplate(
            name="bad-parameter-width",
            matches=lambda dist: isinstance(dist, _BadParameterDistribution),
            data=lambda encoded: (np.asarray(encoded, dtype=np.float64),),
            params=lambda components: {"bad": np.zeros(0, dtype=np.float64)},
            expr=lambda values, params: values[0],
            acc_names=("sx",),
            acc_stmt=lambda values, accumulators, weight: f"{accumulators['sx']}[k] += {weight} * {values[0]}",
            to_value=lambda stats, count: (count, stats[0]),
        )
        with patch.object(fc, "_TEMPLATES", list(fc._TEMPLATES)):
            fc.register_leaf_template(template)
            with self.assertRaisesRegex(ValueError, "component width 0; expected 1"):
                fc.fused_seq_log_density(_BadParameterDistribution(), np.asarray([1.0]))

    def test_log_weight_component_width_is_checked(self):
        model = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )
        model.log_w = np.zeros(1)
        encoded = model.dist_to_encoder().seq_encode([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "2 components but log_w has shape \\(1,\\)"):
            fc.fused_seq_log_density(model, encoded)


if __name__ == "__main__":
    unittest.main()
