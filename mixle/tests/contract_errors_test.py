"""Catalog test pinning error-message quality at the composer contract boundaries (A6).

Every entry below feeds one combinator family (composite/mixture/sequence/conditional/optional) a
canonical malformed input at its ``seq_encode`` or ``estimate`` entry point, and asserts:

  - the call raises a ``ContractError`` (this module's own validation, not an incidental error
    surfacing from deep inside numpy/whatever),
  - the message contains the expected field path substring,
  - the message names both the expected and actual type/shape.

This mirrors the catalog-of-every-distribution shape used by ``sampler_seed_test.py``: a flat list
of cases drives a single parametrized loop, so message quality is pinned per-case rather than
merely aspirational.
"""

import unittest
from collections.abc import Callable
from typing import Any

import mixle.stats as stats
from mixle.stats.compute.pdist import ContractError

# --- Fixture distributions shared by the catalog below ------------------------------------------

_CAT_AB = stats.CategoricalDistribution({"a": 0.5, "b": 0.5})
_CAT_XY = stats.CategoricalDistribution({"x": 0.5, "y": 0.5})
_GAUSS = stats.GaussianDistribution(0.0, 1.0)


def _composite_encoder():
    return stats.CompositeDistribution([_CAT_AB, _CAT_XY]).dist_to_encoder()


def _composite_of_sequence_encoder():
    seq = stats.SequenceDistribution(_GAUSS)
    return stats.CompositeDistribution([_CAT_AB, seq]).dist_to_encoder()


def _sequence_encoder():
    return stats.SequenceDistribution(_GAUSS).dist_to_encoder()


def _mixture_encoder():
    return stats.MixtureDistribution([_GAUSS, stats.GaussianDistribution(1.0, 2.0)], [0.5, 0.5]).dist_to_encoder()


def _conditional_encoder():
    return stats.ConditionalDistribution({"a": _GAUSS}, given_dist=_CAT_AB).dist_to_encoder()


def _optional_encoder():
    return stats.OptionalDistribution(_GAUSS).dist_to_encoder()


def _composite_estimator():
    return stats.CompositeEstimator([_CAT_AB.estimator(), _CAT_XY.estimator()])


def _sequence_estimator():
    return stats.SequenceDistribution(_GAUSS).estimator()


def _mixture_estimator():
    return stats.MixtureDistribution([_GAUSS, stats.GaussianDistribution(1.0, 2.0)], [0.5, 0.5]).estimator()


def _conditional_estimator():
    return stats.ConditionalDistribution({"a": _GAUSS}, given_dist=_CAT_AB).estimator()


def _optional_estimator():
    return stats.OptionalDistribution(_GAUSS).estimator()


# --- The catalog ----------------------------------------------------------------------------
# Each entry: (family, malformation, callable-under-test, expected field-path substring).
# The callable is a zero-arg thunk so construction (which itself must NOT raise) is separated
# from the malformed call under test.

CATALOG: list[tuple[str, str, Callable[[], Any], str]] = [
    # --- composite -------------------------------------------------------------------------
    (
        "composite",
        "wrong_tuple_arity_short",
        lambda: _composite_encoder().seq_encode([("a",)]),
        "CompositeDistribution.dists (row 0)",
    ),
    (
        "composite",
        "wrong_tuple_arity_long",
        lambda: _composite_encoder().seq_encode([("a", "x", "extra")]),
        "CompositeDistribution.dists (row 0)",
    ),
    (
        "composite",
        "wrong_field_type",
        lambda: _composite_encoder().seq_encode([5]),
        "CompositeDistribution.dists (row 0)",
    ),
    (
        "composite",
        "ragged_rows",
        lambda: _composite_encoder().seq_encode([("a", "x"), ("b",)]),
        "CompositeDistribution.dists (row 1)",
    ),
    (
        "composite",
        "malformed_nested_sequence_element",
        lambda: _composite_of_sequence_encoder().seq_encode([("a", [1.0, "bad", 3.0])]),
        "CompositeDistribution.dists[1] -> SequenceDistribution.entries",
    ),
    (
        "composite",
        "mismatched_estimator_data_shape",
        lambda: _composite_estimator().estimate(1.0, (1,)),
        "CompositeEstimator.estimate(suff_stat)",
    ),
    # --- sequence ----------------------------------------------------------------------------
    (
        "sequence",
        "wrong_field_type_row_not_iterable",
        lambda: _sequence_encoder().seq_encode([5]),
        "SequenceDistribution.entries (row 0)",
    ),
    (
        "sequence",
        "malformed_element_type",
        lambda: _sequence_encoder().seq_encode([[1.0, "bad", 3.0]]),
        "SequenceDistribution.entries",
    ),
    (
        "sequence",
        "mismatched_estimator_data_shape",
        lambda: _sequence_estimator().estimate(1.0, "not-a-tuple"),
        "SequenceEstimator.estimate(suff_stat)",
    ),
    # --- mixture -------------------------------------------------------------------------------
    (
        "mixture",
        "wrong_container_type",
        lambda: _mixture_encoder().seq_encode(5),
        "MixtureDistribution.seq_encode",
    ),
    (
        "mixture",
        "malformed_field_type",
        lambda: _mixture_encoder().seq_encode([1.0, "bad", 3.0]),
        "MixtureDistribution.components",
    ),
    (
        "mixture",
        "mismatched_estimator_data_shape",
        lambda: _mixture_estimator().estimate(1.0, "not-a-tuple"),
        "MixtureEstimator.estimate(suff_stat)",
    ),
    (
        "mixture",
        "mismatched_component_count",
        lambda: _mixture_estimator().estimate(1.0, ([1.0], (1,))),
        "MixtureEstimator.estimate(suff_stat)",
    ),
    # --- conditional ---------------------------------------------------------------------------
    (
        "conditional",
        "wrong_tuple_arity",
        lambda: _conditional_encoder().seq_encode([("a",)]),
        "ConditionalDistribution.seq_encode (row 0)",
    ),
    (
        "conditional",
        "malformed_field_type",
        lambda: _conditional_encoder().seq_encode([("a", "bad")]),
        "ConditionalDistribution.estimator_map['a']",
    ),
    (
        "conditional",
        "mismatched_estimator_data_shape",
        lambda: _conditional_estimator().estimate(1.0, "not-a-tuple"),
        "ConditionalDistributionEstimator.estimate(suff_stat)",
    ),
    (
        "conditional",
        "unknown_conditioning_key",
        lambda: _conditional_estimator().estimate(1.0, ({"unknown-key": (1.0, 2.0)}, None, {"a": 1.0})),
        "ConditionalDistributionEstimator.estimator_map['unknown-key']",
    ),
    # --- optional ------------------------------------------------------------------------------
    (
        "optional",
        "wrong_container_type",
        lambda: _optional_encoder().seq_encode(5),
        "OptionalDistribution.seq_encode",
    ),
    (
        "optional",
        "malformed_field_type",
        lambda: _optional_encoder().seq_encode([1.0, "bad", None]),
        "OptionalDistribution.dist",
    ),
    (
        "optional",
        "mismatched_estimator_data_shape",
        lambda: _optional_estimator().estimate(1.0, "not-a-tuple"),
        "OptionalEstimator.estimate(suff_stat)",
    ),
]


class ContractErrorsTestCase(unittest.TestCase):
    def test_catalog_families_and_malformations_covered(self):
        families = {family for family, _, _, _ in CATALOG}
        self.assertEqual(families, {"composite", "mixture", "sequence", "conditional", "optional"})

    def test_malformed_input_raises_field_path_annotated_contract_error(self):
        for family, malformation, thunk, expected_path_substring in CATALOG:
            with self.subTest(family=family, malformation=malformation):
                with self.assertRaises(ContractError) as ctx:
                    thunk()
                err = ctx.exception
                message = str(err)
                self.assertIn(
                    expected_path_substring,
                    message,
                    "expected field path %r in message: %r" % (expected_path_substring, message),
                )
                self.assertIn("expected", message)
                self.assertIn("got", message)


if __name__ == "__main__":
    unittest.main()
