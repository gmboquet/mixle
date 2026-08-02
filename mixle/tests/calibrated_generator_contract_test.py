"""Seed derivation is a function of a prompt's value, and the oracle is spent only where it counts.

MXR-080-1848: ``_derive_seed`` promised cross-process stability while hashing ``repr(prompt)``. A set's
iteration order follows element hashes and a dict's repr follows insertion order, so two equal prompts
encoded differently; and any type merely defining ``__repr__`` was taken as proof of canonicality.

MXR-080-1849: ``calibrate`` scored every row with the correctness oracle although only certification-half
verdicts reach the bound.
"""

import itertools
import unittest
import warnings

from mixle.task.calibrated_generator import (
    CalibratedGenerator,
    _derive_seed,
    _is_canonically_representable,
    _seed_key,
)


class AddressBearingRepr:
    """A custom ``__repr__`` that is not canonical -- defining one proves nothing about its content."""

    def __repr__(self) -> str:
        return f"<AddressBearingRepr {id(self):#x}>"


class SeedKeyTest(unittest.TestCase):
    def test_a_set_encodes_independently_of_iteration_order(self):
        # repr({"a","b"}) depends on element hashes, which PYTHONHASHSEED changes between processes.
        self.assertEqual(_seed_key({"alpha", "beta"}), _seed_key({"beta", "alpha"}))
        self.assertEqual(_derive_seed(7, {"alpha", "beta"}), _derive_seed(7, {"beta", "alpha"}))

    def test_a_mapping_encodes_independently_of_insertion_order(self):
        self.assertEqual(_derive_seed(7, {"a": 1, "b": 2}), _derive_seed(7, {"b": 2, "a": 1}))

    def test_sequence_order_is_still_part_of_the_value(self):
        self.assertNotEqual(_derive_seed(7, (1, 2)), _derive_seed(7, (2, 1)))
        self.assertNotEqual(_derive_seed(7, [1, 2]), _derive_seed(7, [2, 1]))

    def test_distinct_values_still_seed_distinctly(self):
        seeds = {_derive_seed(7, value) for value in ("a", b"a", 1, 1.5, True, None, (1,), frozenset({1}))}
        self.assertEqual(len(seeds), 8)

    def test_a_string_and_its_bytes_do_not_collide(self):
        self.assertNotEqual(_seed_key("61"), _seed_key(b"a"))

    def test_nesting_is_encoded_through(self):
        self.assertEqual(_derive_seed(7, {"k": {1, 2}}), _derive_seed(7, {"k": {2, 1}}))

    def test_a_self_referential_container_terminates(self):
        cycle: list = [1]
        cycle.append(cycle)
        self.assertIsNone(_seed_key(cycle))

    def test_a_member_containing_the_delimiter_does_not_collide(self):
        """The encoding is length-delimited, not comma-joined (MXR-080-1858).

        A comma-joined encoding is not injective when a member may itself contain the delimiter:
        ``["x,s:y"]`` and ``["x", "y"]`` both produced ``q:[s:x,s:y]`` and drew the same seed.
        """
        self.assertNotEqual(_seed_key(["x,s:y"]), _seed_key(["x", "y"]))
        self.assertNotEqual(_derive_seed(7, ["x,s:y"]), _derive_seed(7, ["x", "y"]))

    def test_key_equality_tracks_value_equality(self):
        # The general property the collision above violated: two prompts share a key exactly when
        # they are equal. Booleans are the one deliberate exception (see _seed_key).
        distinct = [
            "",
            "a",
            "x,s:y",
            "s1:x",
            ["x", "y"],
            ["x,s:y"],
            ["xy"],
            ("x", "y"),
            {"a": 1},
            {"a1": None},
            {1, 2},
            [["a"], ["b"]],
            [["a", "b"]],
            1.5,
            None,
            b"a",
            "a1:a",
            {"ab": "c"},
            {"a": "bc"},
            [[]],
            [],
            [[[]]],
            {"": ""},
            (),
            ("",),
        ]
        for left, right in itertools.combinations(distinct, 2):
            with self.subTest(left=repr(left), right=repr(right)):
                self.assertEqual(_seed_key(left) == _seed_key(right), left == right)

    def test_equal_numbers_share_a_seed(self):
        # 1 == 1.0, so they are the same prompt; encoding repr() gave them different draws.
        self.assertEqual(_seed_key(1), _seed_key(1.0))
        self.assertEqual(_derive_seed(7, 1), _derive_seed(7, 1.0))

    def test_a_list_and_a_tuple_are_different_prompts(self):
        self.assertNotEqual(["x"], ("x",))
        self.assertNotEqual(_seed_key(["x"]), _seed_key(("x",)))

    def test_a_set_and_a_frozenset_are_the_same_prompt(self):
        self.assertEqual({1, 2}, frozenset({1, 2}))
        self.assertEqual(_seed_key({1, 2}), _seed_key(frozenset({1, 2})))


class CanonicalityTest(unittest.TestCase):
    def test_canonical_types_are_accepted(self):
        for value in ("a", b"a", bytearray(b"a"), 1, 1.5, True, None, (1, "b"), [1], {1}, {"a": 1}):
            with self.subTest(value=value):
                self.assertTrue(_is_canonically_representable(value))

    def test_defining_a_repr_is_not_proof_of_canonicality(self):
        self.assertFalse(_is_canonically_representable(AddressBearingRepr()))

    def test_an_uncanonical_member_disqualifies_its_container(self):
        self.assertFalse(_is_canonically_representable(("ok", AddressBearingRepr())))

    def test_an_uncanonical_prompt_warns_rather_than_failing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _derive_seed(7, AddressBearingRepr())
        self.assertTrue(any("MXR-080-1848" in str(entry.message) for entry in caught))

    def test_a_canonical_prompt_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _derive_seed(7, {"a": 1})
        self.assertEqual([entry for entry in caught if "1848" in str(entry.message)], [])


class OracleSpendTest(unittest.TestCase):
    """The oracle is consulted once per certification row, and not at all on proposal rows."""

    def _calibrate(self, n_prompts):
        calls = []

        def generate(prompt, k):
            return [prompt * 10 + offset for offset in range(k)]

        def score(candidate):
            return float(candidate % 7)

        def is_correct(prompt, candidate):
            calls.append(prompt)
            return candidate % 2 == 0

        model = CalibratedGenerator(generate, score, alpha=0.5, k=3, seed=0)
        model.calibrate(list(range(n_prompts)), is_correct)
        return calls, model

    def test_only_the_certification_half_reaches_the_oracle(self):
        calls, model = self._calibrate(10)
        self.assertEqual(len(calls), 5)
        self.assertEqual(model.risk_receipt["oracle_calls"], 5)
        self.assertEqual(model.risk_receipt["certification_count"], 5)
        self.assertEqual(model.risk_receipt["proposal_count"], 5)

    def test_the_oracle_sees_a_prefix_in_order(self):
        # Load-bearing: an oracle with no row index can only recover identity by counting its calls,
        # so the rows it is given must be prompts[0], prompts[1], ... with no gaps.
        calls, _ = self._calibrate(10)
        self.assertEqual(calls, [0, 1, 2, 3, 4])

    def test_an_odd_count_gives_certification_the_larger_half(self):
        calls, model = self._calibrate(9)
        self.assertEqual(len(calls), 5)
        self.assertEqual(model.risk_receipt["proposal_count"], 4)

    def test_every_row_is_scored_exactly_once(self):
        calls, _ = self._calibrate(10)
        self.assertEqual(len(calls), len(set(calls)))


if __name__ == "__main__":
    unittest.main()
