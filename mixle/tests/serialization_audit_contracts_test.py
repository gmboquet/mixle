"""Focused regressions for the 0.8.0 serialization-boundary audit."""

from __future__ import annotations

import asyncio
import json
import unittest

import numpy as np

from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.utils.serialization import (
    SerializationError,
    deserialization_is_trusted,
    from_json,
    from_serializable,
    register_serializable_class,
    to_json,
    to_serializable,
    trusted_deserialization,
)

TAG = "__pysp_type__"


class StrictDecodeTest(unittest.TestCase):
    def test_nonstandard_raw_nonfinite_json_is_rejected(self):
        for text in ("NaN", "Infinity", "-Infinity", "[NaN]"):
            with self.subTest(text=text), self.assertRaises(SerializationError):
                from_json(text)

    def test_malformed_tag_fields_are_not_coerced(self):
        bad_payloads = [
            {TAG: "bytes", "data": "!!!!"},
            {TAG: "range", "start": 1.9, "stop": 5, "step": 1},
            {TAG: "ndarray", "dtype": "float64", "shape": [1.9], "data": [1.0]},
            {TAG: "float", "value": "nan", "extra": True},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(SerializationError):
                from_serializable(payload)

    def test_duplicate_decoded_dictionary_keys_are_rejected(self):
        payload = {TAG: "dict", "items": [["x", 1], ["x", 2]]}
        with self.assertRaisesRegex(SerializationError, "duplicate"):
            from_serializable(payload)


class ArrayCodecTest(unittest.TestCase):
    def test_complex_datetime_and_structured_arrays_round_trip(self):
        arrays = [
            np.asarray([1 + 2j, 3 - 4j], dtype=np.complex128),
            np.asarray(["2025-01-01", "2026-02-03"], dtype="datetime64[D]"),
            np.asarray([(1, 2.5), (3, 4.5)], dtype=[("count", "<i4"), ("value", "<f8")]),
        ]
        for array in arrays:
            with self.subTest(dtype=array.dtype):
                decoded = from_json(to_json(array))
                self.assertEqual(decoded.dtype, array.dtype)
                np.testing.assert_array_equal(decoded, array)


class ObjectGraphTest(unittest.TestCase):
    def test_shared_registered_object_identity_is_preserved(self):
        shared = GaussianDistribution(0.0, 1.0)
        decoded = from_serializable(to_serializable([shared, shared]))
        self.assertIs(decoded[0], decoded[1])

    def test_one_class_cannot_acquire_two_canonical_type_ids(self):
        class LocalValue:
            pass

        register_serializable_class(LocalValue, "test.serialization.LocalValue")
        with self.assertRaisesRegex(SerializationError, "already registered"):
            register_serializable_class(LocalValue, "test.serialization.OtherLocalValue")

    def test_mapping_order_uses_canonical_state_not_default_repr(self):
        class StableKey:
            def __init__(self, value):
                self.value = value

            __hash__ = object.__hash__

        register_serializable_class(StableKey, "test.serialization.StableKey")
        first = to_json({StableKey(2): "b", StableKey(1): "a"})
        second = to_json({StableKey(1): "a", StableKey(2): "b"})
        self.assertEqual(json.loads(first), json.loads(second))


class TrustScopeTest(unittest.TestCase):
    def test_trust_does_not_survive_in_a_child_task(self):
        async def scenario():
            release = asyncio.Event()

            async def child():
                await release.wait()
                return deserialization_is_trusted()

            with trusted_deserialization():
                self.assertTrue(deserialization_is_trusted())
                task = asyncio.create_task(child())
            release.set()
            return await task

        self.assertFalse(asyncio.run(scenario()))


if __name__ == "__main__":
    unittest.main()
