"""Focused regressions for the 0.8.0 serialization-boundary audit."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

import numpy as np

import mixle.utils.serialization as serialization
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

    def test_decode_limits_depth_nodes_and_container_size(self):
        nested = None
        for _ in range(serialization.MAX_DECODE_DEPTH + 1):
            nested = [nested]
        with self.assertRaisesRegex(SerializationError, "depth"):
            from_serializable(nested)

        with (
            mock.patch.object(serialization, "MAX_DECODE_CONTAINER_ITEMS", 2),
            self.assertRaisesRegex(SerializationError, "item"),
        ):
            from_serializable([1, 2, 3])

        with (
            mock.patch.object(serialization, "MAX_DECODE_NODES", 2),
            self.assertRaisesRegex(SerializationError, "node"),
        ):
            from_serializable([1, 2])

    def test_constructor_schema_rejects_injected_or_inconsistent_state(self):
        payload = to_serializable(GaussianDistribution(0.0, 1.0))
        payload["state"]["items"].append(["injected", True])
        with self.assertRaisesRegex(SerializationError, "constructor-owned schema"):
            from_serializable(payload)

        payload = to_serializable(GaussianDistribution(0.0, 1.0))
        for pair in payload["state"]["items"]:
            if pair[0] == "log_const":
                pair[1] = 123.0
        with self.assertRaisesRegex(SerializationError, "constructor-owned schema"):
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


class RegistryInitializationTest(unittest.TestCase):
    def test_failed_initialization_is_atomic_and_retried(self):
        original_registry = serialization._CLASS_REGISTRY
        original_ids = serialization._CLASS_IDS
        original_ready = serialization._REGISTRY_READY
        calls = 0

        def fail_once(_package_name):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient registry failure")
            return ()

        serialization._CLASS_REGISTRY = {}
        serialization._CLASS_IDS = {}
        serialization._REGISTRY_READY = False
        try:
            with mock.patch.object(serialization, "_iter_distribution_modules", side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError, "transient registry failure"):
                    serialization.ensure_pysp_serialization_registry()
                self.assertFalse(serialization._REGISTRY_READY)
                self.assertEqual(serialization._CLASS_REGISTRY, {})
                self.assertEqual(serialization._CLASS_IDS, {})

                serialization.ensure_pysp_serialization_registry()
                self.assertTrue(serialization._REGISTRY_READY)
                self.assertGreaterEqual(calls, 3)
        finally:
            serialization._CLASS_REGISTRY = original_registry
            serialization._CLASS_IDS = original_ids
            serialization._REGISTRY_READY = original_ready


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
