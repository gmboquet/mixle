"""Tests for the encoded-data backend registry (register, don't branch)."""

import unittest

from pysp.planner import (
    LocalEncodedData,
    available_encoded_data_backends,
    encoded_data,
    register_encoded_data_backend,
)
from pysp.stats.leaf.gaussian import GaussianDistribution


class EncodedDataBackendRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = GaussianDistribution(0.0, 1.0)
        self.data = list(self.dist.sampler(0).sample(50))
        self.encoder = self.dist.dist_to_encoder()

    def test_builtin_backends_registered(self):
        names = available_encoded_data_backends()
        for expected in ("local", "mp", "multiprocessing", "mpi", "spark", "dask", "torchrun"):
            self.assertIn(expected, names)

    def test_local_dispatch_still_works(self):
        handle = encoded_data(self.data, encoder=self.encoder, backend="local")
        self.assertIsInstance(handle, LocalEncodedData)

    def test_unknown_backend_lists_registered(self):
        with self.assertRaises(ValueError) as ctx:
            encoded_data(self.data, encoder=self.encoder, backend="quantum")
        msg = str(ctx.exception)
        self.assertIn("quantum", msg)
        self.assertIn("local", msg)  # error names the registered backends

    def test_custom_backend_is_dispatched(self):
        seen = {}

        def fake_backend(data, *, encoder=None, **kwargs):
            seen["data_len"] = len(data)
            seen["encoder"] = encoder
            return "FAKE_HANDLE"

        register_encoded_data_backend("fake-test-backend", fake_backend, aliases=("ftb",))
        try:
            self.assertEqual(encoded_data(self.data, encoder=self.encoder, backend="fake-test-backend"), "FAKE_HANDLE")
            self.assertEqual(encoded_data(self.data, encoder=self.encoder, backend="FTB"), "FAKE_HANDLE")  # alias, case
            self.assertEqual(seen["data_len"], 50)
            self.assertIs(seen["encoder"], self.encoder)
        finally:
            from pysp.planner import _ENCODED_DATA_BACKENDS

            _ENCODED_DATA_BACKENDS.pop("fake-test-backend", None)
            _ENCODED_DATA_BACKENDS.pop("ftb", None)

    def test_non_callable_factory_rejected(self):
        with self.assertRaises(TypeError):
            register_encoded_data_backend("bad", object())

    def test_passthrough_of_existing_handle(self):
        handle = encoded_data(self.data, encoder=self.encoder, backend="local")
        self.assertIs(encoded_data(handle, backend="mpi"), handle)  # already a handle: returned as-is


if __name__ == "__main__":
    unittest.main()
