"""Tests for the encoded-data backend registry (register, don't branch)."""

import unittest

from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.utils.parallel.planner import (
    LocalEncodedData,
    available_encoded_data_backends,
    encoded_data,
    register_encoded_data_backend,
)


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
            from mixle.utils.parallel.planner import _ENCODED_DATA_BACKENDS

            _ENCODED_DATA_BACKENDS.pop("fake-test-backend", None)
            _ENCODED_DATA_BACKENDS.pop("ftb", None)

    def test_non_callable_factory_rejected(self):
        with self.assertRaises(TypeError):
            register_encoded_data_backend("bad", object())

    def test_registering_over_a_builtin_name_is_rejected(self):
        # Regression: registering under an already-taken name used to silently overwrite it -- a
        # third-party package colliding with 'local' (or any other backend) would hijack every
        # subsequent backend='local' call with no error, at whatever moment it happened to import.
        def other_local(data, **kwargs):
            return "HIJACKED"

        with self.assertRaises(ValueError) as ctx:
            register_encoded_data_backend("local", other_local)
        self.assertIn("local", str(ctx.exception))
        # the original registration must be untouched
        handle = encoded_data(self.data, encoder=self.encoder, backend="local")
        self.assertIsInstance(handle, LocalEncodedData)

    def test_registering_over_a_colliding_alias_is_rejected(self):
        def fake_backend(data, **kwargs):
            return "FAKE"

        with self.assertRaises(ValueError) as ctx:
            register_encoded_data_backend("fake-test-backend-2", fake_backend, aliases=("mp",))
        self.assertIn("mp", str(ctx.exception))
        from mixle.utils.parallel.planner import _ENCODED_DATA_BACKENDS

        # the whole call is rejected before any key is written, including the non-colliding name
        self.assertNotIn("fake-test-backend-2", _ENCODED_DATA_BACKENDS)

    def test_override_true_deliberately_replaces_a_registration(self):
        def fake_backend(data, *, encoder=None, **kwargs):
            return "OVERRIDDEN"

        register_encoded_data_backend("fake-test-backend-3", fake_backend)
        try:
            register_encoded_data_backend("fake-test-backend-3", fake_backend, override=True)  # same factory: no error

            def other_factory(data, **kwargs):
                return "OTHER"

            register_encoded_data_backend("fake-test-backend-3", other_factory, override=True)
            self.assertEqual(encoded_data(self.data, encoder=self.encoder, backend="fake-test-backend-3"), "OTHER")
        finally:
            from mixle.utils.parallel.planner import _ENCODED_DATA_BACKENDS

            _ENCODED_DATA_BACKENDS.pop("fake-test-backend-3", None)

    def test_re_registering_the_identical_factory_is_not_a_collision(self):
        # Re-importing a module that calls register_encoded_data_backend at import time (a legitimate,
        # idempotent scenario) must not raise just because Python re-ran the top-level call.
        def fake_backend(data, **kwargs):
            return "FAKE"

        register_encoded_data_backend("fake-test-backend-4", fake_backend)
        try:
            register_encoded_data_backend("fake-test-backend-4", fake_backend)  # same factory object again: fine
        finally:
            from mixle.utils.parallel.planner import _ENCODED_DATA_BACKENDS

            _ENCODED_DATA_BACKENDS.pop("fake-test-backend-4", None)

    def test_passthrough_of_existing_handle(self):
        handle = encoded_data(self.data, encoder=self.encoder, backend="local")
        self.assertIs(encoded_data(handle, backend="mpi"), handle)  # already a handle: returned as-is


if __name__ == "__main__":
    unittest.main()
