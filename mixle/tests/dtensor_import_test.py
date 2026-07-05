"""Regression: TorchEngine's DTensor API resolves on both torch>=2.5 (public) and 2.0-2.4 (private).

The multi-GPU DTensor component-sharding path was silently broken on torch 2.0-2.4: mixle imported the
symbols only from the public `torch.distributed.tensor`, which is EMPTY before torch 2.5 (the symbols
live in the private `torch.distributed._tensor`). Found on a 2-GPU box; this pins the fallback so it
cannot regress.
"""

import importlib
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class DTensorImportTest(unittest.TestCase):
    def test_dtensor_api_resolves_when_available(self):
        # torch.distributed.tensor OR ._tensor must yield the 4 symbols on any torch that ships DTensor

        has_public = importlib.util.find_spec("torch.distributed.tensor") is not None
        has_private = importlib.util.find_spec("torch.distributed._tensor") is not None
        if not (has_public or has_private):
            self.skipTest("this torch build ships no DTensor")
        from mixle.engines.torch_engine import _dtensor_api

        dtensor, shard, replicate, distribute = _dtensor_api()
        for sym in (dtensor, shard, replicate, distribute):
            self.assertIsNotNone(sym)
        self.assertEqual(dtensor.__name__, "DTensor")

    def test_engine_registry_registers_dtensor(self):
        # the array-engine registry must map DTensor -> TorchEngine when DTensor exists (used for dispatch)
        import mixle.engines as eng

        if getattr(eng, "DTensor", None) is not None:
            self.assertIn(eng.DTensor, eng._ARRAY_ENGINE_REGISTRY)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class DTensorShardingGateTest(unittest.TestCase):
    """The DTensor component-sharding path is gated to torch >= 2.5 (older torch lacks op strategies)."""

    def test_ops_supported_matches_torch_version(self):
        import torch

        from mixle.engines.torch_engine import _dtensor_ops_supported

        major, minor = (int(p) for p in torch.__version__.split(".")[:2])
        self.assertEqual(_dtensor_ops_supported(), (major, minor) >= (2, 5))

    def test_component_sharding_gated_on_old_torch(self):
        # a *sentinel* mesh object is enough: the gate fires on version before touching the mesh
        from mixle.engines.torch_engine import TorchEngine, _dtensor_ops_supported

        if _dtensor_ops_supported():
            # torch >= 2.5: construction is allowed (no process group needed just to store the mesh)
            TorchEngine(device="cpu", mesh=object(), shard="components")
        else:
            with self.assertRaises(ValueError) as ctx:
                TorchEngine(device="cpu", mesh=object(), shard="components")
            self.assertIn("torch >= 2.5", str(ctx.exception))
            self.assertIn("model_parallel", str(ctx.exception))  # points to the working alternative

    def test_no_mesh_never_gated(self):
        from mixle.engines.torch_engine import TorchEngine

        TorchEngine(device="cpu")  # the ordinary single-device engine is unaffected on any torch


if __name__ == "__main__":
    unittest.main()
