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


if __name__ == "__main__":
    unittest.main()
