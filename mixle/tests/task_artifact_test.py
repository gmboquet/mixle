"""Task artifact contract (mixle.task.artifact): durable save/load with a fresh-process round trip.

The acceptance bar for the artifact keystone: a torch-backed model saved here reloads from the manifest alone
-- in a brand-new interpreter -- to bit-identical outputs, with tied weights (the LM's tied head) intact.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task import artifact as A  # noqa: E402


class TorchRoundTripTest(unittest.TestCase):
    def test_mlp_round_trip_in_process(self):
        from mixle.models.neural import make_mlp

        cfg = {"input_dim": 4, "hidden_dims": [8, 8], "output_dim": 2, "activation": "relu"}
        module = make_mlp(**cfg)
        x = torch.randn(5, 4)
        with torch.no_grad():
            before = module(x).numpy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            A.save_module(path, module, "mixle.mlp", cfg, task="toy regressor")
            reloaded, manifest = A.load_module(path)
            self.assertEqual(manifest.payload, "torch")
            self.assertEqual(manifest.task, "toy regressor")
            with torch.no_grad():
                after = reloaded(x).numpy()
        self.assertTrue(np.allclose(before, after, atol=1e-6))

    def test_causal_lm_tied_weights_round_trip(self):
        # the LM ties head.weight = tok.weight; safetensors save_model/load_model must preserve that
        from mixle.models.transformer import build_causal_lm

        cfg = {"vocab": 32, "d_model": 16, "n_layer": 2, "n_head": 2, "block": 8}
        module = build_causal_lm(**cfg)
        x = torch.randint(0, 32, (3, 8)).float()
        with torch.no_grad():
            before = module(x).numpy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lm")
            A.save_module(path, module, "mixle.causal_lm", cfg, task="char LM")
            reloaded, _ = A.load_module(path)
            self.assertIs(reloaded.head.weight, reloaded.tok.weight)  # tie survived reconstruction
            with torch.no_grad():
                after = reloaded(x).numpy()
        self.assertTrue(np.allclose(before, after, atol=1e-6))

    def test_fresh_process_round_trip(self):
        # the real bar: save here, load in a brand-new interpreter, identical outputs
        from mixle.models.neural import make_mlp

        cfg = {"input_dim": 3, "hidden_dims": [5], "output_dim": 1, "activation": "tanh"}
        module = make_mlp(**cfg)
        x = np.asarray([[0.1, -0.2, 0.3], [1.0, 0.5, -0.5]], dtype=np.float32)
        with torch.no_grad():
            before = module(torch.from_numpy(x)).numpy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            A.save_module(path, module, "mixle.mlp", cfg)
            out_file = os.path.join(d, "out.json")
            script = (
                "import json, numpy as np, torch\n"
                "from mixle.task import artifact as A\n"
                f"m,_ = A.load_module({path!r})\n"
                f"x = np.asarray({x.tolist()!r}, dtype=np.float32)\n"
                "with torch.no_grad(): y = m(torch.from_numpy(x)).numpy()\n"
                f"open({out_file!r},'w').write(json.dumps(y.tolist()))\n"
            )
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=os.getcwd())
            with open(out_file) as f:
                after = np.asarray(json.load(f), dtype=np.float32)
        self.assertTrue(np.allclose(before, after, atol=1e-6))


class JsonPayloadTest(unittest.TestCase):
    def test_pure_distribution_round_trip(self):
        import mixle.stats as st

        model = st.GaussianDistribution(1.5, 2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g")
            A.save_json(path, model, task="density")
            reloaded, manifest = A.load_json(path)
            self.assertEqual(manifest.payload, "json")
        self.assertAlmostEqual(reloaded.log_density(1.5), model.log_density(1.5), places=9)


class BuilderRegistryTest(unittest.TestCase):
    def test_unknown_builder_raises_before_writing(self):
        from mixle.models.neural import make_mlp

        module = make_mlp(input_dim=2, hidden_dims=[2], output_dim=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x")
            with self.assertRaises(KeyError):
                A.save_module(path, module, "not.a.builder", {})
            self.assertFalse(os.path.exists(os.path.join(path, A.WEIGHTS_NAME)))

    def test_register_custom_builder(self):
        calls = {}

        def build(width):
            calls["width"] = width
            return torch.nn.Linear(width, width)

        A.register_builder("test.custom_linear", build)
        module = build(3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lin")
            A.save_module(path, module, "test.custom_linear", {"width": 3})
            reloaded, _ = A.load_module(path)
        self.assertEqual(reloaded.in_features, 3)


if __name__ == "__main__":
    unittest.main()
