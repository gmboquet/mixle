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
from unittest import mock

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
                "import torch; torch.set_num_threads(1)\n"
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


class SaveModuleAtomicUpdateTest(unittest.TestCase):
    """save_module's weights.safetensors + manifest.json must move together as one atomic update.

    Previously they were written as two separate in-place steps (weights, then manifest); a failure in
    between left `path` holding NEW weights paired with the OLD manifest/provenance -- a mismatched pairing
    that load_module cannot detect as corrupt when the builder/config didn't change (it just silently loads
    the new weights next to stale metadata describing the run that produced the *previous* weights). These
    are the failure-injection regression tests for the fix (`_atomic_artifact_write`): the torch-free
    equivalents for save_json/save_arrays live in artifact_atomic_write_test.py.
    """

    @staticmethod
    def _mlp(fill: float):
        from mixle.models.neural import make_mlp

        cfg = {"input_dim": 4, "hidden_dims": [4], "output_dim": 2, "activation": "relu"}
        module = make_mlp(**cfg)
        with torch.no_grad():
            for p in module.parameters():
                p.fill_(fill)
        return module, cfg

    def test_manifest_write_failure_leaves_old_weights_and_manifest_both_intact(self):
        """Weights write succeeds, then the manifest write fails: the update must not stick -- old weights
        and old manifest/provenance must BOTH survive byte-for-byte, never a new/old mismatch."""
        module_v1, cfg = self._mlp(1.0)
        module_v2, _ = self._mlp(2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            A.save_module(path, module_v1, "mixle.mlp", cfg, task="v1", meta={"data_hash": "AAA"})
            with open(os.path.join(path, A.WEIGHTS_NAME), "rb") as f:
                old_weights = f.read()
            with open(os.path.join(path, A.MANIFEST_NAME), "rb") as f:
                old_manifest = f.read()

            with mock.patch.object(A, "_write_manifest", side_effect=RuntimeError("simulated manifest failure")):
                with self.assertRaises(RuntimeError):
                    A.save_module(path, module_v2, "mixle.mlp", cfg, task="v2", meta={"data_hash": "BBB"})

            # never a mismatched pairing: since the failure was injected on the manifest side, BOTH files
            # must still be exactly the v1 originals -- not "new weights, old manifest".
            with open(os.path.join(path, A.WEIGHTS_NAME), "rb") as f:
                self.assertEqual(f.read(), old_weights)
            with open(os.path.join(path, A.MANIFEST_NAME), "rb") as f:
                self.assertEqual(f.read(), old_manifest)

            reloaded, manifest = A.load_module(path)
            self.assertEqual(manifest.task, "v1")
            self.assertEqual(manifest.meta["data_hash"], "AAA")
            with torch.no_grad():
                self.assertEqual(next(reloaded.parameters()).flatten()[0].item(), 1.0)

            # no staging/backup directory leaked as a sibling of `path`
            self.assertEqual(os.listdir(d), ["mlp"])

    def test_publish_failure_restores_backup_path_never_missing(self):
        """A failure during the final directory-rename (not the write itself) must restore the old
        generation from its backup rather than leaving `path` missing or half-swapped."""
        module_v1, cfg = self._mlp(1.0)
        module_v2, _ = self._mlp(2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            A.save_module(path, module_v1, "mixle.mlp", cfg, task="v1")
            with open(os.path.join(path, A.WEIGHTS_NAME), "rb") as f:
                old_weights = f.read()

            real_replace = os.replace
            state = {"n": 0}

            def flaky_replace(src, dst, *a, **kw):
                if dst == path:
                    state["n"] += 1
                    if state["n"] == 1:  # fail only the first "staging -> path" commit, not the restore
                        raise OSError("simulated publish failure")
                return real_replace(src, dst, *a, **kw)

            with mock.patch("os.replace", side_effect=flaky_replace):
                with self.assertRaises(OSError):
                    A.save_module(path, module_v2, "mixle.mlp", cfg, task="v2")

            self.assertTrue(os.path.exists(path))  # must never be left missing
            with open(os.path.join(path, A.WEIGHTS_NAME), "rb") as f:
                self.assertEqual(f.read(), old_weights)
            _, manifest = A.load_module(path)
            self.assertEqual(manifest.task, "v1")
            self.assertEqual(os.listdir(d), ["mlp"])  # backup cleaned up (or restored, not left as debris)

    def test_successful_update_fully_replaces_old_generation_no_leaks(self):
        module_v1, cfg = self._mlp(1.0)
        module_v2, _ = self._mlp(2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            A.save_module(path, module_v1, "mixle.mlp", cfg, task="v1")
            A.save_module(path, module_v2, "mixle.mlp", cfg, task="v2", meta={"data_hash": "CCC"})

            reloaded, manifest = A.load_module(path)
            self.assertEqual(manifest.task, "v2")
            self.assertEqual(manifest.meta["data_hash"], "CCC")
            with torch.no_grad():
                self.assertEqual(next(reloaded.parameters()).flatten()[0].item(), 2.0)
            self.assertEqual(os.listdir(d), ["mlp"])  # no leftover staging/backup directories

    def test_first_save_failure_leaves_no_artifact_at_all(self):
        """A first-time save (no prior artifact) that fails during the write must not leave a half-written
        `path` behind -- it should not exist at all."""
        module_v1, cfg = self._mlp(1.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp")
            with mock.patch.object(A, "_write_manifest", side_effect=RuntimeError("simulated failure")):
                with self.assertRaises(RuntimeError):
                    A.save_module(path, module_v1, "mixle.mlp", cfg, task="v1")
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.listdir(d), [])  # no staging debris either


class TaskManifestValidationTest(unittest.TestCase):
    """TaskManifest.from_dict (the manifest-load boundary) must reject anything it can't safely interpret:
    a wrong/missing artifact_type, an unknown payload kind, a malformed payload shape, and any
    schema_version outside what this code was written to understand -- rather than silently accepting it
    and misinterpreting the artifact later.
    """

    def _good_dict(self, **overrides):
        d = {
            "artifact_type": A.ARTIFACT_TYPE,
            "schema_version": A.SCHEMA_VERSION,
            "payload": "torch",
            "builder": "mixle.mlp",
            "config": {},
            "task": "t",
            "io": {},
            "meta": {},
            "created_at": "",
        }
        d.update(overrides)
        return d

    def test_valid_manifest_still_accepted(self):
        m = A.TaskManifest.from_dict(self._good_dict())
        self.assertEqual(m.payload, "torch")
        self.assertEqual(m.builder, "mixle.mlp")

    def test_wrong_artifact_type_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(artifact_type="some.other.format"))

    def test_missing_artifact_type_rejected(self):
        d = self._good_dict()
        del d["artifact_type"]
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(d)

    def test_unknown_payload_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(payload="quantum-flux-capacitor"))

    def test_malformed_payload_shape_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(payload={"nested": "wrong shape"}))

    def test_torch_payload_missing_builder_rejected(self):
        d = self._good_dict()
        del d["builder"]
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(d)

    def test_arrays_payload_missing_builder_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(payload="arrays", builder=None))

    def test_non_dict_config_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(config="not-a-dict"))

    def test_non_dict_meta_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(meta="not-a-dict"))

    def test_non_dict_io_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(io="not-a-dict"))

    def test_unsupported_future_schema_version_999_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(schema_version=999))

    def test_unsupported_future_schema_version_999_string_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(self._good_dict(schema_version="999"))

    def test_manifest_that_is_not_a_dict_rejected(self):
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict("not-a-dict-at-all")  # type: ignore[arg-type]

    def test_missing_payload_key_rejected(self):
        d = self._good_dict()
        del d["payload"]
        with self.assertRaises(ValueError):
            A.TaskManifest.from_dict(d)

    def test_read_manifest_on_disk_enforces_the_same_validation(self):
        """The validation applies at the actual load boundary (read_manifest), not just from_dict directly."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad")
            os.makedirs(path)
            with open(os.path.join(path, A.MANIFEST_NAME), "w") as f:
                json.dump(self._good_dict(schema_version=999), f)
            with self.assertRaises(ValueError):
                A.read_manifest(path)


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
