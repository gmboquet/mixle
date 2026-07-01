"""TaskModel (mixle.task.model): a callable str->label model that survives a save/load (incl. fresh process).

The featurizer is deterministic and the adapter self-describes, so loading from the artifact alone reconstructs
the whole raw->result function -- the package's point: a plain program loads a small local model and calls it.
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

from mixle.models.neural import make_mlp  # noqa: E402
from mixle.task.model import HashedNGram, TaskModel, TextClassifierIO  # noqa: E402


def _toy_classifier():
    # a tiny MLP over hashed n-gram features, two labels; weights are arbitrary (we only test plumbing here)
    feat = HashedNGram(n=3, dim=64, seed=1)
    cfg = {"input_dim": 64, "hidden_dims": [16], "output_dim": 2, "activation": "relu"}
    module = make_mlp(**cfg)
    adapter = TextClassifierIO(feat, ["ham", "spam"])
    return TaskModel(module, adapter, builder="mixle.mlp", config=cfg, task="toy spam classifier"), cfg


class FeaturizerTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        f = HashedNGram(n=3, dim=64, seed=7)
        a = f.transform(["hello world"])
        b = f.transform(["hello world"])
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a[0])), 1.0, places=5)

    def test_spec_round_trip(self):
        f = HashedNGram(n=4, dim=128, seed=3)
        g = HashedNGram.from_spec(f.to_spec())
        self.assertTrue(np.array_equal(f.transform(["abc"]), g.transform(["abc"])))


class TaskModelCallTest(unittest.TestCase):
    def test_call_returns_a_label(self):
        task, _ = _toy_classifier()
        out = task("buy now")
        self.assertIn(out, {"ham", "spam"})

    def test_batch_matches_single(self):
        task, _ = _toy_classifier()
        xs = ["hi there", "free money", "meeting at 3"]
        batch = task.batch(xs)
        singles = [task(x) for x in xs]
        self.assertEqual(batch, singles)


class TaskModelPersistenceTest(unittest.TestCase):
    def test_save_load_preserves_predictions(self):
        task, _ = _toy_classifier()
        xs = ["alpha", "beta gamma", "delta"]
        before = task.batch(xs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spam")
            task.save(path)
            loaded = TaskModel.load(path)
            self.assertEqual(loaded.task, "toy spam classifier")
            self.assertEqual(loaded.adapter.labels, ["ham", "spam"])
            after = loaded.batch(xs)
        self.assertEqual(before, after)

    def test_fresh_process_predictions_match(self):
        task, _ = _toy_classifier()
        xs = ["one two three", "spammy spam"]
        before = task.batch(xs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spam")
            task.save(path)
            out_file = os.path.join(d, "out.json")
            script = (
                "import torch; torch.set_num_threads(1)\n"
                "import json\n"
                "from mixle.task.model import TaskModel\n"
                f"t = TaskModel.load({path!r})\n"
                f"open({out_file!r},'w').write(json.dumps(t.batch({xs!r})))\n"
            )
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=os.getcwd())
            with open(out_file) as f:
                after = json.load(f)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
