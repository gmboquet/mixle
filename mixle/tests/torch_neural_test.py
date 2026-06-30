"""StreamingTokenEncodedData: the distributed streaming neural-training handle (the inverted reduction).

Single-process: ``seq_estimate(handle, estimator, None)`` dispatches to the handle and trains a transformer LM.
Two-rank (gated torchrun smoke): each rank keeps its OWN shard and the only cross-rank collective is the
in-backward gradient all-reduce -- no gather-to-root -- and the ranks end with a consistent model.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TorchNeuralHandleTest(unittest.TestCase):
    def _corpus(self):
        text = "the model is the message. a tight model spends computation where it earns its keep. " * 14
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return np.array([stoi[c] for c in text]), len(chars)

    def test_single_process_handle_trains_via_seq_estimate(self):
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeafEstimator
        from mixle.models.transformer import build_causal_lm
        from mixle.stats.compute.sequence import seq_estimate
        from mixle.utils.parallel.torch_neural import StreamingTokenEncodedData

        torch.manual_seed(0)
        ids, v = self._corpus()
        block = 32
        est = StreamingTransformerLeafEstimator(
            build_causal_lm(v, d_model=64, n_layer=2, n_head=4, block=block), lr=3e-3
        )
        handle = StreamingTokenEncodedData(ids, block=block, batch_size=64, epochs=40)  # world=1, single process
        self.assertEqual(handle.world, 1)
        leaf = seq_estimate(handle, est, None)  # dispatches to handle.pysp_seq_estimate -> trains
        ctx = np.stack([ids[i : i + block] for i in range(48)]).astype("float32")
        nll = -np.mean(leaf.seq_log_density((ctx, ids[block : block + 48])))
        self.assertLess(nll, 0.5)  # the streamed model learned next-token prediction

    @unittest.skipUnless(os.environ.get("MIXLE_RUN_TORCHRUN_SMOKE"), "gated: set MIXLE_RUN_TORCHRUN_SMOKE=1")
    def test_two_rank_distributed_streaming_is_consistent(self):
        script = """
import os, sys, numpy as np, torch
import torch.distributed as dist
sys.path.insert(0, %(repo)r)
from mixle.models.transformer import build_causal_lm
from mixle.models.streaming_transformer_leaf import StreamingTransformerLeafEstimator
from mixle.utils.parallel.torch_neural import StreamingTokenEncodedData
from mixle.stats.compute.sequence import seq_estimate
torch.manual_seed(0)
text = "the model is the message. a tight model spends computation where it earns its keep. " * 16
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text]); block = 32
module = build_causal_lm(V, d_model=64, n_layer=2, n_head=4, block=block)
est = StreamingTransformerLeafEstimator(module, lr=3e-3, device="cpu")
handle = StreamingTokenEncodedData(ids, block=block, batch_size=64, epochs=50)  # inits PG from torchrun env, shards by rank
leaf = seq_estimate(handle, est, None)  # SPMD: per-rank shard, in-backward all-reduce, no gather-to-root
ctx = np.stack([ids[i:i+block] for i in range(48)]).astype('float32')
nll = -float(np.mean(leaf.seq_log_density((ctx, ids[block:block+48]))))
chk = sum(float(p.detach().double().sum()) for p in module.parameters())
t = torch.tensor([chk], dtype=torch.float64); dist.all_reduce(t)
w = dist.get_world_size(); consistent = abs(t.item() - chk * w) < 1e-6 * abs(chk * w)
if dist.get_rank() == 0:
    print('TORCH-NEURAL-OK world=%%d nll=%%.3f consistent=%%s' %% (w, nll, consistent), flush=True)
dist.barrier(); handle.close()
""" % {"repo": REPO}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "dist_neural.py")
            with open(path, "w") as f:
                f.write(script)
            env = dict(os.environ, PYTHONPATH=REPO, MASTER_ADDR="127.0.0.1")
            if sys.platform != "darwin":
                env.setdefault("GLOO_SOCKET_IFNAME", "lo")
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--rdzv_endpoint=127.0.0.1:0",
                    "--local-addr=127.0.0.1",
                    "--nproc_per_node=2",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
        self.assertEqual(res.returncode, 0, "torchrun failed:\n%s\n%s" % (res.stdout, res.stderr))
        self.assertIn("TORCH-NEURAL-OK", res.stdout)
        self.assertIn("consistent=True", res.stdout)


if __name__ == "__main__":
    unittest.main()
