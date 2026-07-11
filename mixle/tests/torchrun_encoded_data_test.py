import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mixle.inference import seq_estimate
from mixle.inference.estimation import optimize
from mixle.inference.streaming import streaming_accumulate
from mixle.stats import GaussianDistribution, GaussianEstimator, seq_encode, seq_log_density_sum
from mixle.utils.parallel.planner import Resources, encoded_data, is_encoded_data_handle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from mixle.utils.parallel.torchrun import TorchRunEncodedData
else:
    TorchRunEncodedData = None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TorchRunEncodedDataTestCase(unittest.TestCase):
    def test_single_rank_handle_matches_local_scoring_and_estimate(self):
        data = list(np.linspace(-2.0, 2.0, 40))
        model = GaussianDistribution(0.25, 1.5)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with TorchRunEncodedData(data, model=model, estimator=estimator, sub_chunks=2) as enc:
            self.assertTrue(is_encoded_data_handle(enc))
            self.assertEqual(len(enc), len(data))
            cnt_h, ll_h = seq_log_density_sum(enc, model)
            fitted_h = seq_estimate(enc, estimator, model)
            n_h, acc_h = streaming_accumulate(enc, estimator, model)

        cnt_l, ll_l = seq_log_density_sum(enc_local, model)
        fitted_l = seq_estimate(enc_local, estimator, model)
        n_l, acc_l = streaming_accumulate(enc_local, estimator, model)

        self.assertEqual(cnt_h, cnt_l)
        self.assertEqual(n_h, n_l)
        self.assertAlmostEqual(ll_h, ll_l, places=10)
        self.assertAlmostEqual(fitted_h.mu, fitted_l.mu, places=10)
        self.assertAlmostEqual(fitted_h.sigma2, fitted_l.sigma2, places=10)
        np.testing.assert_allclose(acc_h.value(), acc_l.value(), rtol=1.0e-12, atol=1.0e-12)

    def test_factory_and_optimize_can_use_torchrun_backend_single_rank(self):
        data = list(np.linspace(-2.0, 2.0, 60))
        start = GaussianDistribution(1.0, 4.0)
        estimator = GaussianEstimator()

        with encoded_data(data, model=start, estimator=estimator, backend="torchrun", sub_chunks=2) as enc:
            self.assertIsInstance(enc, TorchRunEncodedData)

        fitted = optimize(
            data,
            estimator,
            prev_estimate=start,
            backend="torchrun",
            sub_chunks=2,
            max_its=2,
            delta=None,
            out=io.StringIO(),
        )
        local = optimize(data, estimator, prev_estimate=start, max_its=2, delta=None, out=io.StringIO())

        self.assertAlmostEqual(fitted.mu, local.mu, places=10)
        self.assertAlmostEqual(fitted.sigma2, local.sigma2, places=10)

    def test_resources_can_describe_torchrun_world(self):
        resources = Resources.from_torchrun()
        self.assertGreaterEqual(len(resources.devices), 1)
        self.assertTrue(all(device.engine == "torch" for device in resources.devices))

    def test_torchrun_two_rank_smoke(self):
        if os.environ.get("MIXLE_RUN_TORCHRUN_SMOKE") != "1":
            self.skipTest("set MIXLE_RUN_TORCHRUN_SMOKE=1 to launch a local two-rank torchrun smoke test")
        script = r"""
import io
import os
import sys
import numpy as np

sys.path.insert(0, %(repo)r)

from mixle.engines import TorchEngine
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator, \
    estimate_component_shard_value, seq_encode, seq_log_density_sum, tie_component_shard_values
from mixle.inference import seq_estimate
from mixle.inference.estimation import constant, optimize
from mixle.inference.streaming import StreamingEstimator, streaming_accumulate
from mixle.utils.parallel.torchrun import TorchRunEncodedData, torchrun_out

import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard

data = list(np.linspace(-2.0, 2.0, 60))
start = GaussianDistribution(1.0, 4.0)
estimator = GaussianEstimator()
enc = TorchRunEncodedData(data, model=start, estimator=estimator, sub_chunks=2, backend='gloo')

cnt, ll = enc.pysp_seq_log_density_sum(start)
if torch.distributed.get_rank() == 0:
    enc_local = seq_encode(data, model=start)
    cnt_l, ll_l = seq_log_density_sum(enc_local, start)
    assert cnt == cnt_l, (cnt, cnt_l)
    assert abs(ll - ll_l) < 1.0e-8, (ll, ll_l)

n_h, acc_h = streaming_accumulate(enc, estimator, start)
stream = StreamingEstimator(estimator, schedule=constant(0.5), model=start)
stream_model = stream.update(enc_data=enc)
stream_values = (stream_model.mu, stream_model.sigma2)

gathered_stream = [None for _ in range(torch.distributed.get_world_size())]
torch.distributed.all_gather_object(gathered_stream, stream_values)
if torch.distributed.get_rank() == 0:
    enc_local = seq_encode(data, model=start)
    n_l, acc_l = streaming_accumulate(enc_local, estimator, start)
    stream_l = StreamingEstimator(estimator, schedule=constant(0.5), model=start)
    stream_model_l = stream_l.update(enc_data=enc_local)
    assert n_h == n_l, (n_h, n_l)
    np.testing.assert_allclose(acc_h.value(), acc_l.value(), rtol=1.0e-12, atol=1.0e-12)
    assert all(abs(v[0] - gathered_stream[0][0]) < 1.0e-12 and
               abs(v[1] - gathered_stream[0][1]) < 1.0e-12 for v in gathered_stream), gathered_stream
    assert abs(stream_model.mu - stream_model_l.mu) < 1.0e-10, (stream_model.mu, stream_model_l.mu)
    assert abs(stream_model.sigma2 - stream_model_l.sigma2) < 1.0e-10, (stream_model.sigma2, stream_model_l.sigma2)

model = optimize(None, estimator, enc_data=enc, prev_estimate=start,
                 max_its=2, delta=None, out=torchrun_out())
values = (model.mu, model.sigma2)
gathered = [None for _ in range(torch.distributed.get_world_size())]
torch.distributed.all_gather_object(gathered, values)
if torch.distributed.get_rank() == 0:
    assert all(abs(v[0] - gathered[0][0]) < 1.0e-12 and
               abs(v[1] - gathered[0][1]) < 1.0e-12 for v in gathered), gathered
    local = optimize(data, estimator, prev_estimate=start, max_its=2,
                     delta=None, out=io.StringIO())
    assert abs(model.mu - local.mu) < 1.0e-10, (model.mu, local.mu)
    assert abs(model.sigma2 - local.sigma2) < 1.0e-10, (model.sigma2, local.sigma2)

mesh = DeviceMesh('cpu', list(range(torch.distributed.get_world_size())))
engine = TorchEngine(dtype=torch.float64, mesh=mesh, shard='components')
mix = MixtureDistribution([
    GaussianDistribution(-3.0, 0.6),
    GaussianDistribution(-1.0, 0.8),
    GaussianDistribution(1.0, 1.0),
    GaussianDistribution(3.0, 1.2),
], [0.15, 0.25, 0.30, 0.30])
mix_est = MixtureEstimator([GaussianEstimator(keys='shared_mp_gaussian') for _ in range(4)])
mix_data = mix.sampler(seed=23).sample(size=80)
mix_enc = mix.dist_to_encoder().seq_encode(mix_data)
mix_kernel = mix.kernel(engine=engine, estimator=mix_est)

component_scores = mix_kernel.component_scores(mix_enc)
assert isinstance(component_scores, DTensor)
assert isinstance(component_scores.placements[0], Shard)
assert component_scores.placements[0].dim == 1
np.testing.assert_allclose(engine.to_numpy(mix_kernel.score(mix_enc)),
                           mix.seq_log_density(mix_enc), rtol=1.0e-10, atol=1.0e-10)

resident = mix_kernel.resident_accumulate(mix_enc, engine.asarray(np.ones(len(mix_data))))
local_start, local_value = resident.local_value()
local_shard = resident.estimate_component_shard(mix_est, total_count=float(len(mix_data)))
assert local_shard.component_start == local_start

gathered_shards = [None for _ in range(torch.distributed.get_world_size())]
torch.distributed.all_gather_object(gathered_shards, (local_start, local_value))
ranges = sorted((start, start + len(value[0])) for start, value in gathered_shards)
assert ranges[0][0] == 0 and ranges[-1][1] == mix.num_components, ranges
assert all(a[1] == b[0] for a, b in zip(ranges[:-1], ranges[1:])), ranges

tied_values = tie_component_shard_values(mix_est, tuple(gathered_shards))
tied_local = [item for item in tied_values if item[0] == local_start][0]
tied_shard = estimate_component_shard_value(
    mix_est, tied_local[0], tied_local[1], total_count=float(len(mix_data)))
expected_tied = seq_estimate([(len(mix_data), mix_enc)], mix_est, mix)
np.testing.assert_allclose(
    tied_shard.weights,
    expected_tied.w[tied_shard.component_start:tied_shard.component_stop],
    rtol=1.0e-10,
    atol=1.0e-10,
)
for offset, component in enumerate(tied_shard.components):
    idx = tied_shard.component_start + offset
    expected_component = expected_tied.components[idx]
    np.testing.assert_allclose(
        [component.mu, component.sigma2],
        [expected_component.mu, expected_component.sigma2],
        rtol=1.0e-10,
        atol=1.0e-10,
    )

summary = (
    tied_shard.component_start,
    tied_shard.component_stop,
    tied_shard.weights.tolist(),
    [(float(component.mu), float(component.sigma2)) for component in tied_shard.components],
)
gathered_summary = [None for _ in range(torch.distributed.get_world_size())]
torch.distributed.all_gather_object(gathered_summary, summary)
if torch.distributed.get_rank() == 0:
    assert sorted((item[0], item[1]) for item in gathered_summary) == ranges, gathered_summary
    print('TORCHRUN-BACKEND-OK', flush=True)
torch.distributed.barrier()
enc.close()
""" % {"repo": REPO}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "torchrun_check.py")
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
                timeout=90,
                env=env,
            )
        self.assertEqual(res.returncode, 0, "torchrun failed:\n%s\n%s" % (res.stdout, res.stderr))
        self.assertIn("TORCHRUN-BACKEND-OK", res.stdout)


if __name__ == "__main__":
    unittest.main()
