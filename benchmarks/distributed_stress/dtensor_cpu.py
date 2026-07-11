"""Reproduce + verify the torch-2.4 DTensor logsumexp fix on CPU. Launch:
    torchrun --nproc_per_node=2 dtensor_cpu.py
DTensor works on CPU via gloo; the logsumexp-sharding gap is torch-VERSION specific (2.0-2.4), not GPU
specific, so CPU reproduces the exact bug. Rank 0 prints receipts.
"""

import os

import numpy as np
import torch
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    from torch.distributed._device_mesh import init_device_mesh

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))


def emit(*a):
    if RANK == 0:
        print("CPUDT", *a, flush=True)


def main():
    import mixle.stats as st
    from mixle.engines.torch_engine import TorchEngine
    from mixle.inference import optimize

    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (WORLD,))
    emit("torch", torch.__version__, "world", WORLD)

    # 0. confirm the RAW bug on this torch: native logsumexp over a sharded axis
    try:
        from torch.distributed.tensor import Shard, distribute_tensor
    except ImportError:
        from torch.distributed._tensor import Shard, distribute_tensor

    x = torch.randn(6, WORLD * 2)
    dt = distribute_tensor(x, mesh, [Shard(1)])
    try:
        torch.logsumexp(dt, dim=1)
        emit("raw_native_logsumexp", "WORKS", "(torch>=2.5 has the strategy)")
    except NotImplementedError as e:
        emit("raw_native_logsumexp", "RAISES", f"(the bug: {str(e)[:60]})")

    # the real payload: a full DTensor-sharded mixture EM fit (the path that hit the bug)
    rng = np.random.RandomState(0)
    k = WORLD * 2
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(k)]
    true = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))
    data = true.sampler(1).sample(3000)
    est = st.MixtureEstimator([st.GaussianEstimator() for _ in range(k)])
    init = st.MixtureDistribution([st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(k)], [1.0 / k] * k)

    try:
        eng = TorchEngine(device="cpu", mesh=mesh, shard="components")
    except ValueError as e:
        emit("sharded_mixture_fit", "GATED", f"(expected on torch<2.5: {str(e)[:70]})")
        dist.destroy_process_group()
        emit("DONE")
        return
    try:
        fit = optimize(data, est, prev_estimate=init, max_its=6, out=None, engine=eng)
        if RANK == 0:
            base = optimize(data, est, prev_estimate=init, max_its=6, out=None, backend="local")
            d = float(np.abs(np.asarray(fit.w) - np.asarray(base.w)).max())
            emit("sharded_mixture_fit", "PASS" if d < 1e-2 else "FAIL", f"max|Δw vs serial|={d:.2e}")
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        emit("sharded_mixture_fit", "BUG", f"{type(e).__name__}: {str(e)[:90]}")

    dist.destroy_process_group()
    emit("DONE")


if __name__ == "__main__":
    main()
