"""Real multi-GPU DTensor component-sharding test — launch: torchrun --nproc_per_node=2 dtensor_torchrun.py

One rank per GPU, a device mesh over the ranks, TorchEngine(mesh, shard='components'). Exercises, in
escalating order (each stage a separate receipt so we see exactly where it breaks):
  1. process group + device mesh init
  2. a sharded component-parameter tensor round-trips (asarray on Shard(0) -> gather to host)
  3. a full mixle EM fit with the sharded engine; rank 0 compares to the serial numpy baseline.
"""

import json
import os
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(LOCAL)


def emit(stage, status, detail=""):
    if RANK == 0:
        print(f"DTENSOR {stage} {status} {detail}", flush=True)


def main():
    import mixle.stats as st
    from mixle.engines import TorchEngine
    from mixle.inference import optimize

    # stage 1: process group + mesh
    try:
        dist.init_process_group(backend="nccl")
        mesh = init_device_mesh("cuda", (WORLD,))
        emit("1_mesh", "PASS", f"world={WORLD} local={LOCAL}")
    except Exception as e:  # noqa: BLE001
        emit("1_mesh", "BUG", f"{type(e).__name__}: {str(e)[:100]}")
        return

    # stage 2: a sharded component tensor round-trips through the engine
    try:
        eng = TorchEngine(device=f"cuda:{LOCAL}", mesh=mesh, shard="components")
        stacked = np.arange(WORLD * 3, dtype=np.float64).reshape(WORLD, 3)  # (components, dim)
        placed = eng.stack_components(stacked, axis=0) if hasattr(eng, "stack_components") else eng.asarray(stacked)
        back = eng.to_numpy(placed) if hasattr(eng, "to_numpy") else np.asarray(placed)
        ok = np.allclose(np.asarray(back), stacked)
        emit(
            "2_shard_roundtrip",
            "PASS" if ok else "FAIL",
            f"max|Δ|={float(np.abs(np.asarray(back) - stacked).max()):.2e}",
        )
    except Exception as e:  # noqa: BLE001
        emit("2_shard_roundtrip", "BUG", f"{type(e).__name__}: {str(e)[:120]}")

    # stage 3: a full EM fit with the sharded engine vs serial baseline
    try:
        rng = np.random.RandomState(0)
        k = WORLD * 2  # components a multiple of the mesh size
        comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(k)]
        true = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))
        data = true.sampler(1).sample(3000)
        est = st.MixtureEstimator([st.GaussianEstimator() for _ in range(k)])
        init = st.MixtureDistribution(
            [st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(k)], [1.0 / k] * k
        )

        eng = TorchEngine(device=f"cuda:{LOCAL}", mesh=mesh, shard="components")
        fit = optimize(data, est, prev_estimate=init, max_its=6, out=None, engine=eng)
        if RANK == 0:
            base = optimize(data, est, prev_estimate=init, max_its=6, out=None, backend="local")
            fw = np.asarray(fit.w, dtype=float)
            bw = np.asarray(base.w, dtype=float)
            d = float(np.abs(fw - bw).max())
            emit("3_sharded_fit", "PASS" if d < 1e-2 else "FAIL", f"max|Δw vs serial|={d:.2e}")
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        emit("3_sharded_fit", "BUG", f"{type(e).__name__}: {str(e)[:140]}")
        if RANK == 0:
            print("DTENSOR_TB " + json.dumps(traceback.format_exc().splitlines()[-6:]), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        print("DTENSOR_DONE", flush=True)


if __name__ == "__main__":
    main()
