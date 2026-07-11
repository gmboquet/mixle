# Distributed / parallel / multi-GPU stress harness

An exhaustive bug-hunting harness for mixle's distributed estimation paths. Every check runs against a
serial baseline (bit-identical where the backend claims it, else within tolerance); a crash is recorded
as a BUG with its traceback rather than aborting the run. This harness found a real multi-GPU bug (the
`torch.distributed._tensor` import fallback, fixed in `mixle/engines/torch_engine.py`).

## Run it

```bash
# CPU — all data-parallel + model-parallel + MPI + planner + edge cases:
python dist_stress.py --heavy --only A,B,D,E,F,G

# on a multi-GPU box — adds torch-cuda parity, per-GPU placement, DTensor:
python dist_stress.py --gpu --heavy

# real multi-process DTensor component-sharding (one rank per GPU):
torchrun --nproc_per_node=2 dtensor_torchrun.py
```

## What it covers

| group | what |
|---|---|
| A | data-parallel EM (`num_chunks`, `backend="model_parallel"`) bit-identical to serial, across GMM / categorical-mixture / composite / HMM |
| B | MPI data-parallel EM (`mpi_fit` under `mpirun -n W`) vs serial |
| C | torch engine parity (CPU/CUDA), per-GPU placement, DTensor component-sharding |
| D | `model_sharding_plan` / `auto_parallel_estimator` correctness + more-shards-than-components |
| E | determinism (repeated distributed fits identical) |
| F | edge cases: K=1, tiny data, single point, empty chunks, K=32 stress |
| G | the planner (`plan()`) with and without data |

## Findings from the 2×RTX-3060 run (see `gpu_stress_2xRTX3060.log`)

- **32/33 PASS.** All data-parallel / model-parallel / MPI paths bit-identical to serial, GPU included.
- **BUG (fixed):** DTensor import was broken on torch 2.0–2.4 — the symbols live under
  `torch.distributed._tensor`, not the (empty) public `torch.distributed.tensor`. Fixed with a fallback;
  the sharded component-tensor round-trip then passes.
- **Fully DTensor-sharded mixture EM: works on torch >= 2.5, gated below.** Investigating the
  `logsumexp` crash showed torch 2.0-2.4 register **no** DTensor sharding strategies for the mixture
  E-step's ops (`logsumexp`, `isinf`, ...) — not just one, so backporting each is whack-a-mole. torch
  >= 2.5 runs the whole sharded fit **bit-identical to serial** (verified Δ=1e-15 on torch 2.12, CPU,
  2 ranks). `TorchEngine` now **gates the component-sharding path to torch >= 2.5** with a clear error
  pointing to `backend="model_parallel"` (the engine-agnostic component-parallel route, bit-identical
  on any torch, GPU included). `dtensor_cpu.py` reproduces + verifies this on any torch, no GPU needed
  (`torchrun --nproc_per_node=2 dtensor_cpu.py`): GATED on torch<2.5, PASS on >=2.5.
- Reproducible for pennies: the campaign cost **$0.06** of rented GPU; the DTensor gap was then
  root-caused and fixed entirely on CPU with a throwaway torch-2.4 venv — free.
