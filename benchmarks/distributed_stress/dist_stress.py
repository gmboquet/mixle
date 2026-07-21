"""Exhaustive distributed / parallel / multi-GPU bug-hunting harness for mixle.

Every check is wrapped so a crash is recorded as a BUG (with traceback), not a harness abort. Correctness
is measured against a serial baseline (bit-identical where the backend claims it, else within tol). Runs
locally (CPU/MPS) and on a multi-GPU box; pass --gpu to enable the CUDA/DTensor tests.

Usage:  python dist_stress.py [--gpu] [--heavy] [--only A,B,C]
Report: prints a per-check line + a JSON summary block at the end.
"""

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

import mixle.stats as st
from mixle.inference import optimize

RESULTS: list = []


@dataclass
class Check:
    group: str
    name: str
    status: str  # PASS | FAIL | BUG | SKIP
    detail: str = ""
    extra: dict = field(default_factory=dict)


def record(group, name, status, detail="", **extra):
    RESULTS.append(Check(group, name, status, detail, extra))
    print(f"  [{status:4}] {group}.{name}: {detail}", flush=True)


def run(group, name, fn):
    """Run a check fn() -> (ok: bool, detail: str); a raised exception is a BUG."""
    try:
        ok, detail = fn()
        record(group, name, "PASS" if ok else "FAIL", detail)
    except Exception as e:  # noqa: BLE001 - a crash IS the bug we are hunting
        tb = traceback.format_exc().splitlines()
        record(group, name, "BUG", f"{type(e).__name__}: {str(e)[:120]}", traceback=tb[-4:])


# -- model + data zoo (heterogeneous coverage) ------------------------------------------------------


def gmm(k=4, n=3000, seed=0):
    rng = np.random.RandomState(seed)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(k)]
    true = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))
    data = true.sampler(1).sample(n)
    est = st.MixtureEstimator([st.GaussianEstimator() for _ in range(k)])
    init = st.MixtureDistribution([st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(k)], [1.0 / k] * k)
    return data, est, init


def categorical_mix(k=3, n=2000, seed=1):
    rng = np.random.RandomState(seed)
    levels = list("abcde")
    comps = [st.CategoricalDistribution(dict(zip(levels, rng.dirichlet(np.ones(len(levels)))))) for _ in range(k)]
    true = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))
    data = true.sampler(1).sample(n)
    est = st.MixtureEstimator([st.CategoricalEstimator() for _ in range(k)])
    init = st.MixtureDistribution(
        [st.CategoricalDistribution(dict(zip(levels, [1 / len(levels)] * len(levels)))) for _ in range(k)],
        [1.0 / k] * k,
    )
    return data, est, init


def composite(n=2000, seed=2):
    rng = np.random.RandomState(seed)
    true = st.CompositeDistribution(
        (
            st.GaussianDistribution(3.0, 1.0),
            st.PoissonDistribution(4.0),
            st.CategoricalDistribution({"x": 0.7, "y": 0.3}),
        )
    )
    data = true.sampler(1).sample(n)
    est = st.CompositeEstimator((st.GaussianEstimator(), st.PoissonEstimator(), st.CategoricalEstimator()))
    return data, est, None


def hmm(states=3, n=400, length=20, seed=3):
    rng = np.random.RandomState(seed)
    comps = [st.GaussianDistribution(float(5 * rng.randn()), 1.0) for _ in range(states)]
    trans = rng.dirichlet(np.ones(states), size=states)
    len_dist = st.PoissonDistribution(float(length))
    true = st.HiddenMarkovModelDistribution(
        comps, list(rng.dirichlet(np.ones(states))), trans.tolist(), len_dist=len_dist
    )
    data = true.sampler(1).sample(n)
    est = st.HiddenMarkovEstimator([st.GaussianEstimator() for _ in range(states)], len_estimator=st.PoissonEstimator())
    init = st.HiddenMarkovModelDistribution(
        [st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(states)],
        [1.0 / states] * states,
        (np.ones((states, states)) / states).tolist(),
        len_dist=st.PoissonDistribution(float(length)),
    )
    return data, est, init


ZOO = {"gmm4": gmm, "catmix3": categorical_mix, "composite": composite, "hmm3": hmm}


def _params(model):
    """A flat parameter vector for comparing two fitted models of the same family."""
    parts = []
    for attr in ("w", "taus", "p", "mu", "sigma2", "lam", "pmap"):
        v = getattr(model, attr, None)
        if v is None:
            continue
        if isinstance(v, dict):
            parts.extend(float(x) for x in v.values())
        elif np.ndim(v) == 0:
            parts.append(float(v))
        else:
            parts.extend(np.asarray(v, dtype=float).ravel().tolist())
    # recurse into components / dists
    for attr in ("components", "dists"):
        for c in getattr(model, attr, []) or []:
            parts.extend(_params(c))
    return np.asarray(parts, dtype=float)


def _loglik(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


# -- A. data-parallel backends vs serial baseline ---------------------------------------------------


def group_A_data_parallel(gpu, heavy):
    families = ["gmm4", "catmix3", "composite", "hmm3"] if heavy else ["gmm4", "composite"]
    for fam in families:
        data, est, init = ZOO[fam]()
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
        # local chunked (num_chunks / sub_chunks) must be bit-identical to plain local
        for nc in [1, 2, 4]:

            def chk(nc=nc, data=data, est=est, init=init, base=base):
                m = optimize(data, est, prev_estimate=init, max_its=8, out=None, num_chunks=nc)
                d = float(np.abs(_params(m) - _params(base)).max()) if _params(base).size else 0.0
                return d < 1e-9, f"num_chunks={nc} max|Δ|={d:.2e}"

            run("A_local_chunks", f"{fam}_nc{nc}", chk)

        # model_parallel bit-identical
        def mp(data=data, est=est, init=init, base=base):
            m = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="model_parallel")
            d = float(np.abs(_params(m) - _params(base)).max()) if _params(base).size else 0.0
            return d < 1e-9, f"max|Δ|={d:.2e} ll_base={_loglik(base, data):.3f} ll_mp={_loglik(m, data):.3f}"

        run("A_model_parallel", fam, mp)


# -- B. MPI data-parallel (spawned via mpirun) ------------------------------------------------------

MPI_RUNNER = """
import json, numpy as np
from mpi4py import MPI
import mixle.stats as st
from mixle.inference.estimation import optimize
from mixle.utils.parallel.mpi import MPIEncodedData, mpi_out
comm = MPI.COMM_WORLD
rng = np.random.RandomState(0)
comps=[st.GaussianDistribution(float(6*rng.randn()), float(0.5+rng.rand())) for _ in range(4)]
m=st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(4))))
data=m.sampler(1).sample(4000)
est = m.estimator()
enc = MPIEncodedData(data, estimator=est)
fit=optimize(None, est, enc_data=enc, prev_estimate=m, max_its=10, delta=None, out=mpi_out())
if comm.Get_rank()==0:
    print("MPIRESULT", json.dumps({"w": list(map(float, fit.w)), "size": comm.Get_size()}))
"""


def group_B_mpi(gpu, heavy, mpi_np):
    import shutil
    import tempfile

    if not shutil.which("mpirun"):
        record("B_mpi", "available", "SKIP", "no mpirun on PATH")
        return
    # serial baseline for the same fixed model/data
    rng = np.random.RandomState(0)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(4)]
    m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(4))))
    data = m.sampler(1).sample(4000)
    serial = optimize(data, m.estimator(), prev_estimate=m, max_its=10, out=None, backend="local")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(MPI_RUNNER)
        path = f.name
    for w in mpi_np:

        def chk(w=w, path=path, serial=serial):
            out = subprocess.run(
                ["mpirun", "--oversubscribe", "-n", str(w), sys.executable, path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            line = [x for x in out.stdout.splitlines() if x.startswith("MPIRESULT")]
            if not line:
                return False, f"n={w} no result (stderr: {out.stderr.strip()[-160:]})"
            res = json.loads(line[0].split(" ", 1)[1])
            d = float(np.abs(np.array(res["w"]) - np.array(serial.w)).max())
            return d < 1e-6, f"n={w} max|Δw vs serial|={d:.2e}"

        run("B_mpi", f"np{w}", chk)


# -- C. torch engine parity + multi-GPU -------------------------------------------------------------


def group_C_torch(gpu, heavy):
    from mixle.engines import TorchEngine

    data, est, init = gmm(4, 3000)
    base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")

    def cpu_parity():
        m = optimize(data, est, prev_estimate=init, max_its=8, out=None, engine=TorchEngine(device="cpu"))
        d = float(np.abs(_params(m) - _params(base)).max())
        return d < 1e-4, f"torch-cpu vs numpy max|Δ|={d:.2e}"

    run("C_torch", "cpu_parity", cpu_parity)

    if gpu:
        import torch

        ngpu = torch.cuda.device_count()
        record("C_torch", "device_count", "PASS" if ngpu else "FAIL", f"{ngpu} CUDA devices")

        def gpu_parity():
            m = optimize(data, est, prev_estimate=init, max_its=8, out=None, engine=TorchEngine(device="cuda:0"))
            d = float(np.abs(_params(m) - _params(base)).max())
            return d < 1e-3, f"cuda:0 vs numpy max|Δ|={d:.2e}"

        run("C_torch", "cuda0_parity", gpu_parity)

        # per-GPU placement: fit a separate model on each device
        def per_gpu():
            accs = []
            for i in range(ngpu):
                m = optimize(data, est, prev_estimate=init, max_its=5, out=None, engine=TorchEngine(device=f"cuda:{i}"))
                accs.append(_loglik(m, data))
            spread = float(np.ptp(accs)) if len(accs) > 1 else 0.0
            return spread < 1.0, f"{ngpu} devices, ll spread={spread:.4f}"

        run("C_torch", "per_gpu_placement", per_gpu)

        # DTensor component sharding across a device mesh (the multi-GPU model-parallel path)
        if ngpu >= 2:

            def mesh_shard():
                import os

                import torch.distributed as dist
                from torch.distributed.device_mesh import init_device_mesh

                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29501")
                if not dist.is_initialized():
                    dist.init_process_group(backend="gloo", rank=0, world_size=1)
                mesh = init_device_mesh("cuda", (ngpu,))
                eng = TorchEngine(device="cuda:0", mesh=mesh, shard="components")
                m = optimize(data, est, prev_estimate=init, max_its=6, out=None, engine=eng)
                d = float(np.abs(_params(m) - _params(base)).max())
                return d < 1e-2, f"mesh({ngpu}) shard=components max|Δ|={d:.2e}"

            run("C_torch", "dtensor_component_shard", mesh_shard)


# -- D. model sharding plan + auto parallel ---------------------------------------------------------


def group_D_sharding(gpu, heavy):
    from mixle.utils.parallel import Resources, auto_parallel_estimator, model_sharding_plan
    from mixle.utils.parallel.planner import DeviceSpec

    data, est, init = gmm(8, 3000)
    base = optimize(data, est, prev_estimate=init, max_its=6, out=None, backend="local")
    devs = Resources(devices=tuple(DeviceSpec(name=f"w{i}", kind="cpu", memory_bytes=8 << 30) for i in range(3)))

    def plan_valid():
        plan = model_sharding_plan(init, devs, estimator=est, axis="components")
        return plan is not None, f"plan over 8 components / 3 devices -> {type(plan).__name__}"

    run("D_sharding", "plan_valid", plan_valid)

    def auto_parallel():
        ape, _decomp = auto_parallel_estimator(est, init, devs)
        m = optimize(data, ape, prev_estimate=init, max_its=6, out=None)
        d = float(np.abs(_params(m) - _params(base)).max())
        return d < 1e-6, f"auto_parallel vs local max|Δ|={d:.2e}"

    run("D_sharding", "auto_parallel_bit_identical", auto_parallel)

    # edge: more shards than components
    def more_shards():
        many = Resources(devices=tuple(DeviceSpec(name=f"w{i}", kind="cpu", memory_bytes=1 << 30) for i in range(16)))
        d2, e2, i2 = gmm(3, 1500)
        ape, _decomp = auto_parallel_estimator(e2, i2, many)
        m = optimize(d2, ape, prev_estimate=i2, max_its=5, out=None)
        return m is not None, "16 shards / 3 components did not crash"

    run("D_sharding", "more_shards_than_components", more_shards)


# -- E. determinism ---------------------------------------------------------------------------------


def group_E_determinism(gpu, heavy):
    data, est, init = gmm(4, 2500)
    for backend in ["local", "model_parallel"]:

        def det(backend=backend):
            a = optimize(data, est, prev_estimate=init, max_its=7, out=None, backend=backend)
            b = optimize(data, est, prev_estimate=init, max_its=7, out=None, backend=backend)
            d = float(np.abs(_params(a) - _params(b)).max())
            return d == 0.0, f"{backend} two-run max|Δ|={d:.2e}"

        run("E_determinism", backend, det)


# -- F. edge cases / stress -------------------------------------------------------------------------


def group_F_edge(gpu, heavy):
    # K=1 mixture
    def k1():
        d, e, i = gmm(1, 500)
        m = optimize(d, e, prev_estimate=i, max_its=5, out=None, backend="model_parallel")
        return m is not None, "K=1 mixture fit"

    run("F_edge", "k1_mixture", k1)

    # n_data < num_workers (model_parallel shards components, so also tiny data)
    def tiny_data():
        d, e, i = gmm(4, 3)
        m = optimize(d, e, prev_estimate=i, max_its=3, out=None, backend="model_parallel")
        return m is not None, "n=3 data, 4 components, model_parallel"

    run("F_edge", "tiny_data", tiny_data)

    # single data point
    def single():
        d, e, i = gmm(2, 1)
        m = optimize(d, e, prev_estimate=i, max_its=2, out=None, num_chunks=4)
        return m is not None, "single data point, num_chunks=4"

    run("F_edge", "single_point_overchunk", single)

    # num_chunks > n_data (empty chunks)
    def empty_chunks():
        d, e, i = gmm(3, 5)
        m = optimize(d, e, prev_estimate=i, max_its=3, out=None, num_chunks=16)
        return m is not None, "num_chunks=16 > n_data=5 (empty chunks)"

    run("F_edge", "empty_chunks", empty_chunks)

    if heavy:

        def large_k():
            d, e, i = gmm(32, 8000)
            m = optimize(d, e, prev_estimate=i, max_its=4, out=None, backend="model_parallel")
            return m is not None, "K=32, n=8000 model_parallel"

        run("F_edge", "large_k_stress", large_k)


# -- G. planner / balance ---------------------------------------------------------------------------


def group_G_planner(gpu, heavy):
    from mixle.utils.parallel import plan

    data, est, init = gmm(6, 3000)

    def plan_runs():
        p = plan(data=data, model=init, estimator=est)
        return p is not None, f"plan() -> {type(p).__name__}"

    run("G_planner", "plan_runs", plan_runs)

    def plan_no_data():
        p = plan(model=init, estimator=est, num_chunks=4)
        return p is not None, "plan(num_chunks=4) pre-fit sizing"

    run("G_planner", "plan_no_data", plan_no_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--only", default="A,B,C,D,E,F,G")
    ap.add_argument("--mpi", default="1,2,4")
    args = ap.parse_args()
    groups = set(args.only.split(","))
    mpi_np = [int(x) for x in args.mpi.split(",")]

    print(f"=== mixle distributed stress: groups={sorted(groups)} gpu={args.gpu} heavy={args.heavy} ===", flush=True)
    if "A" in groups:
        group_A_data_parallel(args.gpu, args.heavy)
    if "B" in groups:
        group_B_mpi(args.gpu, args.heavy, mpi_np)
    if "C" in groups:
        group_C_torch(args.gpu, args.heavy)
    if "D" in groups:
        group_D_sharding(args.gpu, args.heavy)
    if "E" in groups:
        group_E_determinism(args.gpu, args.heavy)
    if "F" in groups:
        group_F_edge(args.gpu, args.heavy)
    if "G" in groups:
        group_G_planner(args.gpu, args.heavy)

    summary = {"total": len(RESULTS)}
    for s in ("PASS", "FAIL", "BUG", "SKIP"):
        summary[s] = sum(1 for r in RESULTS if r.status == s)
    bugs = [
        {"where": f"{r.group}.{r.name}", "detail": r.detail, "tb": r.extra.get("traceback")}
        for r in RESULTS
        if r.status in ("BUG", "FAIL")
    ]
    print("\n=== SUMMARY ===", flush=True)
    print("SUMMARY_JSON " + json.dumps({"summary": summary, "bugs": bugs}), flush=True)


if __name__ == "__main__":
    main()
