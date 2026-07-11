"""Scaling studies + profiling for mixle — find the bottlenecks worth speeding up.

Emits structured JSON (SCALE_JSON ...) plus readable tables. Sections (select with --only):
  D  data scaling      fit time vs N (log-log slope = empirical complexity), per family
  K  model scaling     fit time vs K (mixture components) and vs dim (MVN)
  W  worker scaling     local vs model_parallel(W) vs mpi(W) — strong-scaling speedup + crossover
  E  engine scaling     numpy vs torch-cpu vs torch-cuda/mps — time vs N and vs dim (where GPU wins)
  P  profiling          cProfile a heavy fit per family -> top functions by cumulative time (the hotspots)

Usage: python scaling_profile.py [--gpu] [--only D,K,W,E,P] [--reps 3] [--mpi 1,2,4,8]
"""

import argparse
import cProfile
import json
import pstats
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import mixle.stats as st
from mixle.inference import optimize

OUT = {"data": {}, "model": {}, "worker": {}, "engine": {}, "profile": {}}


def timed(fn, reps=3):
    """Median wall-clock of ``fn`` over ``reps`` runs after one warm-up."""
    fn()  # warm-up (numba/torch compile, allocations)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# -- model builders (fixed init for determinism + apples-to-apples timing) --------------------------


def gmm(n, k=8, seed=0):
    rng = np.random.RandomState(seed)
    comps = [st.GaussianDistribution(float(8 * rng.randn()), float(0.5 + rng.rand())) for _ in range(k)]
    data = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k)))).sampler(1).sample(n)
    est = st.MixtureEstimator([st.GaussianEstimator() for _ in range(k)])
    init = st.MixtureDistribution([st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(k)], [1.0 / k] * k)
    return data, est, init


def mvn_mix(n, k=4, dim=16, seed=1):
    rng = np.random.RandomState(seed)
    comps = [st.MultivariateGaussianDistribution(rng.randn(dim) * 4, np.eye(dim)) for _ in range(k)]
    data = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k)))).sampler(1).sample(n)
    est = st.MixtureEstimator([st.MultivariateGaussianEstimator(dim=dim) for _ in range(k)])
    init = st.MixtureDistribution(
        [st.MultivariateGaussianDistribution(rng.randn(dim), np.eye(dim)) for _ in range(k)], [1.0 / k] * k
    )
    return data, est, init


def hmm(n, states=6, length=30, seed=2):
    rng = np.random.RandomState(seed)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), 1.0) for _ in range(states)]
    trans = rng.dirichlet(np.ones(states), size=states)
    ld = st.PoissonDistribution(float(length))
    data = (
        st.HiddenMarkovModelDistribution(comps, list(rng.dirichlet(np.ones(states))), trans.tolist(), len_dist=ld)
        .sampler(1)
        .sample(n)
    )
    est = st.HiddenMarkovEstimator([st.GaussianEstimator() for _ in range(states)], len_estimator=st.PoissonEstimator())
    init = st.HiddenMarkovModelDistribution(
        [st.GaussianDistribution(float(rng.randn()), 1.0) for _ in range(states)],
        [1.0 / states] * states,
        (np.ones((states, states)) / states).tolist(),
        len_dist=st.PoissonDistribution(float(length)),
    )
    return data, est, init


FAMILIES = {"gmm": gmm, "mvn_mix": mvn_mix, "hmm": hmm}


def _slope(xs, ys):
    """Log-log least-squares slope = empirical scaling exponent."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


# -- D. data scaling --------------------------------------------------------------------------------


def sec_D(args):
    print("\n== D. data scaling: fit time vs N (slope ~ empirical complexity) ==", flush=True)
    Ns = [2000, 8000, 32000, 128000] if args.heavy else [1000, 4000, 16000]
    for fam, build in FAMILIES.items():
        ts = []
        for n in Ns:
            data, est, init = build(n)
            t = timed(lambda d=data, e=est, i=init: optimize(d, e, prev_estimate=i, max_its=8, out=None), args.reps)
            ts.append(t)
            print(f"  {fam:8} N={n:>7} : {t * 1000:8.1f} ms  ({n / t / 1e6:.2f}M rows/s)", flush=True)
        OUT["data"][fam] = {"N": Ns, "sec": ts, "slope": _slope(Ns, ts)}
        print(f"  {fam:8} -> log-log slope {OUT['data'][fam]['slope']:.2f} (1.0 = linear in N)", flush=True)


# -- K. model scaling -------------------------------------------------------------------------------


def sec_K(args):
    print("\n== K. model scaling ==", flush=True)
    Ks = [2, 8, 32, 128] if args.heavy else [2, 8, 32]
    ts = []
    for k in Ks:
        data, est, init = gmm(16000, k=k)
        t = timed(lambda d=data, e=est, i=init: optimize(d, e, prev_estimate=i, max_its=6, out=None), args.reps)
        ts.append(t)
        print(f"  gmm K={k:>4} (N=16000): {t * 1000:8.1f} ms", flush=True)
    OUT["model"]["gmm_K"] = {"K": Ks, "sec": ts, "slope": _slope(Ks, ts)}
    print(f"  gmm vs K -> slope {OUT['model']['gmm_K']['slope']:.2f}", flush=True)

    Ds = [4, 16, 64, 256] if args.heavy else [4, 16, 64]
    ts = []
    for d in Ds:
        data, est, init = mvn_mix(8000, k=4, dim=d)
        t = timed(lambda dd=data, e=est, i=init: optimize(dd, e, prev_estimate=i, max_its=5, out=None), args.reps)
        ts.append(t)
        print(f"  mvn dim={d:>4} (N=8000,K=4): {t * 1000:8.1f} ms", flush=True)
    OUT["model"]["mvn_dim"] = {"dim": Ds, "sec": ts, "slope": _slope(Ds, ts)}
    print(
        f"  mvn vs dim -> slope {OUT['model']['mvn_dim']['slope']:.2f} (2.0 = quadratic = covariance-bound)", flush=True
    )


# -- W. worker scaling ------------------------------------------------------------------------------

MPI_RUNNER = """
import json, time, numpy as np
from mpi4py import MPI
import mixle.stats as st
from mixle.inference.mpi_executor import mpi_fit
comm = MPI.COMM_WORLD
rng = np.random.RandomState(0)
comps=[st.GaussianDistribution(float(8*rng.randn()), float(0.5+rng.rand())) for _ in range(16)]
m=st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(16))))
data=m.sampler(1).sample(200000)
comm.Barrier(); t0=time.perf_counter()
fit=mpi_fit(comm, m, data, max_its=8)
dt=time.perf_counter()-t0
if comm.Get_rank()==0: print("MPITIME", json.dumps({"w": comm.Get_size(), "sec": dt}))
"""


def sec_W(args):
    print("\n== W. worker scaling: strong scaling (fixed problem, more workers) ==", flush=True)
    data, est, init = gmm(200000, k=16)
    t_local = timed(
        lambda: optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local"), max(1, args.reps - 1)
    )
    print(f"  local (1 worker)          : {t_local * 1000:8.0f} ms  [baseline]", flush=True)
    t_mp = timed(
        lambda: optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="model_parallel"),
        max(1, args.reps - 1),
    )
    print(f"  model_parallel            : {t_mp * 1000:8.0f} ms  speedup {t_local / t_mp:.2f}x", flush=True)
    OUT["worker"]["local"] = t_local
    OUT["worker"]["model_parallel"] = t_mp
    # MPI strong scaling
    if shutil.which("mpirun"):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(MPI_RUNNER)
            path = f.name
        OUT["worker"]["mpi"] = {}
        for w in args.mpi:
            out = subprocess.run(
                ["mpirun", "--oversubscribe", "-n", str(w), sys.executable, path],
                capture_output=True,
                text=True,
                timeout=600,
            )
            line = [x for x in out.stdout.splitlines() if x.startswith("MPITIME")]
            if line:
                r = json.loads(line[0].split(" ", 1)[1])
                OUT["worker"]["mpi"][str(w)] = r["sec"]
                print(
                    f"  mpi n={w:<2}                  : {r['sec'] * 1000:8.0f} ms  speedup {OUT['worker']['mpi'].get('1', r['sec']) / r['sec']:.2f}x",
                    flush=True,
                )
    else:
        print("  (no mpirun -> skip MPI)", flush=True)


# -- E. engine scaling ------------------------------------------------------------------------------


def sec_E(args):
    from mixle.engines import TorchEngine

    print("\n== E. engine scaling: numpy vs torch-cpu vs GPU (where does the accelerator win?) ==", flush=True)
    engines = [("numpy", lambda: None), ("torch-cpu", lambda: TorchEngine(device="cpu"))]
    if args.gpu:
        engines.append(("torch-cuda", lambda: TorchEngine(device="cuda:0")))
    elif __import__("torch").backends.mps.is_available():
        engines.append(("torch-mps", lambda: TorchEngine(device="mps")))
    # low-dim (overhead-bound) vs high-dim (compute-bound) MVN mixture
    for dim in [16, 128] if args.heavy else [16, 64]:
        row = {}
        for name, mk in engines:
            data, est, init = mvn_mix(20000, k=8, dim=dim)
            eng = mk()
            try:
                t = timed(
                    lambda d=data, e=est, i=init, g=eng: optimize(d, e, prev_estimate=i, max_its=5, out=None, engine=g),
                    max(1, args.reps - 1),
                )
                row[name] = t
                print(f"  MVN dim={dim:<4} {name:11}: {t * 1000:8.0f} ms", flush=True)
            except Exception as ex:  # a CRASH is a finding, not a harness abort
                row[name] = f"CRASH: {type(ex).__name__}"
                print(f"  MVN dim={dim:<4} {name:11}: CRASH {type(ex).__name__}: {str(ex)[:50]}", flush=True)
        OUT["engine"][f"mvn_dim{dim}"] = row


# -- P. profiling -----------------------------------------------------------------------------------


def sec_P(args):
    print("\n== P. profiling: where the time actually goes (top functions by cumtime) ==", flush=True)
    jobs = {
        "gmm": lambda: gmm(120000, k=16),
        "mvn_mix": lambda: mvn_mix(30000, k=8, dim=64),
        "hmm": lambda: hmm(4000, states=8, length=40),
    }
    for fam, build in jobs.items():
        data, est, init = build()
        optimize(data, est, prev_estimate=init, max_its=3, out=None)  # warm-up numba
        pr = cProfile.Profile()
        pr.enable()
        optimize(data, est, prev_estimate=init, max_its=8, out=None)
        pr.disable()
        ps = pstats.Stats(pr)
        # use the stats dict directly (robust vs text parsing): key=(file,line,func), val=(cc,nc,tt,ct,callers)
        rows = []
        for (fn, ln, func), (cc, nc, tt, ct, callers) in ps.stats.items():
            where = f"{fn.split('/')[-1]}:{ln}({func})"
            rows.append({"cum": round(ct, 4), "tot": round(tt, 4), "ncalls": nc, "where": where[-70:]})
        rows.sort(key=lambda r: -r["cum"])
        OUT["profile"][fam] = rows[:14]
        top = rows
        print(f"\n  [{fam}] top by cumulative time:", flush=True)
        for r in top[:10]:
            print(f"    {r['cum']:7.3f}s cum {r['tot']:7.3f}s self  {r['where']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="D,K,W,E,P")
    ap.add_argument("--mpi", default="1,2,4,8")
    args = ap.parse_args()
    args.mpi = [int(x) for x in args.mpi.split(",")]
    secs = set(args.only.split(","))
    import os

    print(
        f"=== mixle scaling+profile: cpus={os.cpu_count()} gpu={args.gpu} heavy={args.heavy} reps={args.reps} ===",
        flush=True,
    )
    if "D" in secs:
        sec_D(args)
    if "K" in secs:
        sec_K(args)
    if "W" in secs:
        sec_W(args)
    if "E" in secs:
        sec_E(args)
    if "P" in secs:
        sec_P(args)
    print("\nSCALE_JSON " + json.dumps(OUT), flush=True)


if __name__ == "__main__":
    main()
