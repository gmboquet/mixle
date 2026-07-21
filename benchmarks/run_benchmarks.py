"""Run the mixle vs scikit-learn / pomegranate / hmmlearn scaling benchmarks.

Emits a human-readable table and ``results/results.json`` (consumed by the write-up /
website charts). Every reported point is correctness-gated: the packages must agree on
the final mean log-likelihood, or the point is flagged.

    python run_benchmarks.py [--quick] [--reps N]

Hardware / versions are captured into the JSON so the numbers are self-describing.
"""

import argparse
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _bench as B  # noqa: E402
from benchmark_provenance import stamp_result  # noqa: E402

LL_TOL = 1e-3  # absolute tolerance on mean log-likelihood agreement across packages


def _versions():
    import importlib.metadata as md

    v = {}
    for p in ("scikit-learn", "pomegranate", "hmmlearn", "mixle", "numpy", "scipy", "torch", "numba"):
        try:
            v[p] = md.version(p)
        except Exception:  # noqa: BLE001 - metadata probing over optional packages; absence is the recorded fact
            v[p] = None
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:  # noqa: BLE001 - sysctl is a macOS nicety; every other platform keeps the generic name
        pass
    return {"packages": v, "cpu": cpu, "platform": platform.platform(), "threads": os.environ.get("OMP_NUM_THREADS")}


def _check_parity(label, results):
    lls = [r["mean_ll"] for r in results.values() if r and r.get("mean_ll") is not None]
    failed = [p for p, r in results.items() if r and r.get("failed")]
    if failed:
        print(f"      note: {', '.join(failed)} failed ({', '.join(results[p]['failed'] for p in failed)})")
    if len(lls) < 2:
        return 0.0, True
    dmax = float(max(lls) - min(lls))
    ok = dmax < LL_TOL
    flag = "" if ok else "  << LL MISMATCH"
    print(f"      parity: max|Δ mean_ll| = {dmax:.2e}{flag}")
    return dmax, ok


def _ms(r):
    return f"{r['sec'] * 1e3:8.0f}ms" if r and r.get("sec") is not None else "   FAILED"


def _ratio(a, b):
    if a and b and a.get("sec") and b.get("sec"):
        return b["sec"] / a["sec"]
    return float("nan")


def panel_gmm_scale_n(reps, quick):
    print("\n== GMM (full covariance) -- data scaling: fit time vs N ==")
    Ns = [4000, 16000, 64000] if quick else [5000, 20000, 80000, 200000]
    dim, k, its = 32, 16, 15
    print(f"   dim={dim} K={k} full-covariance, {its} EM iters from shared init, reps={reps}")
    points = []
    for n in Ns:
        X, init = B.make_full_cov_gmm(n, dim, k)
        res = {
            "sklearn": B.timed(B.gmm_sklearn(X, init, its), reps),
            "pomegranate": B.timed(B.gmm_pomegranate(X, init, its), reps),
            "mixle": B.timed(B.gmm_mixle(X, init, its), reps),
        }
        dmax, ok = _check_parity(f"N={n}", res)
        print(
            f"   N={n:>7}: sklearn {_ms(res['sklearn'])}  pomegranate {_ms(res['pomegranate'])}  "
            f"mixle {_ms(res['mixle'])}   speedup vs sklearn={_ratio(res['mixle'], res['sklearn']):.2f}x"
        )
        points.append(
            {
                "n": n,
                "dim": dim,
                "k": k,
                "its": its,
                "ll_delta": dmax,
                "ll_ok": ok,
                "times": {p: r.get("sec") for p, r in res.items()},
                "raw": res,
            }
        )
    return {"axis": "n", "x": Ns, "fixed": {"dim": dim, "k": k, "its": its}, "points": points}


def panel_gmm_scale_dim(reps, quick):
    print("\n== GMM (full covariance) -- dimension scaling: fit time vs dim ==")
    Ds = [8, 16, 32, 64] if quick else [8, 16, 32, 64, 128]
    n, k, its = 20000, 8, 12
    print(f"   N={n} K={k} full-covariance, {its} EM iters from shared init, reps={reps}")
    points = []
    for d in Ds:
        X, init = B.make_full_cov_gmm(n, d, k)
        res = {
            "sklearn": B.timed(B.gmm_sklearn(X, init, its), reps),
            "pomegranate": B.timed(B.gmm_pomegranate(X, init, its), reps),
            "mixle": B.timed(B.gmm_mixle(X, init, its), reps),
        }
        dmax, ok = _check_parity(f"dim={d}", res)
        print(
            f"   dim={d:>4}: sklearn {_ms(res['sklearn'])}  pomegranate {_ms(res['pomegranate'])}  "
            f"mixle {_ms(res['mixle'])}   speedup vs sklearn={_ratio(res['mixle'], res['sklearn']):.2f}x"
        )
        points.append(
            {
                "dim": d,
                "n": n,
                "k": k,
                "its": its,
                "ll_delta": dmax,
                "ll_ok": ok,
                "times": {p: r.get("sec") for p, r in res.items()},
                "raw": res,
            }
        )
    return {"axis": "dim", "x": Ds, "fixed": {"n": n, "k": k, "its": its}, "points": points}


def panel_hmm_scale_n(reps, quick):
    print("\n== Gaussian HMM -- data scaling: fit time vs number of sequences ==")
    Ns = [200, 800, 1600] if quick else [250, 1000, 4000]
    L, S, its = 60, 8, 12
    print(f"   length={L} states={S} Gaussian emissions, {its} Baum-Welch iters from shared init, reps={reps}")
    points = []
    for nseq in Ns:
        seqs, xcat, lengths, init = B.make_gaussian_hmm(nseq, L, S)
        res = {
            "hmmlearn": B.timed(B.hmm_hmmlearn(seqs, xcat, lengths, init, its), reps),
            "mixle": B.timed(B.hmm_mixle(seqs, init, L, its), reps),
        }
        dmax, ok = _check_parity(f"nseq={nseq}", res)
        print(
            f"   nseq={nseq:>5} ({nseq * L:>7} obs): hmmlearn {_ms(res['hmmlearn'])}  mixle {_ms(res['mixle'])}   "
            f"speedup={_ratio(res['mixle'], res['hmmlearn']):.1f}x  ll/seq={res['mixle']['mean_ll']:.3f}"
        )
        points.append(
            {
                "nseq": nseq,
                "obs": nseq * L,
                "length": L,
                "states": S,
                "its": its,
                "ll_delta": dmax,
                "ll_ok": ok,
                "times": {p: r.get("sec") for p, r in res.items()},
                "raw": res,
            }
        )
    return {"axis": "nseq", "x": Ns, "fixed": {"length": L, "states": S, "its": its}, "points": points}


def panel_hmm_scale_states(reps, quick):
    print("\n== Gaussian HMM -- model scaling: fit time vs number of states ==")
    Ss = [4, 8, 16] if quick else [4, 8, 16, 32]
    nseq, L, its = 1000, 60, 12
    print(f"   nseq={nseq} length={L} Gaussian emissions, {its} Baum-Welch iters from shared init, reps={reps}")
    points = []
    for S in Ss:
        seqs, xcat, lengths, init = B.make_gaussian_hmm(nseq, L, S)
        res = {
            "hmmlearn": B.timed(B.hmm_hmmlearn(seqs, xcat, lengths, init, its), reps),
            "mixle": B.timed(B.hmm_mixle(seqs, init, L, its), reps),
        }
        dmax, ok = _check_parity(f"S={S}", res)
        print(
            f"   states={S:>3}: hmmlearn {_ms(res['hmmlearn'])}  mixle {_ms(res['mixle'])}   "
            f"speedup={_ratio(res['mixle'], res['hmmlearn']):.1f}x"
        )
        points.append(
            {
                "states": S,
                "nseq": nseq,
                "length": L,
                "its": its,
                "ll_delta": dmax,
                "ll_ok": ok,
                "times": {p: r.get("sec") for p, r in res.items()},
                "raw": res,
            }
        )
    return {"axis": "states", "x": Ss, "fixed": {"nseq": nseq, "length": L, "its": its}, "points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes for a fast smoke run")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--only", default="gmm_n,gmm_dim,hmm_n,hmm_states")
    args = ap.parse_args()
    B.pin_torch()

    meta = _versions()
    print("=== mixle scaling benchmarks vs scikit-learn / pomegranate / hmmlearn ===")
    print(f"    {meta['cpu']}  |  threads={meta['threads']}  |  {meta['platform']}")
    print("    " + "  ".join(f"{k}={v}" for k, v in meta["packages"].items() if v))

    panels = {}
    sel = set(args.only.split(","))
    if "gmm_n" in sel:
        panels["gmm_scale_n"] = panel_gmm_scale_n(args.reps, args.quick)
    if "gmm_dim" in sel:
        panels["gmm_scale_dim"] = panel_gmm_scale_dim(args.reps, args.quick)
    if "hmm_n" in sel:
        panels["hmm_scale_n"] = panel_hmm_scale_n(args.reps, args.quick)
    if "hmm_states" in sel:
        panels["hmm_scale_states"] = panel_hmm_scale_states(args.reps, args.quick)

    out = stamp_result({"meta": meta, "panels": panels, "ll_tol": LL_TOL})
    # --quick is a smoke run: keep it away from results.json, the tracked full-sweep reference
    # artifact that the write-up and B7.3's version-stamp gate consume.
    fname = "results_quick.json" if args.quick else "results.json"
    path = os.path.join(HERE, "results", fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
