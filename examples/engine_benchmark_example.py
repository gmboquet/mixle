"""Honest engine benchmarks: the same fit on numpy vs torch (CPU / GPU), with wall-clock numbers.

Times three representative workloads across engines and prints a table:

  * Gaussian-mixture EM (the bread-and-butter engine-resident path),
  * default HMM EM (the numba encoding scored + E-stepped on the engine),
  * batch scoring for a spread of families (Gaussian / GPD / wrapped Cauchy / Watson).

The point is honesty, not marketing: on small data the numpy/numba host path usually WINS (kernel-launch
and transfer overhead dominate), and the GPU only pays past a size threshold — the table shows where.
Sizes scale with --scale so you can find the crossover on your hardware. Apple-silicon (MPS) runs
float32 (no float64 on MPS), so its numbers trade precision for speed by construction.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    GeneralizedParetoDistribution,
    HiddenMarkovEstimator,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
    MixtureEstimator,
    WatsonDistribution,
    WrappedCauchyDistribution,
)
from mixle.stats.compute.backend import backend_seq_log_density


def _engines():
    from mixle.engines import NUMPY_ENGINE

    out = [("numpy", NUMPY_ENGINE)]
    try:
        import torch

        from mixle.engines import TorchEngine

        out.append(("torch-cpu", TorchEngine(device="cpu", dtype="float64")))
        if torch.backends.mps.is_available():
            out.append(("torch-mps", TorchEngine(device="mps")))
        if torch.cuda.is_available():
            out.append(("torch-cuda", TorchEngine(device="cuda")))
    except ImportError:
        pass
    return out


def _time(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_gmm(n: int, engines) -> dict[str, float]:
    rng = np.random.RandomState(0)
    data = np.concatenate([rng.normal(-4, 1, n // 2), rng.normal(4, 1, n // 2)]).tolist()
    init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
    est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
    return {
        nm: _time(lambda e=e: optimize(data, est, max_its=10, engine=e, prev_estimate=init, out=None))
        for nm, e in engines
    }


def bench_hmm(n_seq: int, seq_len: int, engines) -> dict[str, float]:
    rng = np.random.RandomState(0)
    seqs = [[float(rng.normal(-4 if rng.rand() < 0.5 else 4, 1.0)) for _ in range(seq_len)] for _ in range(n_seq)]
    init = HiddenMarkovModelDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]]
    )
    est = HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])  # default (numba encoding)
    return {
        nm: _time(lambda e=e: optimize(seqs, est, max_its=5, engine=e, prev_estimate=init, out=None))
        for nm, e in engines
    }


def bench_scoring(n: int, engines) -> dict[str, dict[str, float]]:
    rng = np.random.RandomState(0)
    v = rng.randn(n, 3)
    cases = {
        "gaussian": (GaussianDistribution(0.0, 1.0), rng.randn(n)),
        "gpd": (GeneralizedParetoDistribution(2.0, 0.3), rng.gamma(2.0, 2.0, n)),
        "wrapped_cauchy": (WrappedCauchyDistribution(0.8, 0.6), rng.uniform(-np.pi, np.pi, n)),
        "watson": (WatsonDistribution(np.array([0.0, 0.0, 1.0]), 5.0), v / np.linalg.norm(v, axis=1, keepdims=True)),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (dist, raw) in cases.items():
        enc = dist.dist_to_encoder().seq_encode(raw)
        out[name] = {nm: _time(lambda e=e, d=dist, x=enc: backend_seq_log_density(d, x, e)) for nm, e in engines}
    return out


def _table(title: str, rows: dict[str, dict[str, float]]) -> None:
    cols = sorted({c for r in rows.values() for c in r})
    print(f"\n{title}")
    print("  %-16s" % "" + "".join("%12s" % c for c in cols))
    for name, r in rows.items():
        base = r.get("numpy")
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("%9.1f ms" % (1e3 * v) if v is not None else "%12s" % "-")
        speed = (
            ""
            if base is None
            else "   (numpy/x: " + ", ".join(f"{c}={base / r[c]:.2f}" for c in cols if c != "numpy" and c in r) + ")"
        )
        print("  %-16s" % name + "".join(cells) + speed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1, help="multiply workload sizes (find your crossover)")
    a = p.parse_args()
    engines = _engines()
    print("engines:", ", ".join(nm for nm, _ in engines))

    _table(
        "EM fits (10 its GMM / 5 its default-HMM)",
        {
            "gmm n=%d" % (20_000 * a.scale): bench_gmm(20_000 * a.scale, engines),
            "hmm %dx%d" % (400 * a.scale, 20): bench_hmm(400 * a.scale, 20, engines),
        },
    )
    _table("batch scoring (n=%d)" % (200_000 * a.scale), bench_scoring(200_000 * a.scale, engines))
    print(
        "\nhonest reading: numpy/numba wins on small data (launch+transfer overhead); the engine pays off"
        "\nas n grows or when the model itself is torch-resident. MPS is float32 by construction."
    )


if __name__ == "__main__":
    main()
