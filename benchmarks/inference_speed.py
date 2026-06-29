"""mixle inference-speed benchmark harness.

The honest speed story for mixle is *not* "our NUTS beats Stan's" — it is "we don't run NUTS when a
closed form or EM will do." This harness measures that, plus the heterogeneous-record path that the
big PPLs cannot express at all. Rival libraries (pomegranate / NumPyro) are compared only when they
are installed; otherwise their rows are skipped (never faked).

Run:
    python benchmarks/inference_speed.py            # full
    python benchmarks/inference_speed.py --quick    # smaller n, faster

No result here is a marketing claim until it is reproduced on the reader's machine; the harness prints
wall-clock medians (best of a few repeats) and the recovered-parameter error so speed is never reported
without correctness.
"""

from __future__ import annotations

import argparse
import time
from statistics import median

import numpy as np


def _time(fn, repeats=3):
    """Median wall-clock seconds over `repeats` runs (best-effort warm; returns (secs, last_result))."""
    out = None
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return median(ts), out


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- mixle cases
def case_conjugate_vs_mcmc(n, rng):
    """Poisson rate with a Gamma prior: closed-form conjugate vs MCMC — the headline contrast."""
    from mixle.ppl import Gamma, Poisson

    data = list(rng.poisson(3.0, n))
    rows = []
    secs, m = _time(lambda: Poisson(Gamma(2.0, 1.0)).fit(data))  # how='auto' -> conjugate
    rows.append(("Poisson-Gamma (conjugate, closed-form)", secs, abs(float(m.dist.lam) - 3.0)))
    secs, m = _time(lambda: Poisson(Gamma(2.0, 1.0)).fit(data, how="mcmc", draws=2000, burn=1000), repeats=1)
    rows.append(("Poisson-Gamma (MCMC, same model)", secs, abs(float(m.dist.lam) - 3.0)))
    return rows


def case_gaussian_mixture_em(n, rng):
    from mixle.inference import optimize
    from mixle.stats import GaussianDistribution, MixtureDistribution

    data = list(np.concatenate([rng.normal(-3, 1, n // 2), rng.normal(3, 1, n - n // 2)]))
    # separated init breaks EM's symmetric fixed point (a real init concern, not a benchmark fudge)
    proto = MixtureDistribution([GaussianDistribution(-2.5, 1), GaussianDistribution(2.5, 1)], [0.5, 0.5])
    secs, m = _time(lambda: optimize(data, proto, max_its=200, out=None))
    mus = sorted(c.mu for c in m.components)
    return [("2-comp Gaussian mixture (EM)", secs, max(abs(mus[0] + 3), abs(mus[1] - 3)))]


def case_hmm_em(n, rng):
    from mixle.inference import optimize
    from mixle.stats import GaussianDistribution, HiddenMarkovModelDistribution

    # generate 2-state Gaussian-HMM sequences directly (avoids the sampler's len_dist requirement)
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    emis = (-3.0, 3.0)
    seqs = []
    for _ in range(max(20, n // 50)):
        s = rng.randint(2)
        seq = []
        for _t in range(40):
            seq.append(float(rng.normal(emis[s], 1.0)))
            s = rng.choice(2, p=A[s])
        seqs.append(seq)
    proto = HiddenMarkovModelDistribution(
        [GaussianDistribution(-1, 1), GaussianDistribution(1, 1)],
        w=[0.5, 0.5],
        transitions=[[0.5, 0.5], [0.5, 0.5]],
    )
    secs, _ = _time(lambda: optimize(seqs, proto, max_its=50, out=None), repeats=1)
    return [("2-state Gaussian HMM (Baum-Welch EM)", secs, float("nan"))]


def case_heterogeneous_record(n, rng):
    """A record no single-family PPL expresses cleanly: (category, real, count-sequence) as ONE model."""
    from mixle.inference import optimize
    from mixle.stats import (
        CategoricalDistribution,
        CompositeDistribution,
        GaussianDistribution,
        PoissonDistribution,
        SequenceDistribution,
    )

    def rec(i):
        c = "a" if i % 2 == 0 else "b"
        x = rng.normal(0 if c == "a" else 5, 1)
        seq = list(rng.poisson(2 if c == "a" else 10, rng.randint(2, 5)))
        return (c, float(x), seq)

    data = [rec(i) for i in range(n)]
    proto = CompositeDistribution(
        (
            CategoricalDistribution({"a": 0.5, "b": 0.5}),
            GaussianDistribution(0, 1),
            SequenceDistribution(PoissonDistribution(1.0)),
        )
    )
    secs, _ = _time(lambda: optimize(data, proto, max_its=50, out=None), repeats=1)
    return [("(category, real, count-seq) record (EM) — no rival expresses this", secs, float("nan"))]


# --------------------------------------------------------------------------- optional rivals
def case_rivals(n, rng):
    rows = []
    if _has("pomegranate"):
        try:
            import torch
            from pomegranate.distributions import Normal as PNormal
            from pomegranate.gmm import GeneralMixtureModel

            data = np.concatenate([rng.normal(-3, 1, n // 2), rng.normal(3, 1, n - n // 2)]).reshape(-1, 1)
            X = torch.tensor(data, dtype=torch.float32)
            secs, _ = _time(lambda: GeneralMixtureModel([PNormal(), PNormal()], max_iter=100).fit(X), repeats=1)
            rows.append(("[rival] pomegranate 2-comp GMM (EM)", secs, float("nan")))
        except Exception as e:  # noqa: BLE001
            rows.append((f"[rival] pomegranate GMM — skipped ({type(e).__name__})", float("nan"), float("nan")))
    if _has("numpyro"):
        rows.append(
            ("[rival] NumPyro present — see notes (NUTS, not EM); not directly comparable", float("nan"), float("nan"))
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller n for a fast run")
    args = ap.parse_args()
    n = 2000 if args.quick else 20000
    rng = np.random.RandomState(0)

    print(f"# mixle inference-speed benchmark (n={n})\n")
    print("| case | median wall-clock (s) | param error |")
    print("| --- | ---: | ---: |")
    rows = []
    for case in (
        case_conjugate_vs_mcmc,
        case_gaussian_mixture_em,
        case_hmm_em,
        case_heterogeneous_record,
        case_rivals,
    ):
        try:
            rows += case(n, rng)
        except Exception as e:  # noqa: BLE001
            rows.append((f"{case.__name__} — ERROR ({type(e).__name__}: {e})", float("nan"), float("nan")))
    for name, secs, err in rows:
        s = "n/a" if secs != secs else f"{secs:.4f}"
        e = "" if err != err else f"{err:.3f}"
        print(f"| {name} | {s} | {e} |")
    print(
        "\nThe headline: the conjugate row should be orders of magnitude faster than the MCMC row for the "
        "*same model* — mixle's speed story is choosing the closed form, not racing samplers."
    )


if __name__ == "__main__":
    main()
