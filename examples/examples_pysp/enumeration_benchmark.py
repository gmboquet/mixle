"""Benchmark + evaluation harness for the enumeration / rank / cumulative-probability toolkit.

Run: python examples/examples_pysp/enumeration_benchmark.py

Exercises and times the pieces added for descending-order enumeration, arbitrary-index access, and
rank / cumulative-probability queries across decomposable families (composite/sequence), mixtures,
and HMMs. Each section prints timing and an accuracy cross-check against an exact reference.
"""

import itertools
import math
import time

import numpy as np

import pysp.utils.quantization.core as qcore
from pysp.stats.composite import CompositeDistribution
from pysp.stats.hidden_markov import HiddenMarkovModelDistribution
from pysp.stats.int_range import IntegerCategoricalDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.poisson import PoissonDistribution
from pysp.stats.sequence import SequenceDistribution
from pysp.utils.density_rank import count_dp_rank, density_rank
from pysp.utils.enumeration import freeze, rerank_by_density, stable_top_k


def _time(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def bench_kronecker_convolution():
    print("\n[1] count-DP convolution: Kronecker substitution vs naive double loop")
    import random

    rng = random.Random(0)
    for width, bits in [(2000, 40), (6000, 64)]:
        a = qcore.CountHistogram(0, [rng.randrange(0, 1 << bits) for _ in range(width)])
        b = qcore.CountHistogram(0, [rng.randrange(0, 1 << bits) for _ in range(width)])
        _, tk = _time(lambda: a.convolve(b))
        saved = qcore._KRONECKER_MIN_PRODUCT
        qcore._KRONECKER_MIN_PRODUCT = 10**18
        try:
            r_naive, tn = _time(lambda: a.convolve(b))
        finally:
            qcore._KRONECKER_MIN_PRODUCT = saved
        exact = a.convolve(b).data == r_naive.data
        print(f"    width={width:5d} bits={bits:3d}: naive={tn * 1e3:8.1f}ms kron={tk * 1e3:7.1f}ms "
              f"speedup={tn / tk:5.1f}x exact={exact}")


def bench_hmm_index_build():
    print("\n[2] HMM count-budget index build (Kronecker accelerates the forward DP)")
    rng = np.random.RandomState(0)
    topics = [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(20) * 0.4))) for _ in range(3)]
    hmm = HiddenMarkovModelDistribution(
        topics, list(rng.dirichlet(np.ones(3))), rng.dirichlet(np.ones(3), size=3).tolist(),
        len_dist=PoissonDistribution(5.0),
    )
    for budget in (36, 48):
        idx, tk = _time(lambda b=budget: qcore.count_budget_index(hmm, budget_bits=b, oversample=8))
        saved = qcore._KRONECKER_MIN_PRODUCT
        qcore._KRONECKER_MIN_PRODUCT = 10**18
        try:
            _, tn = _time(lambda b=budget: qcore.count_budget_index(hmm, budget_bits=b, oversample=8))
        finally:
            qcore._KRONECKER_MIN_PRODUCT = saved
        print(f"    budget=2^{budget}: naive={tn * 1e3:7.1f}ms kron={tk * 1e3:6.1f}ms "
              f"speedup={tn / tk:4.1f}x  pairs counted={idx.total_count:,}")


def _tiers(pairs):
    out = {}
    for v, lp in pairs:
        out.setdefault(round(lp, 8), set()).add(freeze(v))
    return out


def bench_true_order_recovery():
    print("\n[3] true-marginal order from the (approximate) tropical seek index")
    rng = np.random.RandomState(0)
    topics = [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(20) * 0.4))) for _ in range(3)]
    hmm = HiddenMarkovModelDistribution(
        topics, list(rng.dirichlet(np.ones(3))), rng.dirichlet(np.ones(3), size=3).tolist(),
        len_dist=PoissonDistribution(5.0),
    )
    k = 30
    exact = list(itertools.islice(hmm.enumerator(), k))
    true_rank = {freeze(v): r for r, (v, _) in enumerate(exact)}
    raw = list(itertools.islice(hmm.count_budget_distinct(budget_bits=36), 600))

    def disp(order):
        return float(np.mean([abs(p - true_rank[freeze(v)]) for p, (v, _) in enumerate(order[:k]) if freeze(v) in true_rank]))

    reranked = list(itertools.islice(rerank_by_density(iter(raw), window=200), k))
    stable, ts = _time(lambda: stable_top_k(lambda: hmm.count_budget_distinct(budget_bits=36), k))
    print(f"    raw tropical order:   mean|rank displacement|={disp(raw):.1f}")
    print(f"    rerank (window=200):  mean|rank displacement|={disp(reranked):.1f}")
    print(f"    stable_top_k (auto):  matches exact top-{k}={_tiers(stable) == _tiers(exact)}  time={ts * 1e3:.0f}ms")


def bench_rank_queries():
    print("\n[4] rank + cumulative probability of an arbitrary observation")
    rng = np.random.RandomState(0)
    # decomposable: exact rank at any depth via count_dp_rank
    comp = CompositeDistribution(tuple(IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(s)))) for s in (5, 4, 6)))
    support = [list(t) for t in itertools.product(range(5), range(4), range(6))]
    lp = {tuple(x): comp.log_density(x) for x in support}
    brute = {tuple(x): sum(1 for y in support if lp[tuple(y)] > lp[tuple(x)] + 1e-12) for x in support}
    errs = [abs(count_dp_rank(comp, x, oversample=32).rank - brute[tuple(x)]) for x in support]
    print(f"    composite count_dp_rank vs brute (|support|={len(support)}): max|err|={max(errs)} (exact)")

    seq = SequenceDistribution(IntegerCategoricalDistribution(0, [0.4, 0.3, 0.2, 0.1]), len_dist=PoissonDistribution(8.0))
    r, tr = _time(lambda: count_dp_rank(seq, [3] * 12, oversample=16))
    print(f"    deep sequence [3]*12 rank={r.rank:.3e} (enumeration infeasible) time={tr * 1e3:.0f}ms")

    # mixture / HMM: hybrid density_rank (exact head + sampling tail)
    mix = MixtureDistribution([IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(12)))) for _ in range(4)],
                              list(rng.dirichlet(np.ones(4))))
    mlp = {y: mix.log_density(y) for y in range(12)}
    head = density_rank(mix, max(range(12), key=mlp.get))
    exact_G = sum(math.exp(mlp[y]) for y in range(12) if mlp[y] >= mlp[max(range(12), key=mlp.get)] - 1e-12)
    print(f"    mixture mode: density_rank G={head.cumulative_probability:.4f} exact={exact_G:.4f} "
          f"rank={head.rank} method={head.method}")


def main():
    print("=" * 78)
    print("Enumeration / rank / cumulative-probability benchmark")
    print("=" * 78)
    bench_kronecker_convolution()
    bench_hmm_index_build()
    bench_true_order_recovery()
    bench_rank_queries()


if __name__ == "__main__":
    main()
