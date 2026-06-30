"""Composable / structured HMMs — a tour of mixle.stats.latent.structured_hmm.

A standard HMM is a dense K x K transition + per-state emissions. mixle factors the transition behind a
small TransitionOperator interface (forward: alpha @ A, backward: A @ v, plus an expected-mass M-step), so
rich structure plugs into ONE forward-backward / EM: low-rank transitions, block / factorial (Kronecker)
combinators, sparse (left-to-right) transitions, sticky / Dirichlet priors. On top sit decoding (Viterbi /
posterior), enumeration (top-k / rank / nucleus over sequences), terminal (absorbing) states, an
input-output HMM, an explicit-duration HMM (HSMM) with segment decoding, forgetting-based parallel
Baum-Welch, and a JAX fast path.

Run: ``python examples/structured_hmm_example.py``
"""

from __future__ import annotations

import numpy as np

import mixle.stats as S
from mixle.inference import optimize
from mixle.stats.latent.structured_hmm import (
    DenseTransition,
    ExplicitDurationHMM,
    InputOutputHMM,
    KroneckerTransition,
    LowRankTransition,
    SparseTransition,
    StructuredHMM,
    _row_normalize,
    left_to_right_edges,
    sticky_transition,
)


def gaussians(centers, sd=1.0):
    return [S.GaussianDistribution(float(c), sd) for c in centers]


def low_rank():
    rng = np.random.RandomState(0)
    k, r = 8, 2
    gen = StructuredHMM(
        gaussians(range(0, 4 * k, 4)),
        np.ones(k) / k,
        LowRankTransition(_row_normalize(rng.rand(k, r)), _row_normalize(rng.rand(r, k))),
    )
    seqs = [gen.sampler(seed=s).sample(50) for s in range(60)]
    init = StructuredHMM(
        gaussians([4 * i + rng.uniform(-1, 1) for i in range(k)]),
        np.ones(k) / k,
        LowRankTransition(_row_normalize(rng.rand(k, r)), _row_normalize(rng.rand(r, k))),
    )
    fit = optimize(seqs, init.estimator(), prev_estimate=init, max_its=40, out=None)
    print(
        f"1. Low-rank HMM (K={k}, rank={r}): transition params {2 * k * r} vs dense {k * k}; "
        f"recovered means {sorted(round(e.mu, 1) for e in fit.emissions)[:4]}..."
    )


def factorial():
    rng = np.random.RandomState(0)
    a1, a2 = _row_normalize(rng.rand(3, 3)), _row_normalize(rng.rand(4, 4))
    kt = KroneckerTransition(DenseTransition(a1), DenseTransition(a2))
    print(
        f"2. Factorial (Kronecker) HMM: state=(s1,s2), {kt.n_states}=3x4 joint states, A=A1(x)A2 "
        f"(forward O(K1 K2 (K1+K2)) not O((K1K2)^2))"
    )


def sticky_and_sparse():
    sp = SparseTransition(5, left_to_right_edges(5, skip=1))
    a = sp.as_matrix()
    st = sticky_transition(np.full((3, 3), 1 / 3), kappa=10.0)
    print(
        f"3. Sparse left-to-right (5 states): lower-triangle zero={np.allclose(np.tril(a, -1), 0)}; "
        f"sticky prior biases self-transitions (segmentation)"
    )


def decode_and_enumerate():
    a = np.array([[0.92, 0.08], [0.08, 0.92]])
    cat = [S.CategoricalDistribution({0: 0.8, 1: 0.2}), S.CategoricalDistribution({0: 0.2, 1: 0.8})]
    hmm = StructuredHMM(cat, [0.5, 0.5], DenseTransition(a), len_dist=S.CategoricalDistribution({2: 0.5, 3: 0.5}))
    en = hmm.enumerator()
    print(
        f"4. Decoding + enumeration: Viterbi works for any operator; top-3 most-probable sequences "
        f"{[seq for seq, _ in en.top_k(3)]}, nucleus(0.9) covers {en.nucleus_size(0.9).covered_mass:.2f}"
    )


def terminal():
    a = _row_normalize(np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0, 0, 1.0]]))
    hmm = StructuredHMM(gaussians([-3, 0, 3]), [0.7, 0.3, 0.0], DenseTransition(a), terminal_states={2})
    lengths = [len(hmm.sampler(seed=s).sample(50)) for s in range(20)]
    print(
        f"5. Terminal (absorbing) states: state 2 stops the sequence -> variable lengths "
        f"{min(lengths)}..{max(lengths)} (length is a stopping time)"
    )


def hsmm():
    d = 6
    dur = np.zeros((2, d))
    dur[0, 3] = 1.0
    dur[1, 1] = 1.0
    gen = ExplicitDurationHMM(gaussians([-6, 6], 0.4), [1.0, 0.0], np.array([[0, 1.0], [1.0, 0]]), dur, d)
    rng = np.random.RandomState(0)
    true = [0] * 4 + [1] * 2 + [0] * 4 + [1] * 2
    seq = [float(rng.normal([-6, 6][s], 0.4)) for s in true]
    segs = gen.viterbi_segments(seq)
    print(
        f"6. Explicit-duration HMM (HSMM): non-geometric durations; decoded segments "
        f"{[(s, dd) for s, _, dd in segs]} (state, duration)"
    )


def iohmm():
    a0, a1 = np.array([[0.95, 0.05], [0.05, 0.95]]), np.array([[0.05, 0.95], [0.95, 0.05]])
    io = InputOutputHMM(gaussians([-5, 5], 0.5), [0.5, 0.5], [DenseTransition(a0), DenseTransition(a1)])
    print(
        "7. Input-output HMM: an exogenous input selects the transition each step (input 0=sticky, "
        "1=flip) -- a controlled Markov model"
    )


def main():
    print("# Composable / structured HMMs in mixle\n")
    low_rank()
    factorial()
    sticky_and_sparse()
    decode_and_enumerate()
    terminal()
    hsmm()
    iohmm()
    print(
        "\nAll of these share one TransitionOperator interface: the operator's structure makes FITTING "
        "cheaper (O(K r) low-rank, O(edges) sparse, factorial Kronecker), while as_matrix() feeds decoding "
        "and enumeration. Forgetting-parallel Baum-Welch (fit_chunked) and a JAX fast path "
        "(jit_forward_loglik) round out the toolkit."
    )


if __name__ == "__main__":
    main()
