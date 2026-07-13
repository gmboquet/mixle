"""Structure fuzzing for the fused compiler: random model trees vs the host oracle.

Hand-picked fixtures prove the paths I thought of; this searches the structure space I didn't. A
seeded generator composes mixtures of composites from a factor alphabet spanning every template
kind (scalar, vector, matrix, tabulated, categorical, chain) plus bridged factors (nested mixtures,
length-model chains, untemplated leaves), with random component counts, parameters, and data. Every
sample must satisfy, against the untouched host path:

  P1  fusibility: every generated shape compiles (a fusibility REGRESSION gate);
  P2  score parity at 1e-9;
  P3  M-step parity: estimate(fused stats) and estimate(host stats) score identically at 1e-9;
  P4  the parallel scorer matches and its reruns are bit-identical;
  P5  three manual EM steps through the fused path are monotone (the E-step normalizer never lies).

Compile-cost discipline: kernels cache by STRUCTURE signature (factor-type sequence), so the
generator draws many samples over a bounded signature pool -- parameters, component counts, and data
vary freely without recompiling. The unbounded version for deep local soaks is
``scripts/fuzz_fused_soak.py`` (same generator, CLI seed/sample count).
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    from mixle.stats import (
        CategoricalDistribution,
        CompositeDistribution,
        DiagonalGaussianDistribution,
        ExponentialDistribution,
        GaussianDistribution,
        LaplaceDistribution,
        MarkovChainDistribution,
        MixtureDistribution,
        MultivariateGaussianDistribution,
        PoissonDistribution,
    )
    from mixle.stats.compute import fused_codegen as fc

STATES = ["a", "b", "c"]


def _factor_alphabet():
    """kind name -> (make_dist(rng, jitter), make_datum(rng)) covering every template kind + bridges."""

    def chain_dist(rng, len_dist=None):
        init = rng.dirichlet(np.ones(3))
        trans = rng.dirichlet(np.ones(3), size=3)
        kwargs = {} if len_dist is None else {"len_dist": len_dist}
        return MarkovChainDistribution(
            dict(zip(STATES, init)), {s: dict(zip(STATES, trans[i])) for i, s in enumerate(STATES)}, **kwargs
        )

    def chain_datum(rng):
        return [STATES[rng.randint(3)] for _ in range(int(rng.randint(1, 6)))]

    return {
        "gaussian": (
            lambda rng: GaussianDistribution(float(rng.randn() * 2), 0.5 + float(rng.rand())),
            lambda rng: float(rng.randn() * 2),
        ),
        "exponential": (
            lambda rng: ExponentialDistribution(0.5 + float(rng.rand()) * 2),
            lambda rng: float(rng.exponential(1.0)) + 1e-3,
        ),
        "poisson": (lambda rng: PoissonDistribution(1.0 + float(rng.rand()) * 4), lambda rng: int(rng.poisson(3))),
        "categorical": (
            lambda rng: CategoricalDistribution(dict(zip("xyz", rng.dirichlet(np.ones(3))))),
            lambda rng: "xyz"[rng.randint(3)],
        ),
        "diaggaussian": (
            lambda rng: DiagonalGaussianDistribution(list(rng.randn(2)), [0.5 + float(rng.rand())] * 2),
            lambda rng: [float(v) for v in rng.randn(2)],
        ),
        "mvgaussian": (
            lambda rng: MultivariateGaussianDistribution(list(rng.randn(2)), (np.eye(2) * (0.8 + rng.rand())).tolist()),
            lambda rng: [float(v) for v in rng.randn(2)],
        ),
        "chain": (chain_dist, chain_datum),
        "inner_mixture": (
            lambda rng: MixtureDistribution(
                [GaussianDistribution(float(rng.randn() - 2), 1.0), GaussianDistribution(float(rng.randn() + 2), 1.0)],
                list(rng.dirichlet(np.ones(2))),
            ),
            lambda rng: float(rng.randn() * 2),
        ),
        "len_chain": (lambda rng: chain_dist(rng, len_dist=PoissonDistribution(2.0 + float(rng.rand()))), chain_datum),
        "laplace": (
            lambda rng: LaplaceDistribution(float(rng.randn()), 0.5 + float(rng.rand())),
            lambda rng: float(rng.laplace(0, 1)),
        ),
    }


# The bounded signature pool: each entry is one factor-type sequence = one compiled structure.
SIGNATURE_POOL = [
    ("gaussian",),
    ("gaussian", "categorical"),
    ("gaussian", "poisson", "categorical"),
    ("chain", "gaussian"),
    ("chain", "categorical", "exponential"),
    ("inner_mixture", "gaussian"),
    ("inner_mixture", "chain", "categorical"),
    ("len_chain", "gaussian"),
    ("laplace", "poisson"),
    ("mvgaussian", "gaussian"),
    ("diaggaussian", "categorical"),
    ("chain", "inner_mixture", "laplace", "poisson"),
]


def sample_model_and_data(rng, sig=None, n_rows=None):
    """One random (model, data) over a signature from the pool (or a given one)."""
    alphabet = _factor_alphabet()
    sig = sig if sig is not None else SIGNATURE_POOL[rng.randint(len(SIGNATURE_POOL))]
    K = int(rng.randint(2, 5))
    n = int(n_rows if n_rows is not None else rng.randint(200, 800))
    comps = []
    for _ in range(K):
        factors = tuple(alphabet[name][0](rng) for name in sig)
        comps.append(CompositeDistribution(factors) if len(factors) > 1 else factors[0])
    model = MixtureDistribution(comps, list(rng.dirichlet(np.ones(K) * 4)))
    data = []
    for _ in range(n):
        row = tuple(alphabet[name][1](rng) for name in sig)
        data.append(row if len(sig) > 1 else row[0])
    return model, data, sig


def check_sample(tc, rng, sig=None):
    model, data, sig = sample_model_and_data(rng, sig=sig)
    enc = model.dist_to_encoder().seq_encode(data)
    n = len(data)
    label = f"sig={sig}"

    # P1 fusibility regression gate
    tc.assertTrue(fc.fusible(model), f"{label}: generated shape must be fusible")
    tc.assertTrue(fc.fusible_estep(model), f"{label}: generated shape must have a fused E-step")

    # P2 score parity
    host = model.seq_log_density(enc)
    fused = fc.fused_seq_log_density(model, enc, parallel=False)
    np.testing.assert_allclose(fused, host, rtol=1e-9, atol=1e-9, err_msg=f"{label}: score parity")

    # P4 parallel parity + bit-stable reruns
    par = fc.fused_seq_log_density(model, enc, parallel=True)
    np.testing.assert_allclose(par, host, rtol=1e-9, atol=1e-9, err_msg=f"{label}: parallel score parity")
    tc.assertTrue(np.array_equal(par, fc.fused_seq_log_density(model, enc, parallel=True)), f"{label}: rerun bits")

    # P3 M-step parity (through estimate: the stats' consumers, not their raw layout)
    w = np.ones(n)
    est = model.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, w, model)
    new_host = est.estimate(n, acc.value())
    new_fused = est.estimate(n, fc.fused_accumulate(model, enc, w, parallel=bool(rng.randint(2))))
    np.testing.assert_allclose(
        new_fused.seq_log_density(enc),
        new_host.seq_log_density(enc),
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"{label}: M-step parity",
    )

    # P5 EM monotonicity through the fused path (the normalizer must be the input model's true ll)
    cur = model
    lls = []
    for _ in range(3):
        suff, ll = fc.fused_accumulate(cur, enc, w, return_ll=True)
        lls.append(ll)
        cur = est.estimate(n, suff)
    tc.assertTrue(
        all(b - a >= -1e-9 * max(1.0, abs(a)) for a, b in zip(lls, lls[1:])),
        f"{label}: EM through the fused E-step must be monotone, got {lls}",
    )


@unittest.skipUnless(HAS_NUMBA, "the fused compiler requires numba")
class FusedCompilerFuzzTest(unittest.TestCase):
    def test_every_pool_signature_once_then_random_samples(self):
        rng = np.random.RandomState(20260713)
        for sig in SIGNATURE_POOL:  # cover every structure deterministically first
            check_sample(self, rng, sig=sig)
        for _ in range(24):  # then randomized parameters/K/data over the same compiled pool
            check_sample(self, rng)


if __name__ == "__main__":
    unittest.main()
