"""Consolidated numerical stress panel (worklist Q5.2).

A single place that drives the closed-form families through extreme-scale and degenerate inputs and
asserts the one invariant that matters: an in-support evaluation never silently returns ``NaN`` or a
non-finite density. Individual guards for these conditions exist scattered across the suite (variance
floors, overflow guards, logsumexp extremes); this panel is the retained, parametrized receipt that
they hold together across dtype, scale, and degeneracy -- the check the release contract asks for.

A fit that legitimately has no closed form for a point returns a finite log-density here because every
probe is in support and near the fitted mass; the failure this catches is a silent ``NaN`` (or ``inf``)
from overflow, a zero/negative variance, or cancellation -- garbage that would flow into a downstream
decision without an error.

A second panel covers the required "long sequences with underflow pressure" scenario for HMMs: a single
long observation sequence multiplies thousands of per-step emission/transition probabilities, so a naive
linear-space forward pass underflows to ``0`` (log-likelihood ``-inf``) long before the sequence is
implausible. The scaled forward recursion must instead return a finite log-likelihood, and the scalar
``log_density`` and vectorized ``seq_log_density`` paths must agree (the scaling is what keeps them equal
over thousands of steps). The documented recovery behavior for a genuinely impossible observation buried
in an otherwise ordinary long sequence is a clean ``-inf`` in *both* paths -- never a silent ``NaN``.
"""

import math
import unittest

import numpy as np

import mixle.stats as st


def _fit(dist, data):
    est = dist.estimator()
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return est.estimate(float(len(data)), acc.value())


def _scaled(base, scale):
    return (np.asarray(base, dtype=float) * scale).tolist()


# (label, distribution, data, probe points). Data and probes are in support at the stated scale.
_CASES = []
for _s in (1e-8, 1.0, 1e8):
    _d = _scaled([-2.0, -1.0, 0.0, 1.0, 2.0], _s)
    _CASES.append((f"Gaussian@{_s:.0e}", st.GaussianDistribution(0.0, 1.0), _d, _d))
for _s in (1e-6, 1.0, 1e6):
    _d = _scaled([0.1, 0.5, 1.0, 2.0, 4.0], _s)
    _CASES.append((f"Exponential@{_s:.0e}", st.ExponentialDistribution(1.0), _d, _d))
for _s in (1e-4, 1.0, 1e4):
    _d = _scaled([0.5, 1.0, 2.0, 4.0, 8.0], _s)
    _CASES.append((f"Gamma@{_s:.0e}", st.GammaDistribution(2.0, 1.0), _d, _d))
for _lam in (1.0, 1e3, 1e6):
    _d = np.random.RandomState(0).poisson(_lam, size=64).tolist()
    _CASES.append((f"Poisson(lam={_lam:.0e})", st.PoissonDistribution(1.0), _d, _d[:5]))
# Degenerate: constant data -> zero scatter. The variance floor must keep this finite, not NaN/inf.
_CASES.append(("Gaussian(constant)", st.GaussianDistribution(0.0, 1.0), [3.0] * 8, [3.0, 3.0001, 2.9999]))


class NumericalStressPanelTest(unittest.TestCase):
    def test_no_silent_nonfinite_density_under_stress(self):
        for label, dist, data, probes in _CASES:
            with self.subTest(case=label):
                fitted = _fit(dist, data)
                for p in probes:
                    ld = float(fitted.log_density(p))
                    self.assertTrue(
                        math.isfinite(ld),
                        f"{label}: log_density({p!r}) = {ld} is not finite (overflow / zero variance / cancellation?)",
                    )

    def test_hmm_long_sequence_underflow_pressure(self):
        # Build two 3-state HMMs directly (no fit needed -- this panel is about density-evaluation
        # numerics, not fit quality): a discrete-emission chain and a Gaussian-emission chain.
        trans = [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]
        w = [1.0 / 3.0] * 3
        discrete = st.HiddenMarkovModelDistribution(
            [st.CategoricalDistribution({str(k): (0.6 if k == s else 0.1) for k in range(5)}) for s in range(3)],
            w=w,
            transitions=trans,
        )
        gaussian = st.HiddenMarkovModelDistribution(
            [st.GaussianDistribution(m, 0.5) for m in (-5.0, 0.0, 5.0)], w=w, transitions=trans
        )
        rng = np.random.RandomState(0)
        n = 5000  # thousands of factors: a linear-space forward pass would underflow to -inf here.
        discrete_seq = [str(int(rng.randint(5))) for _ in range(n)]
        gaussian_seq = [float(rng.randn() * 0.5 + (-5.0, 0.0, 5.0)[rng.randint(3)]) for _ in range(n)]

        for label, hmm, seq in (("discrete", discrete, discrete_seq), ("gaussian", gaussian, gaussian_seq)):
            with self.subTest(case=f"{label}/in-support"):
                scalar = float(hmm.log_density(seq))
                vectorized = float(hmm.seq_log_density(hmm.dist_to_encoder().seq_encode([seq]))[0])
                self.assertTrue(
                    math.isfinite(scalar),
                    f"{label}: log_density of a length-{n} sequence underflowed to {scalar}",
                )
                self.assertTrue(math.isfinite(vectorized), f"{label}: seq_log_density underflowed to {vectorized}")
                # The scaling is what keeps the two paths equal over thousands of steps.
                self.assertAlmostEqual(scalar, vectorized, places=6, msg=f"{label}: scalar/vectorized disagree")

            with self.subTest(case=f"{label}/impossible-observation"):
                # A single impossible observation mid-sequence: documented recovery is a clean -inf in
                # both paths, never a silent NaN that would flow into a downstream decision.
                bad = list(seq)
                bad[n // 2] = "OUT_OF_SUPPORT_ZZZ" if label == "discrete" else 1e12
                scalar = float(hmm.log_density(bad))
                vectorized = float(hmm.seq_log_density(hmm.dist_to_encoder().seq_encode([bad]))[0])
                self.assertFalse(math.isnan(scalar), f"{label}: impossible obs gave NaN (scalar), not -inf")
                self.assertFalse(math.isnan(vectorized), f"{label}: impossible obs gave NaN (vectorized), not -inf")
                if label == "discrete":  # an unseen categorical symbol has exactly zero mass -> -inf
                    self.assertEqual(scalar, float("-inf"))
                    self.assertEqual(vectorized, float("-inf"))


if __name__ == "__main__":
    unittest.main()
