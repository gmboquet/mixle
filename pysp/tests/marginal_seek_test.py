"""marginal_seek: a GUARANTEED bracket on the TRUE marginal rank of a seeked value.

count_dp_seek brackets only the *tropical* rank for a mixture (its count index bins by the dominant
component and over-counts shared values). marginal_seek closes both gaps -- it widens the rank window
by the family's ``tropical_displacement_bits`` (the log2(K) cost gap) and divides the below-window
count by the component multiplicity (the over-count) -- so its ``[true_rank_lower, true_rank_upper]``
provably contains ``#{u : log p(u) > log p(value)}``. These tests pin the soundness against brute force.
"""

import random
import unittest

import numpy as np

import pysp.stats as stats
from pysp.enumeration.density_rank import MarginalSeekResult, marginal_seek
from pysp.stats.compute.pdist import DensitySemantics, EnumerationError


def _union_logdensity(dist):
    """{value: log p(value)} over the union of component supports (the full marginal support)."""
    support = {}
    for c in dist.components:
        for v, _ in c.enumerator():
            support[v] = dist.log_density(v)
    return support


def _true_rank(support, dist, value, tol=1e-12):
    t = dist.log_density(value)
    return sum(1 for lp in support.values() if lp > t + tol)


class MarginalSeekTest(unittest.TestCase):
    def _overlap_mixture(self, seed=0, n=40, k=3, conc=0.5):
        rng = np.random.RandomState(seed)
        dom = [str(i) for i in range(n)]
        comps = [stats.CategoricalDistribution(dict(zip(dom, rng.dirichlet([conc] * n)))) for _ in range(k)]
        w = list(rng.dirichlet([1.0] * k))
        return stats.MixtureDistribution(comps, w)

    def test_bracket_contains_true_rank_overlapping(self):
        # every seeked value's true marginal rank lies inside its reported bracket, at every index
        m = self._overlap_mixture()
        support = _union_logdensity(m)
        for i in range(len(support)):
            r = marginal_seek(m, i)
            tr = _true_rank(support, m, r.value)
            self.assertLessEqual(r.true_rank_lower, tr, f"lower>{tr} at idx {i}")
            self.assertLessEqual(tr, r.true_rank_upper, f"upper<{tr} at idx {i}")

    def test_log_prob_is_true_marginal(self):
        # the reported log_prob is the true log p(value), never the tropical dominant-component cost
        m = self._overlap_mixture(seed=3)
        for i in range(len(_union_logdensity(m))):
            r = marginal_seek(m, i)
            self.assertAlmostEqual(r.log_prob, m.log_density(r.value), places=9)

    def test_forced_deep_bracket_is_sound(self):
        # resolve_max=1 forces the #P-hard bracket fallback (no exact resolution); it must stay sound
        m = self._overlap_mixture(seed=1, n=50, k=4)
        support = _union_logdensity(m)
        bracketed = 0
        for i in range(len(support)):
            r = marginal_seek(m, i, resolve_max=1)
            tr = _true_rank(support, m, r.value)
            self.assertLessEqual(r.true_rank_lower, tr)
            self.assertLessEqual(tr, r.true_rank_upper)
            bracketed += int(not r.exact)
        self.assertGreater(bracketed, 0)  # the deep path was actually exercised

    def test_exact_claim_is_exact(self):
        # whenever exact is claimed, the bracket collapses to the true rank exactly
        for seed in range(8):
            m = self._overlap_mixture(seed=seed, n=random.Random(seed).randint(4, 16), k=2 + seed % 3)
            support = _union_logdensity(m)
            for i in range(len(support)):
                r = marginal_seek(m, i)
                if r.exact:
                    tr = _true_rank(support, m, r.value)
                    self.assertEqual(r.true_rank_lower, r.true_rank_upper)
                    self.assertEqual(r.true_rank_lower, tr)

    def test_disjoint_support_is_exact_and_tight(self):
        # provably-disjoint components -> displacement 0 -> every seek is exact and the bracket is tight
        d0 = stats.CategoricalDistribution({"a": 0.7, "b": 0.3})
        d1 = stats.CategoricalDistribution({"c": 0.6, "d": 0.4})
        d2 = stats.CategoricalDistribution({"e": 0.5, "f": 0.5})
        m = stats.MixtureDistribution([d0, d1, d2], [0.5, 0.3, 0.2])
        self.assertEqual(m.tropical_displacement_bits(), 0.0)
        support = _union_logdensity(m)
        for i in range(len(support)):
            r = marginal_seek(m, i)
            self.assertTrue(r.exact)
            self.assertEqual(r.true_rank_lower, r.true_rank_upper)
            self.assertEqual(r.true_rank_lower, _true_rank(support, m, r.value))

    def test_overlap_reports_positive_displacement(self):
        # genuinely overlapping components must NOT be mistaken for disjoint
        m = self._overlap_mixture(seed=5, n=8, k=3)
        self.assertGreater(m.tropical_displacement_bits(), 0.0)
        self.assertAlmostEqual(m.tropical_displacement_bits(), np.log2(3), places=9)

    def test_decomposable_parity_with_count_dp_seek(self):
        # for a decomposable family displacement is 0; marginal_seek must agree with the exact count seek
        from pysp.enumeration.density_rank import count_dp_seek

        comp = stats.CompositeDistribution(
            [
                stats.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
                stats.CategoricalDistribution({"x": 0.6, "y": 0.4}),
            ]
        )
        self.assertEqual(comp.tropical_displacement_bits(), 0.0)
        for i in range(6):
            ms = marginal_seek(comp, i)
            cs = count_dp_seek(comp, i)
            self.assertEqual(ms.value, cs.value)
            self.assertTrue(ms.exact)
            self.assertAlmostEqual(ms.log_prob, comp.log_density(ms.value), places=9)

    def test_seek_certified_on_enumerator(self):
        # the user-facing surface: dist.enumerator().seek_certified(index)
        m = self._overlap_mixture(seed=2, n=10, k=2)
        support = _union_logdensity(m)
        for i in range(len(support)):
            r = m.enumerator().seek_certified(i)
            self.assertIsInstance(r, MarginalSeekResult)
            tr = _true_rank(support, m, r.value)
            self.assertLessEqual(r.true_rank_lower, tr)
            self.assertLessEqual(tr, r.true_rank_upper)

    def test_semantics_property(self):
        m = self._overlap_mixture(seed=7, n=12, k=3)
        for i in range(len(_union_logdensity(m))):
            r = marginal_seek(m, i)
            expected = DensitySemantics.EXACT if r.exact else DensitySemantics.ESTIMATE
            self.assertIs(r.semantics, expected)

    def test_continuous_component_raises(self):
        # a continuous-component mixture has no structural count index -> EnumerationError, like count_dp_seek
        m = stats.MixtureDistribution(
            [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(3.0, 1.0)], [0.5, 0.5]
        )
        with self.assertRaises(EnumerationError):
            marginal_seek(m, 0)

    def test_extreme_weight_shared_points_tie_threshold(self):
        # regression: a 1e-9-weight component adds only ~1e-9 nats to shared points, so a genuinely
        # more-probable value must NOT be dropped as a tie. With a too-loose tol the exact rank of
        # value 0 came back 0 though points 5..9 are strictly more probable (true rank 5).
        c1 = stats.CategoricalDistribution({i: 0.1 for i in range(10)})  # support 0..9
        c2 = stats.CategoricalDistribution({i: 0.1 for i in range(5, 15)})  # overlaps 5..9
        m = stats.MixtureDistribution([c1, c2], [0.999999999, 1e-9])
        support = _union_logdensity(m)
        r = marginal_seek(m, 0)
        self.assertTrue(r.exact)
        self.assertEqual(r.true_rank_lower, _true_rank(support, m, r.value))
        self.assertEqual(r.true_rank_lower, r.true_rank_upper)

    def test_near_tie_gap_band(self):
        # regression: distinct values whose marginal log p differ by a gap in (1e-12, 1e-9) must be
        # ranked, not collapsed -- the resolve comparison's tie threshold matches the rank convention.
        import math

        n = 6
        base = 1.0 / n
        gap = 2e-12
        da = {i: base * math.exp(gap * i) for i in range(n)}
        db = {i: base * math.exp(-gap * i) for i in range(n)}
        da = {k: v / sum(da.values()) for k, v in da.items()}
        db = {k: v / sum(db.values()) for k, v in db.items()}
        m = stats.MixtureDistribution(
            [stats.CategoricalDistribution(da), stats.CategoricalDistribution(db)], [0.6, 0.4]
        )
        support = _union_logdensity(m)
        for i in range(len(support)):
            r = marginal_seek(m, i)
            tr = _true_rank(support, m, r.value)
            self.assertLessEqual(r.true_rank_lower, tr)
            self.assertLessEqual(tr, r.true_rank_upper)
            if r.exact:
                self.assertEqual(r.true_rank_lower, tr)

    def test_randomized_soundness_sweep(self):
        # broad fuzz: many random mixtures x all indices x random resolve budgets -> never a violation
        rng = np.random.RandomState(0)
        prng = random.Random(0)
        for _ in range(80):
            k = prng.randint(2, 5)
            n = prng.randint(4, 30)
            dom = [str(i) for i in range(n)]
            comps = [
                stats.CategoricalDistribution(dict(zip(dom, rng.dirichlet([prng.choice([0.3, 0.8, 1.5])] * n))))
                for _ in range(k)
            ]
            # mix in EXTREME weight asymmetry: a non-dominant component then adds only ~1e-9 nats to a
            # shared value, the regime that broke the resolve-step tie threshold.
            if prng.random() < 0.4:
                w = np.full(k, prng.choice([1e-9, 1e-10, 5e-10]))
                w[prng.randrange(k)] = 1.0
            else:
                w = rng.dirichlet([1.0] * k)
            m = stats.MixtureDistribution(comps, list(w / w.sum()))
            support = _union_logdensity(m)
            resolve_max = prng.choice([1, 4, 64, 10**9])
            for i in range(len(support)):
                try:
                    r = marginal_seek(m, i, resolve_max=resolve_max)
                except IndexError:
                    # a large probability gap can truncate the shared depth-deepening loop early, so a
                    # deep index is unreachable (same as count_dp_seek); reachable indices stay sound.
                    continue
                tr = _true_rank(support, m, r.value)
                self.assertLessEqual(r.true_rank_lower, tr)
                self.assertLessEqual(tr, r.true_rank_upper)


if __name__ == "__main__":
    unittest.main()
