"""Torsion / twisted composition (mixle.inference.torsion), CARD A4-a -- research spike, sibling of A3-a.

Kill criterion (stated up front, per the card): if the twisted (shared-base) model does not beat
independently-fit per-group models on held-out per-group log-likelihood in the small-sample regime, this is a
negative result to record in notes/a4-torsion-negative.md, not to paper over.
"""

import unittest

import numpy as np

from mixle.inference.torsion import (
    CyclicGroup,
    fit_independent_mixtures,
    fit_twisted_mixture,
    independent_log_density,
    log_circular_normalizer,
)


def _integrate_over_period(log_density_fn, group, nodes=20001):
    xs = np.linspace(0.0, group.period, nodes)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.exp(log_density_fn(xs)), xs))


def _tight_group_samples(group, *, n_per_group=40, seed=0):
    rng = np.random.RandomState(seed)
    step = group.period / group.order
    return {
        k: (rng.normal(0.2 * group.period + k * step, 0.03 * group.period, size=n_per_group) % group.period)
        for k in range(group.order)
    }


def _cluster_angles_deg():
    # two fixed clusters on the period-360 circle -- the shared base pattern every group is a rotation of
    return [40.0, 200.0]


def _make_group_samples(group: CyclicGroup, *, n_per_group=10, noise=6.0, seed=0):
    """Real cyclic structure: every group k's sample is a noisy draw from ONE shared base pattern (two
    von-Mises-like clusters), rotated by k group elements -- so the groups literally differ only by a twist."""
    rng = np.random.RandomState(seed)
    base_angles = _cluster_angles_deg()
    train, test = {}, {}
    for k in range(group.order):
        shift_deg = k * (group.period / group.order)

        def draw(n, rng=rng, shift_deg=shift_deg):
            centers = rng.choice(base_angles, size=n)
            return (centers + shift_deg + rng.normal(0, noise, size=n)) % group.period

        train[k] = draw(n_per_group)
        test[k] = draw(50)
    return train, test


class CyclicGroupTest(unittest.TestCase):
    def test_twist_survives_composition(self):
        # the load-bearing algebraic check: composing two group elements and applying once must match
        # applying them one after another -- the twist is a genuine homomorphism, not an ad hoc transform
        group = CyclicGroup(order=6, period=360.0)
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 360.0, size=20)
        embedded = group.embed(x)
        for k1 in range(group.order):
            for k2 in range(group.order):
                sequential = group.act(group.act(embedded, k1), k2)
                combined = group.act(embedded, group.compose(k1, k2))
                np.testing.assert_allclose(sequential, combined, atol=1e-9)

    def test_inverse_act_undoes_the_twist(self):
        group = CyclicGroup(order=5, period=360.0)
        rng = np.random.RandomState(1)
        x = rng.uniform(0, 360.0, size=15)
        embedded = group.embed(x)
        for k in range(group.order):
            roundtrip = group.inverse_act(group.act(embedded, k), k)
            np.testing.assert_allclose(roundtrip, embedded, atol=1e-9)

    def test_embedding_is_norm_preserving(self):
        # act() must be an orthogonal transform (exact Jacobian-1), or scoring through it would silently
        # distort densities -- this is what makes log_density comparable across group elements at all
        group = CyclicGroup(order=4, period=360.0)
        embedded = group.embed(np.array([10.0, 200.0, 359.0]))
        rotated = group.act(embedded, 3)
        np.testing.assert_allclose(np.linalg.norm(embedded, axis=-1), np.linalg.norm(rotated, axis=-1), atol=1e-9)


class CircularNormalizationTest(unittest.TestCase):
    """MXR-080-1622: the ``(cos, sin)`` embedding is not a change of variables, so the ambient 2-D score
    is not a density on the periodic coordinate until its circular normalizer is divided out."""

    def setUp(self):
        self.group = CyclicGroup(order=4, period=1.0)
        train = _tight_group_samples(self.group, n_per_group=40, seed=0)
        self.twisted = fit_twisted_mixture(self.group, train, n_components=2, seed=0, max_its=60)
        self.independent = fit_independent_mixtures(self.group, train, n_components=2, seed=0, max_its=60)

    def test_twisted_log_density_integrates_to_one_over_the_period(self):
        for k in range(self.group.order):
            total = _integrate_over_period(lambda xs, k=k: self.twisted.log_density(xs, k), self.group)
            self.assertAlmostEqual(total, 1.0, places=4, msg=f"twisted density does not normalize for k={k}")

    def test_independent_log_density_integrates_to_one_over_the_period(self):
        for k in range(self.group.order):
            total = _integrate_over_period(
                lambda xs, k=k: independent_log_density(self.independent, self.group, xs, k), self.group
            )
            self.assertAlmostEqual(total, 1.0, places=4, msg=f"independent density does not normalize for k={k}")

    def test_the_two_routes_carry_different_raw_normalizers(self):
        # Why normalization is load-bearing rather than cosmetic: the shared base density and each
        # independent fit sit at different ambient scales, so an unnormalized comparison of the two
        # is decided partly by normalization instead of by fit quality.
        twisted_z = self.twisted.log_normalizer()
        independent_z = [log_circular_normalizer(self.independent[k], self.group) for k in range(self.group.order)]
        self.assertTrue(any(abs(z - twisted_z) > 1e-3 for z in independent_z))

    def test_normalizer_is_group_element_invariant(self):
        # inverse_act(embed(x), k) == embed(x - k*period/order); integrating over a full period is
        # invariant to that shift, so ONE cached normalizer is correct for every k.
        grid = np.arange(2048, dtype=np.float64) * (self.group.period / 2048)
        base = self.twisted.log_normalizer()
        for k in range(self.group.order):
            aligned = self.group.inverse_act(self.group.embed(grid), k)
            enc = self.twisted.base_density.dist_to_encoder().seq_encode([row for row in aligned])
            log_q = np.asarray(self.twisted.base_density.seq_log_density(enc), dtype=np.float64)
            shifted = float(np.log(np.mean(np.exp(log_q))) + np.log(self.group.period))
            self.assertAlmostEqual(shifted, base, places=6)


class TwistedMixtureEfficiencyTest(unittest.TestCase):
    def test_twisted_shared_base_beats_independent_per_group_in_small_sample_regime(self):
        group = CyclicGroup(order=6, period=360.0)
        train, test = _make_group_samples(group, n_per_group=10, seed=2)

        twisted = fit_twisted_mixture(group, train, n_components=2, seed=0, max_its=100)
        independent = fit_independent_mixtures(group, train, n_components=2, seed=0, max_its=100)

        twisted_ll = np.mean([np.mean(twisted.log_density(test[k], k)) for k in range(group.order)])
        independent_ll = np.mean(
            [np.mean(independent_log_density(independent, group, test[k], k)) for k in range(group.order)]
        )

        # KILL CRITERION: record a negative result if the twisted shared-base model does not win here.
        self.assertGreater(
            twisted_ll,
            independent_ll,
            f"A4 kill criterion failed: twisted={twisted_ll:.3f} <= independent={independent_ll:.3f}; "
            "record the negative result in notes/a4-torsion-negative.md",
        )

    def test_deterministic_given_seed(self):
        group = CyclicGroup(order=4, period=360.0)
        train, _test = _make_group_samples(group, n_per_group=8, seed=3)
        a = fit_twisted_mixture(group, train, n_components=2, seed=7, max_its=40)
        b = fit_twisted_mixture(group, train, n_components=2, seed=7, max_its=40)
        probe = [10.0, 100.0, 250.0]
        np.testing.assert_allclose(a.log_density(probe, 0), b.log_density(probe, 0), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
