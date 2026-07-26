"""Out-of-support and impossible rows through the compiled fused paths.

Every case pins fused-vs-host parity in a regime where a naive kernel silently diverges:

* scoring parity with a leaf whose expr carries a real ``-np.inf`` branch (Pareto below its scale):
  full ``fastmath=True`` (ninf/nnan) genuinely miscompiled this -- positive log-densities -- so the
  parity here is the regression net for the no-ninf/nnan compile policy;
* a row impossible under EVERY component: the E-step must assign zero responsibility and score
  the row -inf instead of fabricating statistics or NaN-poisoning the fused-EM normalizer;
* nested mixture-of-mixtures: the same guarantees at every mixture NODE, and ``wants_minmax``
  leaves (Pareto) must decline nested fusion cleanly instead of crashing mid-fit;
* the fused Pareto E-step's support minimum is PER-COMPONENT over rows with responsibility > 0
  (the legacy statistic ``ParetoEstimator`` uses directly as the scale xm), never the global data
  minimum -- including across parallel chunk combines, where minima must min-reduce, not sum.
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    import mixle.stats as stats
    from mixle.stats.compute import fused_codegen as fc
    from mixle.stats.compute import fused_nested as fn


def _legacy_suff_stats(model, enc, weights):
    est = model.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, weights, model)
    return est, acc.value()


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class ParetoScoringParityTest(unittest.TestCase):
    """D-1 regression: scoring parity with an out-of-support leaf, sequential and parallel."""

    def _model(self):
        return stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.0, alpha=2.5),
                stats.ParetoDistribution(xm=2.0, alpha=1.5),
            ],
            w=[0.5, 0.5],
        )

    def test_rows_below_one_component_scale_match_host_exactly(self):
        # 1.2 / 1.5 / 1.1 sit below comp-2's xm: its -inf branch is live while the mixture stays
        # finite. Under fastmath=True these rows scored POSITIVE log-densities (+1.38 / +1.57 /
        # +1.36 measured); the subset-compiled kernel must agree with the host to the last digit.
        model = self._model()
        enc = model.dist_to_encoder().seq_encode([1.2, 1.5, 3.0, 5.0, 1.1])
        host = model.seq_log_density(enc)
        self.assertTrue(np.all(np.isfinite(host)))
        for parallel in (False, True):
            fused = fc.fused_seq_log_density(model, enc, parallel=parallel)
            np.testing.assert_allclose(fused, host, rtol=1e-12, atol=0.0, err_msg=f"parallel={parallel}")

    def test_row_below_every_scale_scores_minus_inf(self):
        # complements the edge panel's categorical case with a SCALAR -inf leaf
        model = self._model()
        enc = model.dist_to_encoder().seq_encode([0.5, 3.0])
        host = model.seq_log_density(enc)
        self.assertTrue(np.isneginf(host[0]))
        for parallel in (False, True):
            fused = fc.fused_seq_log_density(model, enc, parallel=parallel)
            self.assertTrue(np.isneginf(fused[0]), f"parallel={parallel}")
            np.testing.assert_allclose(fused[1], host[1], rtol=1e-12)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class EstepImpossibleRowTest(unittest.TestCase):
    """D-2: a row impossible under every component must not NaN-poison the fused E-step."""

    def _model(self):
        return stats.MixtureDistribution(
            components=[
                stats.CategoricalDistribution(pmap={"a": 0.7, "b": 0.3}),
                stats.CategoricalDistribution(pmap={"a": 0.3, "b": 0.7}),
            ],
            w=[0.5, 0.5],
        )

    def test_impossible_row_has_zero_responsibility_and_minus_inf_ll(self):
        model = self._model()
        data = ["a", "b", "c", "a"]  # "c" is impossible under BOTH components
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        est, legacy = _legacy_suff_stats(model, enc, w)
        self.assertTrue(np.all(np.isfinite(legacy[0])))
        host_ll = float(w @ model.seq_log_density(enc))
        self.assertTrue(np.isneginf(host_ll))
        for parallel in (False, True):
            suff, ll = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=parallel)
            # counts: finite, and exactly the host zero-responsibility policy for the impossible row
            np.testing.assert_allclose(suff[0], legacy[0], rtol=1e-12, err_msg=f"parallel={parallel}")
            # the row's log-likelihood is -inf, matching seq_log_density (not NaN, not finite)
            self.assertTrue(np.isneginf(ll), f"parallel={parallel}: ll={ll}")
            # end to end: the M-step sees the same statistics the legacy accumulator produced
            new_fused = est.estimate(len(data), suff)
            new_host = est.estimate(len(data), legacy)
            np.testing.assert_allclose(new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=1e-12)

    def test_zero_weighted_impossible_row_contributes_exactly_zero_not_nan(self):
        # `out_ll[0] += wi * (m + log(s)) if ok else wi * -np.inf` computes 0.0 * -np.inf = NaN when a
        # row is BOTH impossible (ok=False) AND zero-weighted -- poisoning the total log-likelihood for
        # the whole batch, not just that one (rightly excluded) row. A zero-weighted row must
        # contribute exactly 0.0 regardless of whether it's possible under the model.
        model = self._model()
        data = ["a", "b", "c", "a"]  # "c" (index 2) is impossible under BOTH components
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.array([1.0, 1.0, 0.0, 1.0])
        host_ll = model.seq_log_density(enc)
        self.assertTrue(np.isneginf(host_ll[2]))
        # the correct weighted total excludes the zero-weighted row's -inf entirely (0.0 * -inf is
        # NOT a valid way to compute this -- it's NaN, not 0.0 -- so filter before summing).
        expected_ll = float(np.dot(w[w != 0], host_ll[w != 0]))
        self.assertTrue(np.isfinite(expected_ll))
        for parallel in (False, True):
            suff, ll = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=parallel)
            self.assertFalse(np.isnan(ll), f"parallel={parallel}: ll={ll}")
            self.assertAlmostEqual(ll, expected_ll, places=10, msg=f"parallel={parallel}")
            self.assertTrue(np.all(np.isfinite(suff[0])), f"parallel={parallel}: suff={suff[0]}")


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class NestedImpossibleRowTest(unittest.TestCase):
    """D-3: nested mixture nodes guard their log-sum-exp and responsibility pushdown.

    Gaussian leaves overflow to a real ``-inf`` score at |x - mu| ~ 1e200 (the squared residual
    exceeds the float64 range) on the host and fused paths alike, which makes an every-leaf
    impossible row constructible from nested-fusible leaves.
    """

    def _model(self):
        inner1 = stats.MixtureDistribution(
            components=[
                stats.GaussianDistribution(mu=-1.0, sigma2=1.0),
                stats.GaussianDistribution(mu=0.0, sigma2=2.0),
            ],
            w=[0.6, 0.4],
        )
        inner2 = stats.MixtureDistribution(
            components=[stats.GaussianDistribution(mu=3.0, sigma2=1.0), stats.GaussianDistribution(mu=5.0, sigma2=0.5)],
            w=[0.5, 0.5],
        )
        return stats.MixtureDistribution(components=[inner1, inner2], w=[0.7, 0.3])

    def test_impossible_row_scores_minus_inf_at_every_level(self):
        model = self._model()
        data = [0.5, 1.0e200, 3.2]
        enc = model.dist_to_encoder().seq_encode(data)
        with np.errstate(over="ignore"):
            host = model.seq_log_density(enc)
        self.assertTrue(np.isneginf(host[1]) and np.isfinite(host[0]) and np.isfinite(host[2]))
        for parallel in (False, True):
            fused = fn.fused_nested_seq_log_density(model, enc, parallel=parallel)
            np.testing.assert_allclose(fused, host, rtol=1e-12, err_msg=f"parallel={parallel}")

    def test_estep_impossible_row_has_zero_responsibility_recursively(self):
        model = self._model()
        data = [0.5, 1.0e200, 3.2]
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        with np.errstate(over="ignore"):
            suff, ll = fn.fused_nested_accumulate(model, enc, w, return_ll=True)
            valid_enc = model.dist_to_encoder().seq_encode([data[0], data[2]])
            _, expected = _legacy_suff_stats(model, valid_enc, np.ones(2))
        # Root and inner mixture counts match the host policy: the impossible row contributes
        # no latent assignment or leaf arithmetic at any level. Compare against
        # physically removing that row; multiplying its overflowing values by
        # zero would otherwise manufacture NaN sufficient statistics.
        np.testing.assert_allclose(np.asarray(suff[0]), np.asarray(expected[0]), rtol=1e-12)
        for j in range(2):
            np.testing.assert_allclose(np.asarray(suff[1][j][0]), np.asarray(expected[1][j][0]), rtol=1e-12)
            for leaf_fused, leaf_expected in zip(suff[1][j][1], expected[1][j][1]):
                np.testing.assert_allclose(np.asarray(leaf_fused), np.asarray(leaf_expected), rtol=1e-12)
        self.assertTrue(np.isneginf(ll))

    def test_zero_weighted_impossible_row_contributes_exactly_zero_not_nan(self):
        # Same NaN-vs-inf gap as the flat mixture's E-step, but here root_expr's -inf comes from the
        # RECURSIVE per-node finite-max guard in _emit_score rather than a single top-level `ok`
        # check -- out_ll[0] += wi * (root_expr) still computes 0.0 * -inf = NaN for a row that's both
        # impossible at every level AND zero-weighted.
        model = self._model()
        data = [0.5, 1.0e200, 3.2]  # index 1 overflows every leaf to -inf
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.array([1.0, 0.0, 1.0])
        with np.errstate(over="ignore"):
            host_ll = model.seq_log_density(enc)
        self.assertTrue(np.isneginf(host_ll[1]))
        expected_ll = float(np.dot(w[w != 0], host_ll[w != 0]))
        self.assertTrue(np.isfinite(expected_ll))
        with np.errstate(over="ignore"):
            for parallel in (False, True):
                suff, ll = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=parallel)
                self.assertFalse(np.isnan(ll), f"parallel={parallel}: ll={ll}")
                self.assertAlmostEqual(ll, expected_ll, places=10, msg=f"parallel={parallel}")
                self.assertTrue(np.all(np.isfinite(np.asarray(suff[0]))), f"parallel={parallel}")


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class NestedMinmaxDeclineTest(unittest.TestCase):
    """D-4: minmax leaves decline the NESTED template with the DECLARED ValueError, never a mid-fit
    crash -- and the bare-bridge last resort then covers the shape with host-exact semantics (it
    drives each inner mixture's own accumulator, so weighted per-component minima survive)."""

    def _model(self):
        inner1 = stats.MixtureDistribution(
            components=[stats.ParetoDistribution(xm=1.0, alpha=2.0), stats.ParetoDistribution(xm=2.0, alpha=3.0)],
            w=[0.6, 0.4],
        )
        inner2 = stats.MixtureDistribution(
            components=[stats.ParetoDistribution(xm=1.5, alpha=1.5), stats.ParetoDistribution(xm=3.0, alpha=2.5)],
            w=[0.5, 0.5],
        )
        return stats.MixtureDistribution(components=[inner1, inner2], w=[0.7, 0.3])

    def test_nested_pareto_mixture_declines_the_template_and_bridges_with_parity(self):
        model = self._model()
        data = [1.2, 3.0, 0.3]  # 0.3 is below every xm: out of support everywhere
        enc = model.dist_to_encoder().seq_encode(data)
        # the nested TEMPLATE still refuses up front (previously: scoring returned NaN on the 0.3
        # row and the E-step crashed with ``TypeError: 'NoneType' object is not callable`` mid-fit)
        with self.assertRaises(ValueError):
            fn.fused_nested_seq_log_density(model, enc)
        with self.assertRaises(ValueError):
            fn.fused_nested_accumulate(model, enc, np.ones(len(data)))
        # ...but the shape is fusible anyway: the bare-bridge last resort picks it up
        self.assertTrue(fc.fusible(model))
        self.assertTrue(fc.fusible_estep(model))
        host = model.seq_log_density(enc)
        self.assertTrue(np.isfinite(host[0]) and np.isfinite(host[1]) and np.isneginf(host[2]))
        fused = fc.fused_seq_log_density(model, enc)
        np.testing.assert_allclose(fused[:2], host[:2], rtol=1e-9)
        self.assertTrue(np.isneginf(fused[2]), "the out-of-support row must stay -inf, never NaN")
        # M-step parity through the bridge E-step (weighted minima handled by the host accumulator)
        w = np.ones(len(data))
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        new_host = est.estimate(len(data), acc.value())
        new_fused = est.estimate(len(data), fc.fused_accumulate(model, enc, w))
        np.testing.assert_allclose(new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=1e-9, atol=1e-12)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class ParetoEstepMinimaTest(unittest.TestCase):
    """D-5: the fused Pareto suff-stat minimum is per-component and weighted, not the global min."""

    def _model(self):
        return stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.0, alpha=2.5),
                stats.ParetoDistribution(xm=2.0, alpha=1.5),
            ],
            w=[0.5, 0.5],
        )

    def test_per_component_weighted_minimum_matches_legacy(self):
        model = self._model()
        data = [1.2, 1.5, 3.0, 5.0, 1.1]  # comp-2 has responsibility > 0 only on 3.0 and 5.0
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        est, legacy = _legacy_suff_stats(model, enc, w)
        self.assertEqual([v[2] for v in legacy[1]], [1.1, 3.0])
        for parallel in (False, True):
            suff = fc.fused_accumulate(model, enc, w, parallel=parallel)
            # the global data minimum (1.1) is WRONG for comp-2: its scale would collapse below xm
            self.assertEqual(
                [v[2] for v in suff[1]], [1.1, 3.0], f"parallel={parallel}: fused minima are not per-component"
            )
            np.testing.assert_allclose(suff[0], legacy[0], rtol=1e-12)
            for k in range(2):
                np.testing.assert_allclose(np.asarray(suff[1][k]), np.asarray(legacy[1][k]), rtol=1e-12)
            # ParetoEstimator uses the minimum directly as xm: the M-step must land on legacy's scales
            new_fused = est.estimate(len(data), suff)
            new_host = est.estimate(len(data), legacy)
            self.assertEqual([c.xm for c in new_fused.components], [c.xm for c in new_host.components])

    def test_component_with_no_support_keeps_infinite_minimum(self):
        # no row reaches comp-2 (xm=100): its weighted minimum must stay +inf exactly like the
        # legacy accumulator's untouched init, not 0.0 or the global data minimum
        model = stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.0, alpha=2.0),
                stats.ParetoDistribution(xm=100.0, alpha=2.0),
            ],
            w=[0.5, 0.5],
        )
        data = [1.5, 2.5, 4.0]
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        _, legacy = _legacy_suff_stats(model, enc, w)
        suff = fc.fused_accumulate(model, enc, w)
        self.assertEqual([v[2] for v in suff[1]], [v[2] for v in legacy[1]])
        self.assertTrue(np.isposinf(suff[1][1][2]))

    def test_parallel_chunk_combine_min_reduces_the_minima(self):
        # enough rows for two REAL chunks (n // 16384 == 2): a sum-combine of per-chunk minima
        # (the classic threaded-merge bug) would corrupt xm; the combine must min-reduce
        rng = np.random.RandomState(7)
        model = self._model()
        data = np.concatenate(
            [
                (rng.pareto(2.5, size=20_000) + 1.0) * 1.0,
                (rng.pareto(1.5, size=20_000) + 1.0) * 2.0,
            ]
        ).tolist()
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        _, legacy = _legacy_suff_stats(model, enc, w)
        suff = fc.fused_accumulate(model, enc, w, parallel=True)
        self.assertEqual([v[2] for v in suff[1]], [v[2] for v in legacy[1]])
        np.testing.assert_allclose(suff[0], legacy[0], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
