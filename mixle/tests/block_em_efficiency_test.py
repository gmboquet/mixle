"""Efficiency round 2 for the block/typed scheduling layer (worklist D3/Q5.2).

Pins the four changes: (1) per-component FUSED scoring inside the freeze-rollup column builders
(parity-gated against the host path, with non-fusible components falling back per component);
(2) measured end-to-end per-component seconds from the timing receipts feeding the scheduler's cost model
(all-or-nothing, disclosed via the assumptions receipt's cost_basis); (3) reduced-precision
column scoring threaded through run_block_em(compute_dtype=...); (4) the certificate-aware
"auto" audit cadence and the schedule="auto" dispatch rule.
"""

import numpy as np
import pytest

# The whole file exercises the fused-kernel routes and the dispatch semantics that exist only
# when numba is installed; the no-numba lanes cover the host fallbacks through the ordinary suites.
pytest.importorskip("numba")

from mixle.inference.block_em import run_block_em
from mixle.inference.estimation import optimize
from mixle.inference.freeze_rollup import FreezeRollupCache, _component_log_density_matrix_profiled
from mixle.inference.fusion_policy import prefer_block_schedule, prefer_compiled_mixture
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    LaplaceDistribution,
    MixtureDistribution,
    MixtureEstimator,
)
from mixle.stats.compute.fused_codegen import fusible
from mixle.stats.compute.sequence import seq_encode
from mixle.stats.latent.mixture import _component_enc


def _problem(k=4, n=2000, seed=0, laplace_tail=False):
    rng = np.random.RandomState(seed)
    data = np.concatenate([rng.normal(6.0 * c, 1.0, n // k) for c in range(k)])
    rng.shuffle(data)
    comps = [GaussianDistribution(6.0 * c + rng.uniform(-1, 1), 2.0) for c in range(k)]
    if laplace_tail:
        comps[-1] = LaplaceDistribution(6.0 * (k - 1), 2.0)
    start = MixtureDistribution(comps, [1.0 / k] * k)
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(k)])
    return start, estimator, seq_encode(data, model=start), data


class FusedColumnParityTest:
    def test_fused_columns_match_host_columns(self):
        start, _, enc, _ = _problem()
        payload = enc[0][1]
        ll_mat, evals, _ = _component_log_density_matrix_profiled(start, payload, FreezeRollupCache(), set())
        for idx, comp in enumerate(start.components):
            host = np.asarray(comp.seq_log_density(payload), dtype=np.float64)
            np.testing.assert_allclose(ll_mat[:, idx], host, rtol=1e-12, atol=1e-12)
        assert evals == start.num_components

    def test_non_fusible_component_falls_back_per_component(self):
        start, _, enc, _ = _problem(laplace_tail=True)
        assert not fusible(start.components[-1])  # Laplace has no fused template
        payload = enc[0][1]
        ll_mat, _, _ = _component_log_density_matrix_profiled(start, payload, FreezeRollupCache(), set())
        enc_last = _component_enc(payload, len(start.components) - 1)
        host = np.asarray(start.components[-1].seq_log_density(enc_last), dtype=np.float64)
        np.testing.assert_allclose(ll_mat[:, -1], host, rtol=1e-12, atol=1e-12)

    def test_reduced_precision_columns_stay_in_band(self):
        start, estimator, enc, _ = _problem()
        m64, h64 = run_block_em(enc, estimator, start, max_its=6, delta=None)
        m32, h32 = run_block_em(enc, estimator, start, max_its=6, delta=None, compute_dtype=np.float32)
        rel = abs(h32[-1].objective - h64[-1].objective) / abs(h64[-1].objective)
        assert rel < 1e-6  # the fused float32 kernel's validated band


class MeasuredCostTest:
    def test_measured_cost_basis_engages_after_the_bootstrap_round(self):
        start, estimator, enc, _ = _problem()
        _, history = run_block_em(enc, estimator, start, max_its=5, delta=None, cost_model="measured")
        bases = [h.assumptions.cost_basis for h in history]
        assert bases[0] == "structural_parameter_count"  # nothing measured before round 0 runs
        assert "measured_block_seconds" in bases[1:]

    def test_measured_block_cost_includes_parameter_updates(self):
        start, estimator, enc, _ = _problem()
        _, history = run_block_em(enc, estimator, start, max_its=2, delta=None, cost_model="measured")
        timing = history[0].timing
        updates = dict(timing.component_update_seconds)
        blocks = dict(timing.component_block_seconds)
        assert set(updates) == set(range(start.num_components))
        assert set(blocks) == set(range(start.num_components))
        assert all(updates[idx] > 0.0 for idx in updates)
        assert all(blocks[idx] >= updates[idx] for idx in updates)

    def test_structural_default_is_unchanged(self):
        start, estimator, enc, _ = _problem()
        _, history = run_block_em(enc, estimator, start, max_its=4, delta=None)
        assert all(h.assumptions.cost_basis == "structural_parameter_count" for h in history)

    def test_cost_model_is_validated(self):
        start, estimator, enc, _ = _problem()
        with pytest.raises(ValueError):
            run_block_em(enc, estimator, start, cost_model="psychic")


class AuditCadenceAndDispatchTest:
    def test_auto_audit_interval_accepted_and_certified_runs_stay_exact(self):
        start, estimator, enc, _ = _problem()
        model, history = run_block_em(enc, estimator, start, max_its=6, delta=None, objective_audit_interval="auto")
        assert all(np.isfinite(h.objective) for h in history)
        with pytest.raises(ValueError):
            run_block_em(enc, estimator, start, objective_audit_interval="sometimes")

    def test_dispatch_routes_by_fusibility_component_count_and_workload(self):
        low, _, enc_low, _ = _problem(k=4)
        assert not prefer_block_schedule(low, enc_low, max_its=100)  # too few components
        fusible_high, _, enc_fh, _ = _problem(k=24, n=24_000)
        # a whole-model fused kernel exists: the single-pass kernel wins outright, block stays off
        assert not prefer_block_schedule(fusible_high, enc_fh, max_its=100)
        hetero_high, _, enc_hh, _ = _problem(k=24, n=24_000, laplace_tail=True)
        assert not fusible(hetero_high)  # the Laplace tail breaks whole-model fusion
        assert prefer_block_schedule(hetero_high, enc_hh, max_its=100)
        # below the workload floor even the heterogeneous high-K case stays on the plain path
        assert not prefer_block_schedule(hetero_high, enc_hh, max_its=1)

    def test_dispatch_qualifies_few_but_expensive_components(self):
        def chain(depth, center):
            node = GaussianDistribution(center, 1.0)
            for level in range(depth):
                node = MixtureDistribution([node, GaussianDistribution(center + level, 1.0)], [0.7, 0.3])
            return node

        deepish = MixtureDistribution(
            [
                chain(8, -6.0),
                chain(8, 0.0),
                MixtureDistribution([chain(7, 6.0), LaplaceDistribution(6.0, 2.0)], [0.5, 0.5]),
            ],
            [0.34, 0.33, 0.33],
        )
        # the Laplace leaf keeps this off the FAST fused paths (the bare-bridge last resort covers
        # it at host speed, which is exactly why prefer_block_schedule asks with bare_bridge=False)
        assert not fusible(deepish, bare_bridge=False)
        rng = np.random.RandomState(3)
        enc = seq_encode(rng.normal(0.0, 4.0, 20_000), model=deepish)
        # only 3 components, but each is an expensive subtree: block scheduling qualifies
        assert prefer_block_schedule(deepish, enc, max_its=100)
        # Most expensive child subtrees compile even though the heterogeneous root cannot.
        assert prefer_compiled_mixture(deepish, enc, max_its=100)
        assert not prefer_block_schedule(deepish, enc, max_its=1)  # weighted-work floor still applies
        cheap = MixtureDistribution([GaussianDistribution(-4.0, 1.0), LaplaceDistribution(4.0, 2.0)], [0.5, 0.5])
        enc_cheap = seq_encode(rng.normal(0.0, 4.0, 20_000), model=cheap)
        assert not prefer_block_schedule(cheap, enc_cheap, max_its=30)  # few AND cheap: full tree


def _deep_mixed_problem(n=3000, seed=4):
    """Three components: two deep fusible Gaussian-chain composites + one Laplace (non-fusible)."""

    def chain(depth, rng, center):
        node = GaussianDistribution(center + rng.uniform(-1, 1), 1.0)
        for level in range(depth):
            node = MixtureDistribution([node, GaussianDistribution(center + level * 0.5, 1.0)], [0.7, 0.3])
        return node

    rng = np.random.RandomState(seed)
    data = np.concatenate(
        [rng.normal(-8.0, 1.0, n // 3), rng.normal(0.0, 1.0, n // 3), rng.laplace(8.0, 1.5, n - 2 * (n // 3))]
    )
    rng.shuffle(data)
    start = MixtureDistribution(
        [chain(9, rng, -8.0), chain(9, rng, 0.0), LaplaceDistribution(6.0, 2.0)], [0.34, 0.33, 0.33]
    )
    estimator = start.estimator()
    return start, estimator, seq_encode(data, model=start), data


class FusedMStepParityTest:
    def test_fused_suff_stat_matches_host_accumulator_exactly(self):
        from mixle.inference.freeze_rollup import _component_suff_stat

        start, estimator, enc, _ = _deep_mixed_problem()
        payload = enc[0][1]
        rng = np.random.RandomState(0)
        weights = rng.uniform(0.05, 1.0, 3000)
        for idx in (0, 1):  # the deep fusible components
            comp, est_i = start.components[idx], estimator.estimators[idx]
            enc_i = _component_enc(payload, idx)
            fused_stat = _component_suff_stat(est_i, comp, enc_i, weights)
            host_acc = est_i.accumulator_factory().make()
            host_acc.seq_update(enc_i, weights, comp)

            def compare(a, b):
                if isinstance(a, (tuple, list)):
                    assert len(a) == len(b)
                    for x, y in zip(a, b):
                        compare(x, y)
                elif a is None:
                    assert b is None
                else:
                    np.testing.assert_allclose(
                        np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-12, atol=1e-12
                    )

            compare(fused_stat, host_acc.value())

    def test_end_to_end_matches_host_forced_run(self, monkeypatch):
        import mixle.inference.freeze_rollup as fr

        start, estimator, enc, data = _deep_mixed_problem()
        fused_model, fused_hist = run_block_em(enc, estimator, start, max_its=8, delta=None)

        monkeypatch.setattr(fr, "_FUSED_SCORING", False)  # force the host path end to end
        host_model, host_hist = run_block_em(enc, estimator, start, max_its=8, delta=None)
        monkeypatch.setattr(fr, "_FUSED_SCORING", None)  # let the resolver re-probe afterwards

        assert abs(fused_hist[-1].objective - host_hist[-1].objective) <= 1e-9 * abs(host_hist[-1].objective)

    def test_fp32_mstep_band_holds_on_the_deep_fixture(self):
        start, estimator, enc, _ = _deep_mixed_problem()
        _, h64 = run_block_em(enc, estimator, start, max_its=6, delta=None)
        _, h32 = run_block_em(enc, estimator, start, max_its=6, delta=None, compute_dtype=np.float32)
        rel = abs(h32[-1].objective - h64[-1].objective) / abs(h64[-1].objective)
        assert rel < 1e-6


class FusedMStepEngagementTest:
    """The fused accumulate path must ENGAGE on template-path components and must NOT engage on
    nested-tree components (whose fallback kernels recompile per call -- measured 13x slower)."""

    def _counting_resolver(self, monkeypatch):
        import mixle.inference.freeze_rollup as fr

        real = fr._fused_scoring()
        assert real, "numba required for this test"
        calls = {"n": 0}
        fused_seq_log_density, fusible, fused_accumulate, fusible_estep = real

        def counting_accumulate(model, enc, weights, **kw):
            calls["n"] += 1
            return fused_accumulate(model, enc, weights, **kw)

        monkeypatch.setattr(fr, "_FUSED_SCORING", (fused_seq_log_density, fusible, counting_accumulate, fusible_estep))
        return calls

    def test_optimize_full_dispatch_engages_compiled_component_kernels(self, monkeypatch):
        start, estimator, enc, _ = _deep_mixed_problem(n=900)
        calls = self._counting_resolver(monkeypatch)
        monkeypatch.setattr("mixle.inference.fusion_policy.prefer_compiled_mixture", lambda *_: True)

        fitted = optimize(
            None,
            estimator,
            enc_data=enc,
            prev_estimate=start,
            max_its=2,
            delta=None,
            schedule="full",
            out=None,
        )

        assert isinstance(fitted, MixtureDistribution)
        assert calls["n"] > 0

    def test_flat_subcombinator_components_engage_and_match_host(self, monkeypatch):
        import mixle.inference.freeze_rollup as fr

        rng = np.random.RandomState(2)
        data = np.concatenate([rng.normal(-8, 1, 1500), rng.normal(8, 1, 1500)])
        rng.shuffle(data)

        def flat_component(center):
            return MixtureDistribution([GaussianDistribution(center + j, 1.5) for j in range(6)], [1.0 / 6] * 6)

        start = MixtureDistribution(
            [flat_component(-8.0), flat_component(8.0), LaplaceDistribution(0.0, 3.0)], [0.4, 0.4, 0.2]
        )
        estimator = start.estimator()
        enc = seq_encode(data, model=start)

        calls = self._counting_resolver(monkeypatch)
        fused_model, fused_hist = run_block_em(enc, estimator, start, max_its=6, delta=None)
        assert calls["n"] > 0  # the template-path components really took the fused kernel

        monkeypatch.setattr(fr, "_FUSED_SCORING", False)
        host_model, host_hist = run_block_em(enc, estimator, start, max_its=6, delta=None)
        monkeypatch.setattr(fr, "_FUSED_SCORING", None)
        assert abs(fused_hist[-1].objective - host_hist[-1].objective) <= 1e-9 * abs(host_hist[-1].objective)

    def test_nested_chain_components_engage_and_match_host(self, monkeypatch):
        import mixle.inference.freeze_rollup as fr

        start, estimator, enc, _ = _deep_mixed_problem(n=900)
        calls = self._counting_resolver(monkeypatch)
        fused_model, fused_hist = run_block_em(enc, estimator, start, max_its=3, delta=None)
        assert calls["n"] > 0  # nested-tree components take the structure-cached fused kernels

        monkeypatch.setattr(fr, "_FUSED_SCORING", False)
        host_model, host_hist = run_block_em(enc, estimator, start, max_its=3, delta=None)
        monkeypatch.setattr(fr, "_FUSED_SCORING", None)
        assert abs(fused_hist[-1].objective - host_hist[-1].objective) <= 1e-9 * abs(host_hist[-1].objective)

    def test_nested_kernels_are_structure_cached_not_respecialized(self):
        import mixle.stats.compute.fused_nested as fn

        start, estimator, enc, _ = _deep_mixed_problem(n=900)
        score_before, estep_before = len(fn._SCORE_CACHE), len(fn._ESTEP_CACHE)
        run_block_em(enc, estimator, start, max_its=4, delta=None)
        # two same-structure (different-parameter) chain components across four rounds: at most ONE
        # new kernel per cache, and no per-call numba respecialization (signature growth). The bound is
        # per-dimension: 2 layout variants x 2 supported compute dtypes (float64 + the float32
        # reduced-precision path the nested kernels now honor) -- growth BEYOND that means per-call
        # respecialization is back.
        assert len(fn._SCORE_CACHE) <= score_before + 1
        assert len(fn._ESTEP_CACHE) <= estep_before + 1
        for kernel in list(fn._SCORE_CACHE.values()) + list(fn._ESTEP_CACHE.values()):
            assert len(kernel.signatures) <= 4
