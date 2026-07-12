"""Efficiency round 2 for the block/typed scheduling layer (worklist D3/Q5.2).

Pins the four changes: (1) per-component FUSED scoring inside the freeze-rollup column builders
(parity-gated against the host path, with non-fusible components falling back per component);
(2) measured per-component seconds from the timing receipts feeding the scheduler's cost model
(all-or-nothing, disclosed via the assumptions receipt's cost_basis); (3) reduced-precision
column scoring threaded through run_block_em(compute_dtype=...); (4) the certificate-aware
"auto" audit cadence and the schedule="auto" dispatch rule.
"""

import numpy as np
import pytest

from mixle.inference.block_em import run_block_em
from mixle.inference.freeze_rollup import FreezeRollupCache, _component_log_density_matrix_profiled
from mixle.inference.fusion_policy import prefer_block_schedule
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
        assert "measured_seconds" in bases[1:]

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
        assert not fusible(deepish)  # the Laplace leaf breaks whole-model fusion
        rng = np.random.RandomState(3)
        enc = seq_encode(rng.normal(0.0, 4.0, 20_000), model=deepish)
        # only 3 components, but each is an expensive subtree: block scheduling qualifies
        assert prefer_block_schedule(deepish, enc, max_its=100)
        assert not prefer_block_schedule(deepish, enc, max_its=1)  # weighted-work floor still applies
        cheap = MixtureDistribution([GaussianDistribution(-4.0, 1.0), LaplaceDistribution(4.0, 2.0)], [0.5, 0.5])
        enc_cheap = seq_encode(rng.normal(0.0, 4.0, 20_000), model=cheap)
        assert not prefer_block_schedule(cheap, enc_cheap, max_its=30)  # few AND cheap: full tree
