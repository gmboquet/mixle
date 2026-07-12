"""Rejection-livelock escape in run_block_em.

A rejected round restores the exact prior state, so the scheduler regenerates a near-identical
proposal; if every proposal it can produce regresses observed likelihood, the pre-fix loop paid
full proposal cost every round forever with zero progress (observed on a deep heterogeneous
mixture with GradLeaf components). These tests force that regime deterministically by corrupting
the partial-M-step proposals, then assert the vanilla escape fires, strictly improves when a real
full-tree step exists, and the run terminates -- with the whole story visible in the receipts.
"""

import pytest

import mixle.inference.block_em as block_em_module
from mixle.inference.block_em import run_block_em
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator
from mixle.stats.compute.sequence import seq_encode


def _problem(n=300, seed=3):
    # Three components with a one-block-per-round budget: three DISTINCT blocks must reject in a
    # row before their scores zero out, so the corrupted runs hit the escape threshold before the
    # tie-degeneracy fallback (all-zero scores -> dense full round) can quietly self-rescue.
    truth = MixtureDistribution(
        [GaussianDistribution(-6.0, 0.8), GaussianDistribution(0.0, 0.8), GaussianDistribution(6.0, 0.8)],
        [0.34, 0.33, 0.33],
    )
    data = truth.sampler(seed=seed).sample(size=n)
    start = MixtureDistribution(
        [GaussianDistribution(-2.0, 3.0), GaussianDistribution(0.5, 3.0), GaussianDistribution(2.0, 3.0)],
        [0.34, 0.33, 0.33],
    )
    estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator(), GaussianEstimator()])
    return start, estimator, seq_encode(data, model=start)


def _shift_components(candidate, indices, shift=8.0):
    """A deterministically WORSE proposal: shove the given components' means far off the data."""
    comps = list(candidate.components)
    for idx in indices:
        comps[idx] = GaussianDistribution(float(comps[idx].mu) + shift, float(comps[idx].sigma2))
    return MixtureDistribution(comps, [float(v) for v in candidate.w])


def _corrupt_sparse(monkeypatch):
    real = block_em_module._sparse_block_m_step

    def wrapper(enc_payload, estimator, model, active_indices, gamma_active, boundary_weight_step):
        candidate, active_counts, inactive_scale = real(
            enc_payload, estimator, model, active_indices, gamma_active, boundary_weight_step
        )
        return _shift_components(candidate, active_indices), active_counts, inactive_scale

    monkeypatch.setattr(block_em_module, "_sparse_block_m_step", wrapper)


def _wreck_active(candidate, model, indices):
    """Contract-consistent catastrophe: keep the CURRENT model's weights (a weight no-op, so the
    real M-step's weight rebalancing cannot sneak in an improvement) and untouched components,
    but push each updated component's mean 50 units past the CURRENT model's. Means only ever
    move outward from the (bounded) data, so the proposal strictly regresses from any state --
    including from a previously accepted wrecked state."""
    comps = list(candidate.components)
    for idx in indices:
        comps[idx] = GaussianDistribution(float(model.components[idx].mu) + 50.0, float(model.components[idx].sigma2))
    return MixtureDistribution(comps, [float(v) for v in model.w])


def _corrupt_all_catastrophic(monkeypatch):
    real_sparse = block_em_module._sparse_block_m_step
    real_dense = block_em_module._m_step

    def dense(enc_payload, estimator, model, gamma, scheduled_inactive):
        candidate = real_dense(enc_payload, estimator, model, gamma, scheduled_inactive)
        active = [i for i in range(model.num_components) if i not in scheduled_inactive]
        return _wreck_active(candidate, model, active)

    def sparse(enc_payload, estimator, model, active_indices, gamma_active, boundary_weight_step):
        candidate, active_counts, _ = real_sparse(
            enc_payload, estimator, model, active_indices, gamma_active, boundary_weight_step
        )
        # weights are frozen at the current model's, so the consistent inactive scale is exactly 1.
        return _wreck_active(candidate, model, active_indices), active_counts, 1.0

    monkeypatch.setattr(block_em_module, "_m_step", dense)
    monkeypatch.setattr(block_em_module, "_sparse_block_m_step", sparse)


class RejectionLivelockTest:
    def test_escape_fires_improves_and_run_terminates(self, monkeypatch):
        start, estimator, enc = _problem()
        _corrupt_sparse(monkeypatch)  # every PARTIAL proposal regresses; the dense path stays honest

        model, stats = run_block_em(
            enc,
            estimator,
            start,
            max_its=40,
            delta=None,
            budget_fraction=0.3,
            escape_after_rejections=3,
            tie_tol=0.0,  # zeroed scores after rejections must not trigger the tie-degeneracy full round
            full_refresh_interval=None,
            objective_audit_interval=None,
        )

        bases = [s.acceptance_basis for s in stats]
        # round 0 full sweep accepts; three corrupted sparse rounds reject; the escape fires;
        # three more rejections exhaust the post-escape streak and the run stops early.
        assert bases.count("vanilla_escape") == 1
        escape_index = bases.index("vanilla_escape")
        assert escape_index == 4
        assert bases[1:4] == ["rejected"] * 3
        assert all(b == "rejected" for b in bases[escape_index + 1 :])
        assert len(stats) == 8 < 40

        # the escape is a REAL vanilla EM step from the frozen state: strictly better objective
        assert stats[escape_index].accepted
        assert stats[escape_index].objective > stats[0].objective
        # termination is disclosed on the final receipt
        assert stats[-1].stop_reason == "rejection_livelock"
        assert all(s.stop_reason is None for s in stats[:-1])

    def test_terminates_even_when_the_escape_step_cannot_help(self, monkeypatch):
        start, estimator, enc = _problem()
        _corrupt_all_catastrophic(monkeypatch)

        model, stats = run_block_em(
            enc,
            estimator,
            start,
            max_its=40,
            delta=None,
            escape_after_rejections=3,
            # dense-with-all-active every round: wreck-all is contract-clean and strictly
            # regressing (no zero-responsibility component is available for a harmless no-op
            # acceptance that would keep resetting the rejection streak)
            full_tree_every_round=True,
            full_refresh_interval=None,
            objective_audit_interval=None,
        )

        bases = [s.acceptance_basis for s in stats]
        # three rejections from the start; one escape is spent (taken unconditionally, honestly
        # recorded even though it regresses); the second streak stops the run.
        assert bases[:3] == ["rejected"] * 3
        assert bases.count("vanilla_escape") == 1
        assert bases.index("vanilla_escape") == 3
        assert stats[3].accepted
        assert stats[-1].stop_reason == "rejection_livelock"
        assert len(stats) == 7 < 40

    def test_healthy_fit_never_escapes(self):
        start, estimator, enc = _problem()
        model, stats = run_block_em(enc, estimator, start, max_its=25, delta=None)

        assert all(s.acceptance_basis != "vanilla_escape" for s in stats)
        assert all(s.stop_reason is None for s in stats)
        objectives = [s.objective for s in stats]
        assert objectives[-1] >= objectives[0]

    def test_escape_parameter_is_validated(self):
        start, estimator, enc = _problem(n=40)
        with pytest.raises(ValueError):
            run_block_em(enc, estimator, start, escape_after_rejections=0)
