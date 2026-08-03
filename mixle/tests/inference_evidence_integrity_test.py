"""MXR-080-1899: inference results must be complete, self-consistent, and un-editable after the fact.

One finding, one defect class across six surfaces: a public entry point accepted evidence it had not
finished checking (an under-delivered posterior batch, a Boolean where a probability level belongs,
two evidence keys naming the same field), or it mutated state before it knew the operation would
succeed, or it handed back a result whose own invariants the holder could then break.

Every test below names the behaviour that was reproduced against the pre-fix code, so a regression
reads as the original defect rather than as an unexplained assertion.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from mixle.inference.causal import CausalIdentification
from mixle.inference.condition import condition
from mixle.inference.decision import bayes_action
from mixle.inference.em import AnnealedEM, PosteriorTransformEM
from mixle.inference.event_study import EventStudyIdentification, poisson_lograte_effect
from mixle.inference.mcmc.samplers import MCMCResult
from mixle.inference.streaming import IncrementalEstimator
from mixle.inference.uncertainty import cluster_samples, decompose_variance
from mixle.stats import GaussianDistribution, GaussianEstimator, MultivariateGaussianDistribution


class _FixedDrawPosterior:
    """A ``samples(n, rng)`` implementation that ignores ``n`` -- the under-delivery defect."""

    def __init__(self, draws):
        self.draws = list(draws)

    def samples(self, n, rng):
        return list(self.draws)


class _GaussianPosterior:
    """A well-behaved posterior: exactly ``n`` draws, as the ``samples(n, rng)`` contract requires."""

    def samples(self, n, rng):
        return rng.randn(n)


def _abs_loss(action, draw):
    return abs(action - draw)


# --------------------------------------------------------------------------------------------- #
# decision: under-delivered posterior batches and weak quantile / CVaR flags
# --------------------------------------------------------------------------------------------- #


def test_bayes_action_refuses_a_posterior_that_under_delivers_draws():
    """Reproduced: ``bayes_action(..., n=2000)`` against a posterior returning 3 draws returned a
    complete-looking answer whose CVaR/quantiles were computed from three order statistics."""
    with pytest.raises(ValueError, match="returned 3 draw"):
        bayes_action(_FixedDrawPosterior([0.0, 10.0, 100.0]), _abs_loss, [0.0, 10.0], n=2000)

    # The honest spelling of the same request still works: ask for the draws that actually exist.
    out = bayes_action(_FixedDrawPosterior([0.0, 10.0, 100.0]), _abs_loss, [0.0, 10.0], n=3)
    assert out["action"] == 10.0


def test_bayes_action_refuses_boolean_and_string_cvar_alpha():
    """Reproduced: ``cvar_alpha=True`` became a tail mass of 1.0, so the reported "tail risk" was the
    mean of the whole loss distribution, and ``cvar_alpha='0.5'`` was coerced from configuration text."""
    for alpha in (True, "0.5", None):
        with pytest.raises(TypeError, match="cvar_alpha"):
            bayes_action(_GaussianPosterior(), _abs_loss, [0.0], n=32, cvar_alpha=alpha)


def test_bayes_action_refuses_boolean_quantile_levels():
    """Reproduced: ``quantiles=(True, False)`` was reported as the levels 1.0 and 0.0."""
    with pytest.raises(TypeError, match="quantiles"):
        bayes_action(_GaussianPosterior(), _abs_loss, [0.0], n=32, quantiles=(True, False))


def test_bayes_action_checks_quantile_levels_before_calling_the_loss():
    """Reproduced: an out-of-range level was caught inside ``np.quantile``, i.e. only after the loss
    had already been evaluated once per draw for every action. A loss is caller code."""
    calls = []

    def counting_loss(action, draw):
        calls.append(action)
        return abs(action - draw)

    counting_loss.vectorized = False
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bayes_action(_GaussianPosterior(), counting_loss, [0.0, 1.0], n=32, quantiles=(0.5, 1.5))
    assert calls == []


def test_bayes_action_still_accepts_the_quantile_spellings_callers_legitimately_use():
    """Guard-overreach check: no quantiles at all, a repeated level, and the 0.0/1.0 endpoints are
    all coherent requests and must keep working -- only non-numbers and out-of-range levels fail."""
    empty = bayes_action(_GaussianPosterior(), _abs_loss, [0.0], n=64, quantiles=())
    assert empty["risk_profile"]["quantiles"] == {}

    repeated = bayes_action(_GaussianPosterior(), _abs_loss, [0.0], n=64, quantiles=(0.5, 0.5, 0.0, 1.0))
    assert sorted(repeated["risk_profile"]["quantiles"]) == ["0.0", "0.5", "1.0"]

    numpy_levels = bayes_action(
        _GaussianPosterior(), _abs_loss, [0.0], n=64, cvar_alpha=np.float64(0.25), quantiles=np.array([0.1, 0.9])
    )
    assert numpy_levels["risk_profile"]["cvar_alpha"] == 0.25


# --------------------------------------------------------------------------------------------- #
# conditioning: truncated sample counts and duplicate normalized evidence
# --------------------------------------------------------------------------------------------- #


def _bivariate_model():
    return MultivariateGaussianDistribution(np.array([0.0, 0.0]), np.array([[1.0, 0.6], [0.6, 1.0]]))


def test_condition_posterior_sample_refuses_a_truncated_draw_count():
    """Reproduced: ``post.sample(2.9)`` silently returned two records and ``post.sample(True)`` one."""
    post = condition(_bivariate_model(), {0: 1.0}, method="exact")
    for bad in (2.9, True, np.float64(4.0)):
        with pytest.raises(TypeError, match="exact positive integer"):
            post.sample(bad)
    assert len(post.sample(3, seed=0)) == 3
    assert len(post.sample(np.int64(2), seed=0)) == 2  # numpy integers stay legal


def test_condition_refuses_two_evidence_keys_naming_the_same_field():
    """Reproduced: ``{0: 1.0, (0,): 99.0}`` normalized to one path and the dict comprehension kept
    whichever came last, so the posterior was conditioned on evidence the caller did not choose."""
    with pytest.raises(ValueError, match="both refer to field path"):
        condition(_bivariate_model(), {0: 1.0, (0,): 99.0}, method="exact")

    # Distinct fields spelled either way remain fine -- the guard is about collisions, not tuples.
    post = condition(_bivariate_model(), {(0,): 1.0}, method="exact")
    assert post.receipt.method == "exact"


def test_condition_refuses_a_boolean_or_fractional_particle_count():
    """Reproduced: ``int(n_particles)`` accepted ``True``, so SIR ran with ONE particle -- a single
    prior draw that reports a perfect ``ess_ratio`` of 1.0 -- and truncated a fractional count.

    Checked on ``method='auto'`` too: the particle count is part of the request, so it is validated
    whether or not the exact path ends up making it unnecessary."""
    for bad in (True, 4096.5):
        for method in ("sir", "auto"):
            with pytest.raises(TypeError, match="exact positive integer"):
                condition(_bivariate_model(), {0: 1.0}, method=method, n_particles=bad, seed=0)


# --------------------------------------------------------------------------------------------- #
# causal: truthy strings read as identified, Boolean Poisson counts
# --------------------------------------------------------------------------------------------- #


def test_causal_identification_refuses_truthy_string_assumptions():
    """Reproduced: a receipt whose assumptions all read ``"false"`` reported ``identified=True`` and
    passed every causal gate, because ``identified`` was a truthiness conjunction."""
    with pytest.raises(TypeError, match="exchangeability"):
        CausalIdentification(
            graph_source="domain_asserted",
            estimand="ate",
            evidence=("protocol://dag",),
            assumptions=("pre-specified",),
            exchangeability="false",
            positivity="false",
            consistency="false",
            no_interference="false",
        )


def test_causal_identification_refuses_a_truthy_structural_counterfactual_flag():
    """Reproduced: ``structural_counterfactuals='no'`` opened ``counterfactual()``'s gate."""
    with pytest.raises(TypeError, match="structural_counterfactuals"):
        CausalIdentification.domain_asserted("protocol://dag", structural_counterfactuals="no")


def test_causal_identification_still_accepts_numpy_booleans():
    """Guard-overreach check: ``np.bool_`` is a genuine Boolean and the library's own array paths
    produce it, so it is canonicalized rather than refused."""
    receipt = CausalIdentification(
        graph_source="randomized_design",
        estimand="ate",
        evidence=("trial://registration",),
        assumptions=("randomized assignment",),
        exchangeability=np.True_,
        positivity=True,
        consistency=True,
        no_interference=True,
    )
    assert receipt.identified
    assert receipt.to_dict()["exchangeability"] is True


def test_event_study_identification_refuses_truthy_string_assumptions():
    """Reproduced: the same truthiness conjunction in the difference-in-differences receipt, where a
    deserialized ``no_anticipation: "false"`` still read as an identified DiD estimate."""
    with pytest.raises(TypeError, match="no_anticipation"):
        EventStudyIdentification(
            design_evidence=("study://matched",),
            parallel_trends_evidence=("analysis://placebo",),
            exchangeability=True,
            positivity=True,
            consistency=True,
            no_interference=True,
            no_anticipation="false",
        )


def test_poisson_lograte_effect_refuses_boolean_event_counts():
    """Reproduced: ``poisson_lograte_effect(True, 1.0, False, 1.0)`` read a Boolean indicator as
    "one event before, zero after" and returned a confident negative log-rate shift."""
    for bad in (True, False, np.bool_(True)):
        with pytest.raises(TypeError, match="must be a number, not a Boolean"):
            poisson_lograte_effect(bad, 1.0, 2.0, 1.0)
        with pytest.raises(TypeError, match="must be a number, not a Boolean"):
            poisson_lograte_effect(3.0, bad, 2.0, 1.0)
    # Ordinary integer counts (including zero, via the Haldane correction) are untouched.
    effect, var = poisson_lograte_effect(0, 2.0, 4, 2.0)
    assert effect > 0.0 and var > 0.0


# --------------------------------------------------------------------------------------------- #
# streaming: partial mutation before failure
# --------------------------------------------------------------------------------------------- #


class _RefusingEstimator:
    """A GaussianEstimator whose M-step can be made to reject the state it is handed."""

    def __init__(self):
        self._inner = GaussianEstimator()
        self.refuse = False

    def accumulator_factory(self):
        return self._inner.accumulator_factory()

    def estimate(self, nobs, value):
        if self.refuse:
            raise RuntimeError("M-step refused this state")
        return self._inner.estimate(nobs, value)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_incremental_update_that_fails_leaves_the_chunk_store_untouched():
    """Reproduced: revisiting a chunk overwrote its stored payload, advanced ``nobs`` and the running
    accumulator, and only then ran the M-step -- so a rejected batch destroyed the previous good
    contribution for that chunk id, left ``model`` disagreeing with the statistics behind it, and was
    silently baked into the next successful update."""
    estimator = _RefusingEstimator()
    incremental = IncrementalEstimator(estimator, model=GaussianDistribution(0.0, 1.0))
    incremental.update([0.0, 1.0, 2.0], chunk_id="a")

    good_chunk = copy.deepcopy(incremental.chunk_value("a"))
    good_model = incremental.model
    good_nobs = incremental.nobs
    good_step = incremental.step
    good_running = copy.deepcopy(incremental.running_accumulator.value())

    estimator.refuse = True
    with pytest.raises(RuntimeError, match="refused"):
        incremental.update([100.0, 200.0, 300.0], chunk_id="a")

    assert incremental.chunk_value("a") == good_chunk
    assert incremental.running_accumulator.value() == good_running
    assert incremental.nobs == good_nobs
    assert incremental.step == good_step
    assert incremental.model is good_model

    # A brand-new chunk that fails must not be half-registered either.
    with pytest.raises(RuntimeError, match="refused"):
        incremental.update([7.0], chunk_id="b")
    assert sorted(incremental.chunk_values) == ["a"]
    assert incremental.nobs == good_nobs

    # And the estimator is still usable: the retry lands on the pre-failure state, not on top of it.
    estimator.refuse = False
    incremental.update([7.0, 9.0], chunk_id="b")
    assert sorted(incremental.chunk_values) == ["a", "b"]
    assert incremental.nobs == good_nobs + 2.0


def test_incremental_replacement_still_replaces_on_success():
    """Guard-overreach check: staging must not turn a successful revisit into a no-op -- the chunk
    payload is replaced (not accumulated) and ``nobs`` reflects the new batch size."""
    incremental = IncrementalEstimator(GaussianEstimator(), model=GaussianDistribution(0.0, 1.0))
    incremental.update([0.0, 1.0, 2.0], chunk_id="a")
    incremental.update([5.0, 5.0], chunk_id="b")
    incremental.update([10.0, 10.0, 10.0, 10.0], chunk_id="a")
    assert incremental.nobs == 6.0
    assert incremental.nobs_by_chunk == {"a": 4.0, "b": 2.0}
    assert abs(float(incremental.model.mu) - (10.0 * 4 + 5.0 * 2) / 6.0) < 1e-9


# --------------------------------------------------------------------------------------------- #
# MCMC / uncertainty: results that exposed mutable arrays
# --------------------------------------------------------------------------------------------- #


def test_mcmc_result_owns_read_only_evidence():
    """Reproduced: ``accepted`` and ``log_probs`` were aliases of the caller's arrays (so editing
    them afterwards rewrote ``acceptance_rate``), and ``samples.append(...)`` broke the one-log-prob-
    per-sample agreement that ``__post_init__`` had just verified."""
    accepted = np.array([True, False, True])
    log_probs = np.array([-1.0, -2.0, -3.0])
    result = MCMCResult(samples=[1.0, 2.0, 3.0], log_probs=log_probs, accepted=accepted)
    rate = result.acceptance_rate

    accepted[:] = False
    log_probs[:] = -99.0
    assert result.acceptance_rate == rate
    assert result.log_probs.tolist() == [-1.0, -2.0, -3.0]

    for array in (result.log_probs, result.accepted):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = array[0]
    assert isinstance(result.samples, tuple)
    with pytest.raises(AttributeError):
        result.samples.append(4.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.log_probs = np.zeros(3)

    # The summaries still read the sealed arrays, and dataclasses.replace still round-trips.
    assert result.acceptance_rate == pytest.approx(2.0 / 3.0)
    assert dataclasses.replace(result, samples=(4.0, 5.0, 6.0)).samples == (4.0, 5.0, 6.0)


def test_uncertainty_decomposition_owns_read_only_arrays():
    """Reproduced: ``decomposition.total[...] = x`` left the record asserting a total that no longer
    equalled ``aleatoric + epistemic``, and ``fraction_epistemic`` reported the edited version."""
    means = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    variances = np.ones_like(means)
    decomposition = decompose_variance(means, variances)

    means[0, 0] = -1e9  # the record took its own copy
    assert decomposition.total == pytest.approx(decomposition.aleatoric + decomposition.epistemic)
    for array in (decomposition.total, decomposition.aleatoric, decomposition.epistemic):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0.0

    # The single-point path still collapses to plain floats.
    single = decompose_variance(np.array([1.0, 3.0]))
    assert isinstance(single.item().total, float)


def test_clustering_owns_its_classes():
    """Reproduced: ``clustering.probs[...] = 5.0`` produced a class distribution summing to 10, and
    ``representatives.append(...)`` added a class with no mass and no members."""
    clustering = cluster_samples(["a", "b", "a"])
    assert isinstance(clustering.representatives, tuple)
    assert clustering.probs.sum() == pytest.approx(1.0)
    for array in (clustering.probs, clustering.labels):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0
    with pytest.raises(AttributeError):
        clustering.representatives.append("z")


# --------------------------------------------------------------------------------------------- #
# EM controls that coerced invalid values
# --------------------------------------------------------------------------------------------- #


def test_em_hard_flag_is_not_coerced():
    """Reproduced: ``PosteriorTransformEM(hard="false")`` ran classification EM -- a different
    algorithm than the configuration names, visible only in the fitted model."""
    with pytest.raises(TypeError, match="hard"):
        PosteriorTransformEM(temperature=1.0, hard="false")
    with pytest.raises(TypeError, match="hard_final"):
        AnnealedEM([1.0, 0.0], hard_final="false")
    assert PosteriorTransformEM(temperature=1.0, hard=np.True_).hard is True


def test_em_temperature_rejects_nan_and_booleans():
    """Reproduced: ``temperature=nan`` passed ``if temperature < 0.0`` (NaN compares false against
    every bound) and then made every transformed responsibility zero, so the M-step ran on an
    all-zero E-step instead of reporting the nonsense schedule."""
    for bad in (float("nan"), True, "1.0"):
        with pytest.raises((TypeError, ValueError)):
            PosteriorTransformEM(temperature=bad)
        with pytest.raises((TypeError, ValueError)):
            AnnealedEM([bad])
    assert PosteriorTransformEM(temperature=np.float64(2.0)).temperature == 2.0
    assert AnnealedEM([2.0, 1.0, 0.0]).temperatures == (2.0, 1.0, 0.0)
