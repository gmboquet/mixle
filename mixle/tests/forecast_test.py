"""forecast(): horizon predictions with honest intervals from a fitted HMM."""

import numpy as np
import pytest

from mixle.inference import forecast
from mixle.stats import GaussianDistribution, HiddenMarkovModelDistribution


def _hmm(stay=0.9):
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(-4.0, 1.0), GaussianDistribution(4.0, 1.0)],
        [0.5, 0.5],
        [[stay, 1 - stay], [1 - stay, stay]],
    )


def test_short_horizon_tracks_the_current_regime_and_long_horizon_mixes():
    m = _hmm(stay=0.97)
    history = [3.8, 4.2, 3.9, 4.1, 4.0]  # clearly the +4 regime
    f = forecast(m, history, horizon=200, level=0.9, n=8000, seed=0)

    # step 1: p(stay)=0.97 -> exact mean 0.97*4 + 0.03*(-4) = 3.76
    assert abs(f.mean[0] - 3.76) < 0.15
    np.testing.assert_allclose(f.state_probs[0], [0.03, 0.97], atol=1e-6)

    # long horizon: the chain mixes to its (0.5, 0.5) stationary law -> mean ~ 0
    np.testing.assert_allclose(f.state_probs[-1], [0.5, 0.5], atol=0.01)
    assert abs(f.mean[-1]) < 0.35

    # the 90% central band is honest for the predictive shape at each horizon:
    # step 1 the switch lobe holds 3% < the 5% tail -> the band sits in the +4 regime...
    assert f.lo[0] > 0.0
    # ...at long horizon the predictive is an even bimodal -> the band must span both regimes
    assert f.lo[-1] < -3.0 and f.hi[-1] > 3.0


def test_interval_covers_simulated_continuations():
    m = _hmm(stay=0.85)
    history = [-4.1, -3.9, -4.0]
    f = forecast(m, history, horizon=5, level=0.9, n=8000, seed=1)

    # simulate true continuations from the model, starting from the filtered state
    rng = np.random.RandomState(2)
    a = np.asarray(m.transitions)
    hits = 0
    total = 0
    for _ in range(400):
        s = 0  # history pins state 0 with near-certainty
        for h in range(5):
            s = rng.choice(2, p=a[s])
            y = rng.normal(-4.0 if s == 0 else 4.0, 1.0)
            hits += int(f.lo[h] <= y <= f.hi[h])
            total += 1
    coverage = hits / total
    assert 0.85 <= coverage <= 0.97  # nominal 0.9, honest tolerance


def test_rejects_non_hmm():
    with pytest.raises(TypeError):
        forecast(GaussianDistribution(0.0, 1.0), [1.0, 2.0], horizon=3)


class _FlatTopic:
    """A minimal duck-typed topic: every history point scores with the SAME (zero) log-density
    under every state, so ``_filtered_state_posterior`` reduces to the model's own prior/transition
    structure (no real emission discrimination needed) -- lets the test below fully control the
    per-step state marginal via `w`/`transitions` alone."""

    def __init__(self, sample_fn):
        self._sample_fn = sample_fn

    def dist_to_encoder(self):
        return self

    def seq_encode(self, hist):
        return list(hist)

    def seq_log_density(self, enc):
        return np.zeros(len(enc))

    def sampler(self, seed):
        return self

    def sample(self, n):
        return self._sample_fn(n)


class _AlternatingHMM:
    """A minimal duck-typed stand-in for a fitted HMM, good enough for ``forecast()`` /
    ``_filtered_state_posterior``: state 0 emits real floats (a scalar emission), state 1 emits
    strings (never coercible to a numeric array -- a genuinely non-scalar emission). A REAL
    ``HiddenMarkovModelDistribution`` cannot mix these -- every topic must share one scalar-real
    observation domain -- so this fake is what it takes to reproduce the scalar/non-scalar MIX
    across horizon steps that ``forecast()``'s per-step flag has to get right. The transition
    matrix is a pure swap and the start state is deterministic, so the state marginal at each
    horizon step is EXACTLY state 0 or state 1 in alternation, with no Monte Carlo noise.
    """

    n_states = 2
    w = np.array([1.0, 0.0])
    transitions = np.array([[0.0, 1.0], [1.0, 0.0]])

    def __init__(self):
        self.topics = [
            _FlatTopic(lambda n: np.random.RandomState(0).normal(3.0, 1.0, size=n)),
            _FlatTopic(lambda n: [f"cat_{i}" for i in range(n)]),
        ]


def test_a_non_scalar_step_does_not_corrupt_a_later_scalar_step():
    """The per-step scalar/non-scalar decision used to latch globally via one shared flag: the
    FIRST non-scalar step set it, and every LATER step was then misrouted too regardless of its
    own outcome -- even a step whose own emission was perfectly scalar had its computed mean/lo/hi
    silently discarded in favor of raw draws. `_AlternatingHMM` alternates every horizon step
    between the scalar and non-scalar state, so step 0 is non-scalar and step 1 is scalar --
    exactly the sequence that exposes the bug.
    """
    model = _AlternatingHMM()
    f = forecast(model, [0.0], horizon=4, level=0.9, n=200, seed=0)

    # step 0: state 1 (strings) -- genuinely non-scalar
    assert f.lo[0] is None and f.hi[0] is None
    assert isinstance(f.mean[0], list)
    # step 1: state 0 (floats) -- must still be a real computed interval, not corrupted by step 0
    assert isinstance(f.mean[1], float)
    assert f.lo[1] is not None and f.hi[1] is not None
    assert f.lo[1] <= f.mean[1] <= f.hi[1]
    # steps 2/3 continue the alternation -- confirms it's not a one-off, order-dependent fluke
    assert f.lo[2] is None and isinstance(f.mean[2], list)
    assert isinstance(f.mean[3], float) and f.lo[3] is not None and f.hi[3] is not None


def _one_state_hmm():
    return HiddenMarkovModelDistribution([GaussianDistribution(4.5, 1.0)], [1.0], [[1.0]])


def test_level_outside_the_unit_interval_is_rejected_not_returned_as_an_inverted_band():
    # MXR-080-1608: only the horizon was checked. level=-0.5 makes the lower tail probability
    # (1-level)/2 = 0.75 cross the upper one (0.25), so the "90% band" came back with lo above hi --
    # returned as an ordinary Forecast, with `level` echoed back as if it meant something.
    m = _one_state_hmm()
    for bad in (-0.5, 0.0, 1.0, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="level"):
            forecast(m, [4.5, 4.5], horizon=1, level=bad, n=200)


def test_draw_count_must_be_an_exact_positive_integer():
    # n=0 first warned about an empty-slice mean and then raised an opaque IndexError partway
    # through the loop; a fractional n silently produced a forecast off a coerced draw count.
    m = _one_state_hmm()
    for bad in (0, -3, 2.5, True, float("nan")):
        with pytest.raises(ValueError, match="n must be"):
            forecast(m, [4.5, 4.5], horizon=1, n=bad)


def test_horizon_must_be_an_exact_positive_integer():
    # `horizon < 1` is False for NaN (every NaN comparison is) and for 2.5, both of which then died
    # inside range()/np.empty with a bare TypeError naming neither the argument nor forecast().
    m = _one_state_hmm()
    for bad in (0, -1, 2.5, True, float("nan")):
        with pytest.raises(ValueError, match="horizon must be"):
            forecast(m, [4.5, 4.5], horizon=bad, n=200)


def test_empty_history_is_a_domain_error_not_an_internal_index_error():
    # An otherwise valid HMM with an empty history failed inside the forward pass with
    # "IndexError: index 0 is out of bounds for axis 1 with size 0".
    m = _one_state_hmm()
    with pytest.raises(ValueError, match="history"):
        forecast(m, [], horizon=2, n=200)


def test_valid_controls_still_produce_an_ordered_band():
    m = _one_state_hmm()
    f = forecast(m, [4.5, 4.5], horizon=3, level=0.9, n=2000, seed=0)
    assert f.mean.shape == (3,)
    assert bool((f.lo <= f.hi).all())
