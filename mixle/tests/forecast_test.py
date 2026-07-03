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
