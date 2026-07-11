"""P14 (experimental, speculative) -- verified component trade beats isolation.

The card's first experiment + kill criterion: verified component trade must recover >= 50% of the
oracle's (full data sharing) gain over isolation, while exchanging models only, never data -- and
because the buyer verifies each offered component on its own held-out set, a spurious component the
seller offers is rejected, not blindly adopted.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.model_economy import _mse, _verify_and_adopt, run_economy


def test_trade_recovers_at_least_half_the_oracle_gain() -> None:
    fractions = [run_economy(seed=s).recovered_fraction for s in range(6)]
    assert min(fractions) >= 0.5, f"trade recovered < 50% of the oracle gain on some seed: {fractions}"


def test_trade_beats_isolation() -> None:
    for s in range(4):
        r = run_economy(seed=s)
        assert r.trade_mse < r.isolation_mse, f"seed {s}: trade did not beat isolation"
        assert r.trade_gain > 0.0
        assert r.adopted > 0, "no components were adopted -- the market found no surplus"


def test_kill_criterion_surplus_exists() -> None:
    """There must be surplus to allocate: the oracle beats isolation, and trade captures most of it."""
    r = run_economy(seed=0)
    assert r.oracle_gain > 0.0, "no complementary surplus in the fixture"
    assert r.trade_gain >= 0.5 * r.oracle_gain


def test_buyer_rejects_a_spurious_offered_component() -> None:
    """Verification without trust: a coefficient that does not help the buyer is not adopted."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((300, 6)))
    true = np.zeros(6)
    true[1] = 2.0  # only column 1 is real
    y = q @ true + 0.05 * rng.standard_normal(300)

    buyer = np.zeros(6)
    seller = np.zeros(6)
    seller[1] = 2.0  # a genuinely useful component...
    seller[4] = 3.0  # ...and a spurious one on a true-zero column

    adopted_coef, n_adopted = _verify_and_adopt(buyer, seller, q, y)
    assert n_adopted == 1, f"buyer should adopt only the useful component, adopted {n_adopted}"
    assert np.isclose(adopted_coef[1], 2.0), "the useful component should be adopted"
    assert adopted_coef[4] == 0.0, "the spurious component must be rejected"
    assert _mse(adopted_coef, q, y) < _mse(buyer, q, y)


def test_determinism() -> None:
    a = run_economy(seed=3)
    b = run_economy(seed=3)
    assert (a.trade_mse, a.oracle_mse, a.adopted) == (b.trade_mse, b.oracle_mse, b.adopted)
