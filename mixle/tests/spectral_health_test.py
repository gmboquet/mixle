"""P16 (experimental) -- data-free spectral-health receipts discriminate training regimes.

The card's kill criterion: if the spectral exponents cannot tell an under-trained (near-random)
layer from a well-trained one from a memorizing (over-correlated) one, drop the idea. These tests
validate the receipt on matrices with KNOWN spectral laws -- constructed via SVD so the singular
spectrum is exactly the design -- and require the metrics to order and classify the three regimes
as documented, using only the weights (no data).
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.spectral_health import (
    effective_rank,
    model_spectral_report,
    singular_values,
    spectral_health,
    stable_rank,
)


def _rand_orth(n, m, rng):
    q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    return q[:, : min(n, m)]


def _with_spectrum(sv, *, n=512, m=256, seed=0):
    """A matrix whose singular values are exactly ``sv`` (random orthogonal U, V)."""
    rng = np.random.default_rng(seed)
    return _rand_orth(n, min(n, m), rng) @ np.diag(sv) @ _rand_orth(m, min(n, m), rng).T


def _regimes():
    idx = np.arange(1, 257, dtype=float)
    rng = np.random.default_rng(0)
    under = rng.standard_normal((512, 256))  # iid Gaussian -> Marchenko-Pastur bulk
    well = _with_spectrum(idx**-0.35, seed=1)  # mild power-law tail
    sv = idx**-1.3
    sv[0] *= 6.0
    sv[1] *= 3.0
    memorizing = _with_spectrum(sv, seed=2)  # steep tail + planted spikes
    return under, well, memorizing


def test_tail_exponent_orders_the_three_regimes() -> None:
    under, well, memorizing = _regimes()
    a_under = spectral_health(under).alpha
    a_well = spectral_health(well).alpha
    a_mem = spectral_health(memorizing).alpha
    # Under-trained (near-random) has the lightest tail; memorizing the heaviest.
    assert a_under > a_well > a_mem, f"alphas not ordered: {a_under:.2f}, {a_well:.2f}, {a_mem:.2f}"
    # And the well-trained layer sits in the documented [2, 4] band.
    assert 2.0 <= a_well <= 4.0, f"well-trained alpha {a_well:.2f} left the [2,4] band"


def test_verdicts_match_the_documented_bands() -> None:
    under, well, memorizing = _regimes()
    assert spectral_health(under).verdict == "under-trained"
    assert spectral_health(well).verdict == "well-trained"
    assert spectral_health(memorizing).verdict == "memorizing"


def test_kill_criterion_random_is_distinguished_from_trained() -> None:
    """The receipt must separate a noise/random layer from a trained one by a wide margin."""
    under, well, _ = _regimes()
    gap = spectral_health(under).alpha - spectral_health(well).alpha
    assert gap > 3.0, f"random vs trained alpha gap only {gap:.2f} -- not discriminative"


def test_stable_and_effective_rank_track_correlation() -> None:
    under, well, memorizing = _regimes()
    # A near-random matrix is close to full-rank; a heavy-tailed one collapses toward rank 1.
    sr = [stable_rank(singular_values(w)) for w in (under, well, memorizing)]
    assert sr[0] > sr[1] > sr[2], f"stable rank not ordered: {sr}"
    assert stable_rank(singular_values(memorizing)) < 3.0
    er = [effective_rank(singular_values(w)) for w in (under, well, memorizing)]
    assert er[0] > er[2], "effective rank should be highest for the random layer"


def test_receipt_is_data_free_and_deterministic() -> None:
    well = _with_spectrum(np.arange(1, 257, dtype=float) ** -0.35, seed=1)
    r1 = spectral_health(well)  # note: only the weight matrix is passed -- no data
    r2 = spectral_health(well)
    assert r1.as_dict() == r2.as_dict()
    assert set(r1.as_dict()) >= {"alpha", "ks_d", "stable_rank", "effective_rank", "verdict"}


def test_power_law_fit_is_good_on_structured_spectra() -> None:
    well = _with_spectrum(np.arange(1, 257, dtype=float) ** -0.35, seed=3)
    assert spectral_health(well).ks_d < 0.1, "power-law tail fit should be tight on a power-law spectrum"


def test_model_report_over_named_layers() -> None:
    idx = np.arange(1, 257, dtype=float)
    report = model_spectral_report(
        {
            "layer.0": _with_spectrum(idx**-0.35, seed=1),
            "layer.1": _with_spectrum(idx**-1.3, seed=2),
        }
    )
    assert set(report) == {"layer.0", "layer.1"}
    assert report["layer.0"].verdict == "well-trained"
    assert report["layer.1"].verdict == "memorizing"


def test_handles_small_and_degenerate_matrices() -> None:
    # Too few singular values to fit a tail -> alpha is inf, verdict falls back to under-trained.
    tiny = spectral_health(np.eye(3))
    assert tiny.verdict == "under-trained"
    assert tiny.stable_rank > 0
