"""P5 (experimental) -- exact unlearning certificates for closed-form leaves.

The certified operation is *re-reduce the retained shards' stored statistics*, not *subtract the
deleted shard's statistics*. These tests pin both halves of the card's finding:

  * re-reduce yields the never-saw-it fit bit-for-bit across closed-form families;
  * subtraction is neither bitwise nor even safe -- under an adversarial large-magnitude shard it
    catastrophically cancels and returns a negative variance, while re-reduce stays exact.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.unlearning import (
    certify_unlearning,
    shard_statistic,
    unlearn,
)
from mixle.inference.estimation import optimize
from mixle.stats import CategoricalEstimator, GaussianEstimator, PoissonEstimator


def _shards(rng, kind):
    if kind == "gaussian":
        return [rng.normal(3.0, 2.0, 30).tolist() for _ in range(4)]
    if kind == "categorical":
        return [rng.choice(list("abcd"), 25).tolist() for _ in range(4)]
    return [rng.poisson(4.0, 30).tolist() for _ in range(4)]


def test_certificate_is_bitwise_exact_for_closed_form_leaves() -> None:
    for kind, est in [
        ("gaussian", GaussianEstimator()),
        ("categorical", CategoricalEstimator()),
        ("poisson", PoissonEstimator()),
    ]:
        rng = np.random.default_rng(0)
        shards = _shards(rng, kind)
        _, cert = certify_unlearning(est, shards, exclude={1})
        assert cert.bitwise_exact, f"{kind}: re-reduce unlearning was not bitwise exact"
        assert cert.method == "re-reduce"
        assert cert.n_excluded == 1 and cert.n_retained_shards == 3 and cert.n_shards_total == 4


def test_unlearn_matches_the_never_saw_it_fit() -> None:
    rng = np.random.default_rng(1)
    shards = _shards(rng, "gaussian")
    stored = [shard_statistic(GaussianEstimator(), s) for s in shards]
    unlearned = unlearn(GaussianEstimator(), stored, exclude={2})
    # Reference: fit from scratch on the concatenation of the retained shards.
    retained = [x for i, s in enumerate(shards) if i != 2 for x in s]
    scratch = optimize(retained, GaussianEstimator(), out=None)
    assert np.isclose(unlearned.mu, scratch.mu)
    assert np.isclose(unlearned.sigma2, scratch.sigma2)


def test_subtraction_is_not_the_certified_method_and_can_go_invalid() -> None:
    """Adversarial shard: subtract catastrophically cancels to a negative variance; re-reduce is fine."""
    # Retained shards: small, well-conditioned. Excluded shard: huge magnitude.
    rng = np.random.default_rng(7)
    retained = [rng.normal(0.0, 1.0, 200) for _ in range(3)]
    excluded = rng.normal(1e10, 1.0, 200)  # huge magnitude -> Q's ULP swamps the retained sum-of-squares

    def stats(x):
        x = np.asarray(x, dtype=float)
        return float(x.size), float(x.sum()), float((x * x).sum())

    # total over ALL shards, then subtract the excluded shard (the WRONG method).
    all_shards = [*retained, excluded]
    N = sum(stats(s)[0] for s in all_shards)
    S = sum(stats(s)[1] for s in all_shards)
    Q = sum(stats(s)[2] for s in all_shards)
    n_e, s_e, q_e = stats(excluded)
    n_sub, s_sub, q_sub = N - n_e, S - s_e, Q - q_e
    mu_sub = s_sub / n_sub
    var_subtract = q_sub / n_sub - mu_sub**2

    # re-reduce over only the retained shards (the CERTIFIED method).
    n_r = sum(stats(s)[0] for s in retained)
    s_r = sum(stats(s)[1] for s in retained)
    q_r = sum(stats(s)[2] for s in retained)
    mu_re = s_r / n_r
    var_rereduce = q_r / n_r - mu_re**2

    # Re-reduce recovers the true retained variance (~1). Subtraction catastrophically cancels:
    # the excluded shard's ~1e20 sum-of-squares has a ULP far larger than the retained ~600, so the
    # subtracted result is dominated by rounding noise -- materially wrong, and here negative.
    assert 0.5 < var_rereduce < 2.0, f"re-reduce variance {var_rereduce:.3f} should be ~1"
    assert var_subtract < 0.0 or abs(var_subtract - var_rereduce) > 0.5, (
        f"subtraction should be materially wrong; got {var_subtract:.3e} vs re-reduce {var_rereduce:.3f}"
    )


def test_per_user_unlearning_of_multiple_shards() -> None:
    rng = np.random.default_rng(3)
    shards = _shards(rng, "gaussian")
    _, cert = certify_unlearning(GaussianEstimator(), shards, exclude={0, 3})
    assert cert.bitwise_exact
    assert cert.n_excluded == 2 and cert.n_retained_shards == 2


def test_determinism() -> None:
    rng = np.random.default_rng(5)
    shards = _shards(rng, "gaussian")
    m1, c1 = certify_unlearning(GaussianEstimator(), shards, exclude={1})
    m2, c2 = certify_unlearning(GaussianEstimator(), shards, exclude={1})
    assert m1.to_json() == m2.to_json()
    assert c1.as_dict() == c2.as_dict()
