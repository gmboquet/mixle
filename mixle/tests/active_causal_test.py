"""P15 (experimental) -- active causal discovery via EIG over a structure posterior.

The card's kill criterion: EIG-designed interventions must beat random by >= 20%
experiments-to-identification, or fall back to random. These tests measure that margin on the
chain/reverse/fork Markov-equivalence triple, confirm the posterior converges to the true
structure, and confirm observation-only is materially slower and less reliable -- i.e. the
interventions are what pay.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.active_causal import (
    StructurePosterior,
    active_discovery,
    default_regimes,
    expected_information_gain,
    markov_equivalent_triple,
)

WEIGHT, N_BATCH, THR = 0.5, 2, 0.95


def _run(strategy, *, seeds, true_idx=0, max_experiments=60):
    cands = markov_equivalent_triple(weight=WEIGHT)
    ns, correct = [], 0
    for s in seeds:
        r = active_discovery(
            cands[true_idx],
            cands,
            strategy=strategy,
            n_batch=N_BATCH,
            threshold=THR,
            seed=s,
            max_experiments=max_experiments,
        )
        ns.append(r.n_experiments)
        correct += int(r.correct)
    return float(np.mean(ns)), correct


def test_eig_beats_random_by_more_than_20_percent() -> None:
    seeds = range(30)
    eig_mean, eig_correct = _run("eig", seeds=seeds)
    rand_mean, _ = _run("random", seeds=seeds)
    assert eig_mean <= 0.8 * rand_mean, (
        f"EIG must beat random by >=20%: eig={eig_mean:.2f} vs random={rand_mean:.2f} "
        f"({100 * (1 - eig_mean / rand_mean):.0f}% fewer)"
    )
    assert eig_correct >= 27, f"EIG should identify the true structure reliably: {eig_correct}/30"


def test_observation_only_is_materially_slower_and_less_reliable() -> None:
    """Interventions are what pay: obs-only lags far behind EIG in speed and correctness."""
    seeds = range(30)
    eig_mean, eig_correct = _run("eig", seeds=seeds)
    obs_mean, obs_correct = _run("obs", seeds=seeds)
    assert obs_mean > 2.0 * eig_mean, f"obs-only should be much slower: obs={obs_mean:.1f} eig={eig_mean:.1f}"
    assert obs_correct <= eig_correct, "observation alone should not orient more reliably than EIG"


def test_posterior_converges_to_the_true_structure() -> None:
    cands = markov_equivalent_triple(weight=WEIGHT)
    for true_idx in range(3):
        r = active_discovery(cands[true_idx], cands, strategy="eig", n_batch=N_BATCH, seed=1, max_experiments=60)
        assert r.identified == true_idx, f"failed to identify {cands[true_idx].name}"
        assert r.final_probs[true_idx] >= THR


def test_eig_prefers_a_discriminating_intervention_over_observation() -> None:
    """From the uniform prior, at least one do() regime must out-score pure observation."""
    cands = markov_equivalent_triple(weight=WEIGHT)
    post = StructurePosterior(cands)
    rng = np.random.default_rng(0)
    regimes = default_regimes(3)
    eigs = [expected_information_gain(post, r, n_batch=N_BATCH, rng=rng, n_outcomes=8) for r in regimes]
    obs_eig = eigs[0]  # regimes[0] is None (observation)
    best_do = max(eigs[1:])
    assert best_do > obs_eig, f"an intervention should be more informative than observation: {eigs}"


def test_determinism() -> None:
    cands = markov_equivalent_triple(weight=WEIGHT)
    r1 = active_discovery(cands[0], cands, strategy="eig", n_batch=N_BATCH, seed=5)
    r2 = active_discovery(cands[0], cands, strategy="eig", n_batch=N_BATCH, seed=5)
    assert (r1.identified, r1.n_experiments, r1.final_probs) == (r2.identified, r2.n_experiments, r2.final_probs)
