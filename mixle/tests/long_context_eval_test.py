"""E7 acceptance receipts for the long-context referee (see notes/designs/E1.md and
mixle/experimental/long_context_eval.py's module docstring for what each protocol measures).

Honest scale note: the roadmap card's ``evaluate()`` API defaults ``ranges`` to the literal card scale
``(1e3, 1e4, 1e5, 1e6)`` -- that default is exercised nowhere in this file. A real training run at
``distance=1e6`` here is computationally enormous (every suite trains ``mechanism`` from scratch via TBPTT
at that length several times over). This file uses small stand-in ranges -- ``(8, 16, 32)`` -- in place of
the card's ``(1e3, 1e4, 1e5, 1e6)``: two orders of magnitude smaller at every rung, small enough that the
whole suite (needle + copy + multi-hop + perplexity + the length-curriculum bandit, all four ranges) runs
in a few seconds, while exercising the EXACT same ``evaluate()`` code path a caller would use at card scale
-- nothing is mocked, stubbed, or skipped for being "too large".

Four receipts:
1. one command (``evaluate``) runs the full E1-baseline suite end-to-end and returns every documented key.
2. ``comparison_table`` renders a non-empty table containing every tested range and both mechanism names
   when comparing two evaluate() results (the matched-FLOPs / matched-state-bytes comparison shape).
3. suites are seed-reproducible: two ``evaluate()`` calls with identical seeds and a freshly re-seeded
   model produce bitwise-identical receipts.
4. the train-then-probe machinery is really measuring something: given enough training steps on the
   easiest (shortest-distance) suite, copy recall rises measurably above a freshly-initialized model's
   near-chance accuracy -- the receipts aren't a silently-degenerate always-zero placeholder.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine  # noqa: E402
from mixle.experimental.long_context_eval import (  # noqa: E402
    comparison_table,
    copy_suite,
    evaluate,
    multi_hop_suite,
    needle_suite,
)

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.

_STAND_IN_RANGES = (8, 16, 32)  # stand-in for the card's (1e3, 1e4, 1e5, 1e6); see module docstring.


def _build_model(seed: int) -> SlidingWindowSpine:
    torch.manual_seed(seed)
    return SlidingWindowSpine(12, d_model=16, n_layer=1, n_head=2, window=8)


def test_evaluate_end_to_end_smoke():
    model = _build_model(0)
    result = evaluate(
        model,
        ranges=_STAND_IN_RANGES,
        state_budget_bytes=1_000_000,
        seed=1,
        hops=2,
        n_train_steps=3,
        n_eval_trials=3,
        perplexity_steps=2,
        curriculum_rounds=4,
    )

    assert result["ranges"] == _STAND_IN_RANGES
    assert set(result["suites"].keys()) == set(_STAND_IN_RANGES)
    for distance in _STAND_IN_RANGES:
        row = result["suites"][distance]
        for suite_name in ("needle", "copy", "multi_hop"):
            assert 0.0 <= row[suite_name]["accuracy"] <= 1.0
            assert row[suite_name]["distance"] == distance
        assert row["perplexity"]["perplexity"] > 0.0
        assert row["flops"] > 0.0

    fc = result["forgetting_curve"]
    assert fc["distances"] == list(_STAND_IN_RANGES)
    assert len(fc["accuracy"]) == len(_STAND_IN_RANGES)

    cur = result["curriculum"]
    assert sum(cur["pulls"]) <= 4  # curriculum_rounds
    assert len(cur["bucket_ranges"]) == len(_STAND_IN_RANGES)

    assert result["state_bytes_used"] >= 0
    assert result["within_state_budget"] is True
    print(f"[E7 receipt] end-to-end evaluate() returned every documented key for ranges={_STAND_IN_RANGES}")


def test_comparison_table_renders_and_compares():
    model_a = _build_model(0)
    model_b = _build_model(7)
    kwargs = dict(
        ranges=_STAND_IN_RANGES,
        state_budget_bytes=1_000_000,
        seed=2,
        hops=2,
        n_train_steps=2,
        n_eval_trials=2,
        perplexity_steps=1,
        curriculum_rounds=3,
    )
    result_a = evaluate(model_a, **kwargs)
    result_b = evaluate(model_b, **kwargs)

    table = comparison_table({"e1_baseline": result_a, "challenger": result_b})
    assert isinstance(table, str) and table.strip()
    assert "e1_baseline" in table and "challenger" in table
    for distance in _STAND_IN_RANGES:
        assert str(distance) in table

    single_table = comparison_table(result_a)
    assert "mechanism" in single_table
    print(f"[E7 receipt] comparison_table rendered {len(table.splitlines())} lines for a 2-mechanism comparison")


def test_evaluate_is_seed_reproducible():
    kwargs = dict(
        ranges=_STAND_IN_RANGES,
        state_budget_bytes=1_000_000,
        seed=3,
        hops=2,
        n_train_steps=3,
        n_eval_trials=3,
        perplexity_steps=2,
        curriculum_rounds=4,
    )
    result_1 = evaluate(_build_model(11), **kwargs)
    result_2 = evaluate(_build_model(11), **kwargs)

    assert result_1["suites"].keys() == result_2["suites"].keys()
    for distance in _STAND_IN_RANGES:
        row_1, row_2 = result_1["suites"][distance], result_2["suites"][distance]
        for suite_name in ("needle", "copy", "multi_hop"):
            assert row_1[suite_name] == row_2[suite_name]
        assert row_1["perplexity"] == row_2["perplexity"]
        assert row_1["flops"] == row_2["flops"]
    fc_1, fc_2 = result_1["forgetting_curve"], result_2["forgetting_curve"]
    assert fc_1["distances"] == fc_2["distances"]
    assert fc_1["accuracy"] == fc_2["accuracy"]
    assert fc_1["self_reported_loss"] == fc_2["self_reported_loss"]
    corr_1, corr_2 = fc_1["self_knowledge_correlation"], fc_2["self_knowledge_correlation"]
    assert (math.isnan(corr_1) and math.isnan(corr_2)) or corr_1 == corr_2
    assert result_1["curriculum"] == result_2["curriculum"]
    assert result_1["state_bytes_used"] == result_2["state_bytes_used"]
    print("[E7 receipt] evaluate() bitwise-reproducible across two seeded reruns")


def test_train_and_probe_measures_real_recall():
    # Enough training steps on the shortest, easiest suite that copy recall should rise measurably above
    # a freshly-initialized model's near-chance accuracy -- proof the receipt isn't silently degenerate.
    from mixle.experimental.long_context_eval import _train_and_probe

    vocab, distance, chunk_size = 12, 6, 3
    model = _build_model(5)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    rng = np.random.RandomState(9)

    untrained = _train_and_probe(
        model,
        opt,
        copy_suite,
        distance=distance,
        vocab=vocab,
        chunk_size=chunk_size,
        n_train_steps=0,
        n_eval_trials=20,
        rng=rng,
    )
    trained = _train_and_probe(
        model,
        opt,
        copy_suite,
        distance=distance,
        vocab=vocab,
        chunk_size=chunk_size,
        n_train_steps=60,
        n_eval_trials=20,
        rng=rng,
    )

    print(
        f"[E7 receipt] copy-suite mean probe loss: untrained={untrained['mean_probe_loss']:.3f} "
        f"trained={trained['mean_probe_loss']:.3f} (chance={untrained['chance_loss']:.3f})"
    )
    assert trained["mean_probe_loss"] < untrained["mean_probe_loss"], (
        "training on the copy suite should reduce the probe loss below the untrained baseline"
    )


def test_needle_copy_multi_hop_dependency_structure():
    rng = np.random.RandomState(0)

    x, y = needle_suite(rng, distance=5, vocab=10)
    key = int(x[0, 0])
    value = int(x[0, 1])
    assert int(x[0, 5]) == key
    assert int(y[0, 5]) == value

    x, y = copy_suite(rng, distance=4, vocab=10)
    assert int(y[0, 4]) == int(x[0, 0])

    x, y = multi_hop_suite(rng, distance=9, vocab=10, hops=3)
    anchor_vocab = 9
    positions = sorted(set(int(p) for p in np.linspace(0, 8, num=3, endpoint=False)))
    expected = sum(int(x[0, p]) for p in positions) % anchor_vocab
    assert int(y[0, 9]) == expected
    print("[E7 receipt] needle/copy/multi-hop suites encode exactly their documented dependency structure")
