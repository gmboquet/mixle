"""P12 (experimental) -- wake-sleep library learning cuts search cost via learned fragments.

Receipts (the card's first experiment + kill criterion): the abstraction discovers the shared
motif as a library fragment; that fragment is reused across the corpus (>= 3, the kill threshold);
and it cuts the median held-out search cost by >= 2x versus flat search. The honest control: when
the tasks share no motif, the abstraction returns no fragment -- it does not invent structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.wake_sleep import (
    Atom,
    abstract_fragment,
    greedy_search,
    primitive_library,
    wake_sleep,
)

MOTIF = (2, 5, 8, 11, 3, 13, 6)


def test_fragment_is_the_true_motif() -> None:
    r = wake_sleep(seed=0, motif=MOTIF)
    assert r.fragment is not None
    assert set(r.fragment.cols) == set(MOTIF), f"abstracted {r.fragment.cols}, expected {MOTIF}"


def test_fragment_reuse_meets_kill_criterion() -> None:
    r = wake_sleep(seed=1, motif=MOTIF)
    assert r.fragment_reuse >= 3, f"fragment reused only {r.fragment_reuse} times (kill threshold 3)"


def test_median_search_cost_cut_by_at_least_2x() -> None:
    speedups = [wake_sleep(seed=s, motif=MOTIF).speedup for s in range(6)]
    assert np.median(speedups) >= 2.0, f"median speedup {np.median(speedups):.2f}x below the 2x target"
    assert min(speedups) >= 1.85, f"a seed regressed badly: {min(speedups):.2f}x ({speedups})"


def test_abstraction_does_not_invent_structure_without_a_motif() -> None:
    """Control: solutions that share no common column must yield no fragment."""
    solutions = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]  # disjoint, no shared column
    assert abstract_fragment(solutions, min_support=0.6, min_size=2) is None


def test_abstraction_finds_the_common_set() -> None:
    solutions = [(1, 3, 5), (1, 3, 7), (1, 3, 9), (1, 3, 2)]  # {1,3} shared, rest task-specific
    frag = abstract_fragment(solutions, min_support=0.75, min_size=2)
    assert frag is not None and set(frag.cols) == {1, 3}


def test_abstraction_never_unions_individually_frequent_but_rarely_joint_columns() -> None:
    # Each column appears in 60% of solutions, but the pair appears in only 20%.
    solutions = [(0, 1), (0,), (0,), (1,), (1,)]
    assert abstract_fragment(solutions, min_support=0.6, min_size=2) is None


def test_abstraction_selects_a_fragment_with_exact_joint_support() -> None:
    solutions = [(0, 1, 2), (0, 1), (0, 1, 3), (0, 2), (1, 3)]
    fragment = abstract_fragment(solutions, min_support=0.6, min_size=2)
    assert fragment is not None
    assert fragment.cols == (0, 1)
    assert sum(set(fragment.cols).issubset(solution) for solution in map(set, solutions)) == 3


def test_greedy_recovers_planted_structure() -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((100, 10)))
    true_cols = [1, 4, 7]
    y = q[:, true_cols] @ np.array([3.0, -2.5, 2.0]) + 0.05 * rng.standard_normal(100)
    res = greedy_search(y, q, primitive_library(10))
    assert set(res.used_cols) == set(true_cols), f"recovered {res.used_cols}, planted {true_cols}"


def test_fragment_atom_reaches_motif_in_one_step() -> None:
    """With the fragment in the library, one greedy step selects the whole motif."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((160, 16)))
    y = q[:, list(MOTIF)] @ rng.uniform(2, 3, len(MOTIF)) + 0.05 * rng.standard_normal(160)
    frag = Atom("motif", MOTIF)
    res = greedy_search(y, q, [*primitive_library(16), frag])
    assert "motif" in res.selected, "the fragment should be selected in a single step"


def test_determinism() -> None:
    a = wake_sleep(seed=3, motif=MOTIF)
    b = wake_sleep(seed=3, motif=MOTIF)
    assert (a.speedup, a.fragment_reuse, a.fragment.cols) == (b.speedup, b.fragment_reuse, b.fragment.cols)


@pytest.mark.parametrize(
    ("y", "features", "library", "message"),
    [
        (np.ones(3), np.ones((2, 1)), primitive_library(1), "one row per target"),
        (np.ones((3, 1)), np.ones((3, 1)), primitive_library(1), "one-dimensional"),
        (np.ones(3), np.ones((3, 1)), [Atom("bad", (1,))], "outside"),
        (np.ones(3), np.array([[1.0], [np.nan], [2.0]]), primitive_library(1), "finite"),
    ],
)
def test_greedy_search_validates_target_design_and_library(y, features, library, message) -> None:
    with pytest.raises(ValueError, match=message):
        greedy_search(y, features, library)


def test_wake_sleep_validates_orthonormal_design_and_task_dimensions() -> None:
    with pytest.raises(ValueError, match="at least n_primitives"):
        wake_sleep(n_t=4, n_primitives=5)
    with pytest.raises(ValueError, match="outside"):
        wake_sleep(n_primitives=4, motif=(0, 4))
    with pytest.raises(ValueError, match="n_specific"):
        wake_sleep(n_primitives=4, motif=(0, 1, 2), n_specific=2)


@pytest.mark.parametrize(("support", "size"), [(0.0, 2), (1.1, 2), (0.6, 0)])
def test_abstraction_validates_thresholds(support, size) -> None:
    with pytest.raises(ValueError):
        abstract_fragment([(0, 1)], min_support=support, min_size=size)
