"""P12 (experimental) -- wake-sleep library learning over a model-structure grammar.

DreamCoder's loop, in miniature, for mixle's structure grammar:

* **WAKE** -- solve each modeling task by greedy search over a library of structure *atoms*
  (each atom is a group of basis components), scoring candidates by description length (MDL).
* **SLEEP-ABSTRACTION** -- compress the solutions: a set of primitives that recurs across many
  solved tasks is anti-unified into a single reusable *library fragment*.
* Re-solving tasks that need that motif now reaches it in ONE search step (pick the fragment)
  instead of composing it primitive by primitive -- so search cost drops.

The card's claim: on a corpus of related tasks sharing a latent motif, the wake-sleep loop
discovers the motif as a fragment and cuts held-out search cost, with the fragment reused across
the corpus. The kill criterion is measured, not assumed: if no fragment is reused enough, or the
speedup does not materialize, the receipt says so.

Exploratory ``mixle.experimental`` code (P12 card).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from mixle.utils.immutable import detach_receipt_container


@dataclass(frozen=True)
class Atom:
    """A library atom: a named group of feature-column indices (primitive = 1 col; fragment = many)."""

    name: str
    cols: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("atom name must be a non-empty string")
        if not isinstance(self.cols, tuple) or not self.cols:
            raise ValueError("atom cols must be a non-empty tuple")
        if any(isinstance(col, bool) or not isinstance(col, (int, np.integer)) or col < 0 for col in self.cols):
            raise ValueError("atom cols must contain non-negative integer column indices")
        normalized = tuple(sorted(int(col) for col in self.cols))
        if len(set(normalized)) != len(normalized):
            raise ValueError("atom cols must not contain duplicates")
        object.__setattr__(self, "cols", normalized)


def _mdl(y: np.ndarray, features: np.ndarray, cols: tuple[int, ...], penalty: float) -> float:
    """Description length of explaining ``y`` with the given feature columns (lower is better)."""
    n = len(y)
    if not cols:
        sse = float(np.sum((y - y.mean()) ** 2))
    else:
        x = features[:, list(cols)]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        sse = float(np.sum((y - x @ coef) ** 2))
    return 0.5 * n * np.log(sse / n + 1e-12) + penalty * len(cols)


@dataclass
class SearchResult:
    selected: list[str]
    used_cols: tuple[int, ...]
    n_evals: int


def greedy_search(y: np.ndarray, features: np.ndarray, library: list[Atom], *, penalty: float = 2.0) -> SearchResult:
    """Forward-selection structure search; returns the chosen atoms and the number of fit evaluations."""
    y = np.asarray(y, dtype=float)
    features = np.asarray(features, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a non-empty finite one-dimensional array")
    if features.ndim != 2 or features.shape[0] != y.size or features.shape[1] == 0:
        raise ValueError("features must be a non-empty matrix with one row per target")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    if not isinstance(library, list) or any(not isinstance(atom, Atom) for atom in library):
        raise TypeError("library must be a list of Atom objects")
    if len({atom.name for atom in library}) != len(library):
        raise ValueError("library atom names must be unique")
    if any(col >= features.shape[1] for atom in library for col in atom.cols):
        raise ValueError("library atom column is outside the feature matrix")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")

    chosen: list[Atom] = []
    used: tuple[int, ...] = ()
    current = _mdl(y, features, used, penalty)
    n_evals = 0
    remaining = list(library)
    while True:
        best_atom, best_mdl = None, current
        for atom in remaining:
            n_evals += 1
            cand = tuple(sorted(set(used) | set(atom.cols)))
            m = _mdl(y, features, cand, penalty)
            if m < best_mdl:
                best_atom, best_mdl = atom, m
        if best_atom is None:
            break
        chosen.append(best_atom)
        used = tuple(sorted(set(used) | set(best_atom.cols)))
        current = best_mdl
        remaining = [a for a in remaining if a is not best_atom]
    return SearchResult([a.name for a in chosen], used, n_evals)


def primitive_library(n_primitives: int) -> list[Atom]:
    """One atom per basis component."""
    if isinstance(n_primitives, bool) or not isinstance(n_primitives, int) or n_primitives <= 0:
        raise ValueError("n_primitives must be a positive integer")
    return [Atom(f"p{i}", (i,)) for i in range(n_primitives)]


def _validate_task_spec(
    features: np.ndarray,
    motif: tuple[int, ...],
    n_specific: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or min(features.shape) == 0 or not np.all(np.isfinite(features)):
        raise ValueError("features must be a non-empty finite matrix")
    if not isinstance(motif, tuple) or not motif:
        raise ValueError("motif must be a non-empty tuple of column indices")
    if any(isinstance(col, bool) or not isinstance(col, (int, np.integer)) for col in motif):
        raise ValueError("motif must contain integer column indices")
    motif = tuple(int(col) for col in motif)
    if len(set(motif)) != len(motif):
        raise ValueError("motif columns must be unique")
    n_primitives = features.shape[1]
    if any(col < 0 or col >= n_primitives for col in motif):
        raise ValueError("motif column is outside the feature matrix")
    if isinstance(n_specific, bool) or not isinstance(n_specific, int) or n_specific < 0:
        raise ValueError("n_specific must be a non-negative integer")
    if n_specific > n_primitives - len(motif):
        raise ValueError("n_specific exceeds the number of columns outside the motif")
    return features, motif


def make_task(rng: np.random.Generator, features: np.ndarray, motif: tuple[int, ...], n_specific: int) -> np.ndarray:
    """A regression target built from the shared motif plus a few task-specific components.

    The motif components carry a strong coefficient (so greedy reliably recovers all of them),
    the task-specific ones a weaker coefficient.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    features, motif = _validate_task_spec(features, motif, n_specific)
    n_primitives = features.shape[1]
    specific = rng.choice([i for i in range(n_primitives) if i not in motif], size=n_specific, replace=False)
    motif_coef = rng.uniform(2.0, 3.0, size=len(motif)) * rng.choice([-1.0, 1.0], size=len(motif))
    spec_coef = rng.uniform(1.0, 1.5, size=len(specific)) * rng.choice([-1.0, 1.0], size=len(specific))
    signal = features[:, list(motif)] @ motif_coef + features[:, list(specific)] @ spec_coef
    return signal + 0.1 * rng.standard_normal(len(features))


def abstract_fragment(solutions: list[tuple[int, ...]], *, min_support: float = 0.6, min_size: int = 2) -> Atom | None:
    """Return the largest exact itemset jointly present in at least ``min_support`` of solutions.

    Ties are resolved by larger exact support and then lexicographically. Every returned fragment is therefore
    a subset of at least the required number of observed solutions; individually frequent columns are never
    unioned into a fragment unless that exact union is also jointly frequent.
    """
    if not isinstance(solutions, list):
        raise TypeError("solutions must be a list of column tuples")
    if (
        isinstance(min_support, bool)
        or not isinstance(min_support, (int, float))
        or not math.isfinite(min_support)
        or not 0.0 < min_support <= 1.0
    ):
        raise ValueError("min_support must be finite and in (0, 1]")
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    if not solutions:
        return None
    normalized: list[frozenset[int]] = []
    for solution in solutions:
        if not isinstance(solution, tuple):
            raise TypeError("each solution must be a tuple of column indices")
        if any(isinstance(col, bool) or not isinstance(col, (int, np.integer)) or col < 0 for col in solution):
            raise ValueError("solution columns must be non-negative integers")
        if len(set(solution)) != len(solution):
            raise ValueError("solution columns must not contain duplicates")
        normalized.append(frozenset(int(col) for col in solution))

    required = int(math.ceil(float(min_support) * len(normalized)))

    def support(itemset: frozenset[int]) -> int:
        return sum(itemset.issubset(solution) for solution in normalized)

    universe = sorted(set().union(*normalized))
    current = {frozenset((col,)) for col in universe if support(frozenset((col,))) >= required}
    frequent: dict[frozenset[int], int] = {itemset: support(itemset) for itemset in current}
    size = 2
    while current:
        candidates: set[frozenset[int]] = set()
        ordered = sorted(current, key=lambda itemset: tuple(sorted(itemset)))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                candidate = left | right
                if len(candidate) != size:
                    continue
                if all(frozenset(subset) in current for subset in combinations(candidate, size - 1)):
                    candidates.add(candidate)
        current = {candidate for candidate in candidates if support(candidate) >= required}
        frequent.update({itemset: support(itemset) for itemset in current})
        size += 1

    eligible = [(itemset, count) for itemset, count in frequent.items() if len(itemset) >= min_size]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-len(item[0]), -item[1], tuple(sorted(item[0]))))
    cols = tuple(sorted(eligible[0][0]))
    return Atom("frag(" + ",".join(map(str, cols)) + ")", cols)


@dataclass(frozen=True)
class WakeSleepReport:
    fragment: Atom | None
    flat_evals: float  # mean held-out search evals WITHOUT the fragment
    library_evals: float  # mean held-out search evals WITH the fragment
    speedup: float
    fragment_reuse: int  # held-out solutions that used the fragment
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "history", detach_receipt_container(self.history))


def wake_sleep(
    *,
    n_train: int = 30,
    n_heldout: int = 30,
    n_primitives: int = 16,
    n_t: int = 160,
    motif: tuple[int, ...] = (2, 5, 8, 11, 3, 13, 6),
    n_specific: int = 1,
    seed: int = 0,
) -> WakeSleepReport:
    """Run one wake-sleep round and measure held-out search cost with vs without the learned fragment."""
    for value, name in (
        (n_train, "n_train"),
        (n_heldout, "n_heldout"),
        (n_primitives, "n_primitives"),
        (n_t, "n_t"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_t < n_primitives:
        raise ValueError("n_t must be at least n_primitives for the orthonormal feature design")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # Orthonormal component basis: with genuinely orthogonal columns, greedy recovers the exact
    # support (no column is redundant), so the abstraction sees the true motif in every solution.
    q, _ = np.linalg.qr(rng.standard_normal((n_t, n_primitives)))
    features = q[:, :n_primitives]
    features, motif = _validate_task_spec(features, motif, n_specific)

    prims = primitive_library(n_primitives)

    # WAKE: solve the training corpus with the primitive library; collect solutions' column sets.
    train_solutions: list[tuple[int, ...]] = []
    for _ in range(n_train):
        y = make_task(rng, features, motif, n_specific)
        train_solutions.append(greedy_search(y, features, prims).used_cols)

    # SLEEP-ABSTRACTION: compress recurring structure into a library fragment.
    fragment = abstract_fragment(train_solutions, min_support=0.6, min_size=2)
    library = prims + ([fragment] if fragment else [])

    # Measure held-out search cost with and without the fragment; count fragment reuse.
    flat_evals, lib_evals, reuse = [], [], 0
    for _ in range(n_heldout):
        y = make_task(rng, features, motif, n_specific)
        flat_evals.append(greedy_search(y, features, prims).n_evals)
        res = greedy_search(y, features, library)
        lib_evals.append(res.n_evals)
        if fragment and fragment.name in res.selected:
            reuse += 1

    flat_mean = float(np.mean(flat_evals))
    lib_mean = float(np.mean(lib_evals))
    return WakeSleepReport(
        fragment=fragment,
        flat_evals=flat_mean,
        library_evals=lib_mean,
        speedup=flat_mean / lib_mean if lib_mean else 1.0,
        fragment_reuse=reuse,
    )
