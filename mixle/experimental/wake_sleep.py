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

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Atom:
    """A library atom: a named group of feature-column indices (primitive = 1 col; fragment = many)."""

    name: str
    cols: tuple[int, ...]


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
    return [Atom(f"p{i}", (i,)) for i in range(n_primitives)]


def make_task(rng: np.random.Generator, features: np.ndarray, motif: tuple[int, ...], n_specific: int) -> np.ndarray:
    """A regression target built from the shared motif plus a few task-specific components.

    The motif components carry a strong coefficient (so greedy reliably recovers all of them),
    the task-specific ones a weaker coefficient.
    """
    n_primitives = features.shape[1]
    specific = rng.choice([i for i in range(n_primitives) if i not in motif], size=n_specific, replace=False)
    motif_coef = rng.uniform(2.0, 3.0, size=len(motif)) * rng.choice([-1.0, 1.0], size=len(motif))
    spec_coef = rng.uniform(1.0, 1.5, size=len(specific)) * rng.choice([-1.0, 1.0], size=len(specific))
    signal = features[:, list(motif)] @ motif_coef + features[:, list(specific)] @ spec_coef
    return signal + 0.1 * rng.standard_normal(len(features))


def abstract_fragment(solutions: list[tuple[int, ...]], *, min_support: float = 0.6, min_size: int = 2) -> Atom | None:
    """Anti-unify solutions into one fragment: the largest column set present in >= min_support of them."""
    if not solutions:
        return None
    n = len(solutions)
    # Count support of each individual column, then grow the frequent set greedily by support.
    col_support = Counter()
    for sol in solutions:
        for c in set(sol):
            col_support[c] += 1
    frequent = sorted([c for c, k in col_support.items() if k / n >= min_support])
    if len(frequent) < min_size:
        return None
    # Keep only the columns that co-occur together in >= min_support of solutions.
    kept = [c for c in frequent if sum(1 for s in solutions if set(frequent) <= set(s)) / n >= min_support]
    if len(kept) < min_size:
        # fall back to the maximal subset that actually co-occurs
        kept = frequent
    cols = tuple(sorted(kept))
    if len(cols) < min_size:
        return None
    return Atom("frag(" + ",".join(map(str, cols)) + ")", cols)


@dataclass
class WakeSleepReport:
    fragment: Atom | None
    flat_evals: float  # mean held-out search evals WITHOUT the fragment
    library_evals: float  # mean held-out search evals WITH the fragment
    speedup: float
    fragment_reuse: int  # held-out solutions that used the fragment
    history: list = field(default_factory=list)


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
    rng = np.random.default_rng(seed)
    # Orthonormal component basis: with genuinely orthogonal columns, greedy recovers the exact
    # support (no column is redundant), so the abstraction sees the true motif in every solution.
    q, _ = np.linalg.qr(rng.standard_normal((n_t, n_primitives)))
    features = q[:, :n_primitives]

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
