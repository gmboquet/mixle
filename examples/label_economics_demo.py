"""Expert-label economics: the ``acquire()`` receipt as a runnable, standalone artifact.

A5's ``mixle.task.acquire.acquire(pool, model, k, strategy="eig", ...)`` ranks an unlabeled pool by
expected information gain (BALD) against an ensemble of scoreable models. ``mixle/tests/task_acquire_test.py``
already proves the underlying claim as a single test assertion (EIG-ranked labeling reaches a target
held-out likelihood using measurably fewer labels than random, ~4x in that PR's own report). This example
is that same claim, presented as something a human deciding "how many expert labels do I actually need to
buy" can run and read directly: a budgeted labeling loop that ends with an explicit receipt --

    "N_eig EIG-chosen labels matched k * N_eig random labels"

-- plus a small table of held-out likelihood vs. label count for both strategies, so the comparison reads
as a curve rather than a single ratio number.

**Dataset.** B1 (a synthetic captioned-volume multimodal dataset) is being built concurrently and was not
available in this worktree at the time this example was written. Rather than block on it, this demo mirrors
``task_acquire_test.py``'s own synthetic setup directly: a noisy-threshold classification task
(``y = 1{x > theta_true}``, label noise ``EPS_TRUE``) with a small bootstrap ensemble of ``StumpModel``
threshold classifiers as the scoreable model family -- the textbook case where active learning has a real,
large advantage over random sampling, since only pool points near the (unknown) threshold are informative
about where it is. Swapping in a B1 dataset later (once it exists) is a drop-in replacement for
``make_task`` below, not a rewrite of the budgeted loop.

**Plot.** The roadmap item's wording ("receipt plot/table") treats a matplotlib plot as nice-to-have.
matplotlib is not a dependency anywhere else in this repo (not in ``pyproject.toml``, not imported by any
existing module or example), so this demo does not add it just for one example -- it produces the table
only, which carries the same information (held-out likelihood vs. label count, both strategies) without a
new dependency.

Run: ``python examples/label_economics_demo.py``
"""

from __future__ import annotations

import numpy as np

from mixle.task.acquire import acquire

# --- synthetic noisy-threshold task (mirrors mixle/tests/task_acquire_test.py) ----------------------
#
# y = 1{x > theta_true}, flipped with probability EPS_TRUE. The scoreable model family is a small
# bootstrap ensemble of StumpModel members (a noisy-threshold classifier fit by grid MLE) -- the
# discrete weighted hypothesis-set shape acquire()'s "eig" strategy expects.

THETA_TRUE = 0.3
EPS_TRUE = 0.05
EPS_MODEL = 0.1


def _true_p1(x: np.ndarray) -> np.ndarray:
    return np.where(x > THETA_TRUE, 1.0 - EPS_TRUE, EPS_TRUE)


def _teacher(x: float, rng: np.random.RandomState) -> int:
    return int(rng.uniform() < _true_p1(np.asarray(x))[()])


class StumpModel:
    """p(y=1|x) = 1-eps if x>t else eps; ``t`` is fit from labeled data by grid MLE."""

    def __init__(self, t: float = 0.0, eps: float = EPS_MODEL) -> None:
        self.t = t
        self.eps = eps

    def fit(self, xs: np.ndarray, ys: np.ndarray) -> StumpModel:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        uniq = np.unique(xs)
        mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else uniq
        cands = np.concatenate([[uniq.min() - 1.0], mids, [uniq.max() + 1.0]]) if uniq.size else np.array([0.0])
        best_t, best_ll = float(cands[0]), -np.inf
        for t in cands:
            p1 = np.where(xs > t, 1 - self.eps, self.eps)
            p_true = np.where(ys == 1, p1, 1 - p1)
            ll = float(np.sum(np.log(np.clip(p_true, 1e-12, 1.0))))
            if ll > best_ll:
                best_ll, best_t = ll, float(t)
        self.t = best_t
        return self

    def predict_proba(self, items):
        xs = np.asarray(items, dtype=np.float64)
        p1 = np.where(xs > self.t, 1 - self.eps, self.eps)
        return np.stack([1 - p1, p1], axis=1)


class Ensemble:
    """The lighter duck-typed ensemble shape ``acquire``'s dispatch accepts directly (``members`` +
    optional ``weights``)."""

    def __init__(self, members: list) -> None:
        self.members = members
        self.weights = np.full(len(members), 1.0 / len(members))


def _fit_ensemble(xs, ys, rng: np.random.RandomState, n_members: int = 20) -> Ensemble:
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    members = [StumpModel().fit(xs[idx], ys[idx]) for idx in (rng.randint(0, n, size=n) for _ in range(n_members))]
    return Ensemble(members)


def _held_out_ll(ensemble: Ensemble, xs_ho, ys_ho) -> float:
    proba = np.zeros((len(xs_ho), 2))
    for m in ensemble.members:
        proba += m.predict_proba(xs_ho)
    proba /= len(ensemble.members)
    p_true = np.where(np.asarray(ys_ho) == 1, proba[:, 1], proba[:, 0])
    return float(np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def make_task(pool_size: int = 150, ho_size: int = 600, pool_seed: int = 0, ho_seed: int = 999):
    """Build a fresh (pool_x, pool_y, ho_x, ho_y) noisy-threshold labeling task."""
    rng = np.random.RandomState(pool_seed)
    pool_x = list(rng.uniform(-3, 3, size=pool_size))
    pool_y = [_teacher(x, rng) for x in pool_x]

    ho_rng = np.random.RandomState(ho_seed)
    ho_x = list(ho_rng.uniform(-3, 3, size=ho_size))
    ho_y = [_teacher(x, ho_rng) for x in ho_x]
    return pool_x, pool_y, ho_x, ho_y


# --- the budgeted labeling loop ----------------------------------------------------------------------


def budget_curve(
    pool_x,
    pool_y,
    ho_x,
    ho_y,
    seed_size: int,
    strategy: str,
    master_seed: int,
    budgets: list[int],
    batch: int = 1,
    n_members: int = 20,
) -> dict[int, float]:
    """Label ``pool`` under ``strategy`` ("eig" or "random"), refitting the ensemble each round, and
    record held-out log-likelihood at each budget checkpoint in ``budgets``."""
    rng = np.random.RandomState(master_seed)
    remaining = list(range(len(pool_x)))
    rng.shuffle(remaining)
    chosen, remaining = remaining[:seed_size], remaining[seed_size:]
    xs = [pool_x[i] for i in chosen]
    ys = [pool_y[i] for i in chosen]
    results: dict[int, float] = {}
    ensemble = _fit_ensemble(xs, ys, rng, n_members=n_members)
    if seed_size in budgets:
        results[seed_size] = _held_out_ll(ensemble, ho_x, ho_y)
    while len(xs) < max(budgets) and remaining:
        cand_x = [pool_x[i] for i in remaining]
        if strategy == "random":
            pick_local = list(range(min(batch, len(remaining))))
        else:
            picked_items = acquire(cand_x, ensemble, min(batch, len(remaining)), strategy=strategy)
            pick_local = [cand_x.index(p) for p in picked_items]
        picked = [remaining[j] for j in pick_local]
        remaining = [i for j, i in enumerate(remaining) if j not in set(pick_local)]
        xs += [pool_x[i] for i in picked]
        ys += [pool_y[i] for i in picked]
        ensemble = _fit_ensemble(xs, ys, rng, n_members=n_members)
        if len(xs) in budgets:
            results[len(xs)] = _held_out_ll(ensemble, ho_x, ho_y)
    return results


def _smallest_reaching(curve: dict[int, float], budgets: list[int], target: float) -> int | None:
    return next((b for b in budgets if curve.get(b, -np.inf) >= target), None)


def run_demo(
    pool_size: int = 150,
    ho_size: int = 600,
    seed_size: int = 6,
    budgets: list[int] | None = None,
    target: float = -0.25,
    n_random_seeds: int = 5,
    n_members: int = 20,
) -> dict:
    """Run the budgeted labeling loop for both strategies and return the receipt.

    Returns a dict with the eig curve, the seed-averaged random curve, ``n_eig``/``n_random`` (smallest
    budget in ``budgets`` at which each strategy's held-out log-likelihood reaches ``target``, or
    ``None`` if never reached), and ``ratio`` (``n_random / n_eig``, or ``None`` if either is ``None``).
    Factored out of ``main()`` so tests (and other callers) can run the core loop directly at whatever
    scale they need, without going through ``__main__`` script execution.
    """
    if budgets is None:
        budgets = list(range(seed_size, seed_size + 25))
    pool_x, pool_y, ho_x, ho_y = make_task(pool_size=pool_size, ho_size=ho_size)

    eig_curve = budget_curve(
        pool_x, pool_y, ho_x, ho_y, seed_size, "eig", master_seed=1, budgets=budgets, n_members=n_members
    )

    random_curves = [
        budget_curve(
            pool_x, pool_y, ho_x, ho_y, seed_size, "random", master_seed=100 + s, budgets=budgets, n_members=n_members
        )
        for s in range(n_random_seeds)
    ]
    random_avg = {
        b: float(np.mean([c[b] for c in random_curves if b in c]))
        for b in budgets
        if any(b in c for c in random_curves)
    }

    n_eig = _smallest_reaching(eig_curve, budgets, target)
    n_random = _smallest_reaching(random_avg, budgets, target)
    ratio = (n_random / n_eig) if (n_eig and n_random) else None

    return {
        "budgets": budgets,
        "target": target,
        "eig_curve": eig_curve,
        "random_curve": random_avg,
        "n_eig": n_eig,
        "n_random": n_random,
        "ratio": ratio,
    }


def print_report(result: dict) -> None:
    budgets, target = result["budgets"], result["target"]
    eig_curve, random_curve = result["eig_curve"], result["random_curve"]
    n_eig, n_random, ratio = result["n_eig"], result["n_random"], result["ratio"]

    print(f"target held-out log-likelihood: {target}")
    print(f"{'labels':>8}  {'eig ll':>10}  {'random ll':>10}")
    checkpoints = sorted(set(eig_curve) | set(random_curve))
    step = max(1, len(checkpoints) // 8)  # a handful of rows, not every single budget
    shown = sorted(set(checkpoints[::step]) | {b for b in (n_eig, n_random) if b is not None})
    for b in shown:
        eig_v = f"{eig_curve[b]:.4f}" if b in eig_curve else "--"
        rnd_v = f"{random_curve[b]:.4f}" if b in random_curve else "--"
        print(f"{b:>8}  {eig_v:>10}  {rnd_v:>10}")

    print()
    if n_eig is None or n_random is None:
        print("receipt: target not reached by one or both strategies within budget -- widen `budgets`.")
        return
    print(f"receipt: {n_eig} EIG-chosen labels matched {ratio:.2f}x{n_eig} (={n_random}) random labels")
    print(f"         N_eig={n_eig}  N_random={n_random}  N_random/N_eig={ratio:.2f}x")


def main() -> None:
    result = run_demo()
    print_report(result)


if __name__ == "__main__":
    main()
