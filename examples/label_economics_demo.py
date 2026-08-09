"""Expert-label economics: the ``acquire()`` receipt as a runnable, standalone artifact.

``mixle.task.acquire.acquire(pool, model, k, strategy="eig", ...)`` ranks an unlabeled pool by
expected information gain (BALD) against an ensemble of scoreable models. ``mixle/tests/task_acquire_test.py``
already proves the underlying claim as a single test assertion (EIG-ranked labeling reaches a target
held-out likelihood with no more labels than the seeded random baseline in its bounded fixture). This example
is that same claim, presented as something a human deciding "how many expert labels do I actually need to
buy" can run and read directly: a budgeted labeling loop that ends with an explicit PAIRED receipt --
the per-seed win/tie/loss record with its exact sign test -- plus a small table of held-out likelihood
vs. label count for one paired seed, so the comparison reads as a curve and a paired record, never a
single pooled ratio (STAT-RR19-01/STAT-RR21-05).

**Takeaway.** Which points you pay to label is a modeling decision. The evidence printed below is
PAIRED: each master seed runs BOTH strategies on the same pool, and the receipt is the per-seed
win/tie/loss record with an exact sign test -- "often no worse and sometimes better" is what this
synthetic task supports. No pooled multiple is reported: an earlier revision printed a pooled
label-requirement multiple by comparing one EIG seed against the average of five different random
seeds, and the paired replay (18 jointly-reaching seeds: 11 wins, 7 ties, 0 losses; seed 1 itself
tied 7 vs 7) does not support a several-times label-requirement estimate (STAT-RR19-01).

**Dataset.** Deliberately synthetic, mirroring ``task_acquire_test.py``'s own setup: a noisy-threshold
classification task (``y = 1{x > theta_true}``, label noise ``EPS_TRUE``) with a small bootstrap
ensemble of ``StumpModel`` threshold classifiers as the scoreable model family -- a case where pool
points near the unknown threshold are the informative ones, so the ranking has something real to find.
A real dataset drops into ``make_task`` below without touching the budgeted loop.

**Output format.** A table, not a plot: matplotlib is not a dependency anywhere else in this repo (not
in ``pyproject.toml``, not imported by any other module or example), and held-out likelihood vs. label
count for both strategies reads perfectly well as columns.

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
    n_seeds: int = 12,
    n_members: int = 20,
) -> dict:
    """Run the budgeted labeling loop PAIRED -- both strategies under each master seed.

    The estimand is per-seed: on the same pool with the same initial labels and RNG stream, does
    EIG reach the target held-out likelihood with fewer bought labels than random selection? The
    receipt is the win/tie/loss record over the seeds where both strategies reach the target,
    with the exact one-sided sign test over the discordant seeds. An earlier revision compared
    ONE EIG seed against the pointwise AVERAGE of five different random seeds and printed the
    result as a pooled label-requirement multiple; the paired design is the honest estimand and
    it supports a qualitative claim only (STAT-RR19-01). Factored out of ``main()`` so tests can
    run the loop at whatever scale they need.
    """
    if budgets is None:
        budgets = list(range(seed_size, seed_size + 25))
    pool_x, pool_y, ho_x, ho_y = make_task(pool_size=pool_size, ho_size=ho_size)

    per_seed = []
    first_pair: tuple[dict, dict] | None = None
    for master_seed in range(1, n_seeds + 1):
        eig_curve = budget_curve(
            pool_x, pool_y, ho_x, ho_y, seed_size, "eig", master_seed=master_seed, budgets=budgets, n_members=n_members
        )
        random_curve = budget_curve(
            pool_x,
            pool_y,
            ho_x,
            ho_y,
            seed_size,
            "random",
            master_seed=master_seed,
            budgets=budgets,
            n_members=n_members,
        )
        if first_pair is None:
            first_pair = (eig_curve, random_curve)
        per_seed.append(
            {
                "seed": master_seed,
                "n_eig": _smallest_reaching(eig_curve, budgets, target),
                "n_random": _smallest_reaching(random_curve, budgets, target),
            }
        )

    joint = [row for row in per_seed if row["n_eig"] is not None and row["n_random"] is not None]
    wins = sum(row["n_eig"] < row["n_random"] for row in joint)
    losses = sum(row["n_eig"] > row["n_random"] for row in joint)
    ties = len(joint) - wins - losses
    # exact one-sided sign test over the discordant seeds: the pre-registered alternative is
    # "EIG needs fewer labels", so the tail is P(wins >= observed | fair coin on discordants)
    from math import comb

    discordant = wins + losses
    p_sign = sum(comb(discordant, j) for j in range(wins, discordant + 1)) / 2.0**discordant if discordant else 1.0

    eig_reached = [row["n_eig"] for row in joint]
    random_reached = [row["n_random"] for row in joint]
    return {
        "budgets": budgets,
        "target": target,
        "eig_curve": first_pair[0] if first_pair else {},
        "random_curve": first_pair[1] if first_pair else {},
        "per_seed": per_seed,
        "n_joint": len(joint),
        "n_seeds": n_seeds,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_sign": float(p_sign),
        "median_n_eig": float(np.median(eig_reached)) if eig_reached else None,
        "median_n_random": float(np.median(random_reached)) if random_reached else None,
    }


def print_report(result: dict) -> None:
    budgets, target = result["budgets"], result["target"]
    eig_curve, random_curve = result["eig_curve"], result["random_curve"]

    print(f"target held-out log-likelihood: {target}")
    print(f"{'labels':>8}  {'eig ll':>10}  {'random ll':>10}   (seed 1's paired curves, for shape)")
    checkpoints = sorted(set(eig_curve) | set(random_curve))
    step = max(1, len(checkpoints) // 8)  # a handful of rows, not every single budget
    for b in checkpoints[::step]:
        eig_v = f"{eig_curve[b]:.4f}" if b in eig_curve else "--"
        rnd_v = f"{random_curve[b]:.4f}" if b in random_curve else "--"
        print(f"{b:>8}  {eig_v:>10}  {rnd_v:>10}")

    print()
    wins, ties, losses = result["wins"], result["ties"], result["losses"]
    print(
        f"PAIRED receipt over {result['n_seeds']} seeds ({result['n_joint']} reached the target "
        f"under both strategies): EIG needed fewer labels on {wins}, tied on {ties}, "
        f"more on {losses}; exact one-sided sign test p = {result['p_sign']:.4f}"
    )
    if result["median_n_eig"] is not None:
        print(
            f"median labels to target: EIG {result['median_n_eig']:.0f} vs random "
            f"{result['median_n_random']:.0f} (descriptive medians over the jointly-reaching seeds)"
        )
    if result["n_joint"] == 0:
        print("receipt: target not reached under both strategies on any seed -- widen `budgets`.")
    elif result["p_sign"] < 0.05 and losses == 0:
        print("=> on this synthetic task EIG was never worse and often better (paired evidence at 5%)")
    elif result["p_sign"] < 0.05:
        print("=> EIG needed fewer labels more often than not (paired evidence at 5%)")
    else:
        print("=> the paired evidence is inconclusive at the 5% level on this run; add seeds")
    print("   (no pooled multiple is printed: one seed against a seed-average is not a paired")
    print("    estimand, and the old pooled multiple did not survive the paired replay -- STAT-RR19-01)")


def main() -> None:
    result = run_demo()
    print_report(result)


if __name__ == "__main__":
    main()
