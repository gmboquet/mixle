"""H3 — precedence-constrained scheduling: maximum-weight closure and time-phased MILP scheduling.

mixle.precedence_scheduling exposes two general combinatorial-optimization primitives -- maximum-weight
closure (a min-cut construction) and capacity-limited, discounted time-phased scheduling (a MILP) --
whose own module docstring names open-pit "ultimate pit limit" mine planning as just ONE instantiation,
and whose test suite already covers exactly that instantiation. This example demonstrates the same two
functions on a domain with nothing underground in it, to show the generality is real: which parts of a
software release's dependency graph are worth building at all, and in what order, given a fixed
per-sprint engineering capacity.

Ten release items form a small dependency DAG: foundational services with negative net value (their
build/ops cost exceeds their own direct payoff) that unlock customer-facing features with positive net
value, plus a small self-contained legacy-cleanup chain that -- unlike the foundational services -- is
*not* worth paying for on its own. ``maximum_weight_closure`` picks the value-maximizing,
dependency-closed subset of items; ``schedule_activities`` then sequences that subset across four
sprints under fixed per-sprint capacity, discounting later sprints' value.

Run: ``python examples/precedence_scheduling_example.py``
"""

from __future__ import annotations

import numpy as np

from mixle.precedence_scheduling import maximum_weight_closure, schedule_activities

LABELS = [
    "infra", "auth_svc", "billing_svc", "user_dash", "team_dash",
    "invoice_export", "sso", "usage_analytics", "legacy_cleanup", "api_removal",
]  # fmt: skip
VALUE = np.array([-8.0, -5.0, -6.0, 12.0, 15.0, 9.0, 7.0, 11.0, -4.0, 3.0])
# (item, prerequisite) pairs: an item cannot ship before its prerequisite does.
PRECEDENCE = [
    (1, 0), (2, 0),          # auth_svc, billing_svc both require infra
    (3, 1), (4, 1), (4, 2),  # user_dash requires auth_svc; team_dash requires auth_svc + billing_svc
    (5, 2),                  # invoice_export requires billing_svc
    (6, 1), (7, 1), (7, 2),  # sso requires auth_svc; usage_analytics requires auth_svc + billing_svc
    (9, 8),                  # api_removal requires legacy_cleanup
]  # fmt: skip


def demo_closure() -> np.ndarray:
    """Which release items are worth building at all, ignoring timing."""
    print("## 1. maximum_weight_closure -- which items are worth building at all\n")
    mask = maximum_weight_closure(VALUE, PRECEDENCE)
    for label, v, keep in zip(LABELS, VALUE, mask):
        print(f"  {'BUILD' if keep else 'SKIP ':<5}  {label:<16} value={v:+6.1f}")

    closure_value = float(VALUE[mask].sum())
    print(f"\n  closure net value: {closure_value:+.1f}  ({int(mask.sum())}/{len(LABELS)} items selected)")

    # Why the foundational (negative-value) services clear the bar: their cost is paid back, with
    # room to spare, by the positive-value items they unlock.
    infra_cost = float(VALUE[[0, 1, 2]].sum())
    downstream = float(VALUE[[3, 4, 5, 6, 7]].sum())
    print(
        f"\n  infra + auth_svc + billing_svc cost {infra_cost:+.1f} on their own, but unlock "
        f"{downstream:+.1f} of downstream feature value -- net {infra_cost + downstream:+.1f}, so the "
        "whole chain clears the closure."
    )

    # Why the legacy chain does not: closure weighs a chain by what it nets *as a whole*, not by
    # any one item's sign.
    legacy_chain = float(VALUE[[8, 9]].sum())
    print(
        f"  legacy_cleanup ({VALUE[8]:+.1f}) + api_removal ({VALUE[9]:+.1f}) net {legacy_chain:+.1f} as "
        "a chain -- api_removal alone is profitable, but it cannot be taken without its prerequisite, "
        "and the pair together is a net loss, so the closure correctly excludes both."
    )
    return mask


def demo_schedule(mask: np.ndarray) -> None:
    """When to build the items the closure says are worth building, under fixed sprint capacity."""
    print("\n## 2. schedule_activities -- when to build the items worth building\n")
    selected = np.flatnonzero(mask)
    local_of = {b: i for i, b in enumerate(selected)}
    sub_value = VALUE[selected]
    sub_labels = [LABELS[i] for i in selected]
    sub_precedence = [(local_of[b], local_of[p]) for b, p in PRECEDENCE if b in local_of and p in local_of]

    capacity = np.array([2.0, 2.0, 2.0, 2.0])  # 2 items/sprint, 4 sprints -- exactly 8 slots for 8 items
    n_periods = 4
    npv, period = schedule_activities(sub_value, sub_precedence, capacity, n_periods, discount=0.05)

    print(f"  capacity {capacity.tolist()} items/sprint over {n_periods} sprints, 5% discount per sprint\n")
    for t in range(n_periods):
        in_period = [sub_labels[i] for i in range(len(selected)) if period[i] == t]
        print(f"  sprint {t}: {', '.join(in_period) if in_period else '(none)'}")
    never = [sub_labels[i] for i in range(len(selected)) if period[i] == -1]  # -1 == never scheduled
    if never:
        print(f"  never scheduled (capacity-starved): {', '.join(never)}")

    print(f"\n  discounted NPV of this schedule: {npv:+.2f}")

    # Verify the returned schedule honors both constraints, rather than just trusting the solver.
    scheduled = period >= 0
    per_sprint_count = np.bincount(period[scheduled].astype(int), minlength=n_periods)
    assert (per_sprint_count <= capacity).all(), "capacity exceeded in some sprint"
    for b, p in sub_precedence:
        if scheduled[b]:
            assert scheduled[p] and period[p] <= period[b], "an item shipped before its prerequisite"
    print("  (verified: no sprint exceeds capacity, and every shipped item's prerequisite ships at or before it)")


def main():
    print("# mixle.precedence_scheduling on a software-release dependency DAG (deliberately not a mine)\n")
    mask = demo_closure()
    demo_schedule(mask)


if __name__ == "__main__":
    main()
