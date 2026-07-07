"""B4-a: resolve_from_harvest() -- the multi-tier re-solve loop, Solution.improve()'s idea generalized
from one cascade to a whole Router stack. A new tier is trained from the harvested frontier labels and
inserted only if it demonstrably intercepts a real share of what used to always escalate."""

import random
import unittest

from mixle.task.router import Router, resolve_from_harvest
from mixle.task.solve import solve

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

FAMILY_A = ["free money now", "cheap loans fast", "win cash today"]
FAMILY_B = ["urgent wire transfer", "account suspended act now", "verify password immediately"]
HAM = ["meeting at noon", "see you tomorrow", "project update attached", "lunch today", "thanks for the help"]


def _teacher(x):
    """A scalar teacher -- Router.__call__ invokes the frontier tier as teacher(single_input)."""
    return "spam" if any(w in x for fam in (FAMILY_A, FAMILY_B) for w in fam) else "ham"


def _make(n, families, rng):
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append(rng.choice(rng.choice(families)) + " " + rng.choice(["!!!", "", "today"]))
        else:
            out.append(rng.choice(HAM) + " " + rng.choice(["", "thanks", "ok"]))
    return out


def _build_router(seed=0):
    """tier0 is trained ONLY on FAMILY_A, so FAMILY_B spam systematically escalates to the frontier."""
    train_a = _make(200, [FAMILY_A], random.Random(seed))
    tier0 = solve(_teacher, train_a, alpha=0.1, seed=seed)
    return Router.from_solutions([tier0], teacher=_teacher, costs=[0.0001, 0.01]), tier0


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class HarvestResolveTest(unittest.TestCase):
    def test_new_tier_measurably_drops_escalation_on_held_out_traffic(self):
        router, tier0 = _build_router(seed=0)
        router.serve(_make(400, [FAMILY_A, FAMILY_B], random.Random(1)))
        before = router.report()
        self.assertGreater(before["harvested_labels"], 20)  # FAMILY_B genuinely escalates a lot

        result = resolve_from_harvest(router, cost_per_request=0.001, seed=0)
        self.assertTrue(result.accepted)
        self.assertEqual(result.escalation_before, 1.0)
        self.assertLess(result.escalation_after, 1.0)
        self.assertGreater(result.escalation_drop, 0.0)
        self.assertIsNotNone(result.router)

        # the measured drop, on FRESH held-out traffic never seen during resolve_from_harvest, against
        # a control router that never got the new tier -- same traffic, only the stack differs.
        held_out = _make(300, [FAMILY_A, FAMILY_B], random.Random(2))
        control = Router.from_solutions([tier0], teacher=_teacher, costs=[0.0001, 0.01])
        control.serve(held_out)
        control_report = control.report()

        result.router.serve(held_out)
        resolved_report = result.router.report()

        control_frontier_share = next(t["share"] for t in control_report["tiers"] if t["tier"] == "frontier")
        resolved_frontier_share = next(t["share"] for t in resolved_report["tiers"] if t["tier"] == "frontier")
        self.assertLess(resolved_frontier_share, control_frontier_share)

    def test_no_harvest_is_an_honest_no_op(self):
        router, _tier0 = _build_router(seed=0)  # never served -> nothing harvested
        result = resolve_from_harvest(router, cost_per_request=0.001, seed=0)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.router)
        self.assertEqual(result.n_harvested, 0)

    def test_determinism_given_seed(self):
        # two INDEPENDENT routers, identically harvested -- resolve_from_harvest clears the input
        # router's harvest on success (so a repeat call on the SAME router legitimately sees less
        # data), so determinism is checked across two routers with the same harvested state instead.
        router1, _tier0a = _build_router(seed=0)
        router1.serve(_make(400, [FAMILY_A, FAMILY_B], random.Random(1)))
        router2, _tier0b = _build_router(seed=0)
        router2.serve(_make(400, [FAMILY_A, FAMILY_B], random.Random(1)))

        r1 = resolve_from_harvest(router1, cost_per_request=0.001, seed=0)
        r2 = resolve_from_harvest(router2, cost_per_request=0.001, seed=0)
        self.assertEqual(r1.accepted, r2.accepted)
        self.assertEqual(r1.escalation_after, r2.escalation_after)
        self.assertEqual(r1.agreement, r2.agreement)

    def test_successful_resolve_clears_the_input_routers_harvest(self):
        router, _tier0 = _build_router(seed=0)
        router.serve(_make(400, [FAMILY_A, FAMILY_B], random.Random(1)))
        self.assertGreater(len(router.harvested()[0]), 0)

        result = resolve_from_harvest(router, cost_per_request=0.001, seed=0)
        self.assertTrue(result.accepted)
        harvested_inputs, harvested_labels = router.harvested()
        self.assertEqual(harvested_inputs, [])
        self.assertEqual(harvested_labels, [])

    def test_inserted_tier_sits_just_before_the_frontier(self):
        router, _tier0 = _build_router(seed=0)
        router.serve(_make(400, [FAMILY_A, FAMILY_B], random.Random(1)))
        result = resolve_from_harvest(router, cost_per_request=0.001, name="resolved", seed=0)
        self.assertTrue(result.accepted)
        names = [t[0] for t in result.router.tiers]
        self.assertEqual(names[-1], "frontier")
        self.assertEqual(names[-2], "resolved")
        self.assertEqual(len(names), len(router.tiers) + 1)

    def test_a_small_calibration_split_cannot_be_accepted_regardless_of_seed(self):
        """At n_harvested=8 (the old accepted minimum), a 25% holdout gives only 2 calibration
        points -- escalation_rate can only land on 0.0/0.5/1.0, so whether a run happened to be
        "accepted" used to depend entirely on which 2 of 8 points landed in calibration, not on
        anything real. Below the real minimum calibration size, every seed must reject."""
        router, _tier0 = _build_router(seed=0)
        for n_target in (8, 10, 13):
            for seed in range(5):
                router.stats.harvested_inputs.clear()
                router.stats.harvested_labels.clear()
                items = _make(n_target, [FAMILY_B], random.Random(seed + 200))
                router.stats.harvested_inputs.extend(items)
                router.stats.harvested_labels.extend(["spam"] * n_target)
                result = resolve_from_harvest(router, cost_per_request=0.001, seed=seed)
                self.assertFalse(result.accepted, f"n_harvested={n_target} seed={seed} was accepted")


if __name__ == "__main__":
    unittest.main()
