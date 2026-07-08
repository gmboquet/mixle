"""CARD L4: task decomposition as structure learning. Acceptance (per the roadmap item):

1. On held-out synthetic task families the proposer never saw, `discover_decomposition` finds a REAL
   decomposition (not the monolithic no-intermediates case), and fitting that decomposition reaches
   comparable/better held-out accuracy than fitting a monolithic model at matched compute (both are a
   single closed-form linear solve -- no asymptotic compute advantage either way).
2. MDL gain (`mdl_score`) correlates with realized outcome quality across several candidate
   decompositions of varying real quality.
3. The `outcome_decomposer`-style wiring (`DecompositionProposer` / `record_decomposition_outcome`)
   actually changes the proposer's future behavior after logging `(decomposition, outcome)` pairs --
   a real before/after check, not just "ran without error".
"""

import unittest

import numpy as np
from scipy.stats import spearmanr

from mixle.task.design_prior import best_family
from mixle.task.edge import DesignModel
from mixle.task.task_decomposition import (
    MONOLITHIC,
    TaskExample,
    discover_decomposition,
    fit_decomposition,
    init_decomposition_proposer,
    log_decomposition_recipe,
    mdl_score,
    monolithic_predict,
    record_decomposition_outcome,
)


def _examples(fn, n, seed, noise=0.05):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        a, b = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
        y = fn(a, b) + float(rng.normal(0, noise))
        out.append(TaskExample(inputs={"a": a, "b": b}, output=y))
    return out


def _mse(pred, examples):
    y = np.asarray([ex.output for ex in examples], dtype=float)
    return float(np.mean((np.asarray(pred, dtype=float) - y) ** 2))


# Three synthetic task FAMILIES, each genuinely decomposable through simpler intermediate pieces, and
# each with distractor candidate intermediates that do NOT help (so a real search, not a rubber stamp,
# is required). None of these families or their intermediates appear anywhere else in mixle -- a true
# held-out test of the search procedure, which does no pretraining on any family.
FAMILIES = {
    "sine_plus_linear": {  # output = f(g(a), b) with g nonlinear -- a monolithic LINEAR fit of output
        # on raw (a, b) cannot capture sin(a); a decomposition that first computes sin(a) can.
        "fn": lambda a, b: 3.0 * np.sin(a) + 2.0 * b,
        "candidates": {
            "sin_a": lambda i: float(np.sin(i["a"])),
            "sq_a": lambda i: i["a"] ** 2,  # distractor: does not explain the residual
            "sq_b": lambda i: i["b"] ** 2,  # distractor
        },
    },
    "product": {  # output = f(g(a,b)) where g is a genuine cross term -- unrepresentable by ANY
        # linear combination of the raw inputs alone, only by the product intermediate.
        "fn": lambda a, b: 1.5 * a * b,
        "candidates": {
            "prod": lambda i: i["a"] * i["b"],
            "sum_ab": lambda i: i["a"] + i["b"],  # distractor
            "diff_ab": lambda i: i["a"] - i["b"],  # distractor
        },
    },
    "square_plus_cube": {  # output = f(g(a), h(a)) -- TWO intermediates of the SAME input, testing
        # the multi-parent case a single-parent dependency forest cannot represent directly.
        "fn": lambda a, b: 0.5 * a**2 - 0.2 * a**3 + b,
        "candidates": {
            "sq_a": lambda i: i["a"] ** 2,
            "cube_a": lambda i: i["a"] ** 3,
            "abs_a": lambda i: abs(i["a"]),  # distractor
        },
    },
}


class DiscoverDecompositionBeatsMonolithicTest(unittest.TestCase):
    """Held-out synthetic families, never used to tune the search procedure or seen by any proposer."""

    def test_families_get_real_decompositions_that_beat_monolithic(self):
        results = {}
        for name, spec in FAMILIES.items():
            train = _examples(spec["fn"], n=250, seed=1)
            test = _examples(spec["fn"], n=200, seed=999)  # disjoint seed: a real held-out set

            forest = discover_decomposition(train, spec["candidates"], seed=0)
            self.assertTrue(forest.is_decomposed, f"{name}: expected a real decomposition, got monolithic")
            self.assertGreater(forest.mdl_gain, 0.0, f"{name}: decomposition should compress (positive MDL gain)")

            decomposed_pred = [forest.predict(ex.inputs, spec["candidates"]) for ex in test]
            monolithic_pred = monolithic_predict(train, test)

            decomposed_mse = _mse(decomposed_pred, test)
            monolithic_mse = _mse(monolithic_pred, test)
            results[name] = (forest.chosen, decomposed_mse, monolithic_mse)

            # both fits are a single closed-form linear solve on the same n -- compute is matched by
            # construction, no separate budget bookkeeping needed.
            self.assertLessEqual(
                decomposed_mse,
                monolithic_mse,
                f"{name}: decomposed MSE {decomposed_mse:.4f} should not exceed monolithic MSE "
                f"{monolithic_mse:.4f} (chosen={forest.chosen})",
            )

        for name, (chosen, dmse, mmse) in results.items():
            print(f"[L4] {name}: chosen={chosen} decomposed_mse={dmse:.4f} monolithic_mse={mmse:.4f}")


class MdlGainCorrelatesWithOutcomeTest(unittest.TestCase):
    """Score several candidate decompositions of ONE family (some genuinely better, some deliberately
    worse) by MDL gain, then really try each on held-out data and rank-correlate."""

    def test_mdl_gain_rank_correlates_with_held_out_quality(self):
        spec = FAMILIES["sine_plus_linear"]
        train = _examples(spec["fn"], n=400, seed=2, noise=0.02)
        test = _examples(spec["fn"], n=300, seed=888, noise=0.02)

        # a deliberate quality GRADIENT, from the true decomposition down to purely-wrong intermediates,
        # so the rank check is not dominated by near-tied distractors.
        candidate_decompositions = [
            ["sin_a", "b"],  # the real structure: both pieces right
            ["sin_a"],  # right nonlinear piece, missing the linear term
            ["a", "b"],  # monolithic: raw inputs only, sin(a) unmodeled
            ["b"],  # only the linear term, sin(a) entirely missing
            ["sq_a", "b"],  # wrong nonlinear feature, but keeps the linear term
            ["sq_a"],  # wrong nonlinear feature, nothing else
            ["sq_b"],  # a pure distractor, no useful signal at all
        ]

        mdl_gains, outcomes = [], []
        for decomp in candidate_decompositions:
            forest = fit_decomposition(train, decomp, spec["candidates"], seed=0)
            pred = [forest.predict(ex.inputs, spec["candidates"]) for ex in test]
            outcome = -_mse(pred, test)  # higher (less negative) = better held-out quality
            mdl_gains.append(forest.mdl_gain)
            outcomes.append(outcome)

        rho, _ = spearmanr(mdl_gains, outcomes)
        print(f"[L4] MDL gain vs outcome: gains={mdl_gains} outcomes={outcomes} spearman_rho={rho:.3f}")
        self.assertGreater(rho, 0.7, "MDL gain should rank-correlate strongly with realized outcome quality")


class OutcomeDecomposerWiringTest(unittest.TestCase):
    """Smoke test: logging (decomposition, outcome) pairs through the outcome_decomposer-style
    DecompositionProposer really changes what it proposes next, not just "ran without error"."""

    def test_logging_outcomes_shifts_proposal_toward_the_winner(self):
        # two-step "decompositions" (a terminal marker keeps sequences length >= 2, exercising the
        # Markov chain's transition table rather than only its length-1 initial-state table).
        good, bad = ["prod", "done"], ["sum_ab", "done"]
        proposer = init_decomposition_proposer([good, bad, good, bad])  # ambiguous round-0 seed corpus

        rng = np.random.RandomState(0)
        before_samples = [proposer.plan_model.sample(rng) for _ in range(200)]
        before_good_rate = sum(1 for s in before_samples if s == good) / len(before_samples)

        for _ in range(20):
            proposer = record_decomposition_outcome(proposer, good, 10.0)
            proposer = record_decomposition_outcome(proposer, bad, -10.0)

        rng2 = np.random.RandomState(1)
        after_samples = [proposer.plan_model.sample(rng2) for _ in range(200)]
        after_good_rate = sum(1 for s in after_samples if s == good) / len(after_samples)

        print(f"[L4] proposer P(good) before={before_good_rate:.2f} after={after_good_rate:.2f}")
        self.assertGreater(
            after_good_rate,
            before_good_rate,
            "logging (decomposition, outcome) pairs should shift future proposals toward the winner",
        )
        self.assertGreater(after_good_rate, 0.8)


class DesignPriorWiringTest(unittest.TestCase):
    """`log_decomposition_recipe` really persists a decomposed-vs-monolithic family prior into the
    existing `DesignModel` ledger, and `mdl_score` (not just `fit_decomposition`) drives it -- so
    `best_family` reflects which approach has actually won once enough recipes are on file."""

    def test_decomposed_family_outranks_monolithic_once_recorded(self):
        spec = FAMILIES["product"]
        design = DesignModel(signature="task_decomposition_test", n_constraints=0)

        for seed in range(5):
            examples = _examples(spec["fn"], n=200, seed=100 + seed)
            gain = mdl_score(examples, ["prod"], spec["candidates"])
            log_decomposition_recipe(design, gain, family="decomposed")
            mono_gain = mdl_score(examples, ["a", "b"], spec["candidates"])
            log_decomposition_recipe(design, mono_gain, family=MONOLITHIC)

        self.assertEqual(best_family(design), "decomposed")


if __name__ == "__main__":
    unittest.main()
