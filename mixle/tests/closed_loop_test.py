"""L1: closed-loop self-evolution -- drift recovery, operator-credit bandit vs. uniform, genealogy."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from mixle.evolve.closed_loop import (
    ClosedLoopSelfEvolution,
    GenealogyLedger,
    OperatorCreditBandit,
    accuracy_objective,
)
from mixle.evolve.improve import _split
from mixle.inference.estimation import optimize
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator


def _fit_categorical(labels, *, pseudo_count: float = 0.5):
    return optimize(list(labels), CategoricalEstimator(pseudo_count=pseudo_count), max_its=5, out=None)


def _gen_batch(probs, n, rng, labels=("A", "B", "C")):
    return list(rng.choice(list(labels), size=n, p=list(probs)))


class ClosedLoopDriftRecoveryTest(unittest.TestCase):
    """Acceptance criterion 1: a synthetic drifting stream -- the loop recovers accuracy without
    human input."""

    def test_accuracy_recovers_after_concept_drift(self):
        rng = np.random.RandomState(0)
        objective = accuracy_objective()

        # champion trained on the EARLY (pre-drift) label distribution: A is dominant.
        early_probs = [0.7, 0.2, 0.1]
        champion = _fit_categorical(_gen_batch(early_probs, 300, rng))

        # measured BEFORE drift: the champion is accurate on data matching what it was trained on.
        pre_drift_eval = _gen_batch(early_probs, 300, rng)
        acc_before_drift = objective.scalar(champion, pre_drift_eval)
        self.assertGreater(acc_before_drift, 0.55)

        # the true generating distribution DRIFTS partway through the stream: B becomes dominant.
        post_drift_probs = [0.2, 0.7, 0.1]
        post_drift_eval_stale = _gen_batch(post_drift_probs, 300, rng)
        acc_right_after_drift = objective.scalar(champion, post_drift_eval_stale)
        # the stale champion, uncorrected, degrades badly on the drifted distribution.
        self.assertLess(acc_right_after_drift, acc_before_drift - 0.2)

        # run the closed loop over many post-drift batches, WITHOUT human intervention.
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=0, acquire_k=40)
        stream = [_gen_batch(post_drift_probs, 60, rng) for _ in range(24)]
        results = loop.run(stream)

        self.assertTrue(any(r.promoted for r in results), "the loop never adopted a challenger")

        # measured AFTER the loop: accuracy on FRESH post-drift data is recovered.
        post_drift_eval_final = _gen_batch(post_drift_probs, 300, rng)
        acc_after_loop = objective.scalar(loop.champion, post_drift_eval_final)

        self.assertGreater(
            acc_after_loop,
            acc_right_after_drift + 0.2,
            f"drift recovery failed: stale={acc_right_after_drift:.3f} recovered={acc_after_loop:.3f}",
        )
        self.assertGreater(acc_after_loop, 0.5)

        # honest report of the real, measured numbers this test asserts on.
        print(
            f"\n[drift recovery] acc_before_drift={acc_before_drift:.3f} "
            f"acc_right_after_drift={acc_right_after_drift:.3f} acc_after_loop={acc_after_loop:.3f}"
        )


class OperatorCreditBanditTest(unittest.TestCase):
    """Acceptance criterion 2: the operator-credit bandit beats uniform operator choice when one
    operator is known to win more often for a given context."""

    def test_duplicate_operator_names_are_rejected(self):
        # MXR-080-1772: names index the arms positionally and reward() resolves one via
        # `.index(operator)`, which returns the FIRST equal name -- so a duplicate silently credits
        # the wrong arm and report() lists the same name twice with different statistics.
        with self.assertRaises(ValueError):
            OperatorCreditBandit(["distill", "distill", "refine"])

    def test_non_finite_reward_is_refused_rather_than_absorbed(self):
        bandit = OperatorCreditBandit(["distill", "refine"], seed=0)
        bandit.select("ctx")
        bandit.reward("ctx", "distill", 1.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(reward=repr(bad)), self.assertRaises(ValueError):
                bandit.reward("ctx", "distill", bad)
        with self.assertRaises(KeyError):
            bandit.reward("ctx", "not_an_operator", 1.0)
        stats = bandit.report()["ctx"]["distill"]
        self.assertTrue(math.isfinite(stats["mean_reward"]))

    def test_bandit_converges_to_the_better_operator_faster_than_uniform(self):
        operators = ["distill", "refine", "evolve"]
        context = "drift_type_A"
        # ground truth: 'distill' wins 80% of the time in this context, the others rarely.
        true_win_prob = {"distill": 0.8, "refine": 0.15, "evolve": 0.1}
        n_trials = 400
        rng_bandit = np.random.RandomState(1)
        rng_uniform = np.random.RandomState(2)

        bandit = OperatorCreditBandit(operators, c=1.0, seed=1)
        bandit_picks = []
        for _ in range(n_trials):
            op = bandit.select(context)
            bandit_picks.append(op)
            reward = 1.0 if rng_bandit.random_sample() < true_win_prob[op] else 0.0
            bandit.reward(context, op, reward)

        uniform_picks = []
        uniform_rewards = []
        for _ in range(n_trials):
            op = operators[rng_uniform.randint(len(operators))]
            uniform_picks.append(op)
            uniform_rewards.append(1.0 if rng_uniform.random_sample() < true_win_prob[op] else 0.0)

        # win rate of the operator each policy actually PICKED, over the whole run.
        bandit_win_rate = float(np.mean([true_win_prob[op] for op in bandit_picks]))
        uniform_win_rate = float(np.mean([true_win_prob[op] for op in uniform_picks]))

        # convergence: in the SECOND half of the run, the bandit should pick 'distill' (the real
        # winner) far more often than 1/3 of the time (uniform's rate).
        second_half = bandit_picks[n_trials // 2 :]
        distill_share_bandit = second_half.count("distill") / len(second_half)
        distill_share_uniform = 1.0 / len(operators)

        print(
            f"\n[operator-credit bandit] bandit_win_rate={bandit_win_rate:.3f} "
            f"uniform_win_rate={uniform_win_rate:.3f} "
            f"bandit_distill_share(2nd half)={distill_share_bandit:.3f} "
            f"uniform_distill_share={distill_share_uniform:.3f}"
        )

        self.assertGreater(bandit_win_rate, uniform_win_rate)
        self.assertGreater(distill_share_bandit, distill_share_uniform + 0.2)

        report = bandit.report()
        self.assertIn(context, report)
        self.assertGreater(report[context]["distill"]["mean_reward"], report[context]["refine"]["mean_reward"])
        self.assertGreater(report[context]["distill"]["mean_reward"], report[context]["evolve"]["mean_reward"])


class BudgetAccountingTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0046): ``run()`` must check a step's operator cost against
    the REMAINING budget before running it, and settle the charge from what actually happened --
    skipped-by-budget, inapplicable, and failed attempts must all be free, never charged unconditionally
    after the fact."""

    def test_budget_below_every_operator_cost_never_executes_or_charges(self):
        rng = np.random.RandomState(4)
        objective = accuracy_objective()
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=4, acquire_k=40)

        min_cost = min(float(getattr(op, "cost_hint", 1.0)) for op in loop.operators.values())
        budget = min_cost / 10.0  # strictly below EVERY operator's cost -- nothing can ever be afforded.
        self.assertLess(budget, min_cost, "test setup: budget must be below every operator's cost")

        stream = [_gen_batch([0.2, 0.7, 0.1], 60, rng) for _ in range(24)]
        results = loop.run(stream, budget=budget)

        # an unaffordable budget affords exactly zero operator attempts -- the loop burns through the
        # WHOLE stream doing free harvest/acquire-only work rather than letting one paid attempt slip
        # through under the ceiling (the pre-fix bug: exactly one attempt always got through and was
        # charged, regardless of the budget).
        self.assertEqual(len(results), len(stream), "an unaffordable budget must never let a single attempt through")
        self.assertEqual(sum(r.cost for r in results), 0.0, "nothing may be charged when no operator fits the budget")
        self.assertTrue(all(not r.promoted for r in results), "no operator should ever have run, let alone promoted")
        self.assertTrue(all(r.delta == 0.0 for r in results))

        print(
            f"\n[budget accounting] unaffordable budget={budget:.4f} < min_cost={min_cost:.4f}: "
            f"{len(results)}/{len(stream)} steps ran, total charged={sum(r.cost for r in results):.4f}"
        )

    def test_budget_that_fits_still_spends_and_never_exceeds_the_ceiling(self):
        # negative control for the test above: a real, affordable budget must still let operators run
        # and spend -- the fix must not degenerate into "a budget always charges nothing."
        rng = np.random.RandomState(5)
        objective = accuracy_objective()
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=5, acquire_k=40)

        max_cost = max(float(getattr(op, "cost_hint", 1.0)) for op in loop.operators.values())
        budget = max_cost * 6.0  # generous: affords several attempts even of the priciest operator.

        stream = [_gen_batch([0.2, 0.7, 0.1], 60, rng) for _ in range(24)]
        results = loop.run(stream, budget=budget)
        total_spent = sum(r.cost for r in results)

        self.assertGreater(total_spent, 0.0, "a generous budget should let at least one operator genuinely run")
        self.assertLessEqual(total_spent, budget, "spend must never exceed the ceiling")
        # every charged step ran a genuine, complete attempt: its charge is exactly that operator's cost_hint.
        for r in results:
            if r.cost > 0.0:
                self.assertEqual(r.cost, float(getattr(loop.operators[r.operator], "cost_hint", 1.0)))

        print(
            f"\n[budget accounting] affordable budget={budget:.2f}: total charged={total_spent:.2f} "
            f"over {len(results)} steps"
        )


class GenealogyReconstructionTest(unittest.TestCase):
    """Acceptance criterion 3: every champion's lineage is reconstructible."""

    def test_lineage_is_a_real_ordered_chain_back_to_the_root(self):
        rng = np.random.RandomState(3)
        objective = accuracy_objective()

        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=3, acquire_k=40)

        # several successive drift phases, to force multiple adoption cycles across the run.
        phases = [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.1, 0.3],
        ]
        for probs in phases:
            stream = [_gen_batch(probs, 60, rng) for _ in range(12)]
            loop.run(stream)

        n_adoptions = sum(1 for r in loop.history if r.promoted)
        self.assertGreaterEqual(n_adoptions, 2, "need multiple adoptions to test a real multi-hop lineage")

        final_champion = loop.champion
        chain = loop.genealogy.lineage(final_champion)

        self.assertGreaterEqual(len(chain), 1)
        # root-first ordering: each row's child is the NEXT row's parent.
        for a, b in zip(chain, chain[1:]):
            self.assertEqual(a["meta"]["child_hash"], b["parent_hash"])
        # the last row in the chain is the one that produced the final champion.
        self.assertEqual(chain[-1]["meta"]["child_hash"], loop.genealogy._id_for(final_champion))
        # every hop records a real operator name and a non-negative measured gap.
        for row in chain:
            self.assertIn(row["operator"], {"distill", "refine", "evolve"})
            self.assertGreaterEqual(row["delta"], 0.0)

        print(f"\n[genealogy] {len(chain)}-hop lineage: " + " -> ".join(row["operator"] for row in chain))

    def test_unrecorded_model_has_no_lineage(self):
        ledger = GenealogyLedger()
        self.assertEqual(ledger.lineage(object()), [])

    def test_lineage_matches_a_content_identical_reloaded_object(self):
        """Regression coverage (audit MXR-080-0047): identity must be durable/content-addressed, not
        the live object's ``id()`` (a CPython memory address) -- a DISTINCT object with the exact same
        serialized content as a recorded champion (standing in for "the same model, deserialized from
        disk in a later process") must resolve to the SAME lineage as the original, not an empty one."""
        rng = np.random.RandomState(6)
        objective = accuracy_objective()
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=6, acquire_k=40)

        phases = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        for probs in phases:
            stream = [_gen_batch(probs, 60, rng) for _ in range(12)]
            loop.run(stream)
        self.assertTrue(any(r.promoted for r in loop.history), "need at least one adoption to test lineage")

        final_champion = loop.champion
        original_chain = loop.genealogy.lineage(final_champion)
        self.assertGreaterEqual(len(original_chain), 1)

        # a DISTINCT python object with identical content -- deepcopy stands in for "deserialized from
        # disk in a fresh process": never itself passed to record_adoption, and NOT the same id() as
        # final_champion, yet it is the same model.
        reloaded = copy.deepcopy(final_champion)
        self.assertIsNot(reloaded, final_champion)
        self.assertNotEqual(id(reloaded), id(final_champion))

        reloaded_chain = loop.genealogy.lineage(reloaded)
        self.assertEqual(
            reloaded_chain,
            original_chain,
            "a content-identical reloaded model must resolve to the exact same lineage as the original",
        )
        # the durable id itself is content-derived, not object-identity-derived.
        self.assertEqual(loop.genealogy._id_for(reloaded), loop.genealogy._id_for(final_champion))

    def test_lineage_does_not_alias_an_unrelated_model(self):
        """Negative control for the test above: a genuinely DIFFERENT model (never recorded, unrelated
        content) must still resolve to no lineage -- content-addressing must not produce false-positive
        matches against whatever else happens to be in the ledger."""
        rng = np.random.RandomState(7)
        objective = accuracy_objective()
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=objective, seed=7, acquire_k=40)

        phases = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        for probs in phases:
            stream = [_gen_batch(probs, 60, rng) for _ in range(12)]
            loop.run(stream)
        self.assertTrue(any(r.promoted for r in loop.history), "need at least one adoption to test lineage")

        unrelated = _fit_categorical(_gen_batch([0.05, 0.05, 0.9], 300, np.random.RandomState(999)))
        self.assertNotEqual(loop.genealogy._id_for(unrelated), loop.genealogy._id_for(loop.champion))
        self.assertEqual(loop.genealogy.lineage(unrelated), [])


class _RecordingObjective:
    """Wraps a real Objective, recording the exact ``data`` list every ``pointwise``/``scalar`` call
    receives -- lets a test see exactly which rows the loop actually scored on, without step() exposing
    its internal train/verify split as part of its public API."""

    def __init__(self, real):
        self._real = real
        self.name = real.name
        self.lower_is_better = real.lower_is_better
        self.calls: list[list] = []

    def pointwise(self, model, data):
        self.calls.append(list(data))
        return self._real.pointwise(model, data)

    def scalar(self, model, data):
        self.calls.append(list(data))
        return self._real.scalar(model, data)


class _ScoreFailsOnPromotion(ClosedLoopSelfEvolution):
    """A loop whose post-gate scoring fails exactly on a step that would deploy a new champion.

    The condition is written against the champion the step STARTED with, so it fires identically
    whichever side of the champion swap the scoring happens on -- which is precisely what
    MXR-080-1773 is about.
    """

    def step(self, batch, **kwargs):
        self._incumbent = self.champion
        return super().step(batch, **kwargs)

    def _score(self, model, data):
        if model is not self._incumbent:
            raise RuntimeError("scoring the prospective champion blew up")
        return super()._score(model, data)


def _total_pulls(bandit) -> int:
    return sum(int(arm["pulls"]) for ctx in bandit.report().values() for arm in ctx.values())


class PromotionAtomicityTest(unittest.TestCase):
    """MXR-080-1773 (Critical): a loop step must be all-or-nothing.

    Genealogy was recorded and ``self.champion`` replaced BEFORE the new champion's score was
    produced, so a scoring failure left the challenger deployed with one adoption row and one bandit
    reward, while raising to the caller with no ``LoopStepResult`` and no history entry -- the loop's
    durable state and its own record of what happened permanently disagreeing, with nothing to
    reconcile them. A failed step must now change nothing at all and be retryable.
    """

    def test_a_failed_post_gate_score_deploys_nothing(self):
        rng = np.random.RandomState(0)
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = _ScoreFailsOnPromotion(champion, objective=accuracy_objective(), seed=0, acquire_k=40)
        stream = [_gen_batch([0.2, 0.7, 0.1], 60, rng) for _ in range(24)]

        reached_a_promotion = False
        for batch in stream:
            before = (
                loop.champion,
                len(loop.genealogy.ledger.rows),
                len(loop.history),
                _total_pulls(loop.bandit),
            )
            try:
                loop.step(batch)
            except RuntimeError:
                reached_a_promotion = True
                self.assertIs(loop.champion, before[0], "a failed step left the challenger deployed")
                self.assertEqual(len(loop.genealogy.ledger.rows), before[1], "an adoption receipt was stranded")
                self.assertEqual(len(loop.history), before[2], "history and durable state disagree")
                self.assertEqual(_total_pulls(loop.bandit), before[3], "the bandit was rewarded for a lost step")
                break
            # every step that DID return is fully recorded
            self.assertEqual(len(loop.history), before[2] + 1)
        self.assertTrue(reached_a_promotion, "the stream never reached a promoting step")

    def test_an_ordinary_promoting_step_still_commits_everything_together(self):
        rng = np.random.RandomState(0)
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=accuracy_objective(), seed=0, acquire_k=40)
        results = loop.run([_gen_batch([0.2, 0.7, 0.1], 60, rng) for _ in range(24)])

        promotions = [r for r in results if r.promoted]
        self.assertTrue(promotions, "the loop never adopted a challenger")
        self.assertEqual(len(loop.history), len(results))  # every returned step is in history
        self.assertEqual(len(loop.genealogy.ledger.rows), len(promotions))  # one receipt per promotion
        self.assertIs(loop.champion, results[-1].champion)
        self.assertIs(loop.champion, promotions[-1].champion)


class DataRoleDisjointnessTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0042, Critical): step()'s default verify_data used to be the
    WHOLE arriving batch, which CONTAINS train_data (acquire's top-priority subset of it) -- so the gate
    scored a challenger partly on rows it had just been fit on. The fix splits the batch into a
    candidate pool and verify_data FIRST, before harvest/acquire ever touch it, specifically so
    verify_data stays a representative sample rather than the (biased) complement of acquire's own
    hardest-example selection -- see step()'s "Note on verify" docstring."""

    def test_step_default_verify_is_disjoint_from_the_candidate_pool(self):
        rng = np.random.RandomState(20)
        real_obj = accuracy_objective()
        spy = _RecordingObjective(real_obj)
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=spy, seed=20, acquire_k=40)
        batch = _gen_batch([0.2, 0.7, 0.1], 60, rng)
        # replicate step()'s own (documented, deterministic) split exactly: this is step #1, so
        # _n_steps is 1 by the time the split runs.
        expected_pool, expected_verify = _split(batch, loop.holdout, loop.seed + 1)

        loop.step(batch)

        self.assertTrue(spy.calls, "expected the objective to be invoked")
        self.assertTrue(
            all(len(c) < len(batch) for c in spy.calls),
            f"every scored slice must be a strict split, never the full {len(batch)}-row batch: "
            f"{[len(c) for c in spy.calls]}",
        )
        for c in spy.calls:
            self.assertTrue(
                c == expected_pool or c == expected_verify,
                "objective was scored on data outside the (candidate_pool, verify_data) split",
            )
        self.assertTrue(any(c == expected_verify for c in spy.calls), "expected a gate/score call on verify_data")

    def test_default_verify_differs_from_the_pre_fix_whole_batch_alias(self):
        # Explicit before/after: the pre-fix default was literally `verify_data = list(verify) if
        # verify is not None else batch` -- i.e. the WHOLE batch. The new default must be a strict,
        # disjoint subset that never coincides with that old, buggy default.
        rng = np.random.RandomState(22)
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=accuracy_objective(), seed=22, acquire_k=40)
        batch = _gen_batch([0.2, 0.7, 0.1], 60, rng)

        _, new_verify = _split(batch, loop.holdout, loop.seed + 1)
        old_verify = batch  # the exact pre-fix default

        self.assertNotEqual(new_verify, old_verify)
        self.assertLess(len(new_verify), len(old_verify))

    def test_step_explicit_verify_still_respected_and_bypasses_the_auto_split(self):
        # Negative control: a caller that explicitly supplies verify= (its own held-out batch) must
        # still have it used AS GIVEN -- this fix only changes the DEFAULT when verify is omitted.
        rng = np.random.RandomState(21)
        real_obj = accuracy_objective()
        spy = _RecordingObjective(real_obj)
        champion = _fit_categorical(_gen_batch([0.7, 0.2, 0.1], 300, rng))
        loop = ClosedLoopSelfEvolution(champion, objective=spy, seed=21, acquire_k=40)
        batch = _gen_batch([0.2, 0.7, 0.1], 60, rng)
        explicit_verify = _gen_batch([0.2, 0.7, 0.1], 25, rng)

        loop.step(batch, verify=explicit_verify)

        self.assertTrue(spy.calls)
        self.assertTrue(
            any(c == explicit_verify for c in spy.calls), "the explicit verify batch must be used for gating/scoring"
        )


if __name__ == "__main__":
    unittest.main()
