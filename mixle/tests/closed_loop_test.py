"""L1: closed-loop self-evolution -- drift recovery, operator-credit bandit vs. uniform, genealogy."""

from __future__ import annotations

import unittest

import numpy as np

from mixle.evolve.closed_loop import (
    ClosedLoopSelfEvolution,
    GenealogyLedger,
    OperatorCreditBandit,
    accuracy_objective,
)
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


if __name__ == "__main__":
    unittest.main()
