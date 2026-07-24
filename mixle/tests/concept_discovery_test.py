"""CARD L6 -- concept discovery: the library itself under selection.

Acceptance criteria under test (see mixle/evolve/concept_discovery.py):

1. On a corpus of tasks that all secretly share ONE hidden, unmodeled family, the loop discovers and
   admits that family within the first few tasks, then REUSES it (queried, tried, and accepted) on
   the remaining tasks without re-discovering it from scratch each time -- measured reuse rate and
   cumulative MDL gain are both reported and asserted positive.
2. Admission is genuinely reversible: revoke() removes a family from both the active set and the
   design-prior ledger, so query() can never recommend it again.
3. The verify/adopt gate correctly REJECTS a bad candidate family that does not actually improve
   held-out fit over the champion.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from mixle.evolve import challenger_beats_champion, nll_objective
from mixle.evolve.concept_discovery import (
    ConceptLibrary,
    _fit_family,
    run_concept_discovery_loop,
    task_signature,
)


def _skew_normal_corpus(n_tasks: int, *, seed: int = 0, size: int = 1000) -> list[np.ndarray]:
    """A corpus of ``n_tasks`` datasets all drawn from ONE hidden family (skew-normal, shape=8) that
    a starting library containing only ``gaussian`` cannot fit well: the location-scale MLE recovers
    the mean, but the systematic third-moment mismatch recurs identically across every task."""
    rng = np.random.RandomState(seed)
    return [stats.skewnorm.rvs(8.0, loc=0.0, scale=1.5, size=size, random_state=rng) for _ in range(n_tasks)]


class ConceptDiscoveryAcceptanceTest(unittest.TestCase):
    """The card's own acceptance criterion: induce once, reuse thereafter, positive cumulative MDL gain."""

    def setUp(self):
        self.tasks = _skew_normal_corpus(8, seed=0)
        self.library, self.results = run_concept_discovery_loop(self.tasks)

    def test_hidden_family_is_admitted_within_the_first_few_tasks(self):
        admissions = [r.task_index for r in self.results if r.admitted_family is not None]
        self.assertTrue(admissions, "the loop never admitted a new concept over the whole corpus")
        first_admission = admissions[0]
        self.assertLessEqual(first_admission, 3, f"admission took too long: task index {first_admission}")
        self.assertEqual(self.results[first_admission].admitted_family, "skew_normal")
        self.assertIn("skew_normal", self.library.families)
        # only ever admitted once -- later recurrence must reuse, not re-discover.
        self.assertEqual(len(admissions), 1, f"the concept was (re)admitted more than once: {admissions}")

    def test_subsequent_tasks_reuse_the_admitted_concept(self):
        first_admission = next(r.task_index for r in self.results if r.admitted_family is not None)
        later = self.results[first_admission + 1 :]
        self.assertTrue(later, "no tasks left after admission to measure reuse on")

        # every later task must have QUERIED the library and gotten the admitted concept back...
        for r in later:
            self.assertTrue(r.reused_concept, f"task {r.task_index} did not query/try the admitted concept")
            self.assertEqual(r.challenger_family, "skew_normal")
        queried_matches = sum(1 for r in later if r.challenger_family == "skew_normal")
        self.assertEqual(queried_matches, len(later))

        # ...and, being the true generating family, actually get accepted most of the time: a real,
        # measured reuse-and-succeed rate, not just a query that is silently ignored.
        accepted = [r for r in later if r.verdict is not None and r.verdict.promote]
        reuse_success_rate = len(accepted) / len(later)
        print(f"\n[concept_discovery] reuse success rate on {len(later)} later tasks: {reuse_success_rate:.2%}")
        self.assertGreaterEqual(reuse_success_rate, 0.5, f"reuse success rate too low: {reuse_success_rate:.2%}")

    def test_cumulative_mdl_gain_is_positive_and_grows_once_admitted(self):
        first_admission = next(r.task_index for r in self.results if r.admitted_family is not None)
        cumulative = np.cumsum([r.mdl_gain_bits for r in self.results])
        print(f"\n[concept_discovery] per-task MDL gain (bits): {[round(r.mdl_gain_bits, 2) for r in self.results]}")
        print(f"[concept_discovery] cumulative MDL gain (bits): {[round(c, 2) for c in cumulative]}")

        # before admission, no concept is available to save any bits.
        for c in cumulative[:first_admission]:
            self.assertEqual(c, 0.0)
        # by the end of the corpus the concept has paid for itself many times over, in bits.
        self.assertGreater(cumulative[-1], 0.0)
        self.assertGreater(cumulative[-1], cumulative[first_admission])


class ConceptLibraryReversibilityTest(unittest.TestCase):
    """Library growth must be receipted AND reversible (the card's own phrasing)."""

    def test_revoke_removes_the_family_and_the_query_can_no_longer_recommend_it(self):
        library = ConceptLibrary(base_families=("gaussian",))
        sig = "numeric:right_skew"
        library.admit("skew_normal", {"reason": "unit test"}, task_signature=sig, task_index=0, quality=42.0)
        self.assertIn("skew_normal", library.families)
        self.assertEqual(library.query(sig), "skew_normal")

        library.revoke("skew_normal", task_index=1, reason="unit test revoke")

        self.assertNotIn("skew_normal", library.families)
        self.assertIsNone(library.query(sig))
        self.assertIsNone(library.evidence_for("skew_normal"))

    def test_revocation_is_receipted_in_history(self):
        library = ConceptLibrary(base_families=("gaussian",))
        library.admit("laplace", {}, task_signature="sig", task_index=0, quality=1.0)
        library.revoke("laplace", task_index=2, reason="turned out to be a fluke")

        actions = [(e.action, e.family, e.task_index) for e in library.history]
        self.assertEqual(actions, [("admit", "laplace", 0), ("revoke", "laplace", 2)])
        self.assertEqual(library.history[-1].evidence["reason"], "turned out to be a fluke")

    def test_revoke_unknown_family_raises(self):
        library = ConceptLibrary(base_families=("gaussian",))
        with self.assertRaises(KeyError):
            library.revoke("never_admitted")

    def test_base_families_are_always_present(self):
        library = ConceptLibrary(base_families=("gaussian",))
        self.assertIn("gaussian", library.families)
        self.assertIsNone(library.query("anything"))  # nothing admitted yet


class GateCorrectnessTest(unittest.TestCase):
    """A bad candidate family (worse than the status quo, held out) must fail the gate."""

    def test_bad_candidate_family_is_rejected(self):
        rng = np.random.RandomState(3)
        data = rng.normal(loc=0.0, scale=1.0, size=800)  # genuinely Gaussian data
        train, held_out = data[:480], data[480:]

        champion = _fit_family("gaussian", train)  # the correct family, well fit
        challenger = _fit_family("laplace", train)  # a plausible-looking but worse family for this data

        verdict = challenger_beats_champion(
            champion, challenger, held_out, objective=nll_objective(), require_calibration=False
        )
        self.assertFalse(verdict.promote, "a genuinely worse challenger family was incorrectly promoted")
        self.assertIn(verdict.favored, ("champion", "tie"))

    def test_run_loop_does_not_admit_a_family_that_cannot_beat_the_champion(self):
        # Gaussian data: the "hidden family" signal never appears, so nothing should ever be admitted,
        # however long the corpus runs (there's nothing worth discovering here).
        rng = np.random.RandomState(4)
        tasks = [rng.normal(0.0, 1.0, 1000) for _ in range(8)]
        library, results = run_concept_discovery_loop(tasks)
        self.assertEqual(library.families, ("gaussian",))
        self.assertTrue(all(r.admitted_family is None for r in results))


class TaskSignatureTest(unittest.TestCase):
    def test_same_hidden_family_yields_the_same_signature_across_draws(self):
        rng = np.random.RandomState(5)
        a = stats.skewnorm.rvs(8.0, loc=0.0, scale=1.5, size=500, random_state=rng)
        b = stats.skewnorm.rvs(8.0, loc=0.0, scale=1.5, size=500, random_state=rng)
        self.assertEqual(task_signature(a), task_signature(b))

    def test_symmetric_data_gets_a_different_signature_bucket(self):
        rng = np.random.RandomState(6)
        symmetric = rng.normal(0.0, 1.0, 500)
        skewed = stats.skewnorm.rvs(8.0, loc=0.0, scale=1.5, size=500, random_state=rng)
        self.assertNotEqual(task_signature(symmetric), task_signature(skewed))


class TrainFracValidationTest(unittest.TestCase):
    """``train_frac`` outside ``(0, 1)`` must be rejected up front, not silently misapplied -- the old
    ``max(8, int(len(data) * train_frac))`` split absorbed an out-of-range fraction into a floor value
    with no relationship to what was requested (e.g. ``train_frac=-0.5`` silently ran as if 8 rows had
    been asked for)."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_negative_train_frac_raises(self):
        tasks = [self.rng.normal(0.0, 1.0, 100)]
        with self.assertRaisesRegex(ValueError, "train_frac must be"):
            run_concept_discovery_loop(tasks, train_frac=-0.5)

    def test_train_frac_above_one_raises(self):
        tasks = [self.rng.normal(0.0, 1.0, 100)]
        with self.assertRaisesRegex(ValueError, "train_frac must be"):
            run_concept_discovery_loop(tasks, train_frac=1.5)

    def test_train_frac_zero_raises(self):
        # The interval is open: train_frac=0.0 would request an empty train split.
        tasks = [self.rng.normal(0.0, 1.0, 100)]
        with self.assertRaisesRegex(ValueError, "train_frac must be"):
            run_concept_discovery_loop(tasks, train_frac=0.0)

    def test_train_frac_one_raises(self):
        # The interval is open: train_frac=1.0 would request an empty held-out split.
        tasks = [self.rng.normal(0.0, 1.0, 100)]
        with self.assertRaisesRegex(ValueError, "train_frac must be"):
            run_concept_discovery_loop(tasks, train_frac=1.0)


class HoldoutSplitSizeTest(unittest.TestCase):
    """A task too small to yield a non-empty held-out split must raise a clear, actionable error
    instead of silently starving the holdout -- pre-fix, ``max(8, int(len(data) * train_frac))`` always
    claimed all 8-or-fewer rows for training, and the empty held-out set then blew up deep inside
    ``mmd()`` with an error that gives no hint the real problem is task size vs. train_frac."""

    def setUp(self):
        self.rng = np.random.RandomState(7)

    def test_eight_row_task_raises_instead_of_emptying_holdout(self):
        tasks = [self.rng.normal(0.0, 1.0, 8)]
        with self.assertRaisesRegex(ValueError, "min_train"):
            run_concept_discovery_loop(tasks)

    def test_five_row_task_raises_instead_of_emptying_holdout(self):
        tasks = [self.rng.normal(0.0, 1.0, 5)]
        with self.assertRaisesRegex(ValueError, "min_train"):
            run_concept_discovery_loop(tasks)

    def test_one_row_short_of_the_minimum_raises(self):
        tasks = [self.rng.normal(0.0, 1.0, 15)]  # default min_train(8) + min_holdout(8) - 1
        with self.assertRaisesRegex(ValueError, "min_train"):
            run_concept_discovery_loop(tasks)

    def test_exactly_at_the_minimum_succeeds_with_a_nonempty_holdout(self):
        tasks = [self.rng.normal(0.0, 1.0, 16)]  # default min_train(8) + min_holdout(8), exactly
        library, results = run_concept_discovery_loop(tasks)
        self.assertEqual(len(results), 1)
        self.assertTrue(np.isfinite(results[0].discrepancy))

    def test_custom_min_train_and_min_holdout_are_enforced(self):
        tasks = [self.rng.normal(0.0, 1.0, 30)]
        with self.assertRaisesRegex(ValueError, "min_train"):
            run_concept_discovery_loop(tasks, min_train=20, min_holdout=20)  # needs 40, only has 30

    def test_custom_min_train_and_min_holdout_allow_a_smaller_dataset(self):
        tasks = [self.rng.normal(0.0, 1.0, 30)]
        library, results = run_concept_discovery_loop(tasks, min_train=10, min_holdout=10)  # needs 20
        self.assertEqual(len(results), 1)

    def test_normal_sized_dataset_is_unaffected(self):
        # Negative control: a comfortably-sized task must keep working exactly as before the fix.
        tasks = [self.rng.normal(0.0, 1.0, 1000)]
        library, results = run_concept_discovery_loop(tasks)
        self.assertEqual(len(results), 1)
        self.assertTrue(np.isfinite(results[0].discrepancy))


class ModuleCapabilityClaimTest(unittest.TestCase):
    """Doc/naming fix: the module must describe itself as selecting a REGISTERED family, not
    synthesizing a mathematically novel one -- the docstring used to claim it "fits a genuinely new
    family" while the actual mechanism only ever picks among a small hard-coded/registered set."""

    def test_module_docstring_does_not_claim_novel_family_synthesis(self):
        import mixle.evolve.concept_discovery as concept_discovery_module

        doc = concept_discovery_module.__doc__ or ""
        self.assertNotIn("genuinely new family", doc)

    def test_register_family_docstring_count_matches_the_actual_registry(self):
        from mixle.evolve.concept_discovery import known_families, register_family

        self.assertEqual(len(known_families()), 4)
        self.assertIn("four", register_family.__doc__)


if __name__ == "__main__":
    unittest.main()
