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


if __name__ == "__main__":
    unittest.main()
