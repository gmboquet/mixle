"""L5: the discrepancy -> invention loop (mixle.task.discrepancy_invention_loop).

Two scenarios, the crux of the whole item:

* a genuine ceiling case -- data drawn from a two-component Gaussian mixture, a single Gaussian
  champion that CANNOT represent it no matter how it's tuned -- must be detected as ceiling-bound (not
  "needs more tuning"), and the loop must find the correct richer structure, gate-adopt it, and leave a
  replayable reasoning chain in the journal.
* a contrast case -- a single Gaussian champion that is merely undertrained (fit on too little data)
  -- must be correctly diagnosed as "tune it", NOT ceiling-bound, with no invention attempted.
"""

import unittest

import numpy as np

from mixle.evolve import nll_objective
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.task.design_prior import record_accepted_recipe
from mixle.task.discrepancy_invention_loop import (
    reconstruct_reasoning_chain,
    run_discrepancy_invention_loop,
    score_design_prior_surprise,
)
from mixle.task.edge import DesignModel
from mixle.task.imagine import StructuralCandidate


def _bimodal_data(n, rng, sep=3.0, sigma=0.4):
    labels = rng.randint(0, 2, size=n)
    means = np.where(labels == 0, -sep, sep)
    return (means + rng.normal(scale=sigma, size=n)).tolist()


def _fit_single_gaussian(data):
    return optimize(list(data), GaussianEstimator(), out=None)


def _fit_wider_single_gaussian(data):
    """Decoy tuning variant: still a single Gaussian, different regularization -- same structural
    family, cannot represent bimodality no matter how it's tuned."""
    return optimize(list(data), GaussianEstimator(pseudo_count=(0.1, 0.1)), out=None)


def _fit_two_component_mixture(data):
    # seeded: EM mixture init is random, and this fit is called more than once (this test's own probe,
    # then again inside the loop's propose_structure/probe/gate stages) -- must land in the same, truly
    # bimodal optimum every time (some seeds collapse to a degenerate near-unimodal local optimum).
    est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
    return optimize(list(data), est, out=None, max_its=100, rng=np.random.RandomState(2))


class CeilingBoundInventionTest(unittest.TestCase):
    """The core acceptance criterion: a genuinely out-of-class phenomenon."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.train = _bimodal_data(300, rng)
        self.held_out = _bimodal_data(150, np.random.RandomState(1))
        single_probe = _fit_single_gaussian(self.train)
        mixture_probe = _fit_two_component_mixture(self.train)
        single_score = float(np.mean([single_probe.log_density(x) for x in self.held_out]))
        mixture_score = float(np.mean([mixture_probe.log_density(x) for x in self.held_out]))
        self.assertGreater(mixture_score, single_score)  # sanity: benchmark is honest
        self.target = (single_score + mixture_score) / 2.0
        self.candidates = [
            StructuralCandidate(
                "two_component_mixture",
                _fit_two_component_mixture,
                new_information="2-component mixture: represents a bimodal posterior a single Gaussian cannot",
            )
        ]

    def _run(self, design=None):
        return run_discrepancy_invention_loop(
            _fit_single_gaussian,
            self.train,
            self.held_out,
            self.target,
            self.candidates,
            objective=nll_objective(),
            tuning_variants=[_fit_wider_single_gaussian],
            design=design,
            seed=0,
        )

    def test_a_ceiling_bound_phenomenon_is_detected_as_ceiling_bound_not_tune_it(self):
        result = self._run()
        self.assertEqual(result.verdict, "ceiling_bound")
        self.assertTrue(result.ceiling_bound)
        self.assertFalse(result.ceiling.met)

    def test_the_search_finds_the_correct_novel_composition_and_it_is_gate_adopted(self):
        result = self._run()
        self.assertIsNotNone(result.imagine)
        self.assertEqual(result.imagine.breaks_ceiling, "two_component_mixture")
        accepted = {v.name: v for v in result.imagine.verdicts}
        self.assertTrue(accepted["two_component_mixture"].accepted)
        self.assertGreater(accepted["two_component_mixture"].held_out_score, result.ceiling.held_out_score)
        # gate: a genuinely better structure should be promoted over the champion.
        self.assertIsNotNone(result.gate_verdict)
        self.assertEqual(result.gate_verdict.favored, "challenger")
        self.assertTrue(result.gate_verdict.promote)
        self.assertEqual(result.adopted_structure, "two_component_mixture")

    def test_eig_probe_selects_a_real_distinguishing_action(self):
        result = self._run()
        self.assertIsNotNone(result.probe_action)
        self.assertIsNotNone(result.probe_eig)

    def test_journal_reconstructs_the_full_reasoning_chain_in_order(self):
        result = self._run()
        chain = reconstruct_reasoning_chain(result.journal)
        self.assertEqual(len(chain), 5)
        self.assertIn("discrepancy detected", chain[0])
        self.assertIn("ceiling verdict: ceiling_bound", chain[1])
        self.assertIn("structure proposal", chain[2])
        self.assertIn("two_component_mixture", chain[2])
        self.assertIn("EIG probe", chain[3])
        self.assertIn("gate verdict", chain[4])
        self.assertIn("two_component_mixture", chain[4])
        self.assertTrue(result.journal.verify())
        trajectory = result.journal.replay()
        self.assertEqual(len(trajectory), 5)
        # the belief trajectory grows real new hypotheses once proposals are folded in.
        self.assertEqual(len(trajectory[0]), 1)  # champion-only, before any proposal
        self.assertGreaterEqual(len(trajectory[3]), 2)  # champion + accepted candidate(s), at the probe stage

    def test_novelty_scored_as_design_prior_surprise_unprecedented_family_is_maximally_surprising(self):
        result = self._run(design=DesignModel(signature="fresh", n_constraints=0))
        self.assertIn("two_component_mixture", result.novelty_scores)
        self.assertEqual(result.novelty_scores["two_component_mixture"], float("inf"))

    def test_novelty_is_finite_surprise_relative_to_a_seeded_design_prior(self):
        design = DesignModel(signature="seeded", n_constraints=0)
        record_accepted_recipe(design, [0.0], -1.0, [], family="two_component_mixture")
        result = self._run(design=design)
        surprise = result.novelty_scores["two_component_mixture"]
        self.assertTrue(np.isfinite(surprise))
        winning_verdict = next(v for v in result.imagine.verdicts if v.name == "two_component_mixture")
        expected = winning_verdict.held_out_score - (-1.0)
        self.assertAlmostEqual(surprise, expected, places=6)

    def test_score_design_prior_surprise_helper_matches_loop_output(self):
        design = DesignModel(signature="direct", n_constraints=0)
        record_accepted_recipe(design, [0.0], 0.5, [], family="fam")
        self.assertAlmostEqual(score_design_prior_surprise("fam", 0.8, design), 0.3, places=6)
        self.assertEqual(score_design_prior_surprise("never_tried", 0.8, design), float("inf"))


class TuneItContrastTest(unittest.TestCase):
    """The contrast case: same structural family, just needs tuning/more data -- NOT ceiling-bound."""

    def setUp(self):
        rng = np.random.RandomState(7)
        self.true_mu, self.true_sigma2 = 3.0, 1.5
        self.small_train = list(rng.normal(self.true_mu, np.sqrt(self.true_sigma2), 15))
        self.large_train = list(rng.normal(self.true_mu, np.sqrt(self.true_sigma2), 3000))
        self.held_out = list(np.random.RandomState(8).normal(self.true_mu, np.sqrt(self.true_sigma2), 500))

        undertrained = _fit_single_gaussian(self.small_train)
        well_trained = _fit_single_gaussian(self.large_train)
        undertrained_score = float(np.mean([undertrained.log_density(x) for x in self.held_out]))
        well_trained_score = float(np.mean([well_trained.log_density(x) for x in self.held_out]))
        self.assertGreater(well_trained_score, undertrained_score)  # sanity
        # target: close to (but not exactly at) the well-tuned score, so the champion alone falls
        # short but tuning within the SAME family closes almost all of the gap.
        self.target = well_trained_score - 0.01

    def _champion_fit(self, data):
        # ignore ``data`` (the loop always passes ``train``): the "champion" is deliberately the
        # undertrained fit, and the tuning variant below is the same family with more data.
        return _fit_single_gaussian(self.small_train)

    def _tuned_fit(self, data):
        return _fit_single_gaussian(self.large_train)

    def test_a_merely_undertrained_model_is_diagnosed_as_tune_it_not_ceiling_bound(self):
        result = run_discrepancy_invention_loop(
            self._champion_fit,
            self.small_train,
            self.held_out,
            self.target,
            candidates=[],
            objective=nll_objective(),
            tuning_variants=[self._tuned_fit],
            seed=0,
        )
        self.assertEqual(result.verdict, "tune")
        self.assertFalse(result.ceiling_bound)
        self.assertIsNone(result.imagine)
        self.assertIsNone(result.adopted_structure)
        self.assertIsNone(result.gate_verdict)

    def test_no_invention_machinery_is_invoked_when_tuning_suffices(self):
        result = run_discrepancy_invention_loop(
            self._champion_fit,
            self.small_train,
            self.held_out,
            self.target,
            candidates=[],
            objective=nll_objective(),
            tuning_variants=[self._tuned_fit],
            seed=0,
        )
        chain = reconstruct_reasoning_chain(result.journal)
        self.assertEqual(len(chain), 2)
        self.assertIn("discrepancy detected", chain[0])
        self.assertIn("ceiling verdict: tune", chain[1])
        self.assertIn("no invention needed", chain[1])
        self.assertNotIn("ceiling_bound", chain[1].split("--")[0])  # the verdict itself isn't ceiling_bound


if __name__ == "__main__":
    unittest.main()
