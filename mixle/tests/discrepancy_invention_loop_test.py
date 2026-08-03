"""L5: the discrepancy -> invention loop (mixle.task.discrepancy_invention_loop).

Two scenarios, the crux of the whole item:

* a genuine ceiling case -- data drawn from a two-component Gaussian mixture, a single Gaussian
  champion that CANNOT represent it no matter how it's tuned -- must be detected as ceiling-bound (not
  "needs more tuning"), and the loop must find the correct richer structure, gate-adopt it, and leave a
  replayable reasoning chain in the journal.
* a contrast case -- a single Gaussian champion that is merely undertrained (fit on too little data)
  -- must be correctly diagnosed as "tune it", NOT ceiling-bound, with no invention attempted.
"""

import math
import unittest

import numpy as np

from mixle.epistemic.loop import _portfolio_eig_nmc
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio
from mixle.evolve import nll_objective
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.task.design_prior import record_accepted_recipe
from mixle.task.discrepancy_invention_loop import (
    TruncatedProbeLikelihood,
    default_probe_simulate_fn,
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
    # Seeded because EM mixture initialization is random; the loop must retain
    # this one scored fit through proposal, probe, and promotion.
    est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
    return optimize(list(data), est, out=None, max_its=100, rng=np.random.RandomState(2))


def _has_separated_mixture(model):
    components = getattr(model, "components", ())
    return (
        len(components) >= 2
        and max(component.mu for component in components) - min(component.mu for component in components) > 2.0
    )


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
                capability_test=_has_separated_mixture,
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

    def test_decision_stages_use_disjoint_evidence_panels(self):
        result = self._run()
        panels = [set(indices) for indices in result.evidence_indices.values()]
        self.assertEqual(set(result.evidence_indices), {"ceiling", "proposal", "probe", "gate"})
        for i, left in enumerate(panels):
            for right in panels[i + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(len(set().union(*panels)), len(self.held_out))

    def test_scored_stochastic_candidate_is_fit_only_once(self):
        fit_calls = 0

        def counted_fit(data):
            nonlocal fit_calls
            fit_calls += 1
            return _fit_two_component_mixture(data)

        candidate = StructuralCandidate(
            "counted_mixture",
            counted_fit,
            new_information="a separated two-component mixture",
            capability_test=_has_separated_mixture,
        )
        result = run_discrepancy_invention_loop(
            _fit_single_gaussian,
            self.train,
            self.held_out,
            self.target,
            [candidate],
            objective=nll_objective(),
            tuning_variants=[_fit_wider_single_gaussian],
            seed=0,
        )
        self.assertEqual(fit_calls, 1)
        self.assertIsNotNone(result.imagine)
        self.assertIn("counted_mixture", result.imagine.fitted_models)

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


class ProbeSimulationTest(unittest.TestCase):
    def test_exhausted_local_rejection_returns_a_real_model_draw(self):
        class PointMass:
            def sample(self, n):
                return np.full(n, 7.0)

        result = default_probe_simulate_fn(
            Hypothesis("point", PointMass()),
            action=0.0,
            rng=np.random.RandomState(0),
            window=0.1,
        )
        self.assertEqual(result, 7.0)


class _Gauss:
    """Minimal scalar Gaussian with the ``log_density``/``sampler`` surface the probe needs."""

    def __init__(self, mu, sd):
        self.mu, self.sd = float(mu), float(sd)

    def log_density(self, x):
        z = (float(x) - self.mu) / self.sd
        return -0.5 * z * z - math.log(self.sd * math.sqrt(2 * math.pi))

    def sampler(self, seed):
        return _GaussSampler(self, seed)


class _GaussSampler:
    def __init__(self, dist, seed):
        self.dist, self.rng = dist, np.random.RandomState(seed)

    def sample(self, n):
        return self.rng.normal(self.dist.mu, self.dist.sd, size=n)


class _GaussWithCdf(_Gauss):
    def cdf(self, x):
        return 0.5 * (1.0 + math.erf((float(x) - self.mu) / (self.sd * math.sqrt(2.0))))


class ActionConditionedProbeLikelihoodTest(unittest.TestCase):
    """MXR-080-1896: the probe's EIG must score the experiment the probe actually performs.

    Reproduced before the fix: ``default_probe_simulate_fn`` rejection-samples the predictive
    truncated to ``|y - a| <= w``, so its law depends on the action, while the loop scored it with
    ``exp(h.payload.log_density(y))`` -- byte-identical for every action. On a two-Gaussian portfolio
    (``N(0,1)`` vs ``N(3,1)``, ``w=0.5``) with five candidate probe locations that pairing reported
    "EIG" of ``-1.147``/``-1.073``/``+0.107``/``-1.153``/``-1.322`` nats where the true
    action-conditioned EIG was ``+0.041``/``+0.055``/``+0.083``/``+0.026``/``+0.007``. Four of five
    were NEGATIVE, which an expected information gain cannot be.
    """

    WINDOW = 0.5

    def _portfolio(self, sep=3.0):
        hyps = [Hypothesis("champion", _Gauss(0.0, 1.0)), Hypothesis("challenger", _Gauss(sep, 1.0))]
        return HypothesisPortfolio(hyps, np.array([0.5, 0.5]), w_open=0.0)

    def _simulate(self, hypothesis, action, rng):
        return default_probe_simulate_fn(hypothesis, action, rng, window=self.WINDOW)

    @staticmethod
    def _unconditional_likelihood(hypothesis, observation):
        return float(np.exp(hypothesis.payload.log_density(observation)))

    def test_the_likelihood_now_depends_on_the_action(self):
        # The crux of the finding: the OLD density was the same number for every action.
        likelihood = TruncatedProbeLikelihood(window=self.WINDOW)
        hypothesis = Hypothesis("champion", _Gauss(0.0, 1.0))
        at_zero = likelihood(hypothesis, 0.0, 0.2)
        at_half = likelihood(hypothesis, 0.5, 0.2)
        self.assertGreater(at_zero, 0.0)
        self.assertGreater(at_half, 0.0)
        self.assertNotAlmostEqual(at_zero, at_half, places=3)
        # Outside the accepted set the declared experiment cannot produce the observation at all.
        self.assertEqual(likelihood(hypothesis, 6.0, 0.2), 0.0)

    def test_the_quadrature_normalizer_matches_an_exact_cdf_even_in_a_far_tail(self):
        # The normalizer is legitimately ~1e-8 for a probe in a hypothesis's tail -- the regime a
        # Monte Carlo estimate of it would read as exactly zero. Quadrature must not.
        without_cdf = TruncatedProbeLikelihood(window=self.WINDOW)
        with_cdf = TruncatedProbeLikelihood(window=self.WINDOW)
        for action in (-3.0, 0.0, 6.0):
            quadrature = without_cdf.acceptance_mass(Hypothesis("h", _Gauss(0.0, 1.0)), action)
            exact = with_cdf.acceptance_mass(Hypothesis("h", _GaussWithCdf(0.0, 1.0)), action)
            with self.subTest(action=action):
                self.assertGreater(exact, 0.0)
                self.assertLess(abs(quadrature - exact) / exact, 1e-6)

    def test_action_conditioned_eig_is_never_negative_where_the_old_pairing_was(self):
        portfolio = self._portfolio()
        for action, old_value in ((-3.0, -1.147), (0.0, -1.073), (3.0, -1.153), (6.0, -1.322)):
            eig = _portfolio_eig_nmc(
                portfolio,
                action,
                self._unconditional_likelihood,
                self._simulate,
                np.random.RandomState(7),
                n_outer=200,
                n_inner=200,
                action_likelihood=TruncatedProbeLikelihood(window=self.WINDOW),
            )
            with self.subTest(action=action, old=old_value):
                self.assertGreaterEqual(eig, 0.0)
                self.assertLess(eig, 1.0)

    def test_the_old_action_blind_pairing_still_reproduces_the_negative_eig(self):
        # Guards the reproduction itself: without the action-conditioned density the same call still
        # produces the impossible negative number, so this test fails loudly if the pairing is ever
        # silently restored.
        eig = _portfolio_eig_nmc(
            self._portfolio(),
            -3.0,
            self._unconditional_likelihood,
            self._simulate,
            np.random.RandomState(7),
            n_outer=200,
            n_inner=200,
        )
        self.assertLess(eig, -0.5)

    def test_a_zero_width_window_yields_no_likelihood_rather_than_dividing_by_zero(self):
        # A zero-width accepted set carries no probability, so the truncated law does not exist. The
        # honest report is "this hypothesis cannot produce this outcome", not a ZeroDivisionError.
        likelihood = TruncatedProbeLikelihood(window=0.0)
        self.assertEqual(likelihood(Hypothesis("h", _Gauss(0.0, 1.0)), 0.0, 0.0), 0.0)

    def test_quadrature_node_counts_that_simpson_cannot_use_are_rejected(self):
        for bad in (2, 4, 128, 1, 0, -3, True):
            with self.subTest(nodes=repr(bad)), self.assertRaises(ValueError):
                TruncatedProbeLikelihood(window=1.0, quadrature_nodes=bad)

    def test_the_loop_end_to_end_reports_a_non_negative_probe_eig(self):
        # Negative control at the integration level: the real loop still runs, still adopts the
        # richer structure, and its reported EIG is now a number an information gain can take.
        rng = np.random.RandomState(0)
        train = _bimodal_data(300, rng)
        held_out = _bimodal_data(150, np.random.RandomState(1))
        single = float(np.mean([_fit_single_gaussian(train).log_density(x) for x in held_out]))
        mixture = float(np.mean([_fit_two_component_mixture(train).log_density(x) for x in held_out]))
        result = run_discrepancy_invention_loop(
            _fit_single_gaussian,
            train,
            held_out,
            (single + mixture) / 2.0,
            [
                StructuralCandidate(
                    "two_component_mixture",
                    _fit_two_component_mixture,
                    new_information="2-component mixture: represents a bimodal posterior a single Gaussian cannot",
                    capability_test=_has_separated_mixture,
                )
            ],
            objective=nll_objective(),
            tuning_variants=[_fit_wider_single_gaussian],
            probe_reweight_n=1,
            seed=0,
        )
        self.assertEqual(result.adopted_structure, "two_component_mixture")
        self.assertIsNotNone(result.probe_eig)
        self.assertGreaterEqual(result.probe_eig, 0.0)
        # Pre-fix this same call reported 0.2806 nats; the action-conditioned law plus the
        # independent re-estimate put it at ~0.032. The number MOVED -- see the finding's report.
        self.assertLess(result.probe_eig, 0.1)


if __name__ == "__main__":
    unittest.main()
