"""CARD IMAGINE-a: verified structural proposal at a capacity ceiling.

Benchmark: data drawn from a genuine two-component Gaussian mixture (means -3 and +3, tight
variance) -- a paradigm-shift case: no amount of refitting a SINGLE Gaussian closes the gap (its
capacity ceiling is structural, not a data-size problem), but a 2-component Gaussian mixture can
represent it exactly. The single Gaussian is the "current class"; the mixture is the proposed richer
structure, named with the specific capability the single Gaussian provably lacks.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.task.imagine import CeilingReport, StructuralCandidate, ceiling_report, propose_structure


def _bimodal_data(n, rng, sep=3.0, sigma=0.4):
    labels = rng.randint(0, 2, size=n)
    means = np.where(labels == 0, -sep, sep)
    return (means + rng.normal(scale=sigma, size=n)).tolist()


def _fit_single_gaussian(data):
    return optimize(data, GaussianEstimator(), out=None)


def _fit_two_component_mixture(data, n_restarts=8, base_seed=0):
    # Multi-restart EM: a single fixed seed (previously RandomState(0)) is NOT robust -- for this
    # exact bimodal fixture it collapses to two near-identical, near-zero-mean components (fitting
    # one wide Gaussian over both modes) roughly a third of the time, a real degenerate local
    # optimum that the benchmark's old weak `assertGreater` sanity check didn't actually catch (a
    # degenerate 2-component fit still edges out a single Gaussian on held-out log-density purely
    # from the extra free parameters, without ever representing genuine bimodality). Multi-restart
    # keeps the fit -- and this benchmark -- honest regardless of which seed a future edit picks:
    # try several seeds, keep the one with the best TRAINING log-likelihood (the standard, correct
    # way to guard against EM local optima), not the one that happens to make an assertion pass.
    best_fit, best_ll = None, -np.inf
    for i in range(n_restarts):
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        candidate = optimize(data, est, out=None, max_its=50, rng=np.random.RandomState(base_seed + i))
        ll = float(np.mean([candidate.log_density(x) for x in data]))
        if ll > best_ll:
            best_ll, best_fit = ll, candidate
    return best_fit


def _fit_wider_single_gaussian(data):
    """Decoy: still a single Gaussian (same structural class, no new information) -- just a
    different pseudo_count regularization. Cannot represent bimodality no matter how it's tuned."""
    return optimize(data, GaussianEstimator(pseudo_count=(0.1, 0.1)), out=None)


class ParadigmShiftCeilingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.train = _bimodal_data(300, rng)
        self.held_out = _bimodal_data(150, np.random.RandomState(1))
        # target: strictly between the single- and two-component held-out scores, so the benchmark
        # is honest by construction -- the single Gaussian provably cannot reach it, the mixture can.
        single_probe = _fit_single_gaussian(self.train)
        two_comp_probe = _fit_two_component_mixture(self.train)
        single_score = float(np.mean([single_probe.log_density(x) for x in self.held_out]))
        two_comp_score = float(np.mean([two_comp_probe.log_density(x) for x in self.held_out]))
        self.assertGreater(two_comp_score, single_score)  # sanity: the benchmark is honest
        # A degenerate 2-component fit (both components collapsed onto ~the same mean) can still
        # edge out a single Gaussian on held-out score purely from the extra free parameters,
        # WITHOUT ever representing genuine bimodality -- the assertGreater above alone does not
        # catch that failure mode. Directly check the fit actually recovered two well-separated
        # components (the true generating means are -3.0/+3.0), so this benchmark can't silently
        # pass on a degenerate mixture.
        component_means = sorted(c.mu for c in two_comp_probe.components)
        self.assertGreater(
            component_means[1] - component_means[0],
            2.0,
            "two-component fit did not recover well-separated components -- likely a collapsed "
            "EM local optimum, not genuine bimodal recovery",
        )
        self.target = (single_score + two_comp_score) / 2.0

    def test_single_gaussian_never_meets_target_no_matter_how_its_tuned(self):
        single = _fit_single_gaussian(self.train)
        held_out_score = float(np.mean([single.log_density(x) for x in self.held_out]))
        ceiling = ceiling_report(held_out_score, self.target)
        self.assertFalse(ceiling.met)

    def test_mixture_proposal_verified_and_breaks_the_ceiling(self):
        single = _fit_single_gaussian(self.train)
        held_out_score = float(np.mean([single.log_density(x) for x in self.held_out]))
        ceiling = ceiling_report(held_out_score, self.target)

        candidates = [
            StructuralCandidate(
                "two_component_mixture",
                _fit_two_component_mixture,
                new_information="2-component mixture: represents a bimodal posterior a single Gaussian cannot",
            )
        ]
        result = propose_structure(candidates, self.train, self.held_out, ceiling)
        self.assertEqual(result.breaks_ceiling, "two_component_mixture")
        self.assertTrue(result.verdicts[0].accepted)

    def test_a_same_class_decoy_with_no_new_information_is_rejected(self):
        single = _fit_single_gaussian(self.train)
        held_out_score = float(np.mean([single.log_density(x) for x in self.held_out]))
        ceiling = ceiling_report(held_out_score, self.target)

        candidates = [StructuralCandidate("wider_single_gaussian", _fit_wider_single_gaussian, new_information="")]
        result = propose_structure(candidates, self.train, self.held_out, ceiling)
        self.assertIsNone(result.breaks_ceiling)
        self.assertFalse(result.verdicts[0].accepted)
        self.assertIn("no named new information source", result.verdicts[0].reason)

    def test_unnamed_information_source_is_rejected_even_if_it_would_have_improved_held_out(self):
        # the two-component mixture WOULD improve held-out -- but with new_information left empty,
        # the gate must reject it anyway: improvement alone is never sufficient evidence.
        single = _fit_single_gaussian(self.train)
        held_out_score = float(np.mean([single.log_density(x) for x in self.held_out]))
        ceiling = ceiling_report(held_out_score, self.target)

        candidates = [StructuralCandidate("unnamed_mixture", _fit_two_component_mixture, new_information="")]
        result = propose_structure(candidates, self.train, self.held_out, ceiling)
        self.assertFalse(result.verdicts[0].accepted)
        self.assertIsNone(result.breaks_ceiling)

    def test_ceiling_report_reflects_held_out_not_train(self):
        report = ceiling_report(held_out_score=-5.0, target=-4.0)
        self.assertIsInstance(report, CeilingReport)
        self.assertFalse(report.met)
        self.assertTrue(ceiling_report(held_out_score=-3.0, target=-4.0).met)


if __name__ == "__main__":
    unittest.main()
