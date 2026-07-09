"""The language<->belief bridge (roadmap M5, part (c))."""

import unittest

import numpy as np

from mixle.reason.language_bridge import ABSTAIN, Claim, PosteriorDescriber, claim_score, parse_evidence
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

SCHEMA = {"text": "categorical", "brightness": "numeric"}


def _toy_extractor(sentence: str) -> dict:
    """A deliberately simple, deterministic keyword/regex extractor -- stands in for whatever real
    parser (rule-based or a calibrated ``solve_structured`` student) a caller would plug in; this
    module's own contract is validating the extractor's OUTPUT against the declared schema, not how
    the extraction itself happens."""
    out: dict = {}
    for label in ("cat", "dog"):
        if label in sentence:
            out["text"] = label
    import re

    m = re.search(r"brightness(?: is)?(?: of)?(?: about)? ([0-9.]+)", sentence)
    if m:
        out["brightness"] = float(m.group(1))
    return out


class ParseEvidenceTest(unittest.TestCase):
    """Acceptance criterion (c): NL -> evidence reproduces a hand-specified dict bit-for-bit."""

    def test_reproduces_hand_specified_evidence_bit_for_bit(self):
        sentence = "the image looks like a dog and the brightness is about 3.5"
        got = parse_evidence(sentence, SCHEMA, _toy_extractor)
        self.assertEqual(got, {"text": "dog", "brightness": 3.5})

    def test_categorical_normalized_to_str(self):
        got = parse_evidence("a cat, brightness 1.0", SCHEMA, _toy_extractor)
        self.assertEqual(got["text"], "cat")
        self.assertIsInstance(got["text"], str)

    def test_partial_evidence_is_fine(self):
        got = parse_evidence("brightness is 2.0, no animal mentioned", SCHEMA, _toy_extractor)
        self.assertEqual(got, {"brightness": 2.0})

    def test_undeclared_field_rejected(self):
        def bad_extractor(_x):
            return {"smell": "strong"}

        with self.assertRaises(ValueError):
            parse_evidence("whatever", SCHEMA, bad_extractor)

    def test_wrong_type_for_numeric_field_rejected(self):
        def bad_extractor(_x):
            return {"brightness": "very bright"}

        with self.assertRaises(ValueError):
            parse_evidence("whatever", SCHEMA, bad_extractor)

    def test_extractor_must_return_a_dict(self):
        with self.assertRaises(TypeError):
            parse_evidence("whatever", SCHEMA, lambda _x: ["dog", 3.5])

    def test_empty_schema_rejected(self):
        with self.assertRaises(ValueError):
            parse_evidence("whatever", {}, _toy_extractor)


class ClaimScoreStandaloneTest(unittest.TestCase):
    """``claim_score`` is a reusable primitive independent of ``PosteriorDescriber`` -- the shape B2's
    claim-checking needs: score an already-emitted claim against the posterior it describes."""

    def test_well_supported_claim_scores_higher_than_a_wrong_one(self):
        posterior = GaussianDistribution(mu=10.0, sigma2=0.01)
        good = Claim(field="x", lo=9.5, hi=10.5)
        bad = Claim(field="x", lo=-5.0, hi=-4.0)
        self.assertGreater(claim_score(good, posterior), claim_score(bad, posterior))

    def test_cached_probe_reused_without_a_posterior(self):
        posterior = GaussianDistribution(mu=0.0, sigma2=1e-6)
        rng = np.random.RandomState(0)
        probe = tuple((rng.normal(0.0, 1e-3, size=50)).tolist())
        claim = Claim(field="x", lo=-0.5, hi=0.5, probe=probe)
        self.assertGreaterEqual(claim_score(claim), 0.85)

    def test_needs_either_posterior_or_cached_probe(self):
        with self.assertRaises(ValueError):
            claim_score(Claim(field="x", lo=0.0, hi=1.0))


class PosteriorDescriberTest(unittest.TestCase):
    """Acceptance criterion (d): ``describe`` abstains when the posterior is too diffuse relative to
    the caller's declared precision (``tol``) to support any candidate claim."""

    def setUp(self):
        self.tol = 0.5
        self.describer = PosteriorDescriber("temperature", tol=self.tol, k=3, alpha=0.2, n_probe=200, seed=0)
        rng = np.random.RandomState(0)
        # calibration set: a mix of posterior sharpnesses (sigma2 from well-within-tol up to a few
        # multiples of tol) at varied means, each paired with a value REALIZED by an actual draw from
        # that posterior (not its parametric mean) -- the realistic "score a generated answer against
        # what actually happened" conformal setup, not an estimator-bias check.
        sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
        cal = []
        for _ in range(150):
            mu = float(rng.uniform(-20.0, 20.0))
            sigma2 = float(rng.choice(sigmas))
            g = GaussianDistribution(mu=mu, sigma2=sigma2)
            realized = float(g.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample())
            cal.append((g, realized))
        self.describer.calibrate(cal, seed=1)

    def test_sharp_unseen_posterior_gets_a_confident_claim(self):
        posterior = GaussianDistribution(mu=7.0, sigma2=0.01)
        claim = self.describer.describe(posterior, seed=2)
        self.assertIsNotNone(claim)
        self.assertIsInstance(claim, Claim)
        self.assertTrue(claim.contains(7.0))

    def test_diffuse_posterior_abstains(self):
        # spread ~200x tol: no candidate width (up to 10*tol) can meaningfully cover this posterior's
        # mass without also covering the rest of the plausible range -- the honest answer is abstain.
        posterior = GaussianDistribution(mu=7.0, sigma2=10000.0)
        claim = self.describer.describe(posterior, seed=3)
        self.assertIs(claim, ABSTAIN)

    def test_invalid_tol_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=0.0)

    def test_k_exceeding_width_multiples_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, k=99)


if __name__ == "__main__":
    unittest.main()
