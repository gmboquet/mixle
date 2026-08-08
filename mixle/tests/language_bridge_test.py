"""The language<->belief bridge (roadmap M5, part (c))."""

import unittest

import numpy as np

from mixle.reason.language_bridge import (
    ABSTAIN,
    Claim,
    PosteriorDescriber,
    SchemaField,
    _sample_scalar,
    claim_score,
    parse_evidence,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

# "text" is explicitly OPTIONAL (MXR-080-0289: required-ness must be expressible, not implicit for
# every field) -- exercised by test_partial_evidence_is_fine below; "brightness" uses the plain-string
# shorthand, which defaults to required=True.
SCHEMA = {"text": SchemaField(kind="categorical", required=False), "brightness": "numeric"}


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


class EvidenceSchemaValidationTest(unittest.TestCase):
    """Regression coverage for MXR-080-0289: evidence parsing previously validated only fields the
    extractor happened to return -- an invalid schema kind on an omitted field was never detected,
    "required" could not be expressed at all, non-finite numerics passed straight through, and any
    categorical value (including ``None``) was silently ``str()``-coerced into a fabricated label."""

    def test_required_field_missing_is_rejected(self):
        schema = {"a": "numeric", "b": "categorical"}  # both required by default (shorthand)
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"a": 1.0})  # "b" omitted

    def test_optional_field_may_be_omitted(self):
        schema = {"a": "numeric", "b": SchemaField(kind="categorical", required=False)}
        got = parse_evidence("whatever", schema, lambda _x: {"a": 1.0})
        self.assertEqual(got, {"a": 1.0})

    def test_invalid_schema_kind_detected_even_for_a_field_the_extractor_omits(self):
        # "b" is never in the extractor's output at all -- the bad kind must still be caught, because
        # the schema itself is validated before the extractor ever runs.
        schema = {"a": "numeric", "b": "not_a_real_kind"}
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"a": 1.0})

    def test_invalid_schema_kind_rejected_at_construction_even_without_calling_parse_evidence(self):
        with self.assertRaises(ValueError):
            SchemaField(kind="not_a_real_kind")

    def test_numeric_nan_rejected(self):
        schema = {"brightness": "numeric"}
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"brightness": float("nan")})

    def test_numeric_infinity_rejected(self):
        schema = {"brightness": "numeric"}
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"brightness": float("inf")})
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"brightness": float("-inf")})

    def test_categorical_none_is_rejected_not_silently_stringified(self):
        schema = {"text": "categorical"}
        with self.assertRaises(ValueError) as ctx:
            parse_evidence("whatever", schema, lambda _x: {"text": None})
        self.assertNotIn("'None'", str(ctx.exception))  # must not have been coerced to the string "None"

    def test_categorical_arbitrary_object_is_rejected_not_silently_stringified(self):
        schema = {"text": "categorical"}
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"text": ["not", "a", "string"]})
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"text": 5})

    def test_categorical_value_outside_declared_domain_rejected(self):
        schema = {"text": SchemaField(kind="categorical", categories=frozenset({"cat", "dog"}))}
        with self.assertRaises(ValueError):
            parse_evidence("whatever", schema, lambda _x: {"text": "bird"})

    def test_categorical_value_inside_declared_domain_accepted(self):
        schema = {"text": SchemaField(kind="categorical", categories=frozenset({"cat", "dog"}))}
        got = parse_evidence("whatever", schema, lambda _x: {"text": "cat"})
        self.assertEqual(got, {"text": "cat"})

    def test_categories_must_be_non_empty_when_declared(self):
        with self.assertRaises(ValueError):
            SchemaField(kind="categorical", categories=frozenset())

    def test_categories_only_valid_for_categorical_kind(self):
        with self.assertRaises(ValueError):
            SchemaField(kind="numeric", categories=frozenset({"a"}))


class ClaimValidationTest(unittest.TestCase):
    """Regression coverage for MXR-080-0290: ``Claim`` previously permitted NaN/infinite bounds and
    ``hi < lo`` (a reversed interval), so a "validated" claim could be semantically meaningless."""

    def test_nan_lo_rejected(self):
        with self.assertRaises(ValueError):
            Claim(field="x", lo=float("nan"), hi=1.0)

    def test_nan_hi_rejected(self):
        with self.assertRaises(ValueError):
            Claim(field="x", lo=0.0, hi=float("nan"))

    def test_infinite_bounds_rejected(self):
        with self.assertRaises(ValueError):
            Claim(field="x", lo=float("-inf"), hi=1.0)
        with self.assertRaises(ValueError):
            Claim(field="x", lo=0.0, hi=float("inf"))

    def test_hi_less_than_lo_rejected(self):
        with self.assertRaises(ValueError):
            Claim(field="x", lo=5.0, hi=1.0)

    def test_empty_field_name_rejected(self):
        with self.assertRaises(ValueError):
            Claim(field="", lo=0.0, hi=1.0)

    def test_zero_width_point_interval_is_allowed(self):
        # lo == hi is a degenerate but legitimate interval (Claim.text() even special-cases it as
        # "approximately X") -- only hi < lo and non-finite bounds are rejected.
        claim = Claim(field="x", lo=5.0, hi=5.0)
        self.assertEqual(claim.width, 0.0)
        self.assertTrue(claim.contains(5.0))


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


class SampleScalarTest(unittest.TestCase):
    """_sample_scalar's posterior.sample(n, seed=...) / posterior.sample(n) dispatch."""

    def test_a_bug_inside_sample_is_not_masked_by_a_retry_without_seed(self):
        # sample(self, n, seed=None) accepts seed, so it must be called with it exactly once; a
        # TypeError from inside its own body must propagate, not be swallowed and silently retried
        # as sample(n) (which would draw an entirely separate, uncontrolled-seed sample).
        calls = []

        class BuggyPosterior:
            def sample(self, n, seed=None):
                calls.append((n, seed))
                return None + n  # an internal bug unrelated to whether seed is accepted

        with self.assertRaises(TypeError):
            _sample_scalar(BuggyPosterior(), 5, seed=3)
        self.assertEqual(calls, [(5, 3)])  # called once, with seed -- never retried as sample(n)

    def test_sampler_without_seed_support_falls_back_correctly(self):
        class LegacyPosterior:
            def sample(self, n):
                return np.arange(n, dtype=float)

        out = _sample_scalar(LegacyPosterior(), 4, seed=3)
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0])

    def test_multidimensional_array_rejected_not_silently_flattened(self):
        # MXR-080-0290: a matrix must not be silently averaged over via boolean broadcasting
        # downstream (claim_score's np.mean over a 2D boolean array) -- it is a shape violation.
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            _sample_scalar(matrix, 5, None)

    def test_empty_array_rejected_not_left_to_produce_nan_support(self):
        with self.assertRaises(ValueError):
            _sample_scalar(np.array([]), 5, None)

    def test_empty_sequence_rejected(self):
        with self.assertRaises(ValueError):
            _sample_scalar([], 5, None)

    def test_1d_array_and_plain_sequence_agree_via_the_same_validated_path(self):
        arr_out = _sample_scalar(np.array([1.0, 2.0, 3.0]), 5, None)
        seq_out = _sample_scalar([1.0, 2.0, 3.0], 5, None)
        np.testing.assert_allclose(arr_out, seq_out)

    def test_sequence_with_a_non_scalar_non_singleton_element_rejected(self):
        # a plain pass-through sequence must be rejected by the SAME per-element scalar-or-1-tuple
        # rule a live posterior's .sample() output is held to -- not silently accepted, and not
        # misrouted into "do not know how to sample from a list".
        with self.assertRaises(ValueError):
            _sample_scalar([1.0, (2.0, 3.0), 4.0], 5, None)

    def test_n_samples_zero_rejected(self):
        with self.assertRaises(ValueError):
            _sample_scalar(np.array([1.0, 2.0]), 0, None)

    def test_n_samples_negative_rejected(self):
        with self.assertRaises(ValueError):
            _sample_scalar(np.array([1.0, 2.0]), -3, None)

    def test_claim_score_propagates_n_samples_validation(self):
        posterior = GaussianDistribution(mu=0.0, sigma2=1.0)
        claim = Claim(field="x", lo=-1.0, hi=1.0)
        with self.assertRaises(ValueError):
            claim_score(claim, posterior, n_samples=0)


class PosteriorDescriberTest(unittest.TestCase):
    """Acceptance criterion (d): ``describe`` abstains when the posterior is too diffuse relative to
    the caller's declared precision (``tol``) to support any candidate claim."""

    def setUp(self):
        self.tol = 0.5
        # Open-ended candidates are certified as a selective decision: does the
        # top-scored interval cover the truth when its margin clears the gate?
        self.describer = PosteriorDescriber("temperature", tol=self.tol, k=3, alpha=0.1, n_probe=200, seed=0)
        rng = np.random.RandomState(0)
        # calibration set: a mix of posterior sharpnesses (sigma2 from well-within-tol up to a few
        # multiples of tol) at varied means, each paired with a value REALIZED by an actual draw from
        # that posterior (not its parametric mean) -- the realistic "score a generated answer against
        # what actually happened" conformal setup, not an estimator-bias check. 300 (not 150) rows for
        # a stable quantile estimate at this small alpha (verified robust across 30 independent
        # calibration-set draws: qhat cleared the ~0.48 admission ceiling on every one).
        sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
        cal = []
        for _ in range(300):
            mu = float(rng.uniform(-20.0, 20.0))
            sigma2 = float(rng.choice(sigmas))
            g = GaussianDistribution(mu=mu, sigma2=sigma2)
            realized = float(g.sampler(seed=int(rng.randint(0, 2**31 - 1))).sample())
            cal.append((g, realized))
        self.describer.calibrate(cal, seed=1)

    def test_sharp_unseen_posterior_is_accepted_only_if_the_gate_certifies_it(self):
        posterior = GaussianDistribution(mu=7.0, sigma2=0.01)
        # no per-call seed: the certificate covers only the prompt-derived schedule
        # (STAT-RR17-07), and serving under it is deterministic per posterior anyway
        claim = self.describer.describe(posterior)
        if claim is not ABSTAIN:
            self.assertIsInstance(claim, Claim)
            self.assertTrue(claim.contains(7.0))

    def test_diffuse_posterior_abstains(self):
        # spread ~200x tol: no candidate width (up to 10*tol) can meaningfully cover this posterior's
        # mass without also covering the rest of the plausible range -- the honest answer is abstain.
        posterior = GaussianDistribution(mu=7.0, sigma2=10000.0)
        claim = self.describer.describe(posterior)
        self.assertIs(claim, ABSTAIN)

    def test_invalid_tol_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=0.0)

    def test_k_exceeding_width_multiples_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, k=99)


class PosteriorDescriberConstructionValidationTest(unittest.TestCase):
    """Regression coverage for MXR-080-0292: ``k``, ``alpha``, ``n_probe``, ``field_name``, and
    ``width_multiples`` were not required to be positive, finite, ordered, or non-empty. A
    zero/negative multiplier produced a point or reversed :class:`Claim`; ``k<=0`` or an empty
    ``width_multiples`` let an empty candidate set reach the generator; ``tol``'s bare ``<= 0`` check
    could not catch NaN (a NaN comparison is always False)."""

    def test_k_zero_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, k=0)

    def test_k_negative_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, k=-1)

    def test_alpha_zero_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, alpha=0.0)

    def test_alpha_one_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, alpha=1.0)

    def test_alpha_negative_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, alpha=-0.5)

    def test_alpha_greater_than_one_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, alpha=1.5)

    def test_n_probe_zero_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, n_probe=0)

    def test_n_probe_negative_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, n_probe=-5)

    def test_field_name_empty_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("", tol=1.0)

    def test_width_multiples_empty_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, k=1, width_multiples=())

    def test_width_multiples_zero_rejected(self):
        # mult=0 would otherwise create a POINT interval (lo == hi == mean) -- a degenerate candidate.
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(0.0, 3.0, 10.0))

    def test_width_multiples_negative_rejected(self):
        # a negative mult would otherwise create a REVERSED interval (lo > hi).
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(-1.0, 3.0, 10.0))

    def test_width_multiples_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(float("nan"), 3.0, 10.0))
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(1.0, float("inf"), 10.0))

    def test_width_multiples_not_strictly_increasing_rejected(self):
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(10.0, 1.0, 3.0))  # unsorted
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=1.0, width_multiples=(1.0, 1.0, 3.0))  # duplicate

    def test_tol_nan_rejected(self):
        # tol's old bare `<= 0` check could not catch NaN (NaN comparisons are always False).
        with self.assertRaises(ValueError):
            PosteriorDescriber("x", tol=float("nan"))

    def test_calibration_set_with_non_finite_truth_rejected(self):
        describer = PosteriorDescriber("x", tol=1.0, k=1, width_multiples=(1.0,), n_probe=10)
        with self.assertRaises(ValueError):
            describer.calibrate([(_ConstantPosterior(5.0), float("nan"))])


class _ConstantPosterior:
    """A deterministic mock posterior: always samples the same constant regardless of seed. Standing
    in for "one fitted/mock posterior reused across several synthetic calibration points" -- the
    realistic case named in the id()-keyed-truth-lookup bug below -- while keeping candidate
    generation/scoring identical no matter which (or how many) object instances are used, so the
    ONLY thing that can differ between a reused-object and a distinct-objects calibration run is
    whether each row's true_value was looked up correctly."""

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n: int, seed=None):
        return [self.value] * n


def _risk_receipt_for(calibration_set, *, tol: float = 1.0, seed: int = 0) -> dict:
    describer = PosteriorDescriber("x", tol=tol, seed=seed)
    describer.calibrate(calibration_set, seed=seed)
    return dict(describer._gen.risk_receipt)


class CalibrationTruthLookupTest(unittest.TestCase):
    """Regression test for a bug in ``PosteriorDescriber.calibrate``: it built
    ``truth = {id(p): v for p, v in calibration_set}``, an id()-keyed dict. If the SAME posterior
    object appeared more than once in ``calibration_set`` paired with DIFFERENT true values, every
    row referencing that object was silently graded against only the LAST recorded true value --
    ``is_correct`` receives only ``(posterior, claim)`` from ``CalibratedGenerator.calibrate`` (no
    row index), so the fix threads true values through positionally. The
    selective-risk oracle is called once for the selected candidate in each row,
    in order, so a monotone row counter preserves repeated-object identity."""

    def test_reusing_one_posterior_object_calibrates_identically_to_using_distinct_ones(self):
        tol = 1.0
        # a mix of distances from the (deterministic) center 5.0, crossing every width_multiples
        # band boundary (default (1, 3, 10) * tol) and one point outside all of them
        true_values = [5.0 + d for d in (0.2, 1.5, 5.0, 50.0, 0.5, 2.0, 8.0)] * 20

        reused = _ConstantPosterior(5.0)
        calibration_set_shared_object = [(reused, v) for v in true_values]
        calibration_set_distinct_objects = [(_ConstantPosterior(5.0), v) for v in true_values]

        receipt_shared = _risk_receipt_for(calibration_set_shared_object, tol=tol)
        receipt_distinct = _risk_receipt_for(calibration_set_distinct_objects, tol=tol)
        # the receipts differ ONLY in the served-policy uniqueness disclosure: one shared object
        # derives one prompt seed, 100 distinct objects derive their own (STAT-RR17-07 makes the
        # certificate report the schedule it actually covers); the truth-lookup behavior under
        # test must be identical either way
        self.assertEqual(receipt_shared.pop("unique_prompt_count"), 1)
        self.assertGreater(receipt_distinct.pop("unique_prompt_count"), 1)

        # every row's per-candidate draws/scores are identical either way (the mock posterior's
        # sampling ignores its seed and object identity) -- so a correct truth lookup must produce
        # the SAME per-row correctness signal whether or not the posterior object happens to repeat
        self.assertEqual(receipt_shared, receipt_distinct)
        self.assertEqual(receipt_shared["proposal_count"] + receipt_shared["certification_count"], len(true_values))

    def test_adjacent_repeats_of_the_same_posterior_object_are_each_graded_on_their_own_row(self):
        """The harder case: the SAME object at back-to-back rows (not just somewhere in the set)."""
        tol = 1.0
        reused = _ConstantPosterior(5.0)
        true_values = [5.0, 5.0 + 1.5 * tol, 5.0 + 1.5 * tol, 5.0 + 50.0 * tol] * 25
        calibration_set_shared_object = [(reused, v) for v in true_values]
        calibration_set_distinct_objects = [(_ConstantPosterior(5.0), v) for v in true_values]

        receipt_shared = _risk_receipt_for(calibration_set_shared_object, tol=tol)
        receipt_distinct = _risk_receipt_for(calibration_set_distinct_objects, tol=tol)
        # equality modulo the served-policy uniqueness disclosure, as above
        receipt_shared.pop("unique_prompt_count")
        receipt_distinct.pop("unique_prompt_count")
        self.assertEqual(receipt_shared, receipt_distinct)


class CalibrationSelectiveOutcomeTest(unittest.TestCase):
    """The calibrated event is correctness of the one candidate that serving selects."""

    def setUp(self):
        self.tol = 1.0
        self.posterior = _ConstantPosterior(5.0)  # deterministic mean=5.0 regardless of seed/rng
        self.describer = PosteriorDescriber("x", tol=self.tol, k=3, alpha=0.3, n_probe=50, seed=0)

    def _candidates(self):
        # _ConstantPosterior.sample(n, seed=...) ignores seed entirely, so the 3 generated candidates
        # are 100% deterministic regardless of what rng/seed calibrate() derives internally -- calling
        # _generate() directly here reproduces EXACTLY what calibrate() sees.
        return self.describer._generate(self.posterior, self.describer._gen.k, rng=None)

    def _receipt(self, true_value: float) -> dict:
        self.describer.calibrate([(self.posterior, true_value)] * 100, seed=0)
        return dict(self.describer._gen.risk_receipt)

    def test_candidates_are_nested_around_a_shared_center(self):
        # sanity check on the fixture itself before trusting the rest of this test class
        candidates = self._candidates()
        widths = [c.width for c in candidates]
        self.assertEqual(widths, sorted(widths))  # narrowest to widest
        centers = {0.5 * (c.lo + c.hi) for c in candidates}
        self.assertEqual(len(centers), 1)  # all 3 candidates share exactly one center

    def test_truth_at_exact_shared_center_is_covered_by_every_candidate(self):
        true_value = 5.0  # exactly the shared center -> distance 0 from every candidate
        for c in self._candidates():
            self.assertTrue(c.contains(true_value))  # ground truth: every candidate covers it

        receipt = self._receipt(true_value)
        self.assertEqual(receipt["errors"], 0)
        self.assertIsNotNone(receipt["error_upper"])

    def test_wider_covering_candidate_is_not_penalized_for_a_narrower_one_also_covering(self):
        # 2*tol from center: outside the narrowest candidate's half-width (1*tol) but inside BOTH the
        # middle (3*tol) and widest (10*tol) candidates' half-widths -- both genuinely cover it.
        true_value = 5.0 + 2.0 * self.tol
        candidates = self._candidates()
        self.assertFalse(candidates[0].contains(true_value))  # narrowest: does not cover
        self.assertTrue(candidates[1].contains(true_value))  # middle: covers
        self.assertTrue(candidates[2].contains(true_value))  # widest: ALSO covers

        # The selectable outcome is the top-scored candidate, not probability
        # mass pooled over changing candidate identities.
        selected, _statistic = self.describer._gen._selection(self.posterior, seed=0)
        receipt = self._receipt(true_value)
        if selected.contains(true_value):
            self.assertEqual(receipt["errors"], 0)
        else:
            self.assertIsNone(receipt["error_upper"])

    def test_truth_outside_every_candidate_gets_zero_mass(self):
        true_value = 5.0 + 1000.0 * self.tol  # far outside even the widest candidate
        for c in self._candidates():
            self.assertFalse(c.contains(true_value))
        receipt = self._receipt(true_value)
        self.assertIsNone(receipt["error_upper"])
        self.assertEqual(receipt["threshold"], "inf")


if __name__ == "__main__":
    unittest.main()
