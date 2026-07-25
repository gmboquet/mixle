"""The language<->belief bridge (roadmap M5, part (c))."""

import unittest
from unittest import mock

import numpy as np

import mixle.task.calibrated_generator as calibrated_generator_module
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


def _cal_prob_true_for(calibration_set, *, tol: float = 1.0, seed: int = 0) -> list[float]:
    """Calibrate and capture the exact per-row "probability mass on the correct candidate" array
    CalibratedGenerator.calibrate() builds, by spying on conformal_label_threshold -- the first (and
    only) thing that array is passed to, and not otherwise exposed by the public API."""
    real = calibrated_generator_module.conformal_label_threshold
    with mock.patch.object(calibrated_generator_module, "conformal_label_threshold", side_effect=real) as spy:
        PosteriorDescriber("x", tol=tol, seed=seed).calibrate(calibration_set, seed=seed)
    return list(spy.call_args[0][0])


class CalibrationTruthLookupTest(unittest.TestCase):
    """Regression test for a bug in ``PosteriorDescriber.calibrate``: it built
    ``truth = {id(p): v for p, v in calibration_set}``, an id()-keyed dict. If the SAME posterior
    object appeared more than once in ``calibration_set`` paired with DIFFERENT true values, every
    row referencing that object was silently graded against only the LAST recorded true value --
    ``is_correct`` receives only ``(posterior, claim)`` from ``CalibratedGenerator.calibrate`` (no
    row index), so the fix threads true values through positionally instead (counting calls: exactly
    ``k`` candidates are generated and scored per row, in order, so ``calls // k`` is that row's
    index -- correct regardless of how many times, or how adjacently, a posterior object repeats)."""

    def test_reusing_one_posterior_object_calibrates_identically_to_using_distinct_ones(self):
        tol = 1.0
        # a mix of distances from the (deterministic) center 5.0, crossing every width_multiples
        # band boundary (default (1, 3, 10) * tol) and one point outside all of them
        true_values = [5.0 + d for d in (0.2, 1.5, 5.0, 50.0, 0.5, 2.0, 8.0)]

        reused = _ConstantPosterior(5.0)
        calibration_set_shared_object = [(reused, v) for v in true_values]
        calibration_set_distinct_objects = [(_ConstantPosterior(5.0), v) for v in true_values]

        cal_prob_true_shared = _cal_prob_true_for(calibration_set_shared_object, tol=tol)
        cal_prob_true_distinct = _cal_prob_true_for(calibration_set_distinct_objects, tol=tol)

        # every row's per-candidate draws/scores are identical either way (the mock posterior's
        # sampling ignores its seed and object identity) -- so a correct truth lookup must produce
        # the SAME per-row correctness signal whether or not the posterior object happens to repeat
        self.assertEqual(cal_prob_true_shared, cal_prob_true_distinct)
        # and it must not be trivially uninformative (all rows collapsed to one value) -- confirms
        # this calibration set actually exercises different bands across rows
        self.assertGreater(len(set(cal_prob_true_shared)), 1)

    def test_adjacent_repeats_of_the_same_posterior_object_are_each_graded_on_their_own_row(self):
        """The harder case: the SAME object at back-to-back rows (not just somewhere in the set)."""
        tol = 1.0
        reused = _ConstantPosterior(5.0)
        true_values = [5.0, 5.0 + 1.5 * tol, 5.0 + 1.5 * tol, 5.0 + 50.0 * tol]  # rows 1 and 2 adjacent
        calibration_set_shared_object = [(reused, v) for v in true_values]
        calibration_set_distinct_objects = [(_ConstantPosterior(5.0), v) for v in true_values]

        self.assertEqual(
            _cal_prob_true_for(calibration_set_shared_object, tol=tol),
            _cal_prob_true_for(calibration_set_distinct_objects, tol=tol),
        )


if __name__ == "__main__":
    unittest.main()
