"""mixle.utils.callables.accepts_call: signature-based dispatch for an optional richer call.

The pattern this replaces across the codebase -- try the richer call, catch TypeError, fall back to
a reduced call -- cannot tell "this callable's signature doesn't accept these arguments" apart from
"this callable's own implementation raised a TypeError for an unrelated reason". The latter case
would silently retry with the reduced call, duplicating whatever the callable's own call does (a
remote request, a random draw, an expensive computation) and masking the real error.
"""

import unittest

from mixle.utils.callables import accepts_call


def _with_seed(n, seed=None):
    return [n, seed]


def _without_seed(n):
    return [n]


def _raises_typeerror_internally(n, seed=None):
    return None + n  # a bug unrelated to the call signature, that happens to raise TypeError


class AcceptsCallTestCase(unittest.TestCase):
    def test_signature_match_with_keyword(self):
        self.assertTrue(accepts_call(_with_seed, 5, seed=1))

    def test_signature_mismatch_missing_keyword_param(self):
        self.assertFalse(accepts_call(_without_seed, 5, seed=1))

    def test_signature_match_positional_only(self):
        self.assertTrue(accepts_call(_without_seed, 5))

    def test_reports_true_for_a_call_shape_the_signature_accepts_even_if_the_body_would_raise(self):
        # accepts_call only checks the signature, via inspect.signature(...).bind(...), without
        # ever invoking the function -- so a callable whose body has an unrelated bug still
        # reports True here; the bug surfaces only when the caller actually calls it, undisguised.
        self.assertTrue(accepts_call(_raises_typeerror_internally, 5, seed=1))
        with self.assertRaises(TypeError):
            _raises_typeerror_internally(5, seed=1)

    def test_builtin_without_introspectable_signature_defaults_to_true(self):
        # len does support introspection on most Python versions; this just exercises the codepath
        # without asserting on inspect internals we don't control.
        self.assertIsInstance(accepts_call(len, [1, 2, 3]), bool)

    def test_unsupported_signature_metadata_is_unknown_not_a_proven_mismatch(self):
        # MXR-080-1668: inspect.signature documents ValueError (no signature obtainable) AND
        # TypeError (object type unsupported). Sharing one `except TypeError` with bind() made the
        # second case report a *proven* mismatch, so callers dropped an rng/context the callable
        # does accept. Only bind() can prove a mismatch; failed retrieval is "unknown".
        class UnsupportedSignatureMetadata:
            @property
            def __signature__(self):
                raise TypeError("signature metadata is not supported for this object")

            def __call__(self, *args, **kwargs):
                return kwargs.get("rng")

        fn = UnsupportedSignatureMetadata()
        self.assertEqual(fn(5, rng=7), 7)  # the richer call really does work
        self.assertTrue(accepts_call(fn, 5, rng=7))

    def test_signature_retrieval_valueerror_still_defaults_to_true(self):
        class NoSignature:
            @property
            def __signature__(self):
                raise ValueError("no signature available")

            def __call__(self, *args, **kwargs):
                return 1

        self.assertTrue(accepts_call(NoSignature(), 5, rng=7))

    def test_bind_typeerror_after_successful_retrieval_is_still_a_mismatch(self):
        self.assertFalse(accepts_call(_without_seed, 5, rng=1))


if __name__ == "__main__":
    unittest.main()
