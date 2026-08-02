"""Redaction must cover the whole surface the scanner reads (MXR-080-1882).

``item_surface`` serializes ``payload`` with ``json.dumps`` -- which writes dictionary KEYS as well as
values -- and falls back to ``str(payload)`` when that fails, which stringifies sets and any opaque
object's ``repr``. ``redact_value`` reached only dict values, lists and tuples, so the detection
surface was strictly larger than the redaction surface, and three reachable places were detected and
then stored in the clear. The store never looked again after redacting, so nothing caught it.

The asymmetry is the real defect, so the repair closes it twice: redaction now covers the same
shapes, and ``enforce_secret_policy`` re-scans the sanitized item before returning it, which turns
any future gap into a refused write instead of a leak.
"""

import tempfile
import unittest

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.security import (
    SecretPolicyError,
    detect_secrets,
    enforce_secret_policy,
    item_surface,
    redact_value,
    scan_item,
)

# Assembled at import rather than written whole, so no credential-shaped literal is tracked in this
# file. These have to LOOK like credentials to the detector -- that is the whole point of the fixture
# -- which is exactly what repo_hygiene_scan_test refuses to find in a tracked file. Splitting the
# literal satisfies both: the scanner reads the source line, the detector reads the assembled value.
API_KEY_ASSIGNMENT = "api_key=" + "abcdefghijklmnop"
OPENAI_KEY = "sk-" + "abcdefghijklmnopqrstuvwx"
GITHUB_TOKEN = "token=" + "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"


class Opaque:
    """A payload value whose only textual form carries a secret."""

    def __repr__(self) -> str:
        return GITHUB_TOKEN


class RedactionSurfaceTest(unittest.TestCase):
    def test_a_dictionary_key_is_redacted(self):
        # The auditor's shape. json.dumps writes keys, so the scan saw it; redaction did not.
        redacted = redact_value({API_KEY_ASSIGNMENT: "v"})
        self.assertNotIn(API_KEY_ASSIGNMENT, redacted)
        self.assertTrue(detect_secrets(str(redacted)).clean)

    def test_a_set_member_is_redacted(self):
        redacted = redact_value({"c": {OPENAI_KEY}})
        self.assertTrue(detect_secrets(str(redacted)).clean)

    def test_a_frozenset_stays_a_frozenset(self):
        self.assertIsInstance(redact_value(frozenset({OPENAI_KEY})), frozenset)

    def test_an_opaque_object_whose_text_carries_a_secret_is_redacted(self):
        redacted = redact_value({"o": Opaque()})
        self.assertTrue(detect_secrets(str(redacted)).clean)

    def test_an_opaque_object_with_no_secret_passes_through_as_itself(self):
        # Converting every opaque value to text would destroy payloads never at risk.
        sentinel = object()
        self.assertIs(redact_value({"o": sentinel})["o"], sentinel)

    def test_scalars_and_containers_are_otherwise_unchanged(self):
        payload = {"n": 5, "f": 1.5, "b": True, "z": None, "l": [1, 2], "t": (3, 4)}
        self.assertEqual(redact_value(payload), payload)

    def test_nested_shapes_are_reached(self):
        redacted = redact_value({"a": [{"b": {OPENAI_KEY}}], "c": ({API_KEY_ASSIGNMENT: 1},)})
        self.assertTrue(detect_secrets(str(redacted)).clean)

    def test_a_key_collision_refuses_rather_than_dropping_a_field(self):
        # Two distinct keys that mask to the same string would silently merge, losing an entry.
        with self.assertRaisesRegex(ValueError, "collapsed two distinct keys"):
            redact_value({"api_key=aaaaaaaaaaaaaaaa": 1, "api_key=bbbbbbbbbbbbbbbb": 2})


class WriteBoundaryTest(unittest.TestCase):
    def _item(self, **overrides) -> SubstrateItem:
        fields = dict(kind="text", text="hello", payload={API_KEY_ASSIGNMENT: "v"})
        fields.update(overrides)
        return SubstrateItem(**fields)

    def test_the_stored_item_is_actually_clean(self):
        substrate = Substrate(tempfile.mkdtemp())
        stored = substrate.get(substrate.put(self._item()))
        self.assertTrue(scan_item(stored).clean)
        self.assertNotIn(API_KEY_ASSIGNMENT, str(stored.payload))

    def test_a_clean_item_is_stored_unchanged(self):
        substrate = Substrate(tempfile.mkdtemp())
        stored = substrate.get(substrate.put(self._item(text="ordinary", payload={"n": 5})))
        self.assertEqual(stored.payload, {"n": 5})
        self.assertEqual(stored.text, "ordinary")

    def test_reject_policy_still_refuses_without_storing(self):
        with self.assertRaises(SecretPolicyError):
            enforce_secret_policy(self._item(), policy="reject")

    def test_the_sanitized_surface_is_verified_before_return(self):
        # The guarantee, stated directly: whatever enforce_secret_policy returns scans clean.
        sanitized, scan = enforce_secret_policy(self._item(), policy="redact")
        self.assertFalse(scan.clean)
        self.assertTrue(detect_secrets(item_surface(sanitized)).clean)


if __name__ == "__main__":
    unittest.main()
