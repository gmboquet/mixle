"""Secret detection + redaction (N3): keep credentials out of the indexed/served substrate."""

import json
import unittest

from mixle.substrate import (
    Substrate,
    SubstrateItem,
    detect_secrets,
    redact_secrets,
    safe_text,
    scan_item,
    scan_substrate,
)
from mixle.substrate.security import (
    SecretPolicyError,
    enforce_secret_policy,
    item_surface,
    redact_value,
)


class DetectTest(unittest.TestCase):
    def test_detects_common_secret_shapes(self):
        cases = {
            "openai_key": "call with sk-abcdefghij1234567890XYZ please",
            "aws_access_key": "creds AKIA1234567890ABCDEF here",
            "url_credentials": "postgres://admin:hunter2@db.example.com/prod",
            "sensitive_assignment": "password = s3cr3tP@ssw0rd123",
            "bearer_token": "header Bearer abcdef1234567890ABCDEF",
        }
        for expected_rule, text in cases.items():
            scan = detect_secrets(text)
            self.assertFalse(scan.clean, text)
            self.assertIn(expected_rule, scan.rules(), text)

    def test_clean_prose_has_no_findings(self):
        for text in ["refunds are processed within 30 days", "the password is kept safe", "token: short"]:
            self.assertTrue(detect_secrets(text).clean, text)

    def test_empty_text_is_clean(self):
        self.assertTrue(detect_secrets("").clean)

    def test_multiple_secrets_all_found(self):
        text = "key sk-abcdefghij1234567890XYZ and AKIA1234567890ABCDEF"
        scan = detect_secrets(text)
        self.assertEqual(len(scan.findings), 2)
        self.assertEqual(scan.rules(), ["aws_access_key", "openai_key"])


class RedactTest(unittest.TestCase):
    def test_redaction_removes_the_secret(self):
        text = "use sk-abcdefghij1234567890XYZ now"
        red = redact_secrets(text)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", red)
        self.assertIn("[REDACTED:openai_key]", red)

    def test_keep_prefix_leaves_a_recognizable_hint(self):
        red = redact_secrets("creds AKIA1234567890ABCDEF end", keep_prefix=4)
        self.assertIn("AKIA[REDACTED:aws_access_key]", red)
        self.assertNotIn("1234567890ABCDEF", red)

    def test_clean_text_is_unchanged(self):
        text = "nothing secret here"
        self.assertEqual(redact_secrets(text), text)

    def test_safe_text_is_redact_before_store(self):
        stored = safe_text("note with token Bearer abcdef1234567890ABCDEF inside")
        self.assertNotIn("abcdef1234567890ABCDEF", stored)


class SweepTest(unittest.TestCase):
    def test_scan_substrate_flags_dirty_items_that_bypassed_the_store_guard(self):
        """MXR-080-0262: Substrate.put()/add() now redact by default (see
        mixle.tests.substrate_test.SecretHandlingBoundaryTest), so a secret written through the normal
        API never reaches self._items in raw form -- scan_substrate()'s ongoing job is auditing items
        that got into the store some OTHER way (a shard written before this fix and loaded back in, or
        anything else that bypasses put()). Insert directly into _items to simulate exactly that, and
        confirm the sweep's own detection logic still correctly flags it."""
        s = Substrate()
        clean_id = s.add(kind="text", text="clean doc about refunds")
        dirty = SubstrateItem(kind="trace", text="log: sk-abcdefghij1234567890XYZ leaked")
        s._items[dirty.id] = dirty  # bypass put()'s guard -- simulates pre-fix/legacy stored data
        report = scan_substrate(s)
        self.assertEqual(report["n_items"], 2)
        self.assertEqual(report["n_dirty"], 1)
        self.assertEqual(report["dirty"][0]["item_id"], dirty.id)
        self.assertIn("openai_key", report["dirty"][0]["rules"])
        self.assertNotEqual(clean_id, dirty.id)

    def test_scan_substrate_flags_a_payload_only_secret_that_bypassed_the_guard(self):
        """The audit's concrete adversarial case: a secret in `payload`, not `text`. scan_item's surface
        must cover it (item_surface: text + payload + tags), not just `.text`."""
        s = Substrate()
        dirty = SubstrateItem(kind="trace", text="a clean summary", payload={"api_key": "sk-abcdefghij1234567890XYZ"})
        s._items[dirty.id] = dirty  # bypass put(); scan_substrate must still catch it via item_surface
        report = scan_substrate(s)
        self.assertEqual(report["n_dirty"], 1)
        self.assertEqual(report["dirty"][0]["item_id"], dirty.id)
        self.assertIn("openai_key", report["dirty"][0]["rules"])

    def test_safe_text_keeps_the_sweep_clean(self):
        s = Substrate()
        s.add(kind="text", text=safe_text("pasted sk-abcdefghij1234567890XYZ here"))
        self.assertEqual(scan_substrate(s)["n_dirty"], 0)  # redacted before store -> nothing to leak

    def test_put_now_keeps_the_sweep_clean_even_without_a_manual_safe_text_call(self):
        """MXR-080-0262: the redact-before-store guard is no longer opt-in -- put()/add() apply it
        unconditionally, so a caller who never calls safe_text() themselves still ends up with a clean
        store (contrast with test_safe_text_keeps_the_sweep_clean, which redacts manually first)."""
        s = Substrate()
        s.add(kind="text", text="pasted sk-abcdefghij1234567890XYZ here, no manual redaction")
        self.assertEqual(scan_substrate(s)["n_dirty"], 0)


class ItemSurfaceTest(unittest.TestCase):
    """MXR-080-0262: the surface scanning covers must match what lexical retrieval actually serializes
    and indexes (mixle.substrate.core._lexical_score's `text + json.dumps(payload) + tags`), not just
    `.text` -- a payload containing a secret was indexed while the old scan reported clean."""

    def test_surface_joins_text_payload_and_tags(self):
        item = SubstrateItem(kind="record", text="a summary", payload={"k": "v"}, tags=["t1", "t2"])
        surface = item_surface(item)
        self.assertIn("a summary", surface)
        self.assertIn('"k": "v"', surface)
        self.assertIn("t1", surface)
        self.assertIn("t2", surface)

    def test_surface_matches_lexical_scores_own_construction(self):
        item = SubstrateItem(
            kind="record", text="alpha beta", payload={"k": "sk-abcdefghij1234567890XYZ"}, tags=["gamma"]
        )
        # _lexical_score builds " ".join([item.text, json.dumps(item.payload), " ".join(item.tags)]);
        # item_surface must build the identical string so scanning sees exactly what search indexes.
        expected = " ".join([item.text, json.dumps(item.payload), " ".join(item.tags)])
        self.assertEqual(item_surface(item), expected)

    def test_non_json_serializable_payload_does_not_raise(self):
        item = SubstrateItem(kind="record", text="", payload={"odd": {1, 2, 3}})  # a set: not JSON-able
        surface = item_surface(item)  # must not raise
        self.assertIn("odd", surface)


class ScanItemSurfaceCoverageTest(unittest.TestCase):
    """The audit's concrete case, at the scan_item() unit level: a secret in payload or tags must be
    caught, not just one in text."""

    def test_catches_a_secret_in_payload(self):
        item = SubstrateItem(
            kind="trace",
            text="a clean summary of a tool call",
            payload={"api_key": "sk-abcdefghij1234567890XYZ"},
        )
        scan = scan_item(item)
        self.assertFalse(scan.clean)
        self.assertIn("openai_key", scan.rules())

    def test_catches_a_secret_in_tags(self):
        item = SubstrateItem(kind="text", text="clean", tags=["AKIA1234567890ABCDEF"])
        scan = scan_item(item)
        self.assertFalse(scan.clean)
        self.assertIn("aws_access_key", scan.rules())

    def test_still_catches_a_secret_in_text(self):
        item = SubstrateItem(kind="text", text="sk-abcdefghij1234567890XYZ")
        self.assertFalse(scan_item(item).clean)

    def test_clean_when_nothing_leaks_anywhere(self):
        item = SubstrateItem(kind="record", text="refunds within 30 days", payload={"amount": 900}, tags=["finance"])
        self.assertTrue(scan_item(item).clean)


class RedactValueTest(unittest.TestCase):
    """redact_value is redact_secrets recursed over a JSON-like structure -- how `payload` gets the same
    masked-before-store treatment safe_text gives free text."""

    def test_redacts_a_nested_dict_leaf(self):
        payload = {"outer": {"inner": "use sk-abcdefghij1234567890XYZ now"}}
        red = redact_value(payload)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", json.dumps(red))
        self.assertIn("[REDACTED:openai_key]", red["outer"]["inner"])

    def test_redacts_inside_a_list(self):
        payload = {"items": ["clean", "creds AKIA1234567890ABCDEF here"]}
        red = redact_value(payload)
        self.assertNotIn("AKIA1234567890ABCDEF", json.dumps(red))

    def test_non_string_leaves_pass_through_unchanged(self):
        payload = {"amount": 900, "active": True, "ratio": 0.5, "missing": None}
        self.assertEqual(redact_value(payload), payload)

    def test_clean_structure_is_value_equal(self):
        payload = {"a": [1, "clean text", {"b": "also clean"}]}
        self.assertEqual(redact_value(payload), payload)


class EnforceSecretPolicyTest(unittest.TestCase):
    """Unit coverage for the store-boundary choke point Substrate.put()/.update() route every write
    through (mixle.substrate.core.Substrate._store)."""

    def test_clean_item_is_returned_unchanged_same_object(self):
        item = SubstrateItem(kind="text", text="nothing secret here")
        out, scan = enforce_secret_policy(item, policy="redact")
        self.assertIs(out, item)  # no copy made when there's nothing to do
        self.assertTrue(scan.clean)

    def test_redact_policy_masks_text_payload_and_tags(self):
        item = SubstrateItem(
            kind="trace",
            text="use sk-abcdefghij1234567890XYZ now",
            payload={"key": "AKIA1234567890ABCDEF"},
            tags=["Bearer abcdef1234567890ABCDEF"],
        )
        out, scan = enforce_secret_policy(item, policy="redact")
        self.assertFalse(scan.clean)  # reports what WAS found (pre-redaction)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", out.text)
        self.assertNotIn("AKIA1234567890ABCDEF", json.dumps(out.payload))
        self.assertNotIn("abcdef1234567890ABCDEF", " ".join(out.tags))
        self.assertIsNot(out, item)  # a sanitized copy, not the caller's object

    def test_reject_policy_raises_and_returns_nothing(self):
        item = SubstrateItem(kind="text", text="sk-abcdefghij1234567890XYZ", id="fixed-id")
        with self.assertRaises(SecretPolicyError) as ctx:
            enforce_secret_policy(item, policy="reject")
        self.assertEqual(ctx.exception.item_id, "fixed-id")
        self.assertIn("openai_key", ctx.exception.scan.rules())

    def test_unknown_policy_raises_value_error(self):
        item = SubstrateItem(kind="text", text="clean")
        with self.assertRaises(ValueError):
            enforce_secret_policy(item, policy="ignore")  # not a real policy


if __name__ == "__main__":
    unittest.main()
