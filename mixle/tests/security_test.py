"""Secret detection + redaction (N3): keep credentials out of the indexed/served substrate."""

import unittest

from mixle.substrate import (
    Substrate,
    detect_secrets,
    redact_secrets,
    safe_text,
    scan_substrate,
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
    def test_scan_substrate_flags_dirty_items(self):
        s = Substrate()
        s.add(kind="text", text="clean doc about refunds")
        dirty_id = s.add(kind="trace", text="log: sk-abcdefghij1234567890XYZ leaked")
        report = scan_substrate(s)
        self.assertEqual(report["n_items"], 2)
        self.assertEqual(report["n_dirty"], 1)
        self.assertEqual(report["dirty"][0]["item_id"], dirty_id)
        self.assertIn("openai_key", report["dirty"][0]["rules"])

    def test_safe_text_keeps_the_sweep_clean(self):
        s = Substrate()
        s.add(kind="text", text=safe_text("pasted sk-abcdefghij1234567890XYZ here"))
        self.assertEqual(scan_substrate(s)["n_dirty"], 0)  # redacted before store -> nothing to leak


if __name__ == "__main__":
    unittest.main()
