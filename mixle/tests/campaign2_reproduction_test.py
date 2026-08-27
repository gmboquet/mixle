"""Campaign 2, reproduction area: T3-06 and T3-05.

T3-06 (major): ``wheel_provenance``'s ``verified`` key was a hardcoded literal ``True`` -- the
function hashed the wheel only to REPORT a digest and could not return False for altered member
bytes, a deleted RECORD row, or a smuggled member, while the module docstring said "fail-closed".
It now runs the wheel's RECORD self-consistency check (``_wheel_record_hashes``, the same check
``subject_binding`` builds on), collects ``problems``, and derives ``verified`` from them; the
optional ``expected_commit=`` / ``expected_content_sha256=`` arguments close the shape-valid-but-
false provenance field cases when the caller has an expectation to compare against.

T3-05 (minor): one key, two populations -- ``source_content_sha256`` digests the full checkout in
an sdist's embedded record but only the packaged subset in a wheel built from that sdist, so
cross-artifact diffing shows a false mismatch. The consumer (``wheel_provenance``) now documents
both meanings and names the sdist's ``sdist_content_sha256`` as the true cross-artifact check;
disambiguating at the writer (setup.py) is outside this wave's file ownership and is handed off.
"""

import base64
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mixle import reproduction

MODULE_SRC = b"def answer():\n    return 42\n"
SMUGGLED_SRC = b"import os\n"


def _enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def _build_wheel(
    directory,
    *,
    tamper: str | None = None,
    commit: str = "a" * 40,
    tree: str = "b" * 40,
    extra_provenance: dict | None = None,
) -> Path:
    """A minimal internally consistent mixle wheel, with one optional named tampering."""
    wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
    provenance = {
        "artifact": "mixle.build_provenance/v1",
        "source_commit": commit,
        "source_tree": tree,
        "source_dirty": False,
        "source_content_sha256": "c" * 64,
        "source_content_file_count": 3,
        "source_content_universe": "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx}",
    }
    provenance.update(extra_provenance or {})
    members = {
        "mixle/mod.py": MODULE_SRC,
        "mixle/_build_provenance.json": json.dumps(provenance).encode(),
        "mixle-0.8.0.dist-info/METADATA": b"Name: mixle\nVersion: 0.8.0\n",
    }
    rows = [(name, _enc(data), str(len(data))) for name, data in members.items()]

    if tamper == "altered_module_bytes":
        # the RECORD row keeps the original hash; only the member bytes change
        members["mixle/mod.py"] = b"def answer():\n    return 41  # tampered\n"
    elif tamper == "smuggled_record_listed":
        members["mixle/evil.py"] = SMUGGLED_SRC
        rows.append(("mixle/evil.py", _enc(SMUGGLED_SRC), str(len(SMUGGLED_SRC))))
    elif tamper == "smuggled_unlisted":
        members["mixle/evil.py"] = SMUGGLED_SRC
    elif tamper == "deleted_record_row":
        rows = [row for row in rows if row[0] != "mixle/mod.py"]

    record = "".join(f"{name},sha256={digest},{size}\n" for name, digest, size in rows)
    record += "mixle-0.8.0.dist-info/RECORD,,\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if tamper != "no_record":
            archive.writestr("mixle-0.8.0.dist-info/RECORD", record)
    return wheel


class WheelProvenanceIsNotHardcodedTest(unittest.TestCase):
    """T3-06: the verdict must be derived from checks, so tampering can flip it."""

    def test_internally_consistent_wheel_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory))
        self.assertTrue(result["verified"])
        self.assertEqual(result["problems"], [])
        # the three hashed rows: module, provenance record, METADATA (RECORD hashes itself never)
        self.assertEqual(result["record_hashed_entry_count"], 3)

    def test_altered_member_bytes_flip_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, tamper="altered_module_bytes"))
        self.assertFalse(result["verified"])
        self.assertTrue(any("does not match its RECORD hash" in problem for problem in result["problems"]))
        self.assertIsNone(result["record_hashed_entry_count"])

    def test_deleted_record_row_flips_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, tamper="deleted_record_row"))
        self.assertFalse(result["verified"])
        self.assertTrue(any("does not claim" in problem for problem in result["problems"]))

    def test_smuggled_unlisted_member_flips_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, tamper="smuggled_unlisted"))
        self.assertFalse(result["verified"])
        self.assertTrue(any("mixle/evil.py" in problem for problem in result["problems"]))

    def test_missing_record_flips_verified_without_raising(self):
        # past identification the receipt reports tampering instead of refusing to exist
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, tamper="no_record"))
        self.assertFalse(result["verified"])
        self.assertTrue(any("exactly one RECORD" in problem for problem in result["problems"]))

    def test_internally_consistent_smuggle_self_verifies_and_the_limit_is_documented(self):
        # RECORD self-consistency proves the archive matches its own manifest, not that the
        # manifest is the released one: a smuggled module LISTED with a correct hash still
        # self-verifies here. That limit must be stated, and the check that closes it named.
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, tamper="smuggled_record_listed"))
        self.assertTrue(result["verified"])
        doc = reproduction.wheel_provenance.__doc__
        self.assertIn("subject_binding", doc)
        self.assertIn("internally consistent RECORD", doc)


class ExpectedProvenanceValuesTest(unittest.TestCase):
    """T3-06: shape-valid-but-false provenance fields are caught only against an expectation."""

    def test_shape_valid_but_false_fields_pass_without_an_expectation(self):
        # 40 zeros / 40 f's are valid shapes; without an expected value there is nothing to
        # compare against, and the docstring says exactly that.
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, commit="f" * 40, tree="0" * 40))
        self.assertTrue(result["verified"])
        self.assertIn("shape-validated only", reproduction.wheel_provenance.__doc__)

    def test_expected_commit_mismatch_flips_verified_and_names_both_values(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, commit="f" * 40), expected_commit="a" * 40)
        self.assertFalse(result["verified"])
        self.assertTrue(any("f" * 40 in problem and "a" * 40 in problem for problem in result["problems"]))

    def test_expected_content_sha256_mismatch_flips_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory), expected_content_sha256="9" * 64)
        self.assertFalse(result["verified"])
        self.assertTrue(any("source_content_sha256" in problem for problem in result["problems"]))

    def test_matching_expectations_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(
                _build_wheel(directory), expected_commit="a" * 40, expected_content_sha256="c" * 64
            )
        self.assertTrue(result["verified"])
        self.assertEqual(result["problems"], [])

    def test_expected_values_normalize_case_and_whitespace(self):
        # an uppercase or whitespace-wrapped digest names the same bytes; refusing it would be
        # the guard-overreach pattern this campaign exists to remove
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(
                _build_wheel(directory), expected_commit=("A" * 40) + "\n", expected_content_sha256=" " + "C" * 64
            )
        self.assertTrue(result["verified"])

    def test_malformed_expected_values_are_caller_errors_not_mismatches(self):
        # a value that can never match is a typo in the CALL, and reporting it as a provenance
        # mismatch would present the caller's error as evidence of tampering
        with tempfile.TemporaryDirectory() as directory:
            wheel = _build_wheel(directory)
            with self.assertRaisesRegex(ValueError, "expected_commit"):
                reproduction.wheel_provenance(wheel, expected_commit="HEAD")
            with self.assertRaisesRegex(ValueError, "expected_content_sha256"):
                reproduction.wheel_provenance(wheel, expected_content_sha256="c" * 63)


class ContentDigestPopulationsTest(unittest.TestCase):
    """T3-05: the two meanings of ``source_content_sha256`` are documented at the consumer.

    Measured on this tree (setup.py, release env vars): the sdist's embedded record digests the
    full checkout (2,033 files) under ``source_content_sha256`` while the wheel built FROM that
    sdist digests the 816-file packaged subset under the SAME key -- and the sdist's
    ``sdist_content_sha256`` equals the wheel's value, making it the true cross-artifact check.
    Renaming the key at the writer is a setup.py change outside this wave's ownership (handed
    off); the consumer must meanwhile say what the key means and where the real cross-check is.
    """

    def test_both_populations_and_the_cross_check_are_documented_where_consumed(self):
        doc = reproduction.wheel_provenance.__doc__
        self.assertIn("sdist_content_sha256", doc)
        self.assertIn("two populations", doc)
        self.assertIn("source_content_file_count", doc)
        # the false-mismatch reading is explicitly disclaimed
        self.assertIn("not tampering", doc)

    def test_carried_sdist_cross_check_stays_discoverable_in_the_receipt(self):
        # A wheel whose record was carried through the sdist ships ``sdist_content_sha256``;
        # the receipt must pass it through verbatim (``build`` is the whole embedded record),
        # because that field is the only digest a holder of both artifacts can cross-check.
        carried = {
            "sdist_content_sha256": "d" * 64,
            "sdist_content_file_count": 2,
            "source_content_carried_through_sdist": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = reproduction.wheel_provenance(_build_wheel(directory, extra_provenance=carried))
        self.assertTrue(result["verified"])
        self.assertEqual(result["build"]["sdist_content_sha256"], "d" * 64)
        self.assertTrue(result["build"]["source_content_carried_through_sdist"])


if __name__ == "__main__":
    unittest.main()
