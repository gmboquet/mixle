"""Cross-process regression checks for reproducibility-sensitive hashes."""

from __future__ import annotations

import os
import subprocess
import sys

# Each probe pays a COLD mixle import in a fresh interpreter, measured at ~6s unloaded on an Apple
# M4. The previous 10s budget left almost no headroom, and these tests run under `-n 4`, so ordinary
# contention pushed them past it: both failed with TimeoutExpired while the thing they exist to check
# -- that a hash is identical under two PYTHONHASHSEED values -- was never in question. The budget is
# not the measurement here; it is only a guard against a genuine hang, so it is set well clear of the
# import cost while still bounded.
_COLD_IMPORT_TIMEOUT_SECONDS = 120


def _run_with_hash_seed(source: str, seed: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    return subprocess.check_output(
        [sys.executable, "-c", source],
        env=environment,
        text=True,
        timeout=_COLD_IMPORT_TIMEOUT_SECONDS,
    ).strip()


def test_scalar_reduction_is_stable_across_hash_seeds():
    source = (
        "from mixle.reason.zero_shot_bootstrap import _generic_scalar_reduction;"
        "print(_generic_scalar_reduction('new modality sample'))"
    )
    assert _run_with_hash_seed(source, "1") == _run_with_hash_seed(source, "2")


def test_shingles_are_stable_across_hash_seeds():
    source = "from mixle.task.data_mixture import _shingles;print(sorted(_shingles('the same document text', 2)))"
    assert _run_with_hash_seed(source, "1") == _run_with_hash_seed(source, "2")
