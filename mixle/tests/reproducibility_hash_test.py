"""Cross-process regression checks for reproducibility-sensitive hashes."""

from __future__ import annotations

import os
import subprocess
import sys


def _run_with_hash_seed(source: str, seed: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    return subprocess.check_output(
        [sys.executable, "-c", source],
        env=environment,
        text=True,
        timeout=10,
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
