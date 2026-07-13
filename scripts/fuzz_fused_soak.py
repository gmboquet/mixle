"""Unbounded local soak for the fused-compiler fuzzer (mixle/tests/fused_fuzz_test.py's generator).

Usage: python scripts/fuzz_fused_soak.py [--samples 300] [--seed 0]

Every sample checks the five properties (fusibility, score parity, M-step parity, parallel
bit-stability, EM monotonicity) against the host oracle. Failures print the seed + signature so a
one-line reproducer can be pinned as a regression test. Compile cost is real for novel signatures;
the disk cache amortizes repeats across runs.
"""

import argparse
import sys
import traceback
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mixle" / "tests"))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from fused_fuzz_test import SIGNATURE_POOL, check_sample

    tc = unittest.TestCase()
    tc.maxDiff = None
    failures = 0
    rng = np.random.RandomState(args.seed)
    for i in range(args.samples):
        sig = SIGNATURE_POOL[i % len(SIGNATURE_POOL)] if i < len(SIGNATURE_POOL) else None
        try:
            check_sample(tc, rng, sig=sig)
        except Exception:  # noqa: BLE001 - a soak records every failure kind; crashing would hide the rest
            failures += 1
            print(f"FAIL sample={i} seed={args.seed} sig={sig}")
            traceback.print_exc()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.samples} samples, {failures} failures", flush=True)
    print(f"soak complete: {args.samples} samples, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
