"""Reproduce mixle's headline claims from a clean install and emit a receipt (worklist E14).

External reproduction needs two things a reviewer can run without insider knowledge: a capture of the exact
environment, and a set of *deterministic* checks whose numeric outputs an independent run must match. This
script is both. It records the environment (Python, platform, mixle + core dependency versions, git commit
if available) and runs seeded claim checks -- a Gaussian fit recovering known parameters, scalar/vectorized
score agreement, a serialization round-trip, automatic family recovery, deterministic sampling -- printing a
JSON receipt.

Run ``python scripts/reproduce.py`` (add ``--out receipt.json`` to save it). Because every check is seeded,
an independent reproduction on the same mixle version must produce the same ``checks`` values; a difference
is a real environment-dependent discrepancy, not noise. See ``docs/reproduction.rst`` for the protocol.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mixle": _pkg_version("mixle"),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "git_commit": _git_commit(),
    }


def claim_checks() -> dict:
    """Deterministic, seeded checks backing the headline claims. Values must reproduce across runs."""
    import numpy as np

    import mixle.stats as st
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic import get_estimator
    from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable, to_serializable

    ensure_pysp_serialization_registry()
    checks: dict = {}

    # 1. a Gaussian fit recovers the generating parameters (seeded data -> fixed rounded estimate).
    data = np.random.RandomState(0).normal(3.0, 2.0, 2000).tolist()
    fitted = optimize(data, st.GaussianEstimator(), max_its=20, out=None)
    checks["gaussian_fit_mu"] = round(float(fitted.mu), 4)
    checks["gaussian_fit_sigma"] = round(float(np.sqrt(fitted.sigma2)), 4)

    # 2. scalar and vectorized scoring agree.
    g = st.GaussianDistribution(1.0, 2.0)
    xs = list(g.sampler(seed=0).sample(32))
    seq = np.asarray(g.seq_log_density(g.dist_to_encoder().seq_encode(xs)), dtype=float)
    scalar = np.array([float(g.log_density(x)) for x in xs])
    checks["scalar_vectorized_agree"] = bool(np.allclose(seq, scalar, atol=1e-9))

    # 3. serialization round-trips a model's score.
    back = from_serializable(to_serializable(g))
    checks["serialization_score_equal"] = round(float(back.log_density(0.5)), 6) == round(float(g.log_density(0.5)), 6)

    # 4. automatic family recovery on seeded Gaussian data.
    checks["auto_selects"] = type(get_estimator(np.random.RandomState(1).normal(0, 1, 1500).tolist())).__name__

    # 5. deterministic seeded sampling.
    checks["deterministic_sample"] = round(float(st.GammaDistribution(2.0, 1.5).sampler(seed=7).sample(1)[0]), 6)

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce mixle's headline claims and emit a receipt.")
    parser.add_argument("--out", default=None, help="write the JSON receipt here (default: stdout)")
    args = parser.parse_args(argv)

    receipt = {"artifact": "mixle.reproduction_receipt/v1", "environment": environment(), "checks": claim_checks()}
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
