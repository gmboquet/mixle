"""Installed, fail-closed reproduction receipt for Mixle's release claims."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

_EXPECTATIONS = {
    "gaussian_fit_mu": {"expected": 3.0, "absolute_tolerance": 0.2},
    "gaussian_fit_sigma": {"expected": 2.0, "absolute_tolerance": 0.2},
    "scalar_vectorized_agree": {"expected": True},
    "serialization_score_equal": {"expected": True},
    "auto_selects": {"expected": "GaussianEstimator"},
    "deterministic_sample": {"expected": 1.701912, "absolute_tolerance": 1e-6},
}


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _full_git_commit(repository: Path) -> str:
    """Resolve the source checkout containing this module, never the caller's working directory."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return "unknown"
    return commit


def installed_content_provenance() -> dict[str, Any]:
    """Digest the installed distribution's RECORD identities and verify hashed files."""
    try:
        dist = distribution("mixle")
    except PackageNotFoundError:
        return {
            "artifact": "mixle.installed_content/v1",
            "verified": False,
            "reason": "mixle distribution metadata is not installed",
        }
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in sorted(dist.files or (), key=str):
        if item.hash is None:
            continue
        path = Path(dist.locate_file(item))
        if not path.is_file():
            failures.append(f"{item}: missing")
            continue
        algorithm = item.hash.mode
        if algorithm != "sha256":
            failures.append(f"{item}: unsupported RECORD hash {algorithm}")
            continue
        digest, size = _sha256_file(path)
        expected = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
        if expected != item.hash.value:
            failures.append(f"{item}: RECORD hash mismatch")
        entries.append({"path": str(item), "sha256": digest, "size_bytes": size})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "artifact": "mixle.installed_content/v1",
        "digest": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "verified": bool(entries) and not failures,
        "failures": failures,
    }


def wheel_provenance(path: Path) -> dict[str, Any]:
    """Verify a Mixle wheel, its installed version identity, and embedded source provenance."""
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError("reproduction artifact must be an existing .whl file")
    digest, size = _sha256_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            provenance_names = [name for name in archive.namelist() if name == "mixle/_build_provenance.json"]
            if len(metadata_names) != 1 or len(provenance_names) != 1:
                raise ValueError("wheel must contain exactly one METADATA and one Mixle build-provenance record")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
            provenance = json.loads(archive.read(provenance_names[0]))
    except (OSError, UnicodeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError("reproduction artifact is not a valid Mixle wheel") from exc
    fields = {}
    for line in metadata.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value)
    installed_version = _pkg_version("mixle")
    if fields.get("Name", "").lower() != "mixle" or fields.get("Version") != installed_version:
        raise ValueError("wheel name/version does not match the installed Mixle distribution")
    if not isinstance(provenance, dict) or provenance.get("artifact") != "mixle.build_provenance/v1":
        raise ValueError("wheel build provenance has an unsupported schema")
    for field in ("source_commit", "source_tree"):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"wheel build provenance has invalid {field}")
    content_digest = provenance.get("source_content_sha256")
    if (
        not isinstance(content_digest, str)
        or len(content_digest) != 64
        or any(c not in "0123456789abcdef" for c in content_digest)
    ):
        raise ValueError("wheel build provenance has invalid source_content_sha256")
    if provenance.get("source_dirty") is not False:
        raise ValueError("release reproduction requires a wheel built from a clean source candidate")
    return {
        "artifact": "mixle.wheel_provenance/v1",
        "filename": path.name,
        "sha256": digest,
        "size_bytes": size,
        "version": installed_version,
        "build": provenance,
        "verified": True,
    }


def source_tree_provenance() -> dict[str, Any]:
    """Identify an explicitly requested development checkout by its own repository root."""
    repository = Path(__file__).resolve().parents[1]
    commit = _full_git_commit(repository)
    return {
        "artifact": "mixle.source_tree_provenance/v1",
        "commit": commit,
        "repository": str(repository),
        "verified": commit != "unknown",
    }


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mixle": _pkg_version("mixle"),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "installed_content": installed_content_provenance(),
    }


def claim_checks() -> dict[str, Any]:
    """Run deterministic, seeded measurements backing the release claims."""
    import numpy as np

    import mixle.stats as st
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic import get_estimator
    from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable, to_serializable

    ensure_pysp_serialization_registry()
    data = np.random.RandomState(0).normal(3.0, 2.0, 2000).tolist()
    fitted = optimize(data, st.GaussianEstimator(), max_its=20, out=None)
    g = st.GaussianDistribution(1.0, 2.0)
    xs = list(g.sampler(seed=0).sample(32))
    seq = np.asarray(g.seq_log_density(g.dist_to_encoder().seq_encode(xs)), dtype=float)
    scalar = np.array([float(g.log_density(x)) for x in xs])
    back = from_serializable(to_serializable(g))
    return {
        "gaussian_fit_mu": round(float(fitted.mu), 4),
        "gaussian_fit_sigma": round(float(np.sqrt(fitted.sigma2)), 4),
        "scalar_vectorized_agree": bool(np.allclose(seq, scalar, atol=1e-9)),
        "serialization_score_equal": (round(float(back.log_density(0.5)), 6) == round(float(g.log_density(0.5)), 6)),
        "auto_selects": type(get_estimator(np.random.RandomState(1).normal(0, 1, 1500).tolist())).__name__,
        "deterministic_sample": round(float(st.GammaDistribution(2.0, 1.5).sampler(seed=7).sample(1)[0]), 6),
    }


def evaluate_claims(observed: object) -> dict[str, dict[str, Any]]:
    """Bind every observed measurement to a required value/tolerance and pass state."""
    if not isinstance(observed, dict) or set(observed) != set(_EXPECTATIONS):
        missing = sorted(set(_EXPECTATIONS) - set(observed) if isinstance(observed, dict) else _EXPECTATIONS)
        extra = sorted(set(observed) - set(_EXPECTATIONS)) if isinstance(observed, dict) else []
        raise ValueError(f"claim checks do not match the required protocol; missing={missing}, extra={extra}")
    results = {}
    for name, expectation in _EXPECTATIONS.items():
        actual = observed[name]
        expected = expectation["expected"]
        tolerance = expectation.get("absolute_tolerance")
        if tolerance is None:
            passed = type(actual) is type(expected) and actual == expected
        else:
            passed = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and abs(float(actual) - float(expected)) <= tolerance
            )
        results[name] = {
            "observed": actual,
            "expected": expected,
            "absolute_tolerance": tolerance,
            "passed": passed,
        }
    return results


def build_receipt(*, wheel: Path | None, allow_source_tree: bool) -> tuple[dict[str, Any], bool]:
    """Build a versioned receipt and its overall fail-closed verdict."""
    if wheel is not None and allow_source_tree:
        raise ValueError("--wheel and --source-tree are mutually exclusive")
    if wheel is not None:
        artifact = wheel_provenance(wheel)
    elif allow_source_tree:
        artifact = source_tree_provenance()
    else:
        artifact = {
            "artifact": "mixle.unverified_artifact/v1",
            "verified": False,
            "reason": "provide --wheel for release reproduction or --source-tree for development",
        }
    checks = evaluate_claims(claim_checks())
    installed = environment()
    passed = (
        bool(artifact.get("verified"))
        and bool(installed["installed_content"].get("verified"))
        and all(check["passed"] for check in checks.values())
    )
    return (
        {
            "artifact": "mixle.reproduction_receipt/v2",
            "passed": passed,
            "subject": artifact,
            "environment": installed,
            "checks": checks,
        },
        passed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Mixle's release claims and emit a fail-closed receipt.")
    parser.add_argument("--wheel", type=Path, help="exact installed wheel to hash and verify")
    parser.add_argument(
        "--source-tree",
        action="store_true",
        help="development-only: bind the receipt to the checkout containing mixle.reproduction",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON receipt here (default: stdout)")
    args = parser.parse_args(argv)
    try:
        receipt, passed = build_receipt(wheel=args.wheel, allow_source_tree=args.source_tree)
        text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
        if args.out is not None:
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
