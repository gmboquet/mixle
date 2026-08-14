"""Installed, fail-closed reproduction receipt for Mixle's release claims."""

from __future__ import annotations

import argparse
import base64
import binascii
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


def _is_installer_bytecode(name: str) -> bool:
    """True for a byte-compiled module the INSTALLER produced, not a distributed file.

    Deliberately narrow: the name must end in ``.pyc`` AND sit directly inside a ``__pycache__``
    directory, which is the only place ``pip``'s compileall step writes. A ``.pyc`` shipped
    anywhere else in a distribution is not installer output and stays subject to the hash rule.
    """
    if not name.endswith(".pyc"):
        return False
    parts = name.split("/")
    return len(parts) >= 2 and parts[-2] == "__pycache__"


def _bytecode_source_path(name: str) -> str | None:
    """Map ``pkg/__pycache__/mod.cpython-312.pyc`` back to the ``pkg/mod.py`` it was compiled from.

    Returns ``None`` when the name does not decompose, so an undecodable bytecode path is treated
    as unaccounted-for rather than silently exempted.
    """
    parts = name.split("/")
    if len(parts) < 2 or parts[-2] != "__pycache__":
        return None
    stem = parts[-1].split(".")[0]
    if not stem:
        return None
    return "/".join([*parts[:-2], f"{stem}.py"])


def _orphaned_package_files() -> int:
    """Count importable mixle files present with no distribution metadata behind them.

    A wheel install that fails partway -- a corrupt or truncated archive -- leaves the files it had
    already unpacked in place, and pip writes the ``.dist-info`` LAST. The result imports and
    answers ``mixle.__version__`` as ``0+unknown`` while being a fragment of a package (SYS-06).
    """
    try:
        import mixle

        root = Path(next(iter(mixle.__path__)))
    except Exception:  # noqa: BLE001 - nothing importable is a real answer, not an error
        return 0
    try:
        return sum(1 for path in root.rglob("*") if path.is_file())
    except OSError:
        return 0


def installed_content_provenance() -> dict[str, Any]:
    """Digest the installed distribution's RECORD identities and verify hashed files."""
    try:
        dist = distribution("mixle")
    except PackageNotFoundError:
        # Distinguish "no mixle here" from "a partially installed mixle here". Both fail closed,
        # but they are different situations with different remedies, and reporting them
        # identically understated the second: an importable package with no metadata behind it is
        # a broken install that must be REMOVED, not an absent one that can simply be installed
        # over (SYS-06). Installing on top leaves whatever orphaned files the new wheel does not
        # happen to overwrite.
        orphaned = _orphaned_package_files()
        if orphaned:
            return {
                "artifact": "mixle.installed_content/v1",
                "verified": False,
                "reason": (
                    f"partial install: {orphaned} importable mixle files are present with no "
                    f"distribution metadata, so this environment is a fragment of a package rather "
                    f"than an installation of one. Remove the orphaned package directory and "
                    f"install a hash-verified artifact; installing over it can leave orphans behind."
                ),
                "orphaned_file_count": orphaned,
            }
        return {
            "artifact": "mixle.installed_content/v1",
            "verified": False,
            "reason": "mixle distribution metadata is not installed",
        }
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    hashed_names = {str(item) for item in (dist.files or ()) if item.hash is not None}
    bytecode_exempt: list[str] = []
    for item in sorted(dist.files or (), key=str):
        if item.hash is None:
            # Skipping unhashed installed entries is the same tamper vector as the wheel side
            # (SYS-RR8-2): only RECORD may legitimately lack a hash. Installer-written metadata
            # (INSTALLER, REQUESTED, direct_url.json) is not distributed code and is exempt.
            name = str(item)
            if _is_installer_bytecode(name):
                # Ordinary `pip install` byte-compiles every module and writes those .pyc rows to
                # RECORD with no hash, because it generates them AFTER hashing the distributed
                # files. Treating them as tampering made `mixle-reproduce` report passed:false on
                # every normal installation (SYS-02) -- the check refused the standard install path
                # rather than any actual modification.
                #
                # They are exempted, not ignored: each one must correspond to a distributed source
                # file that IS hashed and verified below, the count is published in the receipt,
                # and a .pyc with no such source is still a failure. What this does NOT prove is
                # that the bytecode matches the source it names -- CPython's own invalidation
                # (mtime/size, or hash for `--invalidation-mode`) is what forces a recompile from
                # the verified .py, and a `unchecked_hash` .pyc would bypass that. The verified
                # object is the distributed source, which is what the reproduction claim is about.
                source = _bytecode_source_path(name)
                if source is None or source not in hashed_names:
                    failures.append(f"{item}: installer bytecode has no verified source file")
                else:
                    bytecode_exempt.append(name)
            elif not name.endswith(".dist-info/RECORD") and not name.startswith(("mixle-", "mixle.")):
                failures.append(f"{item}: installed entry carries no RECORD hash")
            elif name.endswith((".py", ".so", ".pyd", ".json")) and ".dist-info/" not in name:
                failures.append(f"{item}: installed code entry carries no RECORD hash")
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
        "installer_bytecode_exempt_count": len(bytecode_exempt),
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


def _wheel_record_hashes(path: Path) -> dict[str, str]:
    """Parse the wheel's RECORD into ``{path: sha256-hex}`` for every hashed entry."""
    with zipfile.ZipFile(path) as archive:
        record_names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("wheel must contain exactly one RECORD")
        text = archive.read(record_names[0]).decode("utf-8")
    hashes: dict[str, str] = {}
    unhashed: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            raise ValueError(f"wheel RECORD line is malformed: {line!r}")
        name, hash_field, _size = parts
        if not hash_field:
            # RECORD cannot hash itself; ANY other unhashed row is the tamper vector -- strip a
            # file's hash and the comparison silently skipped it, so altered code installed and
            # received a passing receipt (SYS-RR8-2). Collected, then rejected below.
            unhashed.append(name)
            continue
        algorithm, _, value = hash_field.partition("=")
        if algorithm != "sha256" or not value:
            raise ValueError(f"wheel RECORD entry {name!r} uses unsupported hash {algorithm!r}")
        if name in hashes:
            raise ValueError(f"wheel RECORD lists {name!r} more than once")
        try:
            hashes[name] = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).hex()
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"wheel RECORD entry {name!r} has a malformed hash") from exc
        if len(hashes[name]) != 64:
            raise ValueError(f"wheel RECORD entry {name!r} has a malformed sha256 digest")
    unexpected = [name for name in unhashed if not name.endswith(".dist-info/RECORD")]
    if unexpected:
        raise ValueError(f"wheel RECORD omits hashes for {len(unexpected)} entrie(s): {sorted(unexpected)[:5]}")
    if len(unhashed) != 1:
        raise ValueError("wheel RECORD must contain exactly one self-referencing unhashed row")
    if not hashes:
        raise ValueError("wheel RECORD carries no hashed entries")
    return hashes


def subject_binding(path: Path, build: dict[str, Any]) -> dict[str, Any]:
    """Bind the receipt's SUBJECT wheel to the distribution that is actually executing.

    Name/version equality is not identity: any clean ``mixle==X`` wheel matches the installed
    version string, so the adversarial review presented an older 0.8.0 wheel as the subject while
    a newer 0.8.0 build executed, and the receipt reported ``passed`` for bytes that never ran
    (SYS-RR7-3). Two comparisons close that: the installed package's embedded build provenance
    must equal the subject wheel's (source commit, tree, and content digest), and every hashed
    entry in the subject wheel's RECORD must appear in the installed RECORD with the same digest.
    The installed RECORD may carry extra installer-written files (INSTALLER, direct_url.json);
    the wheel's own entries are the code, and those are what must match.
    """
    mismatches: list[str] = []
    # Establish WHICH CODE RAN before trusting anything found beside it. A PYTHONPATH shadow
    # package supplied its own __init__/reproduction modules while extending __path__ to the real
    # installation, and produced a receipt byte-identical to the legitimate one (SYS-RR8-3):
    # provenance was read next to the shadow while distribution() independently located the real
    # metadata. Reading provenance first would let a copied-in provenance file decide the outcome,
    # so the import-path identity is settled up front.
    import mixle as _installed
    from mixle import reproduction as _running

    package_roots = [Path(p).resolve() for p in getattr(_installed, "__path__", [])]
    try:
        distribution_root = Path(distribution("mixle").locate_file("mixle")).resolve()
    except PackageNotFoundError:
        return {
            "artifact": "mixle.subject_binding/v1",
            "verified": False,
            "mismatches": ["mixle distribution metadata is not installed"],
        }
    if len(package_roots) != 1:
        mismatches.append(
            f"imported mixle has {len(package_roots)} package roots; a release replay requires exactly one"
        )
    for root in package_roots:
        if root != distribution_root:
            mismatches.append(f"imported package root {root} is not the installed distribution at {distribution_root}")
    running_file = Path(_running.__file__).resolve()
    if not running_file.is_relative_to(distribution_root):
        mismatches.append(f"executing module {running_file} is outside the installed distribution")

    provenance_path = Path(_installed.__file__).resolve().parent / "_build_provenance.json"
    try:
        installed_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "artifact": "mixle.subject_binding/v1",
            "verified": False,
            "mismatches": [*mismatches, f"installed build provenance unreadable: {exc}"],
        }
    for field in ("source_commit", "source_tree", "source_content_sha256"):
        if installed_provenance.get(field) != build.get(field):
            mismatches.append(
                f"build provenance {field}: wheel={build.get(field)!r} installed={installed_provenance.get(field)!r}"
            )
    try:
        wheel_hashes = _wheel_record_hashes(path)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        # append rather than replace: the import-path mismatches found above are the more
        # diagnostic half of a shadowed-install failure and must not be discarded here
        return {
            "artifact": "mixle.subject_binding/v1",
            "verified": False,
            "mismatches": [*mismatches, str(exc)],
        }
    installed_hashes: dict[str, str] = {}
    try:
        dist = distribution("mixle")
    except PackageNotFoundError:
        mismatches.append("mixle distribution metadata is not installed")
    else:
        for item in dist.files or ():
            if item.hash is not None and item.hash.mode == "sha256":
                value = item.hash.value
                installed_hashes[str(item)] = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).hex()
    for name, digest in sorted(wheel_hashes.items()):
        if installed_hashes.get(name) != digest:
            mismatches.append(f"RECORD content differs at {name}")
            if len(mismatches) >= 10:
                mismatches.append("further RECORD mismatches suppressed")
                break
    return {
        "artifact": "mixle.subject_binding/v1",
        "verified": not mismatches,
        "compared_entries": len(wheel_hashes),
        # recorded so two receipts can be compared on WHERE the code ran, not only on its digests
        "executing_module": str(running_file),
        "package_roots": [str(root) for root in package_roots],
        "distribution_root": str(distribution_root),
        "mismatches": mismatches,
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
        # the receipt's subject must be the wheel whose code EXECUTES, not merely one sharing its
        # version string (SYS-RR7-3)
        artifact["installed_binding"] = subject_binding(wheel, artifact["build"])
        artifact["verified"] = bool(artifact["verified"]) and bool(artifact["installed_binding"]["verified"])
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
            # The receipt names its own coverage so it cannot be presented as more than it is
            # (SYS-RR7-4): the installed command runs the self-contained claim checks against the
            # bound artifact; the reproduction-bundle entries are SOURCE-CHECKOUT receipts,
            # produced separately, because the wheel ships neither the bundle nor its inputs.
            "scope": {
                "claim_checks": sorted(_EXPECTATIONS),
                "reproduction_bundle_entries": (
                    "not included; replay each entry from the source checkout with "
                    "`python scripts/run_repro_entry.py --entry <id>` and attach those receipts"
                ),
            },
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
