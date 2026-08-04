#!/usr/bin/env python
"""Generate exact callable signatures for every explicitly stable module."""

from __future__ import annotations

import argparse
import enum
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mixle.maturity import MATURITY_REGISTRY, Maturity

MANIFEST = ROOT / "manifests" / "stable_api_signatures.json"
NON_MODULE_STABLE_NAMES = {"mixle.inference.optimize"}


def _annotation(value: Any) -> str | None:
    return None if value is inspect.Parameter.empty else inspect.formatannotation(value)


def _default(value: Any) -> dict[str, Any]:
    if value is inspect.Parameter.empty:
        return {"present": False}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"present": True, "value": value}
    return {"present": True, "repr": repr(value)}


def _signature(value: Any) -> dict[str, Any]:
    signature = inspect.signature(value)
    return {
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "default": _default(parameter.default),
                "annotation": _annotation(parameter.annotation),
            }
            for parameter in signature.parameters.values()
        ],
        "return_annotation": _annotation(signature.return_annotation),
    }


def stable_modules() -> list[str]:
    modules = []
    for name, (tier, _) in MATURITY_REGISTRY.items():
        if tier is not Maturity.STABLE or name in NON_MODULE_STABLE_NAMES:
            continue
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"stable maturity entry is not an importable module: {name}")
        modules.append(name)
    return sorted(modules)


def _module_entries(module_name: str) -> dict[str, dict[str, Any]]:
    module = importlib.import_module(module_name)
    entries: dict[str, dict[str, Any]] = {}
    for name, value in inspect.getmembers(module):
        if name.startswith("_") or getattr(value, "__module__", None) != module_name:
            continue
        dotted = f"{module_name}.{name}"
        if inspect.isfunction(value):
            entries[dotted] = {"kind": "function", "signature": _signature(value)}
        elif inspect.isclass(value):
            if issubclass(value, enum.Enum):
                # An Enum's "constructor signature" belongs to CPython's EnumMeta.__call__, not to
                # this project, and CPython CHANGED it between supported interpreters: 3.11 reports
                # (value, names=None, ...) and 3.12 reports (*values). Recording it made the manifest
                # interpreter-dependent, so the 3.11 lane could never match a manifest generated on
                # 3.12 -- a permanently red gate reporting a stale API that had not changed.
                # The reviewable surface of an enum is its MEMBER NAMES, which is what is recorded.
                entries[dotted] = {"kind": "enum", "members": sorted(member.name for member in value)}
                continue
            entries[dotted] = {"kind": "class-constructor", "signature": _signature(value)}
            for method_name, method in inspect.getmembers(value):
                if method_name.startswith("_") or not callable(method):
                    continue
                if getattr(method, "__qualname__", "").split(".", 1)[0] != value.__name__:
                    continue
                entries[f"{dotted}.{method_name}"] = {
                    "kind": "class-method",
                    "signature": _signature(method),
                }
    return entries


def build_manifest() -> dict[str, Any]:
    modules = stable_modules()
    entries: dict[str, dict[str, Any]] = {}
    for module_name in modules:
        entries.update(_module_entries(module_name))
    return {
        "artifact": "mixle.stable_api_signatures/v1",
        "stable_modules": modules,
        "compatibility": {
            "parameter_order": "exact",
            "parameter_kinds": "exact",
            "defaults": "exact",
            "annotations": "exact",
            "intentional_exceptions": [],
        },
        "entries": entries,
    }


def render() -> str:
    return json.dumps(build_manifest(), indent=2, sort_keys=True, allow_nan=False) + "\n"


# Must track pyproject's requires-python. inspect.formatannotation's output is version-dependent --
# 3.14 renders Optional[X] as "X | None" and Optional[ForwardRef('X')] as "ForwardRef('X') | None",
# where 3.11 and 3.12 render both the other way -- so a manifest built outside the supported range
# records annotation spellings no supported interpreter can reproduce. Since the manifest declares
# "annotations: exact" and the gate compares it verbatim against a fresh build, that makes the gate
# unpassable everywhere it actually runs. 3.11 and 3.12 agree with each other, so one artifact does
# cover the whole supported range; it just has to be built inside it.
SUPPORTED_PYTHON = ((3, 11), (3, 13))


def _require_supported_interpreter() -> None:
    """Refuse to build or check the manifest on an out-of-support interpreter."""
    low, high = SUPPORTED_PYTHON
    if not (low <= sys.version_info[:2] < high):
        raise RuntimeError(
            "stable API signatures must be generated on Python >=%d.%d,<%d.%d (this is %d.%d): "
            "annotation rendering differs outside that range, so the manifest would not match on "
            "any supported interpreter." % (low + high + sys.version_info[:2])
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        _require_supported_interpreter()
        text = render()
        if args.check:
            if MANIFEST.read_text(encoding="utf-8") != text:
                print("stable API signatures are stale; regenerate the manifest", file=sys.stderr)
                return 1
            return 0
        MANIFEST.write_text(text, encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(build_manifest()['entries'])} signatures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
