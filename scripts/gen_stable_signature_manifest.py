#!/usr/bin/env python
"""Generate exact callable signatures for every explicitly stable module."""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
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
