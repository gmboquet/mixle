#!/usr/bin/env python
"""Import every public ``mixle`` module; exit non-zero if any fails (worklist P2.1).

Run this against a CLEAN base install of the built wheel (no extras). An optional dependency imported
at a module's top level breaks ``import mixle...`` for every base-install user -- a class of regression
that shipped in the wheel before (numba, mpi4py). Sweeping the installed package catches it before
release.

Invoke as ``python scripts/import_sweep.py`` (the script's directory, not the repo root, is on
``sys.path[0]``, so the source tree cannot shadow the installed package). The ``site-packages`` guard
refuses to "pass" against an editable/source checkout -- pass ``--allow-editable`` only for a local
smoke run where you understand it is not the wheel.
"""

import importlib
import pkgutil
import sys

import mixle


def main() -> int:
    if "site-packages" not in mixle.__file__ and "--allow-editable" not in sys.argv:
        print(f"refusing to sweep a source/editable checkout (not the wheel): {mixle.__file__}", file=sys.stderr)
        return 2
    failures: list[str] = []
    swept = 0
    for module in pkgutil.walk_packages(mixle.__path__, prefix="mixle."):
        name = module.name
        if ".tests" in name or name.endswith("_test"):
            continue
        swept += 1
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- any import failure is a base-install regression
            failures.append(f"  {name}: {type(exc).__name__}: {exc}")
    print(f"swept {swept} public modules from {mixle.__file__}")
    if failures:
        print(
            "import failures (an unguarded optional import breaks the base install):",
            file=sys.stderr,
        )
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("all public modules import cleanly from the clean install")
    return 0


if __name__ == "__main__":
    sys.exit(main())
