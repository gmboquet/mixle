"""Documented parameters stay synchronized with runtime signatures (worklist Y4.2).

A docstring that documents a parameter the function no longer takes -- or renames -- is a stale doc that
silently misleads. This gate parses the ``Args:`` section of the stable entry points and asserts every
documented parameter actually exists in the runtime signature, so a renamed/removed argument left in the
docstring fails here. It checks the direction that is unambiguous (documented ⊆ actual); undocumented
parameters are a separate, softer concern and are not failed by this test.
"""

import importlib
import inspect
import re
import unittest

# (module, attribute) for the stable entry points whose parameter docs must not drift.
_STABLE_ENTRY_POINTS = [
    ("mixle.inference.estimation", "optimize"),
    ("mixle.inference.estimation", "fit"),
    ("mixle.inference.estimation", "best_of"),
    ("mixle.lifecycle", "propose"),
    ("mixle.utils.automatic", "get_estimator"),
]

_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
_END = re.compile(r"^\s*(Returns?|Raises|Yields|Examples?|Notes?|Attributes)\s*:\s*$")
_PARAM = re.compile(r"^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))?\s*:")


def _documented_params(fn):
    """Parameter names documented in the function's ``Args:`` section (base-indent entries only)."""
    lines = (inspect.getdoc(fn) or "").splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if _SECTION.match(l)), None)
    if start is None:
        return set()
    params, base = set(), None
    for line in lines[start:]:
        if _END.match(line):
            break
        m = _PARAM.match(line)
        if m:
            indent = len(m.group(1))
            base = indent if base is None else base
            if indent == base:  # a base-level param, not a nested continuation line
                params.add(m.group(2))
    return params


class SignatureDocstringSyncTest(unittest.TestCase):
    def test_documented_params_exist_in_signature(self):
        offenders = []
        for module_name, attr in _STABLE_ENTRY_POINTS:
            fn = getattr(importlib.import_module(module_name), attr)
            actual = set(inspect.signature(fn).parameters)
            stale = _documented_params(fn) - actual
            if stale:
                offenders.append(f"{module_name}.{attr}: documents {sorted(stale)} not in its signature")
        self.assertEqual(
            offenders,
            [],
            "stable entry points document parameters they do not take (stale docstrings):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
