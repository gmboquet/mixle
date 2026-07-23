"""Compatibility shim: ``mixle.scorecard`` moved to :mod:`mixle.system.scorecard`.

``SystemScorecard`` and its evaluation/regression helpers are now part of one cohesive package,
:mod:`mixle.system`, instead of six cross-importing top-level modules. Existing
``from mixle.scorecard import X`` imports keep working (with a :class:`DeprecationWarning`); new code
should import from ``mixle.system`` (or ``mixle.system.scorecard``) directly. Not to be confused with
:class:`mixle.task.scorecard.Scorecard`, a different, narrower comparison (one student solution vs its
teacher on a task) that did not move.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.system.scorecard"


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.scorecard' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.scorecard is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    import importlib

    return sorted(n for n in dir(importlib.import_module(_MOVED)) if not n.startswith("_"))
