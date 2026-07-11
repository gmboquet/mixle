"""Compatibility shim: ``mixle.program`` moved to :mod:`mixle.experimental.program`.

The optimization-program approach (moves + combinators) was a reasonable exploration but not a mature surface --
its closure-taking API (``minimize(lambda: loss, over=params)``) is the PyTorch-style jank it meant to avoid --
so it now lives under :mod:`mixle.experimental`. For the common cases prefer the **declarative neural surface**::

    from mixle.ppl import Categorical, Normal, Net, free

    Categorical(logits=Net(out=10)).fit(y, given={"x": X})   # neural classification, zero closures
    Normal(Net(out=1), free).fit(y, given={"x": X})          # neural mean + learned noise (the blend)

Existing ``from mixle.program import X`` imports keep working (with a :class:`DeprecationWarning`); new
code should import from ``mixle.experimental.program`` (or use the declarative surface above).

This is a stable-namespace module, so it must not pull ``mixle.experimental`` into the stable import
graph. It therefore resolves the moved API lazily, through ``importlib`` on a string rather than a
static ``import mixle.experimental`` -- keeping the experimental boundary (enforced by
``experimental_boundary_test``) clean while the old name keeps redirecting.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.experimental.program"


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.program' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.program is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    import importlib

    return sorted(n for n in dir(importlib.import_module(_MOVED)) if not n.startswith("_"))
