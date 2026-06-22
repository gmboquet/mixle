"""Backward-compatibility shim: ``pysp.stats.leaf`` was renamed to ``pysp.stats.base`` (the "leaf"
terminology was dropped in favour of "base distributions").

Deep imports (``from pysp.stats.leaf.gaussian import X``) resolve to the *identical* ``pysp.stats.base``
submodule objects -- each base submodule is registered under its ``pysp.stats.leaf.*`` name in
``sys.modules``, so there is exactly one module (and one class) object, never a duplicate. Prefer
``pysp.stats`` (the public surface) or ``pysp.stats.base`` going forward.
"""

import importlib
import pkgutil
import sys

from pysp.stats import base as _base

for _info in pkgutil.iter_modules(_base.__path__):
    sys.modules[__name__ + "." + _info.name] = importlib.import_module("pysp.stats.base." + _info.name)
