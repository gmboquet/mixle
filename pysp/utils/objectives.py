"""Backward-compatibility shim — moved to pysp.inference.objectives in the concern-oriented reorg."""
from importlib import import_module as _imp

from pysp.inference.objectives import *  # noqa: F401,F403

_src = _imp("pysp.inference.objectives")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
