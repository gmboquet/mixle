"""Backward-compatibility shim — moved to pysp.inference.em in the concern-oriented reorg."""
from importlib import import_module as _imp

from pysp.inference.em import *  # noqa: F401,F403

_src = _imp("pysp.inference.em")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
