"""Backward-compatibility shim — moved to pysp.enumeration.model_enumeration in the concern-oriented reorg."""
from importlib import import_module as _imp

from pysp.enumeration.model_enumeration import *  # noqa: F401,F403

_src = _imp("pysp.enumeration.model_enumeration")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
