"""Backward-compatibility shim — moved to pysp.sampling.sampling_api in the concern-oriented reorg."""
from importlib import import_module as _imp

from pysp.sampling.sampling_api import *  # noqa: F401,F403

_src = _imp("pysp.sampling.sampling_api")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
