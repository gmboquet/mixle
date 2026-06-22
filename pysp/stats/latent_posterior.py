"""Backward-compatibility shim — moved to pysp.sampling.latent_posterior in the concern-oriented reorg."""
from importlib import import_module as _imp

from pysp.sampling.latent_posterior import *  # noqa: F401,F403

_src = _imp("pysp.sampling.latent_posterior")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
