"""Backward-compatibility shim — the inference facade moved to ``pysp.inference``.

``pysp.infer`` was the engine-agnostic bring-your-own-target sampler/VI facade. All sampling-based
inference now lives in the one inference concern (``pysp.inference.target`` / ``.mcmc`` / ``.backends``
/ ``.diagnostics``); import from :mod:`pysp.inference` going forward.
"""

from importlib import import_module as _imp

from pysp.inference.target import *  # noqa: F401,F403

_src = _imp("pysp.inference.target")
globals().update({_k: getattr(_src, _k) for _k in dir(_src) if not _k.startswith("__")})
