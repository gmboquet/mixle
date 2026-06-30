"""Compatibility shim: ``mixle.program`` moved to :mod:`mixle.experimental.program`.

The optimization-program approach (moves + combinators) was a reasonable exploration but not a mature surface --
its closure-taking API (``minimize(lambda: loss, over=params)``) is the PyTorch-style jank it meant to avoid --
so it now lives under :mod:`mixle.experimental`. For the common cases prefer the **declarative neural surface**::

    from mixle.ppl import Categorical, Normal, Net, free

    Categorical(logits=Net(out=10)).fit(y, given={"x": X})   # neural classification, zero closures
    Normal(Net(out=1), free).fit(y, given={"x": X})          # neural mean + learned noise (the blend)

This module re-exports the old API so existing imports keep working; new code should import from
``mixle.experimental.program`` (or use the declarative surface above).
"""

from mixle.experimental.program import *  # noqa: F401, F403
