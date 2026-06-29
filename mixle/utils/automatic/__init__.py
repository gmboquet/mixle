"""Automatic detection of data type for estimators.

Builds estimators for mixle.stats. By default the plain maximum-likelihood
estimators are produced; pass use_bstats=True to build the Bayesian path, which
attaches the conjugate default prior for each family so estimation performs the
closed-form conjugate / MAP update. get_dpm_mixture fits a Dirichlet process
mixture over automatically-typed data with variational inference.

This package preserves the public surface of the former single-module
``mixle.utils.automatic``: every name that used to live there is re-exported
here, so ``import mixle.utils.automatic`` and ``from mixle.utils.automatic import
X`` keep working unchanged. The implementation is split into:

* ``factories`` -- estimator builders and conjugate default-prior helpers.
* ``profiling`` -- data profiling / model recommendation (DatumNode,
  analyze_structure, the marginal/pairwise scoring and validation helpers).
"""

from . import factories as _factories
from . import profiling as _profiling

# Re-export every public and private name from the two submodules so the
# package namespace is identical to the old flat module's namespace.
for _mod in (_factories, _profiling):
    for _name in dir(_mod):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_mod, _name)

del _mod, _name, _factories, _profiling
