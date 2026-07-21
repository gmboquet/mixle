"""Regression test: importing `mixle.stats.bayes.dirichlet` (or anything that reaches it before
`mixle`'s own top-level `__init__` has warmed `mixle.stats.bayes.dirichlet`) must not raise a
circular-import `ImportError`.

This is exactly the path any out-of-tree consumer package (e.g. `mixle_pde`, which does
`from mixle.ppl.core import RandomVariable` as its own first import) hits in a fresh process: that
chain reaches `mixle.stats.bayes.dirichlet` -> `mixle.inference.fisher` -> `mixle.inference`
package-init -> `mixle.inference.risk` -> `mixle.analysis` -> `mixle.reason` -> ... ->
`mixle.stats.latent.mixture` -> back to `mixle.stats.bayes.dirichlet`, which is still mid-init.
Regression-tests the fix: `mixle.stats.latent.mixture`'s `DirichletDistribution` import deferred to
call time, and `mixle.inference`'s `risk` re-exports made lazy (mirroring the existing
`condition`/`scenario` lazy-export pattern in `mixle/inference/__init__.py`).
"""

import subprocess
import sys


def test_dirichlet_module_imports_standalone_in_a_fresh_process():
    """A fresh process importing `mixle.stats.bayes.dirichlet` directly (not via `import mixle`
    first) must not raise ImportError."""
    result = subprocess.run(
        [sys.executable, "-c", "import mixle.stats.bayes.dirichlet"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_inference_risk_names_are_reachable_after_plain_import_mixle():
    """`mixle.inference.value_at_risk` (a name whose eager import used to sit on the exact cycle)
    must still be reachable -- lazily-exported, not silently dropped."""
    result = subprocess.run(
        [sys.executable, "-c", "import mixle.inference as m; assert callable(m.value_at_risk)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
