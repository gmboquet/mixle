"""mixle.models.transformer: build_causal_lm's torch-absent error path.

``CausalLM`` is defined only inside the module's ``if _HAS_TORCH:`` block (set once,
at import time), but ``build_causal_lm`` (defined unconditionally) referenced it with
no guard -- so on a torch-less install, calling ``build_causal_lm`` raised a bare
``NameError: name 'CausalLM' is not defined`` instead of the actionable ``ImportError``
this codebase's other torch-optional model builders raise (e.g.
``mixle.models.gaussian_process.GaussianProcessRegressor``,
``mixle.models.moe.upcycle_dense_to_moe``).

Torch is genuinely installed in this dev environment, so reproducing the bug needs a
fresh subprocess with ``torch`` blocked from the very first import -- ``_HAS_TORCH`` is
only read once, at module load, so monkeypatching it after the fact on an
already-imported module (where ``CausalLM`` is already bound regardless of the flag)
would not reproduce the failure. Combines this repo's two existing patterns for this
situation: a clean subprocess with PYTHONPATH pinned to this worktree (see
``epistemic_clean_import_test.py``), and blocking ``builtins.__import__`` for one
module name to simulate a genuinely absent optional dependency (see
``system_test.py``'s ``test_ingest_falls_back_to_a_retrievable_substrate_item_...``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SUBPROCESS_SCRIPT = """
import builtins

real_import = builtins.__import__


def _blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("simulated: torch not installed")
    return real_import(name, *args, **kwargs)


builtins.__import__ = _blocked_import

from mixle.models import transformer

assert transformer._HAS_TORCH is False, "torch should appear absent to the module"

try:
    transformer.build_causal_lm(12, 32, 2, 2, 8)
except ImportError as e:
    assert "torch" in str(e), f"expected a torch-mentioning ImportError, got: {e!r}"
    print("ok")
else:
    raise AssertionError("expected ImportError, none was raised")
"""


class BuildCausalLmMissingDependencyTest(unittest.TestCase):
    def test_raises_actionable_import_error_not_name_error(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
