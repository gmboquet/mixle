"""The per-test global-state isolation that makes the suite order-independent (worklist T3.5).

The suite runs under ``-n auto`` (xdist distributes files across workers in a nondeterministic order) and
must give the same verdict regardless of the order tests execute in. That guarantee rests on the autouse
``_isolate_global_process_state`` fixture in ``conftest.py``, which snapshots and restores process-global
state (the default compute engine, the numpy RNG, the numpy error mode, and torch's default dtype + RNG)
around every test -- so a test that sets ``np.seterr(all="raise")`` or a non-default dtype cannot make a
later test flake.

This is the regression test for that mechanism: a first test deliberately corrupts global state, and a
second (running after it in definition order) asserts the corruption did not leak -- i.e. the fixture
restored it. If the isolation fixture were removed or broke, the second test would fail, which is exactly
the order-dependence a shuffled campaign is meant to catch, pinned deterministically here.
"""

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Module-level functions run in definition order, so `leak` is guaranteed to run before `restored`.


def test_1_leak_global_numpy_and_torch_state():
    # Corrupt process-global state that other tests legitimately depend on being at its default.
    np.seterr(all="raise")  # a later test doing 0/0 would now raise instead of producing nan
    np.random.seed(424242)
    np.random.random(16)  # advance the global RNG
    if _HAS_TORCH:
        torch.set_default_dtype(torch.float64)  # a float32 module scheduled next would hit dtype errors


def test_2_global_state_was_restored_by_isolation_fixture():
    # numpy error mode restored: the default is not "raise" (division/overflow warn, underflow ignore).
    err = np.geterr()
    assert err["over"] != "raise", f"numpy error mode leaked from a prior test: {err}"
    assert err["invalid"] != "raise", f"numpy error mode leaked from a prior test: {err}"
    if _HAS_TORCH:
        assert torch.get_default_dtype() == torch.float32, "torch default dtype leaked from a prior test"


def test_3_engine_default_is_restored():
    # The default compute engine is the numpy engine at the start of every test regardless of what ran
    # before (some tests set a torch/jax engine); the isolation fixture restores it.
    from mixle.engines.arithmetic import get_default_engine

    assert type(get_default_engine()).__name__ == "NumpyEngine"
