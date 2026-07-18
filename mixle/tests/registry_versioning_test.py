"""Regression tests for two fixed defects in mixle.inference.production.registry.Registry.

1. ``register`` numbered the next version from ``len(versions())``, not the highest existing version
   number -- so a registry missing a version (deleted out of band, or a future ``delete()``) reused
   that number and the next ``register`` call silently overwrote a DIFFERENT, later version.
2. ``promote`` wrote the alias file directly (``open(..., "w")``), not atomically -- a crash or a
   concurrent reader mid-write could observe a truncated/partial alias file instead of the old or
   new value.
"""

import os
import tempfile

import pytest

import mixle.stats as st
from mixle.inference.production.registry import Registry


def test_register_after_a_missing_version_does_not_reuse_its_number():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        v1 = reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        v2 = reg.register(st.GaussianDistribution(1.0, 1.0), "m")
        v3 = reg.register(st.GaussianDistribution(2.0, 1.0), "m")
        assert (v1, v2, v3) == ("v1", "v2", "v3")

        # simulate v2 having been removed (deleted out of band; the class has no delete() of its own)
        os.remove(os.path.join(reg._dir("m"), "v2.json"))
        assert reg.versions("m") == ["v1", "v3"]

        v4 = reg.register(st.GaussianDistribution(3.0, 1.0), "m")
        assert v4 == "v4", "the next version must come from the highest existing number, not the count"

        # v3 must still be exactly what it was -- never overwritten
        model_v3, _ = reg.get("m", "v3")
        assert model_v3.mu == 2.0
        model_v4, _ = reg.get("m", "v4")
        assert model_v4.mu == 3.0


def test_promote_writes_the_alias_atomically():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        reg.register(st.GaussianDistribution(1.0, 1.0), "m")
        reg.promote("m", "v2")

        alias_path = os.path.join(reg._dir("m"), "production.alias")
        assert os.path.exists(alias_path)
        assert open(alias_path).read() == "v2"
        # no leftover temp file from the atomic-write dance
        assert [f for f in os.listdir(reg._dir("m")) if f.endswith(".tmp")] == []

        model, _ = reg.current("m")
        assert model.mu == 1.0


def test_promote_rejects_an_unknown_version_without_touching_the_alias():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        reg.promote("m", "v1")
        with pytest.raises(KeyError):
            reg.promote("m", "v99")
        # the failed promote must not have clobbered the existing alias
        alias_path = os.path.join(reg._dir("m"), "production.alias")
        assert open(alias_path).read() == "v1"
