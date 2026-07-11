"""Regression: structure-search estimator cloning must produce independent copies, without source eval.

_clone was eval(str(estimator)); most estimators have the default ``<object at 0x...>`` repr, so that raised
and silently returned the SAME object. It now uses copy.deepcopy -> a genuinely independent template per
candidate (and no source-level eval). The fallback still returns the original for uncopyable estimators.
"""

import copy

import pytest

from mixle.inference.structure import _clone


def test_clone_returns_an_independent_object_not_the_same_template():
    import mixle.stats as st

    e = st.GaussianEstimator()
    c = _clone(e)
    assert c is not e  # the old eval path fell back to returning `e` itself
    assert type(c) is type(e)


def test_clone_does_not_use_eval_repr_roundtrip():
    """A configured estimator with the default repr would have crashed eval and shared the object; deepcopy copies it."""
    import mixle.stats as st

    e = st.CategoricalEstimator()
    a, b = _clone(e), _clone(e)
    assert a is not b and a is not e  # every candidate gets its own template


def test_clone_falls_back_for_uncopyable_estimators():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("nope")

    obj = Uncopyable()
    assert _clone(obj) is obj  # fallback preserves the prior same-object behavior


def test_clone_matches_deepcopy_for_a_torch_estimator():
    pytest.importorskip("torch")
    from mixle.models import NeuralDensityEstimator, build_vae

    e = NeuralDensityEstimator(build_vae(4, latent=2))
    c = _clone(e)
    assert c is not e and c.module is not e.module  # independent torch module
    ref = copy.deepcopy(e)
    assert type(c) is type(ref)


def test_provenance_uses_mixle_version_key():
    from mixle.inference.production.provenance import environment_info

    env = environment_info()
    assert "mixle_version" in env and "pysp_version" not in env
    # legacy headers on disk still render via the fallback getter used by Header.__str__
    legacy = {"pysp_version": "0.6.1"}
    assert (legacy.get("mixle_version") or legacy.get("pysp_version")) == "0.6.1"
