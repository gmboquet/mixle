"""L8 DoD -- multi-climate-model ensemble fusion, skill-weighted BMA (notes/exec/workstream-L.md).

``skill_weighted_fuse`` adds no new fusion math: each :class:`ClimateMember` (one CMIP member or AI emulator,
with a per-member ``skill`` = inverse validation-error against held-out observations) maps straight onto an
L5 :class:`ModelClaim` with ``reliability = skill``, then goes through the frozen precision-weighted
:func:`fuse_claims` rule. Two scenarios exercise the ensemble:

* two climate stubs with a 3:1 skill ratio and equal variance fuse with a 3:1 weight ratio, and stay well
  under the 3-sigma disagreement flag.
* a strongly disagreeing pair (``|2 - 4| / sqrt(0.1 + 0.1) = 4.47 > 3`` sigma) trips ``disagreement`` and,
  since neither claim survives cross-model adjudication that far apart, ``abstained`` too.
"""

from __future__ import annotations

import math

import pytest

from mixle.reason.fusion import ClimateMember, FusedBelief, skill_weighted_fuse


def test_skill_weights_track_skill():
    fused = skill_weighted_fuse(
        [
            ClimateMember(
                value=2.0, variance=0.1, model_id="emulator_hi", version="v1", content_hash="a" * 64, skill=3.0
            ),
            ClimateMember(
                value=2.2, variance=0.1, model_id="emulator_lo", version="v1", content_hash="b" * 64, skill=1.0
            ),
        ]
    )
    assert isinstance(fused, FusedBelief)
    assert fused.weights["emulator_hi"] > fused.weights["emulator_lo"]
    ratio = fused.weights["emulator_hi"] / fused.weights["emulator_lo"]
    assert ratio == pytest.approx(3.0, rel=0.05)
    assert fused.disagreement is False
    assert fused.abstained is False

    by_id = {entry["model_id"]: entry for entry in fused.provenance["claims"]}
    assert by_id["emulator_hi"]["skill"] == pytest.approx(3.0)
    assert by_id["emulator_lo"]["skill"] == pytest.approx(1.0)
    assert by_id["emulator_hi"]["content_hash"] == "a" * 64


def test_strongly_disagreeing_ensemble_abstains():
    z = abs(2.0 - 4.0) / math.sqrt(0.1 + 0.1)
    assert z == pytest.approx(4.4721, abs=1e-3)

    fused = skill_weighted_fuse(
        [
            ClimateMember(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="c" * 64),
            ClimateMember(value=4.0, variance=0.1, model_id="cmipB", version="v1", content_hash="d" * 64),
        ]
    )
    assert fused.disagreement is True
    assert fused.abstained is True
