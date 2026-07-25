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


def test_repeated_realizations_of_the_same_model_keep_distinct_weights_and_skill():
    """Regression test: a real CMIP ensemble routinely submits several realizations of the SAME
    model (e.g. CESM2's r1i1p1f1, r2i1p1f1, ...), sharing model_id but with their own variance and
    (sometimes) their own per-realization skill. `skill_weighted_fuse` used to stamp every
    provenance entry sharing a model_id with the LAST such member's skill via an id-keyed
    `{m.model_id: m.skill for m in members}` dict -- the same collapse bug fuse_claims's `weights`
    had, one level up."""
    fused = skill_weighted_fuse(
        [
            ClimateMember(
                value=2.0, variance=1.0, model_id="CESM2", version="r1i1p1f1", content_hash="a" * 64, skill=2.0
            ),
            ClimateMember(
                value=2.2, variance=1.0, model_id="CESM2", version="r2i1p1f1", content_hash="b" * 64, skill=5.0
            ),
        ]
    )
    assert sum(fused.weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert len(fused.provenance["claims"]) == 2

    by_version = {entry["version"]: entry for entry in fused.provenance["claims"]}
    # each realization's OWN skill, not both collapsed onto the last one's (5.0)
    assert by_version["r1i1p1f1"]["skill"] == pytest.approx(2.0, abs=1e-9)
    assert by_version["r2i1p1f1"]["skill"] == pytest.approx(5.0, abs=1e-9)


def test_outlier_realization_of_the_same_model_is_rejected_not_silently_kept():
    """Regression test for MXR-080-0284/0285 at the L8 (ensemble) level: a real CMIP ensemble routinely
    submits several realizations of the SAME model (e.g. CESM2's r1i1p1f1, r2i1p1f1, ...). If ONE
    realization is a wild outlier (a blown run) while another realization of that SAME model agrees with
    an independent third model, the old model_id-keyed adjudication never actually compared the two
    CESM2 realizations against each other (both "CESM2") -- the outlier could silently ride along in the
    fused projection. It must now be identified and excluded, even though it shares a model_id with a
    realization that IS trusted.
    """
    good_realization = ClimateMember(
        value=2.0, variance=0.01, model_id="CESM2", version="r1i1p1f1", content_hash="a" * 64, skill=2.0
    )
    blown_run = ClimateMember(
        value=200.0, variance=0.01, model_id="CESM2", version="r2i1p1f1", content_hash="b" * 64, skill=2.0
    )
    corroborating_other_model = ClimateMember(
        value=2.05, variance=0.01, model_id="GFDL-ESM4", version="r1i1p1f1", content_hash="c" * 64, skill=2.0
    )
    fused = skill_weighted_fuse([good_realization, blown_run, corroborating_other_model])

    assert fused.disagreement is True
    assert fused.abstained is False  # the good realization + GFDL corroborate each other

    by_version = {entry["version"]: entry for entry in fused.provenance["claims"]}
    assert by_version["r1i1p1f1"]["adjudication_status"] == "accepted"
    assert by_version["r2i1p1f1"]["adjudication_status"] == "rejected"  # the blown run, excluded
    assert fused.mean == pytest.approx(2.025, abs=1e-6)  # only the two agreeing members, not the 200.0 outlier
