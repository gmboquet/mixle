"""Volume -> claims -> calibrated report: the mission vertical (workstream B2), as a demo.

"Every sentence carries a receipt" -- literally, here. The full composition on synthetic data:

  1. synthetic 3-D "volumes" with a planted structure, split into octant patches and scored by a small
     frozen numpy "model" (mean intensity per patch) -- the "patches scored by a model" input side.
  2. those patch scores become STRUCTURED CLAIMS via :func:`~mixle.task.structured_out.solve_structured`:
     a categorical claim (``shape``: which planted pattern the volume shows) and a numeric claim
     (``brightness``: its overall intensity), each distilled into its own calibrated local student.
  3. each claim is gated per-claim, not as one top-level generation:
       * ``shape`` (a small finite candidate set) rides :class:`~mixle.task.calibrated_generator.CalibratedGenerator`
         (workstream A1) directly -- draw the 3 candidate labels, score them from the patches, and calibrate
         a conformal accept-singleton-or-abstain rule, so an ambiguous volume abstains instead of guessing.
       * ``brightness`` (continuous) rides its own split-conformal regression machinery directly
         (``RegressionSolution.answers_locally`` / ``.qhat``) -- CalibratedGenerator's finite-candidate-set
         model is the right fit for ``shape``'s 3 discrete labels, not for a continuous estimate; the spec
         explicitly allows composing with "its underlying conformal machinery directly" per claim, so each
         claim uses whichever of the two shapes fits it.
  4. a report object (:class:`VolumeClaimReport`) whose ``.summary()`` explicitly flags, per claim, whether
     it was accepted (with the accepted value) or abstained (with why) -- an inspectable, not asserted,
     receipt.
  5. an OPTIONAL :func:`~mixle.inference.structure.learn_structure` consistency check: does the discovered
     dependency forest over the claim fields recover the ``shape -> brightness`` link the synthetic data
     actually has? (see ``consistency_check()`` below.)

Design note on the synthetic generator: B1 (``examples/multimodal_stage1_demo.py``, PR #132) has a
similar planted-structure volume generator (``blob_left`` / ``blob_right`` / ``stripe``), but it is
entangled with that example's own point (a frozen-encoder/frozen-LM GradLeaf training pipeline), which
this demo does not need -- the claim-extraction step here is a small numpy patch scorer, not a trained
network. Rather than import across branches for a few lines, this file reimplements a standalone,
self-contained version of the same three-pattern idea (credited here); B1's generator was "helpful", not
required, per the roadmap note.

Run: ``python examples/calibrated_report_demo.py``
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.structure import learn_structure
from mixle.task.calibrated_generator import ABSTAIN, CalibratedGenerator
from mixle.task.structured_out import solve_structured

SIZE = 8  # toy volumes are SIZE x SIZE x SIZE voxels
SHAPES = ("blob_left", "blob_right", "stripe")


# --- synthetic volumes + patch scoring ("patches scored by a model") --------------------------------


def synthetic_volume(shape: str, rng: np.random.RandomState, *, ambiguous: bool = False) -> np.ndarray:
    """A planted structure that determines the claims: a small bright cube in the left/right half of the
    volume, or a bright full-width plane. ``ambiguous=True`` plants a much fainter signature -- still the
    same shape, but one a calibrated gate should be honestly unsure about."""
    vol = 0.05 * rng.randn(SIZE, SIZE, SIZE).astype("float64")
    bump = 0.35 if ambiguous else 1.0
    if shape in ("blob_left", "blob_right"):
        lo, hi = (0, SIZE // 2 - 2) if shape == "blob_left" else (SIZE // 2, SIZE - 2)
        cx, cy, cz = rng.randint(lo, hi + 1), rng.randint(1, SIZE - 2), rng.randint(1, SIZE - 2)
        vol[cx : cx + 2, cy : cy + 2, cz : cz + 2] += bump
    else:  # stripe: a bright plane spanning the full width at a random depth
        d = rng.randint(0, SIZE)
        vol[d, :, :] += bump * 0.6
    return vol


def score_patches(volume: np.ndarray) -> tuple[float, ...]:
    """The "model" that scores patches: mean intensity of each of the volume's 8 octants. Frozen and
    deterministic -- any patch-level scorer (a trained encoder, a hand-rolled filter bank) would slot in
    here unchanged; downstream code only depends on the 8-tuple of patch scores it returns."""
    h = SIZE // 2
    patches = []
    for i in (0, h):
        for j in (0, h):
            for k in (0, h):
                patches.append(float(volume[i : i + h, j : j + h, k : k + h].mean()))
    return tuple(patches)


def build_records(n_per_shape: int, seed: int, *, ambiguous_fraction: float = 0.0) -> list[tuple[float, ...]]:
    """Patch-score records for ``n_per_shape`` volumes of each planted shape. A fraction of each shape's
    volumes are planted with a much fainter signature (``ambiguous=True``) -- genuinely harder cases a
    calibrated gate should sometimes abstain on, not a coin flip sprinkled in after the fact."""
    rng = np.random.RandomState(seed)
    records = []
    for shape in SHAPES:
        n_ambiguous = int(round(n_per_shape * ambiguous_fraction))
        for i in range(n_per_shape):
            records.append(score_patches(synthetic_volume(shape, rng, ambiguous=i < n_ambiguous)))
    rng.shuffle(records)
    return records


# --- structured claims via solve_structured ----------------------------------------------------------


def claim_teacher(record: tuple[float, ...]) -> dict[str, Any]:
    """The rigid rule this demo distills: whichever half is brighter wins blob_left/blob_right, unless
    the patches are too uniform to localize a blob at all (small spread) -- then it's a stripe. This
    stands in for "whatever currently produces claims" (a rule, a bigger model, a human annotator) that
    ``solve_structured`` turns into per-field calibrated local students."""
    left = sum(record[:4]) / 4
    right = sum(record[4:]) / 4
    spread = max(record) - min(record)
    shape = "stripe" if spread < 0.15 else ("blob_left" if left > right else "blob_right")
    brightness = sum(record) / len(record)
    return {"shape": shape, "brightness": brightness}


# --- per-claim conformal gates --------------------------------------------------------------------


def _stable_unit(obj: Any) -> float:
    """Deterministic pseudo-noise in [0, 1) from a hash -- keeps candidate scoring a pure function of
    the candidate, matching the pattern in ``task_calibrated_generator_test.py``."""
    digest = hashlib.sha256(repr(obj).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _generate_shape_candidates(record: tuple[float, ...], k: int, rng=None) -> list[tuple[tuple[float, ...], str]]:
    """Every candidate claim is scoreable on its own (per ``CalibratedGenerator``'s contract, ``score``
    only sees the candidate) -- so each candidate embeds the record alongside a guessed label, the same
    ``(prompt, guess)`` shape ``task_calibrated_generator_test.py`` uses."""
    if rng is None:
        rng = np.random.default_rng()
    labels = list(SHAPES[:k]) if k <= len(SHAPES) else list(SHAPES) * ((k // len(SHAPES)) + 1)
    labels = labels[:k]
    rng.shuffle(labels)
    return [(record, label) for label in labels]


def _score_shape_candidate(candidate: tuple[tuple[float, ...], str]) -> float:
    """How well ``label`` matches ``record``'s patch pattern -- a pure function of the candidate, with a
    little deterministic jitter so ties don't collapse identically (mirrors the calibrated-generator test)."""
    record, label = candidate
    left = sum(record[:4]) / 4
    right = sum(record[4:]) / 4
    spread = max(record) - min(record)
    if label == "blob_left":
        base = left - right
    elif label == "blob_right":
        base = right - left
    else:
        base = -spread
    return base + _stable_unit(candidate) * 0.05


def build_shape_gate(cal_records: list[tuple[float, ...]], *, alpha: float = 0.1, seed: int = 0) -> CalibratedGenerator:
    """Calibrate the per-claim conformal gate (workstream A1) for the ``shape`` claim: a singleton
    conformal set -> serve that label; empty/multi -> :data:`ABSTAIN`, exactly as ``CalibratedGenerator``
    already does for open-ended generation, here applied to a 3-way structured claim instead."""

    def is_correct(record: tuple[float, ...], candidate: tuple[tuple[float, ...], str]) -> bool:
        _, label = candidate
        return label == claim_teacher(record)["shape"]

    gate = CalibratedGenerator(_generate_shape_candidates, _score_shape_candidate, alpha=alpha, k=3, seed=seed)
    return gate.calibrate(cal_records, is_correct)


# --- the report object: every claim carries a receipt --------------------------------------------


@dataclass
class ClaimVerdict:
    """One claim's accept-or-abstain decision, with the value (if accepted) and why."""

    field: str
    value: Any
    status: str  # "accepted" | "abstained"
    detail: str = ""


@dataclass
class VolumeClaimReport:
    """Per-volume receipt: which claims were accepted (with values) vs abstained (with why). Mirrors the
    ``name -> "pass"/"fail"/"absent"`` shape of :class:`mixle.inference.receipt.VerificationReport` --
    a report never invents a value for a claim it didn't accept."""

    volume_id: int
    verdicts: dict[str, ClaimVerdict] = field(default_factory=dict)

    @property
    def all_accepted(self) -> bool:
        return all(v.status == "accepted" for v in self.verdicts.values())

    def accepted(self) -> dict[str, Any]:
        return {name: v.value for name, v in self.verdicts.items() if v.status == "accepted"}

    def abstained(self) -> list[str]:
        return [name for name, v in self.verdicts.items() if v.status == "abstained"]

    def summary(self) -> str:
        return ", ".join(f"{name}={v.status}" for name, v in self.verdicts.items())


def build_claim_report(
    volume_id: int,
    record: tuple[float, ...],
    shape_gate: CalibratedGenerator,
    structured: Any,
    *,
    seed: int | None = None,
) -> VolumeClaimReport:
    """Run every claim through its own gate and bind the outcomes into one inspectable report."""
    verdicts: dict[str, ClaimVerdict] = {}

    served = shape_gate.serve(record, seed=seed)
    if served is ABSTAIN:
        verdicts["shape"] = ClaimVerdict(
            "shape", None, "abstained", detail="no shape candidate cleared the conformal threshold"
        )
    else:
        _, label = served
        verdicts["shape"] = ClaimVerdict("shape", label, "accepted", detail="conformal singleton")

    brightness = structured.fields_num["brightness"]
    if brightness.answers_locally:
        yhat, lo, hi = brightness.interval(record)
        verdicts["brightness"] = ClaimVerdict(
            "brightness", round(yhat, 4), "accepted", detail=f"[{lo:.3f}, {hi:.3f}] within tol {brightness.tol}"
        )
    else:
        verdicts["brightness"] = ClaimVerdict(
            "brightness", None, "abstained", detail=f"qhat {brightness.qhat:.3f} exceeds tol {brightness.tol}"
        )

    return VolumeClaimReport(volume_id=volume_id, verdicts=verdicts)


# --- optional: learn_structure consistency check over the claim fields -----------------------------


def consistency_check(records: list[tuple[float, ...]]) -> str:
    """Does an unsupervised structure search over (shape, brightness) pairs -- with no knowledge of how
    this demo built them -- recover the ``shape -> brightness`` dependency the synthetic data actually
    has (a stripe's bright plane spans far more voxels than a blob's small cube, so mean patch brightness
    tracks shape)? A cheap sanity check that "every sentence carries a receipt" extends to the claims
    being mutually consistent, not just individually calibrated."""
    claim_rows = [(claim_teacher(r)["shape"], claim_teacher(r)["brightness"]) for r in records]
    model = learn_structure(claim_rows, min_gain=0.0)
    edges = model.edges()
    if (0, 1) in edges:
        return f"learn_structure recovered shape -> brightness ({model})"
    return f"learn_structure did NOT recover shape -> brightness ({model}) -- reported, not hidden"


# --- pipeline -----------------------------------------------------------------------------------------


def main() -> None:
    train_records = build_records(n_per_shape=120, seed=0)
    cal_records = build_records(n_per_shape=120, seed=1)

    print("structured claims: distilling solve_structured over (shape, brightness)")
    structured = solve_structured(claim_teacher, train_records, tol={"brightness": 0.08}, alpha=0.1, seed=0, epochs=200)
    print(f"   schema: {structured.schema}")

    print("\nper-claim gate: calibrating CalibratedGenerator for the shape claim")
    shape_gate = build_shape_gate(cal_records, alpha=0.1, seed=0)

    print("\nserving reports for a mix of clear and ambiguous volumes")
    probe_records = build_records(n_per_shape=40, seed=2, ambiguous_fraction=0.5)
    reports = [build_claim_report(i, r, shape_gate, structured) for i, r in enumerate(probe_records)]

    n_all_accepted = sum(r.all_accepted for r in reports)
    n_any_abstained = sum(bool(r.abstained()) for r in reports)
    print(f"   {len(reports)} volumes reported; {n_all_accepted} fully accepted, {n_any_abstained} flag >=1 abstention")
    for r in reports[:5]:
        print(f"   volume {r.volume_id}: {r.summary()}  accepted={r.accepted()}")

    print("\noptional consistency check: learn_structure over the claim fields")
    print(f"   {consistency_check(train_records)}")


if __name__ == "__main__":
    main()
