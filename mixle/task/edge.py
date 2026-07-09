"""Edge distillation under hard device budgets.

:func:`mixle.task.tune_recipe` tunes one family's knobs with a soft cost penalty. An edge deployment
is a different problem: the budget is hard (bytes of flash, ops per inference, "no torch on the
device"), and the largest gains often come from choosing the right *kind* of student -- a hashed-feature MLP
versus a structured probabilistic classifier (a learned Bayesian network: kilobytes, torch-free,
exactly calibrated posteriors) -- not from tweaking widths inside one kind.

This module makes both choices searchable, and makes the search itself a model:

* :class:`DeviceSpec` / :class:`EdgeFootprint` -- the deployment budget and a student's **measured**
  cost (serialized bytes, per-inference op count, torch dependence).
* :class:`EdgeSpace` -- one design space spanning *family* (structure) and each family's training
  recipe (process), decoded from a unit cube so any optimizer can drive it.
* :class:`DesignModel` -- the model that writes the model: GP surrogates over design -> (quality,
  budget violations), proposing the next design by feasibility-weighted expected improvement
  (:func:`mixle.doe.propose_next_constrained`). It persists (``to_json``/``from_json``) and keys
  every row on a :func:`task_fingerprint`, so design knowledge **accumulates across tasks**: the
  surrogate learns task-similarity through the fingerprint coords, and a warm-started search -- even
  on a *different* task -- skips the designs that never work anywhere near it.
* :func:`distill_for_edge` -- the front door: screen candidates at reduced fidelity, promote the
  promising ones to full training, return the best student that *fits the device*, with the Pareto
  front over (bytes, agreement) and the updated :class:`DesignModel`.
* :func:`distill_designer` -- distill the accumulated design ledger into a
  torch-free structured student that predicts whether a design is worth
  training.

Teachers are consulted once per dataset (labels are cached), never per candidate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.task.distill import (
    _as_batched,
    agreement,
    distill_from_labels,
    distill_records_from_labels,
    distill_structured_from_labels,
)
from mixle.task.model import TaskModel

__all__ = [
    "DeviceSpec",
    "EdgeFootprint",
    "EdgeSpace",
    "DesignModel",
    "EdgeDistillResult",
    "distill_for_edge",
    "distill_designer",
    "footprint",
    "task_fingerprint",
    "FINGERPRINT_KEYS",
    "measure_inference_seconds",
    "measure_ops_per_second",
]


# --- device budget + measured cost ---------------------------------------------------------------


@dataclass(frozen=True)
class EdgeFootprint:
    """A student's measured deployment cost: serialized ``bytes``, per-inference ``ops`` (multiply-
    accumulates for an MLP; factor evaluations for a structured classifier), and ``torch_free``."""

    bytes: int
    ops: int
    torch_free: bool


@dataclass(frozen=True)
class DeviceSpec:
    """A hard deployment budget. ``None`` leaves an axis unconstrained.

    ``max_bytes``: model size on flash/disk. ``max_ops``: per-inference op budget (a latency proxy --
    calibrate ops/sec once per device to turn a latency target into this number). ``torch_free``:
    the device cannot run torch, so only pure-mixle students qualify.
    """

    max_bytes: int | None = None
    max_ops: int | None = None
    torch_free: bool = False

    @classmethod
    def for_latency(
        cls,
        max_ms: float,
        ops_per_second: float,
        *,
        max_bytes: int | None = None,
        torch_free: bool = False,
    ) -> DeviceSpec:
        """A budget from a latency target: ``max_ops = ops_per_second * max_ms / 1000``.

        ``ops_per_second`` must come from a probe run **on the target device** for the student kind
        you deploy (:func:`measure_ops_per_second` measures it for a representative student) --
        throughput differs by orders of magnitude across devices and student kinds, so there is no
        portable built-in constant.
        """
        if max_ms <= 0 or ops_per_second <= 0:
            raise ValueError("max_ms and ops_per_second must be positive")
        return cls(max_bytes=max_bytes, max_ops=int(ops_per_second * max_ms / 1000.0), torch_free=torch_free)

    def violations(self, fp: EdgeFootprint) -> list[float]:
        """Normalized constraint values, feasible when ``<= 0`` (the form constrained BO consumes)."""
        out: list[float] = []
        if self.max_bytes is not None:
            out.append((fp.bytes - self.max_bytes) / float(self.max_bytes))
        if self.max_ops is not None:
            out.append((fp.ops - self.max_ops) / float(self.max_ops))
        return out

    def feasible(self, fp: EdgeFootprint) -> bool:
        """Return whether a measured footprint satisfies device constraints."""
        if self.torch_free and not fp.torch_free:
            return False
        return all(v <= 0.0 for v in self.violations(fp))


def _torch_macs_and_params(module: Any) -> tuple[int, int]:
    """(multiply-accumulates, parameter count) of a torch module's Linear stack."""
    macs = 0
    params = 0
    for m in module.modules():
        if type(m).__name__ == "Linear":
            macs += int(m.in_features) * int(m.out_features)
    for p in module.parameters():
        params += int(p.numel())
    return macs, params


def footprint(student: TaskModel) -> EdgeFootprint:
    """Measure a student's deployment cost. Bytes are real (fp32 weights, int8 arrays for a quantized
    student, or serialized JSON for a structured one); ops are the closed-form per-inference count for
    the student's kind."""
    if student.payload == "torch":
        macs, params = _torch_macs_and_params(student.model)
        return EdgeFootprint(bytes=4 * params, ops=macs, torch_free=False)
    if student.payload == "arrays":
        # quantized numpy student: int8 weight bytes measured from the arrays; inference needs no torch
        return EdgeFootprint(bytes=student.model.nbytes(), ops=student.model.macs(), torch_free=True)
    # pure-mixle payload: measure the actual serialized size
    from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()
    nbytes = len(json.dumps(to_serializable(student.model)).encode())
    adapter = student.adapter
    n_labels = len(getattr(adapter, "labels", [])) or 1
    n_fields = int(getattr(adapter, "label_index", 1))
    n_comp = int(student.meta.get("recipe", {}).get("n_components", 1) or 1)
    # classification scores every label: one factor evaluation per field (+ label), per component.
    ops = n_labels * n_comp * (n_fields + 1)
    return EdgeFootprint(bytes=nbytes, ops=ops, torch_free=True)


def measure_inference_seconds(student: TaskModel, inputs: Sequence[Any], *, repeats: int = 5) -> float:
    """Median measured wall-clock seconds per single-input inference for ``student`` on this host.

    Runs ``student.batch(inputs)`` ``repeats`` times (after one untimed warm-up) and reports the
    median per-item time. Measured, not modeled -- run it on the machine whose latency you care
    about (the deploy device, not the dev laptop) for a number that means anything there.
    """
    import time

    inputs = list(inputs)
    if not inputs:
        raise ValueError("need at least one input to measure")
    student.batch(inputs)  # warm-up: imports, JITs, caches
    times = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        student.batch(inputs)
        times.append((time.perf_counter() - t0) / len(inputs))
    return float(np.median(times))


def measure_ops_per_second(student: TaskModel, inputs: Sequence[Any], *, repeats: int = 5) -> float:
    """Measured throughput (footprint ops / measured second) for this student kind on this host.

    The calibration constant that turns a latency budget into :class:`DeviceSpec` ``max_ops``
    (:meth:`DeviceSpec.for_latency`): probe once per (device, student kind), reuse across searches.
    """
    seconds = measure_inference_seconds(student, inputs, repeats=repeats)
    return float(footprint(student).ops) / max(seconds, 1e-12)


# --- the joint structure + process design space ---------------------------------------------------


@dataclass
class EdgeSpace:
    """One unit-cube design space over *family* (structure) and each family's recipe (process).

    Coordinate 0 selects the family; 1..4 decode family-specifically. ``families`` defaults to what
    the input kind and device allow: hashed-MLP students need torch; structured students need
    fixed-schema records. Decode is deterministic, so a design point is a reproducible recipe.
    """

    families: tuple[str, ...] = ("mlp", "structured")
    # mlp axes
    dim_choices: Sequence[int] = (64, 128, 256, 512)
    hidden_range: tuple[int, int] = (4, 96)
    epochs_range: tuple[int, int] = (40, 320)
    log10_lr_range: tuple[float, float] = (-3.0, -1.0)
    bits_choices: Sequence[int] = (32, 8)  # weight precision: fp32, or int8 (post-training quantized, numpy inference)
    ngram: int = 3
    # structured axes
    components_range: tuple[int, int] = (1, 4)
    bins_range: tuple[int, int] = (2, 8)
    max_its_range: tuple[int, int] = (10, 60)
    min_gain_range: tuple[float, float] = (0.0, 5.0)

    def dims(self) -> int:
        """Return the normalized design-space dimensionality."""
        return 6

    def bounds(self) -> list[tuple[float, float]]:
        """Return normalized design-space bounds for DOE search."""
        return [(0.0, 1.0)] * self.dims()

    def signature(self) -> str:
        """Fingerprint of the space so persisted design knowledge is only reused where it applies."""
        return json.dumps(
            {
                "families": list(self.families),
                "dim": list(self.dim_choices),
                "hidden": self.hidden_range,
                "epochs": self.epochs_range,
                "lr": self.log10_lr_range,
                "bits": list(self.bits_choices),
                "comp": self.components_range,
                "bins": self.bins_range,
                "its": self.max_its_range,
                "gain": self.min_gain_range,
            },
            sort_keys=True,
        )

    @staticmethod
    def _lin(lo: float, hi: float, u: float) -> float:
        return lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)

    def decode(self, point: np.ndarray) -> tuple[str, dict[str, Any]]:
        """Unit-cube point -> ``(family, recipe kwargs)``; the mlp recipe carries a ``bits`` precision."""
        p = np.clip(np.asarray(point, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if p.size != self.dims():
            raise ValueError(f"design point must have {self.dims()} coordinates, got {p.size}")
        fam = self.families[min(len(self.families) - 1, int(p[0] * len(self.families)))]
        if fam == "mlp":
            dim = int(self.dim_choices[min(len(self.dim_choices) - 1, int(p[1] * len(self.dim_choices)))])
            hidden = int(round(self._lin(*self.hidden_range, p[2])))
            epochs = int(round(self._lin(*self.epochs_range, p[3])))
            lr = float(10.0 ** self._lin(*self.log10_lr_range, p[4]))
            bits = int(self.bits_choices[min(len(self.bits_choices) - 1, int(p[5] * len(self.bits_choices)))])
            return "mlp", {"dim": dim, "hidden": [hidden], "epochs": epochs, "lr": lr, "bits": bits}
        if fam == "structured":
            comp = int(round(self._lin(*self.components_range, p[1])))
            bins = int(round(self._lin(*self.bins_range, p[2])))
            its = int(round(self._lin(*self.max_its_range, p[3])))
            gain = float(self._lin(*self.min_gain_range, p[4]))
            return "structured", {"n_components": comp, "n_bins": bins, "max_its": its, "min_gain": gain}
        raise ValueError(f"unknown family {fam!r}")


# --- the model that writes the model ---------------------------------------------------------------


class DesignModel:
    """A probabilistic model of the design space itself: design point -> (quality, budget violations).

    Every evaluated design is a row; GP surrogates fitted on the rows drive :meth:`propose` (the next
    design worth training, by feasibility-weighted expected improvement) and :meth:`predict` (mean,
    sd, and probability-of-fitting-the-device for *untrained* designs). It serializes, so what was
    learned designing students for one task warm-starts the next -- the design knowledge is itself a
    model artifact. (And :func:`distill_designer` compresses it into a student -- models all the way
    down, each level a real artifact.)
    """

    def __init__(self, signature: str, n_constraints: int, n_fingerprint: int = 0) -> None:
        self.signature = signature
        self.n_constraints = int(n_constraints)
        self.n_fingerprint = int(n_fingerprint)  # trailing task-fingerprint coords appended to each row
        self.X: list[list[float]] = []
        self.quality: list[float] = []
        self.violations: list[list[float]] = []
        self.tags: list[dict[str, Any]] = []

    # -- ledger --
    def add(
        self,
        point: Any,
        quality: float,
        violations: Sequence[float],
        *,
        fingerprint: Sequence[float] | None = None,
        **tag: Any,
    ) -> None:
        """Append one evaluated design point and its feasibility metadata."""
        v = [float(t) for t in violations]
        if len(v) != self.n_constraints:
            raise ValueError(f"expected {self.n_constraints} violation values, got {len(v)}")
        fp = [float(t) for t in (fingerprint or ())]
        if len(fp) != self.n_fingerprint:
            raise ValueError(f"expected {self.n_fingerprint} fingerprint values, got {len(fp)}")
        self.X.append([float(t) for t in np.asarray(point, dtype=np.float64).reshape(-1)] + fp)
        self.quality.append(float(quality))
        self.violations.append(v)
        self.tags.append(dict(tag))

    def __len__(self) -> int:
        return len(self.X)

    def _fingerprint_bounds(
        self, bounds: Sequence[tuple[float, float]], fingerprint: Sequence[float] | None
    ) -> list[tuple[float, float]]:
        """Bounds with the task-fingerprint coords pinned (degenerate ``lo == hi`` dims).

        The fingerprint enters the surrogate as ordinary input coordinates -- constant within a task,
        varying across tasks -- so one ledger serves *different* tasks: the GP learns task-similarity
        through the fingerprint distance, and proposals condition on the current task by searching
        only the slice at its fingerprint.
        """
        fp = [float(t) for t in (fingerprint or ())]
        if len(fp) != self.n_fingerprint:
            raise ValueError(f"expected {self.n_fingerprint} fingerprint values, got {len(fp)}")
        # epsilon-wide, not degenerate: the doe samplers require low < high; 1e-9 is invisible to the GP
        return list(bounds) + [(v - 1e-9, v + 1e-9) for v in fp]

    # -- the two model operations --
    def propose(
        self,
        bounds: Sequence[tuple[float, float]],
        *,
        seed: Any = None,
        n_candidates: int = 256,
        prefilter: Callable[[tuple], Any] | None = None,
        max_tries: int = 8,
        fingerprint: Sequence[float] | None = None,
    ) -> np.ndarray:
        """The next design worth training: feasibility-weighted EI over everything seen so far.

        ``prefilter`` closes the designer loop: pass a design judge -- typically the compact student
        from :func:`distill_designer`, called as ``prefilter(point_tuple) -> label`` -- and any
        proposal it labels ``"weak"`` is vetoed and re-drawn (fresh acquisition seed), up to
        ``max_tries``. The distilled design knowledge thus skips known weak designs before a single
        training run is spent; if every retry is vetoed the last proposal is returned anyway (the
        judge advises, the surrogate decides). ``fingerprint`` conditions the proposal on the current
        task (see :meth:`_fingerprint_bounds`); the returned point has design coords only.
        """
        rng = seed if isinstance(seed, RandomState) else RandomState(seed)
        full_bounds = self._fingerprint_bounds(bounds, fingerprint)
        n_design = len(bounds)

        def _one(s: Any) -> np.ndarray:
            if len(self.X) < 2:
                r = s if isinstance(s, RandomState) else RandomState(s)
                return np.array([r.uniform(lo, hi) for lo, hi in bounds])
            if self.n_constraints == 0:
                from mixle.doe import propose_next

                return propose_next(
                    np.asarray(self.X),
                    np.asarray(self.quality),
                    full_bounds,
                    seed=s,
                    maximize=True,
                    n_candidates=n_candidates,
                )[:n_design]
            from mixle.doe import propose_next_constrained

            return propose_next_constrained(
                np.asarray(self.X),
                np.asarray(self.quality),
                np.asarray(self.violations),
                full_bounds,
                seed=s,
                maximize=True,
                n_candidates=n_candidates,
            )[:n_design]

        point = _one(rng)
        if prefilter is None:
            return point
        for _ in range(max(1, int(max_tries)) - 1):
            if prefilter(tuple(float(v) for v in point)) != "weak":
                return point
            point = _one(rng)
        return point

    def predict(self, points: Any, *, fingerprint: Sequence[float] | None = None) -> dict[str, np.ndarray]:
        """For untrained designs: predicted quality (mean, sd) and P(fits the device).

        ``points`` carry design coords only; ``fingerprint`` (required when the ledger is
        fingerprinted) selects which task's slice the prediction conditions on.
        """
        from mixle.doe.constrained import probability_of_feasibility
        from mixle.models.gaussian_process import GaussianProcessRegressor

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        fp = [float(t) for t in (fingerprint or ())]
        if len(fp) != self.n_fingerprint:
            raise ValueError(f"expected {self.n_fingerprint} fingerprint values, got {len(fp)}")
        if fp:
            pts = np.hstack([pts, np.tile(np.asarray(fp), (len(pts), 1))])
        X = np.asarray(self.X)
        y = np.asarray(self.quality)
        if len(self.X) < 2:
            raise ValueError("need at least two evaluated designs to predict")

        def _posterior(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            scale = float(np.std(target)) or 1.0
            gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=scale, noise=0.1 * scale + 1e-6)
            gp.fit(X, target, out=None)
            mean, cov = gp.predict(X, target, pts, return_cov=True)
            sd = np.sqrt(np.clip(np.diag(np.atleast_2d(cov)), 0.0, None))
            return np.asarray(mean, dtype=float).reshape(-1), sd

        mu, sd = _posterior(y)
        if self.n_constraints == 0:
            p_fit = np.ones(len(pts))
        else:
            C = np.asarray(self.violations)
            c_mean = np.empty((len(pts), self.n_constraints))
            c_sd = np.empty_like(c_mean)
            for k in range(self.n_constraints):
                c_mean[:, k], c_sd[:, k] = _posterior(C[:, k])
            p_fit = probability_of_feasibility(c_mean, c_sd)
        return {"mean": mu, "sd": sd, "p_feasible": np.asarray(p_fit, dtype=float).reshape(-1)}

    # -- persistence: design knowledge outlives the search --
    def to_json(self) -> dict[str, Any]:
        """Serialize the design ledger for reuse across search runs."""
        return {
            "signature": self.signature,
            "n_constraints": self.n_constraints,
            "n_fingerprint": self.n_fingerprint,
            "X": self.X,
            "quality": self.quality,
            "violations": self.violations,
            "tags": self.tags,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> DesignModel:
        """Reconstruct a design ledger from serialized JSON data."""
        m = cls(d["signature"], int(d["n_constraints"]), int(d.get("n_fingerprint", 0)))
        m.X = [list(map(float, r)) for r in d["X"]]
        m.quality = [float(v) for v in d["quality"]]
        m.violations = [list(map(float, r)) for r in d["violations"]]
        m.tags = [dict(t) for t in d.get("tags", [])]
        return m


# --- task fingerprints: how one design ledger serves many tasks -------------------------------------

FINGERPRINT_KEYS = ("log10_examples", "n_labels", "n_fields", "frac_categorical", "label_entropy")


def task_fingerprint(data: Sequence[Any], labels: Sequence[Any]) -> list[float]:
    """A fixed small vector describing *which task this is*: the coords cross-task warm start keys on.

    ``(log10 #examples, #labels, #fields, fraction of categorical fields, normalized label entropy)``
    -- low-overhead invariants of the dataset, O(1)-scaled so the design surrogate's default lengthscale
    treats similar tasks as informative neighbors and dissimilar ones as weakly coupled.
    """
    labels = [str(y) for y in labels]
    counts = np.unique(labels, return_counts=True)[1].astype(float)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    first = data[0]
    if isinstance(first, (dict, tuple, list)) and not isinstance(first, str):
        values = list(first.values()) if isinstance(first, dict) else list(first)
        n_fields = len(values)
        frac_cat = float(np.mean([not isinstance(v, (int, float, np.integer, np.floating)) for v in values]))
    else:  # text
        n_fields, frac_cat = 1, 1.0
    return [
        float(np.log10(max(len(data), 1))),
        float(len(p)),
        float(n_fields),
        frac_cat,
        entropy,
    ]


# --- the front door --------------------------------------------------------------------------------


@dataclass
class EdgeDistillResult:
    """Outcome of an edge search: the winning student (with measured footprint), whether it truly fits
    the device, the (bytes, agreement) Pareto front over everything trained, and the updated
    :class:`DesignModel` carrying the accumulated design knowledge."""

    model: TaskModel
    family: str
    recipe: dict[str, Any]
    agreement: float
    footprint: EdgeFootprint
    feasible: bool
    pareto: list[dict[str, Any]]
    design: DesignModel
    trials: list[dict[str, Any]] = field(default_factory=list)


def _is_record_data(items: Sequence[Any]) -> bool:
    first = items[0]
    return isinstance(first, (dict, tuple, list)) and not isinstance(first, str)


def _scaled(recipe: dict[str, Any], family: str, fidelity: float) -> dict[str, Any]:
    """A recipe at reduced training fidelity (the screen pass): fewer epochs / EM iterations."""
    r = dict(recipe)
    if family == "mlp":
        r["epochs"] = max(5, int(round(recipe["epochs"] * fidelity)))
    else:
        r["max_its"] = max(3, int(round(recipe["max_its"] * fidelity)))
    return r


def distill_for_edge(
    teacher: Callable[..., Any] | None,
    train_data: Sequence[Any],
    val_data: Sequence[Any],
    device: DeviceSpec,
    *,
    labels: Sequence[str] | None = None,
    train_labels: Sequence[Any] | None = None,
    val_labels: Sequence[Any] | None = None,
    space: EdgeSpace | None = None,
    design: DesignModel | None = None,
    designer: Callable[[tuple], Any] | None = None,
    n_init: int = 4,
    n_iter: int = 6,
    screen_fidelity: float = 0.3,
    promote: int = 2,
    seed: int = 0,
    task: str = "",
) -> EdgeDistillResult:
    """Search structure x training-process for the best student that fits ``device``.

    The teacher labels ``train_data``/``val_data`` once (cached) -- or pass ``train_labels``/
    ``val_labels`` when the labels already exist (a harvested dataset, an upstream ``solve`` split)
    and the teacher is then never called (it may be ``None``). Candidates proposed by the
    :class:`DesignModel` are trained at ``screen_fidelity`` (reduced cost), scored by held-out agreement,
    and measured (:func:`footprint`); the top ``promote`` feasible screens are re-trained at full
    fidelity and the best feasible one wins (ties -> smaller). Pass a previous search's ``design``
    (same space + device shape) to warm-start: the surrogate already knows which regions blow the
    budget. Pass ``designer`` (the compact judge from :func:`distill_designer`) to veto known-weak
    proposals before any training is spent. If nothing fits the device, the least-infeasible student
    is returned with ``feasible=False`` -- inspect ``result.pareto`` for the real trade-off frontier.
    """
    train_data = list(train_data)
    val_data = list(val_data)
    records = _is_record_data(train_data)
    space = space or EdgeSpace(families=("mlp", "structured") if records else ("mlp",))
    if device.torch_free:
        # quantized MLPs run on numpy alone, so 'mlp' survives a torch-free device iff a quantized
        # precision is available -- and is then pinned to it (fp32 students would need torch).
        quantized = tuple(b for b in space.bits_choices if b != 32)
        if quantized:
            space.bits_choices = quantized
        else:
            fams = tuple(f for f in space.families if f != "mlp")
            if not fams:
                raise ValueError(
                    "device.torch_free excludes every family in the space "
                    "(fp32 MLP students need torch; add a quantized bits choice or the structured family)"
                )
            space.families = fams

    rng = RandomState(seed)
    n_constraints = len(device.violations(EdgeFootprint(0, 0, True)))
    if design is None:
        design = DesignModel(space.signature(), n_constraints, n_fingerprint=len(FINGERPRINT_KEYS))
    elif (
        design.signature != space.signature()
        or design.n_constraints != n_constraints
        or design.n_fingerprint != len(FINGERPRINT_KEYS)
    ):
        raise ValueError("design model was built for a different space/device shape; start a fresh one")

    # one teacher pass per dataset -- the search itself never re-queries the teacher; precomputed
    # labels (train_labels=/val_labels=) skip the teacher entirely
    if (train_labels is None) != (val_labels is None):
        raise ValueError("pass both train_labels and val_labels, or neither")
    if train_labels is not None:
        if len(train_labels) != len(train_data) or len(val_labels) != len(val_data):
            raise ValueError("precomputed labels must match their data lengths")
        train_y = [str(t) for t in train_labels]
        val_truth = [str(t) for t in val_labels]
    else:
        if teacher is None:
            raise ValueError("teacher may only be None when train_labels/val_labels are provided")
        train_y = [str(t) for t in _as_batched(teacher)(train_data)]
        val_truth = [str(t) for t in _as_batched(teacher)(val_data)]
    label_list = list(labels) if labels is not None else sorted(set(train_y) | set(val_truth))
    fingerprint = task_fingerprint(train_data, train_y)  # conditions the shared ledger on THIS task

    def _train(family: str, recipe: dict[str, Any]) -> TaskModel:
        if family == "structured":
            return distill_structured_from_labels(
                train_data, train_y, labels=label_list, seed=seed, task=task, **recipe
            )
        r = dict(recipe)
        bits = int(r.pop("bits", 32))
        if records:
            student = distill_records_from_labels(train_data, train_y, labels=label_list, seed=seed, task=task, **r)
        else:
            student = distill_from_labels(
                train_data, train_y, labels=label_list, n=space.ngram, seed=seed, task=task, **r
            )
        if bits != 32:
            from mixle.task.quantize import quantize_mlp

            student = quantize_mlp(student, bits=bits)  # scored quantized: measured bytes and fidelity
        return student

    trials: list[dict[str, Any]] = []

    def _evaluate(point: np.ndarray, fidelity: str) -> dict[str, Any]:
        family, recipe = space.decode(point)
        run = _scaled(recipe, family, screen_fidelity) if fidelity == "screen" else recipe
        student = _train(family, run)
        agree = agreement(student, val_truth, val_data)
        fp = footprint(student)
        viol = device.violations(fp)
        hard_ok = fp.torch_free or not device.torch_free
        trial = {
            "point": np.asarray(point, dtype=float).tolist(),
            "family": family,
            "recipe": run,
            "agreement": agree,
            "bytes": fp.bytes,
            "ops": fp.ops,
            "feasible": device.feasible(fp) and hard_ok,
            "fidelity": fidelity,
            "model": student,
            "footprint": fp,
        }
        trials.append(trial)
        design.add(
            point, agree, viol, fingerprint=fingerprint, family=family, fidelity=fidelity, feasible=trial["feasible"]
        )
        return trial

    # -- screen: LHS seed (skipped when warm knowledge already covers it) + surrogate-driven proposals
    from mixle.doe import latin_hypercube

    n_seed = n_init if len(design) == 0 else max(1, n_init // 2)
    for row in latin_hypercube(space.bounds(), n_seed, rng):
        _evaluate(np.asarray(row), "screen")
    for _ in range(n_iter):
        _evaluate(design.propose(space.bounds(), seed=rng, prefilter=designer, fingerprint=fingerprint), "screen")

    # -- promote: full-fidelity re-train of the best feasible screens (fall back to least-infeasible)
    screens = [t for t in trials if t["fidelity"] == "screen"]
    ranked = sorted(screens, key=lambda t: (not t["feasible"], -t["agreement"], t["bytes"]))
    seen: set[str] = set()
    finalists: list[dict[str, Any]] = []
    for t in ranked:
        key = json.dumps({"f": t["family"], "r": t["recipe"]}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            finalists.append(t)
        if len(finalists) >= max(1, promote):
            break
    fulls = [_evaluate(np.asarray(t["point"]), "full") for t in finalists]

    best = min(fulls, key=lambda t: (not t["feasible"], -t["agreement"], t["bytes"]))

    # -- Pareto front over (bytes, agreement), all fidelities, dominated points dropped
    from mixle.doe import pareto_mask

    pts = np.array([[t["bytes"], -t["agreement"]] for t in trials], dtype=float)
    front = [
        {k: t[k] for k in ("family", "recipe", "agreement", "bytes", "ops", "feasible", "fidelity")}
        for t, keep in zip(trials, pareto_mask(pts))
        if keep
    ]
    front.sort(key=lambda d: d["bytes"])

    return EdgeDistillResult(
        model=best["model"],
        family=best["family"],
        recipe=best["recipe"],
        agreement=best["agreement"],
        footprint=best["footprint"],
        feasible=bool(best["feasible"]),
        pareto=front,
        design=design,
        trials=[{k: v for k, v in t.items() if k not in ("model", "footprint")} for t in trials],
    )


# --- the tower's next level ------------------------------------------------------------------------


def distill_designer(
    design: DesignModel,
    *,
    quality_quantile: float = 0.5,
    seed: int = 0,
) -> TaskModel:
    """Distill the design ledger into a compact torch-free student that judges designs: point -> good/weak.

    The :class:`DesignModel`'s rows are records (the design coordinates); a row is labeled ``good``
    when it was feasible on the device *and* its quality reached the ledger's ``quality_quantile``.
    :func:`distill_structured_from_labels` -- the same machinery the design model tunes -- compresses
    that knowledge into a kilobyte Bayesian-network student usable as a zero-cost pre-filter for
    future searches. Teacher trains student; a model designs the students; the designer is distilled
    into a student: each level of the tower is a real artifact.
    """
    if len(design) < 8:
        raise ValueError("need at least 8 evaluated designs to distill a designer")
    cut = float(np.quantile(np.asarray(design.quality), quality_quantile))
    records = [tuple(row) for row in design.X]
    labels = [
        "good" if (all(v <= 0 for v in viol) and q >= cut) else "weak"
        for q, viol in zip(design.quality, design.violations)
    ]
    if len(set(labels)) < 2:
        raise ValueError("design ledger has only one outcome class; nothing to learn yet")
    return distill_structured_from_labels(
        records, labels, labels=["good", "weak"], seed=seed, task="edge design pre-filter"
    )
