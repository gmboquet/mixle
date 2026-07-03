"""Edge distillation: jointly optimize the student's *structure* and its *training process* under a
hard device budget -- driven by a model that designs models.

:func:`mixle.task.tune_recipe` tunes one family's knobs with a soft cost penalty. An edge deployment
is a different problem: the budget is **hard** (bytes of flash, ops per inference, "no torch on the
device"), and the biggest wins come from choosing the right *kind* of student -- a hashed-feature MLP
versus a structured probabilistic classifier (a learned Bayesian network: kilobytes, torch-free,
exactly calibrated posteriors) -- not from tweaking widths inside one kind.

This module makes both choices searchable, and makes the search itself a model:

* :class:`DeviceSpec` / :class:`EdgeFootprint` -- the deployment budget and a student's **measured**
  cost (serialized bytes, per-inference op count, torch dependence).
* :class:`EdgeSpace` -- one design space spanning *family* (structure) and each family's training
  recipe (process), decoded from a unit cube so any optimizer can drive it.
* :class:`DesignModel` -- the model that writes the model: GP surrogates over design -> (quality,
  budget violations), proposing the next design by feasibility-weighted expected improvement
  (:func:`mixle.doe.propose_next_constrained`). It persists (``to_json``/``from_json``), so design
  knowledge **accumulates across tasks** -- a warm-started search skips the designs that never work.
* :func:`distill_for_edge` -- the front door: screen candidates at reduced fidelity, promote the
  promising ones to full training, return the best student that *fits the device*, with the Pareto
  front over (bytes, agreement) and the updated :class:`DesignModel`.
* :func:`distill_designer` -- the recursion made useful: distill the accumulated design ledger into
  a tiny torch-free structured student that predicts whether a design is worth training -- the
  design model, compressed by the very machinery it steers.

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

    def violations(self, fp: EdgeFootprint) -> list[float]:
        """Normalized constraint values, feasible when ``<= 0`` (the form constrained BO consumes)."""
        out: list[float] = []
        if self.max_bytes is not None:
            out.append((fp.bytes - self.max_bytes) / float(self.max_bytes))
        if self.max_ops is not None:
            out.append((fp.ops - self.max_ops) / float(self.max_ops))
        return out

    def feasible(self, fp: EdgeFootprint) -> bool:
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
    """Measure a student's deployment cost. Bytes are real (fp32 weights, or the serialized JSON for a
    torch-free student); ops are the closed-form per-inference count for the student's kind."""
    if student.payload == "torch":
        macs, params = _torch_macs_and_params(student.model)
        return EdgeFootprint(bytes=4 * params, ops=macs, torch_free=False)
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
    ngram: int = 3
    # structured axes
    components_range: tuple[int, int] = (1, 4)
    bins_range: tuple[int, int] = (2, 8)
    max_its_range: tuple[int, int] = (10, 60)
    min_gain_range: tuple[float, float] = (0.0, 5.0)

    def dims(self) -> int:
        return 5

    def bounds(self) -> list[tuple[float, float]]:
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
        """Unit-cube point -> ``(family, recipe kwargs)`` for that family's ``distill_*`` entry."""
        p = np.clip(np.asarray(point, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if p.size != self.dims():
            raise ValueError(f"design point must have {self.dims()} coordinates, got {p.size}")
        fam = self.families[min(len(self.families) - 1, int(p[0] * len(self.families)))]
        if fam == "mlp":
            dim = int(self.dim_choices[min(len(self.dim_choices) - 1, int(p[1] * len(self.dim_choices)))])
            hidden = int(round(self._lin(*self.hidden_range, p[2])))
            epochs = int(round(self._lin(*self.epochs_range, p[3])))
            lr = float(10.0 ** self._lin(*self.log10_lr_range, p[4]))
            return "mlp", {"dim": dim, "hidden": [hidden], "epochs": epochs, "lr": lr}
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

    def __init__(self, signature: str, n_constraints: int) -> None:
        self.signature = signature
        self.n_constraints = int(n_constraints)
        self.X: list[list[float]] = []
        self.quality: list[float] = []
        self.violations: list[list[float]] = []
        self.tags: list[dict[str, Any]] = []

    # -- ledger --
    def add(self, point: Any, quality: float, violations: Sequence[float], **tag: Any) -> None:
        v = [float(t) for t in violations]
        if len(v) != self.n_constraints:
            raise ValueError(f"expected {self.n_constraints} violation values, got {len(v)}")
        self.X.append([float(t) for t in np.asarray(point, dtype=np.float64).reshape(-1)])
        self.quality.append(float(quality))
        self.violations.append(v)
        self.tags.append(dict(tag))

    def __len__(self) -> int:
        return len(self.X)

    # -- the two model operations --
    def propose(
        self, bounds: Sequence[tuple[float, float]], *, seed: Any = None, n_candidates: int = 256
    ) -> np.ndarray:
        """The next design worth training: feasibility-weighted EI over everything seen so far."""
        if len(self.X) < 2:
            rng = seed if isinstance(seed, RandomState) else RandomState(seed)
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        if self.n_constraints == 0:
            from mixle.doe import propose_next

            return propose_next(
                np.asarray(self.X),
                np.asarray(self.quality),
                bounds,
                seed=seed,
                maximize=True,
                n_candidates=n_candidates,
            )
        from mixle.doe import propose_next_constrained

        return propose_next_constrained(
            np.asarray(self.X),
            np.asarray(self.quality),
            np.asarray(self.violations),
            bounds,
            seed=seed,
            maximize=True,
            n_candidates=n_candidates,
        )

    def predict(self, points: Any) -> dict[str, np.ndarray]:
        """For untrained designs: predicted quality (mean, sd) and P(fits the device)."""
        from mixle.doe.constrained import probability_of_feasibility
        from mixle.models.gaussian_process import GaussianProcessRegressor

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
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
        return {
            "signature": self.signature,
            "n_constraints": self.n_constraints,
            "X": self.X,
            "quality": self.quality,
            "violations": self.violations,
            "tags": self.tags,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> DesignModel:
        m = cls(d["signature"], int(d["n_constraints"]))
        m.X = [list(map(float, r)) for r in d["X"]]
        m.quality = [float(v) for v in d["quality"]]
        m.violations = [list(map(float, r)) for r in d["violations"]]
        m.tags = [dict(t) for t in d.get("tags", [])]
        return m


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
    teacher: Callable[..., Any],
    train_data: Sequence[Any],
    val_data: Sequence[Any],
    device: DeviceSpec,
    *,
    labels: Sequence[str] | None = None,
    space: EdgeSpace | None = None,
    design: DesignModel | None = None,
    n_init: int = 4,
    n_iter: int = 6,
    screen_fidelity: float = 0.3,
    promote: int = 2,
    seed: int = 0,
    task: str = "",
) -> EdgeDistillResult:
    """Search structure x training-process for the best student that fits ``device``.

    The teacher labels ``train_data``/``val_data`` once (cached). Candidates proposed by the
    :class:`DesignModel` are trained at ``screen_fidelity`` (cheap), scored by held-out agreement,
    and measured (:func:`footprint`); the top ``promote`` feasible screens are re-trained at full
    fidelity and the best feasible one wins (ties -> smaller). Pass a previous search's ``design``
    (same space + device shape) to warm-start: the surrogate already knows which regions blow the
    budget. If nothing fits the device, the least-infeasible student is returned with
    ``feasible=False`` -- inspect ``result.pareto`` for the real trade-off frontier.
    """
    train_data = list(train_data)
    val_data = list(val_data)
    records = _is_record_data(train_data)
    space = space or EdgeSpace(families=("mlp", "structured") if records else ("mlp",))
    if device.torch_free:
        fams = tuple(f for f in space.families if f != "mlp")
        if not fams:
            raise ValueError("device.torch_free excludes every family in the space (MLP students need torch)")
        space.families = fams

    rng = RandomState(seed)
    n_constraints = len(device.violations(EdgeFootprint(0, 0, True)))
    if design is None:
        design = DesignModel(space.signature(), n_constraints)
    elif design.signature != space.signature() or design.n_constraints != n_constraints:
        raise ValueError("design model was built for a different space/device shape; start a fresh one")

    # one teacher pass per dataset -- the search itself never re-queries the teacher
    train_labels = [str(t) for t in _as_batched(teacher)(train_data)]
    val_truth = [str(t) for t in _as_batched(teacher)(val_data)]
    label_list = list(labels) if labels is not None else sorted(set(train_labels) | set(val_truth))

    def _train(family: str, recipe: dict[str, Any]) -> TaskModel:
        if family == "structured":
            return distill_structured_from_labels(
                train_data, train_labels, labels=label_list, seed=seed, task=task, **recipe
            )
        if records:
            return distill_records_from_labels(
                train_data, train_labels, labels=label_list, seed=seed, task=task, **recipe
            )
        return distill_from_labels(
            train_data, train_labels, labels=label_list, n=space.ngram, seed=seed, task=task, **recipe
        )

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
        design.add(point, agree, viol, family=family, fidelity=fidelity, feasible=trial["feasible"])
        return trial

    # -- screen: LHS seed (skipped when warm knowledge already covers it) + surrogate-driven proposals
    from mixle.doe import latin_hypercube

    n_seed = n_init if len(design) == 0 else max(1, n_init // 2)
    for row in latin_hypercube(space.bounds(), n_seed, rng):
        _evaluate(np.asarray(row), "screen")
    for _ in range(n_iter):
        _evaluate(design.propose(space.bounds(), seed=rng), "screen")

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
    """Distill the design ledger into a tiny torch-free student that judges designs: point -> good/weak.

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
