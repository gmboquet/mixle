"""``solve_regression`` for calibrated replacement of numeric task functions.

The regression shape of the solve loop. ``teacher(x) -> float`` is the scorer/pricer/estimator being
replaced; a small student learns it, and calibration is split conformal for regression: on held-out
CALIBRATION rows the absolute-residual quantile ``qhat`` is set so the interval
``[yhat - qhat, yhat + qhat]`` covers the teacher's answer with probability ``>= 1 - alpha``
(finite-sample, distribution-free, exchangeable inputs). The escalate-or-answer rule is a *precision*
rule: answer locally only when the interval is tight enough for the caller's purpose (``qhat <= tol``);
otherwise run the real code. If the student cannot achieve the required precision, every request
escalates.

**What the coverage statement is -- and what deployment does to it.** The ``1 - alpha`` statement is
marginal over the calibration draw AND the query jointly. Split conformal's randomness includes the
calibration sample, so conditioning on the deployment decision -- the gate event ``qhat <= tol`` is a
function of the calibration draw -- is selection, and coverage conditional on deploying is NOT
guaranteed to be ``1 - alpha``. Exact counterexample (external review pass eleven): iid Uniform(0,1)
residuals, ``n=19``, ``alpha=0.1``, ``tol=0.8`` -- the gate opens with probability 0.083, and coverage
conditional on opening is 0.751, not 0.90. The effect shrinks as the calibration slice grows and as
``tol`` moves away from ``qhat``'s sampling distribution, but it is not zero and this module does not
certify it. What the deployed artifact carries instead is a MEASUREMENT: ``report()`` includes the
empirical coverage of the current interval on the SELECTION rows, a slice the calibration quantile
never touches. Both the guarantee and the measurement assume exchangeability and fail silently under
distribution shift -- re-measure on drifted traffic.

**Two immutable holdout roles.** The held-out rows are split, deterministically at solve time, into
CALIBRATION rows (they set ``qhat`` and are never read by a promotion decision) and SELECTION rows
(they decide every ``improve()`` promotion and produce the reported error/coverage measurements).
Without the split, ``improve()`` chose the student with the same rows that then set its interval, and
repeated promotion took a running minimum over ``qhat`` draws -- expected coverage in the reviewer's
exact construction fell from 0.90 to 0.75 after twenty candidate selections. ``selection_uses`` counts
how many promotion decisions the selection rows have made: after the first, the selection-side numbers
are selection scores, not fresh held-out estimates (the same honesty rule as classification ``solve``).

    def price(item): ...                                   # the rigid pricing routine
    sol = solve_regression(price, items, tol=5.0)          # dataset <- price(i); train; calibrate
    sol(item)                                              # a float: local (width <= tol) or teacher
    sol.interval(item)                                     # (yhat, lo, hi); marginal 1 - alpha coverage,
                                                           #   unconditional on the deployment gate
    sol.improve(); sol.report()                            # promote on selection rows; recalibrate on
                                                           #   untouched calibration rows

``qhat`` is one global width, as in standard split conformal regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task._teacher import TeacherCaller, as_batch_view
from mixle.task.model import HashedNGram, HashedRecord
from mixle.task.solve import _input_kind, _split_holdout_roles


class RecordRegressionFeaturizer:
    """Record featurizer for regression tasks.

    ``HashedRecord`` (built for classification) squashes numerics through ``tanh``, which saturates and
    erases the magnitude signal a regressor needs. Here numeric keys (learned from the fit sample) map to
    dedicated standardized columns; everything else uses hashing."""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.num_keys: list[str] = []
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self._hash = HashedRecord(dim=dim, seed=seed)

    @staticmethod
    def _items(record: Any) -> list[tuple[str, Any]]:
        if isinstance(record, dict):
            return [(str(k), v) for k, v in record.items()]
        if isinstance(record, (list, tuple)):
            return [(str(i), v) for i, v in enumerate(record)]
        return [("0", record)]

    def fit(self, records: list[Any]) -> RecordRegressionFeaturizer:
        """Learn numeric-key normalization statistics from sample records."""
        if not records:
            raise ValueError("regression featurizer needs at least one record")
        cols: dict[str, list[float]] = {}
        for r in records:
            for k, v in self._items(r):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not np.isfinite(v):
                        raise ValueError(f"numeric feature {k!r} must be finite")
                    cols.setdefault(k, []).append(float(v))
        self.num_keys = sorted(cols)
        for k in self.num_keys:
            arr = np.asarray(cols[k], dtype=np.float64)
            self.num_mean[k] = float(arr.mean())
            self.num_std[k] = float(arr.std() or 1.0)
        return self

    def transform(self, records: list[Any]) -> np.ndarray:
        """Transform records into standardized numeric columns plus hashed categorical features."""
        cat_rows = []
        num = np.zeros((len(records), len(self.num_keys)), dtype=np.float32)
        for i, r in enumerate(records):
            cat = {}
            for k, v in self._items(r):
                if k in self.num_mean and isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not np.isfinite(v):
                        raise ValueError(f"numeric feature {k!r} must be finite")
                    num[i, self.num_keys.index(k)] = (float(v) - self.num_mean[k]) / self.num_std[k]
                else:
                    cat[k] = v
            cat_rows.append(cat)
        hashed = np.asarray(self._hash.transform(cat_rows), dtype=np.float32)
        return np.concatenate([num, hashed], axis=1)

    def to_spec(self) -> dict[str, Any]:
        """Serialize numeric-key statistics and hashing settings."""
        return {
            "dim": self.dim,
            "seed": self.seed,
            "num_keys": list(self.num_keys),
            "num_mean": dict(self.num_mean),
            "num_std": dict(self.num_std),
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> RecordRegressionFeaturizer:
        """Reconstruct a regression featurizer from an artifact spec."""
        f = cls(dim=int(spec["dim"]), seed=int(spec["seed"]))
        f.num_keys = list(spec["num_keys"])
        f.num_mean = {k: float(v) for k, v in spec["num_mean"].items()}
        f.num_std = {k: float(v) for k, v in spec["num_std"].items()}
        return f


def featurizer_spec(f: Any) -> dict[str, Any]:
    """Tagged, artifact-ready spec for the featurizers the solve shapes use."""
    kind = "record_regression" if isinstance(f, RecordRegressionFeaturizer) else "ngram"
    return {"kind": kind, **f.to_spec()}


def featurizer_from_spec(spec: dict[str, Any]) -> Any:
    """Reconstruct a task featurizer from a tagged artifact spec."""
    body = {k: v for k, v in spec.items() if k != "kind"}
    if spec.get("kind") == "record_regression":
        return RecordRegressionFeaturizer.from_spec(body)
    return HashedNGram.from_spec(body)


def _fit_reg_mlp(x: np.ndarray, y: np.ndarray, hidden: Sequence[int], epochs: int, lr: float, seed: int):
    import torch

    torch.manual_seed(seed)
    dims = [x.shape[1], *hidden, 1]
    # Build every layer at an explicit float32, independent of torch's process-global default
    # dtype. Leaving `dtype` unset makes `Linear` follow `torch.get_default_dtype()`, and callers
    # (mixle_pde in particular sets `torch.set_default_dtype(torch.float64)` throughout its own PDE
    # code, both in production paths and dozens of test files) routinely leave that global at
    # float64 -- `xt`/`yt` below are already explicitly float32, so an ambient float64 default only
    # poisons the net's own weights, and `net(xt)` then dies with "mat1 and mat2 must have the same
    # dtype, but got Float and Double". Pinning the layer dtype here makes this function's contract
    # (always trains and returns a float32 module) hold regardless of any caller's ambient state.
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float32))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y[:, None], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def _network_prediction(net: Any, feats: np.ndarray) -> np.ndarray:
    """Run a regression network on the network's own device and return validated CPU values."""
    import torch

    params = list(net.parameters())
    if not params:
        raise ValueError("regression network has no parameters")
    tensor = torch.as_tensor(feats, dtype=params[0].dtype, device=params[0].device)
    with torch.no_grad():
        raw = net(tensor).detach().to(device="cpu", dtype=torch.float64).numpy()
    if raw.shape != (len(feats), 1) or not np.all(np.isfinite(raw)):
        raise ValueError(f"regression network must return finite shape {(len(feats), 1)}, got {raw.shape}")
    return raw[:, 0]


def _validated_features(featurizer: Any, inputs: Sequence[Any]) -> np.ndarray:
    feats = np.asarray(featurizer.transform(list(inputs)), dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] != len(inputs) or feats.shape[1] == 0 or not np.all(np.isfinite(feats)):
        raise ValueError("regression features must be a finite, nonempty matrix with one row per input")
    return feats


def _validated_finite_values(values: Sequence[Any], *, name: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be nonempty and contain only finite values")
    return result


@dataclass
class RegressionSolution:
    """A calibrated numeric student in front of the routine it replaces."""

    net: Any
    featurizer: Any
    teacher: Callable[..., Any]
    qhat: float
    alpha: float
    tol: float
    holdout_mae: float
    y_mean: float
    y_scale: float
    train_inputs: list = field(default_factory=list)
    train_ys: list = field(default_factory=list)
    # ``cal_*`` sets the conformal ``qhat`` and is never read by a promotion decision; ``sel_*``
    # decides every improve() promotion and produces the reported error/coverage measurements.
    # ``selection_uses`` records how many promotions the selection rows have decided (see module
    # docstring): past one, the selection-side numbers are selection scores, not held-out estimates.
    cal_inputs: list = field(default_factory=list)
    cal_ys: list = field(default_factory=list)
    sel_inputs: list = field(default_factory=list)
    sel_ys: list = field(default_factory=list)
    selection_uses: int = 0
    hidden: tuple = (64,)
    epochs: int = 300
    lr: float = 1e-2
    seed: int = 0
    n_requests: int = 0
    n_escalated: int = 0
    harvested_inputs: list = field(default_factory=list)
    harvested_ys: list = field(default_factory=list)

    def _predict(self, xs: list) -> np.ndarray:
        feats = _validated_features(self.featurizer, xs)
        out = _network_prediction(self.net, feats) * self.y_scale + self.y_mean
        if not np.all(np.isfinite(out)):
            raise ValueError("regression prediction is non-finite after restoring target scale")
        return out

    def _teacher_call(self) -> TeacherCaller:
        """The teacher view escalation routes through, resolved once and kept.

        Escalation runs the teacher on every escalated request, so rediscovering whether it is
        per-item or batched per call would cost an extra invocation per request forever.
        """
        caller = self.__dict__.get("_teacher_caller")
        if caller is None:
            caller = as_batch_view(self.teacher)
            self.__dict__["_teacher_caller"] = caller
        return caller

    def interval(self, x: Any) -> tuple[float, float, float]:
        """Return ``(yhat, lo, hi)`` with calibrated teacher-answer coverage."""
        yhat = float(self._predict([x])[0])
        return yhat, yhat - self.qhat, yhat + self.qhat

    @property
    def answers_locally(self) -> bool:
        """Whether the calibrated precision meets the tolerance at all (else everything escalates)."""
        return bool(np.isfinite(self.qhat) and self.qhat <= self.tol)

    def decide(self, x: Any) -> float | None:
        """Return the calibrated point estimate when local precision is sufficient.

        If ``answers_locally`` is false, return ``None`` to signal escalation. Unlike ``__call__``, this
        method never falls through to the teacher itself, so a :class:`~mixle.task.router.Router` tier
        can decide whether to escalate to the next tier.
        """
        if self.answers_locally:
            return float(self._predict([x])[0])
        return None

    def __call__(self, x: Any) -> float:
        self.n_requests += 1
        if self.answers_locally:
            return float(self._predict([x])[0])
        self.n_escalated += 1
        y = float(self._teacher_call().one(x))
        self.harvested_inputs.append(x)
        self.harvested_ys.append(y)
        return y

    def selection_coverage(self) -> float | None:
        """Measured coverage of the CURRENT interval on the selection rows, or None when absent.

        The selection rows never touch the calibration quantile, so for a freshly solved model this
        is an unbiased per-artifact measurement of the deployed interval's coverage -- the honest
        substitute for the deployment-conditional guarantee split conformal does not provide (see
        the module docstring). After ``improve()`` promotions the same rows have also chosen the
        model; ``selection_uses`` records that the number is then a selection score.
        """
        if not self.sel_inputs or not np.isfinite(self.qhat):
            return None
        predictions = self._predict(list(self.sel_inputs))
        residuals = np.abs(np.asarray(self.sel_ys, dtype=np.float64) - predictions)
        return float(np.mean(residuals <= self.qhat))

    def report(self) -> dict[str, Any]:
        """Return calibration, precision, measurement, request, and harvest metrics."""
        coverage = self.selection_coverage()
        return {
            "answers_locally": self.answers_locally,
            "qhat": round(float(self.qhat), 6),
            "tol": self.tol,
            "alpha": self.alpha,
            "holdout_mae": round(self.holdout_mae, 6),
            # measured on the selection rows, not guaranteed: deployment conditions on the
            # calibration draw, which split conformal's marginal statement does not cover
            "selection_coverage": None if coverage is None else round(coverage, 6),
            "selection_uses": self.selection_uses,
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "harvested": len(self.harvested_ys),
        }

    def save(self, path: str) -> str:
        """Persist the network, featurizer, and calibration metadata."""
        from mixle.task.artifact import save_module

        first = next(m for m in self.net.modules() if hasattr(m, "in_features"))
        return save_module(
            path,
            self.net,
            "mixle.mlp",
            {
                "input_dim": int(first.in_features),
                "hidden_dims": [int(h) for h in self.hidden],
                "output_dim": 1,
                "activation": "relu",
            },
            task="solve_regression student",
            io=featurizer_spec(self.featurizer),
            meta={
                "regress": {
                    "qhat": float(self.qhat),
                    "alpha": self.alpha,
                    "tol": self.tol,
                    "holdout_mae": self.holdout_mae,
                    "y_mean": self.y_mean,
                    "y_scale": self.y_scale,
                    "hidden": [int(h) for h in self.hidden],
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "seed": self.seed,
                }
            },
        )

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, device: str = "cpu") -> RegressionSolution:
        """Reconstitute a serving RegressionSolution (no training/calibration data; improve() raises)."""
        from mixle.task.artifact import load_module

        net, manifest = load_module(path, device=device)
        m = manifest.meta["regress"]
        return cls(
            net=net,
            featurizer=featurizer_from_spec(manifest.io),
            teacher=teacher,
            qhat=float(m["qhat"]),
            alpha=float(m["alpha"]),
            tol=float(m["tol"]),
            holdout_mae=float(m["holdout_mae"]),
            y_mean=float(m["y_mean"]),
            y_scale=float(m["y_scale"]),
            hidden=tuple(m["hidden"]),
            epochs=int(m["epochs"]),
            lr=float(m["lr"]),
            seed=int(m["seed"]),
        )

    def improve(self) -> bool:
        """Re-fit with harvested pairs; promote on SELECTION evidence, recalibrate on untouched rows.

        The promote/reject decision reads only the selection rows, and the deployed ``qhat`` is
        then computed on the calibration rows, which no promotion decision ever reads -- so the
        promoted model is not chosen with the same data that certifies its interval. The previous
        rule promoted whenever the CALIBRATED width shrank, which took a running minimum of
        ``qhat`` over candidates on one fixed slice; in the reviewer's exact exchangeable
        construction that selection drove expected coverage from 0.90 to 0.75 after twenty
        candidates (STAT-RR11-1). A consequence of honest recalibration: the deployed ``qhat``
        may occasionally grow across a promotion, because it is no longer the selection metric.
        ``selection_uses`` records each decision the selection rows make.
        """
        if not self.harvested_inputs:
            return False
        if not self.cal_inputs or not self.sel_inputs:
            raise RuntimeError(
                "this RegressionSolution was loaded from an artifact and has no calibration/selection "
                "data; collect the harvested pairs and re-solve_regression() to improve."
            )
        inputs = self.train_inputs + list(self.harvested_inputs)
        ys = self.train_ys + [float(v) for v in self.harvested_ys]
        cand = _fit_scaled(inputs, ys, self.featurizer, self.hidden, self.epochs, self.lr, self.seed)
        incumbent_error = _selection_error(
            (self.net, (self.y_mean, self.y_scale)), self.featurizer, self.sel_inputs, self.sel_ys
        )
        candidate_error = _selection_error(cand, self.featurizer, self.sel_inputs, self.sel_ys)
        self.selection_uses += 1
        if not np.isfinite(candidate_error) or candidate_error > incumbent_error + 1e-12:
            return False
        qhat, _ = _calibrate(cand, self.featurizer, self.cal_inputs, self.cal_ys, self.alpha)
        if not np.isfinite(qhat):
            return False
        self.net, (self.y_mean, self.y_scale) = cand[0], cand[1]
        self.qhat, self.holdout_mae = float(qhat), float(candidate_error)
        self.train_inputs, self.train_ys = inputs, ys
        self.harvested_inputs.clear()
        self.harvested_ys.clear()
        return True


def _fit_scaled(inputs: list, ys: list, featurizer: Any, hidden, epochs, lr, seed):
    if len(inputs) != len(ys):
        raise ValueError("regression inputs and targets must have the same length")
    y = np.asarray(_validated_finite_values(ys, name="regression targets"), dtype=np.float64)
    mean, scale = float(y.mean()), float(y.std() or 1.0)
    feats = _validated_features(featurizer, inputs)
    net = _fit_reg_mlp(feats, ((y - mean) / scale).astype(np.float32), hidden, epochs, lr, seed)
    return net, (mean, scale)


def _selection_error(cand, featurizer, sel_inputs, sel_ys) -> float:
    """Mean absolute error on the SELECTION rows -- the only evidence a promotion may read."""
    targets = np.asarray(_validated_finite_values(sel_ys, name="selection targets"), dtype=np.float64)
    net, (mean, scale) = cand
    feats = _validated_features(featurizer, sel_inputs)
    predictions = _network_prediction(net, feats) * scale + mean
    residuals = np.abs(targets - predictions)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("selection residuals must be finite")
    return float(residuals.mean())


def _calibrate(cand, featurizer, cal_inputs, cal_ys, alpha) -> tuple[float, float]:
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if len(cal_inputs) != len(cal_ys):
        raise ValueError("calibration inputs and targets must have the same length")
    targets = np.asarray(_validated_finite_values(cal_ys, name="calibration targets"), dtype=np.float64)
    net, (mean, scale) = cand
    feats = _validated_features(featurizer, cal_inputs)
    pred = _network_prediction(net, feats) * scale + mean
    resid = np.abs(targets - pred)
    if not np.all(np.isfinite(resid)):
        raise ValueError("calibration residuals must be finite")
    n = len(resid)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank < 1 or rank > n:
        raise ValueError(
            f"{n} calibration examples are insufficient for finite {1.0 - alpha:.6g} coverage; "
            f"need at least {int(np.ceil(1.0 / alpha)) - 1}"
        )
    qhat = float(np.sort(resid)[rank - 1])
    return qhat, float(resid.mean())


def solve_regression(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    tol: float,
    alpha: float = 0.1,
    holdout: float = 0.25,
    kind: str | None = None,
    hidden: Sequence[int] = (64,),
    epochs: int = 300,
    lr: float = 1e-2,
    dim: int = 256,
    prelabeled: tuple[Sequence[Any], Sequence[float]] | None = None,
    seed: int = 0,
    teacher_mode: str = "auto",
) -> RegressionSolution:
    """Replace a numeric routine with a conformally-calibrated student (see module docstring).

    Args:
        teacher: the numeric routine (``teacher(x) -> float``); labels the dataset, remains the fallback.
        inputs: example inputs (text or dict/tuple records).
        tol: the caller's precision requirement — answer locally only when the calibrated ``qhat <= tol``.
            NOTE: the coverage statement is marginal over the calibration draw and the query jointly;
            conditioning on this deployment gate is selection over calibration draws and is NOT
            covered by the guarantee — see the module docstring, and read ``report()``'s measured
            ``selection_coverage`` for the deployed artifact.
        alpha: interval miscoverage level (``1 - alpha`` marginal coverage of the teacher's answer).
        holdout: fraction reserved and split into two IMMUTABLE roles at solve time — conformal
            CALIBRATION rows (the larger half; set ``qhat``, never read by promotions) and SELECTION
            rows (decide ``improve()`` promotions, produce the reported measurements).
        prelabeled: already-teacher-labeled ``(inputs, values)`` — typically harvested escalations from
            a serving deployment — folded into the TRAINING split only, never calibration (which stays
            a fresh split of ``inputs``, so ``qhat`` keeps its finite-sample guarantee). The re-solve
            half of the serving loop.
    """
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not np.isfinite(holdout) or not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be finite and in (0, 1)")
    if kind not in (None, "text", "record"):
        raise ValueError("kind must be None, 'text', or 'record'")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    hidden = tuple(hidden)
    if any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in hidden):
        raise ValueError("hidden widths must be positive integers")

    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_regression needs at least 12 example inputs")
    k = kind or _input_kind(items[0])
    # one view for the whole call: the per-item/batched convention is resolved once, not per pass
    call = as_batch_view(teacher, teacher_mode)
    ys = _validated_finite_values(call(items), name="teacher targets")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_holdout = max(4, int(round(len(items) * holdout)))
    if n_holdout >= len(items):
        raise ValueError("holdout leaves no regression training examples")
    # Two IMMUTABLE holdout roles, split deterministically the moment the holdout is drawn
    # (same rule as classification solve): CALIBRATION rows set qhat and are never read by a
    # promotion decision; SELECTION rows decide improve() promotions and produce the reported
    # error/coverage measurements. Conformal takes the larger half because its quantile is the
    # part with a coverage claim attached.
    n_conf = _split_holdout_roles(n_holdout)
    min_cal = int(np.ceil(1.0 / alpha)) - 1
    if n_conf < min_cal:
        raise ValueError(
            f"holdout yields {n_conf} conformal-calibration examples after the selection split "
            f"({n_holdout} held-out rows; calibration takes the larger half), but alpha={alpha} "
            f"requires at least {min_cal}; provide more examples or raise holdout"
        )
    cal_idx, sel_idx, train_idx = order[:n_conf], order[n_conf:n_holdout], order[n_holdout:]
    train_inputs = [items[i] for i in train_idx]
    train_ys = [ys[i] for i in train_idx]
    cal_inputs = [items[i] for i in cal_idx]
    cal_ys = [ys[i] for i in cal_idx]
    sel_inputs = [items[i] for i in sel_idx]
    sel_ys = [ys[i] for i in sel_idx]

    if prelabeled is not None:
        pre_in, pre_ys = prelabeled
        if len(pre_in) != len(pre_ys):
            raise ValueError("prelabeled inputs and values must have equal length")
        train_inputs = train_inputs + list(pre_in)
        train_ys = train_ys + _validated_finite_values(pre_ys, name="prelabeled targets")

    featurizer = (
        HashedNGram(n=3, dim=dim, seed=seed)
        if k == "text"
        else RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_inputs)
    )
    cand = _fit_scaled(train_inputs, train_ys, featurizer, hidden, epochs, float(lr), seed)
    qhat, _ = _calibrate(cand, featurizer, cal_inputs, cal_ys, float(alpha))
    mae = _selection_error(cand, featurizer, sel_inputs, sel_ys)

    return RegressionSolution(
        net=cand[0],
        featurizer=featurizer,
        teacher=teacher,
        qhat=float(qhat),
        alpha=float(alpha),
        tol=float(tol),
        holdout_mae=float(mae),
        y_mean=cand[1][0],
        y_scale=cand[1][1],
        train_inputs=train_inputs,
        train_ys=train_ys,
        cal_inputs=cal_inputs,
        cal_ys=cal_ys,
        sel_inputs=sel_inputs,
        sel_ys=sel_ys,
        hidden=tuple(hidden),
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
    )
