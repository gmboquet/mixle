"""``solve_regression`` for calibrated replacement of numeric task functions.

The regression shape of the solve loop. ``teacher(x) -> float`` is the scorer/pricer/estimator being
replaced; a small student learns it, and calibration is split conformal for regression: on a held-out
slice the absolute-residual quantile ``qhat`` is set so the interval ``[yhat - qhat, yhat + qhat]`` covers
the teacher's answer with probability ``>= 1 - alpha`` (finite-sample, distribution-free, exchangeable
inputs). The escalate-or-answer rule becomes a *precision* rule: answer locally only when the guaranteed
interval is tight enough for the caller's purpose (``qhat <= tol``); otherwise run the real code. So a
locally answered value is covered by the calibrated interval; if the student cannot achieve that
precision, every request escalates.

    def price(item): ...                                   # the rigid pricing routine
    sol = solve_regression(price, items, tol=5.0)          # dataset <- price(i); train; calibrate
    sol(item)                                              # a float: local (±tol guaranteed) or teacher
    sol.interval(item)                                     # (yhat, lo, hi) with 1 - alpha coverage
    sol.improve(); sol.report()                            # the same compounding loop

``qhat`` is one global width, as in standard split conformal regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, HashedRecord
from mixle.task.solve import _input_kind, _label_with


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
        cols: dict[str, list[float]] = {}
        for r in records:
            for k, v in self._items(r):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
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
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
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
    cal_inputs: list = field(default_factory=list)
    cal_ys: list = field(default_factory=list)
    hidden: tuple = (64,)
    epochs: int = 300
    lr: float = 1e-2
    seed: int = 0
    n_requests: int = 0
    n_escalated: int = 0
    harvested_inputs: list = field(default_factory=list)
    harvested_ys: list = field(default_factory=list)

    def _predict(self, xs: list) -> np.ndarray:
        import torch

        feats = np.asarray(self.featurizer.transform(list(xs)), dtype=np.float32)
        with torch.no_grad():
            out = self.net(torch.as_tensor(feats)).numpy()[:, 0]
        return out * self.y_scale + self.y_mean

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
        y = float(_label_with(self.teacher, [x])[0])
        self.harvested_inputs.append(x)
        self.harvested_ys.append(y)
        return y

    def report(self) -> dict[str, Any]:
        """Return calibration, precision, request, and harvest metrics."""
        return {
            "answers_locally": self.answers_locally,
            "qhat": round(float(self.qhat), 6),
            "tol": self.tol,
            "alpha": self.alpha,
            "holdout_mae": round(self.holdout_mae, 6),
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
        """Re-fit with harvested pairs; promote only if the calibrated width shrinks (anti-regression)."""
        if not self.harvested_inputs:
            return False
        if not self.cal_inputs:
            raise RuntimeError(
                "this RegressionSolution was loaded from an artifact and has no calibration data; "
                "collect the harvested pairs and re-solve_regression() to improve."
            )
        inputs = self.train_inputs + list(self.harvested_inputs)
        ys = self.train_ys + [float(v) for v in self.harvested_ys]
        cand = _fit_scaled(inputs, ys, self.featurizer, self.hidden, self.epochs, self.lr, self.seed)
        qhat, mae = _calibrate(cand, self.featurizer, self.cal_inputs, self.cal_ys, self.alpha)
        if not np.isfinite(qhat) or qhat > self.qhat + 1e-12:
            return False
        self.net, (self.y_mean, self.y_scale) = cand[0], cand[1]
        self.qhat, self.holdout_mae = float(qhat), float(mae)
        self.train_inputs, self.train_ys = inputs, ys
        self.harvested_inputs.clear()
        self.harvested_ys.clear()
        return True


def _fit_scaled(inputs: list, ys: list, featurizer: Any, hidden, epochs, lr, seed):
    y = np.asarray(ys, dtype=np.float64)
    mean, scale = float(y.mean()), float(y.std() or 1.0)
    feats = np.asarray(featurizer.transform(list(inputs)), dtype=np.float32)
    net = _fit_reg_mlp(feats, ((y - mean) / scale).astype(np.float32), hidden, epochs, lr, seed)
    return net, (mean, scale)


def _calibrate(cand, featurizer, cal_inputs, cal_ys, alpha) -> tuple[float, float]:
    import torch

    net, (mean, scale) = cand
    feats = np.asarray(featurizer.transform(list(cal_inputs)), dtype=np.float32)
    with torch.no_grad():
        pred = net(torch.as_tensor(feats)).numpy()[:, 0] * scale + mean
    resid = np.abs(np.asarray(cal_ys, dtype=np.float64) - pred)
    n = len(resid)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    qhat = float(np.sort(resid)[min(rank, n) - 1]) if rank <= n else float("inf")
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
) -> RegressionSolution:
    """Replace a numeric routine with a conformally-calibrated student (see module docstring).

    Args:
        teacher: the numeric routine (``teacher(x) -> float``); labels the dataset, remains the fallback.
        inputs: example inputs (text or dict/tuple records).
        tol: the caller's precision requirement — answer locally only when the calibrated ``qhat <= tol``.
        alpha: interval miscoverage level (``1 - alpha`` coverage of the teacher's answer).
        prelabeled: already-teacher-labeled ``(inputs, values)`` — typically harvested escalations from
            a serving deployment — folded into the TRAINING split only, never calibration (which stays
            a fresh split of ``inputs``, so ``qhat`` keeps its finite-sample guarantee). The re-solve
            half of the serving loop.
    """
    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_regression needs at least 12 example inputs")
    k = kind or _input_kind(items[0])
    ys = [float(v) for v in _label_with(teacher, items)]

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_cal = max(4, int(round(len(items) * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_inputs = [items[i] for i in train_idx]
    train_ys = [ys[i] for i in train_idx]
    cal_inputs = [items[i] for i in cal_idx]
    cal_ys = [ys[i] for i in cal_idx]

    if prelabeled is not None:
        pre_in, pre_ys = prelabeled
        if len(pre_in) != len(pre_ys):
            raise ValueError("prelabeled inputs and values must have equal length")
        train_inputs = train_inputs + list(pre_in)
        train_ys = train_ys + [float(v) for v in pre_ys]

    featurizer = (
        HashedNGram(n=3, dim=dim, seed=seed)
        if k == "text"
        else RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_inputs)
    )
    cand = _fit_scaled(train_inputs, train_ys, featurizer, tuple(hidden), int(epochs), float(lr), int(seed))
    qhat, mae = _calibrate(cand, featurizer, cal_inputs, cal_ys, float(alpha))

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
        hidden=tuple(hidden),
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
    )
