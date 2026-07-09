"""``solve_multilabel`` -- replace rigid code that returns a set of labels, with per-label honesty.

The multi-label shape of the solve loop: ``teacher(x) -> list[str]`` (tags, flags, categories -- any
subset of a label universe). The student is one shared-feature net with a sigmoid head per label; the
calibration is per-label conformal thresholds from the held-out slice:

  * ``A_l`` -- the ``1 - alpha`` quantile of label ``l``'s scores among calibration inputs where ``l``
    is absent: a score above it is confidently-present (at most ``~alpha`` of absents score that high);
  * ``P_l`` -- the ``alpha`` quantile among inputs where ``l`` is present: a score below it is
    confidently-absent.

A label is *decided* when its score clears one of those bars; anything in between is ambiguous. The
input is answered locally only when every label is decided -- one ambiguous tag escalates the whole
request to the teacher (whose answer is harvested), so a locally-returned set never contains a guess.
Labels with too few calibration examples on either side are never decided locally (their bars are
``inf``/``-inf``): under-calibrated is treated as ambiguous, not as confident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram
from mixle.task.regress import RecordRegressionFeaturizer, featurizer_from_spec, featurizer_spec
from mixle.task.solve import _input_kind, _label_with


def _fit_multilabel_mlp(x: np.ndarray, y: np.ndarray, hidden: Sequence[int], epochs: int, lr: float, seed: int):
    import torch

    torch.manual_seed(seed)
    dims = [x.shape[1], *hidden, y.shape[1]]
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def _quantile_upper(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample upper bar: the ceil((n+1)(1-alpha)) order statistic; ``inf`` when n is too small."""
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass
class MultiLabelSolution:
    """A per-label-calibrated tagger in front of the routine it replaces."""

    net: Any
    featurizer: Any
    labels: list[str]
    teacher: Callable[..., Any]
    upper_absent: np.ndarray  # A_l: above -> confidently present
    lower_present: np.ndarray  # P_l: below -> confidently absent
    alpha: float
    holdout_set_agreement: float
    train_inputs: list = field(default_factory=list)
    train_sets: list = field(default_factory=list)
    cal_inputs: list = field(default_factory=list)
    cal_sets: list = field(default_factory=list)
    hidden: tuple = (64,)
    epochs: int = 300
    lr: float = 1e-2
    seed: int = 0
    n_requests: int = 0
    n_escalated: int = 0
    harvested_inputs: list = field(default_factory=list)
    harvested_sets: list = field(default_factory=list)

    def _scores(self, xs: list) -> np.ndarray:
        import torch

        feats = np.asarray(self.featurizer.transform(list(xs)), dtype=np.float32)
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.as_tensor(feats))).numpy()

    def try_local(self, x: Any) -> list[str] | None:
        """The decided label set, or ``None`` when any label is ambiguous (= must escalate)."""
        s = self._scores([x])[0]
        present = s > self.upper_absent
        absent = s < self.lower_present
        if bool(np.all(present | absent)):
            return [lab for lab, p in zip(self.labels, present) if p]
        return None

    def decide(self, x: Any) -> list[str] | None:
        """Return the local multilabel decision, or ``None`` when the example should escalate."""
        return self.try_local(x)

    def __call__(self, x: Any) -> list[str]:
        self.n_requests += 1
        local = self.try_local(x)
        if local is not None:
            return local
        self.n_escalated += 1
        got = _label_with(self.teacher, [x])[0]
        tags = [str(v) for v in (got or [])]
        self.harvested_inputs.append(x)
        self.harvested_sets.append(tags)
        return tags

    def report(self) -> dict[str, Any]:
        """Return multi-label agreement, escalation, and harvest metrics."""
        return {
            "labels": len(self.labels),
            "holdout_set_agreement": round(self.holdout_set_agreement, 4),
            "alpha": self.alpha,
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested": len(self.harvested_sets),
        }

    def save(self, path: str) -> str:
        """Persist net + featurizer + per-label bars; :meth:`load` restores a serving tagger."""
        from mixle.task.artifact import save_module

        first = next(m for m in self.net.modules() if hasattr(m, "in_features"))
        return save_module(
            path,
            self.net,
            "mixle.mlp",
            {
                "input_dim": int(first.in_features),
                "hidden_dims": [int(h) for h in self.hidden],
                "output_dim": len(self.labels),
                "activation": "relu",
            },
            task="solve_multilabel student",
            io=featurizer_spec(self.featurizer),
            meta={
                "multilabel": {
                    "labels": list(self.labels),
                    "upper_absent": [float(v) for v in self.upper_absent],
                    "lower_present": [float(v) for v in self.lower_present],
                    "alpha": self.alpha,
                    "holdout_set_agreement": self.holdout_set_agreement,
                    "hidden": [int(h) for h in self.hidden],
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "seed": self.seed,
                }
            },
        )

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, device: str = "cpu") -> MultiLabelSolution:
        """Reconstitute a serving MultiLabelSolution (no training/calibration data; improve() raises)."""
        from mixle.task.artifact import load_module

        net, manifest = load_module(path, device=device)
        m = manifest.meta["multilabel"]
        return cls(
            net=net,
            featurizer=featurizer_from_spec(manifest.io),
            labels=list(m["labels"]),
            teacher=teacher,
            upper_absent=np.asarray(m["upper_absent"], dtype=np.float64),
            lower_present=np.asarray(m["lower_present"], dtype=np.float64),
            alpha=float(m["alpha"]),
            holdout_set_agreement=float(m["holdout_set_agreement"]),
            hidden=tuple(m["hidden"]),
            epochs=int(m["epochs"]),
            lr=float(m["lr"]),
            seed=int(m["seed"]),
        )

    def improve(self) -> bool:
        """Re-fit with harvested sets; promote only if held-out set agreement does not regress."""
        if not self.harvested_inputs:
            return False
        if not self.cal_inputs:
            raise RuntimeError(
                "this MultiLabelSolution was loaded from an artifact and has no calibration data; "
                "collect the harvested pairs and re-solve_multilabel() to improve."
            )
        inputs = self.train_inputs + list(self.harvested_inputs)
        sets = self.train_sets + [list(v) for v in self.harvested_sets]
        cand = _fit_and_calibrate(
            inputs,
            sets,
            self.cal_inputs,
            self.cal_sets,
            self.labels,
            self.featurizer,
            self.alpha,
            self.hidden,
            self.epochs,
            self.lr,
            self.seed,
        )
        if cand["agreement"] < self.holdout_set_agreement - 1e-12:
            return False
        self.net = cand["net"]
        self.upper_absent, self.lower_present = cand["upper_absent"], cand["lower_present"]
        self.holdout_set_agreement = float(cand["agreement"])
        self.train_inputs, self.train_sets = inputs, sets
        self.harvested_inputs.clear()
        self.harvested_sets.clear()
        return True


def _fit_and_calibrate(
    train_inputs, train_sets, cal_inputs, cal_sets, labels, featurizer, alpha, hidden, epochs, lr, seed
) -> dict:
    import torch

    idx = {lab: j for j, lab in enumerate(labels)}
    y = np.zeros((len(train_inputs), len(labels)), dtype=np.float32)
    for i, tags in enumerate(train_sets):
        for t in tags:
            if t in idx:
                y[i, idx[t]] = 1.0
    feats = np.asarray(featurizer.transform(list(train_inputs)), dtype=np.float32)
    net = _fit_multilabel_mlp(feats, y, hidden, epochs, lr, seed)

    cal_feats = np.asarray(featurizer.transform(list(cal_inputs)), dtype=np.float32)
    with torch.no_grad():
        s = torch.sigmoid(net(torch.as_tensor(cal_feats))).numpy()
    y_cal = np.zeros_like(s, dtype=bool)
    for i, tags in enumerate(cal_sets):
        for t in tags:
            if t in idx:
                y_cal[i, idx[t]] = True

    upper_absent = np.array([_quantile_upper(s[~y_cal[:, j], j], alpha) for j in range(len(labels))])
    lower_present = np.array([-_quantile_upper(-s[y_cal[:, j], j], alpha) for j in range(len(labels))])

    present = s > upper_absent[None, :]
    absent = s < lower_present[None, :]
    decided = np.all(present | absent, axis=1)
    agree = float(np.mean([bool(d) and bool(np.array_equal(p, t)) for d, p, t in zip(decided, present, y_cal)]))
    return {"net": net, "upper_absent": upper_absent, "lower_present": lower_present, "agreement": agree}


def solve_multilabel(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    alpha: float = 0.1,
    holdout: float = 0.25,
    kind: str | None = None,
    hidden: Sequence[int] = (64,),
    epochs: int = 300,
    lr: float = 1e-2,
    dim: int = 256,
    prelabeled: tuple[Sequence[Any], Sequence[Sequence[str]]] | None = None,
    seed: int = 0,
) -> MultiLabelSolution:
    """Replace a set-of-labels routine with a per-label-calibrated student (see module docstring).

    ``prelabeled`` — already-teacher-labeled ``(inputs, label_sets)``, typically harvested escalations
    from a serving deployment — folds into the TRAINING split only, never calibration (which stays a
    fresh split of ``inputs``, so the per-label bars keep their finite-sample rank guarantee). Labels
    seen only in ``prelabeled`` still enter the label space.
    """
    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_multilabel needs at least 12 example inputs")
    k = kind or _input_kind(items[0])
    raw = _label_with(teacher, items)
    sets = [[str(v) for v in (tags or [])] for tags in raw]
    pre_in: list = []
    pre_sets: list[list[str]] = []
    if prelabeled is not None:
        pre_in = list(prelabeled[0])
        pre_sets = [[str(v) for v in (tags or [])] for tags in prelabeled[1]]
        if len(pre_in) != len(pre_sets):
            raise ValueError("prelabeled inputs and label sets must have equal length")
    labels = sorted({t for tags in sets for t in tags} | {t for tags in pre_sets for t in tags})
    if not labels:
        raise ValueError("the teacher produced no labels on the example inputs")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_cal = max(4, int(round(len(items) * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_inputs = [items[i] for i in train_idx] + pre_in
    train_sets = [sets[i] for i in train_idx] + pre_sets
    cal_inputs = [items[i] for i in cal_idx]
    cal_sets = [sets[i] for i in cal_idx]

    # records: standardized numeric pass-through + hashed categoricals (HashedRecord's tanh squash
    # saturates and erases the magnitude signal threshold-flags like "high-value" depend on)
    featurizer = (
        HashedNGram(n=3, dim=dim, seed=seed)
        if k == "text"
        else RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_inputs)
    )
    cand = _fit_and_calibrate(
        train_inputs,
        train_sets,
        cal_inputs,
        cal_sets,
        labels,
        featurizer,
        float(alpha),
        tuple(hidden),
        int(epochs),
        float(lr),
        int(seed),
    )
    return MultiLabelSolution(
        net=cand["net"],
        featurizer=featurizer,
        labels=labels,
        teacher=teacher,
        upper_absent=cand["upper_absent"],
        lower_present=cand["lower_present"],
        alpha=float(alpha),
        holdout_set_agreement=float(cand["agreement"]),
        train_inputs=train_inputs,
        train_sets=train_sets,
        cal_inputs=cal_inputs,
        cal_sets=cal_sets,
        hidden=tuple(hidden),
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
    )
