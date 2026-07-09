"""Callable task-model wrapper for serialized local models.

The artifact contract (:mod:`mixle.task.artifact`) makes a model durable.
``TaskModel`` makes it directly usable by pairing a fitted model with an I/O
adapter that converts raw application inputs into model features and converts
model outputs into application results. The adapter is serialized in the
artifact manifest, so ``TaskModel.load(path)`` reconstructs the full
``raw_input -> result`` callable in a fresh process.

Adapters self-describe and rebuild through a registry
(``register_adapter`` / ``IOAdapter.from_spec``). The built-in
:class:`TextClassifierIO` supports the distillation path with a dependency-free
hashed character n-gram featurizer, a small classifier, and a stored label map.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.task import artifact as _artifact

# --- featurizer: dependency-free hashed character n-grams ----------------------------------------------------


class HashedNGram:
    """Map a string to a fixed-width float vector by hashing its character n-grams into ``dim`` buckets.

    The featurizer is deterministic and dependency-free. It serializes as three
    scalar settings and rebuilds without a fitted vocabulary or external
    tokenizer. Counts are L2-normalized per row.
    """

    def __init__(self, n: int = 3, dim: int = 256, seed: int = 0) -> None:
        self.n = int(n)
        self.dim = int(dim)
        self.seed = int(seed)

    def _bucket(self, gram: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{gram}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized hashed n-gram feature rows for ``texts``."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f" {t} "
            for j in range(max(len(s) - self.n + 1, 0)):
                out[i, self._bucket(s[j : j + self.n])] += 1.0
            if len(s) < self.n:  # very short input: hash the whole thing
                out[i, self._bucket(s)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)

    def to_spec(self) -> dict[str, Any]:
        """Return the serializable featurizer configuration."""
        return {"n": self.n, "dim": self.dim, "seed": self.seed}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> HashedNGram:
        """Rebuild a featurizer from :meth:`to_spec` output."""
        return cls(n=spec["n"], dim=spec["dim"], seed=spec["seed"])


class HashedRecord:
    """Map a heterogeneous record to a fixed-width hashed feature vector.

    Each tuple position or dictionary key owns a hashed namespace. Categorical,
    string, and boolean values contribute an indicator feature; numeric values
    contribute a bounded value feature and a presence feature. The transform is
    stateless and deterministic, so it serializes as two scalar settings and
    rebuilds without a fitted encoder or vocabulary.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = int(dim)
        self.seed = int(seed)

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def _items(self, record: Any) -> list[tuple[str, Any]]:
        if isinstance(record, dict):
            return [(str(k), v) for k, v in record.items()]
        if isinstance(record, (list, tuple)):
            return [(str(i), v) for i, v in enumerate(record)]
        return [("0", record)]  # a bare scalar/string record

    def transform(self, records: list[Any]) -> np.ndarray:
        """Return L2-normalized hashed feature rows for heterogeneous records."""
        out = np.zeros((len(records), self.dim), dtype=np.float32)
        for i, record in enumerate(records):
            for key, value in self._items(record):
                if isinstance(value, bool) or value is None or isinstance(value, str):
                    out[i, self._bucket(f"{key}={value}")] += 1.0
                elif isinstance(value, (int, float)):
                    out[i, self._bucket(f"num:{key}")] += float(np.tanh(float(value)))
                    out[i, self._bucket(f"has:{key}")] += 1.0
                else:
                    out[i, self._bucket(f"{key}={value!r}")] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)

    def to_spec(self) -> dict[str, Any]:
        """Return the serializable record-featurizer configuration."""
        return {"dim": self.dim, "seed": self.seed}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> HashedRecord:
        """Rebuild a record featurizer from :meth:`to_spec` output."""
        return cls(dim=spec["dim"], seed=spec["seed"])


# --- I/O adapters: raw <-> model, self-describing -----------------------------------------------------------

_ADAPTERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_adapter(kind: str, from_spec: Callable[[dict[str, Any]], Any]) -> None:
    """Register an adapter's ``from_spec`` factory under ``kind`` so a saved ``io`` block can rebuild it."""
    existing = _ADAPTERS.get(kind)
    if existing is not None and existing is not from_spec:
        raise ValueError(f"adapter {kind!r} already registered to a different factory")
    _ADAPTERS[kind] = from_spec


def adapter_from_spec(spec: dict[str, Any]) -> Any:
    """Rebuild an adapter from its ``io`` spec (the ``kind`` field selects the factory)."""
    kind = spec.get("kind")
    if kind not in _ADAPTERS:
        _register_builtin_adapters()
    if kind not in _ADAPTERS:
        raise KeyError(f"no adapter registered as {kind!r}")
    return _ADAPTERS[kind](spec)


def _register_builtin_adapters() -> None:
    if "text_classifier" not in _ADAPTERS:
        register_adapter("text_classifier", TextClassifierIO.from_spec)
    if "record_classifier" not in _ADAPTERS:
        register_adapter("record_classifier", RecordClassifierIO.from_spec)
    if "structured_classifier" not in _ADAPTERS:
        register_adapter("structured_classifier", StructuredClassifierIO.from_spec)
    if "extraction" not in _ADAPTERS:
        from mixle.task.extract import ExtractionIO

        register_adapter("extraction", ExtractionIO.from_spec)


class _ClassifierIO:
    """Shared ``raw -> label`` plumbing: featurize, run the module, argmax/softmax over a stored label list.

    Subclasses set ``kind`` and the featurizer type; the module-running logic (logits/proba/predict) is common,
    so conformal calibration, density gating, and the cascade work identically for text and record classifiers.
    """

    kind = "classifier"
    _featurizer_cls: type = HashedNGram

    def __init__(self, featurizer: Any, labels: list[str]) -> None:
        self.featurizer = featurizer
        self.labels = list(labels)

    def features(self, raw_inputs: list[Any]) -> np.ndarray:
        return self.featurizer.transform(raw_inputs)

    def logits_batch(self, module: Any, raw_inputs: list[Any]) -> np.ndarray:
        import torch

        if not raw_inputs:  # empty batch: (0, K) with no featurize/forward (reshape can't infer -1 at size 0)
            return np.empty((0, len(self.labels)), dtype=np.float32)
        feats = self.features(raw_inputs)
        module.eval()
        with torch.no_grad():
            out = module(torch.from_numpy(feats)).cpu().numpy()
        return np.asarray(out).reshape(len(raw_inputs), -1)

    def proba_batch(self, module: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Row-stochastic class scores ``(m, K)`` (softmax of the logits) -- the conformal nonconformity input.

        These sum to 1 but are *not* a describable random process; conformal calibration is what turns them
        into a coverage guarantee (see :mod:`mixle.task.calibrate`).
        """
        z = self.logits_batch(module, raw_inputs)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_batch(self, module: Any, raw_inputs: list[Any]) -> list[str]:
        """Predict the most likely label for each raw input."""
        idx = self.logits_batch(module, raw_inputs).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict(self, module: Any, raw_input: Any) -> str:
        """Predict the most likely label for one raw input."""
        return self.predict_batch(module, [raw_input])[0]

    def to_spec(self) -> dict[str, Any]:
        """Return a serializable adapter specification for artifacts."""
        return {"kind": self.kind, "featurizer": self.featurizer.to_spec(), "labels": self.labels}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Any:
        """Rebuild an adapter from its artifact ``io`` specification."""
        return cls(cls._featurizer_cls.from_spec(spec["featurizer"]), spec["labels"])


class TextClassifierIO(_ClassifierIO):
    """``str -> label``: hashed character n-gram features into a small classifier."""

    kind = "text_classifier"
    _featurizer_cls = HashedNGram

    def __init__(self, featurizer: HashedNGram, labels: list[str]) -> None:
        super().__init__(featurizer, labels)


class RecordClassifierIO(_ClassifierIO):
    """``record -> label``: hashed-record features into a small classifier (tuples/dicts of mixed fields)."""

    kind = "record_classifier"
    _featurizer_cls = HashedRecord

    def __init__(self, featurizer: HashedRecord, labels: list[str]) -> None:
        super().__init__(featurizer, labels)


class StructuredClassifierIO:
    """``record -> label`` through a *structured probabilistic* model instead of a neural net.

    The model is a fitted joint over ``(field_1, ..., field_m, label)`` -- a :class:`DependencyTreeDistribution`
    (or mixture) discovered by :func:`mixle.inference.structure.learn_structure`. Classification is the generative
    rule ``argmax_label P(features, label)``: score each candidate label and pick the best. Because
    ``softmax_label log P(features, label) = P(label | features)`` *exactly* (the feature evidence is a shared
    constant across labels), :meth:`proba_batch` returns the true posterior -- not a softmax over arbitrary logits
    -- so conformal calibration (:mod:`mixle.task.calibrate`) and the density gate operate on a real probability.

    The student is interpretable (``model.edges()`` shows the discovered dependencies), kilobytes on disk, and
    round-trips through the json artifact path. It assumes a *fixed schema*: every record exposes the same fields
    (``field_keys`` for dicts, positional for tuples) -- the variable set a Bayesian network is defined over.
    """

    kind = "structured_classifier"

    def __init__(self, field_keys: list[str] | None, label_index: int, labels: list[str]) -> None:
        self.field_keys = list(field_keys) if field_keys is not None else None  # None => positional tuple records
        self.label_index = int(label_index)
        self.labels = list(labels)

    def _values(self, record: Any) -> tuple:
        """The non-label field values of a raw record, in the canonical order the model was fit on."""
        if self.field_keys is not None:
            if not isinstance(record, dict):
                raise TypeError(f"structured classifier expects dict records with keys {self.field_keys}")
            return tuple(record.get(k) for k in self.field_keys)
        if isinstance(record, (list, tuple)):
            return tuple(record)
        return (record,)

    def _augment(self, values: tuple, label: str) -> tuple:
        """Splice ``label`` into the field position it occupied at fit time, giving a full joint record."""
        return values[: self.label_index] + (label,) + values[self.label_index :]

    def logits_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Per-label log-joint ``log P(features, label)`` as an ``(m, K)`` score matrix (the classifier logits)."""
        if not raw_inputs:  # empty batch: (0, K), skip encoding (an empty seq_encode need not be supported)
            return np.empty((0, len(self.labels)), dtype=np.float64)
        values = [self._values(r) for r in raw_inputs]
        out = np.full((len(values), len(self.labels)), -np.inf, dtype=np.float64)
        for k, label in enumerate(self.labels):
            rows = [self._augment(v, label) for v in values]
            try:
                out[:, k] = np.asarray(model.seq_log_density(model.dist_to_encoder().seq_encode(rows)))
            except Exception:  # unseen conditioning value in some row: fall back to per-row scoring
                for i, row in enumerate(rows):
                    try:
                        out[i, k] = float(model.log_density(row))
                    except Exception:
                        out[i, k] = -np.inf
        return out

    def proba_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """The exact posterior ``P(label | features)`` -- softmax of the per-label log-joints (shared evidence cancels)."""
        z = self.logits_batch(model, raw_inputs)
        z = np.where(np.isneginf(z).all(axis=1, keepdims=True), 0.0, z)  # all-(-inf) row -> uniform, avoid nan
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_batch(self, model: Any, raw_inputs: list[Any]) -> list[str]:
        """Predict labels for raw inputs by maximizing the per-label joint score."""
        idx = self.logits_batch(model, raw_inputs).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict(self, model: Any, raw_input: Any) -> str:
        """Predict the label for one raw input."""
        return self.predict_batch(model, [raw_input])[0]

    def to_spec(self) -> dict[str, Any]:
        """Return the serializable structured-classifier adapter specification."""
        return {
            "kind": self.kind,
            "field_keys": self.field_keys,
            "label_index": self.label_index,
            "labels": self.labels,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> StructuredClassifierIO:
        """Rebuild a structured-classifier adapter from its artifact ``io`` specification."""
        return cls(spec.get("field_keys"), spec["label_index"], spec["labels"])


# --- the task model: a callable raw -> result, durable through the artifact ----------------------------------


class TaskModel:
    """A fitted small model plus its I/O adapter, callable as ``task(raw) -> result`` and saveable to a directory."""

    def __init__(
        self,
        model: Any,
        adapter: Any,
        *,
        builder: str | None = None,
        config: dict[str, Any] | None = None,
        payload: str = "torch",
        task: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.builder = builder
        self.config = dict(config or {})
        self.payload = payload
        self.task = task
        self.meta = dict(meta or {})

    def __call__(self, raw_input: Any) -> Any:
        """Run the wrapped model on one raw input through its adapter."""
        return self.adapter.predict(self.model, raw_input)

    def batch(self, raw_inputs: list[Any]) -> list[Any]:
        """Run the wrapped model on a batch of raw inputs through its adapter."""
        if hasattr(self.adapter, "predict_batch"):
            return self.adapter.predict_batch(self.model, raw_inputs)
        return [self.adapter.predict(self.model, x) for x in raw_inputs]

    def save(self, path: str) -> str:
        """Persist as a task artifact: the model payload plus the adapter's ``io`` spec and metadata."""
        io = self.adapter.to_spec()
        if self.payload == "torch":
            if self.builder is None:
                raise ValueError("a torch TaskModel needs builder= to be reconstructable")
            return _artifact.save_module(
                path, self.model, self.builder, self.config, task=self.task, io=io, meta=self.meta
            )
        if self.payload == "arrays":
            if self.builder is None:
                raise ValueError("an arrays TaskModel needs builder= to be reconstructable")
            return _artifact.save_arrays(
                path, self.model.to_arrays(), self.builder, self.config, task=self.task, io=io, meta=self.meta
            )
        return _artifact.save_json(path, self.model, task=self.task, io=io, meta=self.meta)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> TaskModel:
        """Rebuild a TaskModel (model + adapter) from a saved artifact directory."""
        manifest = _artifact.read_manifest(path)
        adapter = adapter_from_spec(manifest.io)
        if manifest.payload == "torch":
            model, _ = _artifact.load_module(path, device=device)
            return cls(
                model,
                adapter,
                builder=manifest.builder,
                config=manifest.config,
                payload="torch",
                task=manifest.task,
                meta=manifest.meta,
            )
        if manifest.payload == "arrays":
            model, _ = _artifact.load_arrays(path)
            return cls(
                model,
                adapter,
                builder=manifest.builder,
                config=manifest.config,
                payload="arrays",
                task=manifest.task,
                meta=manifest.meta,
            )
        model, _ = _artifact.load_json(path)
        return cls(model, adapter, payload="json", task=manifest.task, meta=manifest.meta)
