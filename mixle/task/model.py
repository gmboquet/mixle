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


class ImpossibleEvidenceError(ValueError):
    """Raised when structured evidence has zero support under every declared label."""

    def __init__(self, row_indices: list[int]) -> None:
        self.row_indices = tuple(row_indices)
        super().__init__(f"evidence has zero support under every label for rows {self.row_indices!r}")


def _exact_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be an exact positive integer.")
    return int(value)


def _exact_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact integer.")
    return int(value)


def _validated_labels(labels: Any) -> list[str]:
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("labels must be a non-empty sequence of unique strings.")
    result = list(labels)
    if any(not isinstance(label, str) or not label for label in result):
        raise ValueError("labels must be a non-empty sequence of unique strings.")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique.")
    return result


# --- featurizer: dependency-free hashed character n-grams ----------------------------------------------------


class HashedNGram:
    """Map a string to a fixed-width float vector by hashing its character n-grams into ``dim`` buckets.

    The featurizer is deterministic and dependency-free. It serializes as three
    scalar settings and rebuilds without a fitted vocabulary or external
    tokenizer. Counts are L2-normalized per row.
    """

    def __init__(self, n: int = 3, dim: int = 256, seed: int = 0) -> None:
        self.n = _exact_positive_int(n, "n")
        self.dim = _exact_positive_int(dim, "dim")
        self.seed = _exact_int(seed, "seed")

    def _bucket(self, gram: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{gram}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized hashed n-gram feature rows for ``texts``."""
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings.")
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

    def __init__(
        self,
        dim: int = 256,
        seed: int = 0,
        *,
        record_kind: str | None = None,
        field_keys: list[str] | None = None,
        record_width: int | None = None,
    ) -> None:
        self.dim = _exact_positive_int(dim, "dim")
        self.seed = _exact_int(seed, "seed")
        if record_kind not in (None, "dict", "sequence", "scalar"):
            raise ValueError("record_kind must be None, 'dict', 'sequence', or 'scalar'.")
        if record_kind == "dict":
            if not isinstance(field_keys, list) or not field_keys:
                raise ValueError("dict record schemas require non-empty field_keys.")
            if any(not isinstance(key, str) or not key for key in field_keys) or len(set(field_keys)) != len(field_keys):
                raise ValueError("field_keys must be unique non-empty strings.")
            if record_width is not None:
                raise ValueError("dict record schemas do not use record_width.")
        elif field_keys is not None:
            raise ValueError("field_keys are only valid for dict record schemas.")
        if record_kind == "sequence":
            record_width = _exact_positive_int(record_width, "record_width")
        elif record_width is not None:
            raise ValueError("record_width is only valid for sequence record schemas.")
        self.record_kind = record_kind
        self.field_keys = tuple(field_keys) if field_keys is not None else None
        self.record_width = record_width

    @classmethod
    def for_records(cls, records: list[Any], *, dim: int = 256, seed: int = 0) -> HashedRecord:
        """Construct a featurizer bound to the exact supplied record schema."""

        if not records:
            raise ValueError("records must be non-empty.")
        first = records[0]
        if isinstance(first, dict):
            keys = sorted(first)
            if not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("dict records require a non-empty schema of string keys.")
            if any(not isinstance(record, dict) or sorted(record) != keys for record in records):
                raise ValueError("all dict records must have the same keys.")
            return cls(dim, seed, record_kind="dict", field_keys=keys)
        if isinstance(first, (list, tuple)):
            width = len(first)
            if width == 0 or any(not isinstance(record, (list, tuple)) or len(record) != width for record in records):
                raise ValueError("all sequence records must have one identical positive width.")
            return cls(dim, seed, record_kind="sequence", record_width=width)
        if any(isinstance(record, (dict, list, tuple)) for record in records):
            raise ValueError("scalar records cannot be mixed with structured records.")
        return cls(dim, seed, record_kind="scalar")

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def _items(self, record: Any) -> list[tuple[str, Any]]:
        if self.record_kind == "dict":
            if not isinstance(record, dict) or set(record) != set(self.field_keys or ()):
                raise ValueError(f"record must have exactly the schema {list(self.field_keys or ())!r}.")
            return [(key, record[key]) for key in self.field_keys or ()]
        if self.record_kind == "sequence":
            if not isinstance(record, (list, tuple)) or len(record) != self.record_width:
                raise ValueError(f"record must be a sequence of width {self.record_width}.")
            return [(str(i), value) for i, value in enumerate(record)]
        if self.record_kind == "scalar" and isinstance(record, (dict, list, tuple)):
            raise ValueError("record must be scalar.")
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
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError(f"record numeric field {key!r} must be finite.")
                    out[i, self._bucket(f"num:{key}")] += float(np.tanh(numeric))
                    out[i, self._bucket(f"has:{key}")] += 1.0
                else:
                    out[i, self._bucket(f"{key}={value!r}")] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)

    def to_spec(self) -> dict[str, Any]:
        """Return the serializable record-featurizer configuration."""
        return {
            "dim": self.dim,
            "seed": self.seed,
            "record_kind": self.record_kind,
            "field_keys": list(self.field_keys) if self.field_keys is not None else None,
            "record_width": self.record_width,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> HashedRecord:
        """Rebuild a record featurizer from :meth:`to_spec` output."""
        return cls(
            dim=spec["dim"],
            seed=spec["seed"],
            record_kind=spec.get("record_kind"),
            field_keys=spec.get("field_keys"),
            record_width=spec.get("record_width"),
        )


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
        self.labels = _validated_labels(labels)

    def features(self, raw_inputs: list[Any]) -> np.ndarray:
        return self.featurizer.transform(raw_inputs)

    def logits_batch(self, module: Any, raw_inputs: list[Any]) -> np.ndarray:
        import torch

        if not raw_inputs:  # empty batch: (0, K) with no featurize/forward (reshape can't infer -1 at size 0)
            return np.empty((0, len(self.labels)), dtype=np.float32)
        feats = np.asarray(self.features(raw_inputs))
        if feats.ndim != 2 or feats.shape[0] != len(raw_inputs) or np.any(~np.isfinite(feats)):
            raise ValueError("classifier features must be a finite two-dimensional row-aligned matrix.")
        parameter = next(iter(module.parameters()), None)
        buffer = next(iter(module.buffers()), None)
        device = parameter.device if parameter is not None else buffer.device if buffer is not None else None
        was_training = getattr(module, "training", None)
        module.eval()
        try:
            with torch.no_grad():
                tensor = torch.as_tensor(feats, device=device)
                out = module(tensor)
                if not isinstance(out, torch.Tensor):
                    raise TypeError("classifier module must return a torch.Tensor.")
                result = out.detach().cpu().numpy()
        finally:
            if isinstance(was_training, bool):
                module.train(was_training)
        if result.shape != (len(raw_inputs), len(self.labels)) or np.any(~np.isfinite(result)):
            raise ValueError(
                f"classifier module must return finite logits with shape "
                f"({len(raw_inputs)}, {len(self.labels)})."
            )
        return np.asarray(result)

    def proba_batch(self, module: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Row-stochastic class scores ``(m, K)`` (softmax of the logits) -- the conformal nonconformity input.

        These sum to 1 but are *not* a describable random process; conformal calibration is what turns them
        into a coverage guarantee (see :mod:`mixle.task.calibrate`).
        """
        # Evaluate the normalization in float64 even when the model emits float32.
        # Downstream conformal validation deliberately checks row-stochastic input
        # tightly; float32 accumulation can miss one by several ulps.
        z = np.asarray(self.logits_batch(module, raw_inputs), dtype=np.float64)
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
        if featurizer.record_kind is None:
            raise ValueError("record classifier featurizers must declare a fixed record schema.")
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

    def __init__(
        self,
        field_keys: list[str] | None,
        label_index: int,
        labels: list[str],
        *,
        field_count: int | None = None,
    ) -> None:
        self.label_index = _exact_int(label_index, "label_index")
        if self.label_index < 0:
            raise ValueError("label_index must be non-negative.")
        self.labels = _validated_labels(labels)
        if field_keys is not None:
            if (
                not isinstance(field_keys, list)
                or not field_keys
                or any(not isinstance(key, str) or not key for key in field_keys)
                or len(set(field_keys)) != len(field_keys)
            ):
                raise ValueError("field_keys must be unique non-empty strings.")
            self.field_keys = list(field_keys)
            self.field_count = len(field_keys)
        else:
            self.field_keys = None
            self.field_count = (
                _exact_positive_int(field_count, "field_count") if field_count is not None else None
            )
        if self.field_count is not None and self.label_index > self.field_count:
            raise ValueError("label_index cannot exceed the number of non-label fields.")

    def _values(self, record: Any) -> tuple:
        """The non-label field values of a raw record, in the canonical order the model was fit on."""
        if self.field_keys is not None:
            if not isinstance(record, dict):
                raise TypeError(f"structured classifier expects dict records with keys {self.field_keys}")
            if set(record) != set(self.field_keys):
                raise ValueError(f"structured classifier expects exactly the keys {self.field_keys}")
            return tuple(record[k] for k in self.field_keys)
        if isinstance(record, (list, tuple)):
            if self.field_count is not None and len(record) != self.field_count:
                raise ValueError(f"structured classifier expects {self.field_count} positional fields.")
            return tuple(record)
        if self.field_count not in (None, 1):
            raise ValueError(f"structured classifier expects {self.field_count} positional fields.")
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
            scores = np.asarray(model.seq_log_density(model.dist_to_encoder().seq_encode(rows)), dtype=np.float64)
            if scores.shape != (len(rows),):
                raise ValueError("structured model returned a malformed score vector.")
            if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
                raise ValueError("structured model returned invalid log densities.")
            out[:, k] = scores
        return out

    def proba_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """The exact posterior ``P(label | features)`` -- softmax of the per-label log-joints (shared evidence cancels)."""
        z = self.logits_batch(model, raw_inputs)
        impossible = np.flatnonzero(np.isneginf(z).all(axis=1))
        if impossible.size:
            raise ImpossibleEvidenceError(impossible.tolist())
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_batch(self, model: Any, raw_inputs: list[Any]) -> list[str]:
        """Predict labels for raw inputs by maximizing the per-label joint score."""
        logits = self.logits_batch(model, raw_inputs)
        impossible = np.flatnonzero(np.isneginf(logits).all(axis=1))
        if impossible.size:
            raise ImpossibleEvidenceError(impossible.tolist())
        idx = logits.argmax(axis=1)
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
            "field_count": self.field_count,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> StructuredClassifierIO:
        """Rebuild a structured-classifier adapter from its artifact ``io`` specification."""
        return cls(
            spec.get("field_keys"),
            spec["label_index"],
            spec["labels"],
            field_count=spec.get("field_count"),
        )


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
