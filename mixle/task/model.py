"""``TaskModel`` -- a fitted small model wrapped as a plain callable: ``task(raw_input) -> result``.

The artifact contract (:mod:`mixle.task.artifact`) makes a model *durable*; this makes it *usable*. A task model
pairs a fitted model (a torch module or a mixle distribution) with an **I/O adapter** that turns raw input
(a string, a record) into the model's input and the model's output into a result (a label, a number). The
adapter is serialized into the manifest's ``io`` block, so ``TaskModel.load(path)`` reconstructs the whole
``raw -> result`` function in a fresh process -- the point of the package: a regular program loads a small
local model and just calls it.

Adapters self-describe and self-rebuild through a registry (``register_adapter`` / ``IOAdapter.from_spec``). The
built-in :class:`TextClassifierIO` is the workhorse for the distillation path: a dependency-free hashed
character n-gram featurizer feeds a small classifier whose argmax indexes a stored label list -- the shape of a
"scrape this field" / "classify this line" model you distill from a big teacher and run locally.
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

    Deterministic and dependency-free (stdlib ``hashlib``), so it serializes as three numbers and rebuilds
    identically anywhere -- no fitted vocabulary, no external tokenizer. Counts are L2-normalized per row.
    """

    def __init__(self, n: int = 3, dim: int = 256, seed: int = 0) -> None:
        self.n = int(n)
        self.dim = int(dim)
        self.seed = int(seed)

    def _bucket(self, gram: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{gram}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def transform(self, texts: list[str]) -> np.ndarray:
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
        return {"n": self.n, "dim": self.dim, "seed": self.seed}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> HashedNGram:
        return cls(n=spec["n"], dim=spec["dim"], seed=spec["seed"])


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


class TextClassifierIO:
    """``str -> label``: hashed n-gram features into a small classifier, argmax indexed against a label list."""

    kind = "text_classifier"

    def __init__(self, featurizer: HashedNGram, labels: list[str]) -> None:
        self.featurizer = featurizer
        self.labels = list(labels)

    def features(self, raw_inputs: list[str]) -> np.ndarray:
        return self.featurizer.transform(raw_inputs)

    def predict_batch(self, module: Any, raw_inputs: list[str]) -> list[str]:
        import torch

        feats = self.features(raw_inputs)
        module.eval()
        with torch.no_grad():
            logits = module(torch.from_numpy(feats)).cpu().numpy()
        idx = np.asarray(logits).reshape(len(raw_inputs), -1).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict(self, module: Any, raw_input: str) -> str:
        return self.predict_batch(module, [raw_input])[0]

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "featurizer": self.featurizer.to_spec(), "labels": self.labels}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> TextClassifierIO:
        return cls(HashedNGram.from_spec(spec["featurizer"]), spec["labels"])


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
        return self.adapter.predict(self.model, raw_input)

    def batch(self, raw_inputs: list[Any]) -> list[Any]:
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
        model, _ = _artifact.load_json(path)
        return cls(model, adapter, payload="json", task=manifest.task, meta=manifest.meta)
