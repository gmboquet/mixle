"""``StructuredAdapter`` for adapting a frozen multimodal encoder while preserving transfer.

For a frozen VLM, the encoder is the expensive component and the trainable bridge
on top is the application-specific component. ``StructuredAdapter`` uses a
low-capacity structured map that adapts to a task while preserving zero-shot
transfer to text-specified classes; a full unstructured map can overfit and
damage that transfer even with regularization.

The map is a residual, class-agnostic transform of the image embedding::

    g(x) = x + (diag ⊙ x) + U Vᵀ x          # identity + diagonal reweight + rank-r correction

Three structural choices matter: (1) it is residual with weight decay, so it stays
near the encoder's alignment; (2) it is class-agnostic: targets enter only as
anchor embeddings such as class-text embeddings, so a map fit on some classes
still scores classes it never saw at training time; (3) geometry is cosine
throughout -- ``fit()`` L2-normalizes both the adapted embedding and the anchors
before the temperature-scaled softmax, and ``scores()`` normalizes both the same
way before its dot product. Anchor magnitude is therefore never meaningful at
either stage: what the training loss optimizes for is exactly what ``scores()``
reports at serving time. ``diag + U Vᵀ`` is the same diagonal+low-rank structure
Mixle uses for structured transition operators, here over a VLM bridge.

The same recipe applies to any frozen encoder that emits comparable embeddings.
Torch is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.utils.exact import require_exact_bool

# Rows at or below this L2 norm are treated as degenerate: normalizing one would silently divide
# into NaN (exact zero) or blow up to Inf (float32 underflow), rather than raise a clear error.
_MIN_NORM = 1e-12


def _torch() -> Any:
    import torch

    return torch


def _require_matrix(arr: Any, name: str, width: int) -> np.ndarray:
    """Coerce to ``float32`` and require a non-empty ``(n, width)`` array."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n, {width}), got shape {out.shape}")
    if out.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if out.shape[1] != width:
        raise ValueError(f"{name} has width {out.shape[1]}, expected {width} (StructuredAdapter.dim)")
    return out


def _require_nonzero_rows(arr: np.ndarray, name: str) -> None:
    """Raise if any row of ``arr`` has a (near-)zero L2 norm -- normalizing it would produce NaN/Inf."""
    norm = np.linalg.norm(arr.astype(np.float64), axis=1)
    bad = np.flatnonzero(norm <= _MIN_NORM)
    if bad.size:
        raise ValueError(f"{name} row {int(bad[0])} has a zero (or near-zero) norm; cannot L2-normalize it")


def _l2_normalize_np(arr: np.ndarray, name: str) -> np.ndarray:
    """L2-normalize rows of a numpy array; raises the same way as :func:`_require_nonzero_rows`."""
    _require_nonzero_rows(arr, name)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def _l2_normalize_torch(x: Any, name: str) -> Any:
    """L2-normalize rows of a torch tensor; raises the same way as :func:`_require_nonzero_rows`."""
    norm = x.norm(dim=1, keepdim=True)
    bad = (norm.reshape(-1) <= _MIN_NORM).nonzero(as_tuple=True)[0]
    if bad.numel():
        raise ValueError(f"{name} row {int(bad[0])} has a zero (or near-zero) norm; cannot L2-normalize it")
    return x / norm


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact, positive :class:`int` (mirrors this PR wave's ``_require_count``
    contract, e.g. ``mixle.substrate.multihop._require_count``): never a ``bool`` (an ``int`` subclass
    that would otherwise silently mean 0 or 1) and never a float (a fractional value would otherwise be
    silently truncated by a bare ``int()``, and even a whole-valued one like ``300.0`` is not the exact
    type this promises callers).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive int (at least one optimization step), got {value!r}")
    return value


def _require_labels(labels: Any, n_rows: int, n_anchors: int) -> np.ndarray:
    """Validate ``labels`` are one in-range integer anchor index per embedding row; return as ``int64``."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be a 1-D array, got shape {arr.shape}")
    if arr.shape[0] != n_rows:
        raise ValueError(f"labels has {arr.shape[0]} entries but embeddings has {n_rows} rows")
    if arr.shape[0] == 0:
        raise ValueError("labels must contain at least one entry")
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.bool_):
        raise ValueError(f"labels must be numeric class indices, got dtype {arr.dtype}")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(arr == np.floor(arr)):
        raise ValueError("labels must be integer class indices into anchors, got fractional values")
    idx = arr.astype(np.int64)
    if idx.min() < 0 or idx.max() >= n_anchors:
        raise ValueError(
            f"labels must index into anchors (valid range 0..{n_anchors - 1}), got range "
            f"[{int(idx.min())}, {int(idx.max())}]"
        )
    return idx


class StructuredAdapter:
    """A residual diagonal+low-rank adapter over frozen embeddings.

    ``rank`` sets the low-rank correction's width; ``weight_decay`` pulls the map toward identity (preserve
    the encoder's geometry). ``full=True`` selects the unstructured baseline. Fit on
    ``(embeddings, labels, anchors)``; score any embeddings against any anchors, including anchors for classes
    not seen in training.
    """

    def __init__(self, dim: int, *, rank: int = 8, weight_decay: float = 1.0, full: bool = False) -> None:
        self.dim = int(dim)
        self.rank = int(rank)
        self.weight_decay = float(weight_decay)
        self.full = require_exact_bool(full, "full")
        self._params: list[Any] | None = None
        self._logit_scale: Any = None
        self._built: tuple[list[Any], Any] | None = None

    def _build(self) -> tuple[list[Any], Any]:
        torch = _torch()
        if self.full:
            w = torch.zeros(self.dim, self.dim, requires_grad=True)  # residual full matrix (unstructured)
            return [w], lambda x: x + x @ w.T
        diag = torch.zeros(self.dim, requires_grad=True)
        u = torch.zeros(self.dim, self.rank, requires_grad=True)
        v = (0.01 * torch.randn(self.dim, self.rank)).requires_grad_(True)
        return [diag, u, v], lambda x: x + x * diag + (x @ v) @ u.T

    def _apply(self, x: Any) -> Any:
        if self._built is None:
            raise RuntimeError("StructuredAdapter.transform() called before fit(): the residual map is untrained.")
        _, fn = self._built
        return fn(x)

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        anchors: np.ndarray,
        *,
        epochs: int = 300,
        lr: float = 0.01,
        init_temp: float = 0.07,
        seed: int = 0,
    ) -> StructuredAdapter:
        """Train the residual map so ``g(image)`` matches its label's anchor. ``labels`` index into ``anchors``.

        Both the adapted embedding and the anchors are L2-normalized before the temperature-scaled softmax,
        so the training objective is pure cosine similarity -- exactly what ``scores()`` reports at serving
        time. Anchor magnitude therefore never enters either the loss or the score; only direction does.
        ``seed`` makes the low-rank map's random initialization reproducible (``diag`` and ``U`` start at
        zero -- only ``V``, and the full-matrix baseline's ``w``, involve randomness, and ``w`` starts at
        zero too, so today only ``V`` actually needs the seed).

        Validates its inputs and requires at least one real optimization step before mutating any adapter
        state, so a failed call (bad shapes/labels/epochs, or training that diverges to a non-finite loss)
        leaves the adapter exactly as it was -- still requiring a genuine ``fit()`` before ``transform()``/
        ``scores()`` will work.
        """
        epochs = _require_positive_int(epochs, "epochs")
        embeddings = _require_matrix(embeddings, "embeddings", self.dim)
        anchors = _require_matrix(anchors, "anchors", self.dim)
        _require_nonzero_rows(embeddings, "embeddings")
        labels = _require_labels(labels, embeddings.shape[0], anchors.shape[0])

        torch = _torch()
        # Only V (and, in the full=True path, nothing -- w starts at zero) is randomly initialized.
        # Seeded right here, immediately before the draw, so no earlier unrelated torch call can make
        # this non-reproducible.
        torch.manual_seed(int(seed))
        params, fn = self._build()
        logit_scale = torch.tensor(float(np.log(1.0 / init_temp)), requires_grad=True)
        x = torch.as_tensor(embeddings)
        y = torch.as_tensor(labels)
        a = _l2_normalize_torch(torch.as_tensor(anchors), "anchors")
        opt = torch.optim.Adam(
            [
                {"params": params, "weight_decay": self.weight_decay},
                {"params": [logit_scale], "weight_decay": 0.0},
            ],
            lr=lr,
        )
        loss = None
        for _ in range(epochs):
            g = _l2_normalize_torch(fn(x), "adapted embeddings")
            logits = logit_scale.exp() * (g @ a.T)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                "StructuredAdapter.fit(): training did not converge to a finite loss; the map was not fitted"
            )
        # Only now does fit() mutate adapter state: if anything above raised, the adapter is left exactly
        # as it was before the call, so it still correctly requires a real fit() before transform()/scores().
        self._built = (params, fn)
        self._logit_scale = logit_scale
        self._params = params
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the learned residual map and L2-normalize -- the adapted embedding."""
        if self._built is None:
            raise RuntimeError("StructuredAdapter.transform() called before fit(): the residual map is untrained.")
        embeddings = _require_matrix(embeddings, "embeddings", self.dim)
        _require_nonzero_rows(embeddings, "embeddings")
        torch = _torch()
        with torch.no_grad():
            g = _l2_normalize_torch(self._apply(torch.as_tensor(embeddings)), "adapted embeddings")
        return g.numpy()

    def scores(self, embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Cosine similarity of adapted embeddings to ``anchors``; anchors may represent new classes."""
        g = self.transform(embeddings)
        anchors = _require_matrix(anchors, "anchors", self.dim)
        a = _l2_normalize_np(anchors, "anchors")
        return g @ a.T

    def predict(self, embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Return the highest-scoring anchor index for each embedding."""
        return self.scores(embeddings, anchors).argmax(1)

    def n_params(self) -> int:
        """Return the number of learned adapter parameters."""
        if self._params is None:
            return self.dim * self.dim if self.full else self.dim + 2 * self.dim * self.rank
        return int(sum(p.numel() for p in self._params))
