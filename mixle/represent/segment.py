"""Segmenters -- cut a raw object of any modality into units, WITHOUT committing to a vocabulary.

A ``Segmenter`` turns one raw object (a string, an image, a waveform, a set of node features) into an array of
*units*: ``(n_units,)`` integer ids for a discrete alphabet, or ``(n_units, feat...)`` float features for a
continuous modality. That is the whole of "the tokenizer" that is *not* objective-dependent -- it is a
decomposition, not a vocabulary. Discreteness (mapping units to a codebook of ids) is a separate, optional,
*learned* step (:mod:`mixle.represent.quantize`), so a segmenter never has to guess the right tokens.

Fixed segmenters (bytes, characters, patches, windows, whole-object, element-set) commit to nothing beyond where
to cut and keep all information; the model learns the rest. Learned segmenters (a segmental HMM over boundaries)
are the objective-coupled upgrade and plug into the same contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Segmenter:
    """Base: ``segment(raw) -> np.ndarray``. ``discrete`` says whether units are ids (vs. float features)."""

    discrete: bool = False

    def segment(self, raw: Any) -> np.ndarray:  # pragma: no cover - overridden
        """Segment raw input into model units."""
        raise NotImplementedError


class ByteSegmenter(Segmenter):
    """A string/bytes -> ``(n,)`` byte ids in ``[0, 256)``. The vocabulary-free text decomposition."""

    discrete = True
    num_categories = 256

    def segment(self, raw: Any) -> np.ndarray:
        """Return UTF-8 byte ids for a string or bytes-like object."""
        if isinstance(raw, str):
            data = raw.encode("utf-8")
        else:
            if isinstance(raw, (bool, int, np.bool_, np.integer)):
                raise TypeError("ByteSegmenter accepts text or an explicit bytes-like buffer, not a numeric length")
            try:
                data = memoryview(raw).tobytes()
            except TypeError:
                raise TypeError(f"ByteSegmenter accepts str or a bytes-like buffer, got {type(raw).__name__}") from None
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


class ElementSegmenter(Segmenter):
    """A sequence of hashable symbols (chars, amino acids, k-mers, categories) -> ``(n,)`` ids via a fixed alphabet.

    Given ``alphabet`` (the ordered symbols), each element maps to its index. Out-of-vocabulary symbols map to
    :attr:`unknown_id`, a state reserved *past* the declared alphabet (``len(alphabet)``), so ``num_categories`` is
    ``len(alphabet) + 1``. Unknown symbols used to collapse onto id ``0``, which is already the first declared real
    category -- an unseen residue/character/class then became positive evidence for a genuinely observed symbol.
    The alphabet must also be unique: a repeated entry used to overwrite its own index while still inflating
    ``num_categories``, leaving an id no symbol could ever produce.

    The natural decomposition for proteins/genomes/any categorical sequence, and for characters
    (``alphabet=list(...)``).
    """

    discrete = True

    def __init__(self, alphabet: list[Any]) -> None:
        self.alphabet = list(alphabet)
        self.index: dict[Any, int] = {}
        for i, symbol in enumerate(self.alphabet):
            if symbol in self.index:
                raise ValueError(
                    f"ElementSegmenter requires a unique alphabet; {symbol!r} appears at both index "
                    f"{self.index[symbol]} and {i}."
                )
            self.index[symbol] = i
        self.unknown_id = len(self.alphabet)
        self.num_categories = len(self.alphabet) + 1  # + the reserved out-of-vocabulary state

    def segment(self, raw: Any) -> np.ndarray:
        """Map sequence elements through the fixed alphabet index, unknowns to the reserved :attr:`unknown_id`."""
        return np.asarray([self.index.get(s, self.unknown_id) for s in raw], dtype=np.int64)


class PatchSegmenter(Segmenter):
    """An image ``(H, W)`` or ``(C, H, W)`` -> ``(n_patches, patch_features)`` float units (ViT-style, no vocab)."""

    discrete = False

    def __init__(self, patch: int = 8) -> None:
        if isinstance(patch, (bool, np.bool_)) or not isinstance(patch, (int, np.integer)) or patch <= 0:
            raise ValueError(f"patch must be a positive integer, got {patch!r}")
        self.patch = int(patch)

    def segment(self, raw: Any) -> np.ndarray:
        """Split an image tensor into flattened non-overlapping patches."""
        img = np.asarray(raw, dtype=np.float32)
        if img.ndim == 2:
            img = img[None, :, :]
        if img.ndim != 3:
            raise ValueError(f"PatchSegmenter requires an (H, W) or (C, H, W) image, got shape {img.shape}")
        if not np.isfinite(img).all():
            raise ValueError("PatchSegmenter requires finite image values")
        c, h, w = img.shape
        p = self.patch
        if h < p or w < p:
            # h // p or w // p would silently be 0, producing a (0, features) array -- no patches,
            # no error, no warning. That is almost certainly a caller mistake (wrong patch size for
            # the image, or the wrong image), so fail loudly instead of returning an empty unit set.
            raise ValueError(
                f"PatchSegmenter(patch={p}): image is smaller than the patch size in at least one "
                f"dimension (got h={h}, w={w}); both must be >= patch."
            )
        if h % p or w % p:
            raise ValueError(
                f"PatchSegmenter(patch={p}) requires image height and width to be divisible by patch; "
                f"got h={h}, w={w}. Pad or crop explicitly so the geometry change is visible to the caller."
            )
        hp, wp = h // p, w // p
        img = img.reshape(c, hp, p, wp, p)
        patches = img.transpose(1, 3, 0, 2, 4).reshape(hp * wp, c * p * p)
        return patches.astype(np.float32)

    def unit_features(self, channels: int = 1) -> int:
        """Return feature width of one flattened patch."""
        return channels * self.patch * self.patch


class WindowSegmenter(Segmenter):
    """A 1-D signal ``(T,)`` -> ``(n_frames, window)`` float units by a sliding window (seismic/audio/time-series)."""

    discrete = False

    def __init__(self, window: int = 64, hop: int | None = None) -> None:
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        resolved_hop = window if hop is None else hop
        if (
            isinstance(resolved_hop, (bool, np.bool_))
            or not isinstance(resolved_hop, (int, np.integer))
            or resolved_hop <= 0
        ):
            raise ValueError(f"hop must be a positive integer, got {hop!r}")
        if resolved_hop > window:
            raise ValueError("hop cannot exceed window because that would leave samples uncovered")
        self.window = int(window)
        self.hop = int(resolved_hop)

    def segment(self, raw: Any) -> np.ndarray:
        """Split a one-dimensional signal into fixed windows.

        A signal shorter than ``window`` still yields exactly one window (WindowSegmenter's
        contract is to always return at least one unit, unlike PatchSegmenter which has no such
        contract and rejects a too-small image outright): the real samples fill ``[:len(x)]`` and
        only the remaining slots are zero-padded, so the padding never overwrites real data.
        """
        x = np.asarray(raw, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"WindowSegmenter requires a one-dimensional signal, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("WindowSegmenter requires a finite signal")
        if len(x) < self.window:
            # (len(x) - window) // hop + 1 is <= 0 here, so this used to be treated as "zero
            # windows" and silently answered with one fabricated ALL-zero window instead --
            # discarding the real (if short) signal entirely. [1, 2] and [9, 8] under window=4 both
            # produced the identical all-zero output. Keep the "always >= 1 window" contract, but
            # honor it honestly: zero-pad only the slots beyond the real samples, instead of
            # overwriting the real samples themselves along with everything else.
            padded = np.zeros(self.window, dtype=np.float32)
            padded[: len(x)] = x
            return padded[None, :]
        starts = list(range(0, len(x) - self.window + 1, self.hop))
        frames = [x[start : start + self.window] for start in starts]
        if starts[-1] + self.window < len(x):
            start = starts[-1] + self.hop
            tail = np.zeros(self.window, dtype=np.float32)
            observed = x[start : start + self.window]
            tail[: len(observed)] = observed
            frames.append(tail)
        return np.stack(frames).astype(np.float32)


class WholeSegmenter(Segmenter):
    """A single feature vector -> ``(1, feat)``: the object is one unit (a pooled structure descriptor, a record)."""

    discrete = False

    def segment(self, raw: Any) -> np.ndarray:
        """Treat the whole input as one feature-vector segment."""
        v = np.asarray(raw, dtype=np.float32)
        if v.ndim == 0:
            v = v.reshape(1)
        if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
            raise ValueError("WholeSegmenter requires one non-empty finite feature vector")
        return v[None, :]


class SetSegmenter(Segmenter):
    """A set/list of feature vectors -> ``(n_elements, feat)``: nodes of a graph, atoms of a molecule, taxa of a section.

    The general structured-object decomposition -- a scientific structure becomes its set of element features
    (which a downstream model can further couple with a structure/message-passing embedding).
    """

    discrete = False

    def segment(self, raw: Any) -> np.ndarray:
        """Return set elements as rows of a feature matrix."""
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError("SetSegmenter requires a rectangular numeric feature collection") from None
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0 or not np.isfinite(arr).all():
            raise ValueError("SetSegmenter requires a non-empty finite matrix with fixed positive feature width")
        return arr
