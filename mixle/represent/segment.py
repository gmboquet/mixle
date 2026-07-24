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
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


class ElementSegmenter(Segmenter):
    """A sequence of hashable symbols (chars, amino acids, k-mers, categories) -> ``(n,)`` ids via a fixed alphabet.

    Given ``alphabet`` (the ordered symbols), each element maps to its index; unknown symbols map to ``0``. The
    natural decomposition for proteins/genomes/any categorical sequence, and for characters (``alphabet=list(...)``).
    """

    discrete = True

    def __init__(self, alphabet: list[Any]) -> None:
        self.alphabet = list(alphabet)
        self.index = {s: i for i, s in enumerate(self.alphabet)}
        self.num_categories = len(self.alphabet)

    def segment(self, raw: Any) -> np.ndarray:
        """Map sequence elements through the fixed alphabet index."""
        return np.asarray([self.index.get(s, 0) for s in raw], dtype=np.int64)


class PatchSegmenter(Segmenter):
    """An image ``(H, W)`` or ``(C, H, W)`` -> ``(n_patches, patch_features)`` float units (ViT-style, no vocab)."""

    discrete = False

    def __init__(self, patch: int = 8) -> None:
        self.patch = int(patch)

    def segment(self, raw: Any) -> np.ndarray:
        """Split an image tensor into flattened non-overlapping patches."""
        img = np.asarray(raw, dtype=np.float32)
        if img.ndim == 2:
            img = img[None, :, :]
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
        hp, wp = h // p, w // p
        img = img[:, : hp * p, : wp * p].reshape(c, hp, p, wp, p)
        patches = img.transpose(1, 3, 0, 2, 4).reshape(hp * wp, c * p * p)
        return patches.astype(np.float32)

    def unit_features(self, channels: int = 1) -> int:
        """Return feature width of one flattened patch."""
        return channels * self.patch * self.patch


class WindowSegmenter(Segmenter):
    """A 1-D signal ``(T,)`` -> ``(n_frames, window)`` float units by a sliding window (seismic/audio/time-series)."""

    discrete = False

    def __init__(self, window: int = 64, hop: int | None = None) -> None:
        self.window = int(window)
        self.hop = int(hop) if hop is not None else int(window)

    def segment(self, raw: Any) -> np.ndarray:
        """Split a one-dimensional signal into fixed windows.

        A signal shorter than ``window`` still yields exactly one window (WindowSegmenter's
        contract is to always return at least one unit, unlike PatchSegmenter which has no such
        contract and rejects a too-small image outright): the real samples fill ``[:len(x)]`` and
        only the remaining slots are zero-padded, so the padding never overwrites real data.
        """
        x = np.asarray(raw, dtype=np.float32).ravel()
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
        n = (len(x) - self.window) // self.hop + 1
        return np.stack([x[i * self.hop : i * self.hop + self.window] for i in range(n)]).astype(np.float32)


class WholeSegmenter(Segmenter):
    """A single feature vector -> ``(1, feat)``: the object is one unit (a pooled structure descriptor, a record)."""

    discrete = False

    def segment(self, raw: Any) -> np.ndarray:
        """Treat the whole input as one feature-vector segment."""
        v = np.asarray(raw, dtype=np.float32).ravel()
        return v[None, :]


class SetSegmenter(Segmenter):
    """A set/list of feature vectors -> ``(n_elements, feat)``: nodes of a graph, atoms of a molecule, taxa of a section.

    The general structured-object decomposition -- a scientific structure becomes its set of element features
    (which a downstream model can further couple with a structure/message-passing embedding).
    """

    discrete = False

    def segment(self, raw: Any) -> np.ndarray:
        """Return set elements as rows of a feature matrix."""
        arr = np.asarray(raw, dtype=np.float32)
        return arr if arr.ndim == 2 else arr[None, :]
