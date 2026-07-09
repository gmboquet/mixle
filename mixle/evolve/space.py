"""A small typed search space: the categorical/integer/box gap-filler over the numeric BO backend.

:mod:`mixle.doe` optimizes over a continuous numeric box (a sequence of ``(low, high)`` bounds). Real
configuration search needs categorical and integer knobs too, and the evolutionary/bandit backends want
to ``sample`` and walk ``neighbors`` natively. :class:`Space` is the encoding layer that gives both:

* :meth:`Space.to_bounds` -- a continuous ``(low, high)`` box for the BO backend, with categoricals
  encoded as integer *indices* (``[-0.5, k - 0.5]`` so a round lands on each level with equal width).
* :meth:`Space.encode` / :meth:`Space.decode` -- the lossy round-trip between a config dict and the
  numeric vector the BO backend proposes (decode rounds integers / categorical indices, clips to range).
* :meth:`Space.sample` / :meth:`Space.neighbors` -- native draws and local moves for the
  evolutionary/bandit backends that handle categoricals without going through the numeric box.

The space lives in ``evolve`` (not ``doe``) on purpose: ``doe`` stays a pure numeric-box optimizer, and
the encoding policy that bridges categoricals into it is an ``evolve`` concern.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Real:
    """A continuous dimension over ``[lo, hi]``."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(f"Real bounds must satisfy lo < hi (got {self.lo}, {self.hi}).")

    def bounds(self) -> tuple[float, float]:
        """Return the numeric bounds used by continuous optimizers."""
        return (float(self.lo), float(self.hi))

    def sample(self, rng: np.random.RandomState) -> float:
        """Draw a uniformly distributed value from the interval."""
        return float(rng.uniform(self.lo, self.hi))

    def encode(self, value: Any) -> float:
        """Clip and encode ``value`` as a floating-point coordinate."""
        return float(np.clip(float(value), self.lo, self.hi))

    def decode(self, x: float) -> float:
        """Clip a numeric optimizer coordinate back into the interval."""
        return float(np.clip(float(x), self.lo, self.hi))

    def neighbors(self, value: Any) -> list[float]:
        """A coarse local move: +/- 10% of the range, clipped to bounds."""
        v = float(value)
        step = 0.1 * (self.hi - self.lo)
        out = [self.decode(v - step), self.decode(v + step)]
        return [n for n in out if n != v]


@dataclass(frozen=True)
class Integer:
    """An integer dimension over the inclusive range ``[lo, hi]``."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not int(self.lo) < int(self.hi):
            raise ValueError(f"Integer bounds must satisfy lo < hi (got {self.lo}, {self.hi}).")

    def bounds(self) -> tuple[float, float]:
        """Return widened numeric bounds so rounding covers each integer level."""
        # widen by 0.5 each side so a uniform round lands on each integer with equal width.
        return (float(self.lo) - 0.5, float(self.hi) + 0.5)

    def sample(self, rng: np.random.RandomState) -> int:
        """Draw an integer uniformly from the inclusive range."""
        return int(rng.randint(int(self.lo), int(self.hi) + 1))

    def encode(self, value: Any) -> float:
        """Round, clip, and encode ``value`` as a numeric coordinate."""
        return float(int(np.clip(round(float(value)), self.lo, self.hi)))

    def decode(self, x: float) -> int:
        """Round and clip a numeric optimizer coordinate to an integer value."""
        return int(np.clip(int(round(float(x))), self.lo, self.hi))

    def neighbors(self, value: Any) -> list[int]:
        """Return adjacent integer values within the range."""
        v = int(value)
        out = []
        if v - 1 >= self.lo:
            out.append(v - 1)
        if v + 1 <= self.hi:
            out.append(v + 1)
        return out


@dataclass(frozen=True)
class Categorical:
    """An unordered categorical dimension over a finite list of ``choices``."""

    choices: tuple[Any, ...]

    def __init__(self, choices: Sequence[Any]) -> None:
        choices = tuple(choices)
        if len(choices) == 0:
            raise ValueError("Categorical needs at least one choice.")
        object.__setattr__(self, "choices", choices)

    def bounds(self) -> tuple[float, float]:
        """Return widened index bounds so rounding covers each categorical choice."""
        # encode as an index in [0, k-1]; widen by 0.5 each side for equal-width rounding.
        return (-0.5, float(len(self.choices)) - 0.5)

    def sample(self, rng: np.random.RandomState) -> Any:
        """Draw one choice uniformly at random."""
        return self.choices[int(rng.randint(0, len(self.choices)))]

    def _index_of(self, value: Any) -> int:
        for i, c in enumerate(self.choices):
            if c == value:
                return i
        raise ValueError(f"{value!r} is not a choice of {self.choices!r}.")

    def encode(self, value: Any) -> float:
        """Encode a choice as its floating-point index."""
        return float(self._index_of(value))

    def decode(self, x: float) -> Any:
        """Round and clip an optimizer coordinate back to a choice."""
        idx = int(np.clip(int(round(float(x))), 0, len(self.choices) - 1))
        return self.choices[idx]

    def neighbors(self, value: Any) -> list[Any]:
        """Return all choices except ``value``."""
        idx = self._index_of(value)
        return [c for i, c in enumerate(self.choices) if i != idx]


Dimension = Real | Integer | Categorical


class Space:
    """A typed, named search space over :class:`Real` / :class:`Integer` / :class:`Categorical` dims.

    Construct from a dict mapping each parameter name to its dimension::

        space = Space({"mu": Real(-5, 5), "k": Integer(1, 4), "family": Categorical(["a", "b"])})

    Dimension order is the insertion order of the dict; :meth:`encode` / :meth:`decode` and
    :meth:`to_bounds` all use that same fixed order so the numeric vector and the box align.
    """

    def __init__(self, dims: dict[str, Dimension]) -> None:
        if not dims:
            raise ValueError("Space needs at least one named dimension.")
        for name, dim in dims.items():
            if not isinstance(dim, (Real, Integer, Categorical)):
                raise TypeError(f"dimension {name!r} must be a Real / Integer / Categorical, got {type(dim).__name__}.")
        self.dims: dict[str, Dimension] = dict(dims)
        self.names: tuple[str, ...] = tuple(dims.keys())

    @property
    def ndim(self) -> int:
        """Number of dimensions in the fixed search-space order."""
        return len(self.names)

    def to_bounds(self) -> list[tuple[float, float]]:
        """The continuous ``(low, high)`` box for the BO backend (categoricals as integer indices)."""
        return [self.dims[name].bounds() for name in self.names]

    def sample(self, rng: np.random.RandomState) -> dict[str, Any]:
        """Draw a random config dict, each dimension sampled natively."""
        return {name: self.dims[name].sample(rng) for name in self.names}

    def encode(self, config: dict[str, Any]) -> np.ndarray:
        """Encode a config dict into the numeric vector (in the fixed dimension order)."""
        missing = set(self.names) - set(config)
        if missing:
            raise ValueError(f"config is missing dimensions {sorted(missing)}.")
        return np.asarray([self.dims[name].encode(config[name]) for name in self.names], dtype=float)

    def decode(self, vector: Sequence[float]) -> dict[str, Any]:
        """Decode a numeric vector (BO proposal) back into a config dict (rounding / clipping)."""
        vec = np.asarray(vector, dtype=float).reshape(-1)
        if vec.shape[0] != self.ndim:
            raise ValueError(f"vector has length {vec.shape[0]}, expected {self.ndim}.")
        return {name: self.dims[name].decode(vec[i]) for i, name in enumerate(self.names)}

    def neighbors(self, point: dict[str, Any]) -> list[dict[str, Any]]:
        """All configs one local move away in exactly one dimension (the evolutionary mutation set)."""
        out: list[dict[str, Any]] = []
        for name in self.names:
            for nv in self.dims[name].neighbors(point[name]):
                cand = dict(point)
                cand[name] = nv
                out.append(cand)
        return out

    def __repr__(self) -> str:
        body = ", ".join(f"{n}={self.dims[n]!r}" for n in self.names)
        return f"Space({body})"


__all__ = ["Space", "Real", "Integer", "Categorical", "Dimension"]
