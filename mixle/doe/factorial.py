"""Classical factorial, screening, and response-surface experiment designs.

These complement the space-filling generators in :mod:`mixle.doe.designs` with the structured "named"
designs of classical DOE: two-level fractional factorials and Plackett-Burman screening designs, and
the central-composite and Box-Behnken response-surface designs.

Every generator takes per-dimension ``bounds`` (a sequence of ``(low, high)`` pairs, one per factor)
and returns a ``(n_runs, d)`` array. The design is built in *coded* units -- two-level factors at
``-1`` / ``+1``, response-surface axial/centre points relative to that -- then mapped into ``bounds``
so ``-1`` -> ``low``, ``+1`` -> ``high``, ``0`` -> the midpoint. Pass ``coded=True`` to get the raw
coded matrix instead (the natural input to the analysis routines in :mod:`mixle.doe.analysis`).
"""

from __future__ import annotations

import numpy as np

from mixle.doe.designs import Bounds, _as_bounds

# Standard cyclic generating rows for the non-power-of-two Plackett-Burman designs (length N-1).
_PB_GEN: dict[int, str] = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _coded_to_bounds(coded: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map a coded design (factors centred on 0) into ``bounds``: ``-1`` -> low, ``+1`` -> high."""
    mid = 0.5 * (b[:, 0] + b[:, 1])
    half = 0.5 * (b[:, 1] - b[:, 0])
    return mid + coded * half


def _two_level_full(k: int) -> np.ndarray:
    """Full two-level factorial in coded ``+/-1`` units: ``(2**k, k)``, first factor varying fastest."""
    cols = []
    for i in range(k):
        block = 2**i
        pattern = np.concatenate([np.full(block, -1.0), np.full(block, 1.0)])
        cols.append(np.tile(pattern, 2 ** (k - i - 1)))
    return np.stack(cols, axis=1)


def fractional_factorial(bounds: Bounds, generators: str | list, *, coded: bool = False) -> np.ndarray:
    """Two-level fractional factorial ``2**(k-p)`` from pyDOE-style generator strings.

    ``generators`` names one column per factor as a product of base factors, e.g. ``"a b c ab ac"`` --
    a ``2**(5-2)`` design whose factors ``d, e`` are deliberately aliased as ``d = ab``, ``e = ac``.
    Base factors are the distinct single letters; each token is the elementwise product of its letters,
    optionally negated with a leading ``-``. The number of tokens must equal ``len(bounds)``.

    Returns a ``(2**k, d)`` design (``k`` = number of base factors) mapped into ``bounds`` -- or the
    raw coded ``+/-1`` matrix if ``coded=True``.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    tokens = generators.split() if isinstance(generators, str) else [str(t) for t in generators]
    if len(tokens) != d:
        raise ValueError("generators must name exactly one column per dimension (len(bounds)).")
    letters = sorted({ch for t in tokens for ch in t if ch.isalpha()})
    if not letters:
        raise ValueError("generators must reference at least one base-factor letter.")
    base = _two_level_full(len(letters))
    col_of = {letters[i]: base[:, i] for i in range(len(letters))}
    out = np.empty((base.shape[0], d), dtype=np.float64)
    for j, token in enumerate(tokens):
        name = token.lstrip("+-")
        if not name or any(ch not in col_of for ch in name):
            raise ValueError(f"generator token {token!r} references an undefined base factor.")
        col = np.ones(base.shape[0])
        for ch in name:
            col = col * col_of[ch]
        out[:, j] = -col if token.startswith("-") else col
    return out if coded else _coded_to_bounds(out, b)


def _pb_cyclic(gen: str) -> np.ndarray:
    """Cyclic Plackett-Burman: shift the generating row ``N-2`` times, then append an all-low row."""
    g = np.array([1.0 if c == "+" else -1.0 for c in gen], dtype=np.float64)
    n = g.size + 1  # N runs
    rows = [np.roll(g, i) for i in range(n - 1)]
    rows.append(-np.ones(n - 1))
    return np.array(rows)


def _hadamard_pb(n: int) -> np.ndarray:
    """``(N, N-1)`` Plackett-Burman factor columns (orthogonal, balanced ``+/-1``)."""
    if n & (n - 1) == 0:  # power of two -> Hadamard matrix, drop its all-ones first column
        from scipy.linalg import hadamard

        return hadamard(n).astype(np.float64)[:, 1:]
    if n in _PB_GEN:
        return _pb_cyclic(_PB_GEN[n])
    raise ValueError(f"no Plackett-Burman construction for N={n}.")


def plackett_burman(bounds: Bounds, *, coded: bool = False) -> np.ndarray:
    """Plackett-Burman two-level screening design for ``len(bounds)`` factors.

    Returns ``N`` runs where ``N`` is the smallest multiple of four that is at least ``d + 1`` (so the
    design is saturated or near-saturated, ideal for screening many factors in few runs). For ``N`` a
    power of two the design is a Hadamard matrix (a resolution-III fractional factorial); for ``N`` in
    ``{12, 20, 24}`` a known cyclic generator is used; otherwise ``N`` is rounded up to the next power
    of two so a design always exists. Main effects are mutually orthogonal but aliased with two-factor
    interactions -- use it to find the few large effects, then follow up with a fuller design.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    n = ((d + 1 + 3) // 4) * 4
    if not (n & (n - 1) == 0 or n in _PB_GEN):
        p = 1
        while p < n:
            p *= 2
        n = p
    coded_design = _hadamard_pb(n)[:, :d]
    return coded_design if coded else _coded_to_bounds(coded_design, b)


# Default centre-point counts for Box-Behnken designs (Box & Behnken 1960), keyed by factor count.
_BB_CENTERS: dict[int, int] = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6}


def central_composite(
    bounds: Bounds,
    *,
    center: int = 4,
    alpha: str | float = "rotatable",
    coded: bool = False,
) -> np.ndarray:
    """Central-composite design (CCD) for fitting a full second-order response surface.

    A CCD stacks three parts: the ``2**d`` two-level factorial corners (estimate linear and
    interaction terms), ``2*d`` axial / star points at distance ``alpha`` on each axis (estimate the
    pure-quadratic curvature), and ``center`` replicates at the centre (estimate pure error and
    curvature). ``alpha`` sets the axial distance:

      * ``"rotatable"`` (default) -- ``alpha = (2**d)**0.25``, so prediction variance depends only on
        distance from the centre;
      * ``"orthogonal"`` -- the value making the second-order terms orthogonal (depends on ``center``);
      * ``"face"`` (a face-centred CCD / CCF) -- ``alpha = 1``, keeping every run inside the cube;
      * a positive float -- used directly.

    Returns ``(2**d + 2*d + center, d)`` rows mapped into ``bounds`` (or coded if ``coded=True``).
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    fact = _two_level_full(d)
    nf = fact.shape[0]
    nc = int(center)
    if isinstance(alpha, str):
        if alpha == "rotatable":
            a = nf**0.25
        elif alpha == "face":
            a = 1.0
        elif alpha == "orthogonal":
            ntot = nf + 2 * d + nc
            a = (nf * (np.sqrt(ntot) - np.sqrt(nf)) ** 2 / 4.0) ** 0.25
        else:
            raise ValueError("alpha must be 'rotatable', 'orthogonal', 'face', or a positive float.")
    else:
        a = float(alpha)
        if a <= 0.0:
            raise ValueError("numeric alpha must be positive.")
    axial = np.zeros((2 * d, d))
    for i in range(d):
        axial[2 * i, i] = -a
        axial[2 * i + 1, i] = a
    coded_design = np.vstack([fact, axial, np.zeros((nc, d))])
    return coded_design if coded else _coded_to_bounds(coded_design, b)


def box_behnken(bounds: Bounds, *, center: int | None = None, coded: bool = False) -> np.ndarray:
    """Box-Behnken response-surface design (3 levels per factor, no corner runs).

    For every pair of factors it runs the four ``(+/-1, +/-1)`` combinations with all other factors at
    the centre, plus ``center`` centre replicates. Unlike a CCD it never sets all factors to an extreme
    at once (no cube corners), which is useful when those combinations are expensive or infeasible, and
    it needs only three levels per factor. Requires ``d >= 3``.

    Returns ``(4 * C(d, 2) + center, d)`` rows mapped into ``bounds`` (or coded if ``coded=True``).
    """
    from itertools import combinations

    b = _as_bounds(bounds)
    d = b.shape[0]
    if d < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")
    quad = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    blocks = []
    for i, j in combinations(range(d), 2):
        block = np.zeros((4, d))
        block[:, [i, j]] = quad
        blocks.append(block)
    nc = _BB_CENTERS.get(d, 3) if center is None else int(center)
    coded_design = np.vstack([*blocks, np.zeros((nc, d))])
    return coded_design if coded else _coded_to_bounds(coded_design, b)
