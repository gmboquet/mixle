"""Classical factorial, screening, and response-surface experiment designs.

These complement the space-filling generators in :mod:`mixle.doe.designs` with the structured "named"
designs of classical DOE: two-level fractional factorials and Plackett-Burman screening designs, and
the central-composite and Box-Behnken response-surface designs.

Every generator takes per-dimension ``bounds`` (a sequence of ``(low, high)`` pairs, one per factor)
and returns a ``(n_runs, d)`` array. The design is built in *coded* units -- two-level factors at
``-1`` / ``+1``, response-surface axial/centre points relative to that -- then mapped into ``bounds``
so ``-1`` -> ``low``, ``+1`` -> ``high``, ``0`` -> the midpoint. Pass ``coded=True`` to get the raw
coded matrix instead (the natural input to the analysis routines in :mod:`mixle.doe.analysis`).

``bounds`` is a hard requirement only for the *factorial cube* itself: :func:`central_composite`'s
rotatable/orthogonal axial points are placed by a design-theory property, not clamped to ``bounds``,
and routinely extend past them on purpose -- see its docstring and :func:`central_composite_point_kinds`.
:func:`fractional_factorial`'s generator-string aliasing is likewise published, not just used
internally, via :func:`generator_alias_structure`.
"""

from __future__ import annotations

import numpy as np

from mixle.doe.designs import Bounds, _as_bounds, _require_exact_positive_int, _scale_coded

# Standard cyclic generating rows for the non-power-of-two Plackett-Burman designs (length N-1).
_PB_GEN: dict[int, str] = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _coded_to_bounds(coded: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map a coded design (factors centred on 0) into ``bounds``: ``-1`` -> low, ``+1`` -> high."""
    return _scale_coded(coded, b)


def _two_level_full(k: int) -> np.ndarray:
    """Full two-level factorial in coded ``+/-1`` units: ``(2**k, k)``, first factor varying fastest."""
    cols = []
    for i in range(k):
        block = 2**i
        pattern = np.concatenate([np.full(block, -1.0), np.full(block, 1.0)])
        cols.append(np.tile(pattern, 2 ** (k - i - 1)))
    return np.stack(cols, axis=1)


def _split_generator_tokens(generators: str | list) -> list[str]:
    """Split a ``generators`` spec into its individual token strings."""
    return generators.split() if isinstance(generators, str) else [str(t) for t in generators]


def _reduce_generator_token(token: str, letters: list[str], *, label: str) -> tuple[int, frozenset[str]]:
    """Parse one fractional-factorial generator token into a sign and its base-letter word.

    In ``+/-1`` coding, a letter multiplied by itself is the all-ones column, so repeated letters
    cancel *in pairs* -- e.g. ``"aabc"`` reduces to the same column as ``"bc"``, and ``"aabb"`` reduces
    to the empty word (every letter cancels, leaving a column that is constant across every run).
    Raises ``ValueError`` if the token references a letter outside ``letters``, or reduces to the empty
    word -- a generator that, after cancellation, defines a factor that never varies (MXR-080-0179).
    """
    sign = -1 if token.startswith("-") else 1
    name = token.lstrip("+-")
    if not name or any(ch not in letters for ch in name):
        raise ValueError(f"{label} {token!r} references an undefined base factor.")
    reduced: set[str] = set()
    for ch in name:
        reduced.symmetric_difference_update({ch})
    if not reduced:
        raise ValueError(
            f"{label} {token!r} has every base letter cancel in pairs, which collapses to a constant "
            "(all +1 or all -1) column -- a factor generator must leave an odd number of at least one "
            "base letter."
        )
    return sign, frozenset(reduced)


def fractional_factorial(bounds: Bounds, generators: str | list, *, coded: bool = False) -> np.ndarray:
    """Two-level fractional factorial ``2**(k-p)`` from pyDOE-style generator strings.

    ``generators`` names one column per factor as a product of base factors, e.g. ``"a b c ab ac"`` --
    a ``2**(5-2)`` design whose factors ``d, e`` are deliberately aliased as ``d = ab``, ``e = ac``.
    Base factors are the distinct single letters; each token is the elementwise product of its letters,
    optionally negated with a leading ``-``. The number of tokens must equal ``len(bounds)``.

    Rejects a ``generators`` spec that collapses a factor to a constant column (repeated letters
    cancelling entirely, e.g. ``"aa"``) or that aliases two of the *named* factors with each other
    (two tokens whose reduced letter-words are identical or exact opposites, e.g. tokens ``"a"`` and
    ``"-a"``) -- both silently produced a degenerate design with no diagnostic before (MXR-080-0179).
    This does not affect the intentional aliasing a fractional design is built on: a compound token
    like ``"ab"`` is fine and expected, it just cannot repeat another factor's own word. Use
    :func:`generator_alias_structure` to see which factors are aliased with which interaction.

    Returns a ``(2**k, d)`` design (``k`` = number of base factors) mapped into ``bounds`` -- or the
    raw coded ``+/-1`` matrix if ``coded=True``.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    tokens = _split_generator_tokens(generators)
    if len(tokens) != d:
        raise ValueError("generators must name exactly one column per dimension (len(bounds)).")
    letters = sorted({ch for t in tokens for ch in t if ch.isalpha()})
    if not letters:
        raise ValueError("generators must reference at least one base-factor letter.")
    base = _two_level_full(len(letters))
    col_of = {letters[i]: base[:, i] for i in range(len(letters))}

    seen: dict[frozenset[str], int] = {}
    out = np.empty((base.shape[0], d), dtype=np.float64)
    for j, token in enumerate(tokens):
        sign, reduced = _reduce_generator_token(token, letters, label="generator token")
        if reduced in seen:
            raise ValueError(
                f"generator tokens {tokens[seen[reduced]]!r} (factor {seen[reduced]}) and {token!r} "
                f"(factor {j}) define the same or exactly opposite column -- those two factors would "
                "be perfectly aliased with each other, not merely with an interaction; use distinct "
                "generators per factor."
            )
        seen[reduced] = j
        col = np.ones(base.shape[0])
        for ch in reduced:
            col = col * col_of[ch]
        out[:, j] = -col if sign < 0 else col
    return out if coded else _coded_to_bounds(out, b)


def _alias_word_record(sign: int, factors: frozenset[int]) -> dict[str, object]:
    """Return one JSON-friendly signed factorial word using public positional factor labels."""
    return {"sign": sign, "factors": [f"x{index}" for index in sorted(factors)]}


def _alias_word_label(factors: frozenset[int]) -> str:
    """Canonical public name for an identity, main-effect, or interaction word."""
    return "I" if not factors else ":".join(f"x{index}" for index in sorted(factors))


def generator_alias_structure(generators: str | list, *, max_order: int = 2) -> dict[str, object]:
    """Return signed defining words and complete alias chains through ``max_order``.

    The result is a JSON-friendly record with:

    * ``factor_labels`` mapping every public positional factor name to its source token;
    * ``base_factors`` mapping each latent generator letter to its named factor and sign;
    * ``generators`` expressing every compound generator using those public factor names;
    * ``defining_relations`` and the complete generated ``defining_group`` as signed words; and
    * ``alias_sets`` for the identity and every effect through the requested interaction order.

    A word record has an integer ``sign`` (``+1`` or ``-1``) and a ``factors`` list. For example,
    ``"a b ab"`` defines ``I = +x0:x1:x2``; through order two, ``x0`` is aliased with
    ``+x1:x2`` and ``x2`` with ``+x0:x1``. Negated base or compound tokens propagate their signs
    through every relation and alias. This supplies the actual confounding structure rather than only
    repeating the compound generator fragments (MXR-080-1477).

    Every latent base letter must have one named singleton token so aliases can be mapped
    unambiguously to the positional ``xN`` labels used by :mod:`mixle.doe.analysis`. The generator
    syntax is otherwise validated exactly like :func:`fractional_factorial`.
    """
    order_limit = _require_exact_positive_int(max_order, "max_order", minimum=0)
    tokens = _split_generator_tokens(generators)
    letters = sorted({ch for t in tokens for ch in t if ch.isalpha()})
    if not letters:
        raise ValueError("generators must reference at least one base-factor letter.")
    seen: dict[frozenset[str], int] = {}
    parsed: list[tuple[int, frozenset[str]]] = []
    for j, token in enumerate(tokens):
        sign, reduced = _reduce_generator_token(token, letters, label="generator token")
        if reduced in seen:
            raise ValueError(
                f"generator tokens {tokens[seen[reduced]]!r} (factor {seen[reduced]}) and {token!r} "
                f"(factor {j}) define the same or exactly opposite column -- those two factors would "
                "be perfectly aliased with each other, not merely with an interaction; use distinct "
                "generators per factor."
            )
        seen[reduced] = j
        parsed.append((sign, reduced))

    base_factors: dict[str, tuple[int, int]] = {}
    for index, (sign, reduced) in enumerate(parsed):
        if len(reduced) == 1:
            base_factors[next(iter(reduced))] = (index, sign)
    missing = [letter for letter in letters if letter not in base_factors]
    if missing:
        raise ValueError(
            "every base generator letter must have a named singleton factor for unambiguous aliases; "
            f"missing singleton token(s) for {', '.join(missing)}."
        )

    generator_records: dict[str, dict[str, object]] = {}
    defining_words: list[tuple[int, frozenset[int]]] = []
    for index, (token_sign, reduced) in enumerate(parsed):
        if len(reduced) == 1:
            continue
        effective_sign = token_sign
        base_indices: set[int] = set()
        for letter in reduced:
            base_index, base_sign = base_factors[letter]
            effective_sign *= base_sign
            base_indices.add(base_index)
        generator_records[f"x{index}"] = _alias_word_record(effective_sign, frozenset(base_indices))
        defining_words.append((effective_sign, frozenset({index, *base_indices})))

    defining_group: set[tuple[int, frozenset[int]]] = {(1, frozenset())}
    for relation in defining_words:
        relation_sign, relation_factors = relation
        products = {
            (sign * relation_sign, factors.symmetric_difference(relation_factors))
            for sign, factors in defining_group
        }
        defining_group.update(products)
    ordered_group = sorted(defining_group, key=lambda word: (len(word[1]), tuple(sorted(word[1])), word[0]))

    from itertools import combinations

    alias_sets: dict[str, list[dict[str, object]]] = {}
    for effect_order in range(order_limit + 1):
        for factor_tuple in combinations(range(len(tokens)), effect_order):
            effect = frozenset(factor_tuple)
            aliases = {
                (sign, effect.symmetric_difference(relation_factors))
                for sign, relation_factors in defining_group
                if len(effect.symmetric_difference(relation_factors)) <= order_limit
            }
            ordered_aliases = sorted(aliases, key=lambda word: (len(word[1]), tuple(sorted(word[1])), word[0]))
            alias_sets[_alias_word_label(effect)] = [
                _alias_word_record(sign, factors) for sign, factors in ordered_aliases
            ]

    return {
        "max_order": order_limit,
        "factor_labels": {f"x{index}": token for index, token in enumerate(tokens)},
        "base_factors": {
            letter: {"factor": f"x{index}", "sign": sign}
            for letter, (index, sign) in sorted(base_factors.items())
        },
        "generators": generator_records,
        "defining_relations": [_alias_word_record(sign, factors) for sign, factors in defining_words],
        "defining_group": [_alias_word_record(sign, factors) for sign, factors in ordered_group],
        "alias_sets": alias_sets,
    }


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

    A CCD stacks three parts, in this row order (see :func:`central_composite_point_kinds` to recover
    which output row is which): the ``2**d`` two-level factorial corners (estimate linear and
    interaction terms), ``2*d`` axial / star points at distance ``alpha`` on each axis (estimate the
    pure-quadratic curvature), and ``center`` replicates at the centre (estimate pure error and
    curvature). ``alpha`` sets the axial distance:

      * ``"rotatable"`` (default) -- ``alpha = (2**d)**0.25``, so prediction variance depends only on
        distance from the centre;
      * ``"orthogonal"`` -- the value making the second-order terms orthogonal (depends on ``center``);
      * ``"face"`` (a face-centred CCD / CCF) -- ``alpha = 1``, keeping every run inside the cube;
      * a positive, finite float -- used directly.

    ``bounds`` defines the *factorial cube* (``-1``/``+1`` map to each factor's ``low``/``high``): the
    factorial-corner and centre rows are therefore always exactly within ``bounds``. Axial points sit
    at coded distance ``alpha`` on a single axis, and for ``"rotatable"``/``"orthogonal"`` that distance
    is normally **greater than 1** -- a mathematical requirement of the rotatability/orthogonality
    property, not a clamp to ``bounds`` -- so those rows routinely fall *outside* ``bounds`` on purpose.
    Only ``alpha="face"`` (or a numeric ``alpha <= 1``) guarantees every row stays within ``bounds``
    (MXR-080-0179); use :func:`central_composite_point_kinds` to tell the row kinds apart
    programmatically when that distinction matters to the caller.

    Returns ``(2**d + 2*d + center, d)`` rows mapped into ``bounds`` (or coded if ``coded=True``).
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    fact = _two_level_full(d)
    nf = fact.shape[0]
    nc = _require_exact_positive_int(center, "center", minimum=0)
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
        if isinstance(alpha, (bool, np.bool_)):
            raise TypeError("numeric alpha must be a real number, not bool.")
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("numeric alpha must be finite and positive.")
    axial = np.zeros((2 * d, d))
    for i in range(d):
        axial[2 * i, i] = -a
        axial[2 * i + 1, i] = a
    coded_design = np.vstack([fact, axial, np.zeros((nc, d))])
    return coded_design if coded else _coded_to_bounds(coded_design, b)


def central_composite_point_kinds(bounds: Bounds, *, center: int = 4) -> np.ndarray:
    """Row-kind labels for :func:`central_composite`'s output, in the same row order.

    Returns a length-``(2**d + 2*d + center)`` string array of ``"factorial"`` / ``"axial"`` /
    ``"center"`` entries. ``"factorial"`` and ``"center"`` rows are always exactly within ``bounds``;
    ``"axial"`` rows are guaranteed within ``bounds`` only when :func:`central_composite` was called
    with ``alpha="face"`` (or a numeric ``alpha <= 1``) -- see its docstring for why rotatable/
    orthogonal axial points routinely extend past ``bounds`` on purpose (MXR-080-0179). Pass the same
    ``center`` used in the matching :func:`central_composite` call so the lengths line up, e.g.::

        design = central_composite(bounds, center=4, alpha="rotatable")
        kinds = central_composite_point_kinds(bounds, center=4)
        in_bounds_rows = design[kinds != "axial"]
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    nc = _require_exact_positive_int(center, "center", minimum=0)
    nf = 2**d
    return np.array(["factorial"] * nf + ["axial"] * (2 * d) + ["center"] * nc)


def box_behnken(bounds: Bounds, *, center: int | None = None, coded: bool = False) -> np.ndarray:
    """Box-Behnken response-surface design (3 levels per factor, no corner runs).

    For every pair of factors it runs the four ``(+/-1, +/-1)`` combinations with all other factors at
    the centre, plus ``center`` centre replicates. Unlike a CCD it never sets all factors to an extreme
    at once (no cube corners), which is useful when those combinations are expensive or infeasible, and
    it needs only three levels per factor. Every row is always within ``bounds`` (no analogue of a CCD's
    rotatable axial points here). Requires ``d >= 3``.

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
    nc = _BB_CENTERS.get(d, 3) if center is None else _require_exact_positive_int(center, "center", minimum=0)
    coded_design = np.vstack([*blocks, np.zeros((nc, d))])
    return coded_design if coded else _coded_to_bounds(coded_design, b)
