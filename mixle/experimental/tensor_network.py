"""P4 (experimental) -- tensor-network (matrix-product-state) leaves with exact conditioning.

A matrix-product state (MPS / tensor-train) is a density model over discrete sequences with a
built-in complexity dial -- the **bond dimension** -- and two properties classical mixtures lack:
its marginals and conditionals are **exact by contraction** (no sampling), and its **entanglement
entropy** at any cut is a direct, measured receipt of how much long-range correlation the model
carries (bounded by ``log(bond dimension)``).

This module implements a Born-machine MPS over length-``L`` sequences of a ``d``-symbol alphabet:
``p(x) proportional to |psi(x)|^2`` where ``psi(x)`` is the contraction of one matrix per site. It
provides exact normalization / marginals / conditionals by transfer-operator contraction (verified
against brute-force enumeration), the entanglement entropy from the Schmidt spectrum, and
bond-dimension truncation whose error is exactly the discarded Schmidt weight -- the receipt the
card asks for ("truncation error tracks the entanglement receipt").

Scope: this is the exact-conditioning + entanglement-receipt core (the two verification points the
card lists first). DMRG fitting and the matched-parameter HMM comparison are follow-ups.

Exploratory ``mixle.experimental`` code (P4 card).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


@dataclass
class MPS:
    """A Born-machine matrix-product state: one ``(D_left, d, D_right)`` tensor per site."""

    tensors: list[np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, list) or not self.tensors:
            raise ValueError("tensors must be a non-empty list of rank-3 arrays.")
        validated = []
        physical_dimension = None
        previous_right = None
        for site, tensor in enumerate(self.tensors):
            array = np.asarray(tensor)
            if array.ndim != 3 or any(size <= 0 for size in array.shape):
                raise ValueError(f"tensor {site} must have non-empty (left_bond, symbol, right_bond) shape.")
            if array.dtype.kind not in "fc":
                raise TypeError(f"tensor {site} must use a real or complex floating-point dtype.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"tensor {site} must contain only finite values.")
            if site == 0 and array.shape[0] != 1:
                raise ValueError("an open-boundary MPS must have a unit left boundary bond.")
            if previous_right is not None and array.shape[0] != previous_right:
                raise ValueError(f"tensor {site} left bond does not match the preceding right bond.")
            if physical_dimension is None:
                physical_dimension = array.shape[1]
            elif array.shape[1] != physical_dimension:
                raise ValueError("all tensors must use the same physical symbol dimension.")
            previous_right = array.shape[2]
            validated.append(array)
        if validated[-1].shape[2] != 1:
            raise ValueError("an open-boundary MPS must have a unit right boundary bond.")
        self.tensors = validated

    @property
    def length(self) -> int:
        return len(self.tensors)

    @property
    def d(self) -> int:
        return self.tensors[0].shape[1]

    @staticmethod
    def _exact_integer(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an exact integer.")
        return int(value)

    def _symbol(self, value: Any, name: str) -> int:
        symbol = self._exact_integer(value, name)
        if not 0 <= symbol < self.d:
            raise ValueError(f"{name} must lie in [0, {self.d}).")
        return symbol

    def _sequence(self, x: Any) -> tuple[int, ...]:
        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError("a full MPS event must be a one-dimensional sequence.")
            values = x.tolist()
        elif isinstance(x, (list, tuple)):
            values = list(x)
        else:
            raise TypeError("a full MPS event must be a list, tuple, or one-dimensional ndarray.")
        if len(values) != self.length:
            raise ValueError(f"a full MPS event must contain exactly {self.length} symbols.")
        return tuple(self._symbol(value, f"x[{site}]") for site, value in enumerate(values))

    def _evidence(self, evidence: Any) -> dict[int, int]:
        if not isinstance(evidence, dict):
            raise TypeError("evidence must be a dictionary from site indices to symbols.")
        validated: dict[int, int] = {}
        for raw_site, raw_symbol in evidence.items():
            site = self._exact_integer(raw_site, "evidence site")
            if not 0 <= site < self.length:
                raise ValueError(f"evidence site must lie in [0, {self.length}).")
            validated[site] = self._symbol(raw_symbol, f"evidence[{site}]")
        return validated

    @staticmethod
    def _real_nonnegative(value: Any, name: str) -> float:
        scalar = np.asarray(value).item()
        real = np.real_if_close(scalar, tol=1000)
        if np.iscomplexobj(real):
            raise FloatingPointError(f"{name} has a non-negligible imaginary component.")
        result = float(real)
        if not np.isfinite(result):
            raise FloatingPointError(f"{name} is non-finite.")
        if result < -1e-12:
            raise FloatingPointError(f"{name} is negative beyond numerical tolerance.")
        return max(result, 0.0)

    def amplitude(self, x: Any) -> Any:
        event = self._sequence(x)
        m = self.tensors[0][:, event[0], :]
        for i in range(1, self.length):
            m = m @ self.tensors[i][:, event[i], :]
        return m[0, 0].item()

    def _contract(self, fixed: dict[int, int]) -> float:
        """Unnormalized ``sum_{free} |psi(x)|^2`` with fixed sites clamped."""
        dtype = np.result_type(*(tensor.dtype for tensor in self.tensors))
        env = np.ones((1, 1), dtype=dtype)
        for i, a in enumerate(self.tensors):
            vals = [fixed[i]] if i in fixed else range(self.d)
            new = np.zeros((a.shape[2], a.shape[2]), dtype=dtype)
            for xv in vals:
                ax = a[:, xv, :]
                new += ax.conj().T @ env @ ax
            env = new
        return self._real_nonnegative(env[0, 0], "Born contraction")

    def normalization(self) -> float:
        normalization = self._contract({})
        if normalization <= 0.0:
            raise ValueError("MPS has zero Born normalization.")
        return normalization

    def probability(self, x: Any) -> float:
        """Probability of one complete length-``L`` event.

        Use :meth:`marginal` for an explicitly partial event; prefixes and extra symbols are rejected.
        """
        event = self._sequence(x)
        return self._contract(dict(enumerate(event))) / self.normalization()

    def marginal(self, evidence: dict[int, int]) -> float:
        """``p(X_evidence = values)`` -- exact, by contraction over the free sites."""
        return self._contract(self._evidence(evidence)) / self.normalization()

    def conditional(self, query: int, evidence: dict[int, int]) -> np.ndarray:
        """Exact ``p(X_query = . | evidence)`` as a length-``d`` vector."""
        query = self._exact_integer(query, "query")
        if not 0 <= query < self.length:
            raise ValueError(f"query must lie in [0, {self.length}).")
        ev = self._evidence(evidence)
        denom = self._contract(ev)
        if denom <= 0.0:
            raise ValueError("cannot condition on zero-probability evidence.")
        if query in ev:
            result = np.zeros(self.d)
            result[ev[query]] = 1.0
            return result
        joint = np.array([self._contract({**ev, query: v}) for v in range(self.d)])
        result = joint / denom
        total = float(result.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("conditional probabilities failed to normalize.")
        return result / total

    def all_probabilities(self) -> np.ndarray:
        """Brute-force ``p(x)`` over all ``d**L`` sequences (for verification / entanglement)."""
        amps = np.array([self.amplitude(x) for x in itertools.product(range(self.d), repeat=self.length)])
        w = np.abs(amps) ** 2
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("MPS has zero or non-finite Born normalization.")
        return w / total


def random_mps(length: int, bond: int, *, d: int = 2, seed: int = 0) -> MPS:
    """A random Born-machine MPS with open boundaries and interior bond dimension ``bond``."""
    for name, value in {"length": length, "bond": bond, "d": d}.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive exact integer.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an exact integer.")
    length, bond, d, seed = int(length), int(bond), int(d), int(seed)
    rng = np.random.default_rng(seed)
    dims = [1] + [bond] * (length - 1) + [1]
    tensors = [rng.standard_normal((dims[i], d, dims[i + 1])) for i in range(length)]
    return MPS(tensors)


def product_mps(site_amplitudes: list[np.ndarray]) -> MPS:
    """A bond-1 product state from per-site amplitude vectors (zero entanglement)."""
    if not isinstance(site_amplitudes, list) or not site_amplitudes:
        raise ValueError("site_amplitudes must be a non-empty list.")
    arrays = [np.asarray(amplitudes) for amplitudes in site_amplitudes]
    for site, array in enumerate(arrays):
        if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iufc":
            raise ValueError(f"site amplitude {site} must be a non-empty real or complex vector.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"site amplitude {site} must contain only finite values.")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all site amplitude vectors must use the same symbol dimension.")
    dtype = np.result_type(*[array.dtype for array in arrays], np.float64)
    return MPS([array.astype(dtype, copy=False).reshape(1, -1, 1) for array in arrays])


def schmidt_values(mps: MPS, cut: int) -> np.ndarray:
    """Normalized Schmidt coefficients of the wavefunction at the bond after site ``cut-1``."""
    if not isinstance(mps, MPS):
        raise TypeError("mps must be an MPS.")
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, Integral) or not 1 <= int(cut) < mps.length:
        raise ValueError(f"cut must be an exact integer in [1, {mps.length}).")
    cut = int(cut)
    d, length = mps.d, mps.length
    amps = np.array([mps.amplitude(x) for x in itertools.product(range(d), repeat=length)])
    psi = amps.reshape(d**cut, d ** (length - cut))
    s = np.linalg.svd(psi, compute_uv=False)
    norm = np.linalg.norm(s)
    return s / norm if norm > 0 else s


def entanglement_entropy(mps: MPS, cut: int) -> float:
    """Von Neumann entanglement entropy (nats) across the cut -- the long-range-structure receipt."""
    lam = schmidt_values(mps, cut)
    p = lam**2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def truncate_error(mps: MPS, cut: int, chi: int) -> tuple[float, float]:
    """Truncate the wavefunction to bond ``chi`` at ``cut``; return (discarded_weight, tv_distance).

    ``discarded_weight`` is the Schmidt weight thrown away (Eckart-Young); ``tv_distance`` is the
    total-variation distance between the full and truncated probability distributions. As ``chi``
    shrinks the discarded weight grows and the distribution error grows with it.
    """
    if not isinstance(mps, MPS):
        raise TypeError("mps must be an MPS.")
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, Integral) or not 1 <= int(cut) < mps.length:
        raise ValueError(f"cut must be an exact integer in [1, {mps.length}).")
    if isinstance(chi, (bool, np.bool_)) or not isinstance(chi, Integral) or int(chi) <= 0:
        raise ValueError("chi must be a positive exact integer.")
    cut, chi = int(cut), int(chi)
    d, length = mps.d, mps.length
    amps = np.array([mps.amplitude(x) for x in itertools.product(range(d), repeat=length)])
    psi = amps.reshape(d**cut, d ** (length - cut))
    u, s, vt = np.linalg.svd(psi, full_matrices=False)
    total = float(np.sum(s**2))
    discarded = float(np.sum(s[chi:] ** 2) / total) if total > 0 else 0.0

    s_trunc = s.copy()
    s_trunc[chi:] = 0.0
    psi_trunc = (u * s_trunc) @ vt
    full_weight = np.abs(amps) ** 2
    full_total = float(full_weight.sum())
    truncated_weight = np.abs(psi_trunc.reshape(-1)) ** 2
    truncated_total = float(truncated_weight.sum())
    if full_total <= 0.0 or truncated_total <= 0.0:
        raise ValueError("full and truncated Born states must have positive normalization.")
    p_full = full_weight / full_total
    p_trunc = truncated_weight / truncated_total
    tv = 0.5 * float(np.sum(np.abs(p_full - p_trunc)))
    return discarded, tv
