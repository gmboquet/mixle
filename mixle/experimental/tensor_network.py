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

import numpy as np


@dataclass
class MPS:
    """A Born-machine matrix-product state: one ``(D_left, d, D_right)`` tensor per site."""

    tensors: list[np.ndarray]

    @property
    def length(self) -> int:
        return len(self.tensors)

    @property
    def d(self) -> int:
        return self.tensors[0].shape[1]

    def amplitude(self, x) -> float:
        m = self.tensors[0][:, x[0], :]
        for i in range(1, self.length):
            m = m @ self.tensors[i][:, x[i], :]
        return float(m[0, 0])

    def _contract(self, fixed: dict[int, int]) -> float:
        """Unnormalized ``sum_{free} psi(x)^2`` with sites in ``fixed`` clamped (transfer operator)."""
        env = np.ones((1, 1))
        for i, a in enumerate(self.tensors):
            vals = [fixed[i]] if i in fixed else range(self.d)
            new = np.zeros((a.shape[2], a.shape[2]))
            for xv in vals:
                ax = a[:, xv, :]
                new += ax.T @ env @ ax
            env = new
        return float(env[0, 0])

    def normalization(self) -> float:
        return self._contract({})

    def probability(self, x) -> float:
        return self._contract({i: int(v) for i, v in enumerate(x)}) / self.normalization()

    def marginal(self, evidence: dict[int, int]) -> float:
        """``p(X_evidence = values)`` -- exact, by contraction over the free sites."""
        return self._contract({int(k): int(v) for k, v in evidence.items()}) / self.normalization()

    def conditional(self, query: int, evidence: dict[int, int]) -> np.ndarray:
        """Exact ``p(X_query = . | evidence)`` as a length-``d`` vector."""
        ev = {int(k): int(v) for k, v in evidence.items()}
        joint = np.array([self._contract({**ev, query: v}) for v in range(self.d)])
        denom = self._contract(ev)
        return joint / denom

    def all_probabilities(self) -> np.ndarray:
        """Brute-force ``p(x)`` over all ``d**L`` sequences (for verification / entanglement)."""
        amps = np.array([self.amplitude(x) for x in itertools.product(range(self.d), repeat=self.length)])
        w = amps**2
        return w / w.sum()


def random_mps(length: int, bond: int, *, d: int = 2, seed: int = 0) -> MPS:
    """A random Born-machine MPS with open boundaries and interior bond dimension ``bond``."""
    rng = np.random.default_rng(seed)
    dims = [1] + [bond] * (length - 1) + [1]
    tensors = [rng.standard_normal((dims[i], d, dims[i + 1])) for i in range(length)]
    return MPS(tensors)


def product_mps(site_amplitudes: list[np.ndarray]) -> MPS:
    """A bond-1 product state from per-site amplitude vectors (zero entanglement)."""
    return MPS([np.asarray(a, dtype=float).reshape(1, -1, 1) for a in site_amplitudes])


def schmidt_values(mps: MPS, cut: int) -> np.ndarray:
    """Normalized Schmidt coefficients of the wavefunction at the bond after site ``cut-1``."""
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
    d, length = mps.d, mps.length
    amps = np.array([mps.amplitude(x) for x in itertools.product(range(d), repeat=length)])
    psi = amps.reshape(d**cut, d ** (length - cut))
    u, s, vt = np.linalg.svd(psi, full_matrices=False)
    total = float(np.sum(s**2))
    discarded = float(np.sum(s[chi:] ** 2) / total) if total > 0 else 0.0

    s_trunc = s.copy()
    s_trunc[chi:] = 0.0
    psi_trunc = (u * s_trunc) @ vt
    p_full = amps**2 / np.sum(amps**2)
    w_trunc = psi_trunc.reshape(-1) ** 2
    p_trunc = w_trunc / w_trunc.sum()
    tv = 0.5 * float(np.sum(np.abs(p_full - p_trunc)))
    return discarded, tv
