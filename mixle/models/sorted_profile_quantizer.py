"""Sorted-profile (permutation x monotone) quantizer (roadmap G4): head-exact + parametric-tail per tensor.

Per the R1 copula note (roadmap doc, R1 -> G4, F6, I2, H4), any flattened tensor value vector ``v``
decomposes as ``v = P . s``: a sorted **profile** ``s`` (the empirical quantile function) composed with a
**permutation** ``P`` (the arrangement mapping sorted rank back to original position). H4's
``mixle/experimental/tying_discovery.py`` (``tensor_profile`` / ``profile_distance``, see that module's
docstring) already uses the marginal half of this decomposition -- a fixed-length RESAMPLED profile -- as a
tying-discovery signal, and deliberately throws the permutation away. G4 keeps BOTH halves and turns the
decomposition into an actual per-tensor storage format:

* ``s`` is not stored as a raw sorted array -- it is FIT as a parametric mixle distribution (reusing this
  codebase's real ``mixle.stats``/``mixle.inference.estimate`` machinery, not a hand-rolled curve fit), so
  the non-outlier bulk of the tensor collapses to a handful of distribution parameters instead of one float
  per element;
* ``P`` is stored as literal permutation indices (an integer array) -- per the R1 note's honest
  acknowledgment that "arbitrary permutations are gather ops": there is no closed-form compact encoding of
  an arbitrary permutation short of ``n*log2(n)`` bits, so this module does not pretend otherwise. The sort
  itself is an exact, free (deterministic, non-iterative) operation -- unlike G2's
  :func:`mixle.models.sigma_weighted_projection.sigma_weighted_permutation`, no Sinkhorn/OT solver is
  needed here, because there is nothing to OPTIMIZE: sorting a tensor's own values against itself has one
  unambiguous answer. (G2's Sinkhorn permutation solver is for the DIFFERENT problem of matching one
  tensor's rows to another's under a Sigma-weighted cost -- not reused here.)
* the head (top-``k`` largest-magnitude values) is carved out and stored EXACTLY before any of the above,
  because outliers are exactly where a smooth parametric quantile fit is worst -- this is the "head-exact"
  half of "head-exact + parametric-tail";
* a per-tensor goodness-of-fit RECEIPT (a real, computed Kolmogorov-Smirnov statistic, reusing
  :func:`mixle.utils.evaluation.ks_test` rather than a hand-rolled discrepancy measure) is attached to every
  encoding, and a bad receipt triggers a DENSE FALLBACK rather than silently accepting a bad lossy fit.

Honest scope (do not read this module as a general weight quantizer): the roadmap doc scopes G4 to exactly
three use cases --

1. optimizer states (F6) -- e.g. Adam's second-moment buffer, which is positive, heavy-tailed, and mostly
   smooth (a good match for a Gamma/log-normal-family tail fit); this module builds the mechanism generically
   enough to apply there without F6 itself existing yet;
2. KV-cache tails (E2/I2) -- same story, not built here;
3. anomaly detection (:func:`detect_anomaly`) -- the goodness-of-fit receipt IS the anomaly signal: a tensor
   that suddenly stops matching its own historical value-profile family is itself worth flagging.

Hardware reality (R1): arbitrary permutations are memory-bound gather ops with no FLOP savings, so this
scheme is honestly a STORAGE/regularization/receipt-structure win (real when the permutation indices fit in
fewer bits than the values they replace -- e.g. ``uint16`` indices against ``float32`` values for tensors
under 65536 elements) rather than a speed win, unless restricted to block forms that map to tensor cores
(not attempted here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference import estimate
from mixle.stats import GaussianEstimator
from mixle.utils.evaluation import ks_test

__all__ = [
    "SortedProfileEncoding",
    "AnomalyReport",
    "fit_sorted_profile",
    "reconstruct",
    "detect_anomaly",
]

# Default dense-fallback trigger: the Kolmogorov-Smirnov D-statistic between the fitted tail distribution
# and the actual non-outlier values. D is bounded in [0, 1] and, for a WELL-SPECIFIED family, shrinks
# towards 0 as the sample grows (D ~ O(1/sqrt(n)) for a true fit); for a genuinely mismatched family
# (multi-modal data against a unimodal fit, say) D does NOT shrink with n -- it sits at a roughly constant,
# much larger bias. 0.05 is comfortably above the finite-sample noise floor for tensors of a few thousand
# elements or more (see the module tests for measured D values on both sides of this line) while still
# comfortably below the D observed for genuinely bad fits.
DEFAULT_GOF_THRESHOLD = 0.05

# Default anomaly-detection margin: how much WORSE (in absolute KS-D terms, relative to the reference
# encoding's own receipt) a new tensor's fit against the reference's tail-distribution FAMILY has to get
# before it is flagged. A pure ratio threshold breaks down when the reference D is already tiny (any noise
# doubles it), so this combines a relative ratio with an absolute floor -- see :func:`detect_anomaly`.
DEFAULT_ANOMALY_RATIO = 2.0
DEFAULT_ANOMALY_ABS_MARGIN = 0.02

# Conservative fixed per-distribution parameter-storage budget (bytes) used by
# `SortedProfileEncoding.nbytes`: every tail family used here (Gaussian, Gamma, Student-t, ...) is
# parameterized by a small, fixed number of scalars (2-3 floats) plus a family tag; rather than fragile
# introspection across distribution classes (which do not share a uniform `get_parameters()` contract --
# e.g. `GaussianDistribution` exposes `.mu`/`.sigma2` directly, `GammaDistribution` exposes
# `get_parameters()`), we budget a flat, deliberately generous constant here. This is negligible relative
# to tensor sizes this module targets (thousands+ elements), so precision here does not matter to the
# measured compression ratio.
_DISTRIBUTION_PARAM_BYTES = 64


def _index_dtype(n: int) -> np.dtype:
    """Smallest unsigned integer dtype that can address ``n`` distinct positions -- the honest per-index
    storage cost of a literal permutation array (R1: "arbitrary permutations are gather ops", stored as
    literal indices, ``n*log2(n)`` bits in the worst case; we round up to the nearest whole byte width
    numpy actually offers rather than hand-rolling bit-packing).
    """
    if n <= 0:
        return np.dtype(np.uint8)
    if n <= 2**8:
        return np.dtype(np.uint8)
    if n <= 2**16:
        return np.dtype(np.uint16)
    if n <= 2**32:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


@dataclass
class SortedProfileEncoding:
    """Storage format for one tensor's sorted-profile (permutation x monotone) encoding.

    Either the ``used_dense_fallback=False`` branch (``top_k_*`` / ``tail_distribution`` /
    ``permutation_indices`` populated, ``dense_values=None``) or the ``used_dense_fallback=True`` branch
    (``dense_values`` populated, the rest ``None``/empty) is populated -- never both -- so
    :func:`reconstruct` can dispatch on the flag alone.

    Attributes:
        shape (tuple[int, ...]): Original tensor shape (reconstruction reshapes back to this).
        top_k_values (np.ndarray | None): Exact values of the top-``k`` largest-magnitude entries
            ("head-exact"). ``None``/empty when ``used_dense_fallback``.
        top_k_indices (np.ndarray | None): Flat indices (into the original tensor, C order) the
            ``top_k_values`` came from.
        tail_distribution (Any | None): A fitted ``mixle.stats`` distribution object (exposing ``.cdf`` and
            ``.quantile``) over the non-outlier ("tail") values -- the parametric replacement for storing
            those values directly.
        permutation_indices (np.ndarray | None): Length-``n_tail`` array of flat original indices, ordered
            so that ``permutation_indices[r]`` is where the ``r``-th smallest non-outlier value belongs.
            This IS the permutation ``P`` in ``v = P . s``.
        goodness_of_fit (float): KS D-statistic between the fitted ``tail_distribution`` and the actual
            non-outlier values (0 = perfect fit; see :data:`DEFAULT_GOF_THRESHOLD`). Set even when
            ``used_dense_fallback`` (it is the receipt that CAUSED the fallback), so the receipt itself is
            never silently thrown away.
        used_dense_fallback (bool): True if the fit was rejected and the tensor is stored densely instead.
        dense_values (np.ndarray | None): The full flattened tensor, only populated when
            ``used_dense_fallback``.
        n_tail (int): Number of non-outlier elements (``= permutation_indices.size`` in the non-fallback
            case; kept explicitly so ``nbytes``/receipts are meaningful in the fallback case too).
    """

    shape: tuple
    top_k_values: np.ndarray | None
    top_k_indices: np.ndarray | None
    tail_distribution: Any | None
    permutation_indices: np.ndarray | None
    goodness_of_fit: float
    used_dense_fallback: bool
    dense_values: np.ndarray | None = None
    n_tail: int = 0
    _index_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint32), repr=False)

    @property
    def size(self) -> int:
        """Total element count of the original tensor."""
        return int(np.prod(self.shape)) if len(self.shape) else 1

    def nbytes(self) -> int:
        """Measured storage footprint of the encoding, in bytes.

        Dense fallback: exactly the byte count of ``dense_values`` (float32). Otherwise: top-k exact values
        (float32) + top-k indices (minimal dtype) + permutation indices (minimal dtype) +
        :data:`_DISTRIBUTION_PARAM_BYTES` for the fitted tail distribution + one float32 for the
        goodness-of-fit receipt itself (a real, non-decorative receipt is part of what is shipped).
        """
        if self.used_dense_fallback:
            return int(self.dense_values.astype(np.float32).nbytes)
        top_k_n = 0 if self.top_k_values is None else self.top_k_values.size
        idx_dtype_bytes = self._index_dtype.itemsize
        return (
            top_k_n * np.dtype(np.float32).itemsize  # top_k_values
            + top_k_n * idx_dtype_bytes  # top_k_indices
            + self.n_tail * idx_dtype_bytes  # permutation_indices
            + _DISTRIBUTION_PARAM_BYTES  # tail_distribution
            + np.dtype(np.float32).itemsize  # goodness_of_fit receipt
        )


def _as_flat_numpy(tensor: Any) -> np.ndarray:
    """Flatten ``tensor`` (numpy array or torch tensor) to a 1-D float64 numpy array."""
    if hasattr(tensor, "detach"):  # torch.Tensor
        flat = tensor.detach().cpu().numpy()
    else:
        flat = np.asarray(tensor)
    return flat.reshape(-1).astype(np.float64)


def fit_sorted_profile(
    tensor: Any,
    top_k: int = 0,
    tail_family: Any = None,
    gof_threshold: float = DEFAULT_GOF_THRESHOLD,
) -> SortedProfileEncoding:
    """Encode ``tensor`` as head-exact outliers + a fitted parametric tail distribution + permutation.

    Args:
        tensor: A torch tensor or numpy array of any shape.
        top_k (int): Number of largest-magnitude entries to carve out and store EXACTLY ("head-exact"),
            before any fitting happens -- outliers are exactly where a smooth parametric quantile fit is
            worst, so they are never asked to survive the parametric tail model. 0 disables head-exact
            storage entirely (the whole tensor goes through the tail fit).
        tail_family: A ``mixle.stats`` ``ParameterEstimator`` instance (e.g. ``GaussianEstimator()``,
            ``GammaEstimator()``) used to fit the non-outlier values via ``mixle.inference.estimate``.
            Defaults to ``GaussianEstimator()``. Pick a family whose support matches the tensor's actual
            values -- e.g. ``GammaEstimator()`` for a strictly-positive optimizer second-moment buffer,
            per F6's honest scope note (see module docstring); a mismatched family is not silently
            accepted -- it is caught by the goodness-of-fit receipt below and triggers the dense fallback.
        gof_threshold (float): Maximum acceptable KS D-statistic (see :data:`DEFAULT_GOF_THRESHOLD`) before
            falling back to dense storage.

    Returns:
        SortedProfileEncoding: either a populated head/tail/permutation encoding
        (``used_dense_fallback=False``) or a dense fallback (``used_dense_fallback=True``), always carrying
        the real, computed ``goodness_of_fit`` receipt either way.
    """
    if tail_family is None:
        tail_family = GaussianEstimator()

    flat = _as_flat_numpy(tensor)
    n = flat.size
    if n == 0:
        raise ValueError("fit_sorted_profile requires a non-empty tensor")
    shape = tuple(tensor.shape) if hasattr(tensor, "shape") else (n,)
    top_k = int(max(0, min(top_k, n)))
    idx_dtype = _index_dtype(n)

    if top_k > 0:
        abs_vals = np.abs(flat)
        top_k_indices = np.argpartition(-abs_vals, top_k - 1)[:top_k]
        top_k_indices = top_k_indices[np.argsort(-abs_vals[top_k_indices])]
    else:
        top_k_indices = np.array([], dtype=np.int64)

    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[top_k_indices] = True
    tail_original_indices = np.nonzero(~outlier_mask)[0]
    tail_values = flat[tail_original_indices]
    n_tail = tail_values.size

    if n_tail < 2:
        # Nothing left to fit a distribution to -- dense fallback is the only honest option.
        return SortedProfileEncoding(
            shape=shape,
            top_k_values=None,
            top_k_indices=None,
            tail_distribution=None,
            permutation_indices=None,
            goodness_of_fit=float("inf"),
            used_dense_fallback=True,
            dense_values=flat.astype(np.float32),
            n_tail=0,
            _index_dtype=idx_dtype,
        )

    tail_distribution = estimate(list(tail_values), tail_family)
    d_stat, _p_value = ks_test(tail_values, tail_distribution)

    order = np.argsort(tail_values)  # ascending: sorted_tail[r] = tail_values[order[r]]
    permutation_indices = tail_original_indices[order].astype(idx_dtype)

    used_dense_fallback = d_stat > gof_threshold
    if used_dense_fallback:
        return SortedProfileEncoding(
            shape=shape,
            top_k_values=None,
            top_k_indices=None,
            tail_distribution=None,
            permutation_indices=None,
            goodness_of_fit=d_stat,
            used_dense_fallback=True,
            dense_values=flat.astype(np.float32),
            n_tail=0,
            _index_dtype=idx_dtype,
        )

    return SortedProfileEncoding(
        shape=shape,
        top_k_values=flat[top_k_indices].astype(np.float32) if top_k > 0 else np.array([], dtype=np.float32),
        top_k_indices=top_k_indices.astype(idx_dtype) if top_k > 0 else np.array([], dtype=idx_dtype),
        tail_distribution=tail_distribution,
        permutation_indices=permutation_indices,
        goodness_of_fit=d_stat,
        used_dense_fallback=False,
        dense_values=None,
        n_tail=n_tail,
        _index_dtype=idx_dtype,
    )


def reconstruct(encoding: SortedProfileEncoding) -> np.ndarray:
    """Invert a :class:`SortedProfileEncoding` back to an (approximate, or dense-exact) tensor.

    The head (top-k outliers) is EXACT in both branches (either stored verbatim, or -- in the dense
    fallback case -- simply part of the densely-stored tensor). The tail is exact under dense fallback and
    approximate (reconstructed from the fitted parametric quantile function) otherwise.

    Returns:
        np.ndarray: float32 array reshaped to ``encoding.shape``.
    """
    if encoding.used_dense_fallback:
        return encoding.dense_values.reshape(encoding.shape)

    n = encoding.size
    out = np.zeros(n, dtype=np.float64)

    n_tail = encoding.n_tail
    # Reconstruct the sorted tail profile from the fitted quantile function at the midpoint of each rank's
    # probability mass -- the standard "plotting position" for turning n ranks into n quantile queries.
    ranks = (np.arange(n_tail, dtype=np.float64) + 0.5) / n_tail
    sorted_tail_hat = np.array([encoding.tail_distribution.quantile(float(q)) for q in ranks])
    out[encoding.permutation_indices.astype(np.int64)] = sorted_tail_hat

    if encoding.top_k_values is not None and encoding.top_k_values.size > 0:
        out[encoding.top_k_indices.astype(np.int64)] = encoding.top_k_values

    return out.astype(np.float32).reshape(encoding.shape)


@dataclass(frozen=True)
class AnomalyReport:
    """Result of scoring a new tensor against a reference encoding's tail-distribution family.

    Attributes:
        ks_statistic (float): KS D-statistic of the new tensor's non-outlier values against the
            REFERENCE encoding's fitted ``tail_distribution`` (the family is held fixed; only the data
            changes -- this is a re-SCORING, not a re-fit).
        reference_goodness_of_fit (float): The reference encoding's own receipt, for context.
        is_anomaly (bool): Whether ``ks_statistic`` has degraded significantly relative to
            ``reference_goodness_of_fit`` (see :func:`detect_anomaly` for the exact rule).
    """

    ks_statistic: float
    reference_goodness_of_fit: float
    is_anomaly: bool


def detect_anomaly(
    tensor: Any,
    reference_encoding: SortedProfileEncoding,
    ratio_threshold: float = DEFAULT_ANOMALY_RATIO,
    abs_margin: float = DEFAULT_ANOMALY_ABS_MARGIN,
) -> AnomalyReport:
    """Anomaly-detection use of the goodness-of-fit receipt (roadmap G4, use case 3).

    A tensor that historically fit ``reference_encoding.tail_distribution``'s family well and suddenly stops
    fitting it -- a burst of extreme values, a distribution shift -- is itself an anomaly signal, independent
    of whatever downstream task the tensor feeds. This function re-SCORES ``tensor`` against the reference's
    ALREADY-FITTED family (it does not fit a new distribution to ``tensor``), then compares the resulting
    KS D-statistic to the reference's own receipt.

    The new tensor's outliers are excluded using the reference encoding's own top-k COUNT (not its specific
    indices, which belong to a different tensor) so the comparison is apples-to-apples with how the reference
    receipt itself was computed.

    Flagging rule: ``is_anomaly`` fires when the new D-statistic exceeds
    ``max(ratio_threshold * reference_goodness_of_fit, reference_goodness_of_fit + abs_margin)`` -- a ratio
    threshold alone breaks down when the reference D is already tiny (sampling noise alone can double it), so
    it is combined with an absolute floor. Both directions are meaningful test cases: a similarly-distributed
    new draw should score close to (or even below) the reference's own receipt; a genuinely shifted or
    outlier-contaminated tensor should score well past the combined threshold.

    Returns:
        AnomalyReport
    """
    if reference_encoding.used_dense_fallback:
        raise ValueError(
            "detect_anomaly requires a reference_encoding with a fitted tail_distribution "
            "(reference_encoding.used_dense_fallback was True, so there is no fitted family to score against)"
        )

    flat = _as_flat_numpy(tensor)
    n = flat.size
    if n == 0:
        raise ValueError("detect_anomaly requires a non-empty tensor")

    top_k_ref = reference_encoding.top_k_values.size if reference_encoding.top_k_values is not None else 0
    top_k = int(max(0, min(top_k_ref, n - 2)))  # keep >= 2 non-outlier values to score against

    if top_k > 0:
        abs_vals = np.abs(flat)
        outlier_indices = np.argpartition(-abs_vals, top_k - 1)[:top_k]
        outlier_mask = np.zeros(n, dtype=bool)
        outlier_mask[outlier_indices] = True
        tail_values = flat[~outlier_mask]
    else:
        tail_values = flat

    d_stat, _p_value = ks_test(tail_values, reference_encoding.tail_distribution)
    ref_d = reference_encoding.goodness_of_fit
    threshold = max(ratio_threshold * ref_d, ref_d + abs_margin)
    is_anomaly = d_stat > threshold

    return AnomalyReport(
        ks_statistic=d_stat,
        reference_goodness_of_fit=ref_d,
        is_anomaly=is_anomaly,
    )
