"""Shared fail-closed contracts for continuous observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class UnscorableObservation(ValueError):
    """A record a scorer refuses because it lies outside the law's admissible observation space.

    Distinct from an ordinary ``ValueError`` so a caller can tell "this record is not scorable" --
    a fact about the data, which serving reports as an unscorable record -- from "this call is
    malformed", which is a bug. It subclasses ``ValueError`` so existing handlers keep working.
    """


def finite_observations(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Return an owned one-dimensional finite observation array within optional bounds."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a one-dimensional finite real array") from exc
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must be a one-dimensional finite real array")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{label} must be greater than or equal to {minimum!r}")
    if maximum is not None and np.any(result > maximum):
        raise ValueError(f"{label} must be less than or equal to {maximum!r}")
    return np.array(result, dtype=np.float64, copy=True)


def finite_observation(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite scalar observation within optional bounds."""
    result = finite_observations([value], label=label, minimum=minimum, maximum=maximum)
    return float(result[0])


def scored_observation(value: Any, *, label: str, allow_infinite: bool = False) -> float:
    """Return one scalar observation admitted by a scalar scorer's input policy.

    A scalar scorer and its encoder must admit the same observations, otherwise a
    caller sees a plausible score where a batch of the same data is refused. Every
    continuous encoder rejects NaN, so NaN is rejected here as well: it is malformed
    evidence rather than a point carrying zero density.

    ``allow_infinite`` selects between the two per-law encoder policies. Families whose
    encoder is :func:`finite_observations` reject infinities too and leave it ``False``;
    families whose encoder documents "finite or infinite real-valued observations"
    (Exponential, Gumbel, Laplace, Logistic, Uniform) pass ``True`` so an infinity keeps
    scoring as the zero-density limit it already scores as through the encoded path.

    The float coercion itself is intentionally permissive, matching the ``np.asarray``
    coercion the encoders apply; only the finiteness policy is enforced here.
    """

    result = float(value)
    if math.isnan(result):
        raise UnscorableObservation(f"{label} rejects NaN observations.")
    if not allow_infinite and math.isinf(result):
        raise UnscorableObservation(f"{label} rejects infinite observations.")
    return result


def consistent_anchored_triple(suff_stat: Any, sum_x: float, count: float) -> tuple[float, float, float] | None:
    """Return the ``(anchor, a_sum, a_sum2)`` payload of ``suff_stat`` when it is usable, else ``None``.

    Shared by every scalar family whose shift-anchored moment track is a single first/second-moment
    pair riding on the raw ``(sum, sum2, count)`` sufficient statistic -- the Gaussian, Logistic,
    Gumbel, Student-t and generalized-Pareto families (the higher-order families,
    GeneralizedGaussian and GeneralizedExtremeValue, carry more moments and are not this shape).
    Extracted after the duplicate-body scanner caught the two copies drifting apart risk: this is
    exactly the sibling-bug class D-0200/D-0202 spent three release waves closing, and a shared
    implementation means the next family that needs this payload gets the fix for free instead of a
    third copy to keep in sync.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    finite and agrees with the raw first moment it claims to describe -- a hand-built SuffStat whose
    payload contradicts its tuple must not silently change the estimate the tuple alone would have
    produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or count <= 0.0:
        return None
    anchor, a_sum, a_sum2 = anchored
    if not (np.isfinite(anchor) and np.isfinite(a_sum) and np.isfinite(a_sum2)) or a_sum2 < 0.0:
        return None
    implied_sum = a_sum + count * anchor
    tolerance = 1.0e-6 * max(abs(sum_x), abs(count * anchor), 1.0)
    if abs(implied_sum - sum_x) > tolerance:
        return None
    return float(anchor), float(a_sum), float(a_sum2)


def scale_anchored_triple(
    anchor: float | None, a_sum: float, a_sum2: float, c: float
) -> tuple[float | None, float, float]:
    """Scale an ``(anchor, a_sum, a_sum2)`` track by ``c``, the way uniform weight scaling requires.

    Shared by every scalar family's ``scale()`` override for the reason
    :func:`consistent_anchored_triple` gives. Uniform weight scaling is exactly linear in both
    anchored moments and leaves the anchor -- a data value, not a statistic -- alone, so the track
    scales as the raw moments do; an unset anchor (``None``) passes through unchanged.
    """
    if anchor is None:
        return anchor, a_sum, a_sum2
    return anchor, a_sum * c, a_sum2 * c


# --------------------------------------------------------------------------------------------
# Shift-anchored moment track: the shared repair for every scalar location-scale family.
#
# Four release waves fixed this defect one family at a time -- Gaussian, then Multivariate/Diagonal
# Gaussian, GeneralizedGaussian, GeneralizedExtremeValue and Logistic, then Gumbel, Student-t,
# GeneralizedPareto and MultivariateStudentT. Each wave a family that had been recorded as an
# accepted limit came back as a blocking finding. The per-family copy is what kept failing, so the
# mechanism lives here once: a family that differences raw moments in its M-step mixes in
# :class:`AnchoredMomentTrack` and reads the payload back with :func:`consistent_anchored_triple`,
# and it is shift-equivariant without a fifth transcription of the arithmetic.
# --------------------------------------------------------------------------------------------

# Conditioning threshold for the anchored-moment gate: the raw ``E[x^2]-E[x]^2`` variance loses
# about ``eps * (mean/sd)^2`` relative accuracy, so a (mean/sd)^2 up to 4e6 (ratio ~2000) keeps the
# raw form within ~1e-9 relative error -- the historical single-pass path is bit-preserved there.
# Beyond it the anchored track takes over. Chunks pooled from gate-passing content stay
# well-conditioned as a pool (Cauchy-Schwarz: n*mean_pool^2 <= sum_i n_i*mean_i^2), so a pool built
# only from gate-passing chunks never needs the anchor retroactively.
ANCHOR_CONDITION_RATIO = 4.0e6

# Bound on how far a reported mean can sit from the exact sample mean the anchored track knows:
# ~4-8 grid steps of ``abs(mean)``. It bounds a rounding residue of the mean, not a spread, so a
# multiple of the ulp is the right shape here.
MEAN_ROUNDING_BOUND = 8.8817841970012523e-16  # 4 * eps


def needs_anchor(chunk_sum: float, chunk_sum2: float, w_sum: float) -> bool:
    """Whether a chunk's weighted moments are too ill-conditioned for the raw variance form.

    ``spread2`` computed here is itself the cancellation-prone estimate, but as a GATE it is
    reliable: when cancellation has corrupted it, the corruption is bounded by ``eps * m^2``, which
    still leaves ``m*m`` orders of magnitude above ``ANCHOR_CONDITION_RATIO * spread2``.
    A non-positive computed spread activates the anchor outright (constant or near-constant data).
    """
    m = chunk_sum / w_sum
    spread2 = chunk_sum2 / w_sum - m * m
    return spread2 <= 0.0 or m * m > ANCHOR_CONDITION_RATIO * spread2


def spread_is_resolvable(variance: float, magnitude: float) -> bool:
    """Whether a spread of ``sqrt(variance)`` is representable at all at scale ``magnitude``.

    float64 values near ``magnitude`` lie on a grid of spacing ``u = ulp(magnitude)``, so deviations
    from the mean below about ``u/2`` cannot be carried by any sample there. Used only to decide what
    to DISCLOSE through ``numerical_repairs()``, never to clamp.
    """
    if not np.isfinite(magnitude) or magnitude <= 0.0 or not np.isfinite(variance) or variance <= 0.0:
        return False
    half_ulp = 0.5 * float(np.spacing(magnitude))
    return variance > half_ulp * half_ulp


def anchored_pooled_variance(
    anchor: float,
    a_sum: float,
    a_sum2: float,
    count: float,
    mean: float,
    pseudo_count: float | None,
    prior_mean: float | None,
    prior_variance: float | None,
) -> tuple[float, tuple[str, ...]]:
    """Population variance of ``x`` computed from shift-anchored moments, plus any repairs to disclose.

    ``count``, ``mean`` are the RAW data count and the final (possibly prior-blended) location; the
    prior itself is folded in afterward through ``pseudo_count``/``prior_mean``/``prior_variance``,
    never through the anchored sums (which only ever describe the observed data).

    The scatter is SPLIT rather than accumulated in one sum, which is what lets the noise clamp stay
    off the data. Writing ``anchored_mean = a_sum/count`` for the sample's own mean in
    anchor-relative coordinates,

        scatter(mean) = [a_sum2 - a_sum * anchored_mean] + count * (mean - anchor - anchored_mean)**2

    and the two brackets have completely different error characters. The first (``core``) is the
    scatter about the sample's OWN mean: both terms are O(count * spread^2), computed entirely at
    small magnitude, and it carries all the data. The second is the displacement of the mean
    actually reported (``mean``) from that sample mean -- genuine when a pseudo-count prior pulls
    the location, but on the plain maximum-likelihood path pure rounding of ``sum_x / count`` at
    data magnitude, and the ONLY place the large magnitude enters. Clamping the rounding term alone
    leaves the data untouched; a single combined sum could only clamp the total, so its ulp-scale
    threshold would have to be crossed by the spread as well, and any spread below
    ~``4 eps abs(mean)`` per observation would read as constant.
    """
    if count <= 0.0:
        observed_scatter = 0.0
        repairs: tuple[str, ...] = ()
    else:
        anchored_mean = a_sum / count
        # Scatter about the sample's own mean -- the whole of the data, computed at spread scale.
        core = a_sum2 - a_sum * anchored_mean
        # Mathematically >= 0; only last-ulp rounding of the two O(count * spread^2) terms can
        # undershoot -- or overshoot, and a degenerate component's scatter must come out EXACTLY
        # zero on every algebraically equivalent path, or the scale-relative variance floor reads
        # the +O(eps) residue as a genuine spread and two equivalent fits disagree.
        noise_scale = max(abs(a_sum2), abs(a_sum * anchored_mean), 1.0e-300)
        repairs = ()
        if core < 1.0e-12 * noise_scale:
            # Reporting zero for something whose apparent scatter was positive is a repair, not a
            # measurement -- but only worth saying when the spread it stood for was one this
            # magnitude could have represented.
            if spread_is_resolvable(core / count, max(abs(mean), abs(anchor))):
                repairs = ("spread-below-noise(%.3g of %.3g)" % (core / count, noise_scale / count),)
            core = 0.0
        core = max(core, 0.0)
        # Displacement of the reported mean from the sample mean. Below the mean's own rounding
        # granularity it is not a displacement at all, just which order the large-magnitude sum was
        # accumulated in, and squaring it would turn that into variance.
        shift = (mean - anchor) - anchored_mean
        if abs(shift) <= MEAN_ROUNDING_BOUND * max(abs(mean), abs(anchor)):
            shift = 0.0
        observed_scatter = core + count * shift * shift
    if pseudo_count not in (None, 0.0) and prior_variance is not None:
        offset = 0.0 if prior_mean is None else (prior_mean - mean) ** 2
        prior_scatter = pseudo_count * (prior_variance + offset)
        return (observed_scatter + prior_scatter) / (count + pseudo_count), repairs
    if count == 0.0:
        return 0.0, repairs
    return observed_scatter / count, repairs


def warn_uncorrectable_raw_moments(sum_x: float, sum_x2: float, count: float, *, family: str) -> None:
    """Warn when raw-only statistics are too ill-conditioned for the variance they imply.

    The anchored track fixes an accumulator's OWN accumulation. Statistics that arrive already
    reduced and without an anchor -- an engine/GPU kernel's stacked moments, a hand-built tuple, a
    legacy artifact -- cannot be corrected here: the information cancellation destroyed is not in
    them any more. Naming it is the difference between an imprecise fit and a silently wrong one.

    Deliberately NOT a raise: these statistics are the declared exchange format and the raw M-step
    is what the library has always done with them.

    TWO regimes have to speak, and only the first one used to. Partial loss (``spread2`` still
    positive but ``mean^2`` dominating it) is the graduated case, gated by
    :data:`ANCHOR_CONDITION_RATIO`. TOTAL loss -- cancellation has eaten the whole spread and the
    raw form computes ``spread2 <= 0`` -- was excluded on the grounds that it is also what an
    ordinary degenerate or single-point EM component looks like, and that exclusion made the
    WORST case the silent one: raw-only Gumbel, Student-t, logistic and GEV statistics for sd ~2
    data at offset 1.7e9 all computed a non-positive variance and returned a scale collapsed onto
    the family floor (1e-8, 1e-12) with no warning at all, while the families' own docstrings
    promised that ``estimate`` "warns rather than returning a scale it cannot stand behind".

    Both regimes now warn. The total-loss message does NOT claim the data was ill-conditioned,
    because from raw moments alone that is not knowable: at magnitude ``m`` the raw form resolves
    no spread below about ``1.5e-8 * abs(m)``, so a genuinely constant sample and a sample whose
    spread was destroyed produce the same statistics. It says exactly that, and both readings point
    at the same remedy. A single observation (``count <= 1``) and all-zero data have no spread to
    lose and stay quiet. Measured blast radius across the 183 existing test files that exercise
    these families: the total-loss branch fires four times, all on statistics deliberately stripped
    of their anchored payload.
    """
    if count <= 0.0:
        return
    m = sum_x / count
    spread2 = sum_x2 / count - m * m
    if spread2 > 0.0:
        if not m * m > ANCHOR_CONDITION_RATIO * spread2:
            return
        import warnings

        ratio = m * m / spread2
        warnings.warn(
            "%s sufficient statistics arrived without shift-anchored moments and are too "
            "ill-conditioned for the raw E[x^2] - E[x]^2 variance: mean^2/variance is %.3g, so the "
            "fitted scale loses roughly %.0f%% of its significant digits to cancellation. Accumulate "
            "through this family's own accumulator (which anchors automatically), or subtract a "
            "constant origin from the data before fitting."
            % (family, float(ratio), min(100.0, 100.0 * math.log10(ratio) / 16.0)),
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if count <= 1.0 or m == 0.0:
        return
    import warnings

    warnings.warn(
        "%s sufficient statistics arrived without shift-anchored moments and imply a non-positive "
        "E[x^2] - E[x]^2 variance at mean %.6g, so the fitted scale falls onto this family's floor. "
        "Raw moments at that magnitude cannot resolve a spread below about %.3g, so this is either a "
        "genuinely constant sample or one whose spread cancellation destroyed -- they are not "
        "distinguishable from these statistics. Accumulate through this family's own accumulator "
        "(which anchors automatically), or subtract a constant origin from the data before fitting."
        % (family, float(m), 1.5e-8 * abs(float(m))),
        RuntimeWarning,
        stacklevel=3,
    )


class AnchoredSuffStat(tuple):
    """A raw sufficient-statistic tuple of any arity that also carries a shift-anchored payload.

    Behaves exactly like the plain tuple everywhere it is indexed, unpacked, iterated or validated
    (it *is* one); ``anchored`` is ``(anchor, sum_i w_i*(y_i - anchor), sum_i w_i*(y_i - anchor)^2)``
    for whichever quantity ``y`` the family's M-step differences moments of. Code that doesn't know
    about the payload -- generic ``scale_suff_stat``, engine kernels, older serializations -- sees an
    ordinary tuple and the estimate falls back to the historical raw path.

    The five families repaired in earlier waves each grew their own near-identical subclass before
    this existed; new hosts use this one instead, so the payload's pickle contract lives in one place
    rather than in a sixth and seventh copy for the duplicate-body gate to catch.
    """

    def __new__(cls, values: Any, anchored: tuple[float, float, float] | None = None) -> AnchoredSuffStat:
        obj = super().__new__(cls, values)
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A plain tuple subclass with a payload-bearing __new__ does not pickle by default; the
        # Spark/mp reducers round-trip accumulator values through pickle, so keep the payload.
        return (_rebuild_anchored_suff_stat, (tuple(self), self.anchored))


def _rebuild_anchored_suff_stat(values: tuple, anchored: tuple | None) -> AnchoredSuffStat:
    """Unpickle helper for :class:`AnchoredSuffStat` (module-level so pickle can import it)."""
    return AnchoredSuffStat(values, anchored=anchored)


class AnchoredMomentTrack:
    """Mixin carrying a conditioning-gated shift-anchored ``(sum, sum2)`` track beside raw moments.

    For every scalar family whose M-step forms ``E[x^2] - E[x]^2``. That form loses
    ~``2*log2(abs(mean)/sd)`` bits to cancellation, so data with sd ~2 at offset 1.7e9 fits a scale
    that is wildly wrong -- 8.8x too large for Gumbel, collapsed onto the scale floor for Student-t
    -- with no warning at all. Anchoring at a data value keeps every term of the scatter
    ``O(count * spread^2)``, which makes the M-step variance shift-invariant.

    The track is CONDITIONING-GATED: a chunk whose ``abs(mean)/spread`` ratio the raw form handles
    to ~1e-9 relative error (see :func:`needs_anchor`) accumulates exactly the historical
    single-pass way -- bit-identical statistics, no second pass -- and the anchor activates only
    when a chunk (or a scalar ``update``) would corrupt the variance. The raw moments remain the
    exchange format, so the anchored track rides along as a ``.anchored`` payload on a tuple
    subclass rather than replacing them.

    A host class must own float ``sum``, ``sum2`` and ``count`` attributes and call
    :meth:`_init_anchor` from its ``__init__``. Every hook below is written to run BEFORE the host
    folds new content into the raw moments, so an activation only ever converts content the gate
    has already vouched for.

    Those three names describe the moments the M-STEP differences, which are not always the raw
    observations: the log-Gaussian family differences the moments of ``log(x)``, and the Rician and
    Nakagami families difference the moments of ``x**2`` (both invert ``Var(X^2)``). A host whose
    exchange tuple orders those moments differently overrides
    :data:`_ANCHOR_SUM_INDEX` / :data:`_ANCHOR_SUM2_INDEX` / :data:`_ANCHOR_COUNT_INDEX`, which is
    where :meth:`_anchor_absorb` reads an incoming raw-only statistic from -- Rician and Nakagami
    both serialize as ``(count, sum, sum2)`` rather than ``(sum, sum2, count)``.
    """

    _anchor: float | None
    _anchored_sum: float
    _anchored_sum2: float

    #: Positions of ``(sum, sum2, count)`` inside this host's exchange tuple.
    _ANCHOR_SUM_INDEX = 0
    _ANCHOR_SUM2_INDEX = 1
    _ANCHOR_COUNT_INDEX = 2

    def _init_anchor(self) -> None:
        """Start with the track unactivated (the ordinary well-conditioned state)."""
        self._anchor = None
        self._anchored_sum = 0.0
        self._anchored_sum2 = 0.0

    def _activate_anchor(self, anchor: float) -> None:
        """Start the shift-anchored moment track at ``anchor``.

        Any content already accumulated raw-only is converted about the new anchor. The conversion
        is the cancellation-prone form, but it is only ever applied to content that accumulated
        WITHOUT activating the gate -- i.e. content the gate certified as well-conditioned -- or to
        pre-existing raw statistics restored through ``from_value``/``combine``, where the
        conversion is no less accurate than the raw-only estimate those statistics supported before.
        """
        self._anchor = float(anchor)
        if self.sum != 0.0 or self.sum2 != 0.0 or self.count != 0.0:
            self._anchored_sum += self.sum - self._anchor * self.count
            self._anchored_sum2 += max(
                self.sum2 - 2.0 * self._anchor * self.sum + self._anchor * self._anchor * self.count, 0.0
            )

    def _anchor_scalar(self, x: float, weight: float) -> None:
        """Fold one weighted observation into the anchored track. Call BEFORE the raw fold.

        Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        first observation THAT CARRIES POSITIVE WEIGHT -- a zero-cost O(1) bookkeeping track on
        this path. A weight of exactly 0.0 (an EM component's responsibility for a point it does
        not own, say -- ordinary usage of the ``update``/``seq_update`` calling convention, not
        misuse) contributes nothing to either anchored moment no matter what the anchor is, so such
        an observation must never be allowed to SET the anchor: a zero-weight observation at an
        extreme magnitude would otherwise become the permanent reference point every later,
        fully-weighted observation is differenced against, reintroducing exactly the cancellation
        this track exists to avoid. Mirrors the ``if weight > 0.0`` gate already used for
        max-tracking in ``GeneralizedParetoAccumulator.update``.
        """
        if self._anchor is None:
            if weight <= 0.0:
                # No anchor yet, and this observation cannot supply one: its own contribution to
                # both anchored sums is exactly zero (weight times anything is zero), so there is
                # nothing to fold here, and the anchor stays unset for a later, positively-weighted
                # observation to activate correctly.
                return
            self._activate_anchor(x)
        dx = x - self._anchor
        self._anchored_sum += dx * weight
        self._anchored_sum2 += dx * dx * weight

    def _anchor_chunk(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        chunk_sum: float,
        chunk_sum2: float,
        w_sum: float,
        preferred_anchor: float | None = None,
    ) -> None:
        """Fold an encoded chunk into the anchored track when it needs one. Call BEFORE the raw fold.

        The chunk's own moments are the ones the caller already computed for the raw fold, so the
        conditioning gate costs one scalar test rather than a second pass over the data.
        ``preferred_anchor`` lets a family that already knows a natural origin -- a
        peaks-over-threshold level, say -- anchor there instead of at the first observation.
        """
        if len(x) == 0:
            return
        if self._anchor is None and not (w_sum > 0.0 and needs_anchor(chunk_sum, chunk_sum2, w_sum)):
            return
        if self._anchor is None:
            if preferred_anchor is not None:
                self._activate_anchor(float(preferred_anchor))
            else:
                # ``w_sum > 0.0`` is guaranteed by the branch above, so this chunk carries at least
                # one positively-weighted element; anchor at the FIRST one of those rather than at
                # x[0] positionally. Per-point EM responsibilities passed through seq_update
                # routinely carry a weight of exactly 0.0 for a component's first point in a batch
                # (mixle/stats/compute/stacked.py, torch_mixture.py) -- ordinary usage, not misuse --
                # and an x[0] chosen without regard to its own weight would let that zero-weight
                # point, at whatever magnitude it happens to carry, become the permanent anchor
                # every later, real observation is differenced against.
                self._activate_anchor(float(x[np.argmax(weights > 0.0)]))
        dx = x - self._anchor
        wdx = dx * weights
        self._anchored_sum += float(np.sum(wdx))
        self._anchored_sum2 += float(np.dot(wdx, dx))

    def _anchor_fold_chunk(self, x: np.ndarray, weights: np.ndarray) -> None:
        """Accumulate one encoded chunk into BOTH the raw moments and the anchored track.

        The whole of a host's ``seq_update`` for the ``(sum, sum2, count)`` shape. It lives here
        rather than being transcribed per family for the reason this module exists: three families
        had already copied the identical seven-statement fold, and the duplicate-body gate
        (``mixle/tests/duplicate_body_scan_test.py``) is the ratchet that stops a fourth. The raw
        fold keeps the historical ``np.dot`` accumulation order, so a chunk that never activates
        the anchor produces bit-identical statistics.
        """
        chunk_sum = np.dot(x, weights)
        chunk_sum2 = np.dot(x * x, weights)
        w_sum = np.sum(weights, dtype=np.float64)
        self._anchor_chunk(x, weights, float(chunk_sum), float(chunk_sum2), float(w_sum))
        self.sum += chunk_sum
        self.sum2 += chunk_sum2
        self.count += w_sum

    def _anchor_absorb(self, suff_stat: Any) -> None:
        """Fold another sufficient statistic into the anchored track. Call BEFORE the raw fold.

        An incoming ``anchored`` payload merges by Chan's parallel form: re-express its moments
        about this accumulator's anchor. The anchor gap ``d`` is between two data values, so every
        term stays ``O(count * spread^2)`` and no large-offset cancellation is reintroduced. A
        raw-only statistic joining an already-anchored pool converts about our anchor instead --
        see :meth:`_activate_anchor` for why that cancellation-prone conversion is acceptable here.
        """
        anchored = getattr(suff_stat, "anchored", None)
        b_sum = float(suff_stat[self._ANCHOR_SUM_INDEX])
        b_sum2 = float(suff_stat[self._ANCHOR_SUM2_INDEX])
        b_count = float(suff_stat[self._ANCHOR_COUNT_INDEX])
        if anchored is not None:
            b_anchor, b_asum, b_asum2 = anchored
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            d = b_anchor - self._anchor
            self._anchored_sum += b_asum + b_count * d
            self._anchored_sum2 += b_asum2 + 2.0 * d * b_asum + b_count * d * d
        elif self._anchor is not None and (b_sum != 0.0 or b_sum2 != 0.0 or b_count != 0.0):
            a = self._anchor
            self._anchored_sum += b_sum - a * b_count
            self._anchored_sum2 += max(b_sum2 - 2.0 * a * b_sum + a * a * b_count, 0.0)

    def _anchor_payload(self) -> tuple[float, float, float] | None:
        """The ``(anchor, a_sum, a_sum2)`` payload to hang on ``value()``, or ``None`` when unactivated."""
        if self._anchor is None:
            return None
        return (self._anchor, self._anchored_sum, self._anchored_sum2)

    def _anchor_restore(self, x: Any) -> None:
        """Adopt the anchored payload of ``x``, or restart the track unactivated when it has none.

        Raw-only statistics REPLACE the state rather than converting it: a later activation (the
        first ``update``, or an anchored merge) converts this content then.
        """
        anchored = getattr(x, "anchored", None)
        if anchored is None:
            self._init_anchor()
            return
        self._anchor, self._anchored_sum, self._anchored_sum2 = (
            float(anchored[0]),
            float(anchored[1]),
            float(anchored[2]),
        )

    def _anchor_scale(self, c: float) -> None:
        """Scale the anchored track by ``c``, the way uniform weight scaling requires.

        The structural ``StatisticAccumulator.scale`` round-trips through ``value()`` and
        ``from_value()``, and ``scale_suff_stat`` rebuilds the payload as a PLAIN tuple -- which
        drops the ``anchored`` attribute, so ``from_value`` sees raw-only statistics and restarts
        the track unactivated, undoing the whole repair. Scaling every weight by ``c`` is reachable
        from ordinary use: HMM/LDA/hierarchical-mixture child accumulators and streaming EM's batch
        mixing all do it, so every host overrides ``scale`` to call this.
        """
        self._anchor, self._anchored_sum, self._anchored_sum2 = scale_anchored_triple(
            self._anchor, self._anchored_sum, self._anchored_sum2, c
        )


class SquaredPowerSumTrack(AnchoredMomentTrack):
    """Weighted ``(count, sum x^2, sum x^4)`` power sums with the anchored track riding on ``x**2``.

    The Rician and Nakagami families share one accumulator shape: both invert ``Var(X^2)``, so both
    accumulate the second and fourth power sums and both need the shift-anchored track on
    ``y = x**2`` rather than on ``x``. Every method below was identical in the two modules, and the
    duplicate-body gate (``mixle/tests/duplicate_body_scan_test.py``) caught the pair the moment the
    anchored track made the bodies long enough to notice -- which is the whole point of that gate,
    because this defect class is exactly what per-family transcription keeps reintroducing.

    A host subclasses this alongside ``SequenceEncodableStatisticAccumulator``, sets
    :data:`_OBSERVATION_LABEL`, and supplies ``acc_to_encoder``. The mixin stays free of any
    ``mixle.stats.compute`` import so it can live beside the rest of the anchored-moment machinery.
    """

    #: Singular label for the family's observations, e.g. ``"Rician"`` -- used in validation errors.
    _OBSERVATION_LABEL = "observation"

    # The moments the M-step differences are those of ``y = x**2``, and the exchange tuple orders
    # them ``(count, sum y, sum y^2)`` rather than the scalar families' ``(sum, sum2, count)``.
    _ANCHOR_SUM_INDEX = 1
    _ANCHOR_SUM2_INDEX = 2
    _ANCHOR_COUNT_INDEX = 0

    @property
    def sum(self) -> float:
        """Alias of ``s2`` -- the first moment of ``y = x**2``, which the M-step differences."""
        return self.s2

    @sum.setter
    def sum(self, value: float) -> None:
        self.s2 = value

    @property
    def sum2(self) -> float:
        """Alias of ``s4`` -- the second moment of ``y = x**2``, which the M-step differences."""
        return self.s4

    @sum2.setter
    def sum2(self, value: float) -> None:
        self.s4 = value

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s2 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys
        self._init_anchor()

    def update(self, x: float, weight: float, estimate: Any) -> None:
        """Accumulate weighted second and fourth power sums for one observation."""
        x2 = finite_observation(x, label="%s observation" % self._OBSERVATION_LABEL, minimum=0.0) ** 2
        self._anchor_scalar(float(x2), weight)  # BEFORE the raw fold, per AnchoredMomentTrack
        self.count += weight
        self.s2 += weight * x2
        self.s4 += weight * x2 * x2

    def initialize(self, x: float, weight: float, rng: Any) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted second and fourth power sums from encoded data.

        ``Var(X^2) = E[X^4] - E[X^2]^2`` is the same differenced form every location-scale sibling
        had to anchor, with ``y = x**2`` in place of ``x``. The conditioning gate is assessed on
        ``y``'s own moments, so ordinary data never activates the track and the raw fold below stays
        bit-identical to the historical single-pass path.
        """
        x2 = finite_observations(x, label="%s observations" % self._OBSERVATION_LABEL, minimum=0.0) ** 2
        w = np.asarray(weights, dtype=np.float64)
        self._anchor_chunk(x2, w, float(np.dot(w, x2)), float(np.dot(w, x2 * x2)), float(w.sum()))
        self.count += float(w.sum())
        self.s2 += float(np.dot(w, x2))
        self.s4 += float(np.dot(w, x2 * x2))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Any) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> SquaredPowerSumTrack:
        """Merge another ``(count, sum x^2, sum x^4)`` sufficient-statistic tuple."""
        self._anchor_absorb(suff_stat)
        self.count += suff_stat[0]
        self.s2 += suff_stat[1]
        self.s4 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, second power sum, and fourth power sum.

        A plain 3-tuple for every consumer that treats it as one; once the shift-anchored track is
        live it is an :class:`AnchoredSuffStat` additionally carrying the anchored moments of
        ``x**2`` in its ``anchored`` attribute.
        """
        anchored = self._anchor_payload()
        if anchored is None:
            return self.count, self.s2, self.s4
        return AnchoredSuffStat((self.count, self.s2, self.s4), anchored=anchored)

    def from_value(self, x: tuple[float, float, float]) -> SquaredPowerSumTrack:
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.s2, self.s4 = float(x[0]), float(x[1]), float(x[2])
        self._anchor_restore(x)
        return self

    def scale(self, c: float) -> SquaredPowerSumTrack:
        """Scale the accumulated statistics in place by ``c``, anchored track included."""
        self._anchor_scale(c)
        self.count *= c
        self.s2 *= c
        self.s4 *= c
        return self


def centered_batch_moments(x: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(w_sum, mean, m2, m3)``: exactly centered weighted moments of one batch.

    Shared by the families whose accumulator stores Pebay/West central moments rather than raw power
    sums (ExGaussian, SkewNormal). Centering on the computed mean removes the ``E[x^2]-E[x]^2``
    cancellation, but not all of it: ``mean`` itself carries the rounding of a large-magnitude sum,
    ~``ulp(mean) * sqrt(n)``, and that residue enters ``m3`` linearly (``3 * delta * m2``), which at
    offset 1e12 moved a fitted skew-normal shape by 2e-3. A SECOND centering pass fixes it -- the
    residual offset is computed at spread scale, where it is exact -- so the moments describe the
    sample's true centroid to the last ulp and the fit is shift-equivariant.

    ``w_sum <= 0`` returns an empty batch the caller can skip.
    """
    xx = np.asarray(x, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(ww))
    if w_sum <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.dot(xx, ww) / w_sum)
    # ``dx`` is exact whenever ``mean`` is close to the data (Sterbenz); the residual mean offset
    # below is therefore computed entirely at spread scale.
    dx = xx - mean
    residual = float(np.dot(dx, ww) / w_sum)
    dx = dx - residual
    return w_sum, mean + residual, float(np.dot(ww, dx * dx)), float(np.dot(ww, dx * dx * dx))
