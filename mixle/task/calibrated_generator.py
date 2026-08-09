"""``CalibratedGenerator`` -- certified selective generation with abstention.

The generation-side sibling of :class:`~mixle.task.calibrate.CalibratedTaskModel`. That class gates
*classification*: it turns an uncalibrated softmax into conformal label sets and escalates on an
ambiguous label set. Open-ended generation has no fixed label space: candidates are newly sampled
and their identities change by prompt, so applying label-set conformal prediction to candidate slots
would not provide classification coverage.

This module instead certifies one fixed measurable outcome: whether the top-scored generated
candidate is correct when its score clears a threshold. Calibration is split into threshold
proposal and independent certification halves. Candidate thresholds are proposed on the first half;
the second half supplies simultaneous Bonferroni-corrected exact binomial upper bounds, allowing the
most permissive threshold whose accepted-error upper bound is at most ``alpha`` to be selected without
reusing its evidence. The guarantee is a confidence statement about selective risk under the usual
i.i.d./exchangeability and stable-generation assumptions, not conformal coverage of changing strings.

An uncertified prompt returns :data:`ABSTAIN` (``None``), the same sentinel as
:data:`mixle.task.calibrate.ESCALATE`, so cascades can escalate without special casing.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from mixle.utils.callables import accepts_call

ABSTAIN = None  # sentinel returned when no candidate clears the calibrated threshold; equals Cascade's ESCALATE


def _derive_seed(base_seed: int, prompt: Any) -> int:
    """A per-prompt seed derived from ``base_seed``, cross-process stable for canonical prompt types.

    Unlike builtin ``hash()``, which is salted per process, this is reproducible across runs for the
    prompt types :func:`_seed_key` can encode canonically: ``str``, ``bytes``, ``int``, ``float``,
    ``bool``, ``None``, and tuples, lists, sets or mappings of those.

    Sets and mappings are encoded from their CONTENTS, not their ``repr`` (MXR-080-1848). Both were
    previously called canonical while being neither: a set's iteration order follows element hashes, so
    the same set of strings reordered under a different ``PYTHONHASHSEED``, and a dict's ``repr``
    follows insertion order, so two equal dicts built in different orders seeded differently. Sorting
    their encoded members makes the key depend on the value alone, which is what the promise claimed.
    Sequence order is preserved, because for a tuple or list the order IS part of the value.

    A prompt this cannot encode -- including one that merely defines its own ``__repr__``, which proves
    nothing about what that repr contains -- warns and falls back to ``repr``, so a recorded run keeps
    reproducing while the caller learns the promise does not cover it."""
    if not _is_canonically_representable(prompt):
        warnings.warn(
            f"seed derivation for a {type(prompt).__name__} prompt is not reproducible across processes: "
            "its repr is not a canonical encoding of its value, so equal prompts can seed differently. "
            "Pass a str/bytes/number/None, a container of those, or your own stable key when a draw must "
            "reproduce (MXR-080-1848).",
            stacklevel=3,
        )
    digest = hashlib.sha256(f"{base_seed}:{_seed_key(prompt)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _is_canonically_representable(value: Any) -> bool:
    """Whether this value has a canonical encoding -- one determined by its VALUE alone.

    Three shapes used to pass here that should not have (MXR-080-1848):

    * a ``set``/``frozenset``, whose iteration order follows element hashes, so a set of strings
      reorders under a different ``PYTHONHASHSEED`` and two equal sets encoded differently;
    * a ``Mapping``, whose ``repr`` follows insertion order, so ``{"a": 1, "b": 2}`` and
      ``{"b": 2, "a": 1}`` -- equal dicts -- encoded differently;
    * any type merely DEFINING ``__repr__``, which was taken as proof of canonicality. Defining one
      says nothing about what it contains; a custom ``__repr__`` embedding ``id(self)`` was
      classified canonical and gave two equal instances different seeds.

    The first two are now encoded order-independently by :func:`_seed_key` rather than declared
    unusable, so they became correct instead of merely honest. The third cannot be settled by
    inspection, so it is no longer assumed: an unrecognized type warns.
    """
    return _seed_key(value) is not None


def _seed_key(value: Any, _depth: int = 0) -> Any:
    """A canonical string encoding of ``value``, or ``None`` when it has none.

    Returning ``None`` is what drives the warning above; ``_derive_seed`` still falls back to
    ``repr`` in that case, so an unrecognized prompt keeps the seed it always had and previously
    recorded runs keep reproducing.
    """
    if _depth > 20:  # a self-referential container; repr would print "..." but this would not return
        return None
    if isinstance(value, (str, bytes, bytearray)):
        kind = "s" if isinstance(value, str) else "b"
        body = value if isinstance(value, str) else bytes(value).hex()
        return _tagged(kind, body)
    if isinstance(value, bool) or value is None:
        # Tagged apart from the numbers below even though ``True == 1`` and ``False == 0``. A boolean
        # prompt and an integer prompt are different prompts, and every other validator in this
        # package draws the same line (``isinstance(x, bool) or not isinstance(x, int)``). This is
        # the one deliberate departure from "equal values share a key".
        return _tagged("c", repr(value))
    if isinstance(value, (int, float)):
        # 1 == 1.0 in Python, so they are the same prompt and must derive the same seed. Encoding
        # repr() directly gave "1" and "1.0" -- equal values, different draws, which is the same
        # class of defect as MXR-080-1848 in the other direction.
        numeric = float(value) if isinstance(value, float) else value
        if isinstance(numeric, float) and numeric.is_integer():
            numeric = int(numeric)
        return _tagged("n", repr(numeric))
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            encoded_key, encoded_item = _seed_key(key, _depth + 1), _seed_key(item, _depth + 1)
            if encoded_key is None or encoded_item is None:
                return None
            pairs.append(encoded_key + encoded_item)  # each half is self-delimiting, so is the pair
        return _tagged("m", "".join(sorted(pairs)))  # sorted: a dict's equality ignores its order
    if isinstance(value, (set, frozenset)):
        items = [_seed_key(item, _depth + 1) for item in value]
        if any(item is None for item in items):
            return None
        return _tagged("e", "".join(sorted(items)))  # sorted: iteration order is hash-seed dependent
    if isinstance(value, (tuple, list)):
        items = [_seed_key(item, _depth + 1) for item in value]
        if any(item is None for item in items):
            return None
        # Ordered, and tagged by type: order IS part of the value, and ``["x"] != ("x",)`` in Python,
        # so they are different prompts and must not share a seed. Sets and frozensets DO compare
        # equal, so they deliberately share the "e" tag above.
        return _tagged("q" if isinstance(value, tuple) else "l", "".join(items))
    return None


def _tagged(kind: str, body: str) -> str:
    """Wrap ``body`` so the encoding is self-delimiting, and therefore injective.

    An earlier revision joined members with a comma, which is not injective when a member may itself
    contain the delimiter: ``["x,s:y"]`` and ``["x", "y"]`` both encoded to ``q:[s:x,s:y]`` and
    derived the same seed (MXR-080-1858). Length-prefixing every chunk means a reader can parse the
    structure back out unambiguously, so no two distinct values share an encoding regardless of what
    characters they contain.
    """
    return f"{kind}{len(body)}:{body}"


def _binomial_error_upper(errors: int, accepted: int, tail_probability: float) -> float:
    """One-sided Clopper-Pearson upper bound for an accepted-error rate."""
    if accepted <= 0 or errors < 0 or errors > accepted:
        raise ValueError("binomial counts must satisfy 0 <= errors <= accepted and accepted > 0")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be in (0, 1)")
    if errors == accepted:
        return 1.0
    return float(beta_distribution.ppf(1.0 - tail_probability, errors + 1, accepted - errors))


def smallest_certifiable_calibration_set(
    alpha: float, *, confidence: float = 0.95, thresholds_tested: int = 2, limit: int = 100_000
) -> int:
    """Smallest ``calibrate(...)`` set size whose risk certificate can reach ``alpha`` at all.

    :meth:`CalibratedGenerator.calibrate` splits the set in half and certifies on the leading half with a
    one-sided Clopper-Pearson bound, Bonferroni-corrected across the thresholds it tests. That bound has a
    floor set by the certification count alone: with ``c`` certification rows and zero observed errors it
    is still ``1 - (tail)**(1/c)``. Below the size returned here, ``alpha`` sits under that floor, no
    threshold can ever be eligible, ``qhat`` stays ``+inf`` and serving abstains on every input -- for a
    reason that has nothing to do with the model. Sizing a calibration set from this avoids mistaking a
    structurally impossible target for a model that needs tuning.

    ``thresholds_tested`` is the Bonferroni divisor; :meth:`calibrate` tests one threshold per distinct
    proposal-split statistic plus ``-inf``, so 2 is the floor and the realistic best case. Pass the number
    you expect if the statistic takes many distinct values, since more thresholds tighten the tail.
    """
    if not (isinstance(alpha, (int, float, np.integer, np.floating)) and not isinstance(alpha, (bool, np.bool_))):
        raise TypeError("alpha must be a real number")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if not np.isfinite(confidence) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    if isinstance(thresholds_tested, bool) or not isinstance(thresholds_tested, (int, np.integer)):
        raise TypeError("thresholds_tested must be an exact integer")
    if int(thresholds_tested) < 1:
        raise ValueError("thresholds_tested must be at least 1")
    tail = (1.0 - float(confidence)) / int(thresholds_tested)
    for total in range(2, limit + 1):
        certification = total - total // 2
        if _binomial_error_upper(0, certification, tail) <= float(alpha):
            return total
    raise ValueError(
        f"no calibration set up to {limit} can certify alpha={alpha} at confidence={confidence} "
        f"across {thresholds_tested} thresholds"
    )


class CalibratedGenerator:
    """Draw ``k`` candidates and serve the best under a certified selective-risk gate.

    Args:
        generate: ``generate(prompt, k) -> Sequence[candidate]`` (an ``rng`` keyword is passed if the
            callable accepts one; falls back to the two-argument form otherwise). Any generator that can
            draw ``k`` candidates for a prompt works: a wrapped :class:`~mixle.task.llm.CallableLLM`
            sampled ``k`` times, a beam, a stochastic sampler.
        score: ``score(candidate) -> float``, any mixle-scoreable model. Higher is better; the score
            need not be a calibrated probability; calibration thresholds its ordering statistic directly.
        alpha: maximum certified error rate among accepted candidates.
        k: number of candidates to draw per prompt.
        qhat: optional manually supplied acceptance threshold. This remains callable
            but has no risk certificate; only :meth:`calibrate` populates
            :attr:`risk_receipt`.
        confidence: simultaneous confidence level for the held-out risk certificate.
        seed: base seed for candidate draws; combined with the prompt (see :func:`_derive_seed`) so
            different prompts get different, but reproducible, draws.
    """

    def __init__(
        self,
        generate: Callable[..., Sequence[Any]],
        score: Callable[[Any], float],
        alpha: float = 0.1,
        *,
        k: int = 8,
        qhat: float | None = None,
        seed: int = 0,
        confidence: float = 0.95,
    ) -> None:
        if not callable(generate) or not callable(score):
            raise TypeError("generate and score must be callable")
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha)
            or not 0.0 < float(alpha) < 1.0
        ):
            raise ValueError("alpha must be a finite number strictly between 0 and 1")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be an exact positive integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an exact integer")
        if (
            isinstance(confidence, (bool, np.bool_))
            or not isinstance(confidence, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence)
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ValueError("confidence must be a finite number strictly between 0 and 1")
        if qhat is not None and (
            isinstance(qhat, (bool, np.bool_))
            or not isinstance(qhat, (int, float, np.integer, np.floating))
            or np.isnan(qhat)
        ):
            raise ValueError("qhat must be a real threshold, positive infinity, or None")
        self.generate = generate
        self.score = score
        self.alpha = float(alpha)
        self.k = int(k)
        self.qhat = None if qhat is None else float(qhat)
        self.seed = int(seed)
        self.confidence = float(confidence)
        self.risk_receipt: dict[str, Any] | None = None
        # MXR-080-1894: what the certificate, once issued, is a certificate OF.
        self._certified_policy: tuple[Any, ...] | None = None

    def _policy(self) -> tuple[Any, ...]:
        """The exact policy a risk certificate covers.

        ``generate`` and ``score`` are held BY REFERENCE, not by ``id()``: replacing an attribute can
        free the old callable and a new one can land on the same address, which would make an identity
        check silently pass on the very swap it exists to catch (MXR-080-1894). Holding the objects
        keeps the comparison exact, at the cost of pinning two callables the certificate is about
        anyway. ``qhat`` is included because a hand-set threshold after calibration is a different
        acceptance rule than the one that was certified.
        """
        return (self.generate, self.score, self.k, self.seed, self.alpha, self.confidence, self.qhat)

    def _describe_policy(self) -> dict[str, Any]:
        """JSON-able identity of the certified policy, for the receipt itself."""

        def name(fn: Any) -> str:
            return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"

        return {
            "generate": name(self.generate),
            "score": name(self.score),
            "k": self.k,
            "seed": self.seed,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "statistic": "top_score",
        }

    def _require_certified_policy(self) -> None:
        """Refuse to serve under a certificate that was issued for a different policy.

        The receipt has always ASSERTED "candidate generation and scoring policies remain fixed after
        calibration" as an assumption; nothing enforced it. ``generate``, ``score``, ``k``, ``seed`` and
        ``qhat`` are plain attributes, so swapping the generator and the scorer after ``calibrate()``
        left the old certificate attached and still gating -- serving arbitrary output under a bound
        computed for code that no longer runs (MXR-080-1894).

        Deliberately NOT checked: whether a callable's own *behaviour* changed while its identity stayed
        the same (a closure over mutable state, a stateful model reloaded in place). No in-process check
        can see that; it remains a stated assumption in the receipt.
        """
        if self.risk_receipt is None or self._certified_policy is None:
            return  # no certificate to invalidate: a hand-set qhat is documented as uncertified
        if self._policy() != self._certified_policy:
            raise RuntimeError(
                "the risk certificate in risk_receipt was issued for a different generate/score/k/seed/"
                "alpha/confidence/qhat policy than the one now configured; re-run calibrate(...) before "
                "serving, or construct a fresh CalibratedGenerator"
            )

    def _draw(self, prompt: Any, *, seed: int) -> list[Any]:
        rng = np.random.default_rng(seed)
        if accepts_call(self.generate, prompt, self.k, rng=rng):
            cands = self.generate(prompt, self.k, rng=rng)
        else:
            cands = self.generate(prompt, self.k)
        cands = list(cands)
        if len(cands) != self.k:
            raise ValueError(f"generate(...) must return exactly k={self.k} candidates, got {len(cands)}")
        return cands

    def _scored(self, prompt: Any, *, seed: int) -> tuple[list[Any], np.ndarray]:
        cands = self._draw(prompt, seed=seed)
        scores = np.asarray([float(self.score(c)) for c in cands], dtype=float)
        if scores.shape != (self.k,) or np.any(~np.isfinite(scores)):
            raise ValueError("candidate scores must be finite scalars with exactly one score per candidate")
        return cands, scores

    def _selection(self, prompt: Any, *, seed: int) -> tuple[Any, float]:
        cands, scores = self._scored(prompt, seed=seed)
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        # The top score is the fixed acceptance statistic. Calibration makes no
        # probabilistic claim about its scale; it only thresholds the stable ranking
        # policy and certifies the resulting accept/error event independently.
        statistic = float(scores[best])
        return cands[best], statistic

    def calibrate(
        self,
        prompts: Sequence[Any],
        is_correct: Callable[[Any, Any], bool],
        *,
        seed: int | None = None,
        outcomes: str = "per-prompt",
        sampling: str = "constructed",
    ) -> CalibratedGenerator:
        """Certify a held-out accepted-error threshold using an explicit correctness oracle.

        The trailing half proposes score-margin thresholds; the independent leading
        half evaluates every proposal with simultaneous exact binomial bounds. Only
        the certification half consults ``is_correct``, once per row in order, so a
        metered or side-effecting oracle is not spent on rows that cannot reach the
        bound (MXR-080-1849). If no nonempty accepted subset certifies risk
        ``<= alpha``, the threshold is ``+inf`` and serving abstains everywhere.

        Scope of the certificate: it covers exactly the SERVED stochastic policy -- candidate
        draws seeded from the generator's own ``seed`` and the prompt alone, the same
        derivation ``candidate_set()``/``serve()`` use (STAT-RR17-07: certifying under a
        per-row schedule measured 1/150 errors and bounded risk at 0.0366 while the served
        policy produced 1000/1000 errors). Assuming the calibration prompts and future
        queries are i.i.d. draws from the same distribution, with probability at least
        ``confidence`` over the calibration draw, the TRUE accepted-slice error at the
        deployed threshold is ``<= alpha`` (the per-threshold binomial bounds are Bonferroni-corrected across
        the proposal family, so the selection of the loosest passing threshold stays
        covered). Distribution shift voids the statement silently; re-certify on drifted
        traffic.

        WHAT the bound is a bound ON depends on how duplicated prompts got into the stream, and
        only the caller knows that -- ``sampling`` is that declaration (the same
        premise-declaration contract as ``outcomes``):

        * ``sampling="constructed"`` (fail-safe default): the prompt list is a curated set, so
          under ``outcomes="per-prompt"`` duplicates are redundant copies of one Bernoulli and
          COLLAPSE to one certification row (300 copies of one always-correct prompt certify from
          n = 1, bound 0.975 -- STAT-RR18-04). The certified estimand is then UNIFORM over the
          DISTINCT certification prompts. It is NOT traffic-weighted: measured on 40%-heavy
          i.i.d. traffic whose heavy prompt always errs, collapsing certified error_upper 0.080
          while the served traffic risk was 0.37-0.41, every trial (the pass-19 blocker). If your
          serving mix repeats prompts, this default's certificate does not cover it -- declare the
          sampling truthfully instead.
        * ``sampling="iid-traffic"``: the caller's recorded assertion that the calibration rows
          are an i.i.d. draw of the serving traffic itself. Row error indicators are then i.i.d.
          Bernoulli(traffic accepted-error rate) EVEN under per-prompt outcomes -- the randomness
          is the prompt draw, and a duplicate's multiplicity IS its traffic weight -- so rows all
          count and the bound covers the TRAFFIC-WEIGHTED accepted-error rate serving experiences.
          On the measurement above, this declaration refuses to certify alpha = 0.15 (the heavy
          errors are ~40% of rows), which is the correct answer.

        Disagreeing duplicate verdicts are refused under ``outcomes="per-prompt"`` in both
        sampling modes (a deterministic decision with a fixed per-prompt outcome cannot differ
        between copies); the receipt records both declarations and names the certified estimand.
        """
        if not callable(is_correct):
            raise TypeError("is_correct must be callable")
        prompts = list(prompts)
        if len(prompts) < 2:
            raise ValueError("calibrate(...) needs at least two held-out prompts for proposal/certification splitting")
        if outcomes not in ("per-prompt", "per-row"):
            raise ValueError("outcomes must be 'per-prompt' or 'per-row'")
        if sampling not in ("constructed", "iid-traffic"):
            raise ValueError("sampling must be 'constructed' or 'iid-traffic'")
        if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
            raise ValueError("seed must be an exact integer or None")
        if seed is not None and int(seed) != self.seed:
            # STAT-RR17-07: the certificate must cover the policy that SERVES. Serving derives its
            # candidate seed from self.seed and the prompt alone, so calibrating under any other
            # base seed certifies a policy that never runs. Construct the generator with the seed
            # you want and calibrate that.
            raise ValueError(
                "calibrate(seed=...) must equal the generator's own seed: the certificate covers the "
                f"served policy (seed={self.seed}), and certifying under a different seed schedule is "
                "how a 3.7% certificate served 100% errors"
            )
        rng_seed = self.seed
        # ORACLE CONTRACT: is_correct is called exactly once per CERTIFICATION row, in order, starting
        # at the first prompt. The oracle receives only (prompt, candidate) -- no row index -- so an
        # oracle that must recover per-row ground truth has no way to do it except by counting its own
        # calls; mixle.reason.language_bridge.PosteriorDescriber.calibrate does exactly that, closing
        # over a monotone counter and indexing `truths[calls["n"]]`.
        #
        # That counter is why the proposal half used to be scored too (MXR-080-1849): only
        # certification verdicts reach the bound, but simply skipping the first calls restarted any
        # such counter at 0 and scored every certification row against a truth `split` rows earlier.
        # The fix is not to give the oracle an index -- which would change its public signature -- but
        # to notice that WHICH half certifies is arbitrary. Certifying the LEADING rows means the
        # oracle is called on a prefix, so a counting oracle stays aligned with no protocol change,
        # and the proposal half needs statistics only, which cost no oracle calls at all.
        n_certify = len(prompts) - len(prompts) // 2  # the larger half when the count is odd
        statistics: list[float] = []
        errors: list[bool] = []
        for i, prompt in enumerate(prompts):
            # STAT-RR17-07: select with the SERVED seed schedule -- self.seed and the prompt,
            # exactly what candidate_set() derives -- never a per-row schedule. Certifying under
            # (row, prompt) seeds measured 1/150 errors and certified an 0.0366 upper bound while
            # the served (prompt-only) policy produced 1000/1000 errors on a repeated-prompt
            # population: the certified policy never served. With this schedule a repeated prompt
            # contributes the identical served decision to certification, so the certificate can
            # only report what serving would do; duplicated prompts reduce the certificate's
            # effective independent-sample count (see unique_prompt_count in the receipt).
            candidate, statistic = self._selection(prompt, seed=_derive_seed(self.seed, prompt))
            statistics.append(statistic)
            if i >= n_certify:
                continue  # a proposal row contributes a threshold, never a verdict
            verdict = is_correct(prompt, candidate)
            if not isinstance(verdict, (bool, np.bool_)):
                raise TypeError("is_correct must return a boolean")
            errors.append(not bool(verdict))

        # STAT-RR18-04: the bound's exchangeability unit is the (prompt, outcome) PAIR, and the
        # code cannot see outcomes -- so the CALLER declares how they attach. outcomes="per-prompt"
        # (the fail-safe default) means a prompt's outcome is a fixed property of the prompt:
        # duplicates repeat one Bernoulli, so they collapse to one certification row (300 copies
        # of one always-correct prompt certify from n=1, bound 0.975 -- the row count certified
        # 0.024 from one trial's worth of evidence), and duplicates whose verdicts DISAGREE are
        # refused as an oracle inconsistency. outcomes="per-row" is the caller's recorded
        # assertion that each row's outcome is an independent draw given the prompt (e.g. graded
        # against a per-row ground truth); rows then all count, and the receipt records that the
        # effective sample size rests on that declaration.
        rows_by_key: dict[Any, list[int]] = {}
        for row in range(n_certify):
            key = _seed_key(prompts[row])
            key = key if key is not None else repr(prompts[row])
            rows_by_key.setdefault(key, []).append(row)
        if outcomes == "per-prompt":
            # Oracle consistency is checked under BOTH sampling declarations: with a fixed
            # per-prompt outcome and a deterministic served decision, copies cannot disagree.
            for rows in rows_by_key.values():
                verdicts = {bool(errors[row]) for row in rows}
                if len(verdicts) > 1:
                    raise ValueError(
                        "duplicate certification prompts returned different verdicts under "
                        "outcomes='per-prompt': a prompt's served decision is deterministic, so "
                        "with a fixed per-prompt outcome its correctness cannot differ between "
                        "copies -- if each row is graded against its own independent outcome, "
                        "declare outcomes='per-row'"
                    )
        if outcomes == "per-row" or sampling == "iid-traffic":
            # Rows all count. per-row: each row's outcome is an independent draw given its prompt.
            # iid-traffic + per-prompt: the row indicators err(prompt_i) are i.i.d. Bernoulli of
            # the TRAFFIC accepted-error rate because the prompts themselves are the i.i.d. draw;
            # collapsing here would silently swap the estimand to uniform-over-distinct-prompts
            # and under-weight an error-prone heavy prompt (measured: certified 0.080 while the
            # served traffic risk was 0.37-0.41 on 40%-heavy traffic -- the pass-19 blocker).
            effective_rows = list(range(n_certify))
        else:
            effective_rows = sorted(rows[0] for rows in rows_by_key.values())
        certification_stats = np.asarray([statistics[row] for row in effective_rows], dtype=float)
        certification_errors = np.asarray([errors[row] for row in effective_rows], dtype=bool)
        proposal_stats = np.asarray(statistics[n_certify:], dtype=float)
        thresholds = np.unique(np.concatenate((np.asarray([-np.inf]), proposal_stats)))
        per_threshold_tail = (1.0 - self.confidence) / len(thresholds)

        candidates: list[dict[str, Any]] = []
        for threshold in thresholds:
            accepted_mask = certification_stats >= threshold
            accepted = int(accepted_mask.sum())
            if accepted == 0:
                continue
            n_errors = int(certification_errors[accepted_mask].sum())
            upper = _binomial_error_upper(n_errors, accepted, per_threshold_tail)
            candidates.append(
                {
                    "threshold": float(threshold),
                    "accepted": accepted,
                    "errors": n_errors,
                    "error_upper": upper,
                }
            )
        eligible = [candidate for candidate in candidates if candidate["error_upper"] <= self.alpha]
        chosen = min(eligible, key=lambda candidate: candidate["threshold"]) if eligible else None
        self.qhat = float(chosen["threshold"]) if chosen is not None else float("inf")
        # The bound this split could reach with ZERO observed errors. It depends only on the certification
        # count and the Bonferroni-corrected tail, never on the model, so alpha below it is unreachable no
        # matter how good the model is -- calibration then abstains everywhere and the abstention is a fact
        # about the calibration set's size, not about the model. Reported so a caller can tell those two
        # apart: without it, "no threshold certified" looks identical in both cases and invites tuning the
        # generator against a target it cannot reach. ``None`` when nothing was accepted at any threshold.
        widest_accepted = max((candidate["accepted"] for candidate in candidates), default=0)
        best_case = _binomial_error_upper(0, widest_accepted, per_threshold_tail) if widest_accepted else None
        self.risk_receipt = {
            "method": "split-selective-risk/clopper-pearson-bonferroni/v1",
            "target_error": self.alpha,
            "confidence": self.confidence,
            "proposal_count": len(prompts) - n_certify,
            "certification_count": n_certify,
            "oracle_calls": n_certify,
            "thresholds_tested": len(thresholds),
            "attainable_error_upper": best_case,
            "target_attainable": None if best_case is None else bool(best_case <= self.alpha),
            "threshold": ("inf" if np.isposinf(self.qhat) else "-inf" if np.isneginf(self.qhat) else self.qhat),
            "statistic": "top_score",
            "candidate_count": self.k,
            "seed": rng_seed,
            "seed_schedule": "prompt-only (identical to serving)",
            "unique_prompt_count": len({repr(prompt) for prompt in prompts}),
            # the binomial bound's actual sample size under the declared regime (RR18-04 / pass 19)
            "certification_effective_count": len(effective_rows),
            "outcome_declaration": outcomes,
            "sampling_declaration": sampling,
            "certified_estimand": (
                "per-row accepted-error rate (each row's outcome declared an independent draw given its prompt)"
                if outcomes == "per-row"
                else "traffic-weighted accepted-error rate (rows declared an i.i.d. draw of serving traffic; "
                "duplicates carry their traffic weight)"
                if sampling == "iid-traffic"
                else "accepted-error rate UNIFORM over the distinct certification prompts (duplicates "
                "collapsed; NOT traffic-weighted -- repeat-heavy serving is not covered by this certificate)"
            ),
            "accepted": 0 if chosen is None else chosen["accepted"],
            "errors": 0 if chosen is None else chosen["errors"],
            "error_upper": None if chosen is None else chosen["error_upper"],
            # MXR-080-1894: the certificate names the policy it certifies instead of merely assuming
            # one. ``_certified_policy`` below is the enforcing half; this is the readable half that
            # survives serialization.
            "policy": self._describe_policy(),
            "assumptions": [
                "calibration certification and serving cases are exchangeable",
                "certification prompts are distinct enough that their accept/error events are "
                "independent -- duplicated prompts repeat the identical served decision, so the "
                "binomial bound's effective sample is unique_prompt_count, not certification_count",
                "the certified generate/score callables behave identically at serving time "
                "(their identity is enforced; their internal state is not observable here)",
            ],
        }
        self._certified_policy = self._policy()
        return self

    def candidate_set(self, prompt: Any, *, seed: int | None = None) -> list[Any]:
        """Return ``[best_candidate]`` when its calibrated statistic clears the risk gate, else ``[]``."""
        if self.qhat is None:
            raise RuntimeError("call calibrate(...) (or set qhat) before candidate_set(...)")
        self._require_certified_policy()
        if seed is not None and self.risk_receipt is not None:
            # STAT-RR17-07: an explicit per-call seed is a different stochastic policy from the one
            # the certificate covers -- the exact mechanism behind certify-3.7%-serve-100%.
            raise ValueError(
                "candidate_set/serve seed overrides are refused on a certified generator: the "
                "certificate covers only the prompt-derived schedule; build a generator with the "
                "desired seed and calibrate it"
            )
        call_seed = _derive_seed(self.seed, prompt) if seed is None else int(seed)
        candidate, statistic = self._selection(prompt, seed=call_seed)
        return [candidate] if statistic >= self.qhat else []

    def serve(self, prompt: Any, *, seed: int | None = None) -> Any:
        """Return the best candidate only when the certified risk gate accepts, else :data:`ABSTAIN`."""
        admitted = self.candidate_set(prompt, seed=seed)
        return admitted[0] if len(admitted) == 1 else ABSTAIN

    def decide(self, prompt: Any, *, seed: int | None = None) -> Any:
        """Alias for :meth:`serve` with the same name as :meth:`CalibratedTaskModel.decide`, so a
        ``CalibratedGenerator`` drops into :class:`~mixle.task.cascade.Cascade` unmodified."""
        return self.serve(prompt, seed=seed)

    def __call__(self, prompt: Any, *, seed: int | None = None) -> Any:
        return self.serve(prompt, seed=seed)

    def abstention_rate(self, prompts: Sequence[Any], *, seed: int | None = None) -> float:
        """Empirical fraction of ``prompts`` that would abstain -- the generation analogue of
        :meth:`CalibratedTaskModel.escalation_rate`."""
        prompts = list(prompts)
        if not prompts:
            return 0.0
        outcomes = [self.serve(p, seed=seed) for p in prompts]
        return float(np.mean([o is ABSTAIN for o in outcomes]))


__all__ = ["ABSTAIN", "CalibratedGenerator", "smallest_certifiable_calibration_set"]
