"""``learn_inverse`` -- amortized posteriors ``q(theta | y)`` for a simulator, with calibration receipts.

Simulation-based inference done the mixle way: given a forward simulator ``g: theta -> y`` (a bare
Python callable -- no ``mixle.task.imagine``/M2 program required) and a prior ``p(theta)`` (any
fitted mixle ``Model``), :func:`learn_inverse` trains a torch CONDITIONAL density student
``q(theta | y)`` on simulated ``(theta, y)`` pairs, then ships it wrapped as an
:class:`~mixle.inference.condition.Posterior` (M0's type) so downstream ``condition``/``do``
composition and B7 treat a learned inverse exactly like an exactly-conditioned one -- except its
``.receipt`` carries an explicit amortization warning plus a pointer to :class:`InverseReceipts`,
because a trained student is an APPROXIMATION and the whole point of this module is to ship the
numbers that say whether to trust it.

Convention (matches ``build_mdn``/``build_conditional_flow``'s own ``log_density(x, y)``/
``sample_given(x)`` contract): ``theta`` -- the quantity being inferred -- is the student's
``y``-ARGUMENT, and the observed data ``y`` is its ``x``-argument. So ``q(theta | y_obs)`` is
``module.sample_given(y_obs) -> theta`` and its density is ``module.log_density(y_obs, theta)`` --
the inverse of the simulator's own arrow.

The student is trained through the vendored :class:`~mixle.models.grad_leaf.GradLeaf` (a bare torch
module IS the model), not the simpler :class:`~mixle.models.mixture_density.NeuralConditionalDensity`
adapter -- see ``notes/designs/M3.md`` for why: sequential refinement (below) re-scores the module
against freshly generated round data before the next fit commits, which wants the generic
``seq_log_density`` path ``GradLeaf`` gives any bare module (warm-started across ``optimize()``
calls via the SAME underlying ``nn.Module`` object) rather than a second bespoke accumulator.
``NeuralConditionalDensity`` remains the simpler documented alternative for callers who don't need
round-conditioned rescoring.

Algorithm (``notes/designs/M3.md``):

1. **Pair generation (round 1).** ``theta_i ~ p(theta)`` via the prior's own sampler; ``y_i =
   simulator(theta_i)`` in a plain Python loop (no batching assumed on ``simulator``).
2. **Student.** ``build_conditional_flow``/``build_mdn`` wrapped in :class:`GradLeaf`, fit via
   ``optimize(list(zip(y_pairs, theta_pairs)), leaf, ...)`` -- A4.4's tuple-default-loss fix is what
   lets the bare module's two-arg ``log_density(x, y)`` score straight off tuple observations.
3. **Proposal-corrected sequential refinement (rounds 2..R).** Each round draws from the frozen
   current proposal ``q_r(theta | y_obs)``, records ``p(theta) / q_r(theta | y_obs)`` importance
   weights, and refits on ALL retained rounds. Round 1 prior draws keep global support; every later
   block is self-normalized to equal round mass. Thus every round's weighted conditional-density
   objective targets the original prior joint ``p(theta)p(y|theta)`` -- never the proposal joint.
4. **Optional finite-particle target correction.** ``reweight=True`` with
   ``true_log_likelihood(theta, y_obs)`` turns the final neural posterior into a self-normalized
   importance proposal. The returned model stores those particles and normalized weights, and
   ``posterior(y_obs)`` samples and computes means from that weighted empirical posterior. It is an
   importance-resampled approximation, not an exact continuous density; ESS and a warning make that
   contract explicit.
5. **Calibration receipts (always computed).**

   - **SBC.** Ties are randomized and every discrete rank cell is jittered into a continuous
     uniform variate. Each parameter dimension gets an equal-probability-bin chi-square test;
     a Bonferroni-adjusted minimum p-value supplies a valid global test under arbitrary
     cross-dimension dependence.
   - **Coverage.** Per-dimension empirical coverage ships with simultaneous 99% Wilson intervals;
     a level passes only when its nominal probability lies inside every dimension's interval.
   - **Prior-predictive.** Empirical mean/std of the round-1 simulated ``y_i``'s, plus (when
     ``y_obs`` is given) its per-dimension z-score against that empirical distribution -- a
     caller-facing warning, not a gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from mixle.inference.condition import ConditionReceipt, Posterior
from mixle.inference.estimation import optimize
from mixle.models.grad_leaf import GradLeaf
from mixle.utils.immutable import detach_receipt_container

__all__ = ["InverseModel", "InverseReceipts", "learn_inverse"]


def _torch() -> Any:
    import torch

    return torch


def _next_seed(rng: np.random.RandomState) -> int:
    return int(rng.randint(0, 2**31 - 1))


def _as_rows(
    arr: Any,
    n: int,
    *,
    context: str = "sampler",
    expected_width: int | None = None,
) -> np.ndarray:
    """Normalize a sampler's ``sample(n)`` output to shape ``(n, d)``. A univariate sampler
    (e.g. ``GaussianDistribution``) returns a flat ``(n,)`` array of scalar draws -- ``atleast_2d``
    on that would misread it as ONE row of ``n`` dimensions instead of ``n`` rows of one dimension,
    so a 1-D result of length ``n`` is reshaped to ``(n, 1)`` explicitly rather than via ``atleast_2d``."""
    try:
        a = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must return a rectangular numeric sample array") from exc
    if a.ndim == 0:
        if n != 1:
            raise ValueError(f"{context} returned one scalar for {n} requested samples")
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        if n == 1:
            a = a.reshape(1, -1)
        elif a.size == n:
            a = a.reshape(n, 1)
        else:
            raise ValueError(f"{context} returned {a.size} values for {n} requested samples")
    elif a.ndim != 2:
        raise ValueError(f"{context} samples must be scalar or vector rows")
    if a.shape[0] != n or a.shape[1] == 0:
        raise ValueError(f"{context} must return exactly {n} non-empty sample rows")
    if expected_width is not None and a.shape[1] != expected_width:
        raise ValueError(f"{context} must return vectors of width {expected_width}, got {a.shape[1]}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{context} returned non-finite samples")
    return a


def _simulate_rows(
    simulator: Callable[[Any], Any],
    thetas: np.ndarray,
    *,
    expected_width: int | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, theta in enumerate(thetas):
        try:
            row = np.asarray(simulator(theta), dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"simulator output {index} must be numeric") from exc
        if row.size == 0 or not np.all(np.isfinite(row)):
            raise ValueError(f"simulator output {index} must be a non-empty finite vector")
        if expected_width is not None and row.size != expected_width:
            raise ValueError(f"simulator output {index} has width {row.size}; expected {expected_width}")
        if rows and row.size != rows[0].size:
            raise ValueError("simulator outputs must have one consistent vector width")
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _module_placement(module: Any) -> tuple[Any, Any]:
    torch = _torch()
    tensor = next(iter((*module.parameters(), *module.buffers())), None)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    return tensor.device, dtype


def _restore_training_modes(states: list[tuple[Any, bool]]) -> None:
    for child, training in states:
        child.training = training


def _build_student(family: str, *, x_dim: int, y_dim: int, hidden: int, seed: int | None) -> Any:
    from mixle.models.mixture_density import build_conditional_flow, build_mdn

    # nn.Module weight init draws from torch's GLOBAL rng, not our own seeded RandomState -- seed it
    # explicitly here so a given `seed=` to learn_inverse determines the student's starting weights
    # too (determinism is a contract, rule 0.2.4), independent of whatever global torch state a
    # caller/earlier test left behind.
    torch = _torch()
    with torch.random.fork_rng():
        torch.manual_seed(int(seed) if seed is not None else 0)
        if family == "flow":
            return build_conditional_flow(x_dim, y_dim, hidden=hidden)
        if family == "mdn":
            return build_mdn(x_dim, y_dim, hidden=hidden)
    raise ValueError(f"family must be 'flow' or 'mdn', got {family!r}")


def _generate_pairs(
    prior: Any, simulator: Callable[[Any], Any], n: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` fresh ``(theta_i, y_i)`` pairs: ``theta_i ~ prior``, ``y_i = simulator(theta_i)`` (a plain
    Python loop -- no batching assumption on ``simulator``)."""
    thetas = _as_rows(
        prior.sampler(seed=seed).sample(int(n)),
        int(n),
        context="prior sampler",
    )
    ys = _simulate_rows(simulator, thetas)
    return thetas, ys


def _sample_given(module: Any, x_row: np.ndarray, n: int, *, seed: int | None) -> np.ndarray:
    """``n`` draws of ``theta ~ q(theta | x_row)`` from a fitted student ``module``."""
    torch = _torch()
    n = _positive_integer(n, "posterior sample count")
    x_row = np.asarray(x_row, dtype=float).reshape(-1)
    if x_row.size == 0 or not np.all(np.isfinite(x_row)):
        raise ValueError("posterior conditioning input must be a non-empty finite vector")
    rng = np.random.RandomState(seed)
    draw_seed = _next_seed(rng)
    device, dtype = _module_placement(module)
    xt = torch.as_tensor(np.tile(x_row, (n, 1)), dtype=dtype, device=device)
    states = [(child, child.training) for child in module.modules()]
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    try:
        module.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(draw_seed)
            output = module.sample_given(xt)
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        _restore_training_modes(states)
    if not torch.is_tensor(output):
        raise ValueError("conditional student sample_given must return a tensor")
    return _as_rows(
        output.detach().cpu().numpy(),
        n,
        context="conditional student sample_given",
    )


def _log_density_given(module: Any, x_row: np.ndarray, theta_batch: np.ndarray) -> np.ndarray:
    """``log q(theta | x_row)`` for every row of ``theta_batch`` (shape ``(n, theta_dim)``)."""
    torch = _torch()
    x_row = np.asarray(x_row, dtype=float).reshape(-1)
    theta_batch = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    if (
        x_row.size == 0
        or theta_batch.shape[1] == 0
        or not np.all(np.isfinite(x_row))
        or not np.all(np.isfinite(theta_batch))
    ):
        raise ValueError("density inputs must be non-empty finite vectors")
    n = theta_batch.shape[0]
    device, dtype = _module_placement(module)
    xt = torch.as_tensor(np.tile(x_row, (n, 1)), dtype=dtype, device=device)
    yt = torch.as_tensor(theta_batch, dtype=dtype, device=device)
    states = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        with torch.no_grad():
            output = module.log_density(xt, yt)
    finally:
        _restore_training_modes(states)
    if not torch.is_tensor(output) or output.numel() != n:
        raise ValueError("conditional student log_density must return exactly one tensor score per row")
    scores = output.detach().cpu().numpy().reshape(-1)
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("conditional student log_density returned invalid scores")
    return scores


def _fit_round(
    module: Any,
    ys: np.ndarray,
    thetas: np.ndarray,
    *,
    m_steps: int,
    lr: float,
    max_its: int,
    weights: np.ndarray | None = None,
) -> Any:
    """One ``optimize()`` call against ``(y, theta)`` pairs, warm-started from ``module``'s own weights
    (same underlying ``nn.Module`` object -- ``GradEstimator.estimate`` mutates it in place)."""
    data = list(zip(ys.tolist(), thetas.tolist()))
    leaf = GradLeaf(module, m_steps=m_steps, lr=lr)
    if weights is not None:
        row_weights = np.asarray(weights, dtype=float)
        if (
            row_weights.shape != (len(data),)
            or not np.all(np.isfinite(row_weights))
            or np.any(row_weights < 0.0)
            or float(row_weights.sum()) <= 0.0
        ):
            raise ValueError("inverse training weights must be finite, non-negative, and match all rows")
        estimator = leaf.estimator()
        encoded = leaf.dist_to_encoder().seq_encode(data)
        for _ in range(int(max_its)):
            accumulator = estimator.accumulator_factory().make()
            accumulator.seq_update(encoded, row_weights, leaf)
            leaf = estimator.estimate(float(row_weights.sum()), accumulator.value())
        return leaf.module
    fitted = optimize(data, leaf, max_its=max_its, out=None)
    return fitted.module


def _normalized_importance_weights(log_weights: Any, *, context: str) -> tuple[np.ndarray, float]:
    """Normalize log importance ratios and return weights plus their ESS.

    Non-finite entries receive zero mass. A block with no finite positive mass
    cannot represent the target and fails closed.
    """
    log_w = np.asarray(log_weights, dtype=float).reshape(-1)
    finite = np.isfinite(log_w)
    if not finite.any():
        raise ValueError(f"{context} produced no finite importance weights")
    shifted = np.full_like(log_w, float("-inf"))
    shifted[finite] = log_w[finite] - float(np.max(log_w[finite]))
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{context} produced no positive importance mass")
    weights /= total
    ess = float(1.0 / np.sum(weights**2))
    return weights, ess


def _scalar_log_value(value: Any, *, context: str) -> float:
    """Accept scalar or length-one log-density results without lossy coercion."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{context} must return exactly one log-density value per parameter row")
    result = float(array.reshape(-1)[0])
    if np.isnan(result) or np.isposinf(result):
        raise ValueError(f"{context} returned an invalid log-density value")
    return result


def _posterior_sharpness(module: Any, y_obs: np.ndarray, theta_dim: int, *, n: int, seed: int | None) -> float:
    """A scalar "how spread out is q(theta | y_obs)" receipt -- sum of per-dimension TRIMMED sample
    variance (middle 80%, i.e. the top/bottom decile of draws dropped before computing variance).
    Lower is sharper; used to assert refinement rounds measurably sharpen the posterior (test (e)).

    Plain ``np.var`` is not robust to the occasional extreme-value draw an under-trained conditional
    flow can produce -- a single sample landing far outside the flow's well-conditioned region (a
    known pathology, more likely exactly in the small-``n_sims`` regime this receipt is meant to
    characterize) can inflate raw variance by orders of magnitude without the bulk of the posterior
    mass having moved at all. Trimming keeps the receipt reporting the posterior's actual central
    spread instead of one unlucky sample.
    """
    samples = _as_rows(
        _sample_given(module, y_obs, n, seed=seed),
        n,
        context="posterior sharpness sampler",
        expected_width=theta_dim,
    )
    trimmed = np.sort(samples, axis=0)
    cut = int(0.1 * len(trimmed))
    if cut > 0:
        trimmed = trimmed[cut:-cut]
    return float(np.sum(np.var(trimmed, axis=0)))


def _calibration_receipts(
    module: Any,
    prior: Any,
    simulator: Callable[[Any], Any],
    *,
    theta_dim: int,
    y_dim: int,
    n_replications: int,
    n_posterior_samples: int,
    coverage_levels: tuple[float, ...],
    seed: int | None,
) -> tuple[
    float,
    float,
    int,
    list[float],
    dict[float, float],
    dict[float, list[float]],
    dict[float, list[tuple[float, float]]],
    dict[float, bool],
]:
    """Randomized-rank SBC and uncertainty-aware marginal coverage.

    Each dimension gets its own chi-square test. Their p-values are combined
    with a Bonferroni minimum, which remains valid when posterior dimensions
    are correlated. Coverage uses simultaneous Wilson intervals across every
    requested level and dimension.
    """
    rng = np.random.RandomState(seed)
    randomized_ranks = np.zeros((n_replications, theta_dim), dtype=float)
    covered = {c: np.zeros((n_replications, theta_dim), dtype=bool) for c in coverage_levels}
    for r in range(n_replications):
        theta_star = _as_rows(
            prior.sampler(seed=_next_seed(rng)).sample(1),
            1,
            context="prior sampler during calibration",
            expected_width=theta_dim,
        )[0]
        y_star = _simulate_rows(
            simulator,
            theta_star.reshape(1, -1),
            expected_width=y_dim,
        )[0]
        samples = _as_rows(
            _sample_given(module, y_star, n_posterior_samples, seed=_next_seed(rng)),
            n_posterior_samples,
            context="posterior sampler during calibration",
            expected_width=theta_dim,
        )
        for d in range(theta_dim):
            less = int(np.sum(samples[:, d] < theta_star[d]))
            ties = int(np.sum(samples[:, d] == theta_star[d]))
            # Exchangeability makes the rank position discrete-uniform on
            # 0..S. Random tie placement plus within-cell jitter turns it into
            # a continuous U(0,1) variate suitable for equal-probability bins.
            position = less + int(rng.randint(0, ties + 1))
            randomized_ranks[r, d] = (position + float(rng.uniform())) / (n_posterior_samples + 1)
            for c in coverage_levels:
                lo_q, hi_q = (1.0 - c) / 2.0, (1.0 + c) / 2.0
                lo, hi = np.quantile(samples[:, d], [lo_q, hi_q])
                covered[c][r, d] = bool(lo <= theta_star[d] <= hi)

    bins = max(2, min(20, n_replications // 5))
    edges = np.linspace(0.0, 1.0, bins + 1)
    statistics: list[float] = []
    pvalues: list[float] = []
    for d in range(theta_dim):
        hist, _ = np.histogram(randomized_ranks[:, d], bins=edges)
        expected = n_replications / bins
        statistic = float(np.sum((hist - expected) ** 2 / expected))
        statistics.append(statistic)
        pvalues.append(float(chi2.sf(statistic, bins - 1)))
    # Valid under arbitrary cross-dimension dependence; unlike summing the
    # statistics, this makes no independence claim.
    global_pvalue = min(1.0, theta_dim * min(pvalues))

    coverage_emp: dict[float, float] = {}
    coverage_by_dimension: dict[float, list[float]] = {}
    coverage_intervals: dict[float, list[tuple[float, float]]] = {}
    coverage_pass: dict[float, bool] = {}
    n_tests = max(theta_dim * len(coverage_levels), 1)
    z = float(norm.ppf(1.0 - (0.01 / n_tests) / 2.0))
    for level in coverage_levels:
        successes = covered[level].sum(axis=0)
        proportions = successes / n_replications
        intervals: list[tuple[float, float]] = []
        for count in successes:
            phat = float(count) / n_replications
            denominator = 1.0 + z**2 / n_replications
            center = (phat + z**2 / (2.0 * n_replications)) / denominator
            radius = z * np.sqrt(phat * (1.0 - phat) / n_replications + z**2 / (4.0 * n_replications**2)) / denominator
            intervals.append((max(0.0, center - radius), min(1.0, center + radius)))
        coverage_emp[level] = float(np.mean(proportions))
        coverage_by_dimension[level] = [float(value) for value in proportions]
        coverage_intervals[level] = intervals
        coverage_pass[level] = all(lo <= level <= hi for lo, hi in intervals)

    return (
        max(statistics),
        global_pvalue,
        bins,
        pvalues,
        coverage_emp,
        coverage_by_dimension,
        coverage_intervals,
        coverage_pass,
    )


def _prior_predictive_receipt(ys: np.ndarray, y_obs: np.ndarray | None) -> dict[str, Any]:
    mean = ys.mean(axis=0)
    std = ys.std(axis=0) + 1e-12
    report: dict[str, Any] = {"y_mean": mean.tolist(), "y_std": std.tolist(), "n": int(ys.shape[0])}
    if y_obs is None:
        report["y_obs_zscore"] = None
        report["in_distribution_warning"] = None
        return report
    y_obs_arr = np.atleast_1d(np.asarray(y_obs, dtype=float))
    z = (y_obs_arr - mean) / std
    report["y_obs_zscore"] = z.tolist()
    report["max_abs_zscore"] = float(np.max(np.abs(z)))
    report["in_distribution_warning"] = bool(np.max(np.abs(z)) > 3.0)
    return report


def _reweight_receipt(
    module: Any,
    prior: Any,
    true_log_likelihood: Callable[[Any, Any], float],
    y_obs: np.ndarray,
    *,
    n: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float, list[str]]:
    """Self-normalized importance reweighting of ``q(theta | y_obs)`` against the true likelihood
    (same log-sum-exp construction as ``mixle.inference.condition``'s SIR fallback)."""
    thetas = _sample_given(module, y_obs, n, seed=seed)
    log_q = _log_density_given(module, y_obs, thetas)
    log_prior = np.array([_scalar_log_value(prior.log_density(theta), context="prior.log_density") for theta in thetas])
    log_lik = np.array(
        [
            _scalar_log_value(
                true_log_likelihood(theta, y_obs),
                context="true_log_likelihood",
            )
            for theta in thetas
        ]
    )
    log_w = log_prior + log_lik - log_q
    w_norm, ess = _normalized_importance_weights(log_w, context="inverse target correction")

    warnings = [
        "reweight=True returns a finite-particle self-normalized importance posterior; "
        "it is target-corrected but is not an exact continuous posterior density."
    ]
    ess_ratio = ess / n
    if ess_ratio < 0.01:
        warnings.append(f"reweight ESS ratio {ess_ratio:.4f} < 0.01 -- reweighted posterior is not trustworthy.")
    return thetas, w_norm, ess, ess_ratio, warnings


@dataclass
class InverseReceipts:
    """The calibration report that ships with every :class:`InverseModel` -- tells the caller
    whether to trust ``q(theta | y)``, not just a point estimate."""

    sbc_statistic: float
    sbc_pvalue: float
    sbc_bins: int
    sbc_replications: int
    sbc_pass: bool  # dependence-safe Bonferroni global p-value > 0.01
    coverage: dict[float, float]  # nominal level -> empirical coverage
    coverage_pass: dict[float, bool]  # nominal lies in every simultaneous Wilson interval
    prior_predictive: dict[str, Any]
    rounds_trained: int
    sbc_pvalues_by_dimension: list[float] = field(default_factory=list)
    sbc_method: str = "randomized_rank_bonferroni"
    coverage_by_dimension: dict[float, list[float]] = field(default_factory=dict)
    coverage_intervals: dict[float, list[tuple[float, float]]] = field(default_factory=dict)
    coverage_method: str = "simultaneous_99pct_wilson"
    sharpness_by_round: list[float] = field(default_factory=list)
    round_training: list[dict[str, Any]] = field(default_factory=list)
    ess: float | None = None
    ess_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sever the caller's alias on every container this record holds (MXR-080-1894).

        A calibration report is evidence about a fit that already happened. Every container field
        here was stored by reference, so whoever built or passed one kept a live handle: mutating
        ``warnings`` or ``round_training`` afterwards silently rewrote a receipt that had already
        been produced, and nothing re-validates. :func:`~mixle.utils.immutable.detach_receipt_container`
        (not the stronger freezer) because these fields' concrete types are load-bearing --
        ``coverage`` is keyed by float and read as a ``dict``, ``coverage_intervals`` holds lists of
        tuples that callers index, and this dataclass is not frozen, so a ``mappingproxy`` here would
        change the observable type of a documented field for no additional protection.

        Deliberately NOT done: the dataclass is left mutable. Making it frozen is the stronger fix,
        but ``ConditionReceipt`` was frozen by MXR-080-1876 and that silently broke every
        :meth:`InverseModel.posterior` call (see :class:`InverseConditionReceipt`); repeating the
        move on a record with more consumers, in a change whose regression tests are ``slow``-marked
        and therefore outside the default ``-m fast`` gate, is not a trade worth making here.
        """
        for name in (
            "coverage",
            "coverage_pass",
            "prior_predictive",
            "sbc_pvalues_by_dimension",
            "coverage_by_dimension",
            "coverage_intervals",
            "sharpness_by_round",
            "round_training",
            "warnings",
        ):
            setattr(self, name, detach_receipt_container(getattr(self, name)))


@dataclass(frozen=True)
class InverseConditionReceipt(ConditionReceipt):
    """A :class:`~mixle.inference.condition.ConditionReceipt` that can carry its inverse receipts.

    MXR-080-1894. :meth:`InverseModel.posterior` used to attach the calibration report by assigning
    an undeclared attribute onto a plain ``ConditionReceipt``. MXR-080-1876 then froze that class --
    correctly, it is a record -- and a frozen dataclass refuses assignment to *any* name, declared or
    not. So every ``posterior()`` call, on both the amortized and the target-corrected path, began
    raising ``FrozenInstanceError``: the method was completely non-functional. It went unnoticed
    because ``mixle/tests/inverse_test.py`` needs ``torch`` and is ``slow``-marked, so the default
    ``-m fast`` run never collected it.

    Declaring the field on a subclass fixes it inside :mod:`mixle.task.inverse` and keeps the freeze
    intact: the pointer is now part of the record rather than smuggled past its constructor,
    ``isinstance(receipt, ConditionReceipt)`` still holds, and ``receipt.inverse_receipts`` reads
    exactly as the docstring and the existing test already promise.
    """

    inverse_receipts: InverseReceipts | None = None


class InverseModel:
    """A fitted amortized posterior ``q(theta | y)`` plus its :class:`InverseReceipts`."""

    def __init__(
        self,
        *,
        module: Any,
        prior: Any,
        simulator: Callable[[Any], Any],
        family: str,
        theta_dim: int,
        y_dim: int,
        receipts: InverseReceipts,
        seed: int | None = None,
        reweighted_y: np.ndarray | None = None,
        reweighted_particles: np.ndarray | None = None,
        reweighted_weights: np.ndarray | None = None,
    ) -> None:
        theta_dim = _positive_integer(theta_dim, "theta_dim")
        y_dim = _positive_integer(y_dim, "y_dim")
        if not isinstance(receipts, InverseReceipts):
            raise TypeError("receipts must be an InverseReceipts instance")
        correction_parts = (
            reweighted_y is not None,
            reweighted_particles is not None,
            reweighted_weights is not None,
        )
        if any(correction_parts) and not all(correction_parts):
            raise ValueError("target correction requires y, particles, and normalized weights together")
        if all(correction_parts):
            # MXR-080-1894: `np.asarray` does NOT copy an array that already has the requested dtype,
            # and `reshape` returns a view -- so every check below used to be run against storage the
            # CALLER still owned. A caller that mutated its own array afterwards silently invalidated
            # all of them at once: a validated weight vector summing to 1.0 became 5.75 and validated
            # finite particles picked up a NaN, with no check ever re-run and the corrupted state
            # flowing straight into `posterior()`'s weighted sampler and mean. Copy first, then
            # validate the copy, then freeze it -- the same "checked once, true for the object's
            # lifetime" discipline `HypothesisPortfolio.weights` already uses.
            reweighted_y = np.array(reweighted_y, dtype=float).reshape(-1)
            reweighted_particles = np.array(
                _as_rows(
                    reweighted_particles,
                    len(reweighted_weights),
                    context="reweighted particles",
                    expected_width=theta_dim,
                ),
                dtype=float,
            )
            reweighted_weights = np.array(reweighted_weights, dtype=float).reshape(-1)
            if (
                reweighted_y.shape != (y_dim,)
                or not np.all(np.isfinite(reweighted_y))
                or reweighted_weights.shape != (len(reweighted_particles),)
                or not np.all(np.isfinite(reweighted_weights))
                or np.any(reweighted_weights < 0.0)
                or not np.isclose(float(reweighted_weights.sum()), 1.0, atol=1e-12)
            ):
                raise ValueError("target-correction arrays must be finite, aligned, and normalized")
            for array in (reweighted_y, reweighted_particles, reweighted_weights):
                array.flags.writeable = False
        self.module = module
        self.prior = prior
        self.simulator = simulator
        self.family = family
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.receipts = receipts
        self._seed = seed
        self._reweighted_y = reweighted_y
        self._reweighted_particles = reweighted_particles
        self._reweighted_weights = reweighted_weights

    def posterior(self, y: Any) -> Posterior:
        """Wrap ``q(theta | y)`` as an M0 :class:`~mixle.inference.condition.Posterior`: ``sample(n)``
        / ``log_density(theta)`` / ``mean(field)`` / ``.receipt`` -- so downstream condition/do
        composition treats a learned inverse like an exactly-conditioned one, modulo the amortization
        warning on ``.receipt`` and the ``InverseReceipts`` pointer at ``.receipt.inverse_receipts``."""
        y_row = np.atleast_1d(np.asarray(y, dtype=float))
        if y_row.shape != (self.y_dim,) or not np.all(np.isfinite(y_row)):
            raise ValueError(f"posterior observation must be a finite vector of width {self.y_dim}")
        module = self.module
        base_seed = self._seed

        if self._reweighted_particles is not None:
            if self._reweighted_y is None or not np.array_equal(y_row, self._reweighted_y):
                raise ValueError(
                    "this target-corrected inverse is bound to the y_obs used by learn_inverse; "
                    "fit another model or disable reweight for a different observation"
                )
            particles = self._reweighted_particles
            weights = self._reweighted_weights
            if weights is None:
                raise RuntimeError("target-corrected inverse is missing its normalized weights")

            def weighted_sample_fn(n: int, s: int | None) -> np.ndarray:
                rng = np.random.RandomState(s if s is not None else base_seed)
                indices = rng.choice(len(particles), size=n, replace=True, p=weights)
                return np.asarray(particles[indices], dtype=float).copy()

            def weighted_mean_fn(path: tuple[int, ...]) -> float:
                if len(path) > 1:
                    raise ValueError("inverse posterior fields are one-dimensional parameter indices")
                idx = int(path[0]) if len(path) else 0
                if idx < 0 or idx >= self.theta_dim:
                    raise ValueError(f"posterior field must be in [0, {self.theta_dim})")
                return float(np.dot(weights, particles[:, idx]))

            receipt = InverseConditionReceipt(
                method="sir",
                sample_contract="theta_particles",
                ess=self.receipts.ess,
                ess_ratio=self.receipts.ess_ratio,
                n_particles=len(particles),
                warnings=list(self.receipts.warnings),
                inverse_receipts=self.receipts,
            )
            return Posterior(
                sample_fn=weighted_sample_fn,
                log_density_fn=None,
                mean_fn=weighted_mean_fn,
                receipt=receipt,
                model=None,
            )

        def sample_fn(n: int, s: int | None) -> np.ndarray:
            return _sample_given(module, y_row, n, seed=s if s is not None else base_seed)

        def log_density_fn(theta: Any) -> float:
            theta_array = np.asarray(theta, dtype=float)
            if theta_array.ndim == 0:
                theta_array = theta_array.reshape(1)
            theta_row = np.atleast_2d(theta_array)
            if theta_row.shape != (1, self.theta_dim) or not np.all(np.isfinite(theta_row)):
                raise ValueError(
                    f"posterior density query must be one finite parameter vector of width {self.theta_dim}"
                )
            return float(_log_density_given(module, y_row, theta_row)[0])

        def mean_fn(path: tuple[int, ...]) -> float:
            if len(path) > 1:
                raise ValueError("inverse posterior fields are one-dimensional parameter indices")
            idx = int(path[0]) if len(path) else 0
            if idx < 0 or idx >= self.theta_dim:
                raise ValueError(f"posterior field must be in [0, {self.theta_dim})")
            samples = _sample_given(module, y_row, 500, seed=base_seed)
            return float(np.mean(samples[:, idx]))

        receipt = InverseConditionReceipt(
            method="amortized",
            warnings=[
                "InverseModel.posterior: a LEARNED amortized approximation "
                "(mixle.task.inverse.learn_inverse), not exact conditioning -- see "
                ".receipt.inverse_receipts (SBC/coverage/prior-predictive/ESS) before trusting it."
            ],
            inverse_receipts=self.receipts,  # the full calibration report (not an M0 field)
        )
        return Posterior(
            sample_fn=sample_fn, log_density_fn=log_density_fn, mean_fn=mean_fn, receipt=receipt, model=None
        )


def learn_inverse(
    simulator: Callable[[Any], Any],
    prior: Any,
    *,
    family: str = "flow",
    n_sims: int = 2000,
    rounds: int = 1,
    n_sbc_replications: int = 200,
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
    reweight: bool = False,
    true_log_likelihood: Callable[[Any, Any], float] | None = None,
    y_obs: Any = None,
    seed: int | None = None,
    m_steps: int = 200,
    lr: float = 5e-3,
    max_its: int = 1,
    hidden: int = 32,
    n_posterior_samples: int = 200,
    n_reweight_samples: int = 500,
) -> InverseModel:
    """Learn an amortized posterior ``q(theta | y)`` for simulator ``g: theta -> y`` under prior
    ``p(theta)``. See the module docstring for the full algorithm and the calibration receipts
    computed unconditionally.

    ``family="flow"`` (``build_conditional_flow``) requires ``theta`` (the quantity being inferred,
    the student's ``y``-argument) to be >= 2-dimensional -- ``build_conditional_flow`` needs
    ``y_dim >= 2`` for its coupling layers to be non-trivial (see its own docstring). A 1-D ``theta``
    (e.g. a scalar-parameter inverse problem) must use ``family="mdn"``, which has no such
    restriction (a mixture of per-component Gaussians is well-defined for scalar ``theta`` too, and
    is the more direct fit for asserting multimodality component-by-component).

    ``rounds > 1`` performs proposal-corrected SNPE toward a specific observation. Round-one prior
    simulations are retained forever. Every later proposal block is weighted by
    ``p(theta)/q_round(theta|y_obs)`` and the accumulated weighted objective continues to target the
    declared prior posterior rather than the narrower proposal posterior.

    ``reweight=True`` requires ``y_obs`` and returns an observation-bound, finite-particle
    importance-resampled posterior. It improves the target measure when its ESS is healthy but does
    not claim an exact continuous density.
    """
    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if not callable(getattr(prior, "sampler", None)):
        raise TypeError("prior must expose sampler(seed=...).sample(n)")
    if family not in ("flow", "mdn"):
        raise ValueError(f"family must be 'flow' or 'mdn', got {family!r}")
    n_sims = _positive_integer(n_sims, "n_sims")
    rounds = _positive_integer(rounds, "rounds")
    n_sbc_replications = _positive_integer(
        n_sbc_replications,
        "n_sbc_replications",
        minimum=10,
    )
    n_posterior_samples = _positive_integer(
        n_posterior_samples,
        "n_posterior_samples",
        minimum=2,
    )
    n_reweight_samples = _positive_integer(
        n_reweight_samples,
        "n_reweight_samples",
        minimum=2,
    )
    m_steps = _positive_integer(m_steps, "m_steps")
    max_its = _positive_integer(max_its, "max_its")
    hidden = _positive_integer(hidden, "hidden")
    lr = _positive_finite(lr, "lr")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**32
    ):
        raise ValueError("seed must be None or an integer in [0, 2**32)")
    seed = None if seed is None else int(seed)
    if not isinstance(reweight, (bool, np.bool_)):
        raise TypeError("reweight must be boolean")
    reweight = bool(reweight)
    if isinstance(coverage_levels, (str, bytes)):
        raise TypeError("coverage_levels must be a sequence of probabilities")
    try:
        coverage_levels = tuple(float(level) for level in coverage_levels)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage_levels must contain finite probabilities in (0, 1)") from exc
    if (
        not coverage_levels
        or len(set(coverage_levels)) != len(coverage_levels)
        or any(not np.isfinite(level) or not 0.0 < level < 1.0 for level in coverage_levels)
    ):
        raise ValueError("coverage_levels must contain unique finite probabilities in (0, 1)")
    if rounds > 1 and y_obs is None:
        raise ValueError(
            "learn_inverse(rounds > 1) requires y_obs: rounds 2..R perform SNPE-style refinement "
            "toward a SPECIFIC observation (draw theta ~ q(theta | y_obs) from the current round's "
            "student, re-run the simulator, retrain warm-started) -- with no y_obs there is nothing "
            "to refine toward, so rounds > 1 is meaningless. Round 1 alone (unconditional pair "
            "generation) is valid without y_obs; pass y_obs=... for rounds > 1."
        )
    if reweight and not callable(true_log_likelihood):
        raise ValueError("reweight=True requires true_log_likelihood(theta, y_obs) -> float (a LOG likelihood).")
    if reweight and y_obs is None:
        raise ValueError("reweight=True requires y_obs because importance weights are observation-specific.")
    if (rounds > 1 or reweight) and not callable(getattr(prior, "log_density", None)):
        raise TypeError("sequential or target-corrected inverse fitting requires prior.log_density(theta)")

    rng = np.random.RandomState(seed)
    if y_obs is None:
        y_obs_arr = None
    else:
        try:
            y_obs_arr = np.asarray(y_obs, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_obs must be a finite observation vector") from exc
        if y_obs_arr.size == 0 or not np.all(np.isfinite(y_obs_arr)):
            raise ValueError("y_obs must be a finite observation vector")

    # round 1: unconditional pair generation
    thetas, ys = _generate_pairs(prior, simulator, n_sims, _next_seed(rng))
    theta_dim = thetas.shape[1]
    y_dim = ys.shape[1]
    if y_obs_arr is not None and y_obs_arr.shape != (y_dim,):
        raise ValueError(f"y_obs must have simulator output width {y_dim}, got {y_obs_arr.size}")

    if family == "flow" and theta_dim < 2:
        raise ValueError(
            f"family='flow' requires theta (the quantity being inferred, the student's y-argument) "
            f">= 2-dimensional -- build_conditional_flow needs y_dim >= 2 for its coupling layers to "
            f"be non-trivial (see its docstring). Got theta_dim={theta_dim}; use family='mdn' instead."
        )

    module = _build_student(family, x_dim=y_dim, y_dim=theta_dim, hidden=hidden, seed=_next_seed(rng))
    module = _fit_round(module, ys, thetas, m_steps=m_steps, lr=lr, max_its=max_its)
    retained_thetas = [thetas]
    retained_ys = [ys]
    retained_weights = [np.ones(len(thetas), dtype=float)]
    round_training: list[dict[str, Any]] = [
        {
            "round": 1,
            "proposal": "declared_prior",
            "target": "declared_prior_joint",
            "correction": "none_required",
            "rows": len(thetas),
            "retained_rows": len(thetas),
        }
    ]

    sharpness_by_round: list[float] = []
    if y_obs_arr is not None:
        sharpness_by_round.append(_posterior_sharpness(module, y_obs_arr, theta_dim, n=500, seed=_next_seed(rng)))

    for _r in range(2, int(rounds) + 1):
        theta_round = _as_rows(
            _sample_given(module, y_obs_arr, n_sims, seed=_next_seed(rng)),
            n_sims,
            context=f"sequential proposal round {_r}",
            expected_width=theta_dim,
        )
        log_proposal = _log_density_given(module, y_obs_arr, theta_round)
        log_prior = np.asarray(
            [_scalar_log_value(prior.log_density(theta), context="prior.log_density") for theta in theta_round],
            dtype=float,
        )
        block_weights, proposal_ess = _normalized_importance_weights(
            log_prior - log_proposal,
            context=f"inverse sequential round {_r}",
        )
        # Equal expected mass per round: round one has n_sims unit weights;
        # each corrected proposal block is self-normalized then scaled to the
        # same total. This is a multiple-proposal importance objective for the
        # original prior joint, never an uncorrected proposal objective.
        block_weights *= len(theta_round)
        y_round = _simulate_rows(simulator, theta_round, expected_width=y_dim)
        retained_thetas.append(theta_round)
        retained_ys.append(y_round)
        retained_weights.append(block_weights)
        all_thetas = np.concatenate(retained_thetas, axis=0)
        all_ys = np.concatenate(retained_ys, axis=0)
        all_weights = np.concatenate(retained_weights)
        module = _fit_round(
            module,
            all_ys,
            all_thetas,
            m_steps=m_steps,
            lr=lr,
            max_its=max_its,
            weights=all_weights,
        )
        round_training.append(
            {
                "round": _r,
                "proposal": "frozen_previous_q(theta|y_obs)",
                "target": "declared_prior_joint",
                "correction": "self_normalized_p(theta)/q_round(theta|y_obs)",
                "rows": len(theta_round),
                "retained_rows": len(all_thetas),
                "proposal_ess": proposal_ess,
                "proposal_ess_ratio": proposal_ess / len(theta_round),
            }
        )
        sharpness_by_round.append(_posterior_sharpness(module, y_obs_arr, theta_dim, n=500, seed=_next_seed(rng)))

    (
        sbc_stat,
        sbc_pvalue,
        sbc_bins,
        sbc_pvalues_by_dimension,
        coverage_emp,
        coverage_by_dimension,
        coverage_intervals,
        coverage_pass,
    ) = _calibration_receipts(
        module,
        prior,
        simulator,
        theta_dim=theta_dim,
        y_dim=y_dim,
        n_replications=n_sbc_replications,
        n_posterior_samples=n_posterior_samples,
        coverage_levels=coverage_levels,
        seed=_next_seed(rng),
    )
    prior_pred = _prior_predictive_receipt(ys, y_obs_arr)

    ess = ess_ratio = None
    reweighted_particles = reweighted_weights = None
    reweight_warnings: list[str] = []
    if reweight:
        assert true_log_likelihood is not None  # narrowed by the guard above
        reweighted_particles, reweighted_weights, ess, ess_ratio, reweight_warnings = _reweight_receipt(
            module, prior, true_log_likelihood, y_obs_arr, n=n_reweight_samples, seed=_next_seed(rng)
        )

    receipts = InverseReceipts(
        sbc_statistic=sbc_stat,
        sbc_pvalue=sbc_pvalue,
        sbc_bins=sbc_bins,
        sbc_replications=int(n_sbc_replications),
        sbc_pass=bool(sbc_pvalue > 0.01),
        coverage=coverage_emp,
        coverage_pass=coverage_pass,
        prior_predictive=prior_pred,
        rounds_trained=int(rounds),
        sbc_pvalues_by_dimension=sbc_pvalues_by_dimension,
        coverage_by_dimension=coverage_by_dimension,
        coverage_intervals=coverage_intervals,
        sharpness_by_round=sharpness_by_round,
        round_training=round_training,
        ess=ess,
        ess_ratio=ess_ratio,
        warnings=reweight_warnings,
    )

    return InverseModel(
        module=module,
        prior=prior,
        simulator=simulator,
        family=family,
        theta_dim=theta_dim,
        y_dim=y_dim,
        receipts=receipts,
        seed=seed,
        reweighted_y=None if not reweight else y_obs_arr.copy(),
        reweighted_particles=reweighted_particles,
        reweighted_weights=reweighted_weights,
    )
