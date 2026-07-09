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
3. **Sequential refinement (rounds 2..R).** Resolved decision (was open in the design note): eager,
   via an optional ``y_obs`` keyword to THIS function. When ``y_obs`` is given, each subsequent
   round draws ``theta ~ q(theta | y_obs)`` from the CURRENT round's student, re-runs the simulator,
   and retrains warm-started from the previous round's module weights (same object, so ``optimize``
   continues training it in place -- the same warm-start pattern ``GradLeaf``'s own M-step uses
   across EM iterations). ``rounds > 1`` without ``y_obs`` has no observation to sharpen toward, so
   it raises ``ValueError`` rather than silently doing nothing -- round 1 alone (unconditional pair
   generation) is valid without ``y_obs``.
4. **Optional exactness stage.** ``reweight=True`` with ``true_log_likelihood(theta, y_obs) ->
   float`` (a LOG likelihood) treats the final round's ``q(theta | y_obs)`` as a self-normalized-
   importance-sampling proposal: ``log w_j = log p(theta_j) + true_log_likelihood(theta_j, y_obs) -
   log q(theta_j | y_obs)``, normalized by log-sum-exp (the same construction
   ``mixle.inference.condition``'s SIR fallback uses), with ``ESS = 1 / sum(w_norm^2)`` reported so a
   low ESS visibly says "don't trust this reweighted posterior" instead of silently returning a
   degenerate one.
5. **Calibration receipts (always computed).**
   - **SBC.** Resolved decision (was open): a chi-square uniformity test on binned ranks (Talts et
     al.), ``bins = min(20, n_sbc_replications // 5)`` (clamped to >= 2), one chi-square statistic
     per ``theta`` dimension SUMMED (valid under the standard independence-across-dimensions
     simplification -- a sum of independent chi-square variables is chi-square with summed degrees
     of freedom), against threshold ``p-value > 0.01`` (no rejection of uniformity).
   - **Coverage.** Per-dimension, per-replication credible interval containment vs nominal level,
     averaged; pass when within +/-5% of nominal.
   - **Prior-predictive.** Empirical mean/std of the round-1 simulated ``y_i``'s, plus (when
     ``y_obs`` is given) its per-dimension z-score against that empirical distribution -- a
     caller-facing warning, not a gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2

from mixle.inference.condition import ConditionReceipt, Posterior
from mixle.inference.estimation import optimize
from mixle.models.grad_leaf import GradLeaf

__all__ = ["InverseModel", "InverseReceipts", "learn_inverse"]


def _torch() -> Any:
    import torch

    return torch


def _next_seed(rng: np.random.RandomState) -> int:
    return int(rng.randint(0, 2**31 - 1))


def _as_rows(arr: Any, n: int) -> np.ndarray:
    """Normalize a sampler's ``sample(n)`` output to shape ``(n, d)``. A univariate sampler
    (e.g. ``GaussianDistribution``) returns a flat ``(n,)`` array of scalar draws -- ``atleast_2d``
    on that would misread it as ONE row of ``n`` dimensions instead of ``n`` rows of one dimension,
    so a 1-D result of length ``n`` is reshaped to ``(n, 1)`` explicitly rather than via ``atleast_2d``."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a.reshape(n, 1)
    return a


def _build_student(family: str, *, x_dim: int, y_dim: int, hidden: int, seed: int | None) -> Any:
    from mixle.models.mixture_density import build_conditional_flow, build_mdn

    # nn.Module weight init draws from torch's GLOBAL rng, not our own seeded RandomState -- seed it
    # explicitly here so a given `seed=` to learn_inverse determines the student's starting weights
    # too (determinism is a contract, rule 0.2.4), independent of whatever global torch state a
    # caller/earlier test left behind.
    torch = _torch()
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
    thetas = _as_rows(prior.sampler(seed=seed).sample(int(n)), int(n))
    ys = np.asarray([np.atleast_1d(np.asarray(simulator(theta), dtype=float)) for theta in thetas], dtype=float)
    return thetas, ys


def _sample_given(module: Any, x_row: np.ndarray, n: int, *, seed: int | None) -> np.ndarray:
    """``n`` draws of ``theta ~ q(theta | x_row)`` from a fitted student ``module``."""
    torch = _torch()
    module.eval()
    rng = np.random.RandomState(seed)
    torch.manual_seed(_next_seed(rng))
    xt = torch.as_tensor(np.tile(np.atleast_1d(np.asarray(x_row, dtype=float)), (int(n), 1)), dtype=torch.float32)
    with torch.no_grad():
        return module.sample_given(xt).cpu().numpy()


def _log_density_given(module: Any, x_row: np.ndarray, theta_batch: np.ndarray) -> np.ndarray:
    """``log q(theta | x_row)`` for every row of ``theta_batch`` (shape ``(n, theta_dim)``)."""
    torch = _torch()
    module.eval()
    theta_batch = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    n = theta_batch.shape[0]
    xt = torch.as_tensor(np.tile(np.atleast_1d(np.asarray(x_row, dtype=float)), (n, 1)), dtype=torch.float32)
    yt = torch.as_tensor(theta_batch, dtype=torch.float32)
    with torch.no_grad():
        return module.log_density(xt, yt).cpu().numpy().reshape(-1)


def _fit_round(module: Any, ys: np.ndarray, thetas: np.ndarray, *, m_steps: int, lr: float, max_its: int) -> Any:
    """One ``optimize()`` call against ``(y, theta)`` pairs, warm-started from ``module``'s own weights
    (same underlying ``nn.Module`` object -- ``GradEstimator.estimate`` mutates it in place)."""
    data = list(zip(ys.tolist(), thetas.tolist()))
    leaf = GradLeaf(module, m_steps=m_steps, lr=lr)
    fitted = optimize(data, leaf, max_its=max_its, out=None)
    return fitted.module


def _posterior_sharpness(module: Any, y_obs: np.ndarray, theta_dim: int, *, n: int, seed: int | None) -> float:
    """A scalar "how spread out is q(theta | y_obs)" receipt -- sum of per-dimension sample variance.
    Lower is sharper; used to assert refinement rounds measurably sharpen the posterior (test (e))."""
    samples = _sample_given(module, y_obs, n, seed=seed)
    return float(np.sum(np.var(samples, axis=0)))


def _calibration_receipts(
    module: Any,
    prior: Any,
    simulator: Callable[[Any], Any],
    *,
    theta_dim: int,
    n_replications: int,
    n_posterior_samples: int,
    coverage_levels: tuple[float, ...],
    seed: int | None,
) -> tuple[float, float, int, dict[float, float]]:
    """SBC (chi-square on binned ranks) + per-level coverage, sharing the same replications."""
    rng = np.random.RandomState(seed)
    ranks = np.zeros((n_replications, theta_dim), dtype=int)
    covered = {c: np.zeros((n_replications, theta_dim), dtype=bool) for c in coverage_levels}
    for r in range(n_replications):
        theta_star = np.atleast_1d(np.asarray(prior.sampler(seed=_next_seed(rng)).sample(1), dtype=float)).reshape(-1)[
            :theta_dim
        ]
        y_star = np.atleast_1d(np.asarray(simulator(theta_star), dtype=float))
        samples = _sample_given(module, y_star, n_posterior_samples, seed=_next_seed(rng))
        for d in range(theta_dim):
            ranks[r, d] = int(np.sum(samples[:, d] < theta_star[d]))
            for c in coverage_levels:
                lo_q, hi_q = (1.0 - c) / 2.0, (1.0 + c) / 2.0
                lo, hi = np.quantile(samples[:, d], [lo_q, hi_q])
                covered[c][r, d] = bool(lo <= theta_star[d] <= hi)

    # SBC: bins = min(20, n_sbc_replications // 5), clamped to >= 2 -- the resolved binning choice
    # (design note "Resolved" section / mixle.task.inverse docstring).
    bins = max(2, min(20, n_replications // 5))
    edges = np.linspace(0.0, float(n_posterior_samples), bins + 1)
    stat_total, df_total = 0.0, 0
    for d in range(theta_dim):
        hist, _ = np.histogram(ranks[:, d], bins=edges)
        expected = n_replications / bins
        stat_total += float(np.sum((hist - expected) ** 2 / expected))
        df_total += bins - 1
    pvalue = float(chi2.sf(stat_total, df_total)) if df_total > 0 else 1.0

    coverage_emp = {c: float(np.mean(covered[c])) for c in coverage_levels}
    return stat_total, pvalue, bins, coverage_emp


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
) -> tuple[float, float, list[str]]:
    """Self-normalized importance reweighting of ``q(theta | y_obs)`` against the true likelihood
    (same log-sum-exp construction as ``mixle.inference.condition``'s SIR fallback)."""
    thetas = _sample_given(module, y_obs, n, seed=seed)
    log_q = _log_density_given(module, y_obs, thetas)
    log_prior = np.array([float(prior.log_density(theta)) for theta in thetas])
    log_lik = np.array([float(true_log_likelihood(theta, y_obs)) for theta in thetas])
    log_w = log_prior + log_lik - log_q

    warnings: list[str] = []
    finite = np.isfinite(log_w)
    if not finite.any():
        ess = 0.0
        warnings.append("reweight: all importance weights are zero/non-finite.")
    else:
        m = log_w[finite].max()
        w = np.where(finite, np.exp(log_w - m), 0.0)
        sw = w.sum()
        w_norm = w / sw
        ess = float(1.0 / np.sum(w_norm**2))
    ess_ratio = ess / n
    if ess_ratio < 0.01:
        warnings.append(f"reweight ESS ratio {ess_ratio:.4f} < 0.01 -- reweighted posterior is not trustworthy.")
    return ess, ess_ratio, warnings


@dataclass
class InverseReceipts:
    """The calibration report that ships with every :class:`InverseModel` -- tells the caller
    whether to trust ``q(theta | y)``, not just a point estimate."""

    sbc_statistic: float
    sbc_pvalue: float
    sbc_bins: int
    sbc_replications: int
    sbc_pass: bool  # p-value > 0.01 (resolved acceptance threshold -- see module docstring)
    coverage: dict[float, float]  # nominal level -> empirical coverage
    coverage_pass: dict[float, bool]  # within +/-5% of nominal
    prior_predictive: dict[str, Any]
    rounds_trained: int
    sharpness_by_round: list[float] = field(default_factory=list)
    ess: float | None = None
    ess_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)


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
    ) -> None:
        self.module = module
        self.prior = prior
        self.simulator = simulator
        self.family = family
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.receipts = receipts
        self._seed = seed

    def posterior(self, y: Any) -> Posterior:
        """Wrap ``q(theta | y)`` as an M0 :class:`~mixle.inference.condition.Posterior`: ``sample(n)``
        / ``log_density(theta)`` / ``mean(field)`` / ``.receipt`` -- so downstream condition/do
        composition treats a learned inverse like an exactly-conditioned one, modulo the amortization
        warning on ``.receipt`` and the ``InverseReceipts`` pointer at ``.receipt.inverse_receipts``."""
        y_row = np.atleast_1d(np.asarray(y, dtype=float))
        module = self.module
        base_seed = self._seed

        def sample_fn(n: int, s: int | None) -> np.ndarray:
            return _sample_given(module, y_row, n, seed=s if s is not None else base_seed)

        def log_density_fn(theta: Any) -> float:
            theta_row = np.atleast_2d(np.asarray(theta, dtype=float))
            return float(_log_density_given(module, y_row, theta_row)[0])

        def mean_fn(path: tuple[int, ...]) -> float:
            idx = int(path[0]) if len(path) else 0
            samples = _sample_given(module, y_row, 500, seed=base_seed)
            return float(np.mean(samples[:, idx]))

        receipt = ConditionReceipt(
            method="amortized",
            warnings=[
                "InverseModel.posterior: a LEARNED amortized approximation "
                "(mixle.task.inverse.learn_inverse), not exact conditioning -- see "
                ".receipt.inverse_receipts (SBC/coverage/prior-predictive/ESS) before trusting it."
            ],
        )
        receipt.inverse_receipts = self.receipts  # pointer to the full calibration report (not an M0 field)
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

    ``rounds > 1`` (SNPE-style sequential refinement toward a SPECIFIC observation) requires
    ``y_obs``: round 1 alone (unconditional pair generation) is the only round that has meaning
    without an observation to sharpen against.
    """
    if family not in ("flow", "mdn"):
        raise ValueError(f"family must be 'flow' or 'mdn', got {family!r}")
    if int(rounds) < 1:
        raise ValueError("rounds must be >= 1")
    if int(rounds) > 1 and y_obs is None:
        raise ValueError(
            "learn_inverse(rounds > 1) requires y_obs: rounds 2..R perform SNPE-style refinement "
            "toward a SPECIFIC observation (draw theta ~ q(theta | y_obs) from the current round's "
            "student, re-run the simulator, retrain warm-started) -- with no y_obs there is nothing "
            "to refine toward, so rounds > 1 is meaningless. Round 1 alone (unconditional pair "
            "generation) is valid without y_obs; pass y_obs=... for rounds > 1."
        )
    if reweight and true_log_likelihood is None:
        raise ValueError("reweight=True requires true_log_likelihood(theta, y_obs) -> float (a LOG likelihood).")

    rng = np.random.RandomState(seed)
    y_obs_arr = None if y_obs is None else np.atleast_1d(np.asarray(y_obs, dtype=float))

    # round 1: unconditional pair generation
    thetas, ys = _generate_pairs(prior, simulator, n_sims, _next_seed(rng))
    theta_dim = thetas.shape[1]
    y_dim = ys.shape[1]

    if family == "flow" and theta_dim < 2:
        raise ValueError(
            f"family='flow' requires theta (the quantity being inferred, the student's y-argument) "
            f">= 2-dimensional -- build_conditional_flow needs y_dim >= 2 for its coupling layers to "
            f"be non-trivial (see its docstring). Got theta_dim={theta_dim}; use family='mdn' instead."
        )

    module = _build_student(family, x_dim=y_dim, y_dim=theta_dim, hidden=hidden, seed=_next_seed(rng))
    module = _fit_round(module, ys, thetas, m_steps=m_steps, lr=lr, max_its=max_its)

    sharpness_by_round: list[float] = []
    if y_obs_arr is not None:
        sharpness_by_round.append(_posterior_sharpness(module, y_obs_arr, theta_dim, n=500, seed=_next_seed(rng)))

    for _r in range(2, int(rounds) + 1):
        theta_round = _sample_given(module, y_obs_arr, n_sims, seed=_next_seed(rng))
        y_round = np.asarray([np.atleast_1d(np.asarray(simulator(t), dtype=float)) for t in theta_round], dtype=float)
        module = _fit_round(module, y_round, theta_round, m_steps=m_steps, lr=lr, max_its=max_its)
        sharpness_by_round.append(_posterior_sharpness(module, y_obs_arr, theta_dim, n=500, seed=_next_seed(rng)))

    sbc_stat, sbc_pvalue, sbc_bins, coverage_emp = _calibration_receipts(
        module,
        prior,
        simulator,
        theta_dim=theta_dim,
        n_replications=n_sbc_replications,
        n_posterior_samples=n_posterior_samples,
        coverage_levels=coverage_levels,
        seed=_next_seed(rng),
    )
    coverage_pass = {c: bool(abs(coverage_emp[c] - c) <= 0.05) for c in coverage_levels}
    prior_pred = _prior_predictive_receipt(ys, y_obs_arr)

    ess = ess_ratio = None
    reweight_warnings: list[str] = []
    if reweight:
        assert true_log_likelihood is not None  # narrowed by the guard above
        ess, ess_ratio, reweight_warnings = _reweight_receipt(
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
        sharpness_by_round=sharpness_by_round,
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
    )
