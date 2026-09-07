"""Model comparison: rank fitted models by predictive accuracy, not by eye or in-sample fit.

mixle.ppl's Bayesian model-comparison surface -- ``waic``, ``loo``, and ``compare()`` -- scores
fitted models by their estimated *out-of-sample* predictive accuracy (the expected log pointwise
predictive density, elpd), the same quantity held-out validation would estimate, but computed
from a single fit. That is a different question from "which model has the higher in-sample
log-likelihood", which can favor a flexible but predictively wrong model.

To make the comparison have a real, unambiguous answer to recover, the data here is genuinely
bimodal (two well-separated Gaussian modes), and the candidates are deliberately mismatched:

  A. Normal(free, free)            -- WRONG: unimodal, forced to average across both modes
  B. Mix([Normal(free,free)] * 2)  -- RIGHT: a 2-component mixture that can separate the modes
  C. StudentT(free, free, free)    -- a second WRONG answer: heavier-tailed, but still unimodal

All three are fit with ``how='mcmc'`` so each carries real posterior draws rather than one point
estimate -- that is what makes ``waic``/``loo`` (which integrate over parameter uncertainty)
meaningful instead of silently falling back to a single-draw row. ``compare()`` prints the current
ranking and the raw ``waic()``/``loo()`` diagnostics, including PSIS-LOO's khat_max reliability
check; the source does not predeclare a winner.

Run: ``python examples/model_comparison_example.py``
"""

from __future__ import annotations

import math

import numpy as np

from mixle.ppl import Mix, Normal, StudentT, compare, free, loo, waic

N_PER_MODE = 300  # 600 observations total
MODE_SEP = 6.0  # modes at -6 and +6, sigma=1 -> unambiguously two separate clusters
# MCMC posterior draws per model. Not "modest so the demo runs quickly": at 1200/600 the
# random-walk chains on the two scale parameters come back with a bulk ESS around 3, and the
# posterior SUMMARY printed as "B's fitted component means" is then a summary of a chain that has
# not mixed -- sigma2 1.6 against a true 1.0 (P10-F06). These settings reach a worst bulk ESS in the
# hundreds and recover the planted parameters, and the run still takes a couple of seconds.
DRAWS, BURN, THIN = 6000, 2000, 4


def bimodal_data(rng: np.random.RandomState) -> np.ndarray:
    """Genuinely bimodal 1-D data: two Gaussian modes, well separated relative to their spread."""
    lo = rng.normal(-MODE_SEP, 1.0, N_PER_MODE)
    hi = rng.normal(MODE_SEP, 1.0, N_PER_MODE)
    return np.concatenate([lo, hi])


def fit_candidates(y: np.ndarray) -> tuple:
    """Fit the three candidates with how='mcmc' so every model carries posterior draws."""
    a = Normal(free, free, name="A_single_normal").fit(
        y, how="mcmc", draws=DRAWS, burn=BURN, thin=THIN, rng=np.random.RandomState(1)
    )
    b = Mix([Normal(free, free), Normal(free, free)], name="B_two_component_mixture").fit(
        y, how="mcmc", draws=DRAWS, burn=BURN, thin=THIN, rng=np.random.RandomState(2)
    )
    c = StudentT(free, free, free, name="C_student_t").fit(
        y, how="mcmc", draws=DRAWS, burn=BURN, thin=THIN, rng=np.random.RandomState(3)
    )
    return a, b, c


def print_ranking(rows: list[dict], by: str) -> None:
    cols = ["model", "loglik", "aic", "bic", "elpd", "se"] + (["khat_max"] if by == "loo" else []) + ["d_elpd"]
    widths = {"model": 26, "loglik": 11, "aic": 10, "bic": 10, "elpd": 11, "se": 8, "khat_max": 9, "d_elpd": 10}
    print("".join(c.rjust(widths[c]) if c != "model" else c.ljust(widths[c]) for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(v.ljust(widths[c]) if c == "model" else f"{v:.2f}".rjust(widths[c]))
        print("".join(cells))


def print_chain_health(fit, label: str) -> None:
    """Print the chain diagnostics behind a posterior-summary number.

    ``.dist`` on an MCMC fit is a posterior SUMMARY, and a summary of a chain that has not mixed is
    not a fitted parameter: on some seeds this model's scale parameters come back with a bulk ESS of
    about 3 out of 1200 draws and a sigma2 60% above the truth, with nothing printed to say so
    (P10-F06). ESS and split-R-hat are already in ``summary()``; this puts them next to the numbers
    they qualify.
    """
    summary = fit.summary()
    parameters = {name: row for name, row in summary.items() if isinstance(row, dict) and "ess_bulk" in row}
    if not parameters:
        return
    worst = min(parameters.items(), key=lambda item: item[1]["ess_bulk"])
    print(
        f"  {label}'s chain health: worst bulk ESS {worst[1]['ess_bulk']:.0f} on {worst[0]!r}, "
        f"acceptance {summary['_acceptance_rate']:.2f}"
    )
    for name, row in sorted(parameters.items()):
        rhat = row.get("split_r_hat")
        rhat_text = "n/a (single chain)" if rhat is None or math.isnan(rhat) else f"{rhat:.3f}"
        print(
            f"    {name:<12} mean {row['mean']:>8.3f}  ess_bulk {row['ess_bulk']:>7.1f}  "
            f"ess_tail {row['ess_tail']:>7.1f}  split_r_hat {rhat_text}"
        )
    if worst[1]["ess_bulk"] < 100.0:
        print(
            "    -> fewer than 100 effective draws: the posterior summary printed above is noisy, and "
            "the 'loglik' column is that summary's likelihood, not a maximized one."
        )


def main():
    print("# mixle.ppl model comparison -- waic, loo, compare()\n")

    rng = np.random.RandomState(0)
    y = bimodal_data(rng)
    print(f"data: {len(y)} points, bimodal (modes at {-MODE_SEP:.0f} and {MODE_SEP:.0f}, sigma=1)\n")

    a, b, c = fit_candidates(y)

    b_means = sorted(comp.mu for comp in b.dist.components)
    print(f"B's fitted component means (true {-MODE_SEP:.0f}, {MODE_SEP:.0f}): [{b_means[0]:.2f}, {b_means[1]:.2f}]")
    print(f"A's fitted (mean, sd) forced to average both modes: ({a.dist.mu:.2f}, {np.sqrt(a.dist.sigma2):.2f})")
    print_chain_health(b, "B")
    print()

    print("## compare(by='loo')  -- Pareto-smoothed importance-sampling leave-one-out")
    rows_loo = compare([a, b, c], y, by="loo")
    print_ranking(rows_loo, "loo")
    gap = -rows_loo[1]["d_elpd"]  # elpd deficit of the runner-up vs the winner (d_elpd is <= 0 by construction)
    print(
        f"\n-> winner: {rows_loo[0]['model']}  (runner-up trails by {gap:.1f} elpd, ~{gap / rows_loo[0]['se']:.0f}x its se -- not a close call)\n"
    )

    print("## compare(by='waic') -- Widely Applicable Information Criterion")
    rows_waic = compare([a, b, c], y, by="waic")
    print_ranking(rows_waic, "waic")
    print()

    print("## raw waic()/loo() on B's pointwise log-likelihood directly")
    ll_b = b.pointwise_log_likelihood(y)  # (n_draws, n_obs) matrix underlying both diagnostics above
    print(f"pointwise_log_likelihood(y).shape = {ll_b.shape}  (n_draws, n_obs)\n")

    waic_b = waic(ll_b)
    loo_b = loo(ll_b)
    print("waic(ll_b) =", {k: round(v, 3) for k, v in waic_b.items() if k != "pointwise"})
    print("loo(ll_b)  =", {k: round(v, 3) for k, v in loo_b.items() if k != "pointwise"})

    khat = loo_b["khat_max"]
    verdict = "reliable" if khat <= 0.7 else "UNRELIABLE -- refit with more draws or prefer waic"
    print(f"\nkhat_max = {khat:.3f}  -> {verdict} (PSIS-LOO flags khat_max > 0.7 as an unreliable estimate)")


if __name__ == "__main__":
    main()
