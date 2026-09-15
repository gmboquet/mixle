# mixle 0.8.2 candidate -- ten-pass adversarial review campaign (2026-09-11)

## Candidate

- Branch `release/0.8.2`, commit `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013`
  (library-identical to `f3111688`; `866078be` only re-cut receipts). Reviewed from a `git archive` export
  (`source/`), never from the worktree, because two other sessions were committing to it during the campaign.
- Wheel `dist/mixle-0.8.2-py3-none-any.whl` sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da`
  (sdist `71ec826b…`), built with `python -m build --no-isolation` under the pinned
  `release-checklists/0.8.2-build-requirements.txt` closure, `SOURCE_DATE_EPOCH=1789095173`, commit and tree
  stamped in `mixle/_build_provenance.json`. Its `source_content_sha256` (`2b52a709…`) equals that of the wheel
  the retained corpus-execution receipt was measured on.
- Environments: `venv-full` (numba, torch, pandas, pyspark, scikit-learn, mixle-pde/sim/physics), `venv-nonumba`,
  `venv-base` (numpy + scipy only); the published 0.8.1 (`candidate-081/venv`) for before/after. Every reviewer
  proved the loaded tree with `mixle.__path__[0]` and the provenance commit from `/tmp`.
- Corpus: `mixle-notebooks` @ `d99d296` (131 notebooks), `examples/` (57 scripts); every notebook and example
  assigned to at least one pass (`ASSIGNMENTS.md`).

## Execution fact (`exec/EXECUTION.md`)

57 of 57 examples exit 0; 131 of 131 notebooks execute to completion on the wheel (including the Spark tutorial
with a JDK on PATH and the 74-minute malware-embedding notebook). Pass 07 separately executed all 44 application,
geoscience and architecture notebooks (44/44 exit 0) and found no stored number changed by the candidate.

## How the reports were produced (provenance, stated plainly)

Ten fresh reviewers with no shared context each worked 40-75 minutes (executed their notebooks and examples,
wrote 250+ probe scripts with captured outputs) and were then all killed by an API session rate limit before
writing their reports; the session was continued under a new session directory, after which their contexts
could not be resumed. Their work directories survived intact. For each pass a fresh "recovery" reviewer was
given the preserved evidence plus `RECOVERED_NOTES.md` (the original reviewer's command trail and short
reasoning notes, extracted from its transcript by script) and wrote `report.md`/`findings.json` from that
evidence, re-running only what a finding needed and finishing cheap pending items. Four recovery reviewers
were themselves cut by a second rate limit and resumed within the session. Every reproduction in the reports
was run on the candidate by the reviewer that wrote it; the campaign lead independently reproduced every
blocking finding (`VERIFICATION.md`).

## Findings by pass (`SUMMARY.md`, `findings-all.json`)

| pass | area | blocking | real | minor | docs | total |
|---|---|---|---|---|---|---|
| 01 | univariate laws, enumeration | 2 | 3 | 4 | 0 | 9 |
| 02 | latent models | 2 | 7 | 5 | 0 | 14 |
| 03 | inference loop, receipts, engines | 0 | 4 | 17 | 3 | 24 |
| 04 | PPL | 2 | 3 | 9 | 1 | 15 |
| 05 | production / MLOps, tasks, DOE | 1 | 7 | 6 | 3 | 17 |
| 06 | data adapters, structured families | 0 | 10 | 9 | 1 | 20 |
| 07 | application / geoscience / architecture notebooks | 0 | 7 | 10 | 3 | 20 |
| 08 | tutorials, README, migration guide | 1 | 8 | 4 | 5 | 18 |
| 09 | examples 1-28 | 1 | 4 | 9 | 2 | 16 |
| 10 | examples 29-57 | 0 | 3 | 6 | 3 | 12 |
| all | | 9 | 56 | 79 | 21 | 165 |

## The blocking findings: seven distinct defects, all independently confirmed

| # | defect | ids | contradicts | verified |
|---|---|---|---|---|
| B1 | `quantile(q)` still answers an out-of-domain `q` on many families (HalfNormal returns a negative number, BetaBinomial a support point, Uniform/Laplace/StudentT NaN) | Q01-F01, Q08-F08 | R05-F07, migration guide | yes |
| B2 | A support-limited component still cannot be mixed unless it is one of the ten continuous families that implement `supported_rows`: count families (Poisson/Geometric/LogSeries) and Pareto, GeneralizedPareto, Nakagami, Rician, Tweedie, NegativeBinomial, BetaBinomial, Bernoulli refuse at encode or initialization, blaming the data | Q01-F02, Q02-F02 (+ Q06-F08, Q06-F09 real) | P02-F03 | yes |
| B3 | A terminal-state HMM's `latent_posterior` (marginals/mode/sample), `viterbi`, `seq_viterbi` and `posterior_predictive` enforce only "no terminal state before the end" and never require the final state to be terminal: last-row marginal off by 1.0, Viterbi returns zero-probability paths, 200/200 FFBS draws end in a non-terminal state; `seq_posterior` and `log_density` are correct | Q02-F01 | R05-F01 | yes |
| B4 | A conjugate posterior restored from a pickle answers `predict()` with the plug-in predictive (sd 2.028 vs the integrated 2.306) instead of refusing; the refusal is defined on `Posterior` only, and the regression test covers mcmc/laplace only; the conjugate route is the default for every conjugate model | Q04-F01 | R02-F04, P04-F05 | yes |
| B5 | `how='ensemble'` at its default budget returns a wrong posterior on ordinary 5- and 18-parameter models (means up to 3.7-7.8 posterior sds off, sds 3-9x too wide) silently in single-chain mode; mcmc/nuts/hmc are right on the same models; pre-existing in 0.8.1 | Q04-F11 (+ Q07-F06 real) | (unledgered) | yes |
| B6 | `fit_with_provenance`, `Monitor.update` and the scoring verbs bypass the 0.8.2 input front door: a DataFrame or a mapping of columns is fitted/scored as its column names, a bare `str` as its characters, and the inputs `optimize()` now refuses by name still crash the old way | Q05-F01 (+ Q06-F01..F05, Q02-F08, Q03-F04 real on `vdata=`, `best_of`, `learn_bayesian_network`, `learn_structure`, `detect_drift`, `Service.score`) | R06-F03, R06-F06 | yes |
| B7 | 0.8.2 regression: a `TreeHiddenMarkovModelDistribution` fitted by `optimize()` can no longer be JSON-serialized (`_p_level_cache` is left populated and the schema check refuses it); the same fit round-trips on 0.8.1 | Q09-F01 (+ Q02-F09 real) | (new) | yes |

## Themes among the 56 real findings (cross-pass convergence)

- **The front door is not one door** (B6 and eight real findings across passes 02, 03, 05, 06): `vdata=`,
  `best_of`, `learn_bayesian_network`, `learn_structure`, `fit_with_provenance`, `Monitor.update`,
  `Service.score`, `detect_drift`, `Model.fit/evaluate` with a `RecordEstimator`.
- **P02-F03 is partial** (B2; Q06-F08: a zero-weight out-of-support row still cannot be fit through `optimize()`;
  Q01-F03: the R05-F05 zero-weight exemption holds on `seq_update` but not on the scalar `update`/`initialize`).
- **Write-only JSON families remain** (Q02-F09, Q05-F05, Q06-F10, Q07-F03, Q07-F09, Q09-F02): KnowledgeGraph,
  StudentTCopula, GaussianProcessRegressor (unpicklable too), seven fitted directional/gallery families, and
  `Registry.register` accepting eleven families it can never read back.
- **The A-02 under-supported-component note fires from intermediate iterates / the initialization subsample**
  and describes a starvation the returned model does not have (Q02-F03, Q07-F01, Q08-F03/F04, Q10 minors);
  its advice names `restarts=`, which `optimize()`/`MixtureEstimator()`/`Mix().fit()` do not accept.
- **EM convergence bookkeeping**: a fit that settles on its last permitted iteration is stamped `converged=False`
  with a "last objective gain 0" cap note (Q03-F01, Q08-F01); the monotone acceptance gate's absolute 1e-12
  tolerance rejects ulp-level wobbles and float32 engines every run (Q03-F02, Q08-F06).
- **`initialize()` and `estimate()` verbs lag the repairs** (Q02-F05 no A-01 start; Q01-F03 zero-weight rows;
  Q07-F20 `estimate([])` returns a default model where `optimize([])` refuses).
- **Sampler defaults**: ensemble (B5), the hierarchical grouped route returns an unconverged empirical-Bayes fit
  with `max_its`/`delta` silently swallowed (Q04-F14), `how='vi'` on one observation wanders to 1e12 (Q09-F05).
- **Lifecycle/checkpoint seams**: `on_step` never called on the automatic-structure route (Q05-F03), the model
  handed to `on_step` is one EM step behind the returned one (Q05-F04), `fit_with_provenance(print_iter=N)`
  `KeyError` (Q05-F02), drift verbs crash on dict-record models (Q05-F06).

## Coverage gaps the reports declare

Seed sweeps for several example-level repairs (P10-F02/F03/F04/F06/F07, P09 geoscience seeds) were written but
not run because the host carried a load average of 60-90 from other sessions; `venv-nonumba` was verified but
little used by passes 06, 09 and 10; serialization round trips of LDA/IBP/Gaussian-admixture and the
`hidden_association` attacks were not reached; the docstring runtime claims cannot be judged under this load.

## What this means for the release gate

The checklist row "Ten AI adversarial reviews of the notebooks and examples" closes only when every finding is
repaired on the candidate with a regression test and an independent re-run, or dispositioned in
`0.8.2-decisions.md`. Seven distinct blocking defects are open, five of them repair claims in the 0.8.2 CHANGELOG
that do not hold on a documented route (B1-B4, B6) and one a 0.8.2 regression (B7). Any repair moves the tree,
which re-opens all 69 receipts (index-based evidence digest; re-cut order in the release notes). D-0216's
statement that the 0.8.2 review is "one targeted adversarial pass, not a fresh ten-pass round" is superseded by
this campaign and needs amending when the row is re-closed; the row's current receipt cites the 0.8.1 passes.

Retained copies with machine paths replaced by placeholders (`<review-root>`, `<review-store>`, `<worktree>`,
`<notebooks-repo>`) are staged in `retained/pass-NN/` for `release-checklists/0.8.2-reviews/`.
