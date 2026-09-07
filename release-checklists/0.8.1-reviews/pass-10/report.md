# Adversarial review — PASS 10 of 10 (mixle 0.8.1 release candidate)

- Focus: example scripts 29–57 in sorted order (`hierarchical_mixture_example.py` … `win_demo_example.py`), 29 scripts.
- Wheel: `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, built from tree `6040ea38` (branch `release/0.8.1`).
- Version verification (run from this work dir, outside any checkout):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` →
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Interpreter: candidate venv Python 3.12; optional deps present: torch 2.14.0, sklearn 1.9.0, sympy 1.14.0, transformers 5.16.1, pyspark, dask, mpi4py. Absent: `peft`, `pulp`, `openai`, `anthropic`. No API keys in the environment; nothing in this range needs one.
- Corpus copies are byte-identical to the source-tree copies (`cmp` over all 57 files).
- Method: every script executed from `runs/` with `subprocess.run(timeout=1800)`, 3 concurrent, `MPLBACKEND=Agg`, `PYTHONPATH` unset; stdout/stderr per script in `runs/`. Then per-script ground-truth checks (hand arithmetic or scratch scripts in `attacks/`), a seed-offset attack (every literal seed +100/+200/+300, 26 scripts × 3, `attacks/seeds/`), a data-size attack (10 % and a single observation, 25 scripts × 2, `attacks/sizes/`), and a `dump_models`/`load_models` round-trip over the fitted families (`attacks/json_roundtrip.py`).
- Hygiene note: importing one corpus module from a scratch script created `corpus-081/examples/__pycache__`; it was deleted and `PYTHONDONTWRITEBYTECODE=1` used afterwards. Nothing else outside the work dir was touched.

## Execution table (candidate wheel, own path, unmodified scripts)

| # | script | exit | wall (s) | stderr summary |
|---|---|---|---|---|
| 29 | hierarchical_mixture_example.py | 0 | 17.3 | 1 UserWarning: optimize() stopped at max_its cap (2000), unconverged (gain 1.4e-4) |
| 30 | joint_mixture_example.py | 0 | 4.0 | 2 unconverged-at-cap warnings (best_of restarts, cap 100) |
| 31 | label_economics_demo.py | 0 | 3.9 | clean |
| 32 | latent_variable_models_example.py | 0 | 3.5 | 1 unconverged-at-cap warning (LDA, cap 60) |
| 33 | lookback_hmm_example.py | 0 | 20.2 | clean |
| 34 | mixture_reduction_benchmark.py | 0 | 8.2 | 3 unconverged-at-cap warnings (project(), cap 50) |
| 35 | model_comparison_example.py | 0 | 4.5 | clean |
| 36 | multimodal_stage1_demo.py | 0 | 4.8 | 1 unconverged-at-cap warning (cap 6) |
| 37 | peft_lora_grad_leaf.py | **1** | 2.9 | `ModuleNotFoundError: No module named 'peft'` (raw traceback; P10-F08) |
| 38 | ppl_example.py | 0 | 3.0 | clean |
| 39 | precedence_scheduling_example.py | 0 | 3.0 | clean |
| 40 | production_example.py | 0 | 4.8 | clean |
| 41 | project_neural_to_structured.py | 0 | 18.7 | 5 unconverged-at-cap warnings |
| 42 | quickstart_example.py | 0 | 5.1 | 1 unconverged-at-cap warning emitted from inside `propose()` (cap 25, gain 7e-7) |
| 43 | reasoner_investigation_demo.py | 0 | 2.9 | 1 UserWarning: perfect separation in a binomial GLM (structure-search candidate) |
| 44 | scaling_example.py | 0 | 5.6 | 2 unconverged-at-cap warnings (max_its=1 by design) |
| 45 | semi_supervised_mixture_example.py | 0 | 4.6 | clean |
| 46 | shared_embedding_example.py | 0 | 2.9 | clean |
| 47 | skeptic_challenge_example.py | 0 | 11.2 | 28 unconverged-at-cap warnings, one with negative gain (-56.6) |
| 48 | structure_learning_example.py | 0 | 10.5 | 58 unconverged-at-cap warnings |
| 49 | structured_hmm_example.py | 0 | 8.7 | 1 unconverged-at-cap warning |
| 50 | structured_leaves_example.py | 0 | 3.3 | clean |
| 51 | symbolic_export_example.py | 0 | 3.4 | clean (Sage section skipped as documented) |
| 52 | task_cascade_economics_example.py | 0 | 4.6 | 41 unconverged-at-cap warnings from `mixle/task/distill.py` and `density.py` |
| 53 | task_distill_example.py | 0 | 8.8 | 144 unconverged-at-cap warnings from `mixle/task/distill.py`, 5 with negative gain |
| 54 | task_extraction_example.py | 0 | 7.1 | clean |
| 55 | task_llm_active_example.py | 0 | 6.0 | 129 unconverged-at-cap warnings from `mixle/task/distill.py` |
| 56 | vlm_trust_receipts_demo.py | 0 | 2.9 | clean |
| 57 | win_demo_example.py | 0 | 3.7 | 66 unconverged-at-cap warnings from `mixle/task/distill.py`, 1 with negative gain |

28/29 exit 0 on their own path; the one failure is the documented missing `peft` dependency, but the script does not explain itself (P10-F08). No script hangs; slowest is lookback at 20.2 s (manifest says ~20–25 s, matches).

## Ground-truth checks that passed (printed numbers vs independent computation)

- label_economics: sign test p = 0.5^7 = 0.0078 for 7 wins/0 losses; 12 seeds, 11 jointly reaching.
- task_cascade: McNemar exact two-sided p for 13 vs 3 discordant = 0.0213, 10 vs 2 = 0.0386; escalation 18 %/12 % of 300 consistent with the discordant counts; $6.533 = 600×$0.01 + $0.533; 1M-request projection ≈ $8,780.
- task_llm_active: 4 vs 0 discordant → p = 0.125; 0.987 × 300 = 296 agreements.
- win_demo router: $0.02 + $0.045 + $0.15 = $0.215 on 400 requests → $0.00054/req, saved $12 − $0.215 = $11.79.
- precedence: brute force over all 2^10 closed subsets gives +35.0 for exactly the 8 items printed; brute force over all 5^8 sprint assignments gives NPV 30.535 with discount 1/(1.05)^t, matching the printed +30.54 and schedule (assignment (0,0,2,1,2,3,1,3)).
- symbolic_export: Gaussian score −(x−2)/1.5 at x=10,100 = −5.333, −65.333; Student-t(4) score −1.25x/(1+x²/4) = −0.4808, −0.0500; constants (−0.5 log 3π, loggamma(2.5) = 0.2847).
- joint_mixture: recovered `joint_weights` map onto the planted matrix up to component permutation (0.479↔0.48, 0.249↔0.24, 0.077↔0.08); Gamma (1,3)/(3,3) and Gaussian ±6/0 recovered; held-out KL 0.0055 nats.
- lookback (shipped seed): three recovered chains are the planted cyclic shifts of the sticky 0.8 matrix (max entry error ≈ 0.09); `seq_log_density` on the encoded batch equals per-observation `log_density` exactly (10/10 values).
- hierarchical_mixture: fitted log-likelihood on the 2000 documents −22509.97 vs −22517.47 under the generating parameters — the fit is +7.5 nats above truth, so the non-planted-looking taus are an equivalent solution of a weakly identified model, not a bad optimum (three other RNG seeds: +8.0, +7.7, −72.1). Manifest's −22509.97 reproduces to 2 decimals.
- semi_supervised: weights 0.5979/0.2964/0.1057 match the manifest's 0.598/0.296/0.106; component k is the exemplar-named one.
- structured_leaves: all six parameters per component within the script's own tolerance. scaling: local and mp agree with truth. production: git hash `6040ea38…` and version 0.8.1 recorded in the header.
- model_comparison ranking: B wins by 567 elpd (LOO) / 571 (WAIC); EM MLE of the two-Gaussian mixture is −1267.3, so the ranking is robust even though B's printed row is not at the MLE (P10-F06).

## Findings

### P10-F01 (real) — `propose().explain()` dependency notes silently skip any numeric field not recommended as gaussian/poisson; the quickstart's strongest planted dependence never appears
`quickstart_example.py` plants `category → amount` (N(2,.5) vs N(.5,.3), ≈0.97 bits of MI) and `category → flag` (≈0.6 bits). The printed notes list only `dependency: $[1] <-> $[2] (0.7 bits)`; `$[1] → $[0]` is absent on the shipped seed and all three seed offsets. Root cause (`mixle/utils/automatic/profiling.py:1732`, `_encode_for_pairwise`): quantile binning of a numeric field runs only when `profile.recommendation in ("gaussian", "poisson")`; `amount` is recommended `mixture`/`generalized_gaussian`, falls to the discrete branch, exceeds `max_cardinality`, and is dropped (`encoded_pairwise_fields = 2`, `pairwise_pairs_checked = 1`). Control: the same records with `amount ~ N(2,1)/N(0,1)` (recommended `gaussian`) yield hints `(1,2) 0.57, (0,1) 0.38, (0,2) 0.2 bits`. The profile warns that field 0 "looks multimodal" — the multimodality is the dependence it then refuses to test. The frontier's structured candidate does model it (held-out −1.43 vs −2.60), so the winner is right; the explanation is wrong. Reproduction: `attacks/probe_propose_deps.py`, `attacks/probe_deps_control.py`.

### P10-F02 (real) — `LDAEstimator` raises `LDAConvergenceError` on 9 of 12 sampling seeds of the shipped LDA configuration; the shipped seed passes by luck
`latent_variable_models_example.py::demo_lda` with `gen.sampler(seed=s)` for s = 3…11 (and with 300 instead of 400 documents at seed 1) raises `LDAConvergenceError: lda_alpha_fixed_point did not converge after 1000/1000 iterations (iteration_budget_exhausted; residual=1e-8…1e-7)` from `mixle/stats/latent/lda.py:1573` (`if not diagnostics.converged: raise`). The residual is a relative |Δα|/Σα of ~1e-8 against `alpha_threshold=1e-8`: alpha is known to 7–8 significant figures and the whole EM run is discarded. The seed-offset attack reproduces it (S=200, S=300 exit 1). Fail-closed-guard pattern: a budget-exhausted fixed point with a tiny residual should warn (as `optimize()` does at its own cap), not raise. Workarounds exist (`fixed_alpha=`, `alpha_threshold=1e-3`) but the `iteration_budget_exhausted` message does not name them (only the `alpha_diverging` branch does). Reproduction: `attacks/lda_crash.py`.

### P10-F03 (real) — `mixture_reduction_benchmark.py`'s "honest iterative baseline" is one fixed-seed EM restart stuck on a degenerate optimum; its printed conclusion is an initialisation artifact
Printed: EM-refit KL 0.5417/0.5421/0.5435/0.2132 for M = 2/4/6/8 and "EM-refit KL / closed-form KL = 5.09 on average (>1 ⇒ closed-form is actually tighter)". The M = 4 and 6 refits are no better than M = 2; after one EM step from the seed-0 init the weights are [0.024, 0.895, 0.017, 0.063] and stay there for 50, 500 or 2000 iterations. `mixle.ops.project` calls `fit(..., rng=RandomState(seed))` with `seed=0` and uses only the target's family, so every "target_seed" in the script is the same init. Varying `project(seed=…)` at M = 4 gives EM KL 0.542, 0.544, 0.102, 0.377, 0.542, 0.104 (closed-form 0.120); at M = 8: 0.215, 0.066, 0.068, 0.514, 0.070, 0.384 (closed-form 0.023). Warm-starting EM from the closed-form M = 4 solution (`optimize(..., prev_estimate=cf)`) improves it monotonically to KL 0.104 in 8 iterations. The true picture is "closed-form is a good init and EM refines it", the opposite of the printed 5× headline. Reproduction: `attacks/reduction_attack.py`, `attacks/reduction_attack2.py`, `attacks/em_monotone.py`.

### P10-F04 (real) — `structured_hmm_example.py` crashes on its own low-rank recovery assertion for roughly one RNG seed in three (and at 10 % data)
Section 1 raises `RuntimeError: recovery NOT demonstrated: fit error 1.59 must be < 0.5 …` with `RandomState(0)` replaced by `RandomState(200)`. Sweep of the identical section, seeds 0–6, `max_its=40`: errors 0.15, 4.05, 0.11, 0.40, 0.15, 0.58, 0.55 (3/7 fail; seed 1 is a merged-state optimum, means [−0.06, 3.94, 3.95, 11.25, …], still 2.89 after 400 iterations). At 10 % of the sequences the shipped seed fails at 0.65. Reproduction: `attacks/seeds/structured_hmm_example_s200.py`, `attacks/lowrank_sweep.py`.

### P10-F05 (real) — `dump_models()` cannot round-trip `HierarchicalMixtureDistribution` or `JointMixtureDistribution`, both registered as `constructor-validated` in `manifests/serialization_schema_manifest.json`
`mixle.stats.dump_models(model)` raises `SerializationError` for the fitted and the generating models of `hierarchical_mixture_example.py` and `joint_mixture_example.py`: HierarchicalMixture — "registered class … requires a class-owned `__pysp_setstate__` hook; constructor fields are absent: mixture_weights, topic_weights"; JointMixture — "constructor rejected serialized state". `to_json()` succeeds (write-only text), `from_json()` fails; pickle round-trips both exactly. The CHANGELOG discloses this gap only for `mixle.stats.directional` and `mixle.stats.matrix`; the serialization manifest lists both classes under `profiles/base` and `profiles/full` with codec `constructor-validated`, `state_version 1`. Twelve other families from these examples round-trip exactly (max |Δ log p| ≤ 2e-15): Composite(G,Cat,Poisson), Mixture(Composite(G,BernoulliSet,P)), LookbackHMM, IntegerMarkovChain, SemiSupervisedMixture, GaussianMixture, `reduce_mixture` output, LDA (true params), IBP, auto-fitted HeterogeneousBayesianNetwork, ppl Poisson and Mix. Reproduction: `attacks/json_roundtrip.py`, `attacks/serial_probe.py`.

### P10-F06 (real) — `model_comparison_example.py` prints B's parameters and "loglik" from an MCMC chain whose bulk ESS on the scale parameters is 3
On the shipped seeds, `b.summary()` (never called by the script) reports for B: `comp0.arg1` ess_bulk = 3.27, `comp1.arg1` ess_bulk = 2.78, ess_tail ≈ 10, 97.5 % quantile of sigma = 4.7. The printed point estimate is sigma2 = 1.62/1.75 (truth 1.0), means −5.85/5.72 (cluster sample means −5.97/5.86), "loglik −1304.71" — 37 nats below the EM maximum −1267.3 on the same data. Re-running with `RandomState(3)` or `(4)` gives sigma2 ≈ 1.0 and loglik −1267.4; with the data seed offset by 200, B's means are −5.39/5.47 and loglik 150 nats below the MLE. The example presents `loglik/aic/bic` as if fitted maxima and prints no ESS/R-hat although the fit exposes them; the winner is unaffected (567 elpd margin). Reproduction: `attacks/modelcmp_attack.py`, `attacks/modelcmp_diag2.py`, `attacks/seeds/model_comparison_example_s200.py`.

### P10-F07 (minor) — `lookback_hmm_example.py` is a single random-init EM with no recovery check; half the seeds land on a degenerate optimum
With seed 101 or 201 (the example's 1000 iterations, `delta=None`), fitted chains contain entries 1e-14…1e-137 and rows like [0, 0.51, 0.49]; training log-likelihood is 45 and 8 nats below the generating model's, held-out (2000 sequences) 697 and 432 nats below truth. Shipped seed 1 and 301 are ordinary overfits (+20/+16 train, −192/−187 held-out). The script prints only `str(model)`. Also stale: the manifest's 2026-07-21 note "only about 480 of the requested 1000 iterations actually execute" — `fit_provenance().iterations` is 1000 on all four seeds. Reproduction: `attacks/lookback_quality.py`.

### P10-F08 (minor) — `peft_lora_grad_leaf.py` dies with a raw `ModuleNotFoundError` when `peft` is absent, after announcing a Hugging Face Hub asset
stdout: `asset repository=peft-internal-testing/tiny-random-gpt2 revision=2f18a2…`; stderr: traceback ending `ModuleNotFoundError: No module named 'peft'` at line 43. The manifest marks the example "needs peft", but the script neither checks nor explains, and its docstring does not say the checkpoint is downloaded from the Hub (network). Reproduction: run the script in the candidate venv.

### P10-F09 (minor) — the task-cluster examples spray hundreds of library-emitted "unconverged fit" warnings, some with a negative last objective gain
`task_distill_example.py` prints 144 copies of `UserWarning: optimize() stopped at the max_its cap … Raise max_its to fit to convergence` from `mixle/task/distill.py`; `task_llm_active` 129, `win_demo` 66, `task_cascade` 41, `structure_learning` 58, `skeptic` 28. Seven report a negative gain (−0.0014 … −56.6 at cap 8) while advising the user to raise `max_its`. None of the docstrings mention warnings. `quickstart_example.py` emits one from inside `propose()` (cap 25, gain 7e-7) although `optimize(records)` on the same six rows converges in 2 iterations silently. Reproduction: `runs/task_distill_example.py.err`, `grep "objective gain -" runs/*.err`.

### P10-F10 (docs) — `semi_supervised_mixture_example.py` docstring says the script crashes with `NotImplementedError` and is "left as-is so it keeps reproducing it"; it runs clean and recovers the planted weights
Stale "KNOWN ISSUE (0.8.0)" paragraph (exit 0; weights 0.598/0.296/0.106; components anchored). Related: `best_of(data, data, …)` selects the restart on the training rows, which `joint_mixture_example.py` explicitly avoids with a three-way split (STAT-RR23-04).

### P10-F11 (docs) — `hierarchical_mixture_example.py` docstring runtime and inert `print_iter`
Docstring: "Runtime is on the order of one to two minutes"; measured 17.3 s (manifest ~20 s). `optimize(data, est, max_its=2000, print_iter=500, …)` passes no `out=`; `out` defaults to `None`, so `print_iter` produces nothing.

### P10-F12 (minor) — `latent_variable_models_example.py` prints LDA "recovered topics" with exact zeros where the truth has 5–10 % mass, under "recovers the planted structure"
Printed `[[0.0, 0.05, 0.33, 0.63], [0.67, 0.31, 0.02, 0.0]]` vs true `[[0.6, 0.25, 0.1, 0.05], [0.05, 0.1, 0.25, 0.6]]`. To convergence (3000 its): `[[0, 0.026, 0.344, 0.63], [0.667, 0.333, 0, 0]]`, fitted alpha ≈ 2.6 (true 1.0, never printed); at 3000 docs × 60 words the zeros persist (alpha 1.7). Weak identifiability along "sharper topics + larger alpha" rather than a bug, but the side-by-side overstates recovery. Reproduction: `attacks/lvm_attack.py`, `attacks/lda_crash.py` (last line).

### P10-F13 (minor) — degenerate-input rough edges from the size attack
- `model_comparison_example.py` with `N_PER_MODE = 1` (2 points): `OverflowError: (34, 'Result too large')` from `mixle/ppl/_lowering.py:237` via `ppl/core.py:1153 make_dist`.
- `mixture_reduction_benchmark.py` with `n_samples=1`: `project()` returns an 8-component GMM fitted to one point (KL 1.4e9), no components-vs-samples guard.
- `joint_mixture_example.py` / `semi_supervised_mixture_example.py` with 1 row: `TypeError: … components must be generative probability laws; likelihood factors found at indices […]` names a symptom of an empty split, not the cause.
Clear refusals elsewhere (`propose requires at least three records`, `solve() needs at least 8 example inputs`, `val_texts needs at least four rows`, `embedding_health requires at least three observations`) are fine.

## Attacks that did not break anything

- Seed offsets +100/+200/+300 on 26 scripts (78 runs): all exit 0 except P10-F02 (LDA, 2/3) and P10-F04 (low-rank HMM, 1/3); vlm copies need the sibling `multimodal_stage1_demo.py` beside them (import-by-sibling, not a defect) and then pass 3/3. joint_mixture KL 0.0023/0.0014/0.0020; structured_leaves "recovered: True" 3/3; multimodal "OK" 3/3; scaling exact 3/3; task_llm_active reports "INCONCLUSIVE" honestly 3/3; label_economics has no literal seeds (already a 12-seed paired design).
- 10 % data on 25 scripts: all exit 0 except P10-F04 (0.65 error); label_economics prints "target not reached under both strategies on any seed — widen budgets" honestly; vlm no longer flags the merged regime at 4 volumes per class but its "OK" line only claims the diagnostic ran; structured_leaves prints "recovered: False" honestly at 2 rows.
- Single observation on 25 scripts: clean refusals or example-level IndexError/KeyError (printing `data[1]`, `P(a)`) except the three rough edges in P10-F13; hierarchical, ppl, reduction, reasoner, task_cascade, task_extraction, task_llm_active, project_neural, multimodal run to completion.
- JSON round-trip: 12 families exact; pickle exact for all 14 including the two JSON failures.
- EM monotonicity: `optimize(prev_estimate=closed-form GMM)` trace strictly increases (−19414 → −19345 over 8 iterations); `seq_log_density` equals per-observation `log_density` for GMM, lookback HMM and the reduced mixture.
- Hierarchical mixture: four RNG seeds; fitted log-likelihood ≥ truth on three; the manifest's convergence narrative (2000-iteration cap, unconverged by delta=1e-9, ~20 s) holds.
- IBP: fitted `feature_probs` 0.42/0.41/0.42 vs generating Beta mean 0.4 — consistent.
- precedence: closure and schedule are the brute-force optima. symbolic_export: all constants and scores match closed forms.
- reasoner: the perfect-separation warning comes from the discarded `spend → plan` orientation of the structure search; the intervened answer (mean spend under do(plan=pro) = 99.96 vs planted 100) is unaffected.
- No private-name imports in the 29 scripts beyond `_row_normalize` in `structured_hmm_example.py`; no deprecated-argument warnings observed.

## Summary

- Counts: blocking 0, real 6 (F01–F06), minor 5 (F07, F08, F09, F12, F13), docs 2 (F10, F11).
- Worst: P10-F02 — the shipped LDA example survives only because seed 1 happens to converge; 9 of 12 sampling seeds (and the shipped seed at 300 documents) crash `optimize()` with `LDAConvergenceError` on a residual of ~1e-8, a fail-closed guard destroying an otherwise finished fit.
- Runner-up: P10-F05 — `dump_models` refuses the models of two shipped examples while the serialization manifest registers both as `constructor-validated`.
- The examples' printed arithmetic (paired tests, dollar accounting, NPV, symbolic scores) all reproduces by hand; the defects are in what the examples conclude (F03, F06, F12) and in how fragile their own paths are to a seed change (F02, F04).
- All evidence is under `<review-root>/reviews-081/pass-10/{runs,attacks}`.
