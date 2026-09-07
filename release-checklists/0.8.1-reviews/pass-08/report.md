# Adversarial review — PASS 08 — teaching surface (tutorials, data_science notebooks, README snippets)

- Wheel: `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, git tree `6040ea38`, branch `release/0.8.1`.
- Version verification (run from this work dir, outside any checkout):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` ->
  `0.8.1 /Users/grantboquet/mixle/.ci-repro-colima/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Interpreter: `/Users/grantboquet/mixle/.ci-repro-colima/candidate-081/venv/bin/python` (3.12). torch 2.14.0, transformers 5.16.1, pyspark 4.2.0, mpiexec present, JAVA_HOME = openjdk@17.
- Work dir: `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-08/` (executed notebooks in `executed-tut/`, `executed-ds/`; diffs in `tut-diff.txt`, `ds-diff.txt`; README scripts in `readme/`; helper scripts `run_nbs.py`, `nbdiff.py`, `showcell.py`, `imports_ast.py`).
- Nothing outside the work dir was modified; the venv was not touched.

## Executed notebooks (nbconvert --execute, ExecutePreprocessor.timeout=1500, subprocess-bounded at 1800 s, 3 at a time)

### tutorials/ (12 of 12, all exit 0)

| notebook | exit | wall s | cells differing from stored output |
|---|---|---|---|
| distributions_and_combinators | 0 | 6.6 | 15 / 27 (repr `keys=None`, float noise, `WeightedObservation` repr) |
| fitting_and_estimation | 0 | 5.2 | 4 / 7 (new max_its warnings; "iterate" and GEM means moved, see F05/F06) |
| probabilistic_programming | 0 | 11.6 | 9 / 14 (sampler path changed; ESS 3000 for 1000 draws, F08; `increasing` now correct, F13) |
| bayesian_distributions | 0 | 4.7 | 5 / 7 (fit no longer prints iterations; MAP corrected, F12) |
| latent_variable_models | 0 | 62.9 | 16 / 26 (JointMixture DeprecationWarning F03; print_iter silent F04; HMM sampler sequences changed) |
| embedding_with_htsne | 0 | 13.2 | 4 / 5 (purity 0.87 -> 0.86, trust/cont within 0.01) |
| mcmc_sampling | 0 | 8.4 | 3 / 12 (ESS 93 -> 94, repr) |
| enumeration | 0 | 5.4 | 2 / 8 (np.float64 wrappers, timing) |
| accelerated_engines | 0 | 9.0 | 5 / 10 (max_its warnings at gain 8.6e-09; engines list adds `jax`; to_latex symbolic; to_sage unavailable here) |
| parallel_estimation | 0 | 14.4 | 3 / 8 (max_its warnings; timings; streaming ll -85471.0 -> -85470.8) |
| model_parallel_estimation | 0 | 8.3 | 1 / 3 (timestamp only; live 2-rank torch.distributed run passed) |
| estimation_using_spark | 0 | 62.3 | 7 / 15 (print_iter silent; sampled data changed; max_its warnings) |

### data_science/ (14 executed, all exit 0)

| notebook | exit | wall s | differing cells |
|---|---|---|---|
| gaussian_mixtures_in_depth | 0 | 12.2 | 4 / 7 (worst random-init LL -2236 -> -2252; figures) |
| hmm_from_scratch | 0 | 6.4 | 1 / 6 (Viterbi accuracy 0.977 -> 0.993, agreement 1.000 — improved) |
| em_and_map_strategies | 0 | 8.4 | 3 / 13 (35 max_its warnings in the restart cell; prior repr adds `improper_receipt=None`) |
| mle_and_sufficient_statistics | 0 | 8.3 | 0 / 7 |
| regression_and_glms | 0 | 9.3 | 0 / 6 |
| ppl_end_to_end_case_study | 0 | 10.2 | 0 / 6 |
| model_selection_and_cross_validation | 0 | 6.9 | 0 / 7 |
| conjugate_and_nonparametric_bayes | 0 | 19.6 | 8 / 15 (stick weights +-0.001; 4 figure outputs absent in fresh run — not investigated) |
| topic_modeling_lda | 0 | 183.5 | 5 / 8 (F11) |
| mcmc_uncertainty_quantification | 0 | 14.2 | 3 / 13 (ESS(mu)=4764 / 1500 draws, F08) |
| latent_models_in_practice | 0 | 62.6 | 28 / 52 (collapsed component var 8.5e-97, F10; print_iter silent; JointMixture warning) |
| markov_chains_for_sequences | 0 | 5.3 | 1 / 6 (pseudo-words differ for same rng) |
| heterogeneous_mixed_type_modeling | 0 | 44.5 | 3 / 6 (F09) |
| missing_data_and_imputation | 0 | 6.0 | 0 / 5 |

## README python snippets (run verbatim from `readme/`, then varied)

| # | snippet | status | wall s | notes |
|---|---|---|---|---|
| 1 | Quickstart `optimize(records, out=None)` + `log_density` + `sampler().sample(5)` | OK | 2.8 | Composite(Gamma, Categorical, Optional(Categorical)); ld -2.913; 5 samples of the right shape |
| 2 | Distill `solve(teacher, inputs)` / call / `report()` / `save()` | OK, 15 warnings | 3.9 | rule teacher on 80 strings: answers locally, report promoted=True agreement 1.0 escalation 0.3; **15 UserWarnings from mixle/task/distill.py:744 with default filters** (F05) |
| 3 | `optimize(x, my_module)` / `model.module` | ERR then OK | 2.7 | module returning (N,1) scores rejected (F17); with per-row scalar: mu 2.946 sigma 2.027 vs sample 2.946/2.027; `model.module is my_module` True |
| 4 | Nested `HiddenMarkovEstimator([MixtureEstimator([GaussianEstimator()]*5), GradEstimator(my_module)])` | OK but degenerate | 3.4 | collapses to one state at default max_its=10; correct at max_its=300 (F02) |
| 5 | Engines: `TorchEngine(device=,dtype=)`, `precision="auto"`, `backend="spark"` | kwargs accepted | — | `TorchEngine(device="cpu", dtype="float32")`, `precision="auto"`, `backend="mp"` all fit the same mixture; CUDA/spark not exercised here (spark covered by the tutorial) |
| 6 | Enumeration on SmolLM2-135M | **ERR verbatim** | 9.0 | ValueError non-normalized (sum 0.998532) — bf16 default in transformers 5.x (F01); with `dtype=torch.float32` the README's printed outputs reproduce exactly |
| 7 | `mixle.ppl` block (Normal/Mix/Markov/regression) | OK | 3.0 | Normal(free,free) params mean 0.0667 sd 2.0385 (README says "mean + standard deviation": correct, `params` reports sd); prior fit -> ConjugatePosterior; Mix means -1.97/2.10; Markov transitions [[1,0],[0.5,0.5]] initial [1/3,2/3] hand-verified; regression coefficients 2.007/-0.963/0.506 for truth 2/-1/0.5 |

## Findings

### P08-F01 (blocking) — README enumeration snippet errors verbatim on the candidate venv
The 'Enumeration & ranking' snippet fails at `continuations.top_k(3)` with `ValueError: next_logprobs returned a non-normalized distribution (kept probabilities sum to 0.998532, expected ~1.0)`. Cause: transformers 5.16.1 (inside the wheel's declared `transformers<6,>=4.40`) loads SmolLM2-135M as bfloat16; `mixle/enumeration/autoregressive.py::_parse_steps` rejects `|sum exp(lp) - 1| > 1e-4`. With `dtype=torch.float32` the snippet runs in 3.4 s and prints exactly what the README shows: `[' located in the', ' the city of', ' the capital of']`, `unrank(5)` -> `' Paris, the'`, `rank=6, cumulative_probability=0.11397`. Repro: `readme/s6_enum.py` (fails) vs `readme/s6_enum_f32.py` (passes).

### P08-F02 (real) — README nested-HMM example returns a collapsed one-state HMM at the default `max_its=10`
`readme/s4_hmm_nest_fixed.py`: initial weights `[0.99997, 2.9e-05]`, both transition rows point at state 0 (>=0.9995), the five-cluster mixture grows a component at 11.69 with weight 0.64 that absorbs the neural state's data, neural leaf sigma 3.62; plus the max_its warning. `readme/s4_hmm_nest_300.py` (max_its=300): transitions `[[0.838,0.162],[0.133,0.867]]`, clusters at +-19.98/+-9.98/0.06, neural mu 11.94 sigma 0.998. The README says "one call fits the whole thing" and never mentions `max_its`.

### P08-F03 (real) — `JointMixtureDistribution(w1, w2, taus12, taus21)` tutorial: w2/taus21 are discarded
latent_variable_models cell 18 (and data_science/latent_models_in_practice) constructs with `w2=[0.7,0.2,0.1]`, `taus21=I`. The candidate emits `DeprecationWarning: w2 and taus21 describe a different joint law; the canonical law derived from w1 and taus12 is used` and stores `w2=[0.52,0.31,0.17]`, `taus21=[[0.923,0.194,0.353],[0.058,0.774,0.176],[0.019,0.032,0.471]]`. The prose says taus21 "links the hidden states"; the built model is not the stated one. Stored outputs show no warning.

### P08-F04 (minor) — `print_iter=` is inert without `out=`; tutorials rely on it
`optimize`/`best_of` default `out=None`, so `print_iter=50` prints nothing (documented in the docstring). latent_variable_models cells 5/16/25, estimation_using_spark cells 12/25/27, data_science latent_models_in_practice cells 17/31/46 and markov_chains cell 7 pass `print_iter` and their stored outputs show progress lines the candidate never emits. The README quickstart comment `optimize(records, out=None)  # (out=None: quiet)` implies passing it changes something.

### P08-F05 (real) — max_its-cap `UserWarning` fires across ~35 tutorial cells, from library internals, and at gains of 8.6e-09
README `solve()` snippet: 15 warnings raised at `mixle/task/distill.py:744` (library early-stopping loop calls `optimize(max_its=1)` without `delta=None`). fitting_and_estimation cell 3: 7 warnings, 5 attributed to `mixle/inference/estimation.py:2167` inside `best_of`. accelerated_engines cell 3 warns at gain 8.56e-09 vs delta 1e-09 on a deliberately capped timing loop. em_and_map_strategies cell 30: 35 warnings. No stored output contains a warning and no prose explains one. The tutorial's `optimize(max_its=40)` on its pinned seed is also genuinely unconverged (converges at iteration 75; seeds 2-7 converge within 25).

### P08-F06 (docs) — tutorials teach `iterate`, which does not exist in the wheel
`from mixle.inference import iterate` -> ImportError; `grep -rn "def iterate" site-packages/mixle` -> nothing. tutorials/README.md, fitting_and_estimation prose and output label, latent_variable_models cell 25 comment, and latent_models_in_practice prose all describe it. The cell labelled "iterate ->" calls `optimize(max_its=25)` and now prints `[-1.8, 2.2]` (stored `[-2.1, 1.98]`); GEM/ECM cell prints `[-1.32, 2.39]` (stored `[-2.05, 2.02]`).

### P08-F07 (docs) — prose cites removed modules and the old product name
`mixle.utils.em` (fitting_and_estimation x2, parallel_estimation), `mixle.utils.objectives` (fitting_and_estimation), `mixle.utils.mcmc` (mcmc_sampling) are all `ModuleNotFoundError` — the 0.8.0 CHANGELOG lists their removal. "PySparkPlug" appears 7 times across distributions_and_combinators, latent_variable_models, estimation_using_spark.

### P08-F08 (real) — reported ESS exceeds the draw count by up to log10(N)x
`MCMCResult.effective_sample_size()` = n / tau with tau floored at `1/log10(N)` (`mixle/inference/diagnostics.py::_geyer_tau`), uncapped: 1000 HMC draws -> `[3000., 3000.]` while `result.bulk_ess` for the same fit reports 1000. probabilistic_programming cell 7 prints `HMC acc 1.00 ESS 3000` (stored 1000); data_science/mcmc_uncertainty_quantification cell 14 prints `HMC: acceptance=1.000, ESS(mu)=4764 / 1500 draws` (stored 1500/1500) beside prose about "ESS per draw".

### P08-F09 (real) — heterogeneous_mixed_type_modeling now contradicts its own thesis
Seeded with RandomState(1): "ARI numeric-only fields : 0.759" vs "ARI all heterogeneous : 0.716" (stored 0.730 vs 0.912) under prose saying "using the full record should cluster markedly better"; ablation prints `session -0.083` (dropping the field improves ARI); DP mixture used 11 clusters (truth 4).

### P08-F10 (real) — latent_models_in_practice returns a collapsed Gaussian component (variance 8.5e-97) silently
Cell 32 first line: `MixtureDistribution([GaussianDistribution(0.6317, 8.548e-97), GaussianDistribution(7.554, 7.042)], [0.0032, 0.9968])` from `optimize(max_its=500, rng=RandomState(1), init_estimator=...)` with no warning for that cell. Stored output had O(1) variances. (The sampled data for the same seed also changed — cell 11 — so the seed path differs; the silent collapse is the defect.)

### P08-F11 (docs) — topic_modeling_lda stored outputs stale; coupled-conditional now trails unsupervised
Split 793/86 -> 792/87; unsupervised hit@1 0.581 -> 0.460; coupled conditional hit@1 0.628 -> 0.414 and below unsupervised at every k; printed runtime 40 s -> 176 s (host was loaded; re-measure in isolation).

### P08-F12 (docs) — bayesian_distributions cell 3 stored output was wrong; candidate is right
Hand computation of the Normal-Gamma joint MAP (mu_n = 5*3.111/6 = 2.5923, sigma2 = beta_n/(alpha_n - 1/2) = 4.0963) matches the candidate's `MAP mu 2.592, sigma2 4.096`. The stored `MAP mu 3.124, sigma2 5.881` exceeds the sample mean with a prior centred at 0 and cannot be a posterior mode.

### P08-F13 (docs) — probabilistic_programming cell 22 stored `increasing(v)` output was degenerate; candidate is right
Candidate `[0.01 1.44 1.44 3.06 3.99]` (isotonic fit of column means [0,2,1,3,4]); stored `[2. 2. 2. 2. 2.11]` with a code comment justifying the pinned seed. Consistent with the CHANGELOG's Nelder-Mead feasibility fix.

### P08-F14 (minor) — RV algebra "exact" vs sampled moments
`3*Normal(0,1)+1` prints `mean=1.02 var=8.95` from Monte-Carlo `.mean()/.var()` next to the comment `-> Normal(1, 9), exact`; `.dist` is a doubly nested `TransformDistribution`, not a Gaussian.

### P08-F15 (minor) — PPL regression rough edges
Fitted regression prints `RV(bound=None)`; `given=pd.DataFrame(...)` rejected ("must be a mapping from field names to one-dimensional arrays"); `predict({'x':[1.0],'z':[0.0]})` -> "predictive sample count must be an exact positive integer"; `log_density(y, given=...)` unsupported. Coefficients themselves are correct.

### P08-F16 (docs) — stored outputs stale for seeded samplers and reprs across the corpus
Same seed, different sample: HMM sampler (latent cells 21/31), spark cell 21, ppl `sample(3)` and every downstream rng-dependent cell, markov pseudo-words; repr changes (`keys=None`, `WeightedObservation`, engines list adds `jax`, symbolic to_latex). 74 of 152 tutorial code cells differ; 8/14 data-science notebooks differ. Fresh Viterbi path in latent cell 31 verified against its data.

### P08-F17 (minor) — README's "any module exposing log_density(x)" omits the per-row-scalar contract
Rows arrive as an (N,1) column; a module whose scores broadcast to (N,1) is rejected with a clear `must return exactly one score per row; got shape (500, 1)`. `.sum(-1)` fixes it.

## Attacks that did not break anything

- All 12 tutorials and 14 data-science notebooks execute to completion (exit 0), including the Spark tutorial under openjdk@17, the MPI/dask/mp backends in parallel_estimation, and the live two-rank `torch.distributed` run in model_parallel_estimation.
- Every `mixle.*` name the tutorials import resolves (167 distinct `(module, name)` pairs across 16 modules; none private, none missing). Submodules `mixle.inference.em/estimation/gradient_fit/objectives/streaming`, `mixle.stats.compute.backend`, `mixle.utils.optsutil` lack `__all__` but every taught name exists; EM strategies, `fit_mle/fit_map`, `constant/harmonic` are not re-exported at `mixle.inference` top level but the tutorials import them from the submodules that exist.
- No tutorial uses the deprecated `conformal_alpha` alias or the legacy transposed `taus21` layout (the JointMixture warning above is the inconsistent-law path, not the layout path).
- README quickstart variations all produce sane models: pandas DataFrame (same Composite as the tuple form; `log_density` of a row tuple works, a dict row is rejected as expected for a positional Composite), list of dicts (RecordDistribution), numpy 1-D (Gaussian), numpy 200x3 (MultivariateGaussian), ints (NegativeBinomial), strings (Categorical), 2000 rows, explicit `rng=`; sample-then-score round-trips.
- PPL variations: `Mix` with 3 components on 6 points and on 900 points (means -5.04/0.03/4.98), `Mix` with 2 components on 3-cluster data, `Normal` on numpy array / pandas Series (identical params), `Markov(states=3)` and numpy sequences, single-point and two-identical-point fits (sd floor 1e-4, no crash), prior fit returns a ConjugatePosterior with posterior sd 0.408.
- `optimize(..., precision="auto")`, `backend="mp"`, `engine=TorchEngine(device="cpu", dtype="float32")` all accept the README's keyword forms and fit; `TorchEngine(device="cuda", ...)` constructs lazily on a CUDA-less host.
- Hand checks that passed: PPL Markov transition/initial matrices on the README's three sequences; PPL regression coefficients vs truth; Normal-Gamma MAP (candidate correct, stored wrong); ensemble ESS scales with draws (62/195/622 for 100/400/800); fresh Viterbi decoding matches the sign of the data; hmm_from_scratch Viterbi accuracy improved to 0.993 with 1.000 agreement against the from-scratch implementation.
- `solve()` end to end: labels once, answers locally, escalates 30% on holdout, `report()` and `save()` produce the documented artifacts (`manifest.json`, `weights.safetensors`).

## Summary

- Counts: 1 blocking, 6 real, 4 minor, 6 docs (17 findings).
- Worst: P08-F01 — the README's enumeration snippet raises `ValueError` verbatim on the candidate venv because transformers 5.x loads SmolLM2 in bfloat16 and mixle's normalization check tolerates only 1e-4; `dtype=torch.float32` reproduces the README outputs exactly.
- Next worst: P08-F02 — the README's flagship nested-HMM example collapses to one state at the default `max_its=10`; and P08-F03 — the JointMixture tutorial builds a different model than it states.
- Systemic: the new max_its warning fires from library internals (solve, best_of) and across ~35 shipped tutorial cells; reported ESS can exceed the draw count 3x.
- Every stored tutorial output should be regenerated on the candidate; several stored outputs (bayesian MAP, `increasing(v)`) were wrong and the candidate corrects them.
