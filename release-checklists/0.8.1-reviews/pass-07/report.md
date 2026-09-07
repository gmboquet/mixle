# Adversarial review pass 07 — application, geoscience and architecture-study notebooks

- Pass: 07 of 10 (independent; no shared context with other passes)
- Focus: `notebooks/applications/` (19 of 20; `malware_certificate_embedding.ipynb` skipped per brief), `notebooks/exploration_geoscience/` (11), `notebooks/architecture_studies/` (6)
- Candidate wheel: `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, git tree `6040ea38` (branch `release/0.8.1`)
- Version verification (run from the pass work dir, outside any source checkout):
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Work dir: `<review-root>/reviews-081/pass-07/` (executed notebooks in `executed/`, stored-vs-fresh diffs in `diffs/`, attack scripts `atk_*.py`, runner logs in `logs/`)
- Environment note that matters for every timing claim: the candidate venv has **no `numba`** (`mixle[numba]` is an optional extra; `pip show numba` -> not found). Every "numba" number in `architecture_studies/` was therefore produced by mixle's non-numba path. See P07-F11 / P07-F15.
- Companion packages in the venv: mixle-pde / mixle-physics / mixle-sim at `0.8.0.dev0`; 12 notebooks in scope import `mixle_pde`.

## Method

1. All 36 notebooks in scope were executed fresh on the candidate with `jupyter nbconvert --execute --allow-errors --ExecutePreprocessor.timeout=1500`, three at a time, under a `subprocess.run(timeout=2400)` wrapper (`runner.py`). Two architecture notebooks were re-executed alone (`executed_solo/`) so the timing comparison is uncontended.
2. Every executed notebook was diffed cell-by-cell against the corpus copy (`nbdiff.py`; stream/text outputs, figure-object reprs ignored).
3. Narrative attacks: seed perturbation (3-4 alternative seeds) on the DP mixture, HMM detector, 2-regime mixture, topic mixture; iteration-budget sweeps; 10% data truncation for LocalLevel, radon mixed model, behavioral Markov/HMM; normalisation checks on the knowledge-graph likelihood; scipy cross-checks of printed log-likelihoods.
4. Every dataset loader in scope (15 CSVs, 2 JSON, the UMLS/translation/fake-news text files) was loaded the way the notebooks load it and inspected for dtype, header-as-data, NaN and index problems.

## Executed notebooks (exit status, wall time)

Batch run (3 concurrent), all exit 0, no error cells in any executed notebook (`--allow-errors` was set; verified none were swallowed):

| notebook | exit | wall s | stored wall s (June) |
|---|---|---|---|
| applications/baseball_hierarchical_shrinkage | 0 | 5.8 | 158 (MCMC cell now falls back, F04) |
| applications/bayesian_tweet_rhythm | 0 | 8.0 | 15 |
| applications/behavioral_anomaly_detection | 0 | 9.2 | 21 |
| applications/fake_news_detection | 0 | 8.0 | 4 |
| applications/generative_title_decoding | 0 | 29.5 | 19 |
| applications/machine_translation_alignment | 0 | 40.6 | 12 |
| applications/knowledge_graph_umls | 0 | 107.6 | 316 |
| applications/radon_mixed_effects | 0 | 12.2 | 29 |
| applications/geospatial_tweet_clusters | 0 | 122.9 | 309 |
| applications/option_pricing_and_implied_volatility | 0 | 34.3 | 69 |
| applications/social_fake_news_detection | 0 | 14.1 | 8 |
| applications/stock_portfolio_optimization | 0 | 6.7 | 6 |
| applications/retail_trend_statespace | 0 | 30.9 | 63 |
| applications/oil_exploration_decision | 0 | 40.1 | 23 |
| applications/magma_reservoir_gravity_inversion | 0 | 6.5 | 3 |
| applications/flow_inversion | 0 | 75.9 | 35 |
| applications/synthetic_aperture_sonar | 0 | 174.7 | 41 |
| applications/radar_tomography | 0 | 245.0 | 227 |
| applications/seismic_full_waveform_inversion | 0 | 498.2 | 307 |
| exploration_geoscience/00_foundations | 0 | 2.5 | 0 |
| exploration_geoscience/basin_thermal_history | 0 | 7.1 | 3 |
| exploration_geoscience/biostratigraphy_event_ordering | 0 | 88.9 | 58 |
| exploration_geoscience/crosshole_gpr_tomography | 0 | 6.4 | 3 |
| exploration_geoscience/detrital_zircon_provenance | 0 | 29.6 | 9 |
| exploration_geoscience/geochemical_fingerprinting | 0 | 5.3 | 3 |
| exploration_geoscience/magma_reservoir_gravity | 0 | 6.6 | 3 |
| exploration_geoscience/mineral_exploration_kriging | 0 | 15.9 | 6 |
| exploration_geoscience/near_surface_joint_inversion | 0 | 43.1 | 58 |
| exploration_geoscience/well_log_facies_hmm | 0 | 5.4 | 3 |
| exploration_geoscience/where_to_drill_value_of_information | 0 | 68.6 | 54 |
| architecture_studies/import_and_warmup | 0 | 23.6 (solo 15.4) | 20 |
| architecture_studies/engine_benchmarks | 0 | 20.0 | 4 |
| architecture_studies/data_scaling | 0 | 8.1 | 2 |
| architecture_studies/model_scaling | 0 | 9.4 | 3 |
| architecture_studies/parallel_scaling | 0 | 9.8 | 81 |
| architecture_studies/ppl_scaling_vs_pyro_stan | 0 | 18.7 (solo 14.2) | 44 |

Stored outputs date from 2026-06-23 (applications, architecture), 2026-06-26/29 (parallel_scaling, geoscience) and 2026-07-10 (radar); i.e. they predate most of the 0.8.0 fix waves. 33 of 36 notebooks have at least one text-output cell that differs from the stored copy beyond figure reprs; the findings below separate genuine behaviour changes from stale outputs.

## Findings

### P07-F01 (blocking) — HiddenMarkovEstimator EM sits on an initialization plateau; the behavioral-anomaly HMM detector scores below chance
`behavioral_anomaly_detection.ipynb` cell 23 fits `HiddenMarkovEstimator([CategoricalEstimator(pseudo_count=0.5)]*6)` with `max_its=40, rng=RandomState(6)` and reports impersonation-attack AUC. Stored: `HMM AUC 0.837 vs first-order Markov AUC 0.939`. Fresh: `HMM AUC 0.468` — and the notebook still prints "both read the transition/phase structure that a bag-of-process is blind to". The per-iteration log-likelihood (`atk_hmm_trace.py`) is flat for ~35 iterations (-46604.7 -> -46599 by iteration 30, gains of 0.05/iter) and only takes off around iteration 36; at 40 the HMM (-46502) is barely better than the bag-of-processes baseline (-46605). Seeds 6/1/2/3 at 40 iterations: AUC 0.468/0.463/0.662/0.466. 200 iterations: 0.865/0.812/0.880. `delta=1e-4` needs 433 iterations (LL -40637). `init_p` 0.1/0.5/1.0 all plateau. The candidate does emit the new "stopped at the max_its cap" UserWarning and `fit_provenance().converged=False`, but the notebook's first cell suppresses warnings, so its own path prints a wrong conclusion. The same seed/data/budget worked on the June build.

### P07-F02 (blocking) — 2-component Gaussian mixture collapses to w=[0.993, 0.007], reported converged with no warning, on stock_portfolio_optimization's own seed
Cell 17 fits `MixtureEstimator([MultivariateGaussianEstimator(dim=6)]*2)` on a 1500x6 simulated two-regime panel with `rng=RandomState(0), max_its=80`. Stored: calm 0.79 / crisis 0.21 (the simulation's stationary crisis probability is 0.23). Fresh: `calm regime: weight 0.99 ... crisis regime: weight 0.01 ... avg correlation 0.02` and `posterior crisis-probability correlation with the true regime: 0.15`; the markdown asserts the mixture "recovers them". `fit_provenance()`: `converged=True, iterations=24, final_objective=23713.7, repairs=()`, no warning. Seeds 1, 2, 4 and `sklearn.GaussianMixture(2, n_init=5)` all reach LL 24346.5 with w=[0.788, 0.212]; seed 3 also collapses; 400 iterations do not escape (`atk_stock.py`, `atk_stock2.py`). A component carrying ~10 of 1500 rows is delivered as a clean converged fit.

### P07-F03 (real) — PPL `.fit()` on all-constant parameters is a silent no-op; three notebooks print unfitted likelihoods, one a false claim
`Poisson(2.0).fit(x)` returns `lam=2.0`, `.result is None`, no warning (`Poisson(free).fit(x)` fits). The stored June outputs show the constants used to be fitted: `bayesian_tweet_rhythm` cell 9 stored `Poisson -934 vs negative binomial -594` equals the scipy MLE log-likelihoods (-933.67 / -593.81) on the same 164 counts; fresh prints `Poisson -3706 vs negative binomial -1502; AIC favours NB by -4406` (scipy at lam=2 gives -3706.08 exactly — both models unfitted). `option_pricing_and_implied_volatility` cell 23 now prints `Student-t log-lik -1554 beats Gaussian -527` — a false sentence (stored: -464 beats -527). `fake_news_detection` cell 13 builds two unfitted NegativeBinomials (not printed). The ppl module doc defines `free` as the fit marker, so the notebooks misuse the API — but a `.fit()` that changes nothing and returns `result=None` without a word is the library's rough edge, and the semantics changed since June with no CHANGELOG entry. (`atk_tweet_ll.py`)

### P07-F04 (real) — grouped Bernoulli MCMC now raises NotImplementedError; baseball notebook silently falls back
`Bernoulli(Beta(a,b).each()).fit(games, how='mcmc')` -> `NotImplementedError: grouped NUTS currently supports a Normal likelihood.` (`mixle/ppl/inference.py:_grouped_target`). Stored June output for cell 10: "mixle MCMC sampled per-player posteriors under the EB population prior (acceptance rate 0.37)" in a 155 s cell; section 5's markdown promises exactly this. The notebook's `except NotImplementedError` prints the fallback message and the run continues (which also reseeds the shared `rng`, changing every simulated number in cells 14-18). Not in the CHANGELOG.

### P07-F05 (real) — `LocalLevel().fit()` returns an iteration-capped fit silently; the retail notebook's two "matches" claims are stale
On the retail notebook's 739-day series, `LocalLevel().fit(s).result` has `converged=False, termination_reason='iteration_limit', iterations=100` and emits no warning (contrast `optimize()`, which warns). `max_its=1000` -> level_sd 3.033, `max_its=5000` -> 3.024 and still `converged=False` (delta=1e-8 is never met). Fresh cell 6 prints `level-drift sd 3.26` (stored 3.05) and `from-scratch smoother matches mixle LocalLevel: max abs difference 3.3013` (stored 0.0027) under prose saying it "reproduces it to numerical precision"; the 3.30 is at t=0 and decays geometrically to 0 by t~43 (initial-state convention: mixle's `initial_mean/initial_sd` vs the notebook's a0=s[0], P0=1e6). Cell 28: "the two routes reach the same optimum (log-likelihoods within 0.08)" (stored 0.00). Truncated to 73 points: also silent `iteration_limit`. (`atk_retail.py`, `atk_trunc.py`)

### P07-F06 (real) — knowledge_graph_umls prints the joint log p(h,r,t) as "held-out mean log p(t|h,r)" against the conditional uniform baseline
`KnowledgeGraphDistribution.log_density` now returns the normalized joint (docstring), i.e. `log p(t|h,r) - log 135 - log 46`; summing `exp(log_density)` over all 135 tails gives 0.0002. Cell 6 prints `held-out mean log p(t | h, r) on the test facts = -12.118 (uniform baseline = -4.905)` — telling the reader the model is worse than uniform. Stored: -3.558 (so log_density was conditional in June). `atk_kg2.py`: mean `tail_log_posterior(h,r)[t]` = -3.384, joint - conditional = -8.734 = -log(135)-log(46) exactly. The model is actually better than in June (MRR 0.548 -> 0.735, Hits@10 0.818 -> 0.921). No CHANGELOG entry for the semantics change.

### P07-F07 (docs) — baseball DP-mixture cell asserts multimodality that the fresh fit does not show
Cell 12 prints `DP mixture used 6 effective components; the implied talent density has 1 modes at [0.245]` followed unconditionally by "the nonparametric prior recovered the multi-mode structure (true modes ~.230, ~.310)". Stored: 2 modes at [0.235 0.274]. Seeds 3/0/1/2/4 at the notebook's 18 iterations: all unimodal, ELBO still rising. At 100 iterations 3 of 5 seeds show a second mode at ~0.38, which is not the true 0.31. (`atk_dp.py`)

### P07-F08 (docs) — stored outputs in baseball, stock_portfolio and parallel_scaling do not come from the current source
Baseball cell 4 is pure numpy under `RandomState(1)`: current source (`trials=400`) prints `k= 5: 0.97->0.47 ...` on any build; the stored `0.99->0.49` is reproduced by `trials=1000` or `2000`. Cell 18's held-out log-likelihood of 60 players at `n_sim=80` is -2653 deterministically; stored -6723 needs a larger `n_sim`. Stock cell 17 prints three lines, the stored output has two. parallel_scaling's stored rationale strings say N=10000 / N=500 where the source has 1_500 / 250. These cells cannot serve as a regression baseline.

### P07-F09 (docs) — machine_translation_alignment: stale numbers, vanished EM log, visible new warning
Fresh NLLs are much better (sparse valid/test 73.5/73.9 bits vs stored 103.5/101.9; hidden 79.8/77.8 vs 92.5/92.1), alignment accuracy 0.70 -> 1.00 and 0.60 -> 0.80, retrieval MRR 0.41 -> 0.49 / 0.32 -> 0.37; the prose ranking (lexical aligner beats the HMM-style one) still holds. Two visible differences: the stored `Iteration N: ln[p_mat(Data|Model)]=...` lines no longer print because `optimize(print_iter=1)` is silent unless `out=` is passed (documented in the docstring; probe: `optimize(..., print_iter=1)` prints nothing), and cells 8/16 now show the "stopped at the max_its cap (40)" UserWarning inline (this notebook does not suppress warnings).

### P07-F10 (docs) — parallel_scaling's "model too big for 1 worker" case now plans data-parallel with rationale "model fits"
Stored: `1x8 model-parallel x8`. Fresh: `8x1 data-parallel: model fits and N=250 fills 8/8 workers`; other rows gain `[MEMORY FIT UNKNOWN: at least one assigned device has no memory evidence]`; backend list gained `component_parallel`. The case label and the planner disagree.

### P07-F11 (minor) — architecture-study timing claims are stated as facts and are far from what the candidate produces
Uncontended solo re-run (`executed_solo/`): ppl_scaling `Gaussian MLE N=200000: mixle 92 ms vs torch-Adam 378 ms -> 4x` (stored 19 ms vs 1422 ms -> 76x; "median speedup 72x" -> 5x); Poisson-Gamma exact-vs-MCMC 1614x -> 490x; import_and_warmup `cold numba cache 1.78 s / warm 2.84 s / cache saves ~0.6x` (stored 12.95 s / 1.55 s / 8.3x), steady-state per step 0.062 s -> 0.233 s; engine_benchmarks' "numba" column is 7-15x slower than numpy; parallel_scaling thread table 34.7 ms -> 1.1 ms per E-step. torch got ~3.8x faster than the stored run on this host while mixle's own path got ~4x slower, so this is not host speed. Root cause is F15 (no numba in the venv); the notebooks print "numba" timings without checking numba is present.

### P07-F15 (real) — NumbaKernelFactory silently runs without numba; the review venv has no numba at all
`python -c "import numba"` in the candidate venv -> ModuleNotFoundError (the wheel declares `Provides-Extra: numba`). Yet `NumbaKernelFactory().build(model, NUMPY_ENGINE, estimator=...)` returns a `GeneratedNumbaKernel`, encodes and runs, with zero warnings (probe recorded above). engine_benchmarks therefore prints a "numba" column that is 7-15x slower than numpy (`N=200000 numpy 0.0249 numba 0.3833`), and import_and_warmup prints cold/warm "numba cache" timings for a cache that cannot exist. Two consequences: an ordinary `pip install mixle` user who asks for numba kernels gets a silently slow interpreted path labelled numba; and every performance number gathered by this review cycle on this venv is the numpy path — the architecture-study prose (76x, 8.3x) is unverified on the candidate.

### P07-F12 (docs) — fake_news topic-model conclusion is seed-fragile
Cell 11 (6-component mixture of sequences, 40 its) prints "most fake-skewed topic fake-fraction 0.51; most real-skewed 0.24 -- themes carry signal beyond individual words" (stored 0.56 / 0.20). Seeds 0 and 3 give spreads of 0.11 and 0.10 (topic fake-fractions 0.37-0.48), seeds 1 and 2 give 0.25-0.27; all report converged=True. (`atk_fn.py`)

### P07-F13 (minor) — MarkovChainEstimator(pseudo_count=0.1) scores unseen symbols at -inf
`optimize([['a','b','a','b'],['b','a','c']], MarkovChainEstimator(pseudo_count=0.1))` then scoring `['a','z']` gives -inf. On the behavioral notebook trained on 10% of users, "clean" held-out users score `inf` per-event surprise with no warning (the AUC still computes). Not reached on the notebook's own path.

### P07-F14 (docs) — numeric drift with conclusions intact
geochemical_fingerprinting RAW-model numbers (acc 0.909 -> 0.879, "Brier 2.39x worse" -> 3.40x, Galapagos 0.71 -> 0.43) on deterministic data; behavioral mixture-of-sequences surprise 2.95 -> 3.64 (now worse than the bag's 3.22) and impersonate-by-mixture AUC 0.669 -> 0.616; geospatial BIC values shift by up to 4400 (still monotone in K); knowledge-graph conformal set sizes; kriging range 0.78 -> 0.79 kft; near_surface joint velocity RMSE 0.533 -> 0.658 and real-data cross-gradient 0.719 -> 0.057 (mixle-pde dev0); radar corr 0.928 -> 0.939; tweet CI 29.8 -> 29.7. Every prose conclusion in these cells survives.

## Attacks that did not break anything

- No notebook crashed: 36/36 exit 0, zero error-type outputs in the executed copies; the only stderr content is matplotlib's non-interactive `plt.show()` warning and the new `optimize()` max_its-cap warnings.
- Dataset loaders: all 15 CSVs load with the expected dtypes, no header-as-data, no NaN introduced by parsing; `prices.csv` index is monotone datetime with no duplicates; `transactions_top150.csv` parses `invoice_date`; the tweets file's 3 NaN rows are dropped by the notebook; `assay_BABBITT.csv` has NaN grades in the raw file but the kriging notebook drops them before `hole_grade` (verified 0 of 375 holes biased); hugoton PE NaNs are handled by `_geodata.load_hugoton`; `manifest.json`/`option_chain.csv`/UMLS/translation/fake-news files intact.
- Truncation to 10%: radon mixed model on the first 91 rows (8 counties) gives tau 0.351 / sigma 0.648 / floor -0.536 (full: 0.312 / 0.725 / -0.663), converged; behavioral first-order Markov on 30 training users still detects impersonation (AUC 0.889); HMM on 30 users AUC 0.785; LocalLevel on 8 points does not crash.
- Seed perturbation: geospatial "BIC keeps improving toward K=40" holds on the fresh run; geochemical LOG=CLR conclusion holds for all four representations; DP mixture (F07), HMM (F01), 2-regime mixture (F02) and topic mixture (F12) are the ones that did not survive.
- Geoscience notebooks (00_foundations, basin_thermal_history, biostratigraphy, crosshole_gpr, detrital_zircon, magma_reservoir_gravity, well_log_facies_hmm, where_to_drill) and applications oil_exploration_decision, radon_mixed_effects, social_fake_news_detection, generative_title_decoding, flow_inversion, synthetic_aperture_sonar, radar_tomography, seismic_full_waveform_inversion: printed numbers agree with the stored outputs to print precision or drift without touching a claim (well_log facies: HMM 88 switches vs memoryless 189, ECE 0.450 vs 0.128, unchanged; zircon KS/permutation p-values unchanged to 3 decimals; where_to_drill EVOI numbers unchanged).
- The 0.8.1-specific change (log-series CDF/quantile/entropy) is not exercised by any notebook in scope.

## Summary
- Counts: 2 blocking (P07-F01 HMM plateau, P07-F02 mixture collapse), 5 real (F03 silent constant-parameter fit, F04 grouped MCMC removed, F05 LocalLevel silent cap, F06 KG log_density semantics, F15 numba-less NumbaKernelFactory), 2 minor (F11, F13), 6 docs (F07, F08, F09, F10, F12, F14). Total 15.
- Worst: P07-F01 — the candidate's HMM EM sits on an initialization plateau for ~35 of the shipped notebook's 40 iterations, so a detector that had AUC 0.84 in June now scores 0.47 while the notebook prints the old conclusion; every seed fails at the notebook's budget.
- Two shipped notebooks (behavioral_anomaly_detection, stock_portfolio_optimization) print wrong conclusions on their own path; four more print misleading numbers (tweet rhythm, option pricing, retail, knowledge graph).
- The review venv lacks the `numba` extra, so all performance evidence from this pass (and, presumably, the other passes) is on the numpy path; the architecture-study notebooks do not detect this.
- No crashes, no data-loader misreads, geoscience corpus numerically stable.
