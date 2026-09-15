# Pass 07 -- application, exploration-geoscience and architecture-study notebooks (44) + flagship/demo examples

Recovery report: written from the preserved evidence of the pass-07 reviewer (66 probe scripts with captured outputs, 44 executed notebooks, 6 executed examples, 4 regression-test files run in two environments, `findings_draft.md`, `RECOVERED_NOTES.md`), plus two recovery additions: `probes/r01_confirm_delta_dump.py` (one bounded run) and a read-only stored-vs-fresh sweep of all 44 notebooks (`logs/cmp_all/`). No notebook or example was re-executed.

- Wheel: `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (sdist `mixle-0.8.2.tar.gz` sha256 `71ec826b2960df13f9ca4513da83eac371e25ad942a479378e47d93f1a39c037`)
- Commit `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013` (branch `release/0.8.2`)
- Work dir: `REVIEW_ROOT/pass-07/` (paths below are relative to it)
- Verification command: `cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, run for every environment used (saved in `logs/verify_env_*.txt`):

`venv-full (0.8.2, corpus + most probes)`:

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-nonumba (0.8.2, p05/p12)`:

```
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-base (0.8.2, p05/p17 and the test copies)`:

```
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`candidate-081 (published 0.8.1, before/after only: p03b, p06, p08b, p09b, p19)`:

```
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

## Corpus executed on the candidate

Notebooks: all 44 assigned notebooks executed in place in an APFS clone of the corpus (`corpus/`, mixle-notebooks d99d296) with `REVIEW_ROOT/tools/run_nb.sh REVIEW_ROOT/venv-full/bin/python <nb> [1800|5400]`, 3 at a time (`run_corpus.py`, tally `logs/nb_tally.txt`, 2026-09-11 11:07-12:21). Wall times are under contention (three notebooks plus probes and the background full run) and are not performance evidence. The 'diff cells' column is the number of code cells whose text output differs from the stored corpus copy (`nbcompare.py`, outputs in `logs/cmp_all/`), ignoring figure reprs and memory addresses.

| notebook | exit | wall (s) | diff cells | fresh vs stored |
|---|---|---|---|---|
| applications/malware_certificate_embedding.ipynb | 0 | 4452 | 0 | identical text outputs |
| applications/baseball_hierarchical_shrinkage.ipynb | 0 | 26 | 1 | 1 cell: unseeded how='mcmc' acceptance rate 0.24 -> 0.22 (Q07-F12 class); numbers otherwise identical |
| applications/bayesian_tweet_rhythm.ipynb | 0 | 32 | 3 | 3 cells: unseeded posterior draws (CI 22.7 -> 22.8, GP peak 23.3 -> 23.4, mean diff 4.2 -> 4.3) -- Q07-F12 |
| applications/behavioral_anomaly_detection.ipynb | 0 | 56 | 0 | identical text outputs |
| applications/fake_news_detection.ipynb | 0 | 43 | 0 | identical text outputs |
| applications/flow_inversion.ipynb | 0 | 2054 | 0 | identical text outputs |
| applications/generative_title_decoding.ipynb | 0 | 755 | 3 | 3 cells: 'fit time' / 'seconds' columns only |
| applications/geospatial_tweet_clusters.ipynb | 0 | 1504 | 0 | identical text outputs |
| applications/knowledge_graph_umls.ipynb | 0 | 276 | 0 | identical text outputs |
| applications/machine_translation_alignment.ipynb | 0 | 63 | 4 | 4 cells: fresh copy prints the print_iter=1 traces the stored copy lacks (P08-F04 repair); every NLL number identical |
| applications/magma_reservoir_gravity_inversion.ipynb | 0 | 10 | 0 | identical text outputs |
| applications/oil_exploration_decision.ipynb | 0 | 48 | 1 | 1 cell: wall-clock line only |
| applications/option_pricing_and_implied_volatility.ipynb | 0 | 39 | 0 | identical text outputs |
| applications/radar_tomography.ipynb | 0 | 238 | 1 | 1 cell: wall-clock line only (corr 0.939, contrast 0.51 identical) |
| applications/radon_mixed_effects.ipynb | 0 | 15 | 0 | identical text outputs |
| applications/retail_trend_statespace.ipynb | 0 | 53 | 0 | identical text outputs |
| applications/seismic_full_waveform_inversion.ipynb | 0 | 464 | 2 | 2 cells: wall-clock lines only (SNR 3.7 dB, 4032 data identical) |
| applications/social_fake_news_detection.ipynb | 0 | 10 | 0 | identical text outputs |
| applications/stock_portfolio_optimization.ipynb | 0 | 7 | 0 | identical text outputs |
| applications/synthetic_aperture_sonar.ipynb | 0 | 162 | 1 | 1 cell: wall-clock line only (corr 0.943 identical) |
| applications/pillar_validation/biodiversity_N.ipynb | 0 | 5 | 0 | identical text outputs |
| applications/pillar_validation/calibration_A.ipynb | 0 | 6 | 0 | identical text outputs |
| applications/pillar_validation/climate_L.ipynb | 0 | 6 | 0 | identical text outputs |
| applications/pillar_validation/economics_J.ipynb | 0 | 5 | 0 | identical text outputs |
| applications/pillar_validation/health_K.ipynb | 0 | 6 | 0 | identical text outputs |
| applications/pillar_validation/production_H.ipynb | 0 | 5 | 0 | identical text outputs |
| applications/pillar_validation/simulation_P.ipynb | 0 | 6 | 0 | identical text outputs |
| exploration_geoscience/00_foundations.ipynb | 0 | 2 | 0 | identical text outputs |
| exploration_geoscience/basin_thermal_history.ipynb | 0 | 7 | 0 | identical text outputs |
| exploration_geoscience/biostratigraphy_event_ordering.ipynb | 0 | 91 | 0 | identical text outputs |
| exploration_geoscience/crosshole_gpr_tomography.ipynb | 0 | 6 | 0 | identical text outputs |
| exploration_geoscience/detrital_zircon_provenance.ipynb | 0 | 20 | 0 | identical text outputs |
| exploration_geoscience/geochemical_fingerprinting.ipynb | 0 | 5 | 0 | identical text outputs |
| exploration_geoscience/magma_reservoir_gravity.ipynb | 0 | 7 | 0 | identical text outputs |
| exploration_geoscience/mineral_exploration_kriging.ipynb | 0 | 9 | 0 | identical text outputs |
| exploration_geoscience/near_surface_joint_inversion.ipynb | 0 | 40 | 0 | identical text outputs |
| exploration_geoscience/well_log_facies_hmm.ipynb | 0 | 6 | 0 | identical text outputs |
| exploration_geoscience/where_to_drill_value_of_information.ipynb | 0 | 67 | 0 | identical text outputs |
| architecture_studies/data_scaling.ipynb | 0 | 6 | 2 | 2 cells: timing tables only |
| architecture_studies/engine_benchmarks.ipynb | 0 | 8 | 3 | 3 cells: timing tables only (Q07-F14 concerns its numba branch on other installs) |
| architecture_studies/import_and_warmup.ipynb | 0 | 32 | 4 | 4 cells: timing lines only |
| architecture_studies/model_scaling.ipynb | 0 | 7 | 3 | 3 cells: timing tables only |
| architecture_studies/parallel_scaling.ipynb | 0 | 10 | 5 | 5 cells: timing lines and a host-dependent worker count only |
| architecture_studies/ppl_scaling_vs_pyro_stan.ipynb | 0 | 17 | 3 | 3 cells: timing lines only (fitted mu/means identical) |

44 of 44 exit 0; no executed cell errored. Contradictions of a stored number or a prose claim:

- `applications/pillar_validation/climate_L.ipynb`, `economics_J.ipynb`: stored and fresh both print `using landed ...: False` before their PASS verdicts; the named modules are absent from the 0.8.2 wheel, so the PASS validates the notebooks' inline reference code (Q07-F15, docs).
- `applications/baseball_hierarchical_shrinkage.ipynb` cell 12: the fallback prose 'a longer run or more data is needed' is not supported by longer runs or more data (Q07-F16, docs).
- `applications/bayesian_tweet_rhythm.ipynb` cells 3/5/19 and `baseball` cell 10: printed numbers drift between runs because the posterior draws / the `how='mcmc'` call are unseeded (Q07-F12, minor); no claim changes.
- `applications/machine_translation_alignment.ipynb` cells 7/8/15/16: the stored copy pre-dates the P08-F04 repair (`print_iter` now prints without `out=`); the fresh copy adds the iteration traces, every NLL number is identical.
- `applications/behavioral_anomaly_detection.ipynb`: fresh run identical to the stored its=500 outputs (P07-F01 notebook-side repair holds); at the old its=40 budget the categorical HMM is still on the plateau (Q07-F05, docs, migration-guide wording only).
- `applications/knowledge_graph_umls.ipynb`: identical outputs; the joint-vs-conditional relationship the P07-F06 repair relies on holds exactly (`log_density - tail_log_posterior = -log nE - log nR`, `probes/p09_kg.full.out.txt`).
- The six `architecture_studies` notebooks: only timing lines differ (P07-F11 repaired the prose claims at d99d296; wall times here are not evidence either way).
- Every other notebook: identical text outputs.

Examples: the six assigned scripts, run sequentially with `PYT_CWD=examples_cwd REVIEW_ROOT/tools/pyt.py 1800 REVIEW_ROOT/venv-full/bin/python REVIEW_ROOT/source/examples/<name>.py` (`run_examples.sh`; outputs in `examples_logs/`, tally `logs/examples_tally.txt`).

| script | exit | wall (s) |
|---|---|---|
| examples/flagship_kg_agent.py | 0 | 98.8 |
| examples/flagship_triage_app.py | 0 | 125.6 |
| examples/frontier_ecosystem_demo.py | 0 | 175.1 |
| examples/reasoner_investigation_demo.py | 0 | 115.3 |
| examples/multimodal_stage1_demo.py | 0 | 274.8 |
| examples/vlm_trust_receipts_demo.py | 0 | 110.3 |

6 of 6 exit 0; each script's own self-check line prints (`OK: ...` / `no fact without a schema ...` / `journal.verify(): True`). Three of them warn that their own `optimize(max_its=...)` budget stopped before convergence (flagship_kg_agent max_its=40, frontier_ecosystem_demo max_its=10, multimodal_stage1_demo max_its=6) -- the scripts' choice, disclosed by the 0.8.2 cap warning, not a defect.

Regression tests copied to `tests_copy/` and run with `-p no:randomly -q -m "" -n 0` in venv-full and venv-base (`run_tests.sh`, `logs/tests_tally.txt`): notebook_surface_repairs_test 26 passed (full) / 2 failed + 20 passed + 4 skipped (base; Q07-F18); mixture_kmeans_lloyd_init_test 2 passed + 18 subtests (both); latent_initialization_symmetry_test 12 passed (both); component_family_identifiability_test 9 passed (both).

## Findings

Counts: blocking 0, real 7, minor 10, docs 3 (20 total). Ids renumber the draft's findings in draft order (F-A02 -> Q07-F01 ... F-KEMENY -> Q07-F17); Q07-F18..F20 are written up from probe outputs the draft did not reach (`logs/tests_tally.txt`, `probes/p21_weights.*`, `probes/p22_stock_surfaces.*`). Q07-F08 is downgraded from the draft's `real` to `minor`.

### Q07-F01 (real) -- A-02 under-supported-component note fires on the initialization subsample and describes the returned fit; the MVN free-parameter count is dim-only

- Surface: mixle.inference.optimize(rows, MixtureEstimator([MultivariateGaussianEstimator(dim=6)]*2)) on the stock_portfolio_optimization two-regime panel; mixle.stats.latent.mixture._disclosing_component_support; mixle.inference.structure._num_free_params
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p07_mixture_kmeans.py
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p07b_a02_trace.py
```

- Observed: On the notebook's 1500x6 panel (its=80), seeds 0, 3 and 4 of 8 emit the UserWarning 'this mixture fit left 1 of 2 component(s) with less data than they have parameters: component 1 (12 free parameter(s)) -- see component_row_mass ...' while the returned model's component_row_mass is (1181.53, 318.47), i.e. both components carry hundreds of rows. Tracing the hook shows the only estimate() call with a component under 12 rows is call 0, the init-subsample fit: masses (137.93, 7.07), (128.94, 9.06), (164.92, 11.08) for seeds 0/3/4; seed 2, whose subsample split (80, 74), emits nothing although its final fit is the same. _num_free_params(MultivariateGaussianDistribution(zeros(6), eye(6))) = 12 and dim=2 gives 4 (mean + variance per axis), not d + d(d+1)/2 = 27 and 5.
- Expected: The note describes the fit that is returned (migration guide 0.8.2.md line 74-75: 'mixture fits warn when a component ends with fewer effective rows than parameters'), is computed from the final component masses, and counts the covariance parameters of a multivariate Gaussian.
- Notes: Mechanism: the disclosure runs inside every MixtureEstimator.estimate() call, including the initialization-subsample one, and its text speaks of 'this mixture fit'; the final masses are never below 12 in any of the 8 seeds. component_row_mass survives pickle and is dropped by JSON (documented). Evidence: probes/p07_mixture_kmeans.full.out.txt, probes/p07b_a02_trace.full.out.txt. Repairs concerned: A-02, P07-F02 (the stock panel itself converges to the same (0.212, 0.788) split on all 8 seeds and on the t(3) panel -- that part of the repair holds).
- Repairs attacked: A-02, P07-F02

### Q07-F02 (real) -- ppl Mix re-estimates constant components when only the weights are free; no spelling fits the weights alone

- Surface: mixle.ppl.Mix([<constant components>]).fit(data) (mixle/ppl/distributions.py:376); lowering in mixle/ppl/core.py:3453-3461 and mixle/ppl/_lowering.py:_mix_est
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p08b_mix_constants.py
$R/tools/pyt.py 600 <review-store>/candidate-081/venv/bin/python $R/pass-07/probes/p08b_mix_constants.py   # 0.8.1 comparison
# partial-free and docstring cases (exits 1 at the partial-free NotImplementedError by design):
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p08_mix_constants.py
```

- Observed: Mix([Normal(-3.0, 1.0), Normal(4.0, 2.0)]).fit(x) on 300 N(-3,1) + 700 N(4,2) draws returns MixtureDistribution([Gaussian(3.87, var 3.75), Gaussian(-2.95, var 1.05)], w=(0.696, 0.304)): the constants are replaced by re-estimated values (and re-ordered by the k-means start), with no warning; explain_fit() says 'all-free parameters, no priors -> maximum-likelihood EM'. Mix([Poisson(2.0), Poisson(20.0)]) likewise returns (2.07, 19.90). The all-constant spelling Mix([...], [0.3, 0.7]) warns 'Mixture.fit() had nothing to estimate' and returns the input. Mix([Normal(free, 1.0), Normal(free, 2.0)]) raises NotImplementedError 'partial `free` (some args fixed) is a later slice; use all-free or all-fixed for now', and the all-fixed spelling it recommends is the one that refits. The docstring's other route, Mix([GaussianDistribution(-3,1), GaussianDistribution(4,4)]) ('over PPL variables or concrete distributions'), raises ValueError 'a bound RandomVariable has no estimator to lower to' on fit. Identical on 0.8.1.
- Expected: Constant components are held fixed and only the free weights are estimated, or the call refuses and names the limitation; the docstring's 'concrete distributions' spelling either fits or is not advertised.
- Notes: Mechanism: for a child with no `free` slot, lower(target='estimator') returns lower(rv, 'dist').estimator(), which is the family's fully free estimator; _mix_est then builds MixtureEstimator(estimators, fixed_weights=weights), so the weights are the only thing that can be pinned. The P07-F03 'nothing to estimate' warning covers only the all-constant spelling. Evidence: probes/p08_mix_constants.full.out.txt, probes/p08b_mix_constants.full.out.txt, probes/p08b_mix_constants.081.out.txt.
- Repairs attacked: P07-F03

### Q07-F03 (real) -- KnowledgeGraphDistribution has no working JSON persistence route; the free encoders and dump_models(verify=False) emit a write-only payload

- Surface: KnowledgeGraphDistribution.to_json/to_dict/from_dict; mixle.utils.serialization.to_serializable/from_serializable/to_json; mixle.stats.dump_models/load_models; KnowledgeGraphEnsemble; manifests/serialization_schema_manifest.json (knowledge_graph_umls notebook, 300-epoch fits)
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p09b_kg_serial.py
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p09c_kg_dict.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/r01_confirm_delta_dump.py
# 0.8.1 comparison:
$R/tools/pyt.py 900 <review-store>/candidate-081/venv/bin/python $R/pass-07/probes/p09b_kg_serial.py
```

- Observed: kg.to_json() raises SerializationError 'to_json produced JSON that from_json cannot read back (SerializationError: registered class ...KnowledgeGraphDistribution requires a class-owned __pysp_setstate__ hook; constructor fields are absent: entity_embeddings, relation_embeddings). Refusing to return a write-only serialization'; mixle.stats.dump_models([kg]) refuses with the same text. kg.to_dict() (state keys {'entity', 'relation'}), to_serializable(kg), the free to_json(kg) (1450 chars) and dump_models([kg], verify=False) (2301 chars) all return silently, and from_dict / from_serializable / load_models on those payloads raise the same SerializationError. KnowledgeGraphDistribution.__init__ takes entity_embeddings/relation_embeddings while the instance stores entity/relation. The serialization manifest lists the class (base and full profiles) as codec 'constructor-validated', stability 'provisional'. KnowledgeGraphEnsemble has no to_json. pickle round-trips (log_density equal). Identical on 0.8.1.
- Expected: A registered class round-trips through every JSON route, or every route refuses; the manifest's codec claim is true.
- Notes: Not a 0.8.2 regression. The UMLS notebook fits for 300 epochs and cannot save the result through the library's JSON surface. probes/p09b_kg_serial.py imported dump_models from the wrong module; r01_confirm_delta_dump.py re-ran that route from mixle.stats. Evidence: probes/p09_kg.full.out.txt, probes/p09b_kg_serial.full.out.txt, probes/p09b_kg_serial.081.out.txt, probes/p09c_kg_dict.full.out.txt, probes/r01_confirm_delta_dump.full.out.txt.
- Repairs attacked: none

### Q07-F04 (real) -- LocalLevel/AR1 EM floors the variances at 1e-8 absolute: series at scale <= 1e-5 return level_sd = obs_sd = 1e-4 with converged=True and no disclosure

- Surface: mixle.ppl.LocalLevel().fit, mixle.ppl.AR1().fit (mixle/ppl/statespace.py:322-324); retail_trend_statespace notebook
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p03b_locallevel_floor.py
$R/tools/pyt.py 900 <review-store>/candidate-081/venv/bin/python $R/pass-07/probes/p03b_locallevel_floor.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p03_locallevel_cap.py   # rows 'single point', 'constant series', 'tiny scale 1e-12'
```

- Observed: A 739-point notebook-shaped series (level sd 3.36, observation sd 22.8) rescaled: at scales 1, 1e-2, 1e-3, 1e-4 the fit returns the scale-1 answer times the scale (level_sd 3.228*scale, obs_sd 22.83*scale). At 1e-5: level_sd 1e-4 = 2.98x the true value; at 1e-6: level_sd 1e-4, obs_sd 1e-4 (29.8x and 4.39x); at 1e-8: 2980x and 439x. Every floored fit reports converged=True, termination_reason='objective_tolerance', certificate None; the fitted RV exposes no numerical_repairs()/fit_provenance(), and the result has no attribute containing 'floor' or 'repair'. A constant series [5.0]*50 and a single point likewise return sd 1e-4 for both. Same numbers on 0.8.1.
- Expected: A floor relative to the data scale, or the applied floor disclosed the way 0.8.2 discloses mixture/HMM variance floors (migration guide line 73: 'Variance floors applied inside mixtures and HMMs are disclosed in fit_provenance().repairs').
- Notes: Mechanism: statespace.py:322 q = max(..., 1e-8), :323 r = max(..., 1e-8), :324 P0 = max(..., 1e-8) -- absolute floors on the variances, i.e. sd 1e-4. Not a regression; borders on blocking for a user whose series is measured in units with noise sd below ~2e-4 (the fit is silently wrong by orders of magnitude and certified converged). Evidence: probes/p03b_locallevel_floor.full.out.txt, probes/p03b_locallevel_floor.081.out.txt, probes/p03_locallevel_cap.full.out.txt.
- Repairs attacked: P07-F05

### Q07-F05 (docs) -- Migration guide states the symmetry-broken HMM start without the CHANGELOG's scope caveat; categorical-emission HMMs still sit on the initialization plateau

- Surface: docs/migrations/0.8.2.md line 74; CHANGELOG.md 0.8.2 'Latent-model initialization' paragraph (A-01, R05-F08); behavioral_anomaly_detection.ipynb HMM cell (HiddenMarkovEstimator([CategoricalEstimator(pseudo_count=0.5)]*6))
- Reproduction:

```sh
R=<review-root>; cd /tmp
# cwd must be the behavioral notebook's directory in the pass-07 corpus clone (relative data paths); argv[1] is the notebook's cell dump
PYT_CWD=$R/pass-07/corpus/notebooks/applications $R/tools/pyt.py 1800 $R/venv-full/bin/python $R/pass-07/probes/p06_hmm_plateau.py $R/pass-07/probes/nb_behavioral.py
PYT_CWD=$R/pass-07/corpus/notebooks/applications $R/tools/pyt.py 1800 <review-store>/candidate-081/venv/bin/python $R/pass-07/probes/p06_hmm_plateau.py $R/pass-07/probes/nb_behavioral.py
```

- Observed: 0.8.2, its=40 (the 0.8.1 notebook budget), seeds 6/1/2: impersonation AUC 0.468 / 0.463 / 0.662, objective -46501.699 / -46581.514 / -43319.908, converged=False. 0.8.1, seed 6: AUC 0.468, objective -46501.699 (equal to 10 significant digits; the 0.8.1 run stops after seed 6 because 0.8.1 refuses print_iter=None). At the shipped its=500 the AUC is 0.865 / 0.813. The migration guide says only 'The hidden-Markov families start from a symmetry-broken initialization, and mixture fits warn when a component ends with fewer effective rows than parameters.' The CHANGELOG says the k-means++ start 'declines' for categorical emissions, which 'stay bit-identical to 0.8.1 -- including the plateau, where they had one (R05-F08)'.
- Expected: The migration guide carries the scope (Gaussian and vector-valued emissions; Gamma/Weibull/Beta/Poisson/categorical emissions unchanged), since it is the user-facing list of what changed.
- Notes: The wheel matches the CHANGELOG; only the migration guide overstates. The second half of the same sentence is contradicted by Q07-F01. Evidence: probes/p06_hmm_plateau.full.out.txt, probes/p06_hmm_plateau.081.out.txt; logs/cmp_all/applications_behavioral_anomaly_detection.txt (fresh run identical to the stored its=500 outputs).
- Repairs attacked: A-01, P07-F01, R05-F08

### Q07-F06 (real) -- Grouped Bernoulli-Beta how='ensemble' returns per-player rates many posterior sd from the conjugate answer at ordinary budgets, with no diagnostic flag

- Surface: mixle.ppl.Bernoulli(Beta(a, b).each()).fit(games, how='ensemble', draws=1000, burn=500) and the baseball notebook's spelling (max_its=300); summary() diagnostics
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p02c_ensemble_rates.py
$R/tools/pyt.py 1800 $R/venv-full/bin/python $R/pass-07/probes/p02b_grouped_routes.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p02_grouped_bernoulli.py
```

- Observed: 18 players x 45 games, prior Beta(164.0, 453.9), exact posterior means (a+h)/(a+b+45) known. how='ensemble', draws=1000, burn=500, seeds 0/1/2: 8 / 11 / 7 of the 18 rates lie more than 3 posterior sd from the exact value (worst z 8.8, 13.4, 6.0; e.g. p[2] mean 0.503 vs exact 0.272); the notebook spelling max_its=300 gives max|z| 7.8; p[10]'s 95% interval [0.266, 0.474] excludes the exact 0.2625; mcse 0.0074 against an actual error of 0.102; split_r_hat is NaN (single chain), acceptance 0.35, no warning. draws=6000/burn=3000 gives max|z| 0.2. how='mcmc'/'nuts'/'hmc' at draws=1000/burn=500 give max|z| 0.4 / ~0 / ~0.
- Expected: Draws that are converged at the budget, or a warning that the walkers have not mixed with honest ess/mcse.
- Notes: The regression test GroupedBernoulliTest.test_the_ensemble_sampler_moves_and_converges_with_enough_sweeps runs this route at draws=6000/burn=3000 while the sibling samplers are tested at 1000/500 -- the test knows the budget the user is not told. The notebook itself uses how='mcmc', which is accurate at its budget (P07-F04 repair holds). Evidence: probes/p02_grouped_bernoulli.full.out.txt, probes/p02b_grouped_routes.full.out.txt, probes/p02c_ensemble_rates.full.out.txt.
- Repairs attacked: P07-F04

### Q07-F07 (real) -- MarkovChainEstimator refuses a corpus containing an empty sequence on a seed-dependent subset of runs; seq_log_density on an all-empty batch raises IndexError

- Surface: optimize(seqs, MarkovChainEstimator(pseudo_count=...)) (behavioral_anomaly_detection's Markov detector); MarkovChainDistribution.seq_log_density; mixle/stats/sequences/markov_chain.py:200
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p13c_markov_empty_seed.py
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p13b_markov_empty_fit.py
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p13_empty_sequences.py
```

- Observed: seqs = [['a','b','a','b'], ['b','a','c']] + [[]]: rng seeds 0, 1 and 3 of 0..9 raise ValueError 'MarkovChainEstimator.estimate states cannot be empty.'; the other seven fit. With init_p=1.0 no seed refuses; an 11-sequence corpus refuses on 1 seed of 10; with levels=['a','b','c'] declared it fits; SequenceEstimator (bag) and HiddenMarkovEstimator accept the same corpus on every seed. mc.log_density([]) returns 0.0 and a mixed batch [['a','b'], []] scores [-1.13, 0.0], but mc.seq_log_density(seq_encode([[]], model=mc)[0][1]) raises IndexError 'arrays used as indices must be of integer (or boolean) type'. An all-zero weight vector hits the same 'states cannot be empty' guard (probes/p21_weights.full.out.txt).
- Expected: An empty sequence contributes nothing on every seed (as the bag and HMM treat it), and an all-empty batch scores 0.0 like the mixed batch.
- Notes: Mechanism: optimize()'s initialization subsample (init_p=0.1 by default) can select only the empty sequence; the guard at markov_chain.py:200 then rejects a state set the library's own sampling produced (the fail-closed pattern named in the brief). Evidence: probes/p13_empty_sequences.full.out.txt, probes/p13b_markov_empty_fit.full.out.txt, probes/p13c_markov_empty_seed.full.out.txt. P07-F13 (the -inf/levels repair on the same estimator) holds: probes/p04_markov_unseen.full.out.txt.
- Repairs attacked: P07-F13

### Q07-F08 (minor) -- bootstrap() refuses a plain list of numbers with a message about 'data parts'

- Surface: mixle.inference.bootstrap (detrital_zircon_provenance and biostratigraphy_event_ordering pass arrays)
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p19_bootstrap_gp.py
$R/tools/pyt.py 600 <review-store>/candidate-081/venv/bin/python $R/pass-07/probes/p19_bootstrap_gp.py
```

- Observed: bootstrap([floats], statistic=np.mean, n_boot=100, method='percentile'), the tuple spelling, a list of np.float64 and a list of ints all raise ValueError 'each data part must have a non-empty observation axis'; np.array(a) and (np.array(a),) work. A statistic that returns NaN on list input is reported with the same message. Same on 0.8.1.
- Expected: A list of numbers accepted as one sample (every fit() in the library takes lists), or a message that says a list is read as a tuple of parts and to pass an array.
- Notes: Docstring: 'data: a single array (resampled along axis 0) or a tuple of arrays'. Downgraded from the draft's 'real': it is a refusal with a misleading message, not a wrong result. Evidence: probes/p19_bootstrap_gp.full.out.txt, probes/p19_bootstrap_gp.081.out.txt, probes/p15_geoscience_surfaces.full.out.txt.
- Repairs attacked: none

### Q07-F09 (real) -- GaussianProcessRegressor cannot be pickled and has no JSON route; the kriging notebooks' 400-iteration fit cannot be persisted

- Surface: mixle.models.GaussianProcessRegressor (mixle/models/gaussian_process.py:57 self.torch = torch); mineral_exploration_kriging.ipynb (gp.fit(xy, v, max_its=400)), where_to_drill_value_of_information.ipynb
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p19_bootstrap_gp.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p18_serialization_fitted.py
```

- Observed: pickle.dumps(gp) raises TypeError "cannot pickle 'module' object" for a fresh and for a fitted regressor; the object has no to_json/to_dict and is not in serializable_class_ids(); vars(gp) holds torch (module), engine (TorchEngine), the log-parameters as tensors. Same on 0.8.1.
- Expected: pickle works (drop the module attribute in __getstate__) or a documented persistence route exists.
- Notes: Not a regression. Every other fitted object the area's notebooks produce round-trips (HMM categorical and MVN, mixtures of sequences, DP-Beta, Markov chain, KG via pickle): probes/p18_serialization_fitted.full.out.txt. Evidence: probes/p19_bootstrap_gp.full.out.txt, probes/p19_bootstrap_gp.081.out.txt.
- Repairs attacked: none

### Q07-F10 (minor) -- NegativeBinomialEstimator caps r at 1e7 silently: numerical_repairs() and fit_provenance().repairs are empty and the docstring does not mention the cap

- Surface: mixle.ppl.NegativeBinomial(free, free).fit and mixle.stats.NegativeBinomialEstimator (mixle/stats/univariate/discrete/negative_binomial.py:45, :578, :590); bayesian_tweet_rhythm.ipynb 'fitted: Poisson(...) vs NegativeBinomial(r=..., p=...)'
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p11_nb_ceiling.py
```

- Observed: 164 Poisson(14.8) counts: NegativeBinomial(free, free).fit -> NegativeBinomialDistribution(10000000.0, 0.9999985), converged=True after 3 iterations, repairs=(), numerical_repairs() == (), no warning; the same through optimize(x, NegativeBinomialEstimator()). The log-likelihood equals the Poisson fit's to 1.5e-5. The estimator docstring has no line mentioning the ceiling.
- Expected: The cap disclosed in numerical_repairs()/fit_provenance().repairs and named in the docstring, as the 0.8.2 policy for clamps (P01-F03 class: WeibullEstimator's max_shape).
- Notes: _MAX_NB_SHAPE = 1.0e7 is returned when the bisection runs off to the Poisson limit. The answer is right in effect (NB -> Poisson); the disclosure is what is missing. Evidence: probes/p11_nb_ceiling.full.out.txt.
- Repairs attacked: none

### Q07-F11 (minor) -- LocalLevel().fit's delta contract differs from optimize(): delta=None raises a TypeError naming `tol`, delta=0 is accepted where optimize refuses it

- Surface: mixle.ppl.LocalLevel().fit(delta=...) / AR1 (mixle/ppl/statespace.py:293, :418); the state-space cap warning's advice
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p03_locallevel_cap.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/r01_confirm_delta_dump.py
```

- Observed: LocalLevel().fit(s, delta=None) -> TypeError 'tol must be a finite non-negative real number'; delta=-1 -> ValueError with the same 'tol' wording; delta=0 is accepted and runs to max_its (warning text 'last change 0.00426, delta=0'). optimize(x, GaussianEstimator(), delta=0) -> ValueError 'optimize(): delta must be None (a fixed iteration count) or a finite positive number, got 0'; optimize(delta=None) gives a fixed budget. The state-space cap warning says only 'Raise max_its to fit to convergence' while optimize's warning also offers 'pass delta=None to request a fixed iteration count'.
- Expected: One contract for delta across fit routes, and an error that names the argument the caller wrote.
- Notes: statespace.py:418 `tolerance = delta if tol is None else tol`, validated at :293 as 'tol'. Evidence: probes/p03_locallevel_cap.full.out.txt, probes/r01_confirm_delta_dump.full.out.txt.
- Repairs attacked: P07-F05

### Q07-F12 (minor) -- RandomVariable.posterior(name) draws are unseeded and accept no rng/n; the tweet notebook's printed intervals drift run to run

- Surface: fitted mixle.ppl RandomVariable.posterior (signature (x)); ConjugatePosterior.samples(param, n=4000, rng=None); bayesian_tweet_rhythm.ipynb cells 3, 5, 19
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p14_ppl_notebook_surfaces.py
$R/tools/pyt.py 300 $R/venv-full/bin/python $R/pass-07/probes/p20_misc_light.py
cd $R/pass-07 && $R/venv-full/bin/python nbcompare.py applications/bayesian_tweet_rhythm.ipynb
```

- Observed: m = Poisson(Gamma(2.0, 1.0, name='r')).fit([3,5,4,6]); m.posterior('r') twice -> arrays differ; m.posterior('r', rng=RandomState(0)) and m.posterior('r', n=10) -> TypeError (unexpected keyword); m.result.samples('r', rng=RandomState(0)) is reproducible and samples('r', n=7) works. Fresh vs stored notebook: 'CI [22.7 29.7]' -> '[22.8 29.7]', 'GP smooth diurnal intensity: peak 23.3/h' -> '23.4/h', 'posterior mean difference 4.2/h' -> '4.3/h'.
- Expected: posterior(name, n=..., rng=...) forwarding to samples(), so a seeded notebook prints reproducible intervals.
- Notes: Evidence: probes/p14_ppl_notebook_surfaces.full.out.txt, probes/p20_misc_light.full.out.txt, logs/cmp_tweet.txt, logs/cmp_all/applications_bayesian_tweet_rhythm.txt.
- Repairs attacked: none

### Q07-F13 (minor) -- Mixed model Normal(free*Field + free + Group, free): a NaN group label is accepted as a group named 'nan', a constant covariate raises a raw LinAlgError, and predict(covariates) refuses

- Surface: mixle.ppl regression with Group (mixle/ppl/regression.py:65-72); radon_mixed_effects.ipynb
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p14_ppl_notebook_surfaces.py
```

- Observed: given county[:-1] + [float('nan')] -> fits (floor=0.2701, intercept=1.01) with no error, while county[:-1] + [None] -> ValueError 'given[county] contains a missing or non-finite group label.'; floor all 0.0 -> numpy.linalg.LinAlgError: Singular matrix; mm.predict({'floor': [0.0, 1.0], 'county': ['0', '1']}) -> TypeError 'predict(<covariates>) is for a fitted regression; this model has no conditional fit. Pass a draw count instead.'
- Expected: NaN refused like None; a named rank-deficiency error for a constant covariate; prediction at new covariates or a message that says which call gives it.
- Notes: Mechanism: _group_vector does np.asarray(value) on a list of strings plus one float('nan'), which numpy coerces to a string array ('nan'), so the isinstance(float) test never sees the NaN. Evidence: probes/p14_ppl_notebook_surfaces.full.out.txt.
- Repairs attacked: none

### Q07-F14 (minor) -- engine_benchmarks cell 3 calls NumbaKernelFactory unguarded: on a numba-less install the cell raises although its torch branch degrades to NaN

- Surface: architecture_studies/engine_benchmarks.ipynb cell 3 (em_numba); mixle.stats.compute.kernel.NumbaKernelFactory.build
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-nonumba/bin/python $R/pass-07/probes/p12_engine_nonumba.py
$R/tools/pyt.py 600 $R/venv-base/bin/python $R/pass-07/probes/p05_numba_factory.py
```

- Observed: In venv-nonumba and venv-base, NumbaKernelFactory().build(m, NUMPY_ENGINE, estimator=est) raises KernelCapabilityDeclinedError 'numba is not installed, so a numba kernel cannot be built (install the extra: pip install mixle[numba]). Use GeneratedNumbaKernelFactory to fall back to the generic kernel instead of asking for numba by name.' -- the P07-F15 decline holds. The notebook's prose (cell 2) says 'If numba is not installed at all, mixle declines to build the kernel rather than reporting a numba timing it did not produce', but em_numba has no try/except while em_torch returns np.nan when torch is absent, so the cell fails on that install.
- Expected: The cell catches the decline (or uses GeneratedNumbaKernelFactory) and reports n/a for the numba column the way it does for torch.
- Notes: Notebooks repository. The library behaviour is correct; the corpus was executed in venv-full where the cell passes. Evidence: probes/p12_engine_nonumba.nonumba.out.txt, probes/p05_numba_factory.{full,nonumba,base}.out.txt.
- Repairs attacked: P07-F15, P07-F11

### Q07-F15 (docs) -- pillar_validation README's 'not yet landed' note is stale at 0.8.2, and the climate_L / economics_J PASS verdicts are earned by the notebooks' reference fallbacks

- Surface: notebooks/applications/pillar_validation/README.md (line 8 'A broken pillar fails the notebook, not just a library test', lines 25-31); climate_L.ipynb cells 1 and 13; economics_J.ipynb cells 1 and 15
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-base/bin/python $R/pass-07/probes/p17_base_env.py
$R/tools/pyt.py 300 $R/venv-full/bin/python $R/pass-07/probes/p20_misc_light.py
cd $R/pass-07 && $R/venv-full/bin/python nbcompare.py applications/pillar_validation/climate_L.ipynb && $R/venv-full/bin/python nbcompare.py applications/pillar_validation/economics_J.ipynb
```

- Observed: Stored and fresh copies both print 'using landed mixle_pde.climate_downscale / mixle.analysis.quantile_mapping: False' then 'PASS -- return level rel. error 0.0052 <= 0.05', and 'using landed mixle.analysis.objective: False' then 'PASS -- carbon price removes the highest-emission blocks'. On the 0.8.2 wheel `import mixle.analysis.quantile_mapping` -> ModuleNotFoundError and mixle.analysis.objective exports only priced_liabilities and hard_constraints (no risk_adjusted_plan), so both notebooks run their inline reference implementations; the README's list of modules that 'had not yet landed on release/0.8.0' also names mixle.analysis.sdm, mixle.analysis.health_risk, mixle.analysis.objective, mixle.reason.posterior_protocol and mixle.stochastic_opt, all of which import on 0.8.2.
- Expected: The README dated to the current line and naming exactly what still falls back on 0.8.2, and verdict lines that say 'PASS (reference implementation)' when HAVE_REAL_* is False, so that 'a broken pillar fails the notebook' is true.
- Notes: Notebooks repository. The fallback is by design and the notebook prose says so; the defect is the stale module list and the unqualified PASS for the two pillars whose modules are still absent. Evidence: probes/p17_base_env.base.out.txt, probes/p20_misc_light.full.out.txt, probes/p16_pillar_surfaces.full.out.txt (exit 1 at the quantile_mapping import), probes/p16b_pillar_surfaces.full.out.txt (exit 1 at risk_adjusted_plan), logs/cmp_all/applications_pillar_validation_{climate_L,economics_J}.txt.
- Repairs attacked: none

### Q07-F16 (docs) -- baseball notebook's DP fallback text blames budget and data; neither a longer run nor 10x the players recovers the two modes

- Surface: applications/baseball_hierarchical_shrinkage.ipynb cell 12 (DirichletProcessMixtureEstimator([BetaEstimator()]*6, prior=GammaDistribution(1.0, 4.0)))
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p10b_dp_budget.py
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p10_dp_beta.py
```

- Observed: Stored and fresh copies print 'at this budget (max_its=18) the DP mixture is still unimodal, so the two-mode structure (true modes ~.230, ~.310) is NOT recovered here -- a longer run or more data is needed, and the claim is not made'. Re-running the cell's model at its=60/100/200/400 over seeds 3, 0, 1, 2, 4 gives one mode near 0.245 or spurious extra modes at 0.32-0.39, never the 0.23/0.31 pair (e.g. seed 3 its=400: modes [0.241, 0.356, 0.383], converged=True); 1000 simulated players at its=200 give one mode at 0.238. The 80-at-bat binomial noise (sd ~0.05) exceeds half the 0.08 gap between the modes, so the observed-rate density is unimodal.
- Expected: Prose that says the simulated sample cannot show two modes at 80 at-bats (or a simulation with enough at-bats per player), rather than a budget/data remedy the numbers do not support.
- Notes: Notebooks repository. The DP fit itself and its boundary/NaN/support guards are fine (probes/p10_dp_beta.full.out.txt). Evidence: probes/p10b_dp_budget.full.out.txt.
- Repairs attacked: none

### Q07-F17 (minor) -- Ragged input reaches numpy unguarded: kemeny_consensus and LedoitWolfEstimator (through optimize) raise numpy's inhomogeneous-shape ValueError

- Surface: mixle.analysis.rank_aggregation.kemeny_consensus (rank_aggregation.py:92 _as_rankings; biostratigraphy_event_ordering.ipynb); mixle.analysis.LedoitWolfEstimator via optimize (stock_portfolio_optimization.ipynb)
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-07/probes/p15_geoscience_surfaces.py
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p22_stock_surfaces.py
```

- Observed: kemeny_consensus([[0, 1, 2], [0, 1]]) -> ValueError 'setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions...'; kendall_distance([0,1,2], [0,1]) on the same shape -> 'rankings must be permutations of the same 3 items; got a length-2 ranking'. optimize(rows + [[0.0]*5], LedoitWolfEstimator(dim=6)) -> the same numpy message, while a NaN row is refused by name ('multivariate Gaussian observations must contain only finite values').
- Expected: The named message on every route that takes rankings or rows.
- Notes: Evidence: probes/p15_geoscience_surfaces.full.out.txt, probes/p22_stock_surfaces.full.out.txt.
- Repairs attacked: none

### Q07-F18 (minor) -- notebook_surface_repairs_test.py's TorchModuleShapeTest errors instead of skipping in the numpy+scipy environment

- Surface: mixle/tests/notebook_surface_repairs_test.py (TorchModuleShapeTest, the P08-F17 guard, lines 315-339; the same file guards P07-F03/F04/F05/F13/F15); mixle/tests/conftest.py FILE_MARKERS
- Reproduction:

```sh
R=<review-root>; mkdir -p /tmp/q07_tests && cp $R/source/mixle/tests/notebook_surface_repairs_test.py /tmp/q07_tests/ && cd /tmp/q07_tests && unset PYTHONPATH && $R/venv-base/bin/python -m pytest -p no:randomly -q -m "" -n 0 notebook_surface_repairs_test.py
```

- Observed: venv-base: '2 failed, 20 passed, 4 skipped' -- both TorchModuleShapeTest methods fail with ModuleNotFoundError: No module named 'torch' (bare __import__('torch') at lines 319 and 339); venv-full: 26 passed. The same file guards RegressionSurfaceTest with @unittest.skipUnless(HAS_PANDAS, ...); 120 other test files use a HAS_TORCH guard; conftest.FILE_MARKERS carries no 'torch' marker for this file.
- Expected: A skipUnless(HAS_TORCH) guard or a 'torch' marker, so the suite is green in the environment `pip install mixle` gives.
- Notes: Test hygiene only; the P07 tests in the file pass in both environments (logs/tests_tally.txt). The other three files copied for this area (mixture_kmeans_lloyd_init_test, latent_initialization_symmetry_test, component_family_identifiability_test) pass in venv-full and venv-base. Evidence: logs/test_notebook_surface_repairs_test_venv-base.txt, logs/test_notebook_surface_repairs_test_venv-full.txt.
- Repairs attacked: P08-F17

### Q07-F19 (minor) -- Weight-vector validation differs by family on the accumulator route: negative weights are refused by mixture/HMM/Markov and silently accepted by Beta and Poisson; all-zero weights give a misleading ImpossibleEvidenceError, the empty-state refusal, or silent default models

- Surface: <Estimator>.accumulator_factory().make().seq_initialize/seq_update(enc, weights, ...) + <Estimator>.estimate(...) -- the route engine_benchmarks.ipynb cell 3 drives with k.accumulate(enc, np.ones(n)); MixtureEstimator/HiddenMarkovEstimator/MarkovChainEstimator/BetaEstimator/PoissonEstimator/DirichletProcessMixtureEstimator
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p21_weights.py
```

- Observed: A negative or NaN weight: MVN mixture and HMM -> ValueError '... initialization weights must be finite and non-negative'; Markov -> 'Markov weights must be a finite non-negative vector aligned with encoded rows.'; Beta with weights [1]*99 + [-1] -> BetaDistribution(5.57, 17.13) with no error; Poisson -> PoissonDistribution(5.03). All-zero weights: HMM -> ImpossibleEvidenceError 'zero-probability evidence at batch rows [0, ..., 39]' (the evidence is not impossible, the weights are zero); Markov -> 'states cannot be empty'; MVN mixture -> a default MixtureDistribution; Beta -> Beta(1.0, 1.0); DP -> Beta(1,1) components. Zero, huge (1e300) and integer-list weights behave on every family.
- Expected: The same finite-non-negative check on every accumulator and one message for an all-zero weight vector.
- Notes: The public optimize() takes no weights argument (probes/p10_dp_beta.full.out.txt: 'unexpected keyword argument weights'), so this route is the accumulator protocol itself. Evidence: probes/p21_weights.full.out.txt.
- Repairs attacked: none

### Q07-F20 (minor) -- mixle.inference.estimate([]) returns a default model from no data where optimize([]) refuses

- Surface: mixle.inference.estimate (re-exported from mixle.stats.compute.sequence; stock_portfolio_optimization.ipynb calls estimate(rows(...), MultivariateGaussianEstimator(dim=N)) throughout)
- Reproduction:

```sh
R=<review-root>; cd /tmp
$R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-07/probes/p22_stock_surfaces.py
```

- Observed: estimate([], MultivariateGaussianEstimator(dim=6)) -> MultivariateGaussianDistribution(mean zeros(6), covariance 1.000001*I) silently; optimize([], ...) -> ValueError 'optimize() received no observations: data is empty. Pass a non-empty data sequence.'
- Expected: The same refusal on the one-shot route.
- Notes: Evidence: probes/p22_stock_surfaces.full.out.txt (also: estimate(single row) returns a model; LedoitWolf refuses n<2 by name; the MVN ridge on identical rows / n<dim is disclosed in numerical_repairs()).
- Repairs attacked: none

## Attacks that did not break anything

Each line names its evidence file under `probes/` or `logs/`.

- P07-F03 'nothing to estimate' warning: fires on Poisson/NegativeBinomial/Normal/Gamma/StudentT/Bernoulli/all-constant Mix and Seq; `how='map'|'mcmc'|'vi'|'laplace'` on a constant model refuse by name; `weights=` on a fully specified model refused (p01_nothing_to_fit.full.out.txt).
- P07-F05 state-space cap warning: fires with the right numbers at max_its 1/100/5000, delta 1e-3; max_its 0/-1, empty, NaN (names missing='marginalize'), inf, (n,1) column, string all refused by name (p03_locallevel_cap.full.out.txt).
- P07-F13: MarkovChainEstimator docstring names the -inf and `levels`; `levels=` gives unseen-but-declared symbols mass; length-2 sequences over the declared vocabulary sum to 1.0; JSON/pickle round trips (p04_markov_unseen.full.out.txt).
- P07-F15: NumbaKernelFactory declines by name in venv-nonumba and venv-base, GeneratedNumbaKernelFactory falls back to GenericKernel, HAS_NUMBA flag correct in all three environments (p05_numba_factory.{full,nonumba,base}.out.txt).
- P07-F02 / A-02 on the stock panel: the (0.212, 0.788) two-regime split on all 8 seeds, and on the all-t(3) panel (0.143, 0.857) on all 8; init=/restarts= are not optimize() keywords; component_row_mass survives pickle and is dropped by JSON as documented (p07_mixture_kmeans.full.out.txt).
- A-01 for the shipped budget: behavioral HMM at its=500 reaches AUC 0.865/0.813 and the fresh notebook matches its stored outputs (p06_hmm_plateau.full.out.txt, logs/cmp_all/applications_behavioral_anomaly_detection.txt).
- KnowledgeGraphDistribution: log_density is the normalized joint (sums to 1 over all (h,r,t)); tail/head/relation posteriors normalize; out-of-range, negative, string, 2-tuple ids refused by name; empty triples, dim=0, epochs=0, lr<=0, negatives outside [1, nE) refused; seeded fits reproducible (p09_kg.full.out.txt).
- DP-Beta (baseball): boundary values 0/1, values outside [0,1], NaN refused by name; single/two/identical rows and k=1 fit; JSON round trip exact (p10_dp_beta.full.out.txt).
- Grouped Bernoulli-Beta how='mcmc'/'nuts'/'hmc' at draws=1000/burn=500 recover the conjugate posterior; empty player, zero players, non-binary and NaN values refused by name; ragged groups accepted; int seed / Generator seed accepted; unsupported grouped pair names both supported ones (p02_grouped_bernoulli.full.out.txt, p02b_grouped_routes.full.out.txt).
- Poisson-Gamma conjugate: empty, NaN, negative, fractional, huge counts handled by name; posterior mean exact (p14_ppl_notebook_surfaces.full.out.txt).
- StudentT(df, free, free) and StudentT(free, free, free) (warns that df is not fitted), df<=0 refused; AR1 on single/constant/random-walk series returns with the cap warning (p14_ppl_notebook_surfaces.full.out.txt).
- Empty sequences through SequenceEstimator (bag) and HiddenMarkovEstimator: fit, log_density 0.0, viterbi [] , latent_posterior [] (p13_empty_sequences.full.out.txt).
- Rank aggregation: empty, non-permutation, string ids refused by name; kendall_distance/borda_count/mallows_fit guards; 12-item Kemeny reports exact=False and its search mode (p15_geoscience_surfaces.full.out.txt).
- permutation_test / benjamini_hochberg / calibration metrics (ECE/MCE/Brier/reliability_curve/top_label_confidence): every degenerate spelling (empty, NaN, n_perm=0, alpha outside (0,1), bins=0, conf>1, rows not summing to 1, labels out of range, unknown strategy/alternative) refused by name (p15_geoscience_surfaces.full.out.txt).
- GaussianProcessRegressor constructor/fit/predict guards: noise<=0, lengthscale=0, unknown kernel, NaN input/target, mismatched lengths refused by name; duplicate/single/constant-target fits succeed with empty numerical_repairs(); alm_scores guards (p15_geoscience_surfaces.full.out.txt).
- Pillar surfaces: peaks_over_threshold/return_level (period, exceedance-count, NaN/inf, empty), DoseResponse (model, params, dose), calibrate_variance (target, zero/negative variance, zero residuals -> VarianceCalibrationUnidentifiable), fit_sdm (species id, grid bounds, ridge, cell_area), credible_interval/critical_habitat_mask levels, branch_and_bound_milp sense, two_stage_stochastic_plan k_scenarios/alpha, cvar_epigraph alpha, IPP rates/edges -- all refused by name (p16_pillar_surfaces.full.out.txt, p16b_pillar_surfaces.full.out.txt, p16c_pillar_surfaces.full.out.txt).
- Base environment (numpy+scipy): every module the area's notebooks import that exists in the wheel imports; grouped nuts/hmc and the GP refuse by naming the torch extra; mcmc/ensemble, LocalLevel, conjugate, StudentT, mixed model, AR1, KG, HMM, Markov, DP, MVN mixture, permutation_test, kemeny_consensus all run (p17_base_env.base.out.txt).
- Serialization: categorical HMM, hand-built MVN HMM (well-log construction), mixtures of sequences (behavioral, fake_news), DP-Beta, Markov chain round-trip through JSON and pickle with equal scores; ppl fitted RVs and their results pickle; a pickled grouped posterior's predict() refuses with a message that says why (p18_serialization_fitted.full.out.txt).
- Weights: zero, huge (1e300), integer-list weights behave on every family; negative/NaN weights refused by mixture/HMM/Markov (p21_weights.full.out.txt).
- Stock surfaces: LedoitWolfEstimator refuses n<2, identical rows (no dispersion), NaN rows, and overflow at 1e150 by name; its shrinkage of 1.0 on the iid isotropic test panel is the correct answer; MultivariateStudentTEstimator provenance converged with empty repairs; conjugate_posterior(mvn, rows) returns a NormalInverseWishartPosterior; the MVN ridge on identical rows / n<dim is disclosed in numerical_repairs() (p22_stock_surfaces.full.out.txt).
- The four area regression-test files pass in venv-full; three of four pass in venv-base (logs/tests_tally.txt).

## What was not covered

- The 0.8.1 side of the HMM-plateau comparison covers seed 6 only: 0.8.1 refuses `print_iter=None` (`optimize(): print_iter must be a non-negative integer, got None`), which the probe passes for the untraced seeds, so seeds 1/2 were not re-run there (p06_hmm_plateau.081.out.txt). Also observed on the way: 0.8.2 accepts `print_iter=None` -- consistent with the P08-F04 default.
- The inverse-problem notebooks (flow_inversion, radar_tomography, synthetic_aperture_sonar, seismic_full_waveform_inversion, magma_reservoir_gravity_inversion, geospatial_tweet_clusters, malware_certificate_embedding) were executed and compared but their mixle_pde / mixle_sim surfaces were not attacked directly -- out of this pass's library scope.
- The examples were executed and their self-checks read; their internal surfaces (knowledge substrate, reasoner, hvis, journal) were not attacked -- flagship examples are exercised more deeply by other passes.
- ppl fitted RandomVariables and their results have no to_json (only pickle; p18) -- observed, not written up, because no notebook in the area persists one and the brief's JSON routes are the stats-layer objects.
- two_stage_stochastic_plan accepting a NaN scenario silently and branch_and_bound_milp returning None on an infeasible/unbounded problem (p16c) were observed in the last probe wave and not pursued; the MVT single-row / identical-row 1e-12 covariance disclosure was not checked.
- P07-F06 and P07-F11 (notebook-side repairs at d99d296) were checked only through execution and stored-vs-fresh comparison of knowledge_graph_umls and the architecture studies, not by re-deriving the corrected prose numbers.
- Wall times are contention-inflated (three notebooks in parallel, probes and the background full run); no timing claim in the architecture studies was assessed either way.
- The previous reviewer's session ended before its report; this recovery ran one new probe (r01) and no new attacks.

DONE 07 20
