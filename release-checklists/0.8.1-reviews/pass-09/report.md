# Adversarial review -- PASS 09: example scripts 1-28

- Focus: `examples/` scripts 1-28 in sorted order (auto_example.py .. hidden_association_example.py), executed on the 0.8.1 candidate wheel, docstring/manifest claims checked against independent recomputation, then attacked (seeds, data shrinkage, JSON round-trip, warning audit).
- Wheel: `<review-root>/candidate-081/dist/mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, built from tree `6040ea38` (branch release/0.8.1).
- Version verification (run from this work dir, outside any checkout):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` ->
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Interpreter: Python 3.12.12; optional deps present: torch 2.14.0 (MPS available), scikit-learn, sympy, transformers, datasets, pyspark, mpi4py, dask; `peft` and `pomegranate` absent.
- Method: every script run once by `runner.py` (`<venv>/bin/python <example>`, cwd = `runs/<name>/`, `subprocess.run(timeout=1800)`, 3 concurrent, `PYTHONWARNINGS=default`, `MPLBACKEND=Agg`); stdout/stderr captured under `runs/`. Corpus copies are byte-identical to the source-tree copies (`diff -rq`), so docstrings were read from the corpus. Scratch checks live under `attacks/` (`*.py` with matching `*.out`) and `probe_calgen*.py`.

## Execution table (28/28 exit 0, total wall 119.9 s, 3-way concurrent)

| # | example | exit | wall (s) | stderr | verdict |
|---|---|---|---|---|---|
| 1 | `auto_example.py` | 0 | 2.2 | clean | verified |
| 2 | `autoregressive_enumeration_example.py` | 0 | 2.1 | clean | verified |
| 3 | `calibrated_report_demo.py` | 0 | 2.6 | 44 lines: 20x UserWarning from mixle/task/distill.py:744 (optimize max_its cap 1, library-internal), 2x from inference/estimation.py:2072 (max_its cap 30) | runs; demonstration vacuous (F02) |
| 4 | `capability_layer_example.py` | 0 | 2.1 | clean | verified |
| 5 | `copula_vine_example.py` | 0 | 5.6 | clean | runs; fitted models not JSON-round-trippable (F05) |
| 6 | `cross_modal_fit_receipt.py` | 0 | 2.8 | clean | verified |
| 7 | `doe_example.py` | 0 | 5.2 | clean | runs at seed 0 only (F03) |
| 8 | `engine_benchmark_example.py` | 0 | 4.1 | 1 UserWarning: example's own HMM optimize(max_its=5) unconverged | verified |
| 9 | `enumeration_example.py` | 0 | 2.2 | clean | verified |
| 10 | `enumeration_showcase_example.py` | 0 | 2.5 | clean | verified |
| 11 | `extensibility_seams_example.py` | 0 | 2.4 | clean | verified |
| 12 | `flagship_kg_agent.py` | 0 | 2.8 | 1 UserWarning: example's own KG optimize(max_its=40) unconverged | verified |
| 13 | `flagship_physics_inverse.py` | 0 | 4.1 | clean | runs; coverage receipt inconsistent (F08) |
| 14 | `flagship_triage_app.py` | 0 | 3.3 | clean | verified |
| 15 | `frontier_ecosystem_demo.py` | 0 | 4.5 | 1 UserWarning from inference/estimation.py:253 (library-internal composite optimize, max_its cap 10) | verified |
| 16 | `frontier_family_showcase.py` | 0 | 18.6 | 28 lines: 12x distill.py:744 warnings + 4 torch Kineto 'USDT ... profiler_start/stop' lines | runs; DONE line overstates priced rungs (F10) |
| 17 | `gallery_combinators_example.py` | 0 | 2.6 | clean | verified |
| 18 | `gallery_directional_example.py` | 0 | 2.7 | clean (but see P09-F01) | runs; ProjectedNormal fit wrong (F01) |
| 19 | `gallery_graphs_example.py` | 0 | 2.9 | clean | runs; RDPG truth silently clipped (F12) |
| 20 | `gallery_multivariate_example.py` | 0 | 3.1 | clean | verified |
| 21 | `gallery_processes_example.py` | 0 | 4.6 | clean | verified |
| 22 | `gallery_rankings_example.py` | 0 | 3.3 | clean (but see P09-F06) | runs; PlackettLuce fit is one MM step (F06) |
| 23 | `gallery_structured_example.py` | 0 | 6.2 | 8 UserWarnings: example's own max_its caps (1/40/1/1/8/10/1/1) unconverged | runs; TreeHMM degenerate, Segmental under-converged (F07, F11) |
| 24 | `gallery_univariate_example.py` | 0 | 2.8 | clean | verified |
| 25 | `geoscience_inversion_report.py` | 0 | 5.2 | 3 warnings: task/inverse.py:308 unconverged (max_its=1, library-internal); 2x 'seed derivation for a Posterior prompt is not reproducible' (library-internal, MXR-080-1848) | runs; inversion 100x overconfident by example bug (F04) |
| 26 | `heterogeneous_correctness_example.py` | 0 | 2.8 | clean (pomegranate absent -> comparison skipped, reported honestly) | verified |
| 27 | `heterogeneous_representation_example.py` | 0 | 4.5 | clean | verified |
| 28 | `hidden_association_example.py` | 0 | 12.1 | clean | verified |

"stderr" is the captured stderr with `PYTHONWARNINGS=default`. "verified" = every printed number with a ground truth was recomputed independently and matched (details under "Attacks that did not break anything").

## Findings

### P09-F01 [blocking] gallery_directional_example prints a wrong ProjectedNormal fit: one-pass estimate() is the first EM E-step, not a fit

- Surface: examples/gallery_directional_example.py; mixle.stats ProjectedNormalEstimator.estimate() / ProjectedNormalAccumulator._radius_for; mixle.inference.estimate
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_directional_example.py  -> 'ProjectedNormal (circle) true: ProjectedNormalDistribution(1.5, -0.5) fit: ProjectedNormalDistribution(0.7207, -0.2317)'. Then in a scratch script: t=ProjectedNormalDistribution(1.5,-0.5); data=list(t.sampler(seed=1).sample(8000)); f=estimate(data,t.estimator()); LL(f)=sum(f.seq_log_density(f.dist_to_encoder().seq_encode(data))) = -10482.30 vs LL(t) = -9029.23; scipy Nelder-Mead over the library's own log_density gives (1.5024,-0.4811) with LL -9028.03; optimize(data,t.estimator(),max_its=10,rng=RandomState(1),out=None) gives (1.5018,-0.4809). Same on seeds 2 and 3 (fit ~(0.71,-0.23), truth better by ~1400 nats each). See attacks/gallery_checks.out and attacks/iter_vs_estimate.out.
- Expected: The docstring says each family is 'built / sampled / re-estimated' and 'the fit is a one-liner'; the printed fit should be near (1.5,-0.5) with 8000 samples.
- Observed: Printed fit (0.72,-0.23): correct direction, |mu| = 0.757 = the sample mean resultant length. ProjectedNormalAccumulator._radius_for returns E[r]=1 on the first pass ('first pass: E[r] ~ 1 -> resultant points in the data mean direction'), so estimate() returns the first EM iterate; the M-step then stamps fit_metadata {'converged': True, 'solver': 'em-m-step'} on it. The true MLE is 1453 nats better on the example's own sample.
- Notes: Two defects: (a) the shipped gallery uses estimate() on an iterative (EM) estimator and prints the first iterate as the fit; (b) the library labels that single pass converged=True. optimize() converges to the MLE in <=10 iterations, so the example fix is one line; the library fix is to not claim convergence from a one-pass E[r]=1 heuristic (or to make estimate() iterate).

### P09-F02 [blocking] calibrated_report_demo: the per-claim shape gate certifies threshold=inf and serves 0 of 120 shape claims (60 of them clear); the demo's own teacher mislabels 353 of 360 volumes

- Surface: examples/calibrated_report_demo.py (claim_teacher, _score_shape_candidate, build_shape_gate); mixle.task.calibrated_generator.CalibratedGenerator
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/calibrated_report_demo.py -> '120 volumes reported; 0 fully accepted, 120 flag >=1 abstention' and every listed volume 'shape=abstained'. Scratch (attacks/../probe_calgen.py, probe_calgen2.py): import calibrated_report_demo as d; gate=d.build_shape_gate(d.build_records(120,1)); gate.qhat == inf; gate.risk_receipt['accepted']==0; gate.serve abstains on all 360 calibration records and on 120 fully-clear probes (build_records(40,3,ambiguous_fraction=0.0)). Confusion of planted shape -> claim_teacher label on 360 clear volumes: blob_left->stripe 118/120, blob_right->stripe 120/120, stripe->blob_left 70, ->blob_right 45, ->stripe 5. The scorer's top candidate equals the planted shape for all 240 blobs but equals the TEACHER label on only 117/360.
- Expected: Docstring: 'draw the 3 candidate labels, score them from the patches, and certify a held-out selective-risk gate, so an uncertified volume abstains instead of guessing' on 'a mix of clear and ambiguous volumes' -- i.e. clear volumes should be served, ambiguous ones abstained.
- Observed: Nothing is ever served. Cause is inside the example: claim_teacher calls a volume 'stripe' when max-min octant mean < 0.15, but a clear blob (a 2x2x2 cube of +1.0 inside a 4x4x4 octant) raises one octant mean by exactly 8/64 = 0.125, so blobs have spread 0.04-0.15 (all < 0.15 -> 'stripe') and stripes (16 voxels x 0.6 across four octants) have spread 0.14-0.18 (mostly > 0.15 -> 'blob'). The teacher and the candidate scorer therefore disagree on ~2/3 of records, the selective-risk gate correctly finds no threshold meeting alpha=0.1 at 95% confidence, and the printout reports the result without noting that the gate is degenerate.
- Notes: The library did the right thing (refused to certify an uninformative scorer). The shipped demonstration of 'every sentence carries a receipt' never exercises the accept path for its categorical claim. Any teacher cutoff between the two spread ranges (e.g. 0.14 with the bump set so ranges separate) or a scorer defined from the teacher would fix it. Lowering alpha does not: at alpha=0.2 the gate still abstains on 79/120 clear probes because the labels are wrong, not noisy.

### P09-F03 [real] mixle.doe.minimize crashes with an uncaught torch LinAlgError (GP Cholesky not positive-definite) on 11 of 12 seeds; doe_example works only at seed=0

- Surface: mixle.doe.bayesopt.minimize -> _fit_surrogate -> mixle.models.gaussian_process.GaussianProcessRegressor.log_marginal_likelihood (torch.linalg.cholesky)
- Reproduction: cd <workdir>; <review-root>/candidate-081/venv/bin/python -c "from mixle.doe import minimize; obj=lambda p: float((p[0]-1.0)**2+(p[1]+2.0)**2); print(minimize(obj,[(-5.0,5.0),(-5.0,5.0)],n_init=5,n_iter=15,seed=1).best_x)"  -> torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 17 is not positive-definite). Seeds 1..11 all fail (leading minor orders 11-18); seed 0 (the example's) succeeds. See attacks/doe_seeds.out.
- Expected: The example's exact call with any seed returns a BayesOptResult near (1,-2); at worst a mixle-level error naming the surrogate problem.
- Observed: Raw torch traceback from inside the GP surrogate fit once the acquisition loop has proposed 6-13 points that cluster near the optimum; the fitted noise (log_noise is a trained parameter, lower bound only the 1e-6 jitter) collapses on a noise-free quadratic and K + (noise^2 + 1e-6) I loses positive-definiteness.
- Notes: This is the DOE surface the manifest lists as required ('DOE examples because the current release scope includes pool-based DOE'). No user-facing knob on minimize() sets the surrogate's jitter/noise floor (fit_kwargs go to gp.fit, not the constructor). A robust fix is a noise floor or adaptive jitter in log_marginal_likelihood, or catching the LinAlgError in _fit_surrogate and re-fitting with a larger jitter.

### P09-F04 [real] geoscience_inversion_report: the forward model handed to learn_inverse is noise-free while the observation is noisy, so the amortized posterior is ~100x overconfident, every receipt fails and the report always abstains

- Surface: examples/geoscience_inversion_report.py (invert_new_observation.forward_salt); mixle.task.inverse.learn_inverse
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/geoscience_inversion_report.py -> 'posterior: mean=4.171 std=0.001 km', 'SBC p-value=2.78e-25 (pass=False)', 'coverage@0.5=0.353 (pass=False)', 'calibrated report: ABSTAIN'. Scratch attacks/geo_attack.py: exact grid posterior of depth given the noisy 3-station reading (Gaussian prior from the M2 rollout, SENSOR_NOISE=0.15) has mean 4.205, sd 0.099. Re-running learn_inverse with the identical arguments but forward(theta) += SENSOR_NOISE*randn(3) gives posterior mean 4.204 sd 0.0985, sbc_p=0.63 (pass), coverage@0.5=0.520, coverage@0.9=0.880.
- Expected: Docstring: the inverse is 'trained against the same salt-regime forward physics step 2 just characterized' with receipts that 'say whether the inversion is trustworthy', then a calibrated claim is served 'only if it conformally clears a held-out threshold'. With a correctly specified simulator the posterior should be ~N(4.2, 0.1) and the report served.
- Observed: forward_salt calls _amplitude(depth, formation) with rng=None (zero noise), so q(depth|amplitude) learns a near-deterministic inverse (sd 0.0009 km) of a noisy measurement; SBC and 50%-coverage fail as they should, and stage 4 abstains. The script still prints 'OK: ... real numbers throughout'. The manifest's 2026-07-21 note records only that 'the calibration layer detected a poorly calibrated candidate and abstained', without noting the cause is the example's own simulator.
- Notes: The library's receipts behave correctly; the example demonstrates only its failure path because of a one-line omission in the example. Adding SENSOR_NOISE to forward_salt makes every stage succeed.

### P09-F05 [real] copula_vine_example's fitted models cannot be JSON round-tripped: RVineCopulaDistribution serialization fails on its private _Edge, and GaussianCopulaDistribution produces write-only JSON, although both classes are in the serialization registry

- Surface: mixle.stats.dump_models / load_models; mixle.utils.serialization; mixle.stats.multivariate.rvine_copula.RVineCopulaDistribution (_Edge); mixle.stats.multivariate.gaussian_copula.GaussianCopulaDistribution; mixle.stats.combinator.copula.CopulaDistribution
- Reproduction: Scratch attacks/roundtrip.py: import copula_vine_example as cv; train=cv.generate(seed=0,n=2000); vp=cv.prototype(RVineCopulaDistribution.independence(3)); vf=optimize(train,vp.estimator(),prev_estimate=vp,max_its=3,out=None); mixle.stats.dump_models(vf) -> SerializationError: objects of type mixle.stats.multivariate.rvine_copula._Edge are not JSON serializable by mixle. gp=cv.prototype(GaussianCopulaDistribution(np.eye(3))); gf=optimize(...); dump_models(gf) -> SerializationError: dump_models produced JSON that load_models cannot read back (registered class 'mixle.stats.multivariate.gaussian_copula.GaussianCopulaDistribution' requires a class-owned __pysp_setstate__ hook; constructor fields are absent: corr). serializable_class_ids() contains both RVineCopulaDistribution and GaussianCopulaDistribution.
- Expected: A fitted CopulaDistribution with an R-vine or Gaussian core round-trips through dump_models/load_models (the CHANGELOG's stated safe serialization route), or the classes are not listed as serializable.
- Observed: Both cores fail; the same fits of auto_example and cross_modal_fit_receipt round-trip fine with identical log-likelihoods.
- Notes: The 0.8.0 CHANGELOG lists serialization/lifecycle round-tripping as an audited lens and records __pysp_serializable__ disclosure fixes; the copula family shipped in the examples was missed. Pickle is the only persistence for these fits.

### P09-F06 [real] gallery_rankings_example prints a single Plackett-Luce MM step as the fit: 25 nats below the MLE on its own sample (0.366 vs 0.40 for the top item)

- Surface: examples/gallery_rankings_example.py; mixle.stats.rankings.plackett_luce.PlackettLuceEstimator.estimate ('Return one MM estimate'); mixle.inference.estimate
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_rankings_example.py -> PlackettLuce fit [-1.0059,-1.2085,-1.5125,-2.1601] (probs 0.366/0.299/0.220/0.115) for true log([0.4,0.3,0.2,0.1]). Scratch attacks/gallery_checks.out + iter_vs_estimate.out: on the same 3000 rankings LL(estimate())=-8675.01, LL(true)=-8650.91, scipy BFGS MLE probs [0.4016,0.2986,0.1974,0.1025] with LL -8650.30; optimize(data,t.estimator(),max_its=10,rng=RandomState(1),out=None) reaches [-0.9124,-1.2087,-1.6226,-2.2783] with LL -8650.30.
- Expected: 'the recovered parameters can be read straight off the printed model' -- a fit within sampling error of (0.4,0.3,0.2,0.1).
- Observed: estimate() performs exactly one minorization-maximization step from the zero initialisation; the printed numbers are that step. optimize() converges in <=10 steps.
- Notes: Same shape as P09-F01: a one-pass estimate() on an iterative estimator presented as a fit. Mallows and Matching in the same script are fine (Matching's 6-permutation distribution matches the truth to TV 0.003).

### P09-F07 [real] gallery_structured_example's TreeHiddenMarkov fit is degenerate at the example's seed (both states collapse to N(-0.25, 9.8)); 3 of 5 rng seeds collapse, any informative init recovers the planted +/-3 states

- Surface: examples/gallery_structured_example.py (TreeHiddenMarkov section); mixle.stats TreeHiddenMarkovEstimator.seq_initialize (uniform random state assignment init)
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_structured_example.py -> 'TreeHiddenMarkov ... held-out mean log-density: -17.965'. Scratch attacks/hmm_rdpg.out + iter_vs_estimate.out: same data/estimator, optimize(..., max_its=8|100|400, rng=RandomState(1)) all return topics [Gaussian(-0.229, 9.777), Gaussian(-0.285, 9.753)], transitions ~0.5, held-out -17.965 vs the true model's -15.071 (train -10.879 vs -9.173). rng seeds 1,2,5 collapse identically; seeds 3,4 give means (-3.00, 2.95), vars (1.02, 0.90), held-out -15.180; prev_estimate with means (-2,+2) or the truth also gives -15.180.
- Expected: 'built / sampled / re-estimated with EM' -- a two-state fit near the planted N(-3,1)/N(3,1), or at least a printed number a reader can judge (the true model's held-out density).
- Observed: The uniform-random state-assignment initialisation lands the well-separated problem on the symmetric fixed point (identical emissions) from which EM cannot escape; the example prints the degenerate model's held-out density with no baseline, so the failure is invisible.
- Notes: Not a convergence budget issue (400 its identical). An initialisation that breaks symmetry (e.g. k-means-style or a data-quantile split) or a restart-on-degeneracy check in the estimator would fix the estimator side; printing the true model's held-out density would make the example self-checking.

### P09-F08 [minor] flagship_physics_inverse: MCMC effective sample size ~50 of 1500 draws; the same dataset is printed as a 90%-interval miss (1500 draws) and as a coverage hit (800 draws); exact-posterior coverage is 11/12, not 12/12

- Surface: examples/flagship_physics_inverse.py; mixle.ppl fit(how='mcmc') with potentials
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/flagship_physics_inverse.py -> 'one dataset : posterior mean 1.361, 90% CI [1.321, 1.394]' (true k=1.4 outside) then 'the 90% interval bracketed the truth 12/12 times'. Scratch attacks/physics_check.py: exact grid posterior for observe(0) is mean 1.3611 sd 0.0219 90% CI [1.3255,1.3975] (excludes 1.4); MCMC lag-1 autocorrelation 0.93, ESS ~50 at 1500 draws, ~26 at 800; replicate s=0 (same data, 800 draws) gives [1.321,1.402] -> counted as a hit; exact-posterior 90% intervals bracket 1.4 on 11/12 replicates, MCMC intervals on 12/12.
- Expected: A 'credible interval per dataset' that is the posterior's; the coverage receipt consistent with the top-line interval for the same data.
- Observed: Chains are strongly autocorrelated (no ESS printed); the 800-draw intervals are quantile-noisy enough to flip replicate 0 from miss to hit. The narrative caveat about n=12 is correct but does not cover this.
- Notes: Rough edge rather than wrong result: the posterior mean/sd are right. Reporting ESS or thinning, and using the same draw count in both places, would remove the visible inconsistency.

### P09-F09 [minor] Library-internal optimize() calls emit 'unconverged fit' UserWarnings that the calling example cannot silence (distill._fit_mlp max_its=1 x20; automatic composite path; task.inverse), plus a self-inflicted 'seed derivation ... not reproducible' warning from PosteriorDescriber

- Surface: mixle/task/distill.py:744 (_fit_mlp: optimize(..., max_its=1) without delta=None); mixle/inference/estimation.py:253 (automatic composite optimize, max_its cap 10); mixle/inference/estimation.py:2072 (fit(), max_its 30 / delta 1e-6 from solve_structured); mixle/task/inverse.py:308 (max_its=1); mixle/reason/language_bridge.py:421 and mixle/task/calibrated_generator.py:595 (Posterior objects used as prompts)
- Reproduction: PYTHONWARNINGS=default <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/calibrated_report_demo.py 2>&1 >/dev/null | grep -c 'max_its cap (1)' -> 20. Same for frontier_family_showcase.py (12), frontier_ecosystem_demo.py (estimation.py:253, 1), geoscience_inversion_report.py (inverse.py:308 plus two MXR-080-1848 warnings).
- Expected: Library code that deliberately runs a fixed number of steps passes delta=None (the warning text itself says so); library code that builds its own prompts does not trip its own reproducibility warning.
- Observed: Every distillation chunk warns; the warnings point at library file:line, not at anything the user wrote, and the docstrings of the affected examples do not mention them.
- Notes: Cosmetic but noisy: 44 stderr lines for calibrated_report_demo. Fix is delta=None at the four internal call sites and a stable prompt key in PosteriorDescriber.

### P09-F10 [minor] frontier_family_showcase's DONE line claims 'headline + 2 rung(s), each I1-quantized and priced' while rung_edge was halted and only 2 frontier points exist; the 'size ladder' produced a 0.993x rung that costs 1.32x per request at identical quality

- Surface: examples/frontier_family_showcase.py (main: len(family.rungs) vs family.passed_rungs()); mixle.task.checkpoint_family_ladder / deploy_family
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/frontier_family_showcase.py -> 'halted_at: rung_edge', 'rung_mid: 20128 params (0.993x headline)', 'rung_edge: 20272 params (1.000x headline) -- parameter count increased', frontier lists only headline ($1.00/req, q=0.551, 453972 FLOPs) and rung_mid ($1.32/req, q=0.551, 599704 FLOPs), then 'headline + 2 rung(s), each I1-quantized and priced'. The script's own assert len(serve_receipt.points) == 1 + len(family.passed_rungs()) passes with 2 points.
- Expected: The DONE line counts priced rungs (1), and a compression ladder yields smaller/cheaper rungs.
- Observed: len(family.rungs) counts attempted rungs; the priced/quantized set is headline + rung_mid only. The receipted pipeline runs, but the only accepted rung is 0.7% smaller and 32% more expensive at the same quality, so the demonstration of 'shrinking' is nominal.
- Notes: Printed-narrative inaccuracy in the example; the library receipts (halted_at, is_monotone_frontier) are correct. Also the edge cascade escalates 0/250 so cascade == student on every axis.

### P09-F11 [minor] gallery_structured_example's SegmentalHiddenMarkov section is under-budgeted (max_its=10): printed held-out -19.79 vs -16.36 reachable at 100 iterations (truth -16.35)

- Surface: examples/gallery_structured_example.py (SegmentalHiddenMarkov section)
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_structured_example.py -> 'SegmentalHiddenMarkov ... held-out mean log-density: -19.791' with stderr 'stopped at the max_its cap (10) ... last objective gain 5.23'. Scratch attacks/hmm_rdpg.out: max_its=100 -> held-out -16.359, emissions (3.00,0.97)/(-2.97,1.01); true model held-out -16.347.
- Expected: A converged fit, or a printed baseline making the gap visible.
- Observed: EM stopped while still gaining 5 nats per iteration; the printed number is 3.4 nats/observation worse than the converged fit, with no baseline.
- Notes: Raising max_its to ~100 fixes it (sub-second).

### P09-F12 [minor] RandomDotProductGraphDistribution silently clips latent dot products >1 to probability 1.0 with numerical_repairs() empty; gallery_graphs' own rng.rand(8,2) positions trigger it, so the gallery's 'true' model is not the rank-2 model it fits

- Surface: mixle.stats.RandomDotProductGraphDistribution.__init__ (probs), numerical_repairs(); examples/gallery_graphs_example.py
- Reproduction: <review-root>/candidate-081/venv/bin/python -c "import numpy as np; from mixle.stats import RandomDotProductGraphDistribution as R; X=np.random.RandomState(0).rand(8,2); d=R(X); P=X@X.T; off=~np.eye(8,dtype=bool); print(P[off].max(), (P[off]>1).sum(), np.asarray(d.probs)[off].max(), d.numerical_repairs())" -> 1.0739911270060856 4 1.0 ()   (no warning either). Scratch attacks/hmm_rdpg.out: the gallery's fit has LL -23866.6 vs the clipped truth's -23432.2 on its 2000 graphs and edge probabilities up to 0.116 from the empirical frequencies.
- Expected: Per the 0.8.0 CHANGELOG's disclosure policy, a parameter clamp that changes the model is reported through numerical_repairs() (or rejected).
- Observed: Silent clamp; the gallery then samples from a clipped, non-rank-2 model and fits a rank-2 one, and prints the fitted positions with no comparison.
- Notes: The example should draw positions with |x|<=1/sqrt(2) (or normalise); the library should disclose the clamp.

### P09-F13 [minor] Shipped examples import private names: _design_row, RandomVariable._bound / rv._name, fit._result.samples()

- Surface: examples/cross_modal_fit_receipt.py (from mixle.inference.bayesian_network import _design_row); examples/extensibility_seams_example.py (RandomVariable._bound, rv._name); examples/flagship_physics_inverse.py (fit._result.samples())
- Reproduction: grep -n '_design_row\|_bound\|_result\|_name' in the three files.
- Expected: Examples exercise the public surface; a fitter registered through register_fitter has a public way to build a bound RandomVariable, and an MCMC fit exposes its draws publicly.
- Observed: Three examples depend on underscore names; the extensibility seam demo in particular cannot be written without _bound.
- Notes: No behavioural defect on the candidate; a public-API gap the examples paper over.

### P09-F14 [docs] Examples' own max_its caps emit unconverged-fit UserWarnings on stderr that their docstrings do not mention (engine_benchmark, flagship_kg_agent, gallery_structured)

- Surface: examples/engine_benchmark_example.py (HMM optimize max_its=5, gain 3.9e-10), examples/flagship_kg_agent.py (max_its=40, gain 6.7e-3), examples/gallery_structured_example.py (8 warnings: max_its=1/40/1/1/8/10/1/1)
- Reproduction: PYTHONWARNINGS=default <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_structured_example.py 2>&1 >/dev/null | grep -c 'stopped at the max_its cap' -> 8; likewise 1 each for engine_benchmark_example.py and flagship_kg_agent.py.
- Expected: Fixed-iteration demonstrations pass delta=None (as the warning suggests) or the docstring states that the fits are budgeted; engine_benchmark's HMM row warns at a gain of 3.9e-10 (below any meaningful delta).
- Observed: Warnings printed on every run; the manifest's 'passed' status does not record them.
- Notes: Cosmetic.

### P09-F15 [docs] Stale docstring/narrative details: auto_example calls a float field 'an int'; hidden_association runtime claim 30-60 s vs 12 s measured; gallery_univariate's Binomial prints n=9 for a planted n=10 without comment; gallery_structured's QuantizedHMM prints theta=0.915 for a planted 0.6 with no note that the exponents are re-estimated jointly

- Surface: examples/auto_example.py docstring; examples/hidden_association_example.py docstring; examples/gallery_univariate_example.py (BinomialEstimator uses the sample maximum as n); examples/gallery_structured_example.py (QuantizedHMM section)
- Reproduction: <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/auto_example.py (first field data 1, 3, 3.1 -> GaussianDistribution(2.3667, 0.9356)); <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_univariate_example.py -> 'BinomialDistribution(p=0.444022, n=9, min_val=0)' for true (p=0.4, n=10); <review-root>/candidate-081/venv/bin/python <review-root>/corpus-081/examples/gallery_structured_example.py -> 'theta: 0.915' (fit exponents [[5,0],[5,0]] / [[0,2],[0,0]] / [0,3] vs planted [[0,1],[1,0]] / [[0,1],[1,0]] / [0,1]; LL(fit) -2318.28 vs LL(true) -2316.76, see attacks/gallery_checks.out); hidden_association wall 12.1 s (runs/summary.json).
- Expected: Docstrings and printed 'true vs fit' lines that a reader can reconcile.
- Observed: As listed; none is a wrong computation (the Binomial fit keeps the mean at 4.0 with n = max observed = 9; the quantized HMM reaches a nearby local optimum under a different exponent assignment).
- Notes: Docs only.

## Attacks that did not break anything

- `auto_example.py`: every number in the printed tree reproduces by hand (Gaussian MLE mean 2.3667 / variance 0.9356 of (1, 3, 3.1); Optional p=1/3; Categorical 2/3-1/3; bag Categorical 3/3/2 of 8; length law 1/3-2/3). `dump_models`/`load_models` round-trip preserves per-row log-densities exactly; `optimize(data, est, out=None)` also works as the docstring claims; a single row fits (Optional field becomes `IgnoredDistribution(PointMass(None))`).
- `autoregressive_enumeration_example.py`: brute-force enumeration of all 125 sequences for seeds 0-3 confirms top-5, `unrank(42)`, `rank`, `threshold(20)` and `count` exactly; the terminating model's top-5, rank 4 and cumulative 0.72 (which correctly includes the tied (0,1,2)) match a depth-14 brute force. `attacks/ar_bruteforce.out`.
- `capability_layer_example.py`: Categorical entropy 1.0297, Poisson(2) entropy 1.7049 / skewness 0.7071 / excess kurtosis 0.5 / median 2 / mode 2, Gaussian(0,1) entropy 1.4189 all match closed forms; catalog counts stable.
- `copula_vine_example.py`: the closed-form Clayton(theta=4) joint lower-tail mass (3q^-4-2)^(-1/4)=0.0760 matches the printed 0.0770 (n=1e5 Monte Carlo); seeds 1-3 and n=500, n=50 all recover three Clayton edges, marginals within 3% (n>=500) and vine tail rate 0.074-0.077 vs Gaussian 0.042-0.047; n=1 raises a clear `ValueError: copula fit requires at least 2 observation rows`. Only the JSON round-trip failed (F05). `attacks/copula_attack.out`.
- `cross_modal_fit_receipt.py`: seeds 1-3 recover edges 3<-[1,2] and 0<-[1] with coefficients (30.1, 19.8) etc., corr 0.996, rmse ~3.1 (noise sd 3.0); n=30 and n=10 still recover the price parents; JSON round-trip preserves predictions; n=1 yields a degenerate single-parent fit with corr=nan (garbage in, no crash). `attacks/crossmodal_attack.out`.
- `doe_example.py` at its own seed: the Latin hypercube has one point per stratum in each column; BayesOpt best (0.98,-2.00) with best_y 0.0006; Sobol S1 for y=x0^2+0.5x1 on U(-5,5)^2 is 55.56/57.64=0.964 by hand, printed 0.963/0.036. Other seeds crash (F03).
- `engine_benchmark_example.py`: runs numpy / torch-cpu / torch-mps with its internal parity assertions passing (rtol 2e-4); nothing to recompute beyond that.
- `enumeration_example.py`: all printed probabilities are exact products (0.42/0.28/0.18; mixture 0.5/0.5).
- `enumeration_showcase_example.py`: brute force over a 3 x 120 x 1999 grid confirms top-5 (with Geometric support starting at 1), `seek(10000)` log-prob -53.3228 (also by hand: log 0.2 + log Pois(12;4) + log Geom(122;0.3)), exactly 10000 strictly-more-probable records (bracket [10000,10000] exact=True), the 95% nucleus bracket [212,232] containing the brute-force 231, and mixture `rank(5)=17` (values 0-4 and 14-25 beat p(5)=0.01807) with cumulative 0.9026. `attacks/enum_bruteforce.out`.
- `extensibility_seams_example.py`: printed 4.95 / 1.96 equal numpy's mean/std of the seed-0 sample; `register_fitter` route dispatches (private-name use noted in F13).
- `flagship_kg_agent.py`: withheld fact confidence 0.12 = 3/25 draws at p=0.2; type-violating triple rejected with the correct domain message; typed completion picks acme; abstention on the unlinked question.
- `flagship_physics_inverse.py`: MCMC posterior mean/sd (1.361/0.023) agree with the exact grid posterior (1.361/0.022); the interval-width narrative is right (F08 covers the mixing/consistency issue only).
- `flagship_triage_app.py`: `mixle.inference.nonparametric.ks_2samp` p-values (0.10628, 1.28e-58) equal scipy's to all printed digits; harness answered/escalated/refused as described; the secret is redacted and the substrate scan is clean.
- `frontier_ecosystem_demo.py`: `optimize(records)` discovers the planted graph 0->2, 2->3, 1->3; PIT 0.165 vs floor 0.158 is judged calibrated because `is_calibrated()` uses tol = 2.5x the floor (documented); the Laplace 90% interval [4.775,5.095] brackets 5.0 with the expected ~0.16 half-width at n=400; abstention path does not call the answerer.
- `gallery_combinators_example.py`: all nine fits within sampling error of the planted parameters (Transform's base equals Composite's Gaussian shifted by 5/2 on the same seed, as it must).
- `gallery_directional_example.py`: VonMises, WrappedNormal, WrappedCauchy within 1% and Watson recovers the axis (sign flip is the axial symmetry); only ProjectedNormal fails (F01).
- `gallery_graphs_example.py`: Erdos-Renyi p=0.3026; the SBG fit's `block_probs` are [[0.803,0.102],[0.102,0.700]] (recovered, just not printed); KnowledgeGraph runs. RDPG issue in F12.
- `gallery_multivariate_example.py`, `gallery_processes_example.py`, `gallery_univariate_example.py`: every printed fit within sampling error of the planted parameters (19 univariate families, Hawkes (0.62,0.73,1.38) vs (0.6,0.7,1.3) on 80 realizations, CRP alpha 2.007, birth-death (0.51,0.29,0.20)); Binomial n=9 explained in F15.
- `gallery_rankings_example.py`: Mallows theta 0.82 vs 0.8; the Matching fit, compared as a distribution over the 6 perfect matchings (the weights are only identified up to row/column scaling), is within TV 0.003 of the truth and matches the empirical counts. PlackettLuce in F06.
- `gallery_structured_example.py`: Markov chain transitions 0.895/0.205 vs 0.9/0.2; IBP feature probs [0.81,0.50,0.30,0.09]; Chow-Liu / ICL trees attach the noisy copy to its source; Record/DictRecord fits identical and correct; heterogeneous mixture lands on a plausible local optimum. Tree/Segmental HMM and QuantizedHMM in F07/F11/F15.
- `geoscience_inversion_report.py`: the discovered DAG (formation -> amplitude, formation+amplitude -> depth) and the do(salt) rollout (depth 4.52 +/- 0.48 vs planted 4.5 +/- 0.6) are right; the inversion issue is F04.
- `heterogeneous_correctness_example.py`: seeds 1-3 and 10% data all recover both clusters (TV <= 0.03); its acceptance oracle is genuine (TV <= 0.05 on the full categorical vector). One observation per cluster fails inside the example's own readout (`pmap[k]` KeyError for an unseen category) -- garbage in.
- `heterogeneous_representation_example.py`: held-out accuracy 1.00 on 100 fresh records (the label is carried by the text bytes, so this is expected); VQ + Markov chain path runs.
- `hidden_association_example.py`: diagonal 0.912/0.893/0.906 vs 0.9, marginal (0.294,0.197,0.509) vs (0.3,0.2,0.5); seeds 2-3 and n=100 also recover the diagonal (>=0.84); n=1 returns a degenerate table without crashing.
- Stale-API audit: with `PYTHONWARNINGS=default` no DeprecationWarning/FutureWarning appeared in any of the 28 runs; the only warnings are the unconverged-fit and seed-derivation UserWarnings listed in F09/F14.
- Exception handlers: the only broad `except` in the set (`heterogeneous_correctness_example.try_pomegranate`) reports the caught error verbatim rather than hiding it.

## Summary

- Counts: blocking 2 (P09-F01 ProjectedNormal gallery fit, P09-F02 calibrated_report_demo dead gate), real 5 (P09-F03 doe.minimize crashes on 11/12 seeds, P09-F04 geoscience noise-free forward, P09-F05 copula JSON round-trip, P09-F06 Plackett-Luce one-step fit, P09-F07 TreeHMM degenerate init), minor 6 (F08-F13), docs 2 (F14, F15).
- All 28 examples exit 0 on the candidate (total 119.9 s wall, 3-way concurrent); no crash on the shipped path, no deprecated-API use.
- Worst finding: P09-F03 -- `mixle.doe.minimize(objective, bounds, n_init=5, n_iter=15, seed=s)` raises an uncaught `torch._C._LinAlgError` from the GP surrogate for every seed 1..11; the shipped `doe_example.py` passes only because it uses seed 0, and the manifest lists DOE as a required release workflow.
- Systemic pattern worth a sweep by the library-internals passes: one-pass `estimate()` on iterative estimators (ProjectedNormal, Plackett-Luce) returns a first iterate that the library stamps `converged=True`; and shipped demos whose headline mechanism never fires because of a bug in the demo's own synthetic setup (calibrated_report_demo, geoscience_inversion_report).
- Nothing outside `<review-root>/reviews-081/pass-09/` was modified; the venv and wheel were not touched.
