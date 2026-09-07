# Adversarial review PASS 03 — inference layer (EM loop, provenance, propose/describe/Model, calibration/GoF, seq_* utilities)

- Pass: 03 of 10
- Focus: `mixle.inference.optimize` / `fit` (max_its, delta, delta=None, init_p, prev_estimate, rng, on_step, print_iter), `fit_provenance()`, `propose()` / `describe()` / `Model`, `pit_values` / `ks_1samp` / nonparametric tests, `seq_estimate` / `seq_initialize`, `numerical_repairs()` disclosures.
- Wheel: `mixle-0.8.1-py3-none-any.whl` sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d` (re-hashed locally, matches), built from tree `6040ea38` on `release/0.8.1`.
- Version verification (run from the work dir, not a source checkout):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` ->
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle` (Python 3.12.12)
- Work dir: `<review-root>/reviews-081/pass-03/` (attack scripts and raw outputs under `attacks/`, notebook logs `log-nb-*.txt`, example logs `log-ex-*.txt`, executed notebooks `executed-N-*.ipynb`). Nothing outside the work dir was modified.

## Executed corpus notebooks (nbconvert --execute, kernel pinned to the candidate venv's `python3`, timeout 1200 s)

| # | notebook | exit | wall |
|---|---|---|---|
| 1 | tutorials/fitting_and_estimation.ipynb | 0 | 24 s |
| 2 | data_science/em_and_map_strategies.ipynb | 0 | 41 s |
| 3 | data_science/mle_and_sufficient_statistics.ipynb | 0 | 24 s |
| 4 | data_science/model_based_clustering_evaluation.ipynb | 0 | 42 s |
| 5 | data_science/gaussian_mixtures_in_depth.ipynb | 0 | 50 s |
| 6 | applications/pillar_validation/calibration_A.ipynb | 0 | 23 s |
| 7 | data_science/density_estimation_mixtures_vs_kde.ipynb | 0 | 153 s |
| 8 | architecture_studies/engine_benchmarks.ipynb | 0 | 68 s |

All eight executed to completion with no error outputs. Notebooks 1, 2 and 8 carry `UserWarning: optimize() stopped at the max_its cap ...` on stderr (7, 20 and 4 entries respectively); see P03-F10 for where the surrounding prose contradicts them. Notebooks 3-7 produced no stderr at all.

## Executed examples (candidate venv python, from `exrun/`)

| example | exit | wall |
|---|---|---|
| quickstart_example.py | 0 | 19 s |
| auto_example.py | 0 | 10 s |
| skeptic_challenge_example.py | 0 | 64 s |
| engine_benchmark_example.py | 0 | 27 s |
| project_neural_to_structured.py | 0 | 146 s |
| hidden_association_example.py | 0 | 31 s |
| lookback_hmm_example.py | 0 | 24 s |
| hierarchical_mixture_example.py | 0 | 23 s |

All exit 0. `skeptic_challenge_example.py` emits three cap warnings (one with `last objective gain -56.6`, see P03-F06), `hierarchical_mixture_example.py` one (see P03-F10), `engine_benchmark_example.py` one (max_its=5 benchmark, benign), `project_neural_to_structured.py` many from `mixle/task/distill.py:744` (max_its=1 projection fits, benign). `lookback_hmm_example.py` (`delta=None, max_its=1000`) ran the full 1000 iterations with no shortfall warning, as its docstring claims.

## Findings

### P03-F01 (real) — `optimize()`/`fit()` auto-structure return carries no `fit_provenance()` / `numerical_repairs()` at all
The default entry point (`optimize(rows)` with `estimator=None`, `structure='auto'`) returns a `HeterogeneousBayesianNetwork` when a dependence graph beats the independent composite. That object has neither `fit_provenance` nor `numerical_repairs` (AttributeError), so the receipt the docstring promises ("ships on the model") is absent on the flagship "one `optimize(data)`" path used by `quickstart_example.py` and `skeptic_challenge_example.py`. `Model().fit(rows)` records no `n_iter`/`converged` and adds no note. During that fit the per-factor fits hit their `max_its` cap (warnings with `delta=1e-06`), which a reader of the returned model cannot see. `learn_structure()`'s `DependencyTreeDistribution` *does* carry a receipt, so the `FitProvenanceCarrier` fix landed on one structure learner but not the one `optimize()` returns. Reproduction in `findings.json`; raw output `attacks/a13_hbn_prov_neggain.out`.

### P03-F02 (real) — a one-shot iterator with `estimator=None` silently fits an empty `IgnoredDistribution`
`optimize(x for x in data)`, `optimize(iter(data))`, `optimize(map(float, data))`, `fit(x for x in data)` all return `IgnoredDistribution` with `FitProvenance(n_observations=0, converged=True, final_objective=0.0)` and `log_density(v) = -inf` for every `v`, no warning. Auto-inference consumes the iterator; the empty-data guard tests `hasattr(data, '__len__')` and lets the exhausted iterator through; `seq_encode` encodes zero rows. The explicit-estimator path materializes the same generator and fits all 200 rows, and `propose()` materializes correctly, so the entry points disagree. Output: `attacks/a12_neg_gain_generator.out`, `attacks/a1.out`.

### P03-F03 (real) — empty encoded corpus yields a fabricated converged fit; the changelog's low-level guard does not reject
`optimize(None, GaussianEstimator(), enc_data=seq_encode([], enc))` and `enc_data=[]` return `GaussianDistribution(0.0, 1e-08)` with `iterations=2, converged=True, n_observations=0, repairs=('variance-floored(0 -> 1e-08)',)`; a mixture estimator returns two such components at 0.5/0.5. `seq_initialize` and `seq_estimate` on the same corpora return the same defaults with no error. CHANGELOG 0.8.0 says "the high-level `optimize([], ...)` already guarded this cleanly; the guard now reaches the low-level entry point too" -- what reached it stops the `IndexError`; it does not reject, and `converged=True` over zero observations is precisely the fabricated-fit shape the `data=[]` guard exists to prevent. Output: `attacks/a6b.out`.

### P03-F04 (real) — receipts say `objective='map'` for prior-free Composite/Optional/Sequence estimators
`CompositeEstimator([GaussianEstimator(), PoissonEstimator()]).get_prior()` returns `[None, None]`; `_resolve_objective` tests `get_prior() is not None` and answers `'map'`. The receipt reports `objective='map', algorithm='em'` while `final_objective` equals the plain log-likelihood sum to 1e-13 (no prior term exists). Forcing `objective='mle'` gives `algorithm='fused-em'`. The same composite *inside* a `MixtureEstimator` resolves to `'mle'`/`'fused-em'`, so the mislabel is specific to the top-level container -- which is every `estimator=None` tabular fit (auto-inferred composites, DataFrames, `Optional`-wrapped NaN columns). The other prior detector, `_estimator_carries_prior`, correctly answers `False`. Output: `attacks/a5_prior.out`, `attacks/a8_map_cost.out`.

### P03-F05 (minor) — `delta=0` accepted but unsatisfiable
Validator accepts `delta=0` (rejects `<0`, `nan`, `inf`); the convergence test is `0.0 <= dll < delta`, so it never fires; the run hits the cap and warns "before the objective settled (last objective gain 0, delta=0)". Docstring says `abs(...) < delta`. Output: `attacks/a1.out`.

### P03-F06 (minor) — cap warning under best-seen selection reports a negative gain and wrong advice
With `monotone=False` (or a mutable/neural leaf), a capped run warns "before the objective settled (last objective gain -5.68e+05 ...) Raise max_its to fit to convergence". The Returns docstring describes the cap case as "with the objective still improving". The receipt is honest (`final_objective` = returned best model, `last_accepted_objective` = the downhill trajectory). The corpus reproduces it on real code: `skeptic_challenge_example.py:255` (RealNVP flow, `max_its=8`) prints `last objective gain -56.6`. Torch-free reproduction in `findings.json`; output `attacks/a13_hbn_prov_neggain.out`.

### P03-F07 (docs) — "every fitted mixle distribution's `.cdf`" is false for 134 of 165 distributions
`ks_1samp`, `pit_values`, `evaluate_cdf` docstrings and the CHANGELOG say the scalar-only `cdf` accepted is what every fitted mixle distribution's `.cdf` is. `MixtureDistribution`, `CategoricalDistribution`, `CompositeDistribution`, `HiddenMarkovModelDistribution` and 130 others have no `cdf`; `ks_1samp(y, mixture.cdf)` raises `AttributeError`. The 31 that do have one are handled correctly (statistic matches scipy to machine precision; `math.erf`-based scalar callables work). Output: `attacks/a4.out`.

### P03-F08 (minor) — `propose()` multimodality note leaks raw path tuples
"field(s) () look multimodal" (scalar data -- names nothing), "field(s) (0,) ..." (tuple/DataFrame rows), "field(s) ('key', 'height') ..." (dict rows, internal `'key'` segment exposed), while the adjacent notes spell the same fields `$`, `$[0]`, `$['key']['height']`. CHANGELOG claims the caveat is issued "naming them". DataFrame column names never appear in notes. Output: `attacks/a11_propose_naming.out`.

### P03-F09 (minor) — wrong-type arguments fail with internal AttributeErrors
`rng='abc'` -> `'str' object has no attribute 'randint'`; `rng=random.Random(1)` -> `Random.randint() missing 1 required positional argument`; `prev_estimate=GaussianEstimator()` -> `'GaussianEstimator' object has no attribute 'dist_to_encoder'`; `data={'a': 1}` -> `could not convert string to float: 'a'`; `seq_initialize(rng=None|int|Generator)` -> `... has no attribute 'randint'` (optimize coerces int/Generator, the documented low-level pipeline does not). Output: `attacks/a1.out`, `attacks/a4.out`.

### P03-F10 (docs) — corpus prose contradicts the candidate's own output
`tutorials/fitting_and_estimation.ipynb`: markdown "`optimize` runs EM to convergence and returns the fitted distribution" sits directly above a cell whose executed stderr says it stopped at the cap (40) before settling; the `best_of()` cell's warning is attributed to `mixle/inference/estimation.py:2167` rather than user code (stacklevel through `best_of`). `examples/hierarchical_mixture_example.py`: docstring says the configuration "is capped by iteration count rather than by a convergence delta" but the call omits `delta=None`, so it emits the unconverged warning it describes avoiding. Executed notebooks: `executed-1-*.ipynb`, `executed-2-*.ipynb`; log `log-ex-hierarchical_mixture_example.py.txt`.

### P03-F11 (minor) — `propose()` on exactly three records trains on one row and calls the result verified
`propose([1.0, 2.0, 3.0])` passes the "at least three records" check, holds out `max(2, ...)=2`, trains on one row, and reports both candidates as "held-out mean log-density -124999991.709" with no note about the one-row training split; the degenerate-spike guard is one-sided (positive spikes only). Output: `attacks/a3.out`.

### P03-F12 (minor) — element-wise cdf fallback only on `TypeError`
`pit_values(y, lambda v: 0.5)` and `pit_values(y, lambda v: float('nan'))` are reported as "cdf values must have the same one-dimensional shape as y" -- the NaN case is the FU-04 misattribution shape again. Low impact: every real mixle `.cdf` raises `TypeError` on arrays and is handled. Output: `attacks/a4.out`.

### P03-F13 (minor, boundary) — mixture over `SequenceEstimator` without `len_estimator` fails inside `optimize()` without naming the fix
`TypeError: MixtureDistribution components must be generative probability laws; likelihood factors found at indices [0, 1]`. The guard is latent-model internals (another pass's area); reported because it is reached through `optimize()` and the message does not say "pass `len_estimator=`". All corpus notebooks that do this pass `len_estimator=PoissonEstimator()`. Output: `attacks/a8_map_cost.out`.

## Attacks that did not break anything

- **Control validation**: `max_its` 0 / -1 / 1.0 / True, `delta` <0 / nan / inf, `init_p` 0 / 1.5 / -0.1, `print_iter=-1`, `seed` -1 / 2**40, `rng` + `seed` together, unknown `fused_options` keys -- all raise a `ValueError`/`TypeError` naming the argument. `init_p=1` works. `rng=int` and `rng=np.random.Generator` are coerced as documented.
- **`delta=None` honesty**: Gaussian at its exact MLE (`prev_estimate`) runs the full 1/5/30 iterations (zero gain is accepted); a converged 2-component mixture warm-started with `delta=None` runs 20/20 and 3000/3000 on both the fused and unfused loops with no spurious rejection from float noise. Forced rejections (custom `strategy` returning a worse model) stop at 3/10 with the promised `UserWarning` ("only 3 of them ran ... compare the two") on both Gaussian and mixture; `fit_provenance().iterations=3, converged=False`. WrappedCauchy under `delta=None` rejected its first update in 11/33 seeded fits (fused loop) and warned every time; `on_step` call counts equal `iterations` in all 200 seeded fits across Weibull/GEV/WrappedCauchy/Gamma/Beta.
- **Cap warning** fires whenever `iterations == max_its` and `delta` is in force (Gaussian `max_its=1`, mixtures at 3/10/50/2000); `Model.fit` adds the "EM stopped at the iteration cap" note and records `n_iter`/`converged` for models that carry a receipt.
- **Log-likelihood cross-check**: `fit_provenance().final_objective` equals a direct `sum(model.log_density(x))` to <=6e-13 for Gaussian, categorical, Weibull, 2- and 3-component mixtures (fused and unfused, 50 and 200 iterations), auto-inferred and hand-built composites, and best-seen selection (`monotone=False`, where `final_objective` describes the returned best iterate and `last_accepted_objective` the trajectory).
- **`on_step`**: receives `EMStep(iter, model, log_density, delta)` on every iteration regardless of `print_iter`; the first step reports `delta=inf`; a callback that raises propagates its own exception; a non-callable fails with `TypeError`. `print_iter=3` with `max_its=7` prints iterations 3 and 6; `print_iter=0` prints only the converged line; `out=None` prints nothing.
- **`prev_estimate`** of the wrong family (Poisson prev, Gaussian est), wrong dimension (2-D MVG prev), wrong component count (k=2 prev, k=3 est) -- all raise a `ValueError` naming the mismatch.
- **Data shapes**: `[]`, `np.array([])`, `None` raise the documented "no observations" errors from both `optimize()` and `fit()` (each names its own entry point). One observation fits with a disclosed `variance-floored` repair carried into `fit_provenance().repairs`. `NaN`/`None` with an explicit `GaussianEstimator` raise the documented "contain N NaN entry" error; with `estimator=None` they fit an `OptionalDistribution`; `+inf` with `estimator=None` fits an `OptionalDistribution` *and* warns naming the field, as the docstring promises. `pd.Series`, `pd.DataFrame` (with and without NaN) fit correctly. Constant data fits with a disclosed variance floor. A `set` fits (3 obs). A generator *with* an explicit estimator fits all rows.
- **`propose()`**: empty / `None` / 1 / 2 rows raise clear `ValueError`s; constant float data is correctly rejected as degenerate on every candidate, falls back to the heuristic with a "no candidate could be verified" note, and `fit=True` downgrades the certificate to `attempted`; constant int/str fit a categorical at held-out 0.000; bimodal float data proposes a mixture on both `structured` and `recommended` and `fit=True` returns a `MixtureDistribution` with a converged receipt; mixed-type scalars and rows, ragged rows, NaN/all-NaN/inf, bool, 2-D arrays, 1-D int arrays, generators, one-shot iterators (`fit=True`), and dicts of columns all return a `Model` whose notes describe what was chosen; unequal column lengths, `holdout` 0/1, `max_its=0`, `seed=-1` raise; `max_candidates=0` and `timeout=0.0` record every skipped candidate. The identifier-like column gets the documented unseen-label rescue on the `recommended` candidate.
- **`describe()`**: raw data (list, ndarray, DataFrame, nested list) points at `propose()`; estimators, distributions, classes, `Model` (fitted/unfitted) describe themselves; `None`, ints, strings, dicts, lambdas, modules, bytes, sets, bools, NaN report "no catalogued capability detected" without raising. (`describe([])` says "no catalogued capability" rather than "raw data" -- cosmetic.)
- **`Model`**: `Model(GaussianDistribution)` (a class) raises the documented `TypeError`; `fit([])`, `max_its=0`, `restarts=0`, `calibrate=1.0`, `evaluate()` before `fit()` all raise with named causes; a prototype distribution's parameters are honored as the EM start (`mu=100` moves to `0.03` after one iteration and the receipt says so).
- **`pit_values` / `ks_1samp`**: scalar `.cdf` methods (Gaussian, Poisson), vectorized callables, precomputed arrays, list-returning callables and `math.erf`-based scalar callables all agree to 1e-16 with scipy; out-of-range CDF values (>1, <0, 1.0000001), NaN, wrong shape, `(n,1)` shape, empty/2-D/non-finite `y`, non-monotone CDF (ks), bad `alternative`, non-callable `cdf` all raise with named causes; a callable-signature `TypeError` propagates as `TypeError` (FU-04 holds); `ks_1samp` on one observation works; `greater`/`less` p-values match the documented `exp(-2nD^2)`.
- **`seq_estimate` / `seq_initialize`**: `p` 0 / 1.5 / -1 raise; a wrong-family estimator on encoded data raises; raw (unencoded) data raises; a mixture `prev_estimate` with the wrong component count raises; a Gaussian one-step `seq_estimate` matches `np.mean`/`np.var` to 1e-15; a `p=1e-9` draw that selects no rows is repaired from the first row exactly as the source comment documents.
- **`numerical_repairs()`**: 1200 seeded `GeneralizedParetoEstimator` fits at n=3..6: all 24 fits whose shape landed on `xi_min=-10` carry a `shape-clamped` note, zero silent clamps, one non-clamp `scale-floored-for-support` note; model repairs and `fit_provenance().repairs` agree on every fit. The 0.8.1 changelog claim holds.
- **Corpus output claims** checked against the candidate: quickstart's "held-out-verified frontier", skeptic Act 1's discovered graph and joint-vs-independence gain, lookback HMM's "1000 EM iterations with delta=None, no early stop", the fitting tutorial's AIC/BIC table, and the strategy-recovery counts in `em_and_map_strategies` all reproduce; the only prose-vs-output contradictions found are in P03-F10.

## Summary
- Counts: blocking 0, real 4 (P03-F01..F04), minor 7 (P03-F05, F06, F08, F09, F11, F12, F13), docs 2 (P03-F07, F10).
- Worst: P03-F01 -- the default `optimize(data)`/`fit(data)` path returns a `HeterogeneousBayesianNetwork` with no `fit_provenance()`/`numerical_repairs()` at all, so the receipt the release advertises as the answer to "did this fit converge?" is missing exactly where the quickstart and skeptic examples send users, while the per-factor fits underneath it hit their caps.
- Runner-up: P03-F02 -- `estimator=None` on a generator/iterator silently returns a converged-looking fit of zero observations that scores everything `-inf`.
- All 8 notebooks and 8 examples in scope executed with exit 0 on the candidate; the EM loop's convergence/early-stop disclosures fire as documented on every forced case, and log-likelihoods in receipts match direct sums to 1e-13.
- Nothing outside `<review-root>/reviews-081/pass-03/` was modified.
