# Pass 03 -- the inference loop: optimize/fit/best_of, propose/describe/Model, receipts and provenance, calibration and goodness-of-fit, seq_* utilities, parallel and accelerated engines

**Recovery note.** The original pass-03 reviewer executed the corpus, wrote probes p01-p26 and captured their outputs, then lost its session before writing this report. This report was written by the recovery reviewer from that preserved evidence (`RECOVERED_NOTES.md` is the original command trail), plus the runs the trail had left queued (p09-p13, p15, p16; `run_pending.sh`), a fixed re-run of the crashed p26 (`probes/p26b_emmap_restarts.py`), and two confirmation probes (`probes/p27_recovery_confirm.py`: the roundoff rejection on 0.8.1 vs 0.8.2, the structure-search receipts, the empty-DataFrame structure fit and `fit(vdata=[])`; `probes/p28_jointmixture.py`). No notebook or example was re-executed. The stored-vs-executed cell comparison was re-run into `nb_compare.txt` because the original diff output was not saved.

- Wheel: `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (sdist `mixle-0.8.2.tar.gz` sha256 `71ec826b2960df13f9ca4513da83eac371e25ad942a479378e47d93f1a39c037`; `env/wheel_sha256.txt`)
- Commit `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013`
- Work dir: `REVIEW_ROOT/pass-03/` (all paths below are relative to it)
- Verification: `cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, saved under `env/`; verbatim output for every environment used:

`venv-full (notebooks, examples, most probes)`:

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-nonumba`:

```
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-base`:

```
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`candidate-081 (the published 0.8.1, comparison only)`:

```
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

## Corpus executed on the candidate

Notebooks (venv-full, `run_nb.sh`, executed in place in the APFS clone `corpus/`; results `nb_results.txt`, logs beside each notebook, plain-text dumps in `executed_txt/`, stored-vs-executed cell comparison in `nb_compare.txt`):

| notebook | exit | wall |
|---|---|---|
| data_science/structured_science_context | 0 | 9 s |
| data_science/receipts_and_replay | 0 | 22 s |
| data_science/missing_data_and_imputation | 0 | 22 s |
| data_science/agentic_system_facade | 0 | 17 s |
| data_science/conformal_prediction | 0 | 20 s |
| tutorials/model_parallel_estimation | 0 | 29 s |
| tutorials/accelerated_engines | 0 | 28 s |
| data_science/conformal_uq_on_graphs | 0 | 29 s |
| data_science/verifiable_design_loop | 0 | 36 s |
| data_science/classification_metrics_and_calibration | 0 | 45 s |
| data_science/anomaly_detection | 0 | 46 s |
| data_science/bayesian_workflow_predictive_checks | 0 | 299 s |
| data_science/model_selection_and_cross_validation | 0 | 94 s |
| data_science/em_and_map_strategies | 0 | 323 s |
| tutorials/parallel_estimation | 0 | 352 s |
| tutorials/estimation_using_spark | 0 | 1543 s |

Wall times were taken with three notebooks running at once on a shared machine and are not performance evidence.

Fresh output vs stored output / prose (`nb_compare.txt`; 9 of 16 notebooks reproduce their stored outputs to 3 decimals in every cell):

- `data_science/bayesian_workflow_predictive_checks`: the prose promises 90% intervals, the cell computes 95% ones -- **Q03-F22**. Coverage 0.938 fresh vs 0.927 stored and KS p=0.26 vs 0.30 are Monte Carlo drift and support the prose either way.
- `data_science/em_and_map_strategies`: the prose calls the 60-restart log-likelihoods bimodal while the notebook's own stored and fresh output says all starts are within 1 nat -- **Q03-F23**. The strategies table reproduces exactly (24/24, 24/24, 0/24, 12/24; `probes/p25_emmap_strategies.full.out`); the two differing cells differ only in ipykernel paths inside warnings.
- `data_science/anomaly_detection`: one new line `Iteration 6: ... = 0.000000e+00` -- the cell's explicit `print_iter=100` now prints its converged line (the P08-F04 repair; the stored output predates it). Not a contradiction.
- `tutorials/accelerated_engines`, `tutorials/parallel_estimation`, `tutorials/model_parallel_estimation`: only timings, ipykernel paths and a timestamp differ; log-likelihoods identical.
- `tutorials/estimation_using_spark`: only Spark progress bars and one new `UserWarning: this mixture fit left 7 of 10 component(s) with less data than they have parameters` (the A-02 note) on the 10-component HMM mixture; numbers unchanged.
- all other notebooks: none.

Examples (venv-full, `tools/pyt.py 1800`, `PYT_CWD=examples/cwd_<name>`; outputs `examples/<name>.out`, summary `examples/results.txt`):

| script | exit | wall |
|---|---|---|
| auto_example | 0 | 25.1 s |
| quickstart_example | 0 | 83.8 s |
| model_comparison_example | 0 | 362.6 s |
| engine_benchmark_example | 0 | 137.4 s |
| scaling_example | 0 | 156.4 s |
| skeptic_challenge_example | 0 | 492.4 s |
| cross_modal_fit_receipt | 0 | 155.1 s |
| task_distill_example (P09-F09/P10-F09 ledger example, extra) | 0 | 25.0 s |
| calibrated_report_demo (P09-F09/P10-F09 ledger example, extra) | 0 | 9.1 s |

- `engine_benchmark_example`: its printed header ('10 its GMM', 'parity-checked') is contradicted by its own stderr and by a per-engine replay -- **Q03-F24**.
- `calibrated_report_demo`: at its own default `max_its=30` it prints `dependency_gain scored its candidate fits at the max_its cap (30) ...` and `learn_structure() stopped at the max_its cap (30) before the objective settled (last objective gain 0.0184, delta=1e-06) ... Raise max_its to fit to convergence.` -- the notes name the verb and drop the `delta=None` clause (R07-F06 verified); the example's claims hold.
- `skeptic_challenge_example`: prints as claimed; its stderr shows the library-built A-02 wording ('solve() built this mixture itself, so neither is set here ...', R07-F07 verified) and one rejected-update note from its classical mixture fit (`fell 2.22e-09 ... still gained 2.2e-08`), an instance of Q03-F02.
- others: none.

Regression tests copied from the source export and run against the installed wheel (`tests/run_tests.sh`, `tests/run_tests.out`): `inference_loop_contract_repairs_test.py` 32 passed on venv-full and venv-base; `adversarial_review_082_repairs_test.py -k 'DownhillCapNote or ClosedParallelHandle or CapNoteAdvice or UnidentifiedComponentRemedy or ProposeAdvice or ScalarCdfAndRaggedColumns or an_encoded_batch_is_counted or initialize_takes_what'` 18 passed on both; `notebook_surface_repairs_test.py -k PrintIter` 6 passed on both; `library_surface_repairs_test.py -k 'PairwiseScreen or DegenerateInput'` 14 passed on both.

## Findings

Severity counts: blocking 0, real 4, minor 17, docs 3. Blocking: none.

Every Python reproduction block below was extracted verbatim from `findings.json` and executed on the candidate through `tools/pyt.py` (`repro_check/run_all.sh`; outputs `repro_check/Q03-Fnn.full.out`, plus `repro_check/Q03-F07.base.out` and `repro_check/Q03-F08.base.out` for the two venv-base claims): all 23 runs exit 0 and print the values quoted under Observed.

### Q03-F01 · real · Fused EM loop stamps a fit that settled exactly at the cap converged=False and warns 'before the objective settled (last objective gain 0)'; P03-F04 extended this to composites and the auto route, which reported converged=True on 0.8.1

- **Surface:** mixle.inference.optimize (fused-em route: every closed-form family, CompositeEstimator, CategoricalEstimator, PoissonEstimator, WeibullEstimator, the candidates optimize(rows) fits on the auto-structure route), Model.fit, propose(max_its=...); fit_provenance().converged / objective_gain; the max_its-cap UserWarning
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import warnings, numpy as np
from mixle.inference import optimize, fit
from mixle.stats import GaussianEstimator, CompositeEstimator, CategoricalEstimator
vals = [float(v) for v in np.random.RandomState(0).normal(3, 2, 300)]
rows = [(str(c), v) for c, v in zip(np.random.RandomState(1).choice(['a', 'b', 'c'], 300), vals)]
warnings.simplefilter('always')
for label, f in (("optimize Gaussian max_its=2", lambda: optimize(vals, GaussianEstimator(), max_its=2)),
                 ("fit Gaussian max_its=2", lambda: fit(vals, GaussianEstimator(), max_its=2)),
                 ("optimize(rows) auto max_its=2", lambda: optimize(rows, max_its=2)),
                 ("optimize Composite max_its=2", lambda: optimize(rows, CompositeEstimator([CategoricalEstimator(), GaussianEstimator()]), max_its=2))):
    fp = f().fit_provenance()
    print(label, "it=%s/%s converged=%s gain=%s alg=%s" % (fp.iterations, fp.max_iterations, fp.converged, fp.objective_gain, fp.algorithm))
g = optimize(vals, GaussianEstimator())
print("converged warm start, max_its=1:", optimize(vals, GaussianEstimator(), prev_estimate=g, max_its=1).fit_provenance().converged)
```

- **Observed:** 0.8.2 (probes/p22_zero_gain_cap.full.out): optimize Gaussian max_its=2 -> it=2/2 converged=False gain=0.0 alg=fused-em, with UserWarning 'optimize() stopped at the max_its cap (2) before the objective settled (last objective gain 0, delta=1e-09): the returned model is an unconverged fit ... Raise max_its to fit to convergence'; fit Gaussian max_its=2 -> it=2/2 converged=True gain=0.0 alg=em, no warning; optimize Composite max_its=2 and optimize(rows) max_its=2 -> converged=False gain=0.0 alg=fused-em, whereas on 0.8.1 (p22_zero_gain_cap.081.out) the same two calls took the 'em' loop and reported converged=True. A converged warm start re-run with max_its=1 -> converged=False gain=0.0 (fit(): converged=True). With max_its=3 the same fits report it=3/3 converged=True: one extra iteration is spent to notice. Same shape for Categorical, Poisson and Weibull. probes/p05_capnotes.full.out: optimize(rows, max_its=2) and Model().fit(rows, max_its=2) emit the gain-0 cap note for a candidate model while fit(rows, max_its=2) and learn_bayesian_network(rows, max_its=2) emit none. probes/p16_model.full.out and p22: Model(MixtureEstimator(...)).fit(data, max_its=2) with restarts='auto' emits 13-21 such notes, most reading 'last objective gain 0'.
- **Expected:** A run whose last objective gain is 0 (below delta) is a converged fit: converged=True and no cap note, as fit() reports for the identical call and as optimize() itself reports when granted one more iteration. A note must not assert 'before the objective settled' while printing 'last objective gain 0'.
- **Notes:** Mechanism: mixle/inference/estimation.py _fused_em_loop; its docstring (lines 1305-1313) documents that the convergence test 'lags the standard loop by one iteration'. At the cap the exhausted exit (lines 1374-1391) scores the final model and writes final_objective and objective_gain (0.0) into the receipt but never re-evaluates `converged`, so the receipt carries gain=0.0 and converged=False together and the cap note fires. Regression component: P03-F04 routed prior-free CompositeEstimator, and therefore optimize(rows)/Model.fit(rows) on tabular data, from the 'em' loop onto 'fused-em', so a cap that reported converged=True on 0.8.1 reports converged=False on 0.8.2. Not ledgered. Concerns P03-F04, P03-F06 (cap-note wording), P09-F09/P10-F09/R07-F06 (cap-note routing).
- **Repairs concerned:** P03-F04, P03-F06, P09-F09, P10-F09, R07-F06

### Q03-F02 · real · Monotone acceptance gate uses an absolute 1e-12 tolerance: a converged fit whose fused log-likelihood dips by a few ulps is stopped on a 'rejected update', stamped converged=False and told 'More of the same update would not help'

- **Surface:** mixle.inference.optimize (fused-em route) with engine=NUMPY_ENGINE / FUSED_NUMPY_ENGINE / NumpyEngine(dtype=float32) / TorchEngine (float32, mps), and the default route at n>=200000; fit_provenance().converged; the rejected-update UserWarning; examples/engine_benchmark_example.py
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import warnings, numpy as np
from mixle.inference import optimize
from mixle.stats import MixtureEstimator, MixtureDistribution, GaussianEstimator, GaussianDistribution
from mixle.engines import NUMPY_ENGINE
est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
for seed in range(6):
    rng = np.random.RandomState(seed)
    data = np.concatenate([rng.normal(-4, 1, 10000), rng.normal(4, 1, 10000)]).tolist()
    for label, kw in (("NUMPY_ENGINE", dict(engine=NUMPY_ENGINE)), ("unfused", dict(reuse_estep_ll=False))):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fp = optimize(data, est, max_its=30, prev_estimate=init, out=None, **kw).fit_provenance()
        print(seed, label, "it=%d converged=%s gain=%.3g" % (fp.iterations, fp.converged, fp.objective_gain),
              [str(x.message)[:140] for x in w if 'rejected' in str(x.message)])
```

- **Observed:** probes/p27_recovery_confirm.full.out (a): NUMPY_ENGINE seeds 1, 3, 4 -> it=5 converged=False with 'optimize() stopped at iteration 5 of max_its=30 on a rejected update: the proposal fell 3.64e-11 below the last accepted objective while the last accepted step still gained 1.34e-08 > delta=1e-09. The returned model is the last accepted one, an unconverged fit, and its fit_provenance() reports converged=False. More of the same update would not help.' The unfused loop on the same data and start: 6/6 converged=True. probes/p23_roundoff_reject.full.out: spurious rejections FUSED_NUMPY_ENGINE 3/6, 3/6, 2/6 at n=2000/20000/200000; NUMPY_ENGINE 2/6, 3/6, 1/6; default route 0/6, 0/6, 2/6; unfused fit() 0/6 everywhere; rejected amounts 3.6e-12 .. 1.2e-10, i.e. a few ulps of an objective of magnitude 4e3 .. 4e5. Float32 engines are rejected on every run at 3e-4 .. 2e-2 (probes/p21_engine_gmm.full.out: NumpyEngine(float32) it=4, torch cpu float32 it=4, torch mps it=3 'fell 0.0232 ... still gained 27.3'), and examples/engine_benchmark_example.out carries that MPS note from the example's own GMM benchmark (line 127). Present on 0.8.1 too (probes/p27_recovery_confirm.081.out: NUMPY_ENGINE seeds 1, 3, 4 converged=False; its note additionally advised 'a different initialization or restarts=...').
- **Expected:** An acceptance tolerance scaled to the objective's magnitude and the engine's dtype (a decrease of a few ulps after a step that gained 1e-8 is roundoff, not a failed update): continue to the delta test and report converged=True as the unfused loop does; at minimum the note must not claim 'More of the same update would not help' when the unfused loop converges from the same state.
- **Notes:** Mechanism: mixle/inference/estimation.py:1230 (_em_loop) and :1335 (_fused_em_loop) accept a step only when dll >= -1.0e-12, and :1251/:1352 make `converged` require acceptance, so a rejected step can never be converged. The fused E-step normalizer and the separate score pass sum in different orders, which is why the fused route sees dips the unfused route never does; float32 engines cannot pass an absolute 1e-12 gate near an optimum at all, so their fits are never 'converged' at the default delta. Not a 0.8.2 regression and not ledgered (0.8.2-followups.md has no 'rejected'/'roundoff' entry); the 0.8.2 disclosure of the rejected-step exit is what makes it visible. Consequence for the shipped example: its GMM row bills every engine for '10 its' while the numpy and torch-cpu fits converge at iteration 5 and the MPS fit stops at 3 with a different log-likelihood (Q03-F24). Concerns P03-F06, R02-F05, R07-F07.
- **Repairs concerned:** P03-F06, R02-F05, R07-F07

### Q03-F03 · real · An empty validation set is not refused: fit(data, est, vdata=[]) returns the INITIAL model stamped converged=True, optimize() silently ignores the same argument, best_of() reports a validation log-likelihood of 0.0

- **Surface:** mixle.inference.fit / optimize(reuse_estep_ll=False) / optimize (fused) / best_of / Model.fit with vdata=[] | () | np.array([]) | an empty generator | enc_vdata=[] | ()
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import warnings, numpy as np
from mixle.inference import optimize, fit, best_of
from mixle.stats import GaussianEstimator, MixtureEstimator
vals = [float(v) for v in np.random.RandomState(0).normal(3, 2, 300)]
warnings.simplefilter('always')
for label, f in (("fit vdata=[]", lambda: fit(vals, GaussianEstimator(), vdata=[])),
                 ("optimize vdata=[]", lambda: optimize(vals, GaussianEstimator(), vdata=[])),
                 ("fit no vdata", lambda: fit(vals, GaussianEstimator()))):
    m = f(); fp = m.fit_provenance()
    print(label, "mu=%.4f sigma2=%.4f it=%s converged=%s final=%.3f" % (m.mu, m.sigma2, fp.iterations, fp.converged, fp.final_objective))
print("sample mean/var", np.mean(vals), np.var(vals))
bim = [float(v) for v in np.concatenate([np.random.RandomState(0).normal(-4, 1, 300), np.random.RandomState(1).normal(4, 1, 300)])]
est = MixtureEstimator([GaussianEstimator()] * 2)
vll, m = best_of(bim, [], est, 3, 30, 0.1, 1e-6, rng=np.random.RandomState(1))
print("best_of vdata=[] validation ll:", vll)
print("fit vdata=[] final:", fit(bim, est, max_its=30, rng=np.random.RandomState(1), vdata=[]).fit_provenance())
```

- **Observed:** probes/p27_recovery_confirm.full.out (d): fit vdata=[] -> mu=2.6446 sigma2=2.3063 it=2 converged=True final=-672.510, no warning; optimize vdata=[] -> mu=3.0551 sigma2=4.0056 final=-633.836; fit without vdata -> mu=3.0551 (the sample mean), i.e. fit(vdata=[]) hands back the initialization, 39 nats below the MLE, as a converged fit. probes/p20_vdata_deep.full.out: on a 2-component mixture fit(vdata=[]) returns the iteration-1 model (final -1320.4188, weights [0.636, 0.364]) where optimize(vdata=[]) and the reference fit return -1267.3151 ([0.5, 0.5]); its receipt reads converged=True with last_accepted_objective=-1267.3151; every spelling ([], (), np.array([]), empty generator, enc_vdata=[]/()) behaves the same; best_of(vdata=[]) -> validation ll 0.0. The reproduction block's own mixture case (a different draw) shows the same shape: fit(vdata=[]) -> it=2 converged=True final_objective=-1276.205 with last_accepted_objective=-1250.123 (repro_check/Q03-F03.full.out). Identical on 0.8.1 (p20_vdata_deep.081.out, p27_recovery_confirm.081.out).
- **Expected:** An empty validation set refused by name, the way empty data and empty enc_data are since P03-F02/P03-F03 (or ignored with a warning); never a converged=True receipt on the initialization with no note, and never two verbs returning different models for the same arguments.
- **Notes:** Mechanism: mixle/inference/estimation.py _em_loop (lines 1193-1204, 1243): with a validation set the INITIAL model is made selectable with best_vll = ll_fn(enc_vdata, model) = 0.0 (an empty sum), and later iterates replace it only when best_vll < vll (strict); every iterate also scores 0.0, so the initial model is returned while trace.converged records the trajectory's convergence. _fused_em_loop selects with score >= best_score (line 1349), so there the LAST iterate wins, which is why optimize() looks right and fit() does not. Not ledgered (no 'vdata' entry in 0.8.2-followups.md). Concerns P03-F02, P03-F03 (their empty-input refusals stop at data/enc_data).
- **Repairs concerned:** P03-F02, P03-F03

### Q03-F04 · real · R02-F07 stops at learn_bayesian_network(DataFrame): learn_structure(DataFrame) still fits the column names (an empty frame becomes a 2-observation tree), and both structure learners read a Mapping as its keys and a str as its characters

- **Surface:** mixle.inference.learn_structure and learn_bayesian_network with a pandas DataFrame, a mapping of columns, a str/bytes, numpy.matrix, a 1-d ndarray / pd.Series / range / set / generator of scalars, a masked or 0-d array
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np, pandas as pd
from mixle.inference import learn_structure, learn_bayesian_network, optimize
rng = np.random.RandomState(0); n = 300
plan = rng.choice(['free', 'pro'], n); usage = np.where(plan == 'pro', rng.normal(60, 5, n), rng.normal(10, 3, n)); spend = usage * 2 + rng.normal(0, 1, n)
rows = [(str(p), float(u), float(s)) for p, u, s in zip(plan, usage, spend)]
cols = {'p': [r[0] for r in rows], 'u': [r[1] for r in rows], 's': [r[2] for r in rows]}
cases = (("empty DataFrame", pd.DataFrame(columns=['c', 'v'])), ("DataFrame p,u,s", pd.DataFrame(rows, columns=['p', 'u', 's'])),
         ("DataFrame plan,usage,spend", pd.DataFrame(rows, columns=['plan', 'usage', 'spend'])), ("mapping p,u,s", cols), ("str", 'hello world hello'))
for label, d in cases:
    for f in (learn_structure, learn_bayesian_network):
        try:
            t = f(d); print("%-26s %-22s -> %s n_observations=%s" % (label, f.__name__, t, t.fit_provenance().n_observations))
        except Exception as e: print("%-26s %-22s -> %s: %s" % (label, f.__name__, type(e).__name__, str(e)[:100]))
m = optimize(cols); print("optimize(mapping) for comparison:", type(m).__name__, m.fit_provenance().n_observations)
```

- **Observed:** probes/p27_recovery_confirm.full.out (c), p17_structure_frontdoor.full.out, p17b_dependency_gain.full.out, p02_frontdoor.full.out: learn_structure(empty DataFrame cols c,v) -> DependencyTreeDistribution(fields=1, edges=[none]) n_observations=2 (the two column names fitted as the corpus; learn_bayesian_network refuses the same frame by name); learn_structure(DataFrame p,u,s) -> fields=1, n_observations=3 while learn_bayesian_network -> fields=3, n_observations=300; learn_structure(DataFrame plan,usage,spend) -> ValueError 'structure learning requires records of one fixed width, but record 1 has 5 field(s) where record 0 has 4' (the lengths of the names 'plan' and 'usage'); learn_structure(mapping) and learn_bayesian_network(mapping) -> fields=1, n_observations=3 (the keys); learn_structure('hello world hello') and learn_bayesian_network(str) -> n_observations=17 (the characters), where optimize/fit/best_of/Model.fit/propose refuse a str by name; np.matrix -> RecursionError (learn_structure) / TypeError unhashable 'matrix' (learn_bayesian_network); 1-d ndarray, pd.Series, range, set, generator of scalars -> TypeError "object of type 'numpy.float64' has no len()"; masked array -> 'len() of unsized object'; 0-d array -> 'iteration over a 0-d array'. optimize(mapping) -> HeterogeneousBayesianNetwork with n_observations=300 for the three-column mapping (the R06-F01 route; p02's two-column mapping gives CompositeDistribution n_observations=120). 0.8.1 fits the labels the same way (p17b_dependency_gain.081.out).
- **Expected:** Both structure learners take the DataFrame/mapping conversion optimize() applies (CHANGELOG R02-F07: 'It now takes the same conversion optimize already applied'; R06-F01: a mapping of columns is materialized through column_records) and refuse str/bytes/matrix/0-d/masked/scalar-sequence inputs by name, as every other verb does since P03-F09.
- **Notes:** Mechanism: mixle/inference/structure.py line 1060 (learn_structure) `data = list(data)` before _columns (likewise lines 804 and 976): list(DataFrame) yields the column labels, list(Mapping) its keys, list(str) its characters; the P02-F07 width check then either passes (equal-length labels: a silent wrong fit) or raises about the label lengths. learn_bayesian_network received a DataFrame branch for R02-F07 but no Mapping branch and no str refusal. The targeted re-review rated R02-F07 real; this is the same defect on the sibling verb and the sibling spellings. Concerns R02-F07, R06-F01, P03-F09, P02-F07.
- **Repairs concerned:** R02-F07, R06-F01, P03-F09, P02-F07

### Q03-F05 · minor · One-shot iterators are materialized by optimize()/fit()/best_of() but not by dependency_gain(), seq_encode() or MPEncodedData, which die on a raw len() TypeError

- **Surface:** mixle.inference.structure.dependency_gain (parent or child generator), mixle.stats.seq_encode (generator data), mixle.utils.parallel.multiprocessing.MPEncodedData (generator data)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference.structure import dependency_gain
from mixle.stats import GaussianEstimator, seq_encode
par = [str(c) for c in np.random.RandomState(0).choice(['a', 'b'], 100)]; ch = [float(v) for v in np.random.RandomState(1).normal(0, 1, 100)]
for label, f in (("dependency_gain parent generator", lambda: dependency_gain((p for p in par), ch, GaussianEstimator())),
                 ("dependency_gain child generator", lambda: dependency_gain(par, (c for c in ch), GaussianEstimator())),
                 ("seq_encode generator", lambda: seq_encode((c for c in ch), GaussianEstimator().accumulator_factory().make().acc_to_encoder()))):
    try: print(label, "->", f())
    except Exception as e: print(label, "->", type(e).__name__, e)
```

- **Observed:** 0.8.2: all three -> TypeError: object of type 'generator' has no len() (probes/p17b_dependency_gain.full.out, p12_sequtils.full.out; MPEncodedData(generator) likewise in p13_parallel.full.out). 0.8.1 (p17b_dependency_gain.081.out): dependency_gain consumed a parent generator and raised 'fit() received no observations' and scored a child generator (573.41); the 0.8.2 TypeError is introduced by R02-F16's new length check.
- **Expected:** Materialize one-shot iterators the way optimize()/fit()/best_of() do since P03-F02, or refuse them by name.
- **Notes:** Concerns R02-F16 (its len()-based ragged check is where the generator now dies) and P03-F02.
- **Repairs concerned:** R02-F16, P03-F02

### Q03-F06 · minor · P03-F03's zero-rows disclosure is spelling-dependent: seq_initialize/seq_estimate warn on an empty list but not on an empty tuple, initialize([]) is silent, and a single (size, x) chunk dies on 'cannot unpack non-iterable int object'

- **Surface:** mixle.inference.seq_initialize, seq_estimate, initialize, optimize(enc_data=...) with an empty tuple of chunks, tuple(seq_encode([])), or a single (size, x) chunk
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import warnings, numpy as np
from mixle.inference import seq_initialize, seq_estimate, initialize, optimize
from mixle.stats import GaussianEstimator, GaussianDistribution, seq_encode
est = GaussianEstimator(); enc = seq_encode([], est.accumulator_factory().make().acc_to_encoder())
for label, e in (("list", enc), ("tuple", tuple(enc)), ("()", ())):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); r = seq_initialize(e, est, np.random.RandomState(0), 0.5)
    print("seq_initialize", label, "->", r, "warnings:", len(w))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); r = initialize([], est, np.random.RandomState(0), 0.5)
print("initialize([]) ->", r, "warnings:", len(w))
for f in (lambda: seq_initialize((0, np.array([])), est, np.random.RandomState(0), 0.5), lambda: optimize(None, est, enc_data=(0, np.array([])))):
    try: f()
    except Exception as ex: print(type(ex).__name__, ex)
```

- **Observed:** probes/p12_sequtils.full.out: seq_initialize/seq_estimate on the empty list -> GaussianDistribution(0.0, 1e-08) with the P03-F03 warning 'received an encoded batch with zero rows ...'; on tuple(enc) or () -> the same fabricated model with NO warning; initialize([], ...) -> GaussianDistribution(0.0, 1e-08), no warning; a single (size, x) chunk -> 'TypeError: cannot unpack non-iterable int object' from seq_initialize, seq_estimate and optimize(enc_data=...). optimize(enc_data=()) is refused by name (R02-F10 verified).
- **Expected:** The zero-rows disclosure on every spelling of an empty batch (R02-F10 made optimize count tuples; seq_initialize/seq_estimate/initialize did not follow), and a bare chunk either accepted or refused by name.
- **Notes:** Concerns P03-F03, R02-F10, R02-F13.
- **Repairs concerned:** P03-F03, R02-F10, R02-F13

### Q03-F07 · minor · engine=/precision= spellings fail on raw AttributeError/TypeError, and without numba engine=FUSED_NUMPY_ENGINE dies with ModuleNotFoundError instead of the P07-F15 refusal

- **Surface:** mixle.inference.optimize(engine='numpy'|'torch'|'fused'|3, precision='bogus'); optimize(engine=FUSED_NUMPY_ENGINE) and fused_options= on venv-nonumba / venv-base
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator, MixtureDistribution, GaussianDistribution
from mixle.engines import FUSED_NUMPY_ENGINE
mix = MixtureDistribution([GaussianDistribution(-2, 1), GaussianDistribution(2, 1)], [0.5, 0.5]); data = mix.sampler(seed=1).sample(2000)
est = MixtureEstimator([GaussianEstimator()] * 2)
for eng in ('numpy', 'torch', 'fused', 3, FUSED_NUMPY_ENGINE):
    try: print(repr(eng), "->", optimize(data, est, prev_estimate=mix, max_its=2, engine=eng, out=None).fit_provenance().algorithm)
    except Exception as e: print(repr(eng), "->", type(e).__name__, str(e)[:120])
try: optimize(data, est, prev_estimate=mix, max_its=2, precision='bogus', out=None)
except Exception as e: print("precision='bogus' ->", type(e).__name__, e)
```

- **Observed:** probes/p24_engine_followup.{full,nonumba,base}.out and p14_engines.full.out: engine='numpy'/'torch' -> AttributeError: 'str' object has no attribute 'to_numpy'; engine='fused'/'x'/3 -> AttributeError: ... has no attribute 'name'; precision='bogus' -> TypeError: data type 'bogus' not understood. On venv-nonumba and venv-base, engine=FUSED_NUMPY_ENGINE (and any fused_options= with it) -> ModuleNotFoundError: No module named 'numba', where NumbaKernelFactory().build(...) on the same environments says 'numba is not installed, so a numba kernel cannot be built (install the extra: pip install mixle[numba]). Use GeneratedNumbaKernelFactory ...' (P07-F15).
- **Expected:** engine= and precision= refused by name with the accepted objects/dtypes listed, and the fused engine without numba declined with the P07-F15 message rather than an import error from inside the loop.
- **Notes:** Concerns P07-F15 (the sibling route that was repaired).
- **Repairs concerned:** P07-F15

### Q03-F08 · minor · precision='float32' (and engine=NumpyEngine(dtype=float32)) is a silent no-op without numba while float16 is applied

- **Surface:** mixle.inference.optimize(precision='float32'), optimize(engine=NumpyEngine(dtype=np.float32)) on venv-nonumba and venv-base
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator, MixtureDistribution, GaussianDistribution
mix = MixtureDistribution([GaussianDistribution(-2, 1), GaussianDistribution(2, 1)], [0.5, 0.5]); data = mix.sampler(seed=1).sample(5000)
est = MixtureEstimator([GaussianEstimator()] * 2)
for p in ('float64', 'float32', 'float16'):
    m = optimize(data, est, prev_estimate=mix, max_its=5, precision=p, out=None)
    print(p, [repr(c.mu) for c in m.components])
```

- **Observed:** venv-full (numba): float64 mu=-2.0008092790678464, float32 mu=-2.000809263921426 (differs at the 8th digit), float16 mu=-2.0. venv-nonumba and venv-base: float32 mu=-2.0008092790678424, identical to float64 to all 16 digits (also with engine=NumpyEngine(dtype=np.float32)), while float16 -> -1.9992766533604764 is applied (probes/p24_engine_followup.{full,nonumba,base}.out).
- **Expected:** Either float32 arithmetic on the pure-numpy kernels too, or a note that the requested precision is not honoured on this path; not a silent no-op for one dtype and an applied cast for another.
- **Notes:** Reproduced on both numba-less environments; the mechanism in the generic kernel path was not traced.
- **Repairs concerned:** none

### Q03-F09 · minor · Structure-search receipts read converged=False, final_objective=None and is_approximate()=True for every completed search, and Model.fit(rows) records converged=False

- **Surface:** fit_provenance() of HeterogeneousBayesianNetwork (optimize(rows), fit(rows), learn_bayesian_network) and DependencyTreeDistribution (learn_structure); FitProvenance.is_approximate(); Model.fit(rows)._fit_info
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize, learn_structure
from mixle import Model
rng = np.random.RandomState(0); n = 300
plan = rng.choice(['free', 'pro'], n); usage = np.where(plan == 'pro', rng.normal(60, 5, n), rng.normal(10, 3, n)); spend = usage * 2 + rng.normal(0, 1, n)
rows = [(str(p), float(u), float(s)) for p, u, s in zip(plan, usage, spend)]
for m in (optimize(rows), learn_structure(rows)):
    fp = m.fit_provenance(); print(type(m).__name__, fp.converged, fp.iterations, fp.max_iterations, fp.delta, fp.final_objective, fp.is_approximate())
print(Model().fit(rows)._fit_info)
```

- **Observed:** probes/p27_recovery_confirm.full.out (b), p01_ledger_replay.full.out, p05_capnotes.full.out: both receipts read converged=False iterations=1/1 delta=None final_objective=None objective_gain=None and is_approximate() -> True; Model().fit(rows)._fit_info == {'n': 300, 'n_iter': 1, 'converged': False} with notes == []; the skeptic example prints the search's BIC (25952.8 vs 35868.2) that the receipt leaves as None. FitProvenance's docstring defines converged=False as 'stopped without the gain ever falling below delta', which cannot describe a run with no delta.
- **Expected:** A completed search reports a truthful status (converged=True with delta=None, or a search-specific field), its objective (the BIC it maximized), and is_approximate() False unless a factor was actually capped or repaired.
- **Notes:** Concerns P03-F01 (the receipt that repair added).
- **Repairs concerned:** P03-F01

### Q03-F10 · minor · The receipt's seed field is recorded only through the seed= alias: rng=<int> leaves fit_provenance().seed None

- **Surface:** mixle.inference.optimize(rng=<int>) vs optimize(seed=<int>); FitProvenance.seed
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
data = [float(v) for v in np.random.RandomState(0).normal(3, 2, 100)]
est = MixtureEstimator([GaussianEstimator()] * 2)
a = optimize(data, est, max_its=5, rng=3); b = optimize(data, est, max_its=5, seed=3)
print([c.mu for c in a.components] == [c.mu for c in b.components], a.fit_provenance().seed, b.fit_provenance().seed)
```

- **Observed:** True None 3 (probes/p07_rng.full.out): the same integer through rng= produces the same model but a receipt with seed=None.
- **Expected:** The receipt records the integer seed whichever alias carried it.
- **Notes:** Concerns P03-F09 (rng normalization) and the receipt work of P03-F01.
- **Repairs concerned:** P03-F09, P03-F01

### Q03-F11 · minor · optimize(enc_data=<MPEncodedData>) receipts carry n_observations=None although the handle knows its length

- **Surface:** mixle.inference.optimize with enc_data=MPEncodedData(...); FitProvenance.n_observations
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import io, numpy as np
from mixle.inference import optimize, seq_initialize
from mixle.stats import *
from mixle.utils.parallel.multiprocessing import MPEncodedData
true = MixtureDistribution([CompositeDistribution((GaussianDistribution(-2.0, 1.0), PoissonDistribution(3.0))), CompositeDistribution((GaussianDistribution(2.0, 1.0), PoissonDistribution(8.0)))], [0.5, 0.5])
data = list(true.sampler(seed=1).sample(3000)); est = MixtureEstimator([CompositeEstimator((GaussianEstimator(), PoissonEstimator()))] * 2)
if __name__ == "__main__":
    with MPEncodedData(data, estimator=est, num_workers=2) as enc:
        start = seq_initialize(enc, est, np.random.RandomState(7), 1.0)
        print(len(enc), optimize(None, est, enc_data=enc, prev_estimate=start, max_its=3, out=io.StringIO()).fit_provenance().n_observations)
    print(optimize(data, est, prev_estimate=start, max_its=3, out=io.StringIO()).fit_provenance().n_observations)
```

- **Observed:** 3000 None, then 3000 for the local route (probes/p13_parallel.full.out): the parallel handle reports len 3000 but its receipt records n_observations=None (algorithm 'em'); the mp and local fits agree to 0.0 in log-likelihood.
- **Expected:** n_observations=3000 on both routes; R02-F10 closed the same gap for the tuple-of-chunks spelling.
- **Notes:** Concerns R02-F10, R06-F02.
- **Repairs concerned:** R02-F10, R06-F02

### Q03-F12 · minor · objective='map' with no prior and objective='vb' with no variational family run plain EM but the receipt records 'map'/'vb'

- **Surface:** mixle.inference.optimize(objective='map'|'vb') on a prior-free estimator; FitProvenance.objective
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator
vals = [float(v) for v in np.random.RandomState(0).normal(0, 1, 300)]
for obj in ('mle', 'map', 'vb'):
    m = optimize(vals, GaussianEstimator(), objective=obj); fp = m.fit_provenance()
    print(obj, fp.objective, fp.algorithm, fp.final_objective, sum(m.log_density(x) for x in vals))
```

- **Observed:** mle -> objective 'mle', fused-em, -441.1708; map -> objective 'map', em, -441.1708 (equal to the plain log-likelihood; no prior anywhere); vb -> objective 'vb', em, -441.1708 (probes/p06_objective.full.out; the reproduction block's own draw prints -425.8919 for all three, repro_check/Q03-F12.full.out). The receipt labels a MAP or VB fit that ran plain EM on the plain likelihood.
- **Expected:** objective='map' without a prior and objective='vb' without a variational family refused by name, or recorded as 'mle' -- P03-F04's own rule is that the receipt names the objective actually maximized.
- **Notes:** Concerns P03-F04.
- **Repairs concerned:** P03-F04

### Q03-F13 · minor · best_of() silently coerces max_its=0/-1/True to one iteration and trials=0/-3/True to a run, and dies on a raw TypeError for trials=1.5, where optimize() refuses the same max_its spellings by name

- **Surface:** mixle.inference.best_of(max_its=..., trials=...)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import best_of, optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
data = [float(v) for v in np.random.RandomState(0).normal(3, 2, 200)]; est = MixtureEstimator([GaussianEstimator()] * 2)
for mi in (0, -1, True):
    print("best_of max_its=%r -> iterations" % mi, best_of(data, None, est, 2, mi, 0.1, 1e-6, rng=np.random.RandomState(0))[1].fit_provenance().iterations)
for tr in (0, -3, True, 1.5):
    try: print("best_of trials=%r -> ok, iterations" % tr, best_of(data, None, est, tr, 3, 0.1, 1e-6, rng=np.random.RandomState(0))[1].fit_provenance().iterations)
    except Exception as e: print("best_of trials=%r ->" % tr, type(e).__name__, e)
try: optimize(data, est, max_its=0)
except Exception as e: print("optimize max_its=0 ->", e)
```

- **Observed:** best_of max_its=0/-1/True -> runs 1 iteration (with two cap notes); trials=0/-3/True -> runs; trials=1.5 -> TypeError: 'float' object cannot be interpreted as an integer; optimize(max_its=0) -> 'optimize(): max_its must be a positive integer, got 0' (probes/p04_controls.full.out).
- **Expected:** best_of validates max_its and trials the way optimize validates max_its (P03-F05 class).
- **Notes:** Concerns P03-F05.
- **Repairs concerned:** P03-F05

### Q03-F14 · minor · Cap notes routed through Model.fit still open with 'optimize()', and a restarts='auto' fit at a small cap emits one note per internal trial (21 for one call)

- **Surface:** mixle.Model.fit (restarts=None and restarts='auto') cap notes; R07-F06's verb naming
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import warnings, numpy as np
from mixle import Model
from mixle.inference import fit
from mixle.stats import GaussianEstimator, MixtureEstimator
data = [float(v) for v in np.concatenate([np.random.RandomState(0).normal(-3, 1, 200), np.random.RandomState(1).normal(3, 1, 200)])]
for label, f in (("fit", lambda: fit(data, MixtureEstimator([GaussianEstimator()] * 3), max_its=2, rng=np.random.RandomState(0))),
                 ("Model.fit restarts=None", lambda: Model(MixtureEstimator([GaussianEstimator()] * 3)).fit(data, max_its=2, restarts=None)),
                 ("Model.fit restarts=auto", lambda: Model(MixtureEstimator([GaussianEstimator()] * 3)).fit(data, max_its=2))):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); f()
    caps = [str(x.message) for x in w if 'max_its cap' in str(x.message)]
    print(label, len(caps), caps[0][:60] if caps else None)
```

- **Observed:** fit -> 1 note 'fit() stopped at the max_its cap ...'; Model.fit restarts=None -> 1 note 'optimize() stopped at the max_its cap ...'; Model.fit restarts='auto' -> 21 notes, all reading 'optimize()' or 'best_of()' (probes/p05_capnotes.full.out, p16_model.full.out). The notes do land on the caller's line (P09-F09 verified).
- **Expected:** R07-F06's rule applied to the lifecycle verb: the note names Model.fit and the knobs Model.fit takes; one summary note per Model.fit call rather than one per internal restart trial.
- **Notes:** Concerns R07-F06, P09-F09, P10-F09.
- **Repairs concerned:** R07-F06, P09-F09, P10-F09

### Q03-F15 · minor · Model.fit(calibrate=True) on fewer than 8 records reports status 'not_applicable' with reason 'no calibration holdout requested'

- **Surface:** mixle.Model.fit(calibrate=True | fraction) evidence['calibration'] on small n
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle import Model
from mixle.stats import GaussianEstimator
for n in (5, 7, 8):
    m = Model(GaussianEstimator()).fit([float(v) for v in np.random.RandomState(0).normal(size=n)], calibrate=True, restarts=None)
    print(n, type(m.calibration).__name__, m.evidence.get('calibration'))
```

- **Observed:** n=5 and n=7: calibration None and evidence {'status': 'not_applicable', 'n_holdout': 0, 'reason': 'no calibration holdout requested', ...} although calibrate=True was passed (also with calibrate=0.5 at n=7); n=8: CalibrationReport, status 'succeeded', n_holdout=2 (probes/p19_calibration.full.out).
- **Expected:** A reason that names the cause (too few records for a calibration holdout at this fraction) and a warning; 'no calibration holdout requested' is false.
- **Repairs concerned:** none

### Q03-F16 · minor · Model.fit(calibrate=0.99) trains the model on a single row and reports the calibration as 'succeeded'

- **Surface:** mixle.Model.fit(calibrate=<large fraction>)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle import Model
from mixle.stats import GaussianEstimator
m = Model(GaussianEstimator()).fit([float(v) for v in np.random.RandomState(0).normal(size=100)], calibrate=0.99, restarts=None)
print(m._fit_info['n'], m.evidence['calibration'], m.fitted)
```

- **Observed:** n=1: the returned model is fitted on one row (a variance-floored point mass) and the evidence reads status 'succeeded' with n_holdout=99 (probes/p19_calibration.full.out); propose() refuses exactly this single-row training since P03-F11.
- **Expected:** Refuse a holdout that leaves fewer training rows than the estimator can identify, the way propose() does.
- **Notes:** Concerns P03-F11 (sibling route).
- **Repairs concerned:** P03-F11

### Q03-F17 · minor · HeterogeneousBayesianNetwork and DependencyTreeDistribution, the returns of optimize(data) and learn_structure, have no to_json/to_dict/from_json

- **Surface:** mixle.stats.HeterogeneousBayesianNetwork, mixle.inference.structure.DependencyTreeDistribution; to_json / to_dict / from_json
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize, learn_structure
from mixle.stats import dump_models, load_models
rng = np.random.RandomState(0); n = 300
plan = rng.choice(['free', 'pro'], n); usage = np.where(plan == 'pro', rng.normal(60, 5, n), rng.normal(10, 3, n)); spend = usage * 2 + rng.normal(0, 1, n)
rows = [(str(p), float(u), float(s)) for p, u, s in zip(plan, usage, spend)]
for m in (optimize(rows), learn_structure(rows)):
    print(type(m).__name__, hasattr(m, 'to_json'), hasattr(m, 'to_dict'), type(load_models(dump_models([m]))[0]).__name__)
```

- **Observed:** HeterogeneousBayesianNetwork False False HeterogeneousBayesianNetwork; DependencyTreeDistribution False False DependencyTreeDistribution (probes/p18_hbn_serial.full.out, p08_receipts.full.out): m.to_json() raises AttributeError on the flagship optimize(data) return while every other fitted family answers; dump_models/load_models, pickle and Model.deploy (model.json carrying fit_provenance) do round-trip both with their receipts.
- **Expected:** The to_json/from_json surface the other fitted families have, or a docstring/explain() line naming the routes that work.
- **Notes:** Not ledgered (the to_json entries in 0.8.2-followups.md concern ChowLiuTree, P02-F02). Concerns P03-F01.
- **Repairs concerned:** P03-F01

### Q03-F18 · minor · seq_initialize/initialize refuse rng=None while optimize(rng=None) resolves to a fixed seed, and the CHANGELOG's R02-F13 bullet says seq_initialize takes None

- **Surface:** mixle.inference.seq_initialize(enc, est, None, p), initialize(data, est, None, p); CHANGELOG 0.8.2 R02-F13
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize, seq_initialize, initialize
from mixle.stats import GaussianEstimator, seq_encode
data = [float(v) for v in np.random.RandomState(0).normal(3, 2, 100)]; est = GaussianEstimator()
print(optimize(data, est, rng=None).fit_provenance().converged)
for f in (lambda: seq_initialize(seq_encode(data, est.accumulator_factory().make().acc_to_encoder()), est, None, 0.5), lambda: initialize(data, est, None, 0.5)):
    try: print(f())
    except Exception as e: print(type(e).__name__, e)
```

- **Observed:** optimize(rng=None) fits; seq_initialize(..., None, ...) and initialize(..., None, ...) -> TypeError '...(): rng must be a numpy.random.RandomState, a numpy.random.Generator, or an integer seed, got NoneType' (probes/p07_rng.full.out, p12_sequtils.full.out). CHANGELOG 0.8.2, R02-F13: 'initialize took only a RandomState where seq_initialize takes an int, a Generator or None'. 0.8.1 refused None with a bare AttributeError, so the sentence was never true.
- **Expected:** Either None accepted with the optimize semantics on both, or the CHANGELOG sentence corrected.
- **Notes:** Concerns R02-F13, P03-F09.
- **Repairs concerned:** R02-F13, P03-F09

### Q03-F19 · minor · MPEncodedData / backend='mp' front door: worker counts 0, -1, 1.5 and True run silently, 10**6 dies on a raw TimeoutError, and a str is encoded as its characters

- **Surface:** mixle.utils.parallel.multiprocessing.MPEncodedData(data, num_workers=...), optimize(backend='mp', num_workers=...)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import *
from mixle.utils.parallel.multiprocessing import MPEncodedData
if __name__ == "__main__":
    true = MixtureDistribution([CompositeDistribution((GaussianDistribution(-2.0, 1.0), PoissonDistribution(3.0))), CompositeDistribution((GaussianDistribution(2.0, 1.0), PoissonDistribution(8.0)))], [0.5, 0.5])
    data = list(true.sampler(seed=1).sample(200)); est = MixtureEstimator([CompositeEstimator((GaussianEstimator(), PoissonEstimator()))] * 2)
    for nw in (0, -1, 1.5, True):
        print("num_workers=%r ->" % nw, optimize(data, est, max_its=2, backend='mp', num_workers=nw, out=None).fit_provenance().iterations)
    print("str data ->", len(MPEncodedData('hello', estimator=CategoricalEstimator(), num_workers=2)))
```

- **Observed:** num_workers=0/-1/1.5/True all run silently; num_workers=10**6 -> TimeoutError: parallel worker 0 timed out after 30.000s during setup; MPEncodedData('hello', ...) -> 5 rows (the characters) where optimize('hello') is refused by name (probes/p13_parallel.full.out). backend='' and backend='nope' are refused by name (verified).
- **Expected:** Worker counts validated as positive integers the way max_its/print_iter are, and P03-F09's str refusal applied at the parallel front door.
- **Notes:** Concerns R06-F02 (sibling guard on the same class), P03-F09.
- **Repairs concerned:** R06-F02, P03-F09

### Q03-F20 · minor · JointMixtureEstimator over length-less SequenceEstimators dies in optimize() with 'JointMixtureDataEncoder must implement row_count()' instead of the likelihood-factor refusal Mixture/HMM give

- **Surface:** mixle.inference.optimize with JointMixtureEstimator([SequenceEstimator(GaussianEstimator())]*2, [SequenceEstimator(GaussianEstimator())]*2)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle.inference import optimize
from mixle.stats import *
rng = np.random.RandomState(0); seqs = [list(rng.normal(0, 1, rng.randint(3, 9))) for _ in range(200)]
for est in (MixtureEstimator([SequenceEstimator(GaussianEstimator())] * 2),
            JointMixtureEstimator([SequenceEstimator(GaussianEstimator())] * 2, [SequenceEstimator(GaussianEstimator())] * 2)):
    try: optimize(list(zip(seqs, seqs)) if isinstance(est, JointMixtureEstimator) else seqs, est, max_its=2)
    except Exception as e: print(type(est).__name__, "->", type(e).__name__, str(e)[:120])
```

- **Observed:** MixtureEstimator -> TypeError: MixtureDistribution components must be generative probability laws; likelihood factors found at indices [0, 1] (SequenceDistribution) ... (the P03-F13 advice naming len_estimator); JointMixtureEstimator -> NotImplementedError: JointMixtureDataEncoder must implement row_count() for its encoded payload layout (probes/p15_p10_items.full.out). A well-formed JointMixture over Gaussians fits through optimize/fit/enc_data on both wheels (probes/p28_jointmixture.{full,081}.out).
- **Expected:** The same likelihood-factor refusal, with the len_estimator advice, on the JointMixture family.
- **Notes:** Concerns P03-F13, P10-F13 and R02-F10 (the row_count contract the error names). 0.8.1 was not compared on this spelling.
- **Repairs concerned:** P03-F13, P10-F13, R02-F10

### Q03-F21 · minor · Model('x') and Model.fit(out='x') fail on raw AttributeErrors while the class spelling Model(GaussianEstimator) is refused by name

- **Surface:** mixle.Model(spec=<str>), Model.fit(out=<str>)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```python
import numpy as np
from mixle import Model
from mixle.stats import GaussianEstimator
d = [float(v) for v in np.random.RandomState(0).normal(size=50)]
for f in (lambda: Model('x').fit(d), lambda: Model(GaussianEstimator).fit(d), lambda: Model(GaussianEstimator()).fit(d, out='x')):
    try: f()
    except Exception as e: print(type(e).__name__, str(e)[:110])
```

- **Observed:** Model('x') -> AttributeError: 'str' object has no attribute 'accumulator_factory'; Model(GaussianEstimator) -> TypeError 'Model spec must be a distribution or estimator INSTANCE ...'; fit(out='x') -> AttributeError: 'str' object has no attribute 'write' (probes/p16_model.full.out).
- **Expected:** Both wrong spellings refused by name like the class spelling is.
- **Repairs concerned:** none

### Q03-F22 · docs · bayesian_workflow_predictive_checks: the prose promises 90% intervals and a 0.90 hit-rate, the cell computes 95% intervals and prints 'target 0.95'

- **Surface:** notebooks/data_science/bayesian_workflow_predictive_checks.ipynb, section 3 (coverage cell)
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```text
# Read the markdown before the coverage cell against the cell (executed_txt/data_science_bayesian_workflow_predictive_checks.txt lines 110-125):
#   markdown: "## 3. Calibration: do 90% intervals contain the truth 90% of the time?" ... "record whether the central 90% credible interval
#             contains the truth. Over many replications that hit-rate should be ~0.90"
#   code:     lo, hi = np.percentile(draws, [2.5, 97.5])   # 95% credible interval
#             print('95%% credible-interval coverage over 400 replications: %.3f (target 0.95)' % np.mean(hits))
```

- **Observed:** The prose and the code disagree on the interval level (90% vs 95%). Fresh output 0.938 (stored 0.927), KS p=0.26 (stored 0.30): Monte Carlo drift, both consistent with a calibrated posterior.
- **Expected:** Prose and code agree on the interval level.
- **Notes:** Corpus text, not the wheel.
- **Repairs concerned:** none

### Q03-F23 · docs · em_and_map_strategies: the prose says the 60-restart log-likelihoods are 'bimodal, and only some runs find the good optimum' while the notebook's own output prints 'fraction of starts within 1 nat of the best optimum: 1.00'

- **Surface:** notebooks/data_science/em_and_map_strategies.ipynb, the 60-random-starts section
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```text
# Execute the notebook, or probes/p26b_emmap_restarts.py, and read the markdown above the 60-restart cell against the cell output:
#   markdown (executed_txt/data_science_em_and_map_strategies.txt line 546): "... from 60 random initializations and look at the
#             distribution of final log-likelihoods - it is bimodal, and only some runs find the good optimum."
#   stored and fresh output: "fraction of starts within 1 nat of the best optimum: 1.00"
```

- **Observed:** probes/p26b_emmap_restarts.full.out: the 60 finals span -1824.593 .. -1824.355 (0.24 nats); all 60 fits stop at the max_its=60 cap with converged=False (0/60 converged, 60/60 cap notes, last gains ~1e-3). The stored corpus output already said 1.00.
- **Expected:** Prose that matches the notebook's own numbers (a unimodal spread within a quarter nat, all runs unconverged at the cap), or a workload that actually shows the bimodality.
- **Notes:** Same class as the 0.8.1 P03-F10 (corpus text vs the candidate's own run output), on a different cell.
- **Repairs concerned:** none

### Q03-F24 · docs · engine_benchmark_example bills its GMM row as '10 its' and 'parity-checked' equal work, but its fits converge at 5 (numpy, torch-cpu) and stop at 3 on a rejected update (torch-mps) with a different log-likelihood

- **Surface:** examples/engine_benchmark_example.py (docstring line 14, header line 215, footer line 223) and its printed output
- **Reproduction** (run from /tmp with a venv's python; `REVIEW_ROOT/tools/pyt.py 600 <python> <file>`):

```text
# Run examples/engine_benchmark_example.py (examples/engine_benchmark_example.out) and probes/p21_engine_gmm.py, which replays its GMM
# workload (n=20000, init means (-1, 1), max_its=10) per engine and records the iterations each engine actually ran.
```

- **Observed:** The example prints 'EM fits (10 its GMM / 5 its default-HMM)' and 'Interpret only this synchronized, parity-checked run', and its docstring says 'The fits are budgeted (a fixed max_its) so the timings compare equal work'; its own stderr carries 'optimize() stopped at iteration 3 of max_its=10 on a rejected update: the proposal fell 0.0232 below the last accepted objective while the last accepted step still gained 27.3' from line 127 (the GMM row on torch-mps) and the HMM row's 'stopped at iteration 4 of max_its=5 ... fell 0.000477'. The replay (probes/p21_engine_gmm.full.out): numpy and torch-cpu float64 converge at iteration 5 (final -42042.8828); torch-mps stops at iteration 3 with final -42042.8680.
- **Expected:** Either delta=None so every engine runs the stated budget (the example's own stated intent), or a header that states the iterations each engine actually ran.
- **Notes:** The library-side cause is Q03-F02. Concerns P03-F06's disclosure only as the messenger.
- **Repairs concerned:** P03-F06

## Attacks that did not break anything

Every item below is backed by a probe output in `probes/` (0.8.2 = venv-full unless stated; base = venv-base; 081 = candidate-081).

- **P03-F01..F13 ledger reproductions replayed** (`p01_ledger_replay.{full,base,081}.out`): every reproduction behaves as the CHANGELOG claims on 0.8.2 (full and base identical apart from the pandas case base cannot run) and differently on 0.8.1: HBN carries fit_provenance()/numerical_repairs() (F01); generators/iterators/map objects are materialized, n_observations=200, and an empty generator is refused by name (F02); optimize(enc_data=<empty>) is refused for list/tuple/() spellings and seq_initialize/seq_estimate warn (F03; but see Q03-F06); prior-free Composite/Optional/mixture-of-composites report objective 'mle' and take fused-em (F04); delta=0 refused (F05); the downhill cap note names the BEST iterate and last_accepted_objective (F06); multimodality notes use `$`, `$[0]`, `$['key']['height']` (F08); rng='abc', random.Random, prev_estimate=<estimator>, a scalar mapping, seq_initialize rng=None/3/Generator are named (F09); propose on 3 records refused with a count that works (F11); pit_values takes a scalar cdf and refuses NaN (F12); a Mixture over length-less sequences is refused with the len_estimator advice (F13).
- **Front-door spellings through optimize/fit/best_of/Model.fit/propose** (`p02_frontdoor.{full,base,081}.out`): str, bytes, numpy.matrix, masked arrays and 0-d arrays are refused by name on all five verbs (0.8.1 fitted str/bytes as categoricals over characters and recursed on matrix); structured arrays, dict-of-columns, DataFrame, Series, range, set, tuple rows, list-of-lists, 2-D arrays, None-inside rows all fit with the right family and n_observations; empty generator/empty DataFrame refused; Model.evaluate refuses the same spellings.
- **Control spellings** (`p04_controls.{full,081}.out`): delta=0, 0.0, -0.0, np.float32(0), np.float64(0), True, '0', inf refused by name on optimize/fit/best_of/Model.fit (0.8.1 accepted the zeros); subnormal 1e-320 accepted and echoed; init_p 0/1.5/-0.1/True/'x' refused; P08-F04: an explicit print_iter=1/2/np.int64(2) prints to stdout without out= on optimize, fit, best_of, Model.fit and the auto-structure route, print_iter=0 and None print nothing, True/-1/1.5/'1' refused (0.8.1 printed nothing for any of them and refused None).
- **Cap notes** (`p05_capnotes.{full,081}.out`): notes land on the caller's line (P09-F09) and name fit()/best_of()/propose()/learn_structure() (R07-F06); learn_structure's per-candidate notes are the budget-matched dependency_gain wording; learn_bayesian_network and fit(rows) emit none at max_its=2; the track_best=False downhill note says LAST iterate and last_accepted_objective is None (R02-F05); schedule='auto' block-EM receipts consistent with delta=None/1e-9.
- **Objective labels** (`p06_objective.{full,081}.out`): Categorical with pseudo_count None/0/1/0.5 -> 'mle' fused-em; Composite[Cat(pc=1), Gauss] 'mle' fused-em (0.8.1: 'map' em); Gauss(prior) -> 'map' with final_objective = LL + log prior; forced 'mle' on a prior estimator honoured.
- **rng/seed spellings on every route** (`p07_rng.{full,base,081}.out`): int/np.int64/np.uint8/Generator/RandomState/None accepted; bool refused (0.8.1 accepted True); float/str/list named; -1/2**32/2**63 refused (numpy's message); rng+seed together refused; rng=3 == seed=3 == RandomState(3) models; propose(seed=) validates its own range.
- **Receipts survive persistence** (`p08_receipts.full.out`, `p18_hbn_serial.full.out`): fit_provenance()/numerical_repairs() (incl. 'variance-floored', 'shape-clamped') survive pickle, to_json/from_json, dump_models/load_models and Model.deploy/Model.load (model.json carries fit_provenance; manifest lists format, hashes) for Gaussian, mixture, composite, HBN, DependencyTree.
- **HBN routes** (`p09_hbn.full.out`): optimize/fit/learn_bayesian_network/Model.fit on rows and on a DataFrame give the same edges and n_observations=400 (R02-F07 for learn_bayesian_network); empty DataFrame refused by name; NaN/inf fields handled (inf named as a missingness indicator); ragged rows refused with the width named; a NaN record and an unseen label score -inf on log_density and seq_log_density alike (R02-F09); a constant-given-parent factor fits with the perfect-separation warning.
- **propose()** (`p10_propose.full.out`): R02-F11 split-refusal arithmetic re-fed for holdout 0.25..0.99 and n=3..20 -- every advised count fits; 'lower holdout' offered only where holdout binds; P10-F01 dependency screen reports the ledger's planted (0,)-(1,) 0.699-bit hint and quickstart's structured candidate; propose(iterator) == propose(list); the fitted receipt is present.
- **pit_values / ks_1samp callable contract** (`p11_cdf.{full,base}.out`): scalar, list-1, (1,), (1,1), np.float64, int, bool answers accepted; None/NaN/inf/out-of-range/wrong-length/(n,1) refused by name; a scalar-only cdf returning n values named (R02-F15); real Gaussian/Poisson cdfs and precomputed values fine; y edge shapes refused by name; mixtures have no .cdf (P03-F07 docs repair consistent).
- **seq_* utilities** (`p12_sequtils.{full,base}.out`): tuple-of-chunks and list-of-chunks both count n_observations=100 (R02-F10); num_chunks/chunk_size 0/-1 refused; init p 0/1.5/-0.1/True/'x'/None named; initialize accepts int/Generator/RandomState (R02-F13); mixture/prev_estimate shape mismatch named.
- **Parallel handles** (`p13_parallel.full.out`): a closed MPEncodedData is refused by name on seq_initialize, seq_estimate, seq_log_density_sum, optimize(enc_data=...) warm and cold (R06-F02); mp and local log-likelihoods agree to 0.0; backend='' and 'nope' refused with the registered names; cap notes through backend='mp' land on the caller's line.
- **P10-F13 refusals** (`p15_p10_items.full.out`): project() refuses a target with more components than draws, n_samples=0, delta=0; empirical_kl_divergence names a zero-row batch; Mixture and HMM over length-less sequences refused with the len_estimator advice; the A-02 note through a caller-built estimator advises the estimator knobs.
- **Model lifecycle** (`p16_model.full.out`): fit notes on capped fits; restarts 0/-1/1.5/True refused, 'auto'/None/2/np.int64(2) accepted; calibrate 1.0/-0.1/1/'x'/nan refused, True/False/0.0/0.25 accepted; evaluate on scalars, [], str, NaN, None, nested lists and before fit refused by name; explain()/describe()/repr as documented; fit(delta=0)/max_its=0/rng='x'/seed+rng/bogus kwargs refused.
- **Calibration surfaces** (`p19_calibration.{full,081}.out`): calibration_report on [], NaN, inf, 2-D, str refused by name; ECE binary path; conformal helpers' signatures present.
- **Validation sets that are not empty** (`p20_vdata_deep.{full,081}.out`): vdata=[nan]/[None]/'abc' refused by name; vdata=[1e300] -> 'EM did not produce a model with a finite validation objective'; a one-row vdata selects by validation score as documented.
- **Engines** (`p14_engines.{full,nonumba,base}.out`, `p24_engine_followup.*`): legacy, NUMPY_ENGINE, FUSED, NumpyEngine(float32), TorchEngine agree in log-likelihood to 6e-4 (float32) / exactly (float64) from a fixed start; fused_options unknown keys refused; NumbaKernelFactory without numba declines by name and GeneratedNumbaKernelFactory falls back (P07-F15); the 60k-row auto-fusion gate takes fused-em with or without an explicit engine.
- **em_and_map_strategies replays** (`p25_emmap_strategies.full.out`, `p26b_emmap_restarts.full.out`): the strategy-recovery table reproduces (24/24, 24/24, 0/24, 12/24) and the 'too hot' annealing run is the one that hits the 150-iteration cap with gain 46.7, as the stored output shows; the restart cell's number (1.00) reproduces.
- **JointMixture** (`p28_jointmixture.{full,081}.out`): a well-formed JointMixtureEstimator fits through optimize (fused-em), fit, reuse_estep_ll=False, enc_data and delta=None with identical receipts on both wheels.

## What was not covered

- R07-F07's README `solve(teacher, inputs)` one-liner was not replayed as written: the trail's attempt (`p15`) passed float draws and was refused ('solve() handles text or record inputs'); the library-built A-02 wording was seen only through `skeptic_challenge_example`'s stderr.
- HBN per-factor repair aggregation: `p09` shows `repairs=()` on a network whose spend|plan factor has a constant residual, but no probe looked inside the factors to establish whether a floor was applied and dropped from the aggregate; not pursued.
- The 0.8.1 comparison of Q03-F02 covers only the n=20000 NUMPY_ENGINE/default/unfused sweep (`p27`); the n=2000/200000 sweep and the float32/torch engines (`p21`, `p23`) were run on 0.8.2 only (the 0.8.1 venv has no numba, so FUSED_NUMPY_ENGINE cannot run there).
- The regression tests were run against the wheel (all green on full and base) but their logic was not re-read for vacuity by the recovery reviewer; the original reviewer read `inference_loop_contract_repairs_test.py` and left no note.
- The original reviewer's subagent review of every notebook's prose against fresh numbers was lost with the session; the recovery reviewer re-ran the cell-level comparison for all 16 notebooks (`nb_compare.txt`) and read the prose only around the seven notebooks whose cells changed. `estimation_using_spark`'s prose was not read.
- Q03-F20's spelling was not compared on 0.8.1; Q03-F08's mechanism (which generic-kernel path drops the float32 request) was not traced.
- `Model.deploy` artifact contents beyond the file listing and `fit_provenance` key (`p18`), the torch/MPS engines beyond the GMM replay (`p21`), and weighted rows (optimize has no weights argument; `p12`) were not attacked further.

DONE 03 24
