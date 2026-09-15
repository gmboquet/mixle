# Pass 05 -- production and MLOps: fit_with_provenance / Header / verify_lineage, Registry, Service, drift and Monitor, the lifecycle Model, dump_models/load_models, to_json/from_json, model_hash, checkpoint/resume, the task and DOE surfaces

**Recovery note.** The reviewer originally assigned to this pass executed the corpus, wrote probes P01-P15 and captured their outputs, and was killed by an API rate limit before writing this report. This report was produced from that preserved evidence (`probes/`, `out/`, `example-logs/`, `corpus/`, `RECOVERED_NOTES.md`), plus the minimal additional runs P16-P24 (all in `probes/` and `out/`) needed to finish the notebook comparison, relaunch the one probe that had crashed on its own constructor bug (P05 -> `p05b_lifecycle_fixed.py`), and confirm or reject candidate findings. Every number below comes from a file in this directory.

- **Wheel:** `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (`wheel_sha256.txt`)
- **Commit / tree:** `866078be520b22188110780be957150dc6da964c` / `7922a8c59283ecef6877104b9a7499afd52d9013`
- **Work dir:** `REVIEW_ROOT/pass-05/` (paths below are relative to it)

Environment verification (`cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, saved as `verify_env_*.txt`):

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```
```
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```
```
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```
```
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

Every probe was run through `REVIEW_ROOT/tools/pyt.py` from `/tmp` (PYTHONPATH unset), with `PYT_CWD` pointing into `scratch/`; each probe's stdout is `out/<probe>_<env>.txt`. Every probe ran on `venv-full` unless noted: P04 additionally on `venv-nonumba` and `venv-base`, P08 on `venv-base`, P24 on all three; P04, P07 (write side), P09, P12, P13, P15 and P17 also on the published 0.8.1 for before/after comparison (P07's read side, P13 (b) and P19 (b) read the 0.8.1-written artefacts under `xver/` on 0.8.2).

## Corpus executed on the candidate

Notebooks (`venv-full`, `tools/run_nb.sh`, clone at `corpus/`, three at a time under machine contention; `nb_results.txt`, `nb_results_retry.txt`, logs beside each notebook):

| notebook (data_science/) | exit | wall |
|---|---|---|
| adaptive_clinical_trials | 0 | 36 s |
| bayesian_optimization | 0 | 1310 s |
| experimental_designs | 0 | 285 s |
| gaussian_process_regression | 0 | 75 s |
| hvis_topology_and_streaming | 0 | 62 s |
| sequential_design_for_small_samples | 1 (cell timeout at 1800 s under 3-way contention, 1899 s; `sequential_design_for_small_samples.attempt1.log`) -> retried alone: 0 | 205 s |
| state_space_filtering | 0 | 143 s |
| time_series_forecasting | 0 | 60 s |

Examples (`venv-full`, `tools/pyt.py 1800`, `PYT_CWD=examples-cwd/<name>`; `example_results.txt`, `example-logs/<name>.txt`):

| script | exit | wall |
|---|---|---|
| production_example | 0 | 62.9 s |
| calibrated_report_demo | 0 | 77.5 s |
| doe_example | 0 | 360.5 s |
| capability_layer_example | 0 | 85.2 s |
| extensibility_seams_example | 0 | 86.0 s |
| task_cascade_economics_example | 0 | 139.4 s |
| label_economics_demo | 0 | 140.6 s |
| win_demo_example | 0 | 188.8 s |
| precedence_scheduling_example | 0 | 108.0 s |
| task_distill_example | 0 | 411.5 s |
| task_extraction_example | 0 | 85.2 s |

Fresh output vs stored output / prose (`probes/p11_nb_compare.py` -> `out/p11_nb_compare.txt`; `probes/p16_nb_compare_rest.py` -> `out/p16_nb_compare_rest.txt`, cell by cell on stream and text/plain outputs):

- **experimental_designs**, cell 1: stored `'Halton': 0.048`, fresh `'Halton': 0.08`; the other four designs identical. The notebook's Halton call is the only unseeded one; the number is not reproducible on either version (Q05-F12). No prose claim is contradicted.
- adaptive_clinical_trials, bayesian_optimization, gaussian_process_regression, hvis_topology_and_streaming, sequential_design_for_small_samples, state_space_filtering, time_series_forecasting: none (every code cell's fresh output equals the stored one).
- Examples: all eleven ran to completion with the numbers their prose expects (`example-logs/`); `production_example` reports `lineage verified: True`, `chain intact: True`, `git / mixle : 866078be... / 0.8.2`.

## Findings

Severity counts: 1 blocking, 7 real, 3 docs, 6 minor (17). Repair ids in parentheses are the CHANGELOG/ledger entries the finding concerns.

### Q05-F01 -- blocking -- fit_with_provenance, Monitor.update and the scoring verbs bypass the 0.8.2 fit-verb front door: a DataFrame, a mapping of columns, a str/bytes and a 2-level MultiIndex frame are silently fitted or scored as their column names / characters, and the inputs optimize() now refuses by name still crash the old way (R06-F03, R06-F06, P06-F10, P06-F06, P06-F07, P06-F09, P06-F03)

Surface: `mixle.inference.production.fit_with_provenance` (`provenance.py:_records` + `list()`), `Monitor.update` (retrains through it), `Service.score`, `Service.check_drift`, `detect_drift`, `Monitor.check`.

Reproduction (`venv-full`; full route table in `probes/p05b_lifecycle_fixed.py` section "route parity", scoring verbs in `probes/p18_recovery_misc.py` (a), cross-version in `probes/p15_xver_frontdoor.py`):

```python
import numpy as np, pandas as pd
from mixle.inference import optimize
from mixle.inference.production import fit_with_provenance, Service, detect_drift, Monitor
from mixle.stats import CategoricalEstimator, GaussianEstimator, PoissonEstimator, CompositeEstimator
rng = np.random.RandomState(0)
df = pd.DataFrame({'x': rng.normal(size=400), 'k': rng.poisson(3, 400)})
print(type(optimize(df, max_its=2)).__name__)                        # CompositeDistribution over 400 rows
m, h = fit_with_provenance(df, None, max_its=2); print(m, h.n_records)  # Categorical({'k': .5, 'x': .5}), n_records=2
print(fit_with_provenance('hello world hello', None, max_its=2)[1].n_records)   # 17 -- optimize() refuses the str
print(fit_with_provenance({'x': rng.normal(size=50).tolist(), 'k': rng.poisson(3, 50).tolist()}, None, max_its=2)[0])  # Categorical over the names
labels = rng.choice(list('abc'), 300).tolist(); cm = optimize(labels, CategoricalEstimator())
df2 = pd.DataFrame({'a': labels[:150], 'b': labels[150:]})
s = Service(cm); print(s.score(df2), s.activity[-1]['n'])            # two scores, for the column NAMES 'a' and 'b'
print(detect_drift(cm, labels, df2).score['n_current'])              # 2
print(Monitor(cm, CategoricalEstimator(), labels).update(df2)['action'])  # 'retrained' -- on the two column names
comp = optimize([(float(a), int(b)) for a, b in zip(df['x'], df['k'])], CompositeEstimator([GaussianEstimator(), PoissonEstimator()]))
Service(comp).score(df)                                              # ContractError: row 0 ... got str
```

Observed (`out/p05b_lifecycle_fixed_full.txt`, `out/p18_recovery_misc_full.txt`, `out/p03_drift_service_monitor_full.txt`, `out/p15_xver_frontdoor_082.txt`, `out/p01_provenance_full.txt`): for the same 14 inputs `optimize`, `propose`, `Model().fit` and `Model().evaluate` agree with each other and `fit_with_provenance` does not -- DataFrame -> `CategoricalDistribution` over the 2 column names, n_records=2 (optimize: `HeterogeneousBayesianNetwork` over the 60 rows); mapping of columns and mapping of Series -> Categorical over the names (optimize: HeterogeneousBayesianNetwork); bare str -> Categorical over 17 characters, bytes -> over 11 (the four other verbs: ValueError naming the verb); 2-level MultiIndex DataFrame -> `CompositeDistribution` with n_records=2 (optimize: CopulaDistribution over the rows); structured array -> `TypeError: dataset_hash/model_hash cannot canonically encode numpy.void`; numpy.matrix -> `RecursionError`; timedelta64 -> `TypeError: int() argument must be ... not 'datetime.timedelta'`; masked array -> `GumbelDistribution requires real-valued observations` (mask silently dropped); 0-d array -> `TypeError: iteration over a 0-d array`. On the scoring side: `Service(categorical).score(df2)` returns `[-1.079, -1.04]` with activity `n=2, n_unscorable=0` (the labels 'a' and 'b' scored as observations); `detect_drift(cm, labels, df2)` -> drift=True with `n_current=2`; `Monitor(...).update(df2)['action'] == 'retrained'` (the production model was refitted on the two column names); on a composite model the same frame raises `CompositeDistribution observation must be a tuple-like sequence` / ContractError `row 0 ... got str`; `fit_with_provenance(df[['x']], GaussianEstimator())` -> `could not convert string to float: 'x'`. Identical on 0.8.1 (`out/p15_xver_frontdoor_081.txt`), where `optimize('hello world hello')` was also still accepted.

Expected: the CHANGELOG (R06-F03) says "One table now gets one answer whichever verb reads it" and (R06-F06) that the migration guide's refusal list covers "a bare str/bytes into any fit verb"; the migration guide says a bare str "raises instead of being fitted as a categorical over its individual characters". `fit_with_provenance` is documented as "Fit estimator on data via EM (optimize)" and `Monitor.update` retrains through it; both should read data through the same `normalize_input` front door (DataFrame -> records, mapping -> column records, str/bytes/matrix/masked/0-d refused by name), and `Service`/`detect_drift`/`Monitor` should encode a DataFrame as its rows, as `optimize` does, or refuse it by name -- never score or retrain on its column labels.

Notes: mechanism -- `provenance.py:_records()` returns the object unchanged unless it has `.records()`, and `fit_with_provenance` does `materialized_data = list(_records(data))`; `list(DataFrame)` is its column labels, `list(mapping)` its keys, `list(str)` its characters, so `optimize()` receives a list of strings and (estimator=None) fits a categorical -- the exact P06-F10 symptom the release says is fixed. `drift.py`/`serving.py` hand the batch to `model.dist_to_encoder().seq_encode(batch)`, which iterates a DataFrame the same way. The 0.8.2 front door (`estimation.normalize_input`, R06-F03) was wired into optimize/fit/best_of/propose/Model but not into the production module. Rated blocking: a silent wrong model on the most common container through the production fit verb, a production monitor that retrains on column labels, and a repair claim that is false for this verb; the provenance header even records n_records=2 and a data hash of the column names, so the audit record certifies the wrong dataset.

### Q05-F02 -- real -- fit_with_provenance(print_iter=N) with N != 1: KeyError 'delta' after the whole fit (lineage=True), or a header whose iterations/converged fields describe only every Nth iteration (lineage=False); print_iter=0 records zero iterations (P08-F04)

Surface: `fit_with_provenance` (`provenance.py:455` `setdefault('print_iter', 1)`; `:498-503` `recs[-1]['delta']`); `Monitor.update` forwards `**kw` to it.

```python
import numpy as np
from mixle.inference.production import fit_with_provenance
from mixle.stats import GaussianEstimator, MixtureEstimator
g = np.random.RandomState(0).normal(2, 1.5, 400).tolist()
est = lambda: MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
fit_with_provenance(g, est(), seed=1, max_its=10, print_iter=3)                    # KeyError: 'delta'
m, h = fit_with_provenance(g, est(), seed=1, max_its=10, print_iter=3, lineage=False)
print(h.training['iterations'], m.fit_provenance().iterations, len(h.training['convergence']))   # 9 10 3
m, h = fit_with_provenance(g, est(), seed=1, max_its=10, print_iter=0, lineage=False)
print(h.training['iterations'], h.training.get('converged'))                       # 0 None
```

Observed (`out/p01_provenance_full.txt` section 2, `out/p12_variations_082.txt` A, `out/p18_recovery_misc_full.txt` d): print_iter 3, 7 or 0 with the default lineage=True -> `KeyError: 'delta'` raised from `fit_with_provenance` after `optimize()` completed. With lineage=False: print_iter=3 -> header iterations=9, converged=False, 3 trace records while the model's `fit_provenance().iterations` is 10; print_iter=7 -> iterations=7, 1 record; print_iter=0 -> iterations=0, converged=None, empty trace, model ran 10. print_iter=1 or unset -> 10/10, verify_lineage True. Same on 0.8.1 (`out/p12_variations_081.txt` identical in this section).

Expected: a print cadence must not change the provenance record -- every accepted iteration is recorded (or the argument is refused with a message) and the header's iterations/converged agree with the model's `fit_provenance()` for any print_iter; no KeyError.

Notes: `fit_with_provenance` installs its `_EMHistory` collector as `out=` and only `setdefault()`s print_iter=1, so a caller's print_iter reaches `optimize()` and throttles the `em_record` hook (loglik/delta) to every Nth iteration, while the `on_step` hook (model-hash chain) fires every iteration; the records are merged by iteration, so with lineage=True the last record has lineage fields but no `delta` and `recs[-1]['delta']` raises; with lineage=False only the Nth records exist and `iterations` is read from the last one. 0.8.2 made an explicit print_iter a first-class control (P08-F04) without covering this route; the docstring mentions out= but not print_iter=. Not a wrong model, but the fit is lost to the KeyError or the audit record undercounts what ran. (A user-supplied `out=` alone works as documented: the trace is simply not captured.)

### Q05-F03 -- real -- the automatic-structure route (estimator=None on tabular records) never calls on_step: Registry.checkpointer writes nothing, and fit_with_provenance returns a header with lineage_status='recorded', iterations=0, an empty trace, verify_lineage False and the caller's max_its recorded although the structure search ran its own internal fits

Surface: `optimize`/`fit(data, None, on_step=...)`, `Model().fit(tabular, on_step=...)`, `Registry.checkpointer`, `fit_with_provenance(tabular, None)`.

```python
import numpy as np, tempfile
from mixle.inference import optimize
from mixle.inference.production import Registry, fit_with_provenance, verify_lineage
rng = np.random.RandomState(0); d = rng.normal(0, 1, 300).tolist() + rng.normal(6, 1, 300).tolist()
tab = [(float(x), int(abs(x)), 'ab'[i % 2]) for i, x in enumerate(d)]
calls = []; m = optimize(tab, None, max_its=50, delta=None, out=None, on_step=lambda s: calls.append(s.iter))
print(type(m).__name__, calls, m.fit_provenance().iterations)        # HeterogeneousBayesianNetwork [] 1
reg = Registry(tempfile.mkdtemp()); optimize(tab, None, max_its=3, delta=None, out=None, on_step=reg.checkpointer('auto')); print(reg.versions('auto'))  # []
mm, h = fit_with_provenance(tab, None, max_its=50, delta=None)
print(h.training['lineage_status'], h.training['iterations'], len(h.training['convergence']), h.training['max_its'], verify_lineage(h))  # recorded 0 0 50 False
```

Observed (`out/p12_variations_082.txt` D, `out/p19_auto_route_and_081_dirichlet_full.txt` a): `optimize(tab, None, on_step=cb)`, `fit(tab, on_step=cb)`, `Model().fit(tab, on_step=cb)`, `fit_with_provenance(tab, None, on_step=cb)`: cb never called; the same data with an explicit CompositeEstimator, or scalar data with estimator=None, calls it every iteration. Checkpointer on the automatic route: `versions == []`. `fit_with_provenance(tab, None, max_its=50, delta=None)`: `lineage_status='recorded'`, iterations=0, convergence=[], max_its=50, `verify_lineage(h)=False`, while the returned model's `fit_provenance().iterations == 1`. The 0.8.1 registry written by `probes/p07_write_081.py` shows the same for its `auto_tabular` entry (chain `[]`, verify_lineage False; `out/p07_write_081.txt`, `out/p07_read_082.txt`).

Expected: either the structure search's winning fit runs through the caller's optimize controls (on_step/out/max_its/delta) so checkpoints and the trace exist, or the automatic route states that it ignores them: the header should say `lineage_status='not_recorded'` (as it does for a user out=) and not record a max_its the fit did not use; the checkpointer should not silently write nothing.

Notes: `estimation.optimize` takes the structure='auto' branch when estimator is None and the records are tabular (`_maybe_structured_model`); the winner comes back from the structure search's internal `optimize()` calls (out=None, their own caps) and the caller's on_step/out never reach an EM loop. `fit_with_provenance` sets lineage_status from what it INSTALLED, not from what fired, so an empty trace is labelled recorded and `verify_lineage` then returns False, which reads as tampering. Adjacent to P06-F05 but not ledgered.

### Q05-F04 -- real -- under a fixed iteration budget the model handed to on_step at iteration k -- and so every Registry.checkpointer snapshot -- is the previous iterate: the chain tip is one EM step behind the model optimize()/Model.fit() return and carries its iteration number, while the same call with monotone=False reports the post-step models

Surface: `optimize(on_step=...)` / `EMStep` under the default acceptance policy with `delta=None` (or a max_its cap that ends the run), `Registry.checkpointer(name, every=1)`, `Model.fit(on_step=...)`; `fit_with_provenance`'s convergence trace.

```python
import numpy as np, tempfile
from mixle.inference import optimize
from mixle.inference.production import Registry
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.data.hashing import model_hash
rng = np.random.RandomState(0); d = rng.normal(0, 1, 300).tolist() + rng.normal(6, 1, 300).tolist()
est = lambda: MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
last = {}
m = optimize(d, est(), seed=0, delta=None, max_its=4, out=None, on_step=lambda s: last.update(iter=s.iter, model=s.model))
print(last['iter'], model_hash(last['model'])[:8], model_hash(m)[:8])            # 4 6d5572b0 b36689fe -- different models
print(model_hash(optimize(d, est(), max_its=1, delta=None, out=None, prev_estimate=last['model'])) == model_hash(m))  # True: returned = last step + one EM step
m2 = optimize(d, est(), seed=0, delta=None, max_its=4, out=None, monotone=False, on_step=lambda s: last.update(model=s.model))
print(model_hash(last['model']) == model_hash(m2))                                  # True under monotone=False
reg = Registry(tempfile.mkdtemp()); m3 = optimize(d, est(), seed=0, delta=None, max_its=4, out=None, on_step=reg.checkpointer('run', every=1))
print(reg.versions('run'), reg.verify_chain('run'), model_hash(reg.get('run')[0]) == model_hash(m3))   # ['v1'..'v4'] True False
```

Observed (`out/p24_final_objective_offbyone_full.txt`, `..._venv-nonumba.txt`, `..._venv-base.txt`; `out/p22_modelfit_vs_optimize_full.txt`; `out/p23_restart_notes_full.txt`; `out/p21_restarts_none_control_full.txt`; `out/p20_bo_askTell_and_restarts_full.txt` b; `out/p18_recovery_misc_full.txt` b): `optimize(600 bimodal points, 2-Gaussian mixture, seed=0, delta=None, max_its=4, on_step=cb)`: cb receives (1, hash 01e2d46a, ll -1313.8354), (2, ec6fad8f, -1266.0318), (3, 05383aca, -1265.9010), (4, 6d5572b0, -1265.8945); the returned model hashes b36689fe with ll -1265.8941, is exactly one further EM step from the iteration-4 model (`optimize(max_its=1, delta=None, prev_estimate=<iteration-4 model>)` reproduces it), and is the model an early-stopping run (max_its=200) reports at ITS iteration 5. The same call with `monotone=False` reports (1, ec6fad8f), (2, 05383aca), (3, 6d5572b0), (4, b36689fe) and returns the iteration-4 model; a run that stops on convergence (10 iterations) reports the returned model at its last step. `Registry.checkpointer(every=1)` on the fixed-budget run holds v1..v4 = the four stale iterates labelled checkpoint_iter 1..4, `verify_chain` True, and `reg.get(name)[0]` is not the returned model -- reproduced through plain `optimize` and `Model.fit(restarts=None/'auto'/1)`, identically in venv-full, venv-nonumba and venv-base. `fit_provenance().final_objective` and the header's `final_loglik` are the returned model's (correct); `fit_with_provenance`'s trace records the stale iterate as iter 4, its terminal record binds the returned model, and `verify_lineage` is True.

Expected: `EMStep`'s docstring says `model` "is the current accepted model -- snapshot it to checkpoint, and resume with optimize(prev_estimate=...)": the snapshot labelled iteration k should be the model iteration k produced, on every acceptance policy, and the last snapshot of a run should be the model returned (or the checkpointer should write a final snapshot), so that `reg.get(name)` after a fit is the model the caller was handed and both policies label the same models with the same iteration numbers.

Notes: mechanism measured, not fully traced -- the plain EM loop at `estimation.py:1230-1257` reports the ACCEPTED model of the iteration (`model = nxt`, then `on_step(EMStep(i + 1, model, ...))`) and `monotone=False` runs through it; the default policy evidently runs a loop whose report precedes its M-step, so iteration 1 reports the initial model and the final M-step's product is returned unreported. Effect size per step is one EM update (4e-4 nats here, near convergence; larger early in a run); the harm is the silent mismatch between what was checkpointed and what was returned, plus the iteration label. `fit_with_provenance` is designed around a returned model that differs from the last step ("possibly restored-best" terminal record), which is why its lineage still verifies. Not ledgered; neither the 0.8.1 pass-05 report nor the targeted re-review tested tip == returned.

### Q05-F05 -- real -- Registry.register and Registry.checkpointer accept models that have no readable JSON form, writing versions and checkpoint chains that get/current/Service.from_registry/verify_chain can never read, while dump_models refuses the same models up front (P05-F17, P05-F02)

Surface: `Registry.register / checkpointer / get / current / verify_chain`, `Service.from_registry`; families VonMises, WrappedNormal, WrappedCauchy, Kent, VonMisesFisher, Bingham (`mixle.stats.directional`), Wishart, MatrixNormal (`mixle.stats.matrix`), IntegerBernoulliSet (`mixle.stats.sets`), KnowledgeGraph, RandomDotProductGraph (`mixle.stats.graphs`).

```python
import numpy as np, tempfile
from mixle.inference import optimize
from mixle.inference.production import Registry
from mixle.stats import VonMisesEstimator, dump_models
data = np.random.RandomState(0).vonmises(0, 2, 200).tolist(); m = optimize(data, VonMisesEstimator(), max_its=5)
try: dump_models(m)
except Exception as e: print('dump_models:', type(e).__name__)      # SerializationError, refused up front
reg = Registry(tempfile.mkdtemp()); print(reg.register(m, 'm'), reg.versions('m'))   # v1 ['v1'] -- accepted
reg.get('m')                                                         # SerializationError: ... does not match its constructor-owned schema
optimize(data, VonMisesEstimator(), max_its=3, delta=None, out=None, on_step=reg.checkpointer('ck', every=1)); print(reg.versions('ck'))  # ['v1','v2','v3']
reg.verify_chain('ck')                                               # SerializationError (not False)
```

Observed (`out/p13_writeonly_registry_082.txt`, `out/p02_family_sweep_full.txt`, `out/p02b_family_sweep2_full.txt`, `out/p14_writeonly_families_082.txt`): VonMises -- `dump_models` -> SerializationError "dump_models produced JSON that load_models cannot read back"; `register` -> 'v1', `promote` -> None; `get`/`current`/`Service.from_registry` -> SerializationError "serialized state ... does not match its constructor-owned schema; define __pysp_setstate__"; the checkpointer writes v1..v3 and `verify_chain` raises SerializationError. The same register-ok/get-fails pattern for WrappedNormal, WrappedCauchy, Kent, VonMisesFisher, Bingham, IntegerBernoulliSet, Wishart, MatrixNormal, KnowledgeGraph, RandomDotProductGraph. DirichletMultinomial is refused at `register()` (its FitReceipt class is not registered) -- a clean refusal. 0.8.1 behaves the same for VonMises and additionally for Dirichlet (`out/p13_writeonly_registry_081.txt`); Dirichlet is repaired in 0.8.2. The 64 + 21 other fitted families in the sweeps register, load back hash-equal, and round-trip bit-exactly through dump_models/to_json.

Expected: `register()` and `checkpointer()` should apply the read-back guard `dump_models` already applies (or consult the serialization manifest) and refuse with the same message, so a registry never holds a version that cannot be served and a checkpoint chain that cannot be resumed; `verify_chain` should return False or a registry error rather than leak a SerializationError.

Notes: `registry.register` writes `to_serializable(model)` into the version record without attempting `from_json` on it; `dump_models` does the read-back check. The directional/matrix modules being write-only is disclosed in the 0.8.0/0.8.1 CHANGELOG and P05-F17 covered to_json/dump_models/deploy (refused loudly, pickle fallback); the registry path is the surface with no refusal. `manifests/serialization_schema_manifest.json` lists these classes among the constructor-validated schemas.

### Q05-F06 -- real -- detect_drift, Service.check_drift, Monitor.check and Monitor.update crash with TypeError 'unhashable type' on dict-record and set-record models with the default per_feature=True

Surface: `detect_drift` / `Service.check_drift` / `Monitor.check` / `Monitor.update` on `RecordDistribution` (RecordEstimator, DictRecordEstimator), `BernoulliSetDistribution`, `IndianBuffetProcessDistribution`.

```python
import numpy as np
from mixle.inference import optimize
from mixle.inference.production import detect_drift, Service, Monitor
from mixle.stats import DictRecordEstimator, GaussianEstimator, PoissonEstimator, BernoulliSetEstimator
rng = np.random.RandomState(0)
rows = [{'a': float(a), 'b': int(b)} for a, b in zip(rng.normal(size=400), rng.poisson(3, 400))]
est = DictRecordEstimator({'a': GaussianEstimator(), 'b': PoissonEstimator()}); rm = optimize(rows, est)
print(Service(rm).score(rows[200:])[:2])                                    # works
print(detect_drift(rm, rows[:200], rows[200:], per_feature=False).drift)   # works
detect_drift(rm, rows[:200], rows[200:])                                   # TypeError: unhashable type: 'dict'
Monitor(rm, est, rows[:200]).check(rows[200:])                             # TypeError: unhashable type: 'dict'
srows = [set(rng.choice(list('abcde'), 2, replace=False).tolist()) for _ in range(100)]; sm = optimize(srows, BernoulliSetEstimator())
detect_drift(sm, srows[:50], srows[50:])                                   # TypeError: unhashable type: 'set'
```

Observed (`out/p03_drift_service_monitor_full.txt` B, `out/p02_family_sweep_full.txt`, `out/p02b_family_sweep2_full.txt`, `out/p15_xver_frontdoor_082.txt`): RecordEstimator and DictRecordEstimator models -- `detect_drift(dict rows)` -> TypeError unhashable type: 'dict'; `Monitor.check/update` -> same; `per_feature=False` -> drift=False, ks=0.065; `Service.score(dict rows)` -> [-2.826, -4.85]. BernoulliSet and IndianBuffetProcess models: TypeError unhashable type: 'set'. Identical on 0.8.1 (`out/p15_xver_frontdoor_081.txt`).

Expected: per-feature PSI on record models should read fields by key (and set models by membership), or per_feature should degrade to the score-only report with a note; a default-argument crash on the record types the model itself was fitted from is not acceptable.

Notes: the per-feature pass hashes each observation (or its fields) to bin it; a dict/set row is unhashable. P05-F06 repaired ndarray batches on this same path; dict and set records were not exercised by that repair or its test. Not ledgered.

### Q05-F07 -- real -- Monitor.update still ends a monitoring loop on an unscorable row: check() now counts it (P05-F12) but the retrain that update() runs when drift is flagged refuses the same batch, with a message naming arguments update() does not take (P05-F12)

Surface: `Monitor.update` (retrain via `fit_with_provenance`/`optimize`) on Gaussian (NaN, inf, None), Exponential (negative), and any family with a support refusal.

```python
import numpy as np
from mixle.inference import optimize
from mixle.inference.production import Monitor, detect_drift
from mixle.stats import GaussianEstimator, ExponentialEstimator
rng = np.random.RandomState(0); d = rng.normal(3, 1, 300).tolist(); gm = optimize(d, GaussianEstimator())
ref = d[:200]; batch = ref[:20] + [float('nan')]
print(detect_drift(gm, ref, batch).drift, Monitor(gm, GaussianEstimator(), ref).check(batch).drift)   # True True (counted as unscorable)
Monitor(gm, GaussianEstimator(), ref).update(batch)   # ValueError: GaussianDistribution observations contain 1 NaN entry ... fit(..., missing='marginalize') in mixle.ppl ...
ex = rng.exponential(2, 300).tolist(); em = optimize(ex, ExponentialEstimator())
Monitor(em, ExponentialEstimator(), ex).update([v + 10 for v in ex[:100]] + [-0.5])   # ValueError: ExponentialDistribution has support x >= 0 ...
```

Observed (`out/p04_ledger_replay_082full.txt` P05-F12 block, `out/p12_variations_082.txt` E): 0.8.2 `detect_drift`/`Service.check_drift`/`Monitor.check` on a batch with one NaN -> drift=True, fraction_unscorable_current=0.048 (repaired); `Monitor.update` on the same batch -> ValueError "GaussianDistribution observations contain 1 NaN entry -- missing values. Drop the incomplete rows, or model the gaps: fit(..., missing='marginalize') in mixle.ppl, or wrap the leaf with mixle.stats.marginalized()". Also `update([.., inf])` -> "requires support x in (-inf,inf)", `update([.., None])` -> TypeError float() ... 'NoneType', Exponential `update(shifted + [-0.5])` -> ValueError support x >= 0, with and without `combine_reference`. 0.8.1 raised `UnscorableObservation` from check() and update() alike (`out/p04_ledger_replay_081.txt`).

Expected: the P05-F12 ledger entry names `Monitor.check/update` and says "A Monitor loop dies on the first NaN in production data"; the CHANGELOG says an unscorable record is counted "instead of ending a monitoring loop". `update()` should drop the rows `check()` already classified as unscorable before retraining (or refuse the batch before the drift check, in its own words), and its message should not advise `fit(missing=...)` / `marginalized()`, which `update()` cannot pass (the R06-F05 pattern).

Notes: `Monitor.update` calls `check()` (now tolerant) and, on drift, `fit_with_provenance` on reference+batch (or batch alone); the estimator's 0.8.2 support/NaN refusals then fire on the row `check()` counted. The repair covered the scoring half of the surface the ledger named. Rated real rather than blocking: the failure is loud and the drift verdict is now right; but the loop still dies where the ledger said it must not.

### Q05-F08 -- real -- the ask/tell BayesianOptimizer and propose_next do not disclose the surrogate ridging that mixle.doe.minimize now reports: BayesianOptimizer.best.numerical_repairs() is () on a run whose GP was ridged, and propose_next returns only the point (R07-F05, R02-F14, P09-F03)

Surface: `mixle.doe.BayesianOptimizer.ask/tell/best` (`optimizer.py:136-150`), `mixle.doe.propose_next`, `mixle.doe.bayesopt._record_surrogate_repairs` / `_SURROGATE_REPAIRS`.

```python
import numpy as np
from mixle.doe import BayesianOptimizer, propose_next, minimize
from mixle.models.gaussian_process import GaussianProcessRegressor
seen = []; orig = GaussianProcessRegressor._chol
def patched(self, k, eye):
    before = self._jitter_applied; out = orig(self, k, eye)
    if self._jitter_applied != before: seen.append(float(self._jitter_applied))
    return out
GaussianProcessRegressor._chol = patched
obj = lambda p: float((p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2)
bo = BayesianOptimizer([(-5.0, 5.0), (-5.0, 5.0)], seed=3)
for _ in range(16):
    for x in np.atleast_2d(np.asarray(bo.ask(1), float)): bo.tell(x, obj(x))
print(len(seen), max(seen), bo.best.numerical_repairs(), bo.best.surrogate_repairs)   # 63 3.99 () ()
r = minimize(obj, [(-5.0, 5.0), (-5.0, 5.0)], n_init=5, n_iter=11, seed=3); print(len(r.numerical_repairs()))  # 3, disclosed on this route
```

Observed (`out/p20_bo_askTell_and_restarts_full.txt` a; the 12-seed minimize sweep in `out/p08_gp_doe_full.txt`): `BayesianOptimizer(..., seed=3)`, 16 ask(1)/tell rounds on (p0-1)^2+(p1+2)^2: the default GP escalated its jitter 63 times, largest ridge 3.99 (0.1 x the mean kernel diagonal), yet `bo.best.numerical_repairs() == ()` and `bo.best.surrogate_repairs == ()`; seed=7: 49 escalations, largest 3.7, () again; no warning on either. `propose_next` on a design with five near-duplicate rows (x[:5] + 1e-9): 19 escalations, largest 0.481, returns a bare (2,) ndarray, the `_SURROGATE_REPAIRS` context variable is None afterwards. Parity reference on the same objective and seed: `minimize(..., n_init=5, n_iter=11, seed=3)` escalated 63 times, largest 3.99, and its result's `numerical_repairs()` lists 3 entries starting `covariance-ridged(3.99; beyond jitter=1e-06, to factor the GP kernel matrix)`.

Expected: the migration guide says `BayesOptResult.numerical_repairs()` "reports any covariance ridging the Gaussian-process surrogate needed during the run" and "is () when" none was needed; `BayesianOptimizer.best` returns a `BayesOptResult`, so a repaired surrogate must show there too, and `propose_next` needs some route to the record (a `return_repairs=` or a result object) -- the same way on every route.

Notes: mechanism (`bayesopt.py:417-445, 473`; `optimizer.py:136-150`) -- `_fit_surrogate` records the GP's `numerical_repairs()` into the `_SURROGATE_REPAIRS` ContextVar, which only `minimize()` sets; on the ask/tell route and in a bare `propose_next()` the ContextVar is None and the record is dropped, and `BayesianOptimizer.best` constructs `BayesOptResult` without `surrogate_repairs`, so the field is its default `()`. R07-F05 fixed exactly this loss for `minimize`; the other two public routes to the same surrogate were left as they were. The R02-F14 largest-jitter reporting itself is correct on the GP (`out/p08_gp_doe_full.txt`: 1e-2 then 1e-10 -> reported 0.01 once).

### Q05-F09 -- docs -- migration guide: "mixle models and posteriors ... are unaffected and seed exactly as before" is false -- a mixle model used as a calibrated-generator prompt seeded from the shared constant on 0.8.1 and seeds from its serialized parameters on 0.8.2 (R07-F03, P09-F09)

Surface: `docs/migrations/0.8.2.md` "Seeds derived from a prompt" (lines 95-100); `mixle.task.calibrated_generator._derive_seed/_seed_key`.

```python
import numpy as np, warnings
from mixle.task.calibrated_generator import _derive_seed
from mixle.inference import optimize
from mixle.stats import GaussianEstimator
m = optimize(np.random.RandomState(0).normal(size=100).tolist(), GaussianEstimator())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); print(_derive_seed(7, m), [str(x.message)[:60] for x in w])
# 0.8.1 (candidate-081 venv): 1523943581 == sha256('7:None') with the 'not reproducible' warning; 0.8.2: 2173324335, no warning
```

Observed (`out/p09_calibrated_seed_081.txt` vs `out/p09_calibrated_seed_full.txt`): 0.8.1 -- a fitted GaussianDistribution prompt -> seed 1523943581 (the literal `'7:None'` digest every non-canonical prompt shared) with the reproducibility warning; 0.8.2 -> 2173324335 for the same model (same for a refit and a reloaded copy), no warning. The guide's sentence says such prompts "are unaffected and seed exactly as before".

Expected: the guide should say that models and posteriors gained a canonical key in 0.8.2 (P09-F09) and therefore seed differently from 0.8.1, where they fell into the shared-constant bug (R07-F03) -- the same "draws recorded on 0.8.1 ... will not reproduce" caveat it already gives for custom classes.

Notes: the behaviour change is intended; only the sentence is wrong. Canonical str/bytes/int/float/None/bool/list/tuple/set/dict prompts seed identically on both versions (same probe).

### Q05-F10 -- minor -- the calibrated-generator seed key of a mixle model includes its fit receipt, so parameter-identical models (equal model_hash) seed differently with no warning; numpy integer scalars are non-canonical (np.int64(3) != 3) and are warned as "not reproducible across processes" although their repr is deterministic (P09-F09)

Surface: `mixle.task.calibrated_generator._seed_key / _derive_seed`.

```python
import numpy as np, warnings
from mixle.task.calibrated_generator import _derive_seed, _seed_key
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, GaussianDistribution
from mixle.data.hashing import model_hash
d = np.random.RandomState(0).normal(size=100).tolist()
m1 = optimize(d, GaussianEstimator()); m2 = optimize(d, GaussianEstimator(), max_its=1); m3 = GaussianDistribution(m1.mu, m1.sigma2)
print(model_hash(m1) == model_hash(m2) == model_hash(m3), (m1.mu, m1.sigma2) == (m2.mu, m2.sigma2))   # True True
print(_derive_seed(7, m1), _derive_seed(7, m2), _derive_seed(7, m3))                                 # three different seeds, no warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); print(_derive_seed(7, np.int64(3)), _derive_seed(7, 3), _seed_key(np.int64(3)), _seed_key(3), [str(x.message)[:70] for x in w])
```

Observed (`out/p13_writeonly_registry_082.txt` d, `out/p09_calibrated_seed_full.txt`): model_hash equal for m1/m2/m3 and parameters equal, seeds 814817323 / 1586366294 / 888214437, warnings: 0; `_seed_key(m1)` starts with the serialized object including `fit_provenance`. `np.int64(3)` -> seed 600160120 vs 229914339 for `3` and `np.float64(3.0)`, `_seed_key(np.int64(3))` is None, and the "seed derivation for a int64 prompt is not reproducible across processes" warning fires.

Expected: the CHANGELOG (P09-F09) says seed derivation "recognizes a mixle model by its serialized parameters"; the key should be the parameters the library's own `model_hash` uses (which excludes the fit receipt), so equal models seed equally; numpy scalars should map to the canonical number they represent, and a prompt whose repr IS stable should not be warned as irreproducible.

Notes: `_seed_key` encodes a model through the full serialized form, which carries `fit_provenance` (iterations, objective, seed); `model_hash` excludes that block. numpy scalars hit no branch (only ndarray and Python numbers are canonical) and fall to repr with the warning. Reproducible across processes as long as the fit is identical, hence minor.

### Q05-F11 -- docs -- the content hash of the families re-serialized in 0.8.2 changed between 0.8.1 and 0.8.2 (Dirichlet measured): a 0.8.1 deploy artifact loads with an integrity note claiming a content-hash mismatch, and a 0.8.1 header's model_hash no longer matches the model; the migration guide's "every 0.8.1 pickle, JSON model and checkpoint loads unchanged" does not disclose it (P05-F17, P09-F05, P10-F05)

Surface: `docs/migrations/0.8.2.md` lines 5-6; `mixle.data.hashing.model_hash`; `Model.load` integrity note; the Registry P05-F02 cross-check; families Dirichlet (measured), DirichletProcessMixture, MultivariateStudentT, ProbabilisticPCA, HierarchicalMixture, JointMixture, GaussianCopula, RVineCopula (same re-serialization, not measured).

```
# write on 0.8.1:  PYT_CWD=<workdir> REVIEW_ROOT/tools/pyt.py 600 <review-store>/candidate-081/venv/bin/python probes/p07_write_081.py
# read on 0.8.2:   probes/p19_auto_route_and_081_dirichlet.py (b)
```
```python
import json, pickle
from mixle.lifecycle import Model
from mixle.data.hashing import model_hash
X = 'REVIEW_ROOT/pass-05/xver'
m = Model.load(X + '/deploy_dirichlet', trust_code=True); print(m.notes)
pk = pickle.load(open(X + '/dirichlet.pkl', 'rb')); print(model_hash(pk), json.load(open(X + '/summary_081.json'))['dirichlet']['model_hash'])
```

Observed (`out/p19_auto_route_and_081_dirichlet_full.txt` b, `out/p07_read_082.txt`, `out/p13_writeonly_registry_081.txt`): `Model.load` of the 0.8.1 Dirichlet artifact on 0.8.2 -> note "integrity note: the loaded model's content hash (9cc291be62561230...) does not match the manifest's model_content_hash (81adf347369f3b01...); ... (a swapped model file, or a content-hash change between mixle versions)" plus a UserWarning; 0.8.1-recorded model_hash 81adf347... vs 0.8.2 model_hash of the unpickled model 9cc291be...; the 0.8.1 header's model_hash != model_hash(model) on 0.8.2; Gaussian's hash is identical across versions. 0.8.1 registry records for Dirichlet/VonMises are unreadable on 0.8.2 ("serialized Dirichlet state does not match its constructor parameters (extra=['alpha_ma', 'dim', 'fit_metadata', 'has_invalid', 'log_const'])") -- and were already unreadable on 0.8.1 itself, so that part is not a regression. Every other 0.8.1 registry record, checkpoint chain, deploy artifact, dump_models text and pickle written by `probes/p07_write_081.py` (23 of its 25 family entries) loads on 0.8.2 hash-equal.

Expected: the migration guide should state that the eight re-serialized families hash differently from 0.8.1 (their serialized state is now the constructor parameters), so 0.8.1 headers, manifests and any stored model_hash for them will not match on 0.8.2 and the `Model.load` integrity note is expected, not tampering.

Notes: a consequence of P05-F17/P09-F05/P10-F05 (constructor-parameter serialization changes the canonical bytes `model_hash` digests). The `Model.load` note already anticipates "a content-hash change between mixle versions"; the guide does not.

### Q05-F12 -- docs -- experimental_designs notebook: the Halton design is the only unseeded call in its comparison cell, so the printed min-distance is irreproducible; the stored 0.048 is seed 0's draw, the executed copy printed 0.08, a further run 0.101

Surface: corpus `notebooks/data_science/experimental_designs.ipynb` cell 1 (`halton_design(bounds, 24)` without `seed=`); `mixle.doe.halton_design(seed=None, scramble=True)`.

```python
import numpy as np
from mixle.doe import halton_design
from scipy.spatial.distance import pdist
b = [(0, 1), (0, 1)]
print([round(float(pdist(halton_design(b, 24)).min()), 3) for _ in range(3)])   # differs run to run
print(round(float(pdist(halton_design(b, 24, seed=0)).min()), 3))                # 0.048 == the stored output
```

Observed (`out/p17_halton_xver_082.txt`, `out/p17_halton_xver_081.txt`, `out/p11_nb_compare.txt`, `out/p16_nb_compare_rest.txt`): stored cell text `{'random': 0.027, 'Latin hypercube': 0.043, 'maximin LHS': 0.091, 'Sobol': 0.065, 'Halton': 0.048}`; executed copy `... 'Halton': 0.08` (the four seeded designs identical); a fresh unseeded call gives 0.101 on both 0.8.1 and 0.8.2 with different point sets, seed=0 gives 0.048, seed=1 0.097, scramble=False 0.119.

Expected: the notebook should pass `seed=` to `halton_design` as it does to the other four designs, so the stored number and the ranking sentence ("larger min-distance = better space-filling") are reproducible; no library change is implied (`halton_design(seed=None)` is documented as randomized).

Notes: not a library defect. The prose makes no numeric claim about Halton beyond the printed dict.

### Q05-F13 -- minor -- numpy scalar spellings are refused inconsistently across the production verbs: seed=np.int64(5) is rejected by fit_with_provenance and Monitor.update (optimize accepts it), Monitor.update(retrain=np.bool_(True)) and checkpointer(resume=np.True_) are rejected, and Registry.register(metadata={'x': np.int64(1)}) fails with the raw "Object of type int64 is not JSON serializable" (P05-F09, P05-F13, P05-F16)

Surface: `fit_with_provenance(seed=)`, `Monitor.update(retrain=, combine_reference=, seed=)`, `Registry.checkpointer(resume=)`, `Registry.register(metadata=)`.

```python
import numpy as np, tempfile
from mixle.inference import optimize
from mixle.inference.production import fit_with_provenance, Monitor, Registry
from mixle.stats import GaussianEstimator
d = np.random.RandomState(0).normal(size=300).tolist(); gm = optimize(d, GaussianEstimator())
print(type(optimize(d, GaussianEstimator(), seed=np.int64(5), max_its=2)).__name__)     # accepted
fit_with_provenance(d, GaussianEstimator(), seed=np.int64(5), max_its=2)               # ValueError: seed must be a nonnegative integer or None
Monitor(gm, GaussianEstimator(), d[:200]).update(d[200:], retrain=np.bool_(True))      # TypeError: retrain and combine_reference must be booleans
reg = Registry(tempfile.mkdtemp()); reg.checkpointer('r', resume=np.True_)              # TypeError: resume must be True or False, got np.True_
reg.register(gm, 'm', metadata={'x': np.int64(1)})                                     # TypeError: Object of type int64 is not JSON serializable
```

Observed (`out/p01_provenance_full.txt` 6, `out/p03_drift_service_monitor_full.txt` F, `out/p04_ledger_replay_082full.txt` P05-F09/F13 blocks, `out/p12_variations_082.txt` B): as in the reproduction; `checkpointer(every=np.int64(2))` and `Service(keep=np.int64(3))` ARE accepted, so the same release accepts numpy integers on two knobs and refuses them on the neighbouring ones. Same on 0.8.1 for seed/metadata.

Expected: one rule for numpy scalars across the module -- accept `np.integer` where an int is meant (as optimize, `checkpointer(every=)` and `Service(keep=)` do), accept `np.bool_` where a bool is meant or say so, and let `register()` convert numpy scalars in metadata or refuse them with a registry message rather than json's.

Notes: the validators added for P05-F09/P05-F10/P05-F16 admit `np.integer` for every= and keep= but the seed check (`isinstance(seed, int)`) and the bool checks (`isinstance(x, bool)`) do not; metadata goes to `json.dumps` unconverted.

### Q05-F14 -- minor -- assorted validation gaps and unhelpful messages on the registry, header, serving and monitor surfaces: Header.from_dict on a partial dict raises from str(); Service.score/Monitor.check on a scalar say "'float' object is not iterable"; Service.score([.., None]) raises where [.., nan] is counted unscorable; Registry.promote accepts a version whose file fails the P05-F01 integrity check; a non-UTF-8 version file and a symlink-looped alias file still surface as raw UnicodeDecodeError / OSError(ELOOP); Monitor.suggest_samples accepts any method= string silently (P05-F01, P05-F11, P05-F12, P05-F16)

Surface: `Header.from_dict/__str__`, `Service.score`, `Monitor.check`, `Registry.promote`, `Registry.get/current` on damaged files, `Monitor.suggest_samples`.

```python
import os, json, tempfile, numpy as np
from mixle.inference import optimize
from mixle.inference.production import Header, Service, Monitor, Registry
from mixle.stats import GaussianEstimator, GaussianDistribution
d = np.random.RandomState(0).normal(size=300).tolist(); gm = optimize(d, GaussianEstimator())
str(Header.from_dict({}))                          # AttributeError: 'NoneType' object has no attribute 'items'
Service(gm).score(1.0)                             # TypeError: 'float' object is not iterable
s = Service(gm); print(s.score(d[:3] + [float('nan')]), s.activity[-1]['n_unscorable'])   # [..., -inf] 1
Service(gm).score(d[:3] + [None])                  # ValueError: ... contain 1 NaN entry -- missing values ...
root = tempfile.mkdtemp(); reg = Registry(root); reg.register(GaussianDistribution(1, 1), 'm'); reg.register(GaussianDistribution(9, 1), 'm')
os.rename(os.path.join(root, 'm', 'v2.json'), os.path.join(root, 'm', 'v3.json'))
print(reg.promote('m', 'v3'))                      # None (accepted); reg.current('m', 'production') then raises the integrity failure
open(os.path.join(root, 'm', 'v1.json'), 'wb').write(b'\xff\xfe\x00garbage'); reg.get('m', 'v1')   # UnicodeDecodeError (raw)
print(np.asarray(Monitor(gm, GaussianEstimator(), d[:5]).suggest_samples([(0, 1)] * 2, 4, method='bogus', seed=0)).shape)   # (4, 2) -- LHS, silently
```

Observed (`out/p01_provenance_full.txt` 8, `out/p03_drift_service_monitor_full.txt` F, `out/p04_ledger_replay_082full.txt` P05-F01/P05-F11 blocks, `out/p12_variations_082.txt` E and G): as in the reproduction; in the P05-F11 replay a version file overwritten with bytes `\xff\xfe\x00garbage` -> `reg.get` raises `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0` (raw), and a symlink loop at `m/production.alias` -> `reg.current('m', 'production')` raises `OSError [Errno 62] Too many levels of symbolic links` (raw), whereas the same loop at `m/v5.json` is wrapped as "registry ... could not be read as a registry record". Registry-returned headers round-trip through `Header.from_dict`/`str` fine; only partial dicts break.

Expected: `Header.from_dict` should default the missing blocks (or refuse a non-header dict by name); `Service.score`/`Monitor.check` should refuse a scalar the way optimize refuses a 0-d array; a None row should be counted like NaN (it becomes NaN one frame later) or refused by row index; `promote` should read the record it points at (the P05-F01 check) so an alias can never target an inconsistent file; every damaged or foreign file, including a non-UTF-8 one and a looped alias, should get the registry message P05-F11 promises; `suggest_samples` should refuse an unknown method (its docstring lists 'lhs'/'sobol').

Notes: `suggest_samples` (`monitor.py:102-115`) tests `method == 'sobol'` and falls through to LHS for anything else; `promote` writes the alias without reading the version file; the None-row difference comes from `Service.score`'s unscorable pre-scan seeing a float NaN but not a None that only becomes NaN inside the encoder; the P05-F11 wrapper covers JSON decoding and the version-file read but not the text decoding step nor the alias-file read.

### Q05-F15 -- minor -- mixle.doe.minimize(seed=np.random.default_rng(3)) and seed='3' fail with numpy's cast error instead of a message naming seed=; GaussianProcessRegressor(jitter=True) is accepted as jitter=1.0

Surface: `mixle.doe.minimize(seed=)`, `mixle.models.gaussian_process.GaussianProcessRegressor(jitter=, noise=)`.

```python
import numpy as np
from mixle.doe import minimize
from mixle.models.gaussian_process import GaussianProcessRegressor
obj = lambda p: float((p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2)
minimize(obj, [(-5, 5), (-5, 5)], n_init=4, n_iter=3, seed=np.random.default_rng(3))   # TypeError: Cannot cast scalar from dtype('O') to dtype('int64') according to the rule 'safe'
minimize(obj, [(-5, 5), (-5, 5)], n_init=4, n_iter=3, seed='3')                        # TypeError: Cannot cast scalar from dtype('<U1') to dtype('int64') ...
print(GaussianProcessRegressor(lengthscale=1.0, amplitude=1.0, jitter=True).jitter)     # 1.0
```

Observed (`out/p08_gp_doe_full.txt`): as in the reproduction; `seed=np.int64(3)` and an int work; jitter=0/-1/nan and noise=0/-1 are refused by name. In `venv-base` `minimize`/`propose_next` refuse before any evaluation with the documented torch message (`out/p08_gp_doe_base.txt`).

Expected: `seed=` validated like the fit verbs do (int, RandomState, or refused by name; a Generator either accepted or named); a bool jitter refused like a bool print_iter is.

Notes: requires torch (`venv-full`).

### Q05-F16 -- minor -- the lifecycle Model wrapper handed to the production verbs gets three different wrong-shaped refusals and one silent acceptance: build_header(Model, data) returns a header for model_type 'Model' with model_hash None; Registry.register/dump_models/to_json/model_hash say "callable Model(...) is not registered; use register_serializable_callable()"; Service/detect_drift/Monitor raise AttributeError 'dist_to_encoder'/'log_density'

Surface: `build_header`, `Registry.register`, `Service`, `detect_drift`, `Monitor`, `mixle.stats.dump_models`, `mixle.data.hashing.model_hash` with a `mixle.Model` instead of `model.fitted`.

```python
import numpy as np, tempfile, mixle
from mixle.inference.production import build_header, Registry, Service, detect_drift
from mixle.stats import GaussianEstimator
d = np.random.RandomState(0).normal(size=300).tolist(); wm = mixle.Model(GaussianEstimator()).fit(d)
h = build_header(wm, d); print(h.model_type, h.model_hash)     # Model None  (silently)
Registry(tempfile.mkdtemp()).register(wm, 'w')                 # SerializationError: callable Model(GaussianDistribution, fitted=True) is not registered; use register_serializable_callable()
Service(wm).score(d[:3])                                        # AttributeError: 'Model' object has no attribute 'dist_to_encoder'
detect_drift(wm, d[:100], d[100:])                              # AttributeError: 'Model' object has no attribute 'log_density'
```

Observed (`out/p12_variations_082.txt` F, `out/p14_writeonly_families_082.txt` C, `out/p05b_lifecycle_fixed_full.txt`): as in the reproduction; pickle of the wrapper works; `Model.deploy/load` is the documented path for the wrapper and works.

Expected: one refusal naming the fix ("pass model.fitted, the fitted distribution, not the Model wrapper") on every production verb, and `build_header` should refuse a non-distribution rather than certify 'Model' with a None hash.

Notes: the registry/hasher message comes from the serializer treating a callable object as an unregistered callable; the wrapper is callable and not a distribution.

### Q05-F17 -- minor -- Model.fit(restarts=N) runs its restart candidates without the caller's on_step/out, so a Registry.checkpointer records only the initial fit while a kept candidate is returned; notes say "best-of-3 restart kept" but nothing says the checkpoints predate the replacement

Surface: `mixle.Model.fit(restarts=<int>, on_step=...)` -> `lifecycle.py:_refit_symmetry_broken` (`optimize(..., out=None)` / `best_of(..., out=None)` without on_step).

```python
import numpy as np, tempfile, warnings, mixle
from mixle.inference.production import Registry
from mixle.stats import GaussianEstimator, MixtureEstimator
rng = np.random.RandomState(0); d = rng.normal(0, 1, 300).tolist() + rng.normal(6, 1, 300).tolist()
est = lambda: MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
def params(m): return [(round(float(c.mu), 4), round(float(c.sigma2), 4)) for c in m.components]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    a = mixle.Model(est()).fit(d, restarts=1, max_its=4, delta=None, seed=0)
    calls = []; reg = Registry(tempfile.mkdtemp()); ck = reg.checkpointer('mf', every=1)
    b = mixle.Model(est()).fit(d, restarts=3, max_its=4, delta=None, seed=0, on_step=lambda s: (calls.append(s.iter), ck(s)))
print(a.notes, params(a.fitted)); print(b.notes, params(b.fitted), calls, reg.versions('mf'))
```

Observed (`out/p23_restart_notes_full.txt`, `out/p20_bo_askTell_and_restarts_full.txt` b, `out/p18_recovery_misc_full.txt` b): restarts=3, max_its=4, delta=None, seed=0 -- on_step fires four times (the initial fit's iterations 1..4), the registry holds v1..v4, verify_chain True; the returned model is `[(0.0388, 1.0346), (5.8714, 0.9802)]` with notes `['restarts requested: best-of-3 restart kept (log-lik +0.000)']`, whereas restarts=1 (candidate not kept) returns `[(0.0385, 1.0339), (5.8711, 0.9808)]` with the same four on_step calls. The kept candidate's trajectory never reached on_step or the checkpointer.

Expected: either forward on_step/out to the restart candidates (the winner's trajectory becoming the chain), or append to notes that the on_step/checkpoint record covers the initial fit only, so a user reading the chain knows the served model came from elsewhere.

Notes: mechanism (`lifecycle.py:392-412, 440-500`) -- after the first `optimize()` (with the caller's optimize_kw), `_refit_symmetry_broken` runs hard-partition trials and `best_of()` through `optimize(..., out=None)` with no on_step and replaces `self.fitted` when a candidate beats the initial fit by > 1e-6. The replacement itself is disclosed (notes) and `Model.fit`'s docstring says restarts record "what happened in notes"; the gap is only that the callback-side record is silently partial. Stacks with Q05-F04.

## Attacks that did not break anything

Registry (`probes/p04_ledger_replay.py` on 0.8.2 full/nonumba/base and 0.8.1; outputs `out/p04_ledger_replay_*.txt`, cross-environment diffs identical apart from temp paths):
- P05-F01: swapped, copied-over and renamed version files are refused with "registry integrity failure ... the record in 'v1.json' declares version 'v2'" on every read route (0.8.1 served the other model). A record whose `version` key is deleted and whose `record_digest` is recomputed is served -- consistent with `verify_chain`'s documented "unkeyed content digest" threat model.
- P05-F02: another family under a Gaussian header, same family with other parameters, a caller-built header for a different fit, and a model edited after `fit_with_provenance` are all refused at `get`/`current`/`Service.from_registry` with the family/model_hash message; a header with `model_hash=None`, `model_type=None`, or a non-dict header is served (nothing to check). A short-spelled `model_type='Gaussian'` is refused as a mismatch.
- P05-F03: the attached header is excluded from serialization -- dump_models/to_json write JSON, `Model.deploy` writes json with no fallback, `load_models` returns a header-less model, pickle keeps the header (0.8.1: SerializationError and a pickle fallback).
- P05-F04: `fit_request_digest` stable across identical requests, mixtures and a custom estimator subclass; sensitive to seed, `pseudo_count`, `name`, `keys`, `delta=None` vs default; no memory address in the estimator description (0.8.1 had `0x...`).
- P05-F05: `verify_chain`/`checkpointer(trust_code=)` with 'yes', 1, None, 0, 1.0, np.True_ raise the "must be exactly True" message (0.8.1 returned a false "does not verify"); True/False verify.
- P05-F06: object and float ndarray batches, ndarray reference+current, batches of 0-d arrays, `Service.check_drift`/`Monitor.check`/`Monitor.update` on ndarrays all agree with the list spelling (0.8.1 flagged drift on the ndarray spelling only).
- P05-F08: `DriftReport` is frozen; its containers are copies (a caller's dict mutated after construction does not leak in).
- P05-F09/F10/F16: `checkpointer(every=0/-1/True/1.5/'2'/None/np.True_)`, `resume='no'/1/0/None/np.True_`, `Service(keep=0/-1/True/2.5/None/'3')`, `Service.from_registry(trust_code=..., keep=0, alias='nope', bogus=1)`, thresholds `psi_threshold='0.5'/True/None/-1`, `loglik_shift_threshold=0.5`, `ks_threshold=2/nan`, `min_scorable_fraction='a'`, `unscorable_shift_threshold=-0.1` -- all refused by name on `detect_drift` and at `Monitor` construction alike (R06-F05); `every=2**40` writes nothing; `np.int64` cadences and keeps accepted.
- P05-F11: truncated, array, empty version files -> registry messages; a symlink loop at a version file -> wrapped; `versions()` ignores `vX`, `v-1`, `.bak`; `get('vX')` -> KeyError by name; empty and 'latest' alias files handled.
- P05-F12: NaN, inf, negative Poisson counts, unseen categorical labels in a drift batch are counted as unscorable and raise the unscorable-rate reason (0.8.1 raised `UnscorableObservation`); `Service.score` counts them with `-inf`.
- P05-F13: non-finite header values and metadata are refused as strict JSON; int keys become strings; a tuple schema comes back as lists.
- Registry.register with a symlinked v5 present numbers the next version v6; `names()` ignores stray files; an empty model directory has no versions.

Provenance (`probes/p01_provenance.py`, `probes/p12_variations.py`): baseline header verifies; `seed=None/0/5`, `lineage=False` and a user `out=` give `lineage_status='not_recorded'` and `verify_lineage False` (never a vacuous True); `rng=` refused with the documented message; `lineage='yes'` refused as a non-bool; generator input consumed once; `(n,1)` arrays and empty lists refused by name; a single row fits with a floored variance; `Header` round-trips through dict/JSON; `verify_lineage` on str/int/list/tuple/partial dicts returns False (P05-F16); `model.header.model_hash == model_hash(model)`; exact-zero Beta/Gamma/Weibull data refused by the estimator (the P0x support repairs) before any header exists; Exponential zeros and constant Gaussian data fit, register and reload; VonMises headers get `lineage_status='recorded'` and verify although the family has no JSON form (the chain hashes are computed from `to_serializable`).

Checkpointing (`probes/p06_checkpoint.py`): cadence offsets continue across resumes (3,6,9,12 then 14,16 with `every=2`); `resume=False` roots a new lineage after which `verify_chain` is False and `resume=True` refuses with the documented advice (by design); registering a plain model into a checkpoint name likewise unverifies the chain (documented: every version must carry lineage); a run converging before the first cadence writes no checkpoint (documented cadence; the docstring's resume recipe then raises KeyError "no versions registered"); a user callback raising mid-run leaves exactly the versions written before it and an intact chain; zero-probability symbols and a 1e-300 variance `prev_estimate` produce finite (if absurd) `log_density` metadata; `fit_with_provenance` + checkpointer in one run: checkpoint hashes equal the header's per-iteration hashes; `monotone=False` runs checkpoint the returned model (see Q05-F04 for the default policy); `best_of` takes no `on_step`/`restarts`.

Drift / Service / Monitor (`probes/p03_drift_service_monitor.py`): list, object-ndarray, float-ndarray, list-of-lists and generator batches agree; a Series on a scalar model works; MultivariateGaussian on ndarray rows with a shifted column flags the right feature; ragged, None, scalar-first and string rows are refused (composite ContractErrors name the row); single-row reference/current run; `reference is current` gives zero PSI/KS; `Service(keep=2)` bounds the log and `health(window=np.int64(1))` works; `Service.score([])` returns shape (0,) and logs an event; `availability_errors=()` refused; `Monitor.update` with `seed=` records it in the header, with `rng=` refused, with `out=None` `not_recorded`; empty reference/current refused; `suggest_samples('lhs'/'sobol', n=0)` shapes/refusals; `str(DriftReport(...))`.

Serialization sweep (`probes/p02_family_sweep.py`, `probes/p02b_family_sweep2.py`, `probes/p14_writeonly_families.py`, `probes/p05b_lifecycle_fixed.py`): 64 + 21 fitted families -> `build_header` sets `model_hash` and a finite `final_loglik`; registry round trip hash-equal; `dump_models`/`to_json` bit-exact scores, equal hashes and identical re-serialized text for every family except the write-only list in Q05-F05 and DirichletMultinomial; the eight 0.8.2-repaired families (Dirichlet, DPM, MvT, PPCA, HierarchicalMixture, JointMixture, GaussianCopula, RVineCopula) plus ChowLiu, Mixture and HMM deploy as `format=json` and reload with equal scores and hashes (P05-F17/P09-F05/P10-F05 hold); VonMises deploys as pickle with the documented note; drift and `Service.score` run on every family's own records except the dict/set-record ones (Q05-F06).

Lifecycle (`probes/p05b_lifecycle_fixed.py`): R06-F04 -- manifest `evidence_not_exported` names 'provenance header' (and 'certificate', 'calibration' for a calibrated fit) and `Model.load` notes it; route parity R06-F03 holds for optimize/propose/Model().fit/Model().evaluate on all 14 inputs (the failure is `fit_with_provenance`, Q05-F01); `Model(GaussianEstimator)` (class), `restarts=0/True`, `calibrate=1.0`, `delta=0`, `rng='x'`, unfitted `deploy`/`evaluate`, `deploy` onto a plain file are refused by name; `deploy` twice into one directory then `load` works; an edited `model.json` with a recomputed sha256 is refused for every `trust_code` spelling; `evaluate([])`/`([nan])` refused, generator evaluated.

Calibrated-generator seeds (`probes/p09_calibrated_seed.py` on both versions): R07-F03 holds -- distinct non-canonical prompts seed distinctly (0.8.1: one shared seed 1523943581 for all of them); canonical prompts unchanged between versions; `__pysp_seed_key__` honoured, `NotImplemented`/raising/unencodable providers fall back with the warning; ndarrays canonical by dtype/shape/bytes; containers holding a non-canonical prompt warn.

DOE (`probes/p08_gp_doe.py`, `probes/p20_bo_askTell_and_restarts.py`): P09-F03 holds -- `minimize` needed escalated jitter on 11 of 12 seeds (the CHANGELOG's number) and every run finished with `numerical_repairs()` listing the ridges (0.215 to 3.99) and no torch `_LinAlgError`; an indefinite or NaN matrix raises the "not numerically positive definite even after adding ..." message naming the fix; R02-F14 holds (largest jitter reported); `BayesOptResult` constructs without `surrogate_repairs` (migration-guide compatibility claim); `venv-base` refuses `minimize`/`propose_next` before any evaluation with the `mixle[torch]` message; `n_iter=0`, `n_init=1`, constant and NaN objectives (`objective_failed`), `bounds` lo>hi / lo==hi, jitter/noise 0/-1/nan validation.

Cross-version (`probes/p07_write_081.py` on 0.8.1 -> `probes/p07_read_082.py` on 0.8.2): the 25 family entries the 0.8.1 probe could fit (four of its spellings did not construct on 0.8.1) -- their 0.8.1 registry records (`get`/`current`/`Service.from_registry`), checkpoint chains (`verify_chain` True, `checkpointer` resumes), deploy artifacts (`Model.load`, pickle ones with `trust_code=True`), `dump_models` texts (Markov, Optional: rescored equal) and pickles load on 0.8.2 hash-equal, and their 0.8.1 headers verify -- except the Dirichlet/VonMises registry records (never readable on 0.8.1 either), the Dirichlet hash change (Q05-F11), and the `auto_tabular` entry whose 0.8.1 header never verified and whose checkpoint chain is empty (Q05-F03).

Regression tests (`probes/p10_run_tests.sh` -> `out/p10_run_tests.txt`; 14 test files copied to `tests_copy/` and run against the installed wheel with `-p no:randomly -m "" -n 0`): `venv-base` 333 passed, 20 skipped (torch/pandas/networkx-dependent), 765 subtests; `venv-full` 376 passed, 0 skipped, 772 subtests. No test in my area passes for a reason other than its repair as far as the outputs show; the base-environment skips cover only torch/pandas cases.

## What was not covered

- The other Bayesian-optimization routes' disclosure (`constrained_minimize`, `multiobjective`, `trust_region`, multifidelity, a caller-supplied surrogate): P08 died on `inspect.signature` of the `multiobjective` module and the recovery only re-ran the ask/tell and `propose_next` routes (Q05-F08).
- The hash change of the seven other re-serialized families across 0.8.1 -> 0.8.2 (only Dirichlet was written by the 0.8.1 probe; Q05-F11 infers the rest from the same mechanism and says so).
- `restarts='auto'` refits triggered by a suspected saddle or an iteration-capped latent fit combined with a checkpointer: only explicit `restarts=1/3` and the no-refit cases were run (Q05-F17); the exact loop in `estimation.py` producing the Q05-F04 off-by-one was measured, not traced line by line.
- A comparison of the eleven examples' printed numbers against `examples/README.md` and `manifests/` beyond exit codes and their own printed claims: the trail read the manifests but left no comparison output, and I did not redo it.
- Spark/parallel backends of the production verbs, `lineage=False` performance on large models, concurrent `Service` use, and the internal claims of `calibrated_report_demo` / the task examples beyond their successful runs.
- `sequential_design_for_small_samples` was executed under contention and once alone (exit 0, 205 s); the 1899 s first attempt is recorded, not analysed further.

DONE 05 17
