# Adversarial review of mixle 0.8.2 -- pass 06 of 10

**Focus area.** Data input surfaces and the multivariate/structured families: DataFrame / Series /
structured-array / mapping adapters, records and aliases, missing values and dtypes, the `mp` and
`local` backends, rankings and paired comparisons, graphs, copulas, directional laws, processes and
embeddings; the ten data_science notebooks and the htsne tutorial assigned in `ASSIGNMENTS.md`, the
twelve gallery/structured examples, and the repairs `P06-F01..F10`, `R06-F01`, `R06-F03`, `R06-F07`,
`P09-F12`, `P02-F03`, `P01-F13`.

**Recovery note.** The first reviewer of this pass executed the corpus, wrote probes `p01`-`p18`
and captured their outputs, and was stopped by an API rate limit before writing this report. This
report was written from that preserved evidence (`RECOVERED_NOTES.md` is the reviewer's own command
trail) plus one recovery probe (`probes/p19_recovery_bool_labels_record.py`, run on 0.8.2 and 0.8.1)
and a single retry of the one notebook that timed out. Every number below comes from a file in this
work dir.

**Wheel.** `mixle-0.8.2-py3-none-any.whl`, sha256
`e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (re-hashed with `shasum -a 256`,
matches the brief). Commit `866078be520b22188110780be957150dc6da964c`, tree
`7922a8c59283ecef6877104b9a7499afd52d9013`.

**Environment verification** (`cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`; output
kept as `verify_env.txt`), verbatim:

```
=== venv-full ===
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
=== venv-nonumba ===
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
=== venv-base ===
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
=== 0.8.1 ===
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

Every probe was run through `REVIEW_ROOT/tools/pyt.py` from `/tmp` (PYTHONPATH unset, OMP/BLAS
threads 2) and prints `mixle.__path__[0]` as its first lines; its stdout is the `out/<probe>_<env>.txt`
file cited beside each finding (`_venv-full` = 0.8.2 full env, `_venv-nonumba` = no numba,
`_venv-base` = numpy+scipy only, `_v081` = the published 0.8.1). Probe scripts are in `probes/` and
share `probes/_h.py` (`attempt(label, fn)` prints `OK <type> <repr>` or `EXC <type>: <message>` per
case). The Mac was shared with the other nine passes and the background corpus execution for the
whole session (load average 200-330 on 18 cores during the reviewer's runs, 60-90 during the
recovery runs), so every wall time below is inflated by contention and is not performance evidence.

**Work dir.** `REVIEW_ROOT/pass-06/` -- `probes/`, `out/`, `nb/` (notebook diffs, produced with
`nbdiff.py`), `tests/` (copied regression tests and their pytest logs), `corpus/` (the APFS clone
the notebooks executed in; each notebook's `<name>.log` sits beside it), `corpus-retry/` (the clone
used for the single retry), `examples-out/` and `examples-base-out/` (example stdout/stderr per
environment), `p01_table.py` (tabulates the front-door probe per verb). Nothing outside it was
modified; no git write command was run.

## Corpus executed on the candidate

### Notebooks (venv-full, `tools/run_nb.sh`, 3 at a time, 1800 s timeout; `nb_status.log`)

| notebook | exit | wall |
|---|---|---|
| data_science/directional_statistics.ipynb | 0 | 48 s |
| data_science/copulas_and_gated_mixtures.ipynb | 0 | 52 s |
| data_science/cross_modal_belief_transport.ipynb | 0 | 60 s |
| data_science/inverse_rl_maxent.ipynb | 0 | 56 s |
| data_science/latent_fields_from_proxies.ipynb | 0 | 79 s |
| data_science/knowledge_graph_completion_and_uncertainty.ipynb | 0 | 141 s |
| data_science/networks_and_community_structure.ipynb | 0 | 133 s |
| data_science/heterogeneous_mixed_type_modeling.ipynb | 0 | 961 s |
| tutorials/embedding_with_htsne.ipynb | 0 | 322 s |
| data_science/model_based_embeddings.ipynb | 1 | 1924 s (CellTimeoutError after 1800 s in the `htsne(...)`/`humap(...)` cell, load average 200-330; see retry below) |
| data_science/ranking_and_combinatorial_models.ipynb | 0 | 1776 s |

**Retry of the timed-out notebook** (alone in a fresh clone `corpus-retry/`, `run_retry_nb.sh`, 3600 s cap, load average 50-90 at start and end; `nb_retry_status.log`): `data_science/model_based_embeddings.ipynb` **exit 0 in 3014 s**. The 1800 s timeout was contention, not a hang. Fresh vs stored (`nb/diff_model_based_embeddings.txt`): 4 of 8 code cells identical; the other four differ only by (a) `htsne()`/`humap()` now emitting their own `stopped at the max_its cap (200)` note naming the caller's line where the stored run (executed on 0.8.1, its warning cites `candidate-081/.../estimation.py:2075`) had `optimize()`'s note, and (b) two diagnostics that moved in the direction the prose argues: kNN label purity of the htsne map 0.963 -> 0.988 (numeric-only baseline 0.821 unchanged), two-seed 10-NN Jaccard overlap 0.35 (27x chance) -> 0.43 (34x chance). The markdown quotes neither number, so no prose claim is contradicted.

Fresh output versus stored output (`nb/diff_*.txt`, cell-by-cell text/number comparison at 1e-6
relative): identical for copulas_and_gated_mixtures (6/6 cells), cross_modal_belief_transport (5/5),
embedding_with_htsne (5/5), heterogeneous_mixed_type_modeling (6/6), inverse_rl_maxent (7/7),
latent_fields_from_proxies (6/6), networks_and_community_structure (5/5),
ranking_and_combinatorial_models (9/9). directional_statistics differs only by one extra
`Iteration 3: ... =0.000000e+00` progress line in two cells (the fresh run reports one more EM
iteration at zero gain; every number the prose cites is unchanged).
knowledge_graph_completion_and_uncertainty differs only by blank lines in two cells. Contradictions
between a fresh output and a stored output or a prose claim: **none** among the ten notebooks that
completed on the first run, and none in the retried model_based_embeddings (see the retry paragraph above).

### Examples (`tools/pyt.py 1800`, sequential; `examples_status.log`, `examples_base_status.log`)

| script | venv-full exit | wall | venv-base exit | wall |
|---|---|---|---|---|
| gallery_combinators_example | 0 | 98.2 s | 0 | 32.9 s |
| gallery_directional_example | 0 | 93.8 s | 0 | 33.1 s |
| gallery_graphs_example | 0 | 92.7 s | 0 | 46.5 s |
| gallery_multivariate_example | 0 | 99.5 s | 0 | 47.9 s |
| gallery_processes_example | 0 | 156.8 s | 0 | 111.0 s |
| gallery_rankings_example | 0 | 126.6 s | 0 | 68.6 s |
| gallery_structured_example | 0 | 235.8 s | 0 | 50.4 s |
| copula_vine_example | 0 | 317.5 s | 0 | 65.4 s |
| heterogeneous_correctness_example | 0 | 101.6 s | 0 | 10.8 s |
| heterogeneous_representation_example | 0 | 73.7 s | 1 | 0.5 s (`ModuleNotFoundError: No module named 'torch'` at its own `import torch`; the script's docstring says it needs `mixle[torch]`) |
| shared_embedding_example | 0 | 35.5 s | 1 | 11.3 s (`ImportError: build_causal_lm requires torch.`; docstring says it needs `mixle[torch]`) |
| structured_leaves_example | 0 | 39.4 s | 0 | 11.2 s |

The ten examples that run in both environments produce byte-identical stdout in venv-full (numba)
and venv-base (pure numpy) (`diff examples-out/ examples-base-out/`, exit line excluded). Output
contradicting a stated claim: none; `copula_vine_example` reproduces its "vine recovers 100% of the
true joint-crash rate, Gaussian copula 58%" numbers (0.07736 vs 0.07698 true; 0.04492), and
`heterogeneous_correctness_example` / `structured_leaves_example` print `True` for their recovery
checks. `gallery_structured_example` emits seven `optimize() stopped at the max_its cap` warnings
from its own `max_its=1` calls (documented behaviour, stderr kept in `examples-out/`).

### Regression tests (copied to `tests/`, run there with `-p no:randomly -q -m "" -n 0`)

`tests/pytest_base.txt` (venv-base, all five copied files): 141 passed, 15 skipped, 455 subtests,
399.6 s. `tests/pytest_full.txt` (venv-full, `adversarial_review_082_repairs_test.py` +
`data_adapter_contract_repairs_test.py`): 90 passed, **3 failed**, 412 subtests, 1008 s -- the three
`ClosedParallelHandleTest` tests failed in `MPEncodedData.__init__` with `TimeoutError: parallel
worker 0 timed out after 30.000s during setup` (the mp workers did not come up within the
hard-coded 30 s under load average 200-330; the same construction with `response_timeout=600`
succeeds, `out/p09_mp_closed_venv-full.txt`). Contention, not a wheel defect; see Q06-F16 for the
knob that does not exist.

## Findings

### Q06-F01 -- `real` -- `vdata=` bypasses the 0.8.2 front door: a mapping of columns, a structured array, a single-column mapping, a timedelta64/masked/0-d array and a `str` reach the encoder raw, on `optimize`, `fit` and `best_of`

**Surface.** `mixle.inference.optimize(vdata=)`, `fit(vdata=)`, `best_of(data, vdata, ...)`.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd
from mixle.inference import optimize, best_of
from mixle.stats import GaussianEstimator, CompositeEstimator, CategoricalEstimator
rng = np.random.RandomState(0)
xs = [float(v) for v in rng.normal(size=200)]; ks = ['a', 'b'] * 100
rows = list(zip(xs, ks)); est2 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
print(type(optimize({'x': xs, 'k': ks}, est2, max_its=3)).__name__)        # data route: fits
for v in ({'x': xs, 'k': ks}, np.array(rows, dtype=[('x', 'f8'), ('k', 'U1')]), 'hello'):
    try: optimize(rows, est2, vdata=v, max_its=3)
    except Exception as e: print('vdata', type(v).__name__, '->', type(e).__name__, str(e)[:90])
for v in ({'x': xs}, np.arange(200) * np.timedelta64(1, 's'), 'hello', np.array(1.0)):
    try: print('univariate vdata', type(v).__name__, '->', optimize(xs, GaussianEstimator(), vdata=v, max_its=3))
    except Exception as e: print('univariate vdata', type(v).__name__, '->', type(e).__name__, str(e)[:80])
for v in ((r for r in rows), pd.DataFrame({'x': xs, 'k': ks})):
    try: best_of(rows, v, est2, 1, 3, 0.1, 1e-6, rng=1)
    except Exception as e: print('best_of vdata', type(v).__name__, '->', type(e).__name__, str(e)[:90])
```

**Observed** (`out/p03_vdata_venv-full.txt`): `optimize(rows, est2, vdata={'x':..,'k':..})` ->
`ContractError: CompositeDistribution.dists (row 0): expected a tuple of 2 fields (one per component
distribution), got str`; structured array -> the same with `got void`; `vdata='hello'` ->
`ContractError: ... expected a sequence of 2-tuples, got str`; on the univariate route
`vdata={'x': xs}` -> `ValueError: could not convert string to float: 'x'` (the field NAME is being
scored -- the P06-F10 defect), `vdata=timedelta64 array` -> fits silently and the run reports a
validation log-likelihood computed on whatever the encoder made of `timedelta64` values (the data
route refuses the same array by name), `vdata='hello'` -> `could not convert string to float:
'hello'`, `vdata=np.array(1.0)` -> `TypeError: len() of unsized object`, a masked array -> the NaN
message rather than the masked-array refusal. `best_of(rows, vdata=generator)` -> `TypeError: object
of type 'generator' has no len()`; `best_of(rows, vdata=DataFrame)` -> `ContractError: ... got
DataFrame`. The same inputs as `data` fit or are refused by name on every verb
(`out/p01_frontdoor_venv-full.txt`).

**Expected.** The validation set is "the same shape as data" (docstring: "Optional validation
set"); the CHANGELOG says the fit verbs normalize what they are handed before anything reads it
(P06-F06/F07/F09/F10, R06-F03). A mapping of columns, a structured array, a DataFrame or a
generator handed as `vdata` should be normalized exactly like `data`, and the newly refused inputs
should be refused by name on the `vdata` route too.

**Notes.** Mechanism: `optimize` sends `vdata` through `_data_records_for_encoding`
(`mixle/inference/estimation.py:594`, reached at line 2217), which only handles DataFrame-like
inputs, Series and one-shot iterators; the front-door normalizer that materializes mappings and
structured arrays and refuses `str`/`timedelta64`/masked/0-d inputs runs on `data` alone.
`best_of` does not even do that for `vdata`. Repairs concerned: P06-F06, P06-F07, P06-F09,
P06-F10, R06-F03.

### Q06-F02 -- `real` -- `best_of(data=DataFrame)` and `best_of(data=dict_values)` are refused with a `ContractError` about "2-tuples, got DataFrame" while `optimize`, `fit`, `propose` and `Model.fit` fit the same frame

**Surface.** `mixle.inference.best_of`.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd
from mixle.inference import optimize, best_of
rng = np.random.RandomState(0)
xs = [float(v) for v in rng.normal(size=200)]; ks = ['a', 'b'] * 100
df = pd.DataFrame({'x': xs, 'k': ks})
print(type(optimize(df, max_its=2)).__name__)                                   # CompositeDistribution
try: best_of(df, None, None, 1, 2, 0.1, 1e-6, rng=1)
except Exception as e: print('best_of(DataFrame) ->', type(e).__name__, str(e)[:120])
try: best_of({i: r for i, r in enumerate(zip(xs, ks))}.values(), None, None, 1, 2, 0.1, 1e-6, rng=1)
except Exception as e: print('best_of(dict_values) ->', type(e).__name__, str(e)[:120])
print(type(best_of({'x': xs, 'k': ks}, None, None, 1, 2, 0.1, 1e-6, rng=1)[1]).__name__)  # mapping: fits
```

**Observed** (`out/p01_frontdoor_venv-full.txt`, `best_of` section): `best_of(DataFrame)` ->
`ContractError: CompositeDistribution.seq_encode: expected a sequence of 2-tuples, got DataFrame.
Fix: pass a list/tuple of observations`; `best_of(dict.values)` -> the same with `got dict_values`.
The same call with a mapping of columns, a structured array, a generator, a `map`/`zip`/`filter`
object or a Series fits, and a masked array / `numpy.matrix` / `timedelta64` / `str` is refused by
a message that names `best_of()` -- so `best_of` is wired into the front door for everything except
the frame and a mapping view. `optimize`, `fit`, `propose`, `Model.fit` all fit the frame.

**Expected.** One table gets one answer whichever verb reads it (CHANGELOG, R06-F03). `best_of`
should fit a DataFrame (and a `dict_values` view) exactly as `optimize` does.

**Notes.** The reviewer's trail records that `best_of` infers the estimator from the normalized
records but encodes the raw `data` object (its "encode path" was read in
`mixle/inference/estimation.py`); a `dict_values` view is a non-list iterable the encoder refuses
for the same reason. Repairs concerned: P06-F03, P06-F10, R06-F03.

### Q06-F03 -- `real` -- `learn_bayesian_network` bypasses the front door: a mapping of columns is fit as a one-field network over its two KEYS, a `str` is fit as its characters, and masked / matrix / Series / DataSource inputs surface raw `TypeError`s

**Surface.** `mixle.inference.bayesian_network.learn_bayesian_network`.

**Reproduction** (venv-full):

```python
import numpy as np, warnings
from mixle.inference.bayesian_network import learn_bayesian_network
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
xs = [float(v) for v in rng.normal(size=200)]; ks = [int(v) for v in rng.poisson(3, 200)]
rows = list(zip(xs, ks))
for label, d in (('rows', rows), ('mapping of columns', {'x': xs, 'k': ks}), ('str', 'hello world hello')):
    m = learn_bayesian_network(d, max_its=2)
    print(label, '->', m, 'n_observations =', m.fit_provenance().n_observations)
for label, d in (('masked', np.ma.masked_array(np.array(rows), mask=np.zeros((200, 2), bool) | (np.arange(200) % 10 == 0)[:, None])), ('np.matrix', np.matrix(np.array(rows)))):
    try: learn_bayesian_network(d, max_its=2)
    except Exception as e: print(label, '->', type(e).__name__, str(e)[:80])
```

**Observed** (`out/p04_learn_bn_venv-full.txt`, `out/p01_frontdoor_venv-full.txt` `learn_bn`
section): rows -> `HeterogeneousBayesianNetwork(fields=2, edges=[none])`, n_observations=200;
mapping of columns -> `HeterogeneousBayesianNetwork(fields=1, ...)`, **n_observations=2** (the two
keys `'x'`, `'k'` were the records); `'hello world hello'` -> `fields=1`, n_observations=17 (its
characters); `{'x': sub.x, 'k': sub.k}` (a filtered-index Series pair, R06-F01's case) ->
fields=1, n=2; masked array -> `TypeError: unhashable type: 'MaskedConstant'`; `numpy.matrix` ->
`TypeError: unhashable type: 'matrix'`; `as_source(rows)` -> `TypeError: 'MaterializedSource' object
is not iterable`; a Series -> `TypeError: object of type 'float' has no len()`; `range(200)` ->
`TypeError: object of type 'int' has no len()`. A DataFrame, a structured array and a generator fit
correctly.

**Expected.** The CHANGELOG's front-door repairs ("a mapping of equal-length columns fits the
columns rather than a categorical over their field NAMES", "a bare string ... [is] named instead of
... fitting something nobody asked for", masked/matrix named) and R06-F03's "one table gets one
answer whichever verb reads it" should hold for this verb, or it should refuse what it does not
normalize instead of silently fitting a two-record network.

**Notes.** The docstring types `data` as `Sequence[tuple]`, but the function accepts a DataFrame
and a structured array (normalized), so users will reasonably hand it the same containers the fit
verbs take; the silent wrong model is the P06-F10 / R06-F03 defect surviving on one verb. Repairs
concerned: P06-F05, P06-F06, P06-F09, P06-F10, R06-F01, R06-F03.

### Q06-F04 -- `real` -- the production routes (`detect_drift`, `Service.score`, `Monitor.check/update`) read a DataFrame, a mapping of columns, a structured array or a `str` as its column names: a model fitted from a frame cannot score or drift-check that frame

**Surface.** `mixle.inference.production.drift.detect_drift`, `production.serving.Service.score`,
`production.monitor.Monitor.check/update`.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd, warnings
import mixle, mixle.stats as S
from mixle.inference import optimize
from mixle.inference.production.drift import detect_drift
from mixle.inference.production.serving import Service
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
df = pd.DataFrame({'x': rng.normal(size=300), 'k': ['a', 'b', 'c'] * 100})
m = optimize(df, max_its=5)                       # fits from the frame
print(mixle.Model().fit(df, max_its=5).evaluate(df))   # the lifecycle verb reads the frame
for label, a, b in (('df, df', df, df), ('records, df', list(df.itertuples(index=False, name=None)), df), ('dict of cols', {'x': df.x, 'k': df.k}, {'x': df.x, 'k': df.k}), ('structured', df.to_records(index=False), df.to_records(index=False))):
    try: print(label, '->', detect_drift(m, a, b).drift)
    except Exception as e: print('detect_drift', label, '->', type(e).__name__, str(e)[:80])
try: Service(m).score(df)
except Exception as e: print('Service.score(df) ->', type(e).__name__, str(e)[:100])
g = optimize(df['x'], S.GaussianEstimator(), max_its=3)
try: detect_drift(g, df[['x']], df[['x']])
except Exception as e: print("detect_drift(g, df[['x']], ...) ->", type(e).__name__, str(e)[:80])
```

**Observed** (`out/p18_drift_frame_venv-full.txt`): `detect_drift(m, df, df)`, `(rows, df)`,
`(df, rows)`, the mapping pair, the structured pair and `('hello', 'hello')` all raise `TypeError:
CompositeDistribution observation must be a tuple-like sequence`; `Service.score(df)` /
`(structured)` / `(dict of cols)` / `('hello')` raise `ContractError: CompositeDistribution.dists
(row 0): expected a tuple of 2 fields ..., got str` (resp. `got record`); `Monitor(m, est, rows).check(df)`,
`Monitor(m, est, df).check(df)` and `.update(df)` raise the same `TypeError`;
`detect_drift(g, df[['x']], df[['x']])` and `Service(g).score(df[['x']])` -> `ValueError: could not
convert string to float: 'x'`. `detect_drift(m, rows, rows)` and a generator pair work
(`drift=False`, ks 0.0); `Model.evaluate(df)` works. Identical on 0.8.1 (`out/p18_drift_frame_v081.txt`).

**Expected.** A model fitted by `optimize(df)` should be able to score / drift-check the same
frame; the scoring verbs should read a DataFrame, a mapping of columns and a structured array
through the same front door as the fit verbs, or refuse them by name instead of by "row 0 ... got
str".

**Notes.** Mechanism: `detect_drift` does `reference = list(reference); current = list(current)`
(`mixle/inference/production/drift.py:331-332`) and `Service.score` does `recs = list(records)`
(`serving.py:137`) -- exactly the bare `list(data)` fall-through R06-F03 removed from the lifecycle
verbs. Not a regression (same on 0.8.1). Repairs concerned: R06-F03, P06-F10.

### Q06-F05 -- `real` -- a `RecordEstimator`/`RecordDistribution` cannot be used on the lifecycle and production routes: `Model.fit(df)`, `Model.evaluate(df)`, `Service.score(df)` refuse the frame `optimize(df, est)` fits (P06-F02 covers `optimize`/`fit` only), and `detect_drift`/`Monitor` reject dict rows outright, so no input form works and identical frames report `drift=True`

**Surface.** `mixle.Model.fit/evaluate`, `Service.score`, `detect_drift`, `Monitor` with a record
estimator/model; `RecordEstimator` plain and aliased.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd, warnings
import mixle, mixle.stats as S
from mixle.inference import optimize
from mixle.stats.combinator.record import RecordEstimator, field
from mixle.inference.production.drift import detect_drift
from mixle.inference.production.serving import Service
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
df = pd.DataFrame({'x': rng.normal(size=200), 'k': ['a', 'b'] * 100})
rows = df.to_dict('records')
for label, est in (('plain', RecordEstimator(['x', 'k'], [S.GaussianEstimator(), S.CategoricalEstimator()])), ('aliased', RecordEstimator([field('mean', 'x'), field('kind', 'k')], [S.GaussianEstimator(), S.CategoricalEstimator()]))):
    m = optimize(df, est, max_its=3); print(label, 'optimize(df, est) ->', type(m).__name__)
    for name, fn in (('Model(est).fit(df)', lambda: mixle.Model(est).fit(df, max_its=2)), ('Model(est).fit(rows).evaluate(df)', lambda: mixle.Model(est).fit(rows, max_its=2).evaluate(df)), ('Service(m).score(df)', lambda: Service(m).score(df)), ('detect_drift(m, rows, rows)', lambda: detect_drift(m, rows, rows))):
        try: fn(); print(label, name, '-> ok')
        except Exception as e: print(label, name, '->', type(e).__name__, str(e)[:80])
    r = detect_drift(m, df, df); print(label, 'detect_drift(m, df, df) -> drift =', r.drift, r.reasons)
```

**Observed** (`out/p16_routes_serial_venv-full.txt`, `out/p10_dataframe_venv-full.txt`,
`out/p19_recovery_bool_labels_record_venv-full.txt`): for both the plain and the aliased record,
`optimize(df, est)` and `fit(df, est)` return a `RecordDistribution`, but `Model(est).fit(df)`,
`Model(est).fit(rows).evaluate(df)` and `Service(m).score(df)` raise `TypeError: record observation
at row 0 must be a mapping.`; `detect_drift(m, rows, rows)` and `Monitor(m, est, rows).check(rows)`
raise `TypeError: unhashable type: 'dict'`; `detect_drift(m, df, df)` and `Monitor(m, est,
df).check(df)` return `drift=True` with reasons `['score KS 1.000 > 0.2', 'mean log-likelihood
shift -inf < -0.5', 'only 0.000 of current records were scorable (< 0.5); ...']` on two identical
frames (the frame was read as its two column names, which the record scores `-inf`, see Q06-F06);
tuple rows also give `drift=True` (0 % scorable). `Service(m).score(rows)` and `Model(est).fit(rows)`
work. On 0.8.1 `optimize(df, aliased)` itself failed (`missing=['x', 'k']`), and the other routes
behave as on 0.8.2 (`out/p16_routes_serial_v081.txt`, `out/p19_..._v081.txt`).

**Expected.** P06-F02's "a `RecordEstimator` with aliased fields can be fit from a DataFrame, whose
rows now arrive keyed by the sources the record encoder asks for" should hold on `Model.fit`,
`Model.evaluate` and `Service.score`, which read their data "through the same front door"
(migration guide); and `detect_drift`/`Monitor` should accept the dict rows a record model is
defined on.

**Notes.** Mechanism: `Model.fit`/`Service.score` convert the frame to tuple rows regardless of the
estimator (the reviewer's trail: "how Model.fit forwards rows"), while `optimize` consults
`_recordish(estimator)` in `_data_records_for_encoding` and keys the rows by source;
`detect_drift` builds a hashable set/counter of observations and cannot take dicts. Repairs
concerned: P06-F02, R06-F03.

### Q06-F06 -- `real` -- `RecordDistribution.log_density` returns `-inf` (`density` 0.0) for any malformed or mis-keyed observation -- a tuple/list row, a dict missing a source, an empty dict, a dict with an extra key, `None`, a `str`, or a dict keyed by the model's own logical field names -- where `seq_encode` raises a named error for the same row

**Surface.** `RecordDistribution.log_density` / `density` (all record models, plain or aliased).

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize
from mixle.stats.combinator.record import RecordEstimator, field
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
rows = [{'x': float(v), 'k': k} for v, k in zip(rng.normal(size=200), ['a', 'b'] * 100)]
m = optimize(rows, RecordEstimator(['x', 'k'], [S.GaussianEstimator(), S.CategoricalEstimator()]), max_its=3)
print('good', m.log_density({'x': 0.1, 'k': 'a'}))
for bad in ((0.1, 'a'), [0.1, 'a'], {'x': 0.1}, {}, {'x': 0.1, 'k': 'a', 'extra': 1}, None, 'x'):
    print(repr(bad)[:30], '-> log_density', m.log_density(bad))
    try: m.seq_log_density(m.dist_to_encoder().seq_encode([bad]))
    except Exception as e: print('   seq path ->', type(e).__name__, str(e)[:90])
al = optimize(rows, RecordEstimator([field('mean', 'x'), field('kind', 'k')], [S.GaussianEstimator(), S.CategoricalEstimator()]), max_its=3)
print(al, '\nkeyed by its own field names ->', al.log_density({'mean': 0.1, 'kind': 'a'}))
```

**Observed** (`out/p19_recovery_bool_labels_record_venv-full.txt` section (b),
`out/p16_routes_serial_venv-full.txt`): the good dict scores -1.6337; every malformed row above
scores `-inf` with no exception (`density` -> 0.0), including the aliased model scored by its own
displayed field names (`RecordDistribution({'mean': ..., 'kind': ...})` -> `-inf`), while
`seq_encode` of the same rows raises `TypeError: record observation at row 0 must be a mapping.` or
`ValueError: record observation at row 0 must contain exactly the configured sources;
missing=['k'], extra=[]` (resp. `missing=['x', 'k'], extra=['mean', 'kind']`). Same on 0.8.1.

**Expected.** The scalar and vectorized paths describe the same model (the P02-F03 principle):
a malformed observation should raise the same named error from `log_density`, not be reported as
impossible. A user iterating `log_density` over `df.itertuples()` rows, or keying rows by the field
names the repr shows, gets `-inf` everywhere and no signal.

**Notes.** This is what turns Q06-F05's `detect_drift(record model, tuple rows / df, ...)` into a
silent `drift=True` (0 % scorable). Repairs concerned: P06-F02 (the aliased-fields case).

### Q06-F07 -- `real` -- a DataFrame whose column labels are booleans silently loses the `False`-labelled column: the fit has one component and `dataframe_records` returns 1-tuples, even when `fields=[True, False]` names both columns

**Surface.** `mixle.data.sources.pandas_source.dataframe_records`; every verb that reads a DataFrame
(`optimize`, `fit`, `Model.fit`, `propose`, ...).

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd, warnings
from mixle.inference import optimize
from mixle.data.sources.pandas_source import dataframe_records
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
for cols in ([True, False], [False, True], [0, 1], ['', 'k']):
    f = pd.DataFrame({'x': rng.normal(size=200), 'k': ['a', 'b'] * 100}); f.columns = cols
    m = optimize(f, max_its=2)
    print(cols, 'record 0 =', dataframe_records(f)[0], '| fitted components =', len(m.dists), '| with fields=cols:', len(optimize(f, fields=list(f.columns), max_its=2).dists))
```

**Observed** (`out/p19_recovery_bool_labels_record_venv-full.txt` section (a)): `[True, False]` ->
record 0 = `(1.764,)`, 1 fitted component, 1 with `fields=[True, False]`; `[False, True]` -> record
0 = `('a',)`, 1 component (the OTHER column survives); `[np.False_, np.True_]` likewise;
`[0, 1]`, `[1, 0]`, `[0.0, 1.0]`, `['', 'k']` -> 2-tuples, 2 components. No warning, no error.
Same on 0.8.1 (`..._v081.txt`).

**Expected.** Both columns fit (a Composite of two), or the frame is refused by name; a column
explicitly requested through `fields=` must never vanish silently.

**Notes.** Mechanism: `dataframe_records` selects `df.loc[:, source_list]`
(`mixle/data/sources/pandas_source.py`, the `.itertuples` line inside `dataframe_records`); pandas
reads a list of booleans in `.loc` as a boolean MASK over the columns, not as labels, so
`[True, False]` keeps column 0 and `[False, True]` keeps column 1. The `missing` check just above it
(`name not in df.columns`) passes because the labels do exist. Boolean column labels are what
`pd.crosstab`/`pivot` produce from a boolean column, so the input is reachable; rated `real`
rather than `blocking` because such labels are uncommon, but the loss is silent even when the
user names the column.

### Q06-F08 -- `real` -- a zero-weight out-of-support row, which P02-F03 says is exempt, still cannot be fit: `optimize()` refuses every family with "fused EM did not produce a finite objective from its non-finite initial model", and the scalar `update()`/`estimate()` route refuses seven families by support regardless of weight

**Surface.** `WeightedObservation`/`WeightedEstimator`/`WeightedDistribution` with a zero-weight
row outside the base family's support; `optimize`, `estimate`, the accumulators' `update()`.

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize, estimate
from mixle.stats import WeightedObservation as WO, seq_log_density_sum
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
for name, est, pos, bad in (('Exponential', S.ExponentialEstimator(), rng.exponential(2, 100), -5.0), ('Gamma', S.GammaEstimator(), rng.gamma(2, 2, 100), -5.0), ('Beta', S.BetaEstimator(), rng.beta(2, 3, 100), 1.5), ('Poisson', S.PoissonEstimator(), rng.poisson(3, 100).astype(float), -5.0)):
    wrows = [WO(float(v), 1.0) for v in pos] + [WO(bad, 0.0)]
    acc = est.accumulator_factory().make()
    try: acc.update(bad, 0.0, None); print(name, 'update(x_out, w=0) ok')
    except Exception as e: print(name, 'update(x_out, w=0) ->', str(e)[:60])
    try: print(name, 'estimate ->', estimate(wrows, S.WeightedEstimator(est)))
    except Exception as e: print(name, 'estimate ->', str(e)[:70])
    try: print(name, 'optimize ->', optimize(wrows, S.WeightedEstimator(est), max_its=3, out=None))
    except Exception as e: print(name, 'optimize ->', str(e)[:90])
wd = S.WeightedDistribution(S.ExponentialDistribution(2.0))
enc = wd.dist_to_encoder().seq_encode([WO(1.0, 1.0), WO(-5.0, 0.0)])
print('WeightedDistribution scores the zero-weight row:', wd.seq_log_density(enc), seq_log_density_sum([(2, enc)], wd))
```

**Observed** (`out/p15_zero_weight_matching_venv-full.txt`, `out/p14_support_siblings_venv-full.txt`,
`out/p16_routes_serial_venv-full.txt`): the vectorized accumulator honours the exemption
(`seq_update` with weights `[1, 0]` succeeds for 13 of 14 families), but `optimize(...)` raises
`ValueError: fused EM did not produce a finite objective from its non-finite initial model.` for
Exponential, Gamma, Weibull, LogGaussian, Beta, HalfNormal, Rayleigh, InverseGamma,
InverseGaussian, Poisson, Geometric, LogSeries and Uniform (also with `objective='mle'` and with
`reuse_estep_ll=False`, where the message loses the word "fused"); the scalar `update(x, 0.0)` and
`estimate(wrows, WeightedEstimator(...))` refuse Gamma, Weibull, Beta, HalfNormal, Rayleigh,
InverseGamma and InverseGaussian by support (`GammaDistribution has support x > 0.`) although the
row's weight is zero, while Exponential, LogGaussian, Poisson and Uniform accept them; Pareto
refuses at the encoder on every route. `WeightedDistribution.seq_log_density` scores the zero-weight
row `-inf` and `seq_log_density_sum` is `-inf`, which is why the objective is non-finite. On 0.8.1
every route refused at the encoder with the family's own support message
(`out/p14_support_siblings_v081.txt`).

**Expected.** The CHANGELOG (P02-F03 and its follow-up repair): "`refuse_unsupported_observations`
deliberately exempts a row whose weight is zero" and "A zero-weight row contributes exactly zero to
a sufficient statistic, on all ten continuous families that exempt one". A row the accumulator
exempts must also contribute zero to the weighted log-likelihood (0 x -inf treated as 0), so the
fit proceeds; the scalar and `estimate` routes should apply the same exemption as `seq_update`;
and a refusal, if any, should name the row and its weight rather than blame a "non-finite initial
model" that is in fact finite.

**Notes.** Repairs concerned: P02-F03 (the zero-weight exemption). The reviewer's trail read
`WeightedDistribution` scoring and the Gamma/Beta `estimate`-route support refusals.

### Q06-F09 -- `real` -- P02-F03's encoder admission stops at the ten tested families: Pareto, GeneralizedPareto, Nakagami, Rician and Tweedie still refuse an out-of-support value in `seq_log_density` where `log_density` returns `-inf`, so `Mixture[Gaussian, Pareto]` cannot be encoded or fit; the discrete families refuse a fractional value the same way

**Surface.** `ParetoDistribution`, `GeneralizedParetoDistribution`, `NakagamiDistribution`,
`RicianDistribution`, `TweedieDistribution` encoders (continuous); `PoissonDistribution`,
`GeometricDistribution`, `LogSeriesDistribution`, `BinomialDistribution`, `SkellamDistribution`,
`BernoulliDistribution`, `SymmetricDirichletDistribution` encoders; `MixtureDistribution` /
`HeterogeneousMixtureDistribution` over them.

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize
warnings.simplefilter('ignore')
p = S.ParetoDistribution(2.0, 1.0)
print('scalar', p.log_density(-1.0), p.log_density(0.5))
for x in ([-1.0, 2.0], [0.5, 2.0]):
    try: print('vectorized', x, p.seq_log_density(p.dist_to_encoder().seq_encode(x)))
    except Exception as e: print('vectorized', x, '->', type(e).__name__, str(e)[:60])
rng = np.random.RandomState(0)
data = np.concatenate([rng.normal(-2, 0.5, 200), (rng.pareto(3, 200) + 1)])
for est in (S.MixtureEstimator([S.GaussianEstimator(), S.ParetoEstimator()]), S.HeterogeneousMixtureEstimator([S.GaussianEstimator(), S.ParetoEstimator()]), S.MixtureEstimator([S.GaussianEstimator(), S.GammaEstimator()])):
    try: print(type(est).__name__, '->', str(optimize(data, est, max_its=5, rng=np.random.RandomState(1), out=None))[:70])
    except Exception as e: print(type(est).__name__, '->', type(e).__name__, str(e)[:80])
po = S.PoissonDistribution(2.0); print('Poisson scalar', po.log_density(2.5))
try: po.seq_log_density(po.dist_to_encoder().seq_encode([2.5]))
except Exception as e: print('Poisson vectorized ->', str(e))
```

**Observed** (`out/p14_support_siblings_venv-full.txt`, `out/p06_support_mix_venv-full.txt`):
`Pareto.log_density(-1.0)` = `-inf` but `seq_log_density([-1, 2])` -> `ValueError:
ParetoDistribution requires observations x > 0.` (while `[0.5, 2]`, also outside the support
`x >= x_m = 1`, is admitted and scores `-inf`); `Mixture[Gaussian, Pareto]` -> `TypeError:
MixtureDistribution could not encode the data with all of its component encoders. A finite mixture
treats the component as LATENT ...`; `HeterogeneousMixture[Gaussian, Pareto]` -> `ValueError:
ParetoDistribution requires observations x > 0.`; the same data with a Gamma component fits. The
reviewer's survey of every registered family (scalar `-inf` vs encoder refusal at x = -1.0 and
x = 2.5) lists the continuous mismatches GeneralizedPareto, Nakagami, Pareto, Rician, Tweedie and
the discrete/simplex ones Bernoulli, Binomial, Geometric, LogSeries, Poisson, Skellam,
SymmetricDirichlet (`Poisson.log_density(2.5)` = `-inf`, `seq_log_density([2.5])` -> `Poisson
observations must be finite exact integers.`; `Mixture[Gaussian, Poisson].seq_log_density([2.5, 2])`
cannot encode). Rayleigh, LogGaussian, InverseGaussian, Gumbel, Uniform, Beta and the ten families
of the regression test agree on both paths. On 0.8.1 the mismatch list was longer (Exponential,
Gamma, Weibull, Beta, HalfNormal, ... -- the repaired ones) and Pareto/Nakagami/Rician/Tweedie were
already on it.

**Expected.** "The encoders now admit out-of-support observations and the vectorized scorers return
the `-inf` the scalar path returns" (CHANGELOG, P02-F03) -- for every support-limited family,
since a mixture with a Pareto (or Tweedie, Nakagami, Rician, GeneralizedPareto) component on data
that has a non-positive row is exactly the P02-F03 symptom.

**Notes.** The P02-F03 regression test enumerates its families (`adversarial_review_082_repairs_test.py`,
`_families()`: Exponential, Gamma, Beta, Weibull, HalfNormal, Rayleigh, Uniform, ...); the sibling
families were never in it. For the discrete families a fractional observation is a type mismatch as
much as a support miss, but the scalar path answers `-inf` for it, so the two routes still describe
different models. Repairs concerned: P02-F03.

### Q06-F10 -- `real` -- `StudentTCopulaDistribution` and `KnowledgeGraphDistribution` are write-only for `to_json`/`dump_models` (constructed and fitted; a `CopulaDistribution` with a Student-t core too), while the manifest registers both as codec `constructor-validated` and the CHANGELOG discloses only the directional and matrix modules

**Surface.** `StudentTCopulaDistribution.to_json/from_json`, `KnowledgeGraphDistribution.to_json/from_json`,
`mixle.stats.dump_models/load_models`, `CopulaDistribution(..., StudentTCopulaDistribution)`;
`manifests/serialization_schema_manifest.json`.

**Reproduction** (venv-full):

```python
import numpy as np, pickle
import mixle.stats as S
from mixle.inference import estimate
from mixle.stats import dump_models, load_models
rng = np.random.RandomState(0)
u = S.ClaytonCopulaDistribution(3, theta=4.0).sampler(0).sample(500)
for d, fit_data in ((S.StudentTCopulaDistribution(np.eye(3), 4.0), u), (S.KnowledgeGraphDistribution(rng.randn(6, 3), rng.randn(2, 3)), None)):
    if fit_data is None: fit_data = list(d.sampler(1).sample(200))
    m = estimate(fit_data, d.estimator()); print(type(m).__name__, 'fitted:', m)
    for label, fn in (('to_json', lambda: d.to_json()), ('fitted dump_models', lambda: dump_models([m])), ('fitted dump(verify=False)+load', lambda: load_models(dump_models([m], verify=False)))):
        try: fn(); print('  ', label, 'ok')
        except Exception as e: print('  ', label, '->', type(e).__name__, str(e)[:150])
    print('   pickle round trip ok:', type(pickle.loads(pickle.dumps(m))).__name__)
```

**Observed** (`out/p16_routes_serial_venv-full.txt`, `out/p17_json_writeonly_venv-full.txt`,
`out/p11_copulas_venv-full.txt`, `out/p13_graphs_procs_venv-full.txt`): both families raise
`SerializationError: to_json produced JSON that from_json cannot read back (SerializationError:
registered class 'mixle.stats.multivariate.student_t_copula.StudentTCopulaDistribution' requires a
class-owned __pysp_setstate__ hook; constructor fields are absent: corr...` (resp. `...knowledge_graph.
KnowledgeGraphDistribution' ... constructor fields are absent: entity_embedding...`) from `to_json()`
and `dump_models()`, constructed or fitted; `dump_models(verify=False)` writes text that
`load_models` cannot read; a fitted `CopulaDistribution([Gamma, Gaussian, Gamma], StudentTCopula)`
cannot be dumped either; pickle works. The Clayton, Gaussian-copula and R-vine fits and the ER/SBM/
RDPG graph fits round-trip bit-exactly. Identical on 0.8.1 (`out/p17_json_writeonly_v081.txt`).
`serialization_schema_manifest.json` lists both classes with `"codec": "constructor-validated"`.

**Expected.** Per the CHANGELOG (0.8.0 disclosure, reaffirmed by the P05-F17 repair "Eight more
families round-trip through JSON") the only families that cannot round-trip are the directional and
matrix modules; a Student-t copula reached through the copula core search and a knowledge-graph
model reached through `optimize()` should serialize to JSON, or be named in the disclosure and
carry a manifest codec that says so.

**Notes.** Same defect class as P05-F17 / P10-F05 (rated `real` there for the same reason: every
save path costs these families a code-executing pickle, and `Model.deploy` silently falls back to
it). The fitted directional laws also fail (`does not match its constructor-owned schema`) but the
directional module is disclosed; the constructed directional laws do round-trip on 0.8.2, so that
disclosure is now over-broad rather than wrong. Repairs concerned: P05-F17 (the disclosure claim).

### Q06-F11 -- `minor` -- the row-numbered ordering refusal was added to Mallows, Plackett-Luce, Bradley-Terry and Thurstone-Mosteller only: Thurstone, GeneralizedMallows, GeneralizedMallowsModel, LowRankPermutation, SpearmanRanking, Matching and Ewens still raise Python's "'NoneType' object is not iterable" and numpy's "inhomogeneous shape" for a None/scalar/string/ragged row, and Davidson/Rao-Kupper for a None/scalar tie row; every family reads a dict row as its keys

**Surface.** `ThurstoneEstimator`, `GeneralizedMallowsEstimator`, `GeneralizedMallowsModelEstimator`,
`LowRankPermutationEstimator`, `SpearmanRankingEstimator`, `MatchingEstimator`, `EwensEstimator`,
`DavidsonEstimator`, `RaoKupperEstimator` (and their encoders).

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize
warnings.simplefilter('ignore')
rng = np.random.RandomState(0); perms = [list(rng.permutation(4)) for _ in range(60)]
for name, mk in (('Mallows', lambda: S.MallowsEstimator(4)), ('Thurstone', lambda: S.ThurstoneEstimator(4)), ('GeneralizedMallows', lambda: S.GeneralizedMallowsEstimator(4)), ('LowRankPermutation', lambda: S.LowRankPermutationEstimator(4, 2)), ('SpearmanRanking', lambda: S.SpearmanRankingEstimator(4)), ('Ewens', lambda: S.EwensEstimator(4))):
    for bad in (None, [0, 1], 7, 'abc'):
        try: optimize(perms + [bad], mk(), max_its=2)
        except Exception as e: print('%-19s %-8r -> %s: %s' % (name, bad, type(e).__name__, str(e)[:70]))
    print(name, 'dict row ->', str(optimize(perms + [{0: 1, 1: 0, 2: 3, 3: 2}], mk(), max_its=2))[:60] if name != 'LowRankPermutation' else '(see Q06-F12)')
```

**Observed** (`out/p02_rankings_venv-full.txt`, identical in venv-base): Mallows -> `ValueError:
Mallows orderings row 60 is not a ordering (NoneType); every ordering must be a sequence of exactly
4 items.` / `row 60 has 2 items ...` (P06-F09 repaired; note "a ordering"); Plackett-Luce ->
`Plackett-Luce observations are orderings ... but row 60 is None` (R06-F07 repaired); Thurstone,
GeneralizedMallows, GeneralizedMallowsModel, LowRankPermutation, SpearmanRanking, Matching, Ewens ->
`TypeError: 'NoneType' object is not iterable` / `'int' object is not iterable` and `ValueError:
setting an array element with a sequence. The requested array has an inhomogeneous shape after 1
dimensions ...` for the ragged, string and nested rows, on the fit verb and on the encoder directly;
Davidson/Rao-Kupper `None`/scalar tie row -> `'NoneType' object is not iterable` (their 2-item and
outcome-5 rows are named). A dict row is silently read as its keys `[0, 1, 2, 3]` by every family
(the fitted theta moves). Unchanged from 0.8.1 for these families.

**Expected.** The sibling families of the same shape get the same row-numbered message
(the fractional-entry check `... row 60 must contain exact integer item identifiers` is already
shared by all of them, so the helper exists).

**Notes.** Repairs concerned: P06-F09, R06-F07 (verified for the families they name).

### Q06-F12 -- `minor` -- `LowRankPermutationEstimator` raises `RuntimeError` from inside `optimize()` when its inner solver misses its tolerance by 27 % at 300 iterations, instead of returning the unconverged fit with `converged=False` as the `max_its` cap does

**Surface.** `LowRankPermutationEstimator` via `optimize`/`estimate`.

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize
warnings.simplefilter('ignore')
rng = np.random.RandomState(0); perms = [list(rng.permutation(4)) for _ in range(60)]
try: print(optimize(perms + [[0, 1, 2, 3]], S.LowRankPermutationEstimator(4, 2), max_its=2))
except Exception as e: print(type(e).__name__, str(e))
```

**Observed** (`out/p02_rankings_venv-full.txt`, the `LowRankPermutation + dict row` case, whose dict
row is read as the ordering `[0, 1, 2, 3]`): `RuntimeError: Low-rank permutation fitting did not
converge in 300 iterations (gradient norm 1.26696e-06, tolerance 1e-06).` -- the user asked for two
EM iterations and gets no model at all. Same on 0.8.1.

**Expected.** The library's own convention for a capped fit: return the last accepted model,
warn, and report `converged=False` in `fit_provenance()` (the cap note that `optimize()` itself
emits), or expose the inner cap/tolerance as arguments.

**Notes.** If this reproduction's data happens not to trip the tolerance on another BLAS, the
recorded case in `out/p02_rankings_venv-full.txt` (same seed, `numpy 2.5.3`) is the evidence.

### Q06-F13 -- `docs` -- the migration guide's list of newly refused inputs omits `datetime64` and `timedelta64` arrays: a `datetime64` array that 0.8.1 fitted (as an Ignored categorical over its timestamps) is now refused, and the CHANGELOG names only `timedelta64`

**Surface.** `docs/migrations/0.8.2.md` ("A bare `str` or `bytes` raises ... along with the
refusals of a masked array, a `numpy.matrix` and a 0-dimensional array, and the normalization of a
one-shot iterator, a structured array and a mapping of columns"); CHANGELOG "Data adapters (pass 06)".

**Reproduction** (venv-full, then 0.8.1):

```python
import numpy as np
from mixle.inference import optimize
dt = np.datetime64('2020-01-01') + np.arange(200) * np.timedelta64(1, 'D')
for d in (dt, np.arange(200) * np.timedelta64(1, 's')):
    try: print(d.dtype, '->', str(optimize(d, max_its=2))[:80])
    except Exception as e: print(d.dtype, '->', type(e).__name__, str(e)[:110])
```

**Observed** (`out/p08_datetime_venv-full.txt` vs `out/p08_datetime_v081.txt`): 0.8.2 ->
`ValueError: optimize() received a numpy datetime64[D] array. Fit its numeric representation instead
-- e.g. data.astype('int64') ...` and the same for `timedelta64[s]`; 0.8.1 -> the `datetime64`
array fitted `IgnoredDistribution(CategoricalDistribution({np.datetime64('2020-01-01'): 0.005, ...}))`
and the `timedelta64` array crashed with `TypeError: int() argument must be ... not
'datetime.timedelta'`. Neither dtype appears in `docs/migrations/0.8.2.md` (grep: no `datetime64`,
no `timedelta64`); the CHANGELOG names `timedelta64` only.

**Expected.** R06-F06 made the guide's list "the full list it claims to be"; a behaviour change
from "fits" to "raises" for `datetime64` arrays belongs in it, and the CHANGELOG entry should name
`datetime64` beside `timedelta64`.

**Notes.** Repairs concerned: R06-F06, P06-F06, P06-F09, P06-F10. The related container inconsistency is
Q06-F20.

### Q06-F14 -- `minor` -- the `(n, 1)` refusal is `(n, 1)`- and estimator-route-specific: an `(n, 2)` or `(1, n)` array, an `(n, 1)` list of lists, `log_density(col)` and `detect_drift(model, x, col)` still surface numpy's "only 0-dimensional arrays can be converted to Python scalars" (for `detect_drift` a worse message than 0.8.1's), Bernoulli accepts the column silently, and the message names "row 199" for three families

**Surface.** Univariate estimators' `seq_encode`/`estimate`; `GaussianDistribution.log_density`;
`detect_drift`.

**Reproduction** (venv-full):

```python
import numpy as np, warnings
import mixle.stats as S
from mixle.inference import optimize, estimate
from mixle.inference.production.drift import detect_drift
warnings.simplefilter('ignore')
rng = np.random.RandomState(0); x = rng.normal(size=200); col = x.reshape(-1, 1)
g = optimize(x, S.GaussianEstimator())
for label, fn in (('estimate((n,1), Gamma)', lambda: estimate(np.abs(x).reshape(-1, 1) + 0.1, S.GammaEstimator())), ('estimate((1,n))', lambda: estimate(x.reshape(1, -1), S.GaussianEstimator())), ('estimate((n,2))', lambda: estimate(np.stack([x, x], 1), S.GaussianEstimator())), ('estimate(list of [v])', lambda: estimate(col.tolist(), S.GaussianEstimator())), ('g.log_density(col)', lambda: g.log_density(col)), ('detect_drift(g, x, col)', lambda: detect_drift(g, x, col))):
    try: print(label, '->', fn())
    except Exception as e: print(label, '->', type(e).__name__, str(e)[:110])
print('Bernoulli (n,1) ->', estimate((rng.poisson(3, 200) > 2).astype(float).reshape(-1, 1), S.BernoulliEstimator()))
```

**Observed** (`out/p07_col_array_venv-full.txt` vs `out/p07_col_array_v081.txt`): the 19 univariate
families refuse an `(n, 1)` array with `<Family>Estimator expects a scalar observation per row, but
row 0 is an array of shape (1,). Flatten the data with numpy.ravel(data) ...` (P01-F13/P06-F08
repaired; Gamma, Weibull and HalfNormal say `row 199`), and the scoring routes `seq_encode(col,
model=g)`, `Model(Gaussian).fit(col)`, `Model.evaluate(col)`, `Service.score(col)`,
`optimize(vdata=col)` refuse by name too. But `(1, n)` and `(n, 2)` arrays and an `(n, 1)` list of
lists still raise `TypeError: only 0-dimensional arrays can be converted to Python scalars` /
`float() argument must be ... not 'list'`; `g.log_density(col)` and `g.log_density(col[0])` raise
the numpy `TypeError`; `detect_drift(g, x, col)` raises the numpy `TypeError` where 0.8.1 said
`ValueError: model scoring returned (200, 1) log-densities for 200 records; drift analysis needs
exactly one score per record`; `estimate((n, 1), Bernoulli)` fits `BernoulliDistribution(0.565)`
without complaint.

**Expected.** The same named refusal for every non-1-D array into a univariate estimator or
scorer, on every route, with a consistent first offending row.

**Notes.** Repairs concerned: P01-F13, P06-F08.

### Q06-F15 -- `minor` -- front-door refusals are not named on every verb: a bare `float`/`np.float64` and a mapping with a 0-d array column give "'float' object is not iterable" / "object of type 'float' has no len()" on `optimize`, `fit` and `best_of` but a named `ValueError` on `propose` and `Model.fit`

**Surface.** `optimize`, `fit`, `best_of` vs `propose`, `Model.fit`.

**Reproduction** (venv-full):

```python
import numpy as np, mixle
from mixle.inference import optimize, fit, best_of
for label, d in (('float', 1.5), ('np.float64', np.float64(1.5)), ('mapping with 0-d column', {'x': np.array(1.0)})):
    for verb, f in (('optimize', lambda d: optimize(d, max_its=2)), ('fit', lambda d: fit(d, max_its=2)), ('best_of', lambda d: best_of(d, None, None, 1, 2, 0.1, 1e-6, rng=1)), ('propose', lambda d: mixle.propose(d, max_its=2, fit=True)), ('Model.fit', lambda d: mixle.Model().fit(d, max_its=2))):
        try: f(d)
        except Exception as e: print('%-24s %-9s -> %s: %s' % (label, verb, type(e).__name__, str(e)[:60]))
```

**Observed** (`out/p01_frontdoor_venv-full.txt`; `p01_table.py` flags the rows): `optimize`/`fit`/
`best_of` -> `TypeError: 'float' object is not iterable` (resp. `'numpy.float64' ...`) and
`TypeError: object of type 'float' has no len()` for the 0-d column; `propose`/`Model.fit` ->
`ValueError: data must be a collection of observation records ...`. A 0-d array handed directly is
named by all five (P06-F09).

**Expected.** "The refusals name the verb that was called" (R06-F03) on all five verbs.

**Notes.** Repairs concerned: P06-F09, R06-F03.

### Q06-F16 -- `minor` -- `optimize(..., backend='mp')` cannot raise the 30 s worker setup timeout: `planner.py` builds `MPEncodedData(...)` without `response_timeout`, so on a loaded machine the fit verb fails with "parallel worker 0 timed out after 30.000s during setup" with no argument to control it (the three `ClosedParallelHandleTest` regression tests failed the same way)

**Surface.** `optimize(backend='mp')`, `mixle/utils/parallel/planner.py:1118`,
`MPEncodedData.__init__(response_timeout=30.0)`.

**Reproduction** (venv-full; reproduces when the machine is loaded, e.g. load average > 100 on
18 cores as during this review -- otherwise it shows only the missing knob):

```python
import numpy as np, inspect, warnings
import mixle.stats as S
from mixle.inference import optimize
from mixle.utils.parallel.multiprocessing import MPEncodedData
warnings.simplefilter('ignore')
print('optimize accepts response_timeout:', 'response_timeout' in inspect.signature(optimize).parameters)
xs = [float(v) for v in np.random.RandomState(0).normal(3, 2, 400)]
rows = list(zip(xs, ['a', 'b'] * 200)); ce = S.CompositeEstimator([S.GaussianEstimator(), S.CategoricalEstimator()])
try: print(optimize(rows, ce, max_its=2, backend='mp', num_workers=2))
except Exception as e: print('optimize(backend=mp) ->', type(e).__name__, str(e))
h = MPEncodedData(xs, estimator=S.GaussianEstimator(), num_workers=2, response_timeout=600.0); print('direct handle with response_timeout=600 ->', len(h)); h.close()
for label, fn in (('MPEncodedData(generator)', lambda: MPEncodedData((r for r in rows), estimator=ce, num_workers=2, response_timeout=600.0)), ('MPEncodedData(mapping of columns)', lambda: MPEncodedData({'x': xs, 'k': ['a', 'b'] * 200}, estimator=ce, num_workers=2, response_timeout=600.0))):
    try: fn().close()
    except Exception as e: print(label, '->', type(e).__name__, str(e)[:60])
```

**Observed** (`out/p09_mp_closed_venv-full.txt`, `tests/pytest_full.txt`): `optimize(rows, ...,
backend='mp', num_workers=2)` -> `TimeoutError: parallel worker 0 timed out after 30.000s during
setup` for the rows, generator and structured inputs (one call, the mapping, happened to get
through), so the local-vs-mp agreement check could not run; a handle built directly with
`response_timeout=600.0` served `seq_initialize`/`seq_estimate`/`seq_log_density_sum`/
`optimize(enc_data=)` correctly, and after `close()` refused every entry point with `RuntimeError:
this parallel encoded-data handle is closed: init/update/llsum cannot be served ...` (P06-F04,
R06-F02 verified; a killed worker is reported as `parallel worker disconnected while sending init`
and the handle then reads as closed). The direct handle has no front door: a generator ->
`TypeError: object of type 'generator' has no len()`, a mapping of columns -> `KeyError: 0`; a
closed handle's `__enter__()` succeeds. `optimize`'s signature has `backend`/`num_workers` and no
timeout.

**Expected.** A way to set the worker timeout from the fit verb (or a setup deadline that scales
with `num_workers`/load), and a refusal from the direct handle that names the unsupported input.

**Notes.** The regression-test failures in `tests/pytest_full.txt` are this contention path, not a
defect in the closed-handle repair. Repairs concerned: P06-F04, R06-F02.

### Q06-F17 -- `minor` -- the copula domain refusal for a NaN observation reports "observed min=0.5, max=0.7" -- both inside (0, 1) -- and never says a NaN was found

**Surface.** `ClaytonCopulaDistribution`, `GaussianCopulaDistribution`, `StudentTCopulaDistribution`
(`log_density`, `seq_log_density`, `estimate`).

**Reproduction** (venv-full):

```python
import numpy as np, mixle.stats as S
for c in (S.ClaytonCopulaDistribution(3, theta=4.0), S.GaussianCopulaDistribution(np.eye(3))):
    try: c.log_density(np.array([np.nan, 0.5, 0.7]))
    except Exception as e: print(type(c).__name__, '->', str(e))
```

**Observed** (`out/p11_copulas_venv-full.txt`): `ValueError: copula observations must be finite and
lie strictly inside (0, 1); observed min=0.5, max=0.7` (Clayton, Student-t; Gaussian says
`observation`), the same for `estimate` with a NaN row (`observed min=0.5, max=0.5`).

**Expected.** Name the NaN (and its row), since the reported range satisfies the stated condition.

### Q06-F18 -- `minor` -- `fields=` cannot address a MultiIndex / tuple-labelled column: `fields=[('num', 'x')]` is still read as an `(alias, source)` pair and fails with "DataFrame is missing fields: x", although P06-F03 made the frame itself fit

**Surface.** `optimize(df, fields=...)` (and every verb taking `fields=`) on a 2-level MultiIndex
frame or a flat index of 2-tuples.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd, warnings
from mixle.inference import optimize
warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
mi = pd.DataFrame({'x': rng.normal(size=200), 'k': ['a', 'b'] * 100}); mi.columns = pd.MultiIndex.from_tuples([('num', 'x'), ('cat', 'k')])
print('whole frame ->', type(optimize(mi, max_its=2)).__name__)
for f in ([('num', 'x')], [('num', 'x'), ('cat', 'k')]):
    try: optimize(mi, fields=f, max_its=2)
    except Exception as e: print('fields=%r ->' % f, type(e).__name__, str(e))
```

**Observed** (`out/p10_dataframe_venv-full.txt`): the whole frame fits (P06-F03 verified on
`optimize`, `fit`, `propose`, `Model.fit`, `dataframe_records`, 2- and 3-level MultiIndex and a flat
index of 2-tuples); `fields=[('num', 'x')]` -> `KeyError: 'DataFrame is missing fields: x'`,
`fields=[('num', 'x'), ('cat', 'k')]` -> `... missing fields: x, k`. A `RecordEstimator` with
`field('mean', ('num', 'x'))` sources does fit the frame, so the tuple label is addressable only
through a record estimator.

**Expected.** A `fields=` entry that equals a column label of the frame selects that column
(before the `(alias, source)` interpretation), or the error says the tuple was read as an alias
pair.

**Notes.** Repairs concerned: P06-F03.

### Q06-F19 -- `minor` -- a closed `MPEncodedData` handle still enters as a context manager, and `len()` of it answers the old size, after `close()` made every request refuse

**Surface.** `MPEncodedData.__enter__`, `__len__` after `close()`.

**Reproduction** (venv-full):

```python
import numpy as np, mixle.stats as S
from mixle.utils.parallel.multiprocessing import MPEncodedData
xs = [float(v) for v in np.random.RandomState(0).normal(3, 2, 400)]
h = MPEncodedData(xs, estimator=S.GaussianEstimator(), num_workers=2, response_timeout=600.0); h.close()
print('len after close:', len(h)); print('enter after close:', h.__enter__())
try: __import__('mixle.inference', fromlist=['seq_estimate']).seq_estimate(h, S.GaussianEstimator(), S.GaussianDistribution(0, 1))
except Exception as e: print('seq_estimate after close ->', type(e).__name__, str(e)[:70])
```

**Observed** (`out/p09_mp_closed_venv-full.txt`): `closed len -> 400`, `closed context manager ->
<MPEncodedData object>` (no error), while `seq_initialize`/`seq_estimate`/`seq_log_density_sum`/
`optimize(enc_data=h)` raise the closed-handle `RuntimeError` (P06-F04 / R06-F02 verified) and
`close()` twice is the documented no-op.

**Expected.** Entering a closed handle should raise the same closed-handle error (or re-open), so a
`with` block over a closed handle fails at the `with`, not inside.

**Notes.** Repairs concerned: P06-F04, R06-F02.

### Q06-F20 -- `minor` -- the same timestamps are refused as a `datetime64` ndarray but fitted as an Ignored categorical when handed as a pandas Series, a list of `np.datetime64`, an object array, or a mapping column

**Surface.** `optimize` (auto path) on datetime-valued inputs.

**Reproduction** (venv-full):

```python
import numpy as np, pandas as pd
from mixle.inference import optimize
dt = np.datetime64('2020-01-01') + np.arange(200) * np.timedelta64(1, 'D')
df = pd.DataFrame({'t': pd.to_datetime(dt)})
for label, d in (('ndarray', df['t'].to_numpy()), ('Series', df['t']), ('list of np.datetime64', list(dt)), ('object array', np.array(list(dt), dtype=object)), ('mapping column', {'t': df['t']})):
    try: print(label, '->', str(optimize(d, max_its=2))[:70])
    except Exception as e: print(label, '->', type(e).__name__, str(e)[:70])
```

**Observed** (`out/p08_datetime_venv-full.txt`): ndarray -> `ValueError: optimize() received a numpy
datetime64[s] array. Fit its numeric representation instead ...`; Series, list, object array and
mapping column -> `IgnoredDistribution(CategoricalDistribution({Timestamp('2020-01-01 ...'): 0.005,
...}))` -- the "fitting something nobody asked for" outcome the refusal exists to prevent.

**Expected.** One answer per table regardless of container: either every route refuses datetimes
by name with the numeric-representation hint, or none does.

**Notes.** Repairs concerned: P06-F09, P06-F10 (the timedelta64 refusal). Docs side: Q06-F13.

## Attacks that did not break anything

- **Front door, all five fit verbs, three environments and 0.8.1** (`probes/p01_frontdoor.py`,
  `out/p01_frontdoor_{venv-full,venv-nonumba,venv-base,v081}.txt`, `p01_table.py`): generator,
  `iter(list)`, `map`, `zip`, `filter`, `dict.values`, `range`, `MaterializedSource`, masked array
  (masked and nomask), structured / record / sub-array / datetime-field / object-field arrays,
  `numpy.matrix`, `timedelta64`, `datetime64`, 0-d array, `str`/`bytes`/`bytearray`, mapping of
  lists / ndarrays / Series (filtered index, R06-F01) / tuple keys / int keys, ragged, empty,
  scalar, str, set and DataFrame-valued mappings, 2-d column, masked column, DataFrame, Series and
  empty DataFrame -- on `optimize`, `fit`, `best_of`, `propose`, `Model.fit`: P06-F01, P06-F05,
  P06-F06, P06-F07, P06-F09, P06-F10, R06-F01 and R06-F03 hold on all five verbs (exceptions:
  Q06-F02, Q06-F15), numba and non-numba outputs are identical (the nonumba/base runs skipped
  `propose` only), and every one of these inputs behaves worse or wrongly on 0.8.1 (structured ->
  `unhashable type: 'writeable void-scalar'`, generator -> `Ignored(PointMass(None))`, mapping ->
  categorical over the keys, `numpy.matrix` -> `RecursionError`, str -> categorical over
  characters). `describe()` and `Model.evaluate()` on the same inputs behave like `optimize`.
- **DataFrame labels** (`p10`): 2- and 3-level MultiIndex, a flat index of 2-tuples, shared first
  level, duplicate labels (named on every verb, P06-F09), int / float / None / NaN / mixed
  tuple-and-str labels, `fields=` aliases, repeated and empty `fields=`, unknown field; aliased
  `RecordEstimator` from a frame on `optimize`/`fit`/`prev_estimate`/`vdata`/`seq_encode_dataframe`
  (P06-F02 on those routes); `Float64` Series with `pd.NA` on every route (Optional fit, p=1/3).
- **Ranking refusals** (`p02`): Mallows None/ragged/scalar/string/nested/np.int64/object-array
  rows (P06-F09), Plackett-Luce None/scalar/string (R06-F07), Bradley-Terry and Thurstone-Mosteller
  None/1-item/3-item/string/scalar/nested comparison rows, Davidson/Rao-Kupper 2-item and outcome-5
  rows, fractional entries on every ordering family; identical in venv-base.
- **RDPG clip disclosure P09-F12** (`p05`): the ledger reproduction reports `edge-probability-clipped(2
  of 56 off-diagonal entries; inner products span [0.08637, 1.074], ...)`, negative products are
  reported, a scaled-in `X` reports `()`, the disclosure survives `to_json`/`from_json`, pickle and
  `dump_models`/`load_models`, a `MixtureDistribution` container prefixes it with `components[0].`
  (P02-F05), an exact 1.0 product is not a clip, `1 + 1e-15` is, `inf` positions are refused; the
  fitted route (ASE) produced no clip on the two data sets tried and scored every training graph
  finitely.
- **P02-F03 on the repaired families** (`p06`, `p14`, also venv-base): Exponential, Gamma, Weibull,
  Beta, HalfNormal, InverseGamma, Rayleigh, LogGaussian, InverseGaussian, Uniform, Gumbel admit
  out-of-support rows and score them `-inf` on both paths (`seq == scalar` True), in
  `MixtureEstimator`, `HeterogeneousMixtureEstimator` and `HiddenMarkovModelEstimator`; the mixture
  batch log-likelihood is finite and NaN-free; every one of these failed on 0.8.1.
- **P01-F13 / P06-F08 on 19 univariate families and the scoring routes** (`p07`).
- **Closed and killed `MPEncodedData` handles** (`p09`; P06-F04, R06-F02): every entry point refuses
  after `close()`; a SIGKILLed worker is reported and the handle then reads as closed.
- **Copulas** (`p11`, `p16`): boundary / out-of-range / NaN / wrong-dimension observations refused
  by name on Clayton, Gaussian, Student-t; single-row fit refused (Clayton, Student-t), two-row and
  all-identical and comonotonic data handled (Clayton names its comonotonic boundary); float32 and
  list-of-tuples inputs; Clayton / Gaussian-copula / R-vine fits round-trip through JSON, pickle and
  `dump_models` bit-exactly; vine `seq` vs scalar agree to 1.8e-15; a vine fit from a DataFrame and
  from a mapping of columns; the copula example's claim that bare `optimize()` reaches a copula on
  Clayton-coupled columns holds (`CopulaDistribution`, converged) and independent columns give a
  `CompositeDistribution`.
- **Directional laws** (`p12`, `p17`): periodicity (`x`, `x + 2pi`, `x - 4pi` score identically),
  NaN / inf / None refused by name, `(n, 1)` refused by name, single / identical / empty data raise
  the family's own `FitError`, mixtures of two directional laws fit, sphere laws refuse non-unit,
  zero, NaN and wrong-dimension vectors; fitted directional laws are JSON write-only (disclosed
  since 0.8.0; `Model.deploy` falls back to pickle and records `format_fallback`), constructed ones
  now round-trip; very concentrated von Mises data (sd <= 1e-5) raises `VonMisesFitError` by name.
- **Graphs, processes, multivariate** (`p13`): adjacency validation (self-loop, asymmetry,
  non-binary, wrong size, NaN, dtype variants), single / empty / complete / mixed-size graph fits,
  ER/SBM/RDPG JSON+pickle+dump round-trips; knowledge-graph out-of-range / malformed triples
  refused by name; renewal / inhomogeneous-Poisson / Hawkes / CRP / birth-death out-of-window,
  unsorted, negative, NaN and empty realizations (`-inf` or named refusal) and their round-trips;
  multivariate Gaussian single / identical / collinear (ridge disclosed) / NaN / ragged rows;
  Dirichlet boundary rows.
- **Matching gauge** (`p15`): the `gallery_rankings_example` Matching fit prints weights far from
  the truth (`5.31` vs `2.0`), but the implied matching probabilities agree with the truth to 3e-3
  and with the empirical frequencies, and per-row log-densities agree to 1e-3 -- the weights are
  identified only up to row/column scaling, so the fit is right (the example's printout could say
  so).
- **Examples in venv-base**: the ten non-torch examples print byte-identical output to venv-full.
- **Regression tests in venv-base**: 141 passed, 15 skipped.

## What was not covered

- `data_science/model_based_embeddings.ipynb` was compared only from its single retry (exit 0, 3014 s,
  load average 50-90); its first run timed out at 1800 s under load average 200-330. No second
  timing was taken, so its wall time is not evidence of anything.
- The `mp` backend under contention: the 30 s setup timeout (Q06-F16) prevented the
  local-vs-`mp` agreement check and the `mp`-backend front-door inputs from running; `pyspark`
  was not exercised (out of area).
- Embeddings beyond the htsne tutorial and the `shared_embedding_example` (which only constructs
  its estimators); the two torch examples in venv-base fail at their documented `import torch`.
- The 0.8.1 baseline was run for `p01`, `p02`, `p06`, `p07`, `p08`, `p14`, `p16`, `p17`, `p18`,
  `p19` only; `p10`-`p13` and `p15` ran on 0.8.2 alone.
- The regression tests were run, not audited line by line beyond the P02-F03 family list and the
  R06-F01/F02/F03/F07 tests the reviewer read (the trail records no verdict on them).
- The `local` backend's chunking (`sub_chunks`) and the `ResilientMPEncodedData` / audited handles.

DONE 06 20 findings
