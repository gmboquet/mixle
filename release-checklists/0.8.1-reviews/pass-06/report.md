# Adversarial review — pass 06 of 10 — mixle 0.8.1 release candidate

- **Pass:** 06
- **Focus:** data input surfaces — pandas DataFrame/Series adapters, record distributions reading columns by name, numpy arrays of every dtype/shape, lists of records/dicts/tuples, mappings of columns, DataSource / normalize_input / one-shot iterators, missing-value handling (None, NaN, pd.NA, pd.NaT, inf), datetime/categorical/string columns, the `mp` and `local` backends, rankings/paired-comparison encoders.
- **Wheel:** `<review-root>/candidate-081/dist/mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d` (re-hashed locally, matches)
- **Tree:** `6040ea38` (branch release/0.8.1)
- **Version verification line (run from the work dir, outside any checkout):**
  `0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle`
- **Interpreter / deps:** Python 3.12.12, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1
- **Work dir:** `<review-root>/reviews-081/pass-06/` (attack scripts in `attacks/`, outputs `attacks/*.out`, consolidated reproductions in `repro.py` / `repro.out`). Nothing outside the work dir was modified. macOS has no `timeout(1)`; `tmo.py` in the work dir is the subprocess-timeout wrapper used everywhere.

## Executed corpus notebooks (nbconvert --execute, timeout 1200 s per cell)

| # | Notebook | Exit | Wall | Output file |
|---|---|---|---|---|
| 1 | tutorials/parallel_estimation.ipynb | 0 | 79 s | executed-1.ipynb |
| 2 | data_science/missing_data_and_imputation.ipynb | 0 | 11 s | executed-2.ipynb |
| 3 | data_science/heterogeneous_mixed_type_modeling.ipynb | 0 | 40 s | executed-3.ipynb |
| 4 | data_science/market_basket_ibp.ipynb (reads two CSVs) | 0 | 14 s | executed-4.ipynb |
| 5 | applications/baseball_hierarchical_shrinkage.ipynb (reads CSV) | 0 | 6 s | executed-5.ipynb |
| 6 | architecture_studies/parallel_scaling.ipynb | 0 | 11 s | executed-6.ipynb |
| 7 | applications/retail_trend_statespace.ipynb (reads CSV) | 0 | 28 s | executed-7.ipynb |
| 8 | tutorials/model_parallel_estimation.ipynb | 0 | 9 s | executed-8.ipynb |
| 9 | applications/generative_title_decoding.ipynb (pandas, file read) | 0 | 33 s | executed-9.ipynb |

No error outputs in any executed notebook. Runtime-computed claims in the outputs are true on the candidate: parallel_estimation "all backends agree: True" and "dataframe fit ... matches local: True"; parallel_scaling "model-parallel fit bit-identical to serial: True", "data×model composition matches serial: True"; model_parallel_estimation "matches single-process: True". The only output-level oddity is parallel_scaling cell 8 printing "all **1** backends agree on the fit" because no MPI is present — a degenerate but not false statement. Every `optimize(..., max_its=1)` in these notebooks emits the new unconverged-fit UserWarning; the notebooks do not claim convergence for those cells.

## Executed examples

| Example | Exit | Wall |
|---|---|---|
| scaling_example.py (local vs mp backends) | 0 | 34 s |
| gallery_rankings_example.py | 0 | 28 s |
| heterogeneous_representation_example.py | 0 | 24 s |
| heterogeneous_correctness_example.py | 0 | 5 s |
| quickstart_example.py | 0 | 6 s |
| auto_example.py | 0 | 3 s |

scaling_example prints local and mp fits that agree to the printed precision (mu=2.00, P(a)=0.60, lam=4.01), consistent with its "agreement, not bit-equality" text. Measured directly: local vs mp(num_workers=2,7,None) differ in sigma2 by 6.7e-16 (one reassociation ulp); num_workers=1 and 3 are bit-identical to local.

## Findings

### P06-F01 (real) — `optimize()`/`fit()`/`propose()` with `estimator=None` on a one-shot iterator silently return `IgnoredDistribution(PointMassDistribution(None))`, which scores every observation `-inf`

Surface: `optimize(generator)`, `optimize(iter(list))`, `optimize(map(...))`, `fit(generator)`, `mixle.propose(generator)` with the default `structure='auto'`.

Reproduction (from a directory outside any checkout):
```
PY=<review-root>/candidate-081/venv/bin/python
$PY -c "import numpy as np, mixle; from mixle.inference import optimize, fit
xs=[float(v) for v in np.random.RandomState(0).normal(size=200)]
m=optimize(iter(xs), max_its=3); print(m, m.log_density(xs[0]))
print(fit(v for v in xs)); print(mixle.propose(iter(xs)).fitted)
print(optimize(iter(xs), max_its=3, structure='off')); print(optimize(xs, max_its=3))"
```
Expected: the `GaussianDistribution(0.0709, 1.0433)` that the list input yields. `mixle.utils.automatic.profiling.normalize_input`'s docstring promises one-shot iterators are materialized precisely so that "a one-shot iterator would [not] silently fit a wrong/empty model on the second pass".

Observed: `IgnoredDistribution(PointMassDistribution(None))`, `log_density -> -inf`, `fit()` identical, `propose(...).fitted -> None`. No warning. `structure='off'` or any explicit estimator gives the correct fit.

Cause: the `structure='auto'` front door (`estimation.py` ~1585–1610) hands the raw iterator to `_maybe_structured_model`, whose `rows = list(data)` (line 229) exhausts it; when structure inference declines, the ordinary path re-profiles the now-empty iterator and `get_estimator([])` returns the empty-corpus model. This is the worst finding of the pass: a silent, unflagged wrong model on a documented-as-supported input type.

### P06-F02 (real) — `RecordEstimator` with aliased `field(name, source)` cannot be fit from a DataFrame through `optimize()`/`fit()`

Reproduction:
```
$PY -c "import numpy as np, pandas as pd; from mixle.inference import optimize
from mixle.stats import GaussianEstimator, CategoricalEstimator
from mixle.stats.combinator.record import RecordEstimator, field
xs=list(np.random.RandomState(0).normal(size=200)); df=pd.DataFrame({'x':xs,'k':['a','b']*100})
est=RecordEstimator([field('mean','x'),field('kind','k')],[GaussianEstimator(),CategoricalEstimator()])
m=optimize(df, est, fields=['x','k'], max_its=3)
print(m.log_density({'x':0.1,'k':'a'}), m.log_density({'mean':0.1,'kind':'a'}))
print(optimize(df, est, max_its=3))"
```
Expected: `optimize(df, est)` reads columns `x`,`k` into fields `mean`,`kind` (what `field(name, source)` documents, and what `mixle.data.seq_encode_dataframe(df, estimator=est)` already does).

Observed: `ValueError: record observation at row 0 must contain exactly the configured sources; missing=['x','k'], extra=['mean','kind']` — for `optimize(df, est)`, `fit(df, est)`, and `optimize(df, est, fields=[('mean','x'),('kind','k')])`. Only `fields=['x','k']` works. The fitted model then scores a logically-keyed row `{'mean':..,'kind':..}` at `-inf` silently. Cause: `estimation._data_records_for_encoding` calls `dataframe_records(..., as_dict=True)` with the default `_dict_keys='logical'`, while `seq_encode_dataframe` passes `_dict_keys='source'` (with a comment explaining exactly this contract).

### P06-F03 (real) — 2-level `MultiIndex` columns are misparsed as `(name, source)` field specs

Reproduction:
```
$PY -c "import numpy as np, pandas as pd; from mixle.inference import optimize
df=pd.DataFrame({'x':np.random.RandomState(0).normal(size=200),'k':['a','b']*100})
mi=df.copy(); mi.columns=pd.MultiIndex.from_tuples([('num','x'),('cat','k')]); print(optimize(mi, max_its=3))"
```
Expected: a fit of the two columns (a 3-level MultiIndex and integer column names both work), or a ContractError naming MultiIndex columns.

Observed: `KeyError: 'DataFrame is missing fields: x, k'`; with a repeated first level, `ValueError: logical fields must be unique, got ['num', 'num']`. `mixle.propose(mi)` and a flat `pd.Index` of 2-tuples (`tupleize_cols=False`) fail the same way. `pandas_source._field_source/_field_name` treat any 2-tuple as `(name, source)` and `dataframe_records` feeds `df.columns` through that parser when `fields is None`.

### P06-F04 (real) — a closed `MPEncodedData` handle silently yields a default model and zero log-density

Reproduction (`closed_mp.py`, run with `$PY`):
```
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, seq_log_density_sum
from mixle.utils.parallel.multiprocessing import MPEncodedData
if __name__ == "__main__":
    xs = list(np.random.RandomState(0).normal(size=200))
    h = MPEncodedData(xs, estimator=GaussianEstimator(), num_workers=2); h.close()
    print(len(h)); print(optimize(None, GaussianEstimator(), enc_data=h, max_its=3))
    print(seq_log_density_sum(h, optimize(xs, GaussianEstimator())))
```
Expected: an error stating the handle is closed. Observed: `200`, `GaussianDistribution(0.0, 1e-08)` with no warning, `(0, 0)`. `close()` empties `_conns`/`_procs`; `_broadcast_collect` then returns `[]`, the fold sees zero observations and `estimator.estimate(0, ...)` returns the default model, while `__len__` still reports the pre-close size. A mixture estimator returns `MixtureDistribution([Gaussian(0,1e-8), Gaussian(0,1e-8)],[0.5,0.5])`.

### P06-F05 (minor) — `optimize()` with `estimator=None` rejects a `DataSource`

`optimize(as_source(xs), max_its=3)` -> `TypeError: 'MaterializedSource' object is not iterable` (estimation.py:229); `LazySource` likewise. With `structure='off'` or an explicit estimator the fit is correct. Same root cause as F01 (the structure front door bypasses `normalize_input`, which does unwrap a DataSource), but loud instead of silent.

### P06-F06 (minor) — masked array: auto-inference crashes before the masked-array guard

`optimize(np.ma.masked_array(x, mask=...))` -> `TypeError: len() of unsized object` (profiling `_positional_row_widths`); with `GaussianEstimator()` the intended `ValueError: optimize() received a numpy masked array with 20 masked value(s)...` fires. `_reject_masked_data` runs at estimation.py:1715, after `get_estimator` has already consumed the masked array.

### P06-F07 (minor) — structured/record arrays crash with `unhashable type: 'writeable void-scalar'`

`optimize(np.array([(1.0,'a'),(2.0,'b')]*50, dtype=[('x','f8'),('k','U1')]))` and `np.rec.array(...)` -> `TypeError` at profiling.py:2246. `normalize_input` returns ndarrays unchanged with the comment "numeric/structured array: nothing to freeze", so this is a claimed-supported shape.

### P06-F08 (minor) — `(n,1)` array with an explicit `GaussianEstimator` crashes in `seq_update`, while the same shape encodes and scores correctly

`optimize(x.reshape(-1,1), GaussianEstimator())` -> `ValueError: shapes (200,1) and (200,) not aligned` (gaussian.py:807, `np.dot(x, weights)`); `seq_encode(col, model=g)` + `seq_log_density` gives the correct −288.027; auto-inference reads `(n,1)` as a `SequenceDistribution` of length-1 sequences (numerically the same likelihood).

### P06-F09 (minor) — raw `TypeError`/`RecursionError`/`AttributeError`/numpy messages where neighbouring paths give row-numbered ContractErrors

Reproduce all with `$PY <review-root>/reviews-081/pass-06/repro.py | sed -n '/# F09/,/# F10/p'`:
(a) `np.matrix` input -> `RecursionError` (profiler recurses on matrix rows); (b) numpy `timedelta64` array -> `TypeError: int() argument ... not 'datetime.timedelta'` (a pandas timedelta column fits); (c) 0-d array with an estimator -> `TypeError: len() of unsized object`; (d) `bytes` -> "Integer-categorical observations must be finite exact integers"; (e) DataFrame with a duplicated column name and `fields=['x']` -> `AttributeError: 'DataFrame' object has no attribute 'tolist'`; without `fields` the message "logical fields must be unique" never mentions duplicate column names; (f) `ThurstoneMostellerEstimator` with a `None` row -> `TypeError: 'NoneType' object is not iterable`, whereas `BradleyTerryEstimator` says "row 50 has no items ..."; (g) `MallowsEstimator` with a ragged ranking -> numpy's "inhomogeneous shape" message, whereas `PlackettLuceEstimator` says "full Plackett-Luce rankings must all have length 4". (g) is the same defect class the 0.8.1 changelog says was closed for mixed 2-/3-tuple paired comparisons.

### P06-F10 (minor) — a mapping of columns is silently fit as a categorical over the column names

`optimize({'x': xs, 'k': [...]})` -> `CategoricalDistribution({'k': 0.5, 'x': 0.5})`, no warning; `mixle.describe(d)` -> "dict — no catalogued capability detected" (the 0.8.1 changelog says `describe()` on raw data now points at `propose()`); `propose(d).fitted -> None`. `optimize('hello world')` likewise fits a categorical over characters.

### P06-F11 (docs) — `backend='mp'` with `num_workers` far above the core count fails only after the setup timeout

`optimize(20000 rows, est, backend='mp', num_workers=1000)` on a 10-core Mac: 1000 processes spawned, 384.7 s wall, then `TimeoutError: parallel worker 122 timed out after 30.000s during setup`, with stray `Process SpawnProcess-N:` tracebacks. Workers are cleaned up (no leak). `num_workers=0/-1/2.5/'4'` are silently coerced. The docstring caps only the *default* at cpu_count.

## Attacks that did not break anything

pandas / DataFrame:
- Baseline frame vs list-of-tuples vs `convert_dtypes()`: bit-identical log-likelihoods (−883.0226…).
- Nullable `Int64`/`boolean`/`string`/`Float64` with `pd.NA`, pyarrow `float64[pyarrow]` with null, `NaN`/`None` in object string columns, `pd.NaT` in datetime, tz-aware datetime, timedelta, `Decimal` and complex object columns, all-NaN column, one-row frame, non-default and MultiIndex **row** index, integer column names, categorical with unused categories, numeric-looking string column with one dirty cell (fits continuous, as the changelog claims), object column mixing ints and strings, `+inf/-inf` in a numeric column (fits as disclosed OptionalDistribution with the documented UserWarning), 3-level MultiIndex columns, zero-row frame (clean "no observations" error), empty frame, `fields=` selection/reorder/aliasing with distinct source columns, `fields=[('a','x'),('b','x')]` (fits, degenerate copula as expected for a duplicated column), missing/empty `fields` (clean KeyError/ValueError), vdata as DataFrame or generator.
- Cross-scoring: model fit on numpy-backed frame scores the `convert_dtypes()` twin and vice versa, identical LL; NaN-string frame model scores the None-string frame.
- Series: `Float64`/`Int64`/`boolean`/`string`/categorical with `pd.NA`, float with NaN, all-NaN, all-NA, datetime with NaT, float32, weird index (values only), named Series, Series of tuples/dicts/ragged lists/dicts with missing keys.
- The changelog's pd.NA claims verified: a model fit from `list(nullable_series)` or from the Series scores `pd.NA`, `NaN`, its own list, `series.to_numpy()`, and re-fits via `prev_estimate` and `m.estimator()`. (`mixle.stats.seq_encode(series, ...)` refuses a Series with a clear ContractError, as documented.)

numpy:
- float64/32/16, int8, uint64 (small; large gives a clear "must fit in signed 64-bit" error), bool, object-of-floats, str, bytes, datetime64 (frozen as identifier categorical, documented), Fortran order, read-only arrays, `np.memmap`, `(n,2)`/`(n,3,2)` arrays, `(1,n)`, empty `(0,)`/`(0,2)` (clean error), subnormals, identical values (variance floor), NaN with auto (Optional) and with explicit estimator (clear ValueError naming `marginalized()`), inf with explicit estimator (clear support error), Poisson on float-valued ints / huge ints / non-integers, Bernoulli on bool/int, Categorical on np.str_/np.int64 then scoring plain Python values, float32 vs float64 fits agree to 1e-9.

records / containers:
- Lists/tuples of tuples, lists, dicts; ragged rows (dominant arity raises the row-numbered ContractError with the fix text; 60/40 mix warns and fits a sequence, as documented); dicts with a missing key (Optional), extra key, None/int/str/tuple keys; mixed dict/tuple rows fail loudly; `range`, `set`, nested ragged lists; RecordEstimator on a DataFrame by column name, with an extra column, with an integer-named column, with an omitted optional key; RecordEstimator on tuples (clear "must be a mapping"), duplicate sources (clear "must be unique"), missing column (clear KeyError); Composite with `fields=` reorder; Composite without `fields` on a wider frame (row-numbered ContractError).
- `normalize_input`: generator -> list, DataFrame -> records, flat float list returned as the same object, str/None/int handled or rejected; `MaterializedSource`/`LazySource` from generators replay `records()` twice, `partition(n)`/`encode(num_chunks, chunk_size)` reject 0/negative cleanly; `seq_encode(DataSource, estimator=...)` and `optimize(DataSource, estimator)` work.

mp backend:
- `num_workers` None/0/1/2/3/7/-1/2.5/'4' all fit and agree with local to ≤1 ulp; `backend='multiprocessing'`, `'MP'`; `''`/`'nope'` refused with the registered-backend list; estimator=None, generator, DataFrame, numpy inputs through mp; `sub_chunks` 0/5; `vdata`; unpicklable observations and lambda-holding data fail fast with the pickling error; worker SIGKILLed before the fit -> `RuntimeError: parallel worker disconnected while sending init` in 12 ms; worker SIGKILLed mid-fit -> `RuntimeError: parallel worker 2 disconnected during llsum (exitcode=-9)` in 0.3 s; `close()` twice; workers released on `del` and after `optimize(backend='mp')`; a 2-component Gaussian mixture run to convergence under local, local `num_chunks=3`, mp×1, mp×3 with `init_p` 0.1 and 1.0 all reach the same optimum (LL −105142.6023, parameters equal to 1e-8; only label order/trajectory differ).

rankings / paired comparisons (with `dim`):
- BradleyTerry, ThurstoneMosteller, Davidson, RaoKupper: lists/tuples/numpy int8/uint64/float arrays accepted; 3-tuples into a pair model, 2-tuples into a tie model, mixed 2/3 arity (row-numbered message naming the fix, as the changelog claims), non-integer/negative/out-of-range/self-comparison/string/bytes/bool/nested/1-tuple/NaN rows all rejected with row-numbered messages; empty data clean; disconnected graphs and never-losing competitors give the documented strongly-connected ValueError; `win_probability`/`tie_probability` sum to 1, accept float/numpy ints, reject (i,i) and out-of-range; `fit_diagnostics` carries plain Python scalars; `BradleyTerryDataEncoder(dim=None)` with max id 20000 does not allocate a dim² matrix (maxrss 0.43 GB). PlackettLuce/Mallows: partial rankings, duplicate/out-of-range items, strings, empty rankings all rejected with clear messages (except Mallows ragged, F09g).

## Summary

- Counts: blocking 0, real 4 (P06-F01..F04), minor 6 (P06-F05..F10), docs 1 (P06-F11).
- Worst: P06-F01 — a one-shot iterator (generator/`iter`/`map`) passed to `optimize()`/`fit()`/`propose()` with the default auto-inference is silently fit as `IgnoredDistribution(PointMass(None))` scoring `-inf` everywhere; the `structure='auto'` front door exhausts the iterator before profiling.
- 9 corpus notebooks and 6 examples executed cleanly on the candidate; no false runtime claims found in their outputs.
- The 0.8.1 changelog claims in this area (pd.NA canonicalization, Series-with-missing fits, numeric-looking string columns, inf disclosure, mixed-arity paired-comparison messages, `win_probability`, plain-scalar BT diagnostics) all verified true.
- Rankings encoders and the mp backend are robust apart from F04 (closed handle) and F11 (slow failure on absurd worker counts).
