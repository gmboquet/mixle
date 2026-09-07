# Adversarial review PASS 05 — production / MLOps layer

- Pass: 05 of 10
- Focus: `mixle.inference.production` (fit_with_provenance / Header / verify_lineage, Registry, Service, detect_drift, Monitor), `mixle.lifecycle.Model` (fit / explain / deploy / load), `mixle.stats.dump_models` / `load_models`, `to_json` / `from_json`, `model_hash`, checkpoint / resume.
- Wheel: `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`, built from tree `6040ea38` (branch `release/0.8.1`).
- Version verification (run from `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-05/`):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` ->
  `0.8.1 /Users/grantboquet/mixle/.ci-repro-colima/candidate-081/venv/lib/python3.12/site-packages/mixle` (Python 3.12.12).
  `environment_info()` inside the candidate reports `git_commit=6040ea38c322715a974701898e369f9cc8f40c18`, `provenance_source=installed-artifact-build-provenance`, `mixle_version=0.8.1`.
- Work dir: `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-05/` (attack scripts in `attacks/`, one standalone reproduction per finding in `repro/`, executed notebooks `executed-N.ipynb`, example logs `ex-*.log`, run table `corpus_results.txt`). Nothing outside the work dir was modified.

## Corpus execution

The corpus has very few notebooks that touch this layer (grep for `fit_with_provenance|Registry|Service|detect_drift|dump_models|to_json|checkpoint|verify_chain` hits only `receipts_and_replay.ipynb` and `reasoning_over_real_images.ipynb`). I ran the provenance/registry-adjacent ones plus the pillar-validation notebooks that mention registries/provenance, all via `nbconvert --execute --ExecutePreprocessor.timeout=1200` from the work dir.

| # | Notebook | Exit | Wall |
|---|----------|------|------|
| 1 | data_science/receipts_and_replay.ipynb | 0 | 22 s |
| 2 | applications/pillar_validation/production_H.ipynb | 0 | 20 s |
| 3 | applications/pillar_validation/simulation_P.ipynb | 0 | 21 s |
| 4 | applications/pillar_validation/biodiversity_N.ipynb | 0 | 25 s |
| 5 | applications/pillar_validation/climate_L.ipynb | 0 | 23 s |
| 6 | exploration_geoscience/00_foundations.ipynb | 0 | 12 s |
| 7 | tutorials/fitting_and_estimation.ipynb | 0 | 31 s |
| 8 | data_science/em_and_map_strategies.ipynb | 0 | 64 s |

| Example | Exit | Wall |
|---------|------|------|
| production_example.py | 0 | 10 s |
| flagship_triage_app.py | 0 | 6 s |
| frontier_family_showcase.py | 0 | 25 s |
| label_economics_demo.py | 0 | 5 s |
| reasoner_investigation_demo.py | 0 | 3 s |

All exit 0. `production_example.py` prints the claimed chain (`lineage verified: True`, `versions ['v1','v2'], production -> mu=2.95`, `drift on shifted batch: True (ks=0.84)`, `checkpoints ['v1','v2','v3'] | chain intact: True`) and every claim holds on the candidate. Notebook 1 executes cleanly but its *output* contradicts its own narrative (P05-F07). Notebooks 2-8 are not really MLOps notebooks; their outputs matched their committed outputs and their Definition-of-Done cells passed.

## Findings

### P05-F01 (real) — Registry: swapping two version files is undetected; the promoted alias serves the other version
`repro/P05-F01_registry_swap.py`. Register mu=1 as v1 and mu=9 as v2, promote `production -> v1`, swap the contents of `v1.json` and `v2.json` on disk. `current("m","production")` returns the v2 model (mu=9.0) with no error; `get("m","v1").mu == 9.0`; `v1.json` now literally says `"version": "v2"` and its `record_digest` still verifies because the digest covers the payload, never the file name. `_load_payload` never checks `payload["version"] == version`. A rollback (promote back to a known-good version) is defeated by a plain file swap — exactly what `current()`'s docstring says the alias mechanism exists to guarantee.

### P05-F02 (real) — Registry.get/current serve a model of another family under the original provenance header with no cross-check
`repro/P05-F02_registry_family_swap.py`. Rewrite the `model` block of a registered Gaussian's version file to a `PoissonDistribution`, recompute `record_digest` with the module's own `_record_digest`. `get("m")` returns `(PoissonDistribution, header)` where `header["model_type"] == "GaussianDistribution"` and `header["model_hash"] != model_hash(loaded)`; no error, no warning; `Service.from_registry` then scores with it. `Model.load` performs exactly this cross-check (family -> refuse, content hash -> warn + note); the registry stores the header as an opaque dict and never compares it with the model it accompanies.

### P05-F03 (real) — fit_with_provenance's attached `.header` breaks to_json/dump_models and silently downgrades Model.deploy to a pickle artifact
`repro/P05-F03_header_breaks_serialization.py`. `model, header = fit_with_provenance(...)`; `dump_models(model)` and `model.to_json()` raise `SerializationError (... extra=['header'])`. Wrapping the same fitted Gaussian in `mixle.Model` and calling `deploy()` writes `format='pickle'`, `format_fallback='the serialization registry has no JSON form for GaussianDistribution ...'`, so a plain Gaussian becomes a code-executing artifact that `Model.load` refuses without `trust_code=True`. `del model.header` restores JSON. `Registry.register` and `model_hash` already special-case `header`; `to_serializable` and `Model._write_model` do not. The provenance module docstring says headers "serialize to JSON alongside the model".

### P05-F04 (real) — `fit_request_digest` differs between byte-identical fit requests
`repro/P05-F04_fit_request_digest.py`. Two identical calls give the same `model_hash` and `dataset_hash` but different `training["fit_request_digest"]`, because `fit_request["estimator_repr"]` is `repr(estimator)` = `<...GaussianEstimator object at 0x125a728d0>` (estimators have no `__repr__`). The digest hashes a memory address, so it can never match a header to the request that produced it.

### P05-F05 (real) — verify_chain returns False on a valid chain when trust_code is invalid; checkpointer then blames the data
`repro/P05-F05_verify_chain_swallows_trust_error.py`. `verify_chain("ck")` -> True; `verify_chain("ck", trust_code="yes")` / `trust_code=1` -> False. The `require_explicit_true` ValueError raised inside `self.get()` is swallowed by the blanket `except (KeyError, TypeError, ValueError): return False`. `checkpointer("ck", trust_code="yes")` consequently raises "checkpoint lineage for 'ck' does not verify ... repair or remove the damaged versions" for an intact chain — the docstring's own comment says a raise from `verify_chain` must propagate as "verification could not be performed".

### P05-F06 (real) — detect_drift reports DRIFT for a numpy-array batch of the same rows a list batch reports as no drift (composite models)
`repro/P05-F06_drift_ndarray_false_positive.py`. `rows[:100]` as a list -> `drift=False`, per-feature PSI 0.093/0.004. `np.array(rows[:100])` -> `drift=True`, `reasons=["feature 'field_0' PSI 0.717 > 0.25"]`, only one feature reported. `_raw_columns` tests `isinstance(rows[0], (tuple, list))`; an ndarray row fails, the two-column batch is treated as one scalar column, both fields are flattened into one PSI against the reference's field_0. `Service.check_drift(np.array(...))` -> True; `Monitor.update` would retrain. Score drift (ks=0.058) is identical in both cases, i.e. the model itself scores the array rows fine.

### P05-F07 (real, corpus) — receipts_and_replay.ipynb: the un-tampered Receipt fails verify_receipt on 0.8.1
`repro/P05-F07_receipt_notebook.py` (cell 8 verbatim), or `executed-1.ipynb` cell 8. Committed output: `receipt verifies offline: True`, four `pass`. Candidate output: `receipt verifies offline: False` with `executables_match=fail`, `trace_replayable=fail`, `provenance_present=fail` — while `is_bit_identical_replay(trace, TOOLS)` in the cell above is True. `verify_receipt` now requires `receipt.executables` pre-filled with `implementation_digest(tool)` (nothing in `Receipt`/`record_step` fills it, no public helper) and `provenance={"sources":[{id,digest,content}]}`, while the module docstring still says provenance follows the `SubstrateItem.provenance` dict shape the notebook uses. The notebook's narrative ("corrupt any one claim and verification fails on *that* claim, while the others still stand"; "offline re-verifiable") is false as executed; the cell has no assert so the corpus gate sees rc=0.

### P05-F08 (minor) — DriftReport is frozen but its containers are mutable; str() reports injected values
`repro/P05-F08_driftreport_mutable.py`. `r.score["ks"]=99.0; r.reasons.append("injected")` succeed and `str(r)` prints `DriftReport[ok]  score: ks=99.000 ... reasons: injected`. The `__post_init__` comment claims "a mutation after construction cannot rewrite evidence"; CHANGELOG lists Report records among the "35 durable records ... frozen". Same pattern in `Receipt`.

### P05-F09 (minor) — checkpointer `every` is not validated
`repro/P05-F09_checkpointer_every.py`. `every=0`, `-1`, `True` -> checkpoint every iteration; `every=1.5` -> one checkpoint (iteration 3, `3 % 1.5 == 0`); `every="2"` / `None` -> `TypeError` from inside `optimize()` on the first step. `resume=` is truthiness-coerced (`resume="no"` behaves as True).

### P05-F10 (minor) — Service keep=0 is unbounded, keep=-1 discards every event
`repro/P05-F10_service_keep.py`. `keep=0`: `activity[-0:]` is a full slice, log grows without bound. `keep=-1`: after five `score()` calls `len(activity)==0` and `health()=={'events': 0, ...}` — a monitoring blind spot with no error, while `health(window=0/-1)` in the same class is rejected loudly.

### P05-F11 (minor) — damaged/foreign registry files surface as raw errors; versions() lists non-version names
`repro/P05-F11_registry_raw_errors.py`. Truncated `v1.json` -> bare `json.JSONDecodeError`; symlinked `v5.json` is listed by `versions()` and `get` raises `OSError [Errno 62] Too many levels of symbolic links` (the `O_NOFOLLOW` errno), same for a symlinked `production.alias` via `current()`; a stray `vX.json` is listed as version `'vX'` (sorted first) and `get("m","vX")` raises `KeyError('model')`. Refusals are correct; reporting is not.

### P05-F12 (minor) — a NaN record makes detect_drift / Service.check_drift / Monitor.check raise while Service.score counts it as unscorable
`repro/P05-F12_drift_nan_raises.py`. `Service.score(batch)` -> `-inf`, `n_unscorable=1`. `detect_drift(m, ref, batch)`, `svc.check_drift(batch)`, `Monitor(...).check(batch)` -> `UnscorableObservation: GaussianDistribution rejects NaN observations.` The drift module computes `fraction_unscorable_*` and fails closed on coverage, but nothing catches `UnscorableObservation` the way `Service._safe_logd` does; a Monitor loop dies on the first NaN.

### P05-F13 (minor) — Registry writes NaN/-Infinity literals into version files
`repro/P05-F13_registry_nonstrict_json.py`. A header with `final_loglik=nan` registers; the file contains the literal `NaN`; `Registry.get` reads it back via lenient `json.load`, but `mixle.utils.serialization.from_json(text)` raises `SerializationError("non-standard JSON constant 'NaN' is forbidden")` and any non-Python consumer gets invalid JSON. `register()` uses `json.dump` without `allow_nan=False`, unlike every other mixle artifact writer.

### P05-F14 (docs) — verify_lineage does not bind the data hash or the loglik trace; production_example.py over-claims
`repro/P05-F14_lineage_does_not_bind_data.py`. Rewrite `dataset_hash` to zeros, `n_records` to 1e9, every convergence `loglik` and `final_loglik` to 0.0 -> `verify_lineage(header)` still True. Transition digests cover only iter/run_id/model_hash/parent fields. `production_example.py` says the chain answers "is this served model really the one that came out of **that data** and that fit? without trusting anyone's word".

### P05-F15 (docs) — "authenticated" lineage is unkeyed content hashing; a forged tip verifies
`repro/P05-F15_chain_forgeable.py`. Replace the tip model with `GaussianDistribution(1234, 1)` and recompute `model_hash`, `transition_digest`, `record_digest` with the module's helpers -> `verify_chain` True, `get()[0].mu == 1234.0`. `Model.load`'s docstring is honest about this threat model; `Registry`'s ("authenticated lineage record", "authenticated tip", "Verified digest") is not.

### P05-F16 (minor) — assorted validation gaps
`repro/P05-F16_misc_validation.py`. `verify_lineage("x")` raises `ValueError` instead of returning False; `Service.from_registry(reg, name, trust_code=True)` -> `TypeError` (kwarg forwarded to `Service.__init__`, so a NeuralLeaf registry entry cannot be served through `from_registry`); `detect_drift(psi_threshold="0.5")` passes validation (`float("0.5")` works) and dies later with `TypeError '>' not supported between float and str`; `detect_drift(loglik_shift_threshold=+0.5)` flags identical reference/current as drift.

### P05-F17 (real) — six families outside the CHANGELOG-disclosed modules are write-only for to_json/dump_models
`attacks/e_roundtrip.py` (results in `attacks/e2.out`): 63 fitted families pushed through `to_json/from_json`, `dump_models/load_models` and `utils.to_json/from_json`. 51 round-trip bit-exactly (identical `seq_log_density`, identical `model_hash`, identical re-serialized text) and their `model_hash` is identical in child processes under `PYTHONHASHSEED=1` and `=2`. 12 raise `SerializationError` at encode time (the read-back guard works — no write-only text escapes). Six of the 12 are the disclosed directional module (VonMises, WrappedNormal, WrappedCauchy, Kent, VonMisesFisher, Bingham). The other six are **not** covered by the CHANGELOG line "cannot round-trip any family in `mixle.stats.directional` or `mixle.stats.matrix` (12 families across two whole modules)": `DirichletDistribution`, `DirichletProcessMixtureDistribution` (`mixle.stats.bayes`), `MultivariateStudentTDistribution` (`mixle.stats.multivariate`), `ProbabilisticPCADistribution`, `HierarchicalMixtureDistribution`, `JointMixtureDistribution` (`mixle.stats.latent`). Every save path for these costs a code-executing pickle (`Model.deploy` falls back with the documented warning).

## Attacks that did not break anything

- **Registry integrity (per-file)**: editing any byte of a version file (registered_at, a model parameter, the header) without recomputing `record_digest` -> `ValueError: registry integrity failure` from `get`, `current`, `header`, `metadata`. Editing a model parameter and recomputing the digest -> refused by the schema decoder (`invalid 'object' payload fields`) because parameters live under `state`, not as loose keys. Truncated files are never served.
- **Path safety**: names/aliases `..`, `a/b`, `""`, absolute paths -> `ValueError`; alias file pointing at `v99` or `../../etc/passwd` -> `KeyError` (resolved against the known version list); symlinked version/alias/lock files are never followed (`O_NOFOLLOW`); a model directory that is a symlink is refused.
- **Concurrency**: 16 threads and 8 separate processes registering under one name -> v1..v16 / v1..v8 with no clobber and no error; `expected_tip` semantics correct (None on a non-empty name refuses, stale tip refuses, current tip succeeds); two checkpointers adopting the same tip -> exactly one appends, the other gets `registry conflict`, chain stays verifiable; a failed register (non-JSON metadata) leaves no temp file and does not consume a version number.
- **Chain verification**: truncated tip, deleted middle checkpoint, swapped checkpoint files, replaced tip model with stale transition digest, plain `register()` appended to a chain, backwards `step.iter` -> `verify_chain` False; `checkpointer(resume=True)` refuses on every one of those; `resume=False` roots a fresh lineage; resume from a valid tip continues `checkpoint_iter` (3,6,9 -> 12,15) and verifies; `every=0` chain verifies.
- **trust_code**: `1`, `"yes"`, `"True"`, `"false"`, `""`, `None`, `0`, `1.0`, `np.True_`, `[True]` all rejected by `Registry.get`/`current` and (except the documented `None`/`0` no-trust values) by `Model.load`; only the `True` singleton opens the trusted scope. A planted `model.pkl` with `__reduce__` is refused before unpickling by the manifest digest even with `trust_code=True`.
- **Header / lineage**: swapping, dropping, duplicating records, editing any `model_hash`, changing `run_id`, flipping `lineage_status`, removing the terminal, editing the header `model_hash` -> `verify_lineage` False; `Header.from_dict(to_dict())` round-trips through JSON and still verifies; `lineage="yes"`, `seed=-1`, `seed=True`, `rng=` rejected; generator data materialized once; `header.model_hash == model_hash(model)`; same seed -> same `model_hash`, different seed -> different.
- **Model.load artifact tampering**: edited/truncated `model.json`, wrong `format` ("JSON", "pickle" on a JSON artifact), `model_file="../model.json"`, missing model file, foreign `mixle_artifact` tag -> refused with specific `SerializationError`s; same-family swap with recomputed sha -> loads with the documented content-hash warning + note; other-family swap with recomputed sha -> refused on `family`; deploy onto a plain file -> clear `FileExistsError`; redeploy in place works; mixture deploy/load reproduces `seq_log_density` exactly; VonMises deploy -> pickle fallback disclosed three ways as documented.
- **Model wrapper**: `Model(GaussianEstimator)` (class) rejected; `restarts=0/True`, `calibrate=1.0` rejected; `calibrate=0.5` attaches a calibration report with evidence records; `evaluate([])`, `evaluate([nan])`, unfitted `__call__`/`deploy`, `posterior` on a non-latent model -> clear errors; `explain()` works fitted and unfitted; `propose()` returns an unfitted wrapper whose `_require_fitted` hint names the remedy; `dump_models(Model wrapper)` refused with the documented message and `dump_models(m.fitted)` works.
- **Service.score**: scalar / None / dict / string records -> TypeError/ValueError; tuples on a scalar model, 2-D arrays, nested lists -> the documented cardinality `ValueError`; NaN/inf records -> `-inf` counted as unscorable; composite ragged/extra-field/wrong-type rows -> `ContractError`; unseen categorical -> `-inf`; flaky `seq_log_density` (ConnectionError) + flaky `log_density` (TimeoutError) -> `n_batch_unavailable=4`, `n_unavailable=2`, health rates correct; `availability_errors` validated; empty batch logged with `mean_loglik=None`; `health(window=0/-1/1.0)` rejected; `log_path` JSONL written; `from_registry` with a missing alias refuses.
- **detect_drift / Monitor**: reference==batch -> `drift=False`, ks=0, psi=0; empty reference/batch -> `ValueError`; batch of one / reference of one -> DRIFT (PSI 12-28, fail-closed on tiny evidence); NaN/negative/out-of-range thresholds rejected; constant reference vs moved batch -> DRIFT; categorical complete replacement -> DRIFT with `fraction_unscorable_current=1.0`; generators materialized once; 200 000-record batch scored in 0.07 s; `Monitor` rejects bad thresholds, `retrain="yes"`, empty batches; `Monitor.update` on a shifted batch retrains and returns a new header; `suggest_samples` returns the requested shape.
- **Hangs / memory**: nothing hung; `fit_with_provenance(lineage=True)` on a 4-state HMM for 30 iterations costs 0.28 s vs 0.20 s for bare `optimize` (per-iteration hashing is cheap); verification of an 18-record lineage is instant.

## Summary

- Counts: 0 blocking, 8 real (F01-F07, F17), 7 minor (F08-F13, F16), 2 docs (F14, F15).
- Worst finding: **P05-F01** — a registry's promoted alias silently serves a different version after two version files are swapped on disk (the record digest never binds the file name), defeating the atomic-swap/rollback guarantee the Registry exists to provide; closely followed by P05-F03, where the library's own provenance header turns a plain Gaussian into a pickle artifact through `Model.deploy`.
- Everything the release claims about `trust_code`, per-file integrity, concurrency, and chain corruption detection held under attack; the gaps are in what is *not* bound (file name, header-vs-model, data hash, request identity) and in a corpus notebook whose executed output now contradicts its text.
- 8 notebooks and 5 examples executed on the candidate, all exit 0; 51 of 63 fitted families round-trip through JSON bit-exactly with process-stable hashes.
- Reproductions: one standalone script per finding under `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-05/repro/`.
