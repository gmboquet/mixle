# Pass 08 -- the teaching surface: the 12 tutorials, the neural/vision data_science notebooks, the README's snippets and claims (venv-full and venv-base), the docs quickstart, the README/documentation review records and the 0.8.2 migration guide

This report was assembled by a recovery reviewer from the preserved evidence of the pass-08 reviewer
(command trail in `RECOVERED_NOTES.md`; every probe, output and executed notebook is in this work
dir), plus the few runs named below that were needed to confirm or reject a finding. Every number
here is read from a saved output in this directory.

## Header

- **Wheel:** `mixle-0.8.2-py3-none-any.whl`, sha256
  `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da`
- **Commit / tree:** `866078be520b22188110780be957150dc6da964c` /
  `7922a8c59283ecef6877104b9a7499afd52d9013`
- **Work dir:** `REVIEW_ROOT/pass-08/` (paths below are relative to it)
- **Verification** (`cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, saved under
  `env/`), verbatim:

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
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
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
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

venv-nonumba was verified but not otherwise used. The 0.8.1 interpreter was used only for
before/after comparison. Every probe ran through `tools/pyt.py` from `/tmp` (PYTHONPATH unset,
`PYTHONWARNINGS=default`, 2 BLAS threads); each probe's stdout is saved beside its script. Machine
load during the campaign was 50-60 (18 cores shared), so wall times are not performance evidence.

## Corpus executed on the candidate

### Notebooks (venv-full, `tools/run_nb.sh`, 3 at a time, executed in place in `corpus/`; log `nb/status.log`)

| Notebook | exit | wall |
|---|---|---|
| tutorials/accelerated_engines.ipynb | 0 | 26 s |
| tutorials/bayesian_distributions.ipynb | 0 | 19 s |
| tutorials/distributions_and_combinators.ipynb | 0 | 143 s |
| tutorials/embedding_with_htsne.ipynb | 0 | 45 s |
| tutorials/enumeration.ipynb | 0 | 21 s |
| tutorials/estimation_using_spark.ipynb | 0 | 1803 s |
| tutorials/fitting_and_estimation.ipynb | 0 | 35 s |
| tutorials/latent_variable_models.ipynb | 0 | 1365 s |
| tutorials/mcmc_sampling.ipynb | 0 | 33 s |
| tutorials/model_parallel_estimation.ipynb | 0 | 23 s |
| tutorials/parallel_estimation.ipynb | 0 | 53 s |
| tutorials/probabilistic_programming.ipynb | 0 | 225 s |
| data_science/adaptive_mesh_refinement.ipynb | 0 | 27 s |
| data_science/cifar10_conv_net_and_exact_head.ipynb | 0 | 2008 s |
| data_science/reasoning_over_real_images.ipynb | 0 | 244 s |
| data_science/structured_neural_leaves.ipynb | 0 | 23 s |
| data_science/tiny_vision_language_model.ipynb | 0 | 387 s |

Stored-vs-fresh comparison (`nb/nbdiff.py`, one `nb/diff_<name>.txt` per notebook), cells whose
text differs / fresh errors: accelerated_engines 2/10 / 0; bayesian_distributions 0/7;
distributions_and_combinators 1/27; embedding_with_htsne 0/5; enumeration 0/8; estimation_using_spark
1/15; fitting_and_estimation 2/7; latent_variable_models 9/26; mcmc_sampling 3/12;
model_parallel_estimation 1/3; parallel_estimation 0/8; probabilistic_programming 7/14;
adaptive_mesh_refinement 2/11; cifar10 1/5; reasoning_over_real_images 0/5; structured_neural_leaves
0/4; tiny_vision_language_model 1/5. No fresh errors anywhere.

Fresh output contradicting a stored output or a prose claim:
- tutorials/probabilistic_programming: cell 3 prints `HMC ESS 1000` where the stored copy says
  `3000`; cell 8 prints `mean=1.00 var=9.00` where the stored copy says `1.02 / 8.95` beside the
  comment "exact" -- the wheel is right, the corpus is stale (Q08-F14).
- tutorials/latent_variable_models cells 3/10/12/14/16 and tutorials/estimation_using_spark cell 6
  now print the progress lines their `print_iter=` asks for; the stored copies have none (Q08-F14).
- tutorials/latent_variable_models cell 7: a new stderr warning that the 200-iteration fit stopped at
  iteration 8 "on a rejected update" of 5.46e-12 with `converged=False`, while every printed
  parameter equals the stored output (Q08-F06).
- Everything else is timing (cifar10 87.93% final accuracy and every epoch accuracy identical, 349 s
  stored vs 1855 s under load; tiny_vision 553 slots identical), a swapped state label
  (latent_variable_models cell 19 Viterbi path `[1 0 0 ...]` vs `[0 1 1 ...]` for the same
  segmentation), sampler noise (probabilistic_programming cells 1, 10, 12), or stream splitting.

### Examples

| Script | environment | exit | wall |
|---|---|---|---|
| examples/peft_lora_grad_leaf.py | venv-full | 0 | 569.2 s |
| examples/task_llm_active_example.py | venv-full | 0 | 191.5 s |
| examples/gallery_univariate_example.py (README base-install five) | venv-base | 0 | 22.1 s |
| examples/gallery_structured_example.py | venv-base | 0 | 140.5 s |
| examples/ppl_example.py | venv-base | 0 | 28.1 s |
| examples/production_example.py | venv-base | 0 | 58.5 s |
| examples/scaling_example.py | venv-base | 0 | 53.3 s |

(`examples/*.out`, `examples/base5/`.) peft: base-weight drift 0.0, 10/10 LoRA tensors moved,
held-out mean log-density -55.17 -> -51.22. task_llm: reports its paired test as inconclusive
(p = 0.125) and makes no superiority claim. gallery_structured and scaling print 7 and 2 cap-note
warnings (Q08-F16).

### README snippets (scripts `readme/s*.py`, outputs `readme/out/`)

| Snippet | venv-full | venv-base |
|---|---|---|
| Block 1 `optimize(records, out=None)` verbatim (`s1`) | exit 0; log_density -2.91258746535244; sampler draws; Composite(Gamma, Categorical, Optional(Categorical)); converged=True | identical numbers |
| Block 2 `solve(teacher, inputs)` with a rule teacher and 80 strings (`s2`) | works: answers `spam`, `report()` keys as documented, `save()` writes manifest.json + weights.safetensors; one distinct A-02 note in the library-built form (27 under `always`) | ImportError naming `pip install "mixle[torch]"` and `student="generative"`, as the prose says |
| Block 2 with `student="generative"` (`s2 generative`) | works (model.json); 2 cap notes per call (Q08-F02) | same |
| Block 3 `optimize(x, my_module)` with the obvious (N,1)-scoring module (`s3`) | fits: mu 2.9493 / sigma 1.9963 = sample mean/std; `model.module is my_module`; (N,3), (N,2), (1,N) refused naming `.sum(-1)`; a one-row batch fits | not run (no torch) |
| Block 4 nested HMM, 3 seeds, default and `max_its=300` (`s4`) | default: capped, both transition rows point at one state, neural sigma 3.7-6.6 (prose confirmed); 300: still capped and both rows at one state on 3/3 seeds (Q08-F10) | not run (no torch); 0.8.1 seed 0 digit-identical (`s4_hmm_nest_sum_*`) |
| Block 5 engine / precision / backend one-liners (`s5`, `s5b`, `s10`) | `TorchEngine(device="cuda")` -> AssertionError "Torch not compiled with CUDA enabled"; `device="cpu", dtype="float32"` -> fits but stops at 5/30 on a rejected update, converged=False (Q08-F06); `precision="auto"` = reference; spark -> TypeError needs an RDD; mp -> worker setup timeout 30 s under load 58 (inconclusive); dask, mpi = reference; ray, lightning -> ModuleNotFoundError; `bogus` -> ValueError listing 13 registered backends | TorchEngine -> ImportError naming `mixle[torch]`; precision auto = reference; spark TypeError; mp default workers -> unseparated, unconverged mixture, `num_workers=2` = reference (Q08-F05); dask -> ImportError needs dask.distributed; mpi -> ImportError naming `mixle[mpi]`; ray/lightning/bogus as on full |
| Block 6 enumeration on SmolLM2-135M verbatim (`s6`, `s6_f32`) | runs in bf16; top-3 strings match; `rank(unrank(5))` -> rank=4, cumulative 0.0915 (README: 6, 0.114); float32 reproduces the README (Q08-F11) | not run (no torch/transformers) |
| Block 7 PPL one-liners + regression with placeholders filled (`s7`) | four RandomVariables; regression prints `RV(fitted RegressionResult: x=1.986, z=-0.9743, intercept=0.4753)`; `explain_fit()` names the route | identical (HMM shows `use_numba=False`) |
| Prose claims (`s8`, `s9`): propose/fit=True, explain_fit, compare/waic/loo/Flow/MDN/VAE/Group/potential/Field/Mix/Markov, `.each`, constraints, planner, SymbolicEngine, EnumerationError, multi-chain R-hat/ESS in `m.result.summary()`, `compare(by='waic')` refusal on point fits | all hold; `register_encoded_data_backend` lives in `mixle.utils.parallel` (README names no module) | all hold (s9 not run on base) |
| docs/quickstart.rst transcribed in order (`probes/quickstart_docs.py`) | every printed line as the page says up to `create(...)`, which prints `0` and `False` (Q08-F09) | identical |

## Findings

### Q08-F01 -- real -- Fused EM declares convergence one iteration late: a fit that settles on its last permitted iteration is returned with `converged=False` and a cap note reading "before the objective settled (last objective gain 0)"

**Surface:** `mixle.inference.optimize` fused EM route (closed-form families: Categorical, Gaussian,
Poisson, ...); `fit_provenance().converged`; the cap `UserWarning`; reached by README's
`solve(..., student='generative')` through `mixle/task/generative_text.py:177`.

**Reproduction:**
```
cd /tmp && REVIEW_ROOT/venv-full/bin/python - <<'EOF'
import warnings, numpy as np
from mixle.inference import optimize
from mixle.stats import CategoricalEstimator, GaussianEstimator
words = ["a", "b", "a", "c", "b", "a"] * 10
normal = list(np.random.RandomState(0).normal(size=100))
for est, data, k in ((CategoricalEstimator(), words, 2), (CategoricalEstimator(), words, 3), (GaussianEstimator(), normal, 2), (GaussianEstimator(), normal, 10)):
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        m = optimize(data, est, max_its=k, out=None)
    fp = m.fit_provenance()
    print(type(est).__name__, "max_its=%d" % k, "converged=%s iterations=%s objective_gain=%r" % (fp.converged, fp.iterations, fp.objective_gain), [str(w.message)[:110] for w in c])
EOF
```
**Observed:** Categorical `max_its=2` -> `converged=False iterations=2 objective_gain=0.0` plus
"optimize() stopped at the max_its cap (2) before the objective settled (last objective gain 0,
delta=1e-09) ... Raise max_its ..."; `max_its=3` -> `converged=True iterations=3`, no note. Gaussian
`max_its=2` the same (gain 0.0, False); at the default it converges at iteration 3. Poisson
identical. Same on 0.8.1 (`repairs/out_gain0_capnote_081.txt`), venv-full and venv-base
(`repairs/out_gain0_capnote_venv-full.txt`).
**Expected:** A run whose last iteration gained less than `delta` is converged: `converged=True`, no
cap note; a note must not say the objective did not settle while quoting a gain of 0.
**Notes:** Mechanism (`mixle/inference/estimation.py`, fused loop ~1340-1390): each iteration scores
its input model and tests `dll = ll_model - prev_ll`, i.e. the previous step's gain, so convergence
at step k is only seen at step k+1; on exhaustion the post-loop fold recomputes
`objective_gain = final_ll - prev_ll` (0.0) but never re-evaluates `converged`. Pre-existing on
0.8.1; not ledgered (P03-F05 is the `delta=0` case). Repairs concerned: P08-F05, R07-F06.

### Q08-F02 -- real -- README's torch-free route `solve(teacher, inputs, student='generative')` emits two library-internal cap notes per call that the caller cannot silence, contradicting the migration guide's "library-internal capped calls no longer warn from inside distill"

**Surface:** `mixle.task.solve(student='generative')` (README Quickstart, "Distill" paragraph);
`mixle/task/generative_text.py:177` `optimize(by_class[lab] or [_UNK], est, max_its=2, out=None)`;
`docs/migrations/0.8.2.md` lines 39-40.

**Reproduction:**
```
cd /tmp && REVIEW_ROOT/venv-base/bin/python - <<'EOF'
import warnings
from mixle.task import solve
words = ["alpha beta", "gamma delta", "beta gamma", "delta alpha", "alpha gamma", "beta delta"]
inputs = [words[i % len(words)] + " " + str(i % 7) for i in range(120)]
def teacher(text): return "first" if text.startswith(("alpha", "beta")) else "second"
for kw in ({}, {"max_its": 50}, {"delta": None}):
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        sol = solve(teacher, inputs, seed=0, student="generative", **kw)
    caps = [str(w.message) for w in c if "max_its cap" in str(w.message)]
    print(kw, "cap notes:", len(caps), "|", caps[0][:200] if caps else None)
EOF
```
**Observed:** Every call: 2 x "solve() stopped at the max_its cap (2) before the objective settled
(last objective gain 0, delta=1e-09): the returned model is an unconverged fit, and its
fit_provenance() reports converged=False." -- also with `solve(max_its=50)` and `solve(delta=None)`
(`repairs/out_generative_capnote_venv-base.txt`); the README block-2 snippet with
`student="generative"` shows the same 2 notes on both venvs
(`readme/out/s2_solve_venv-*_generative.txt`). The default torch student shows 0 cap notes.
**Expected:** No note: the call is library-internal, deliberately capped, and settled (gain 0); the
migration guide says such calls no longer warn.
**Notes:** The internal call passes `max_its=2` without `delta=None`, so it hits Q08-F01 on every
class and warns onto the reader's line through `_caller_stacklevel`; R07-F06 then strips the
"Raise max_its / pass delta=None" clauses because `solve()` has neither, leaving a false statement
with no action. Repairs concerned: P08-F05, R07-F06, R07-F07.

### Q08-F03 -- real -- The A-02 note gives the caller-built remedy (`restarts=` / `MixtureEstimator(..., init='dirichlet')`) on routes where the caller built nothing, and `restarts=` is not accepted by `optimize()`, `MixtureEstimator()` or `Mix().fit()`

**Surface:** A-02 note (`component_row_mass` disclosure) via `optimize(data)` with no estimator
(README Quickstart route), `mixle.ppl Mix([...]).fit(data)` (README PPL block, line 241),
`optimize(data, MixtureEstimator(...))`, `best_of()`; regression test
`UnidentifiedComponentRemedyTest` in `adversarial_review_082_repairs_test.py`.

**Reproduction:**
```
cd /tmp && REVIEW_ROOT/venv-full/bin/python - <<'EOF'
import warnings, numpy as np
from mixle.inference import optimize, best_of
from mixle.stats import MixtureEstimator, GaussianEstimator
from mixle.ppl import Normal, Mix, free
rows = list(np.random.RandomState(0).normal(0.0, 1.0, 400)) + [40.0]
def remedy(call):
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        call()
    n = [str(w.message) for w in c if "less data than they have parameters" in str(w.message)]
    return (len(n), n[0][-170:]) if n else (0, None)
print("optimize(rows) no estimator      :", remedy(lambda: optimize(rows, max_its=60, out=None, rng=np.random.RandomState(0))))
print("Mix([N,N]).fit(rows)             :", remedy(lambda: Mix([Normal(free, free), Normal(free, free)]).fit(rows)))
print("optimize(rows, MixtureEstimator) :", remedy(lambda: optimize(rows, MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), max_its=60, out=None, rng=np.random.RandomState(0))))
print("best_of(rows, ..., MixtureEst)   :", remedy(lambda: best_of(rows, rows[:50], MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), 2, 60, 1.0, 1e-8, np.random.RandomState(0))))
for label, fn in (("optimize(..., restarts=3)", lambda: optimize(rows, MixtureEstimator([GaussianEstimator()] * 2), restarts=3, max_its=5)),
                  ("MixtureEstimator(..., restarts=3)", lambda: MixtureEstimator([GaussianEstimator()] * 2, restarts=3)),
                  ("Mix([N,N]).fit(rows, restarts=3)", lambda: Mix([Normal(free, free), Normal(free, free)]).fit(rows, restarts=3, max_its=5)),
                  ("Mix([N,N]).fit(rows, init='dirichlet')", lambda: Mix([Normal(free, free), Normal(free, free)]).fit(rows, init="dirichlet", max_its=5))):
    try: fn(); print(label, "-> accepted")
    except Exception as e: print(label, "->", type(e).__name__, str(e)[:90])
EOF
```
**Observed:** `optimize(rows)` with no estimator (the library's `get_estimator` built the mixture)
and `Mix([...]).fit(rows)` both end with "... and restarts= or MixtureEstimator(..., init='dirichlet')
distinguishes the two." `optimize(rows, MixtureEstimator(...))` gets the same advice, but
`optimize(..., restarts=3)`, `MixtureEstimator(restarts=3)`, `Mix.fit(restarts=3)` and
`Mix.fit(init='dirichlet')` all raise TypeError. `best_of(rows, ..., MixtureEstimator(...))` says
"best_of() built this mixture itself" although the caller passed the estimator
(`repairs/out_r07f07_routes_venv-full.txt`, `repairs/out_r07f07_knobs_venv-full.txt`).
**Expected:** R07-F07's own rule -- only knobs the reader can turn; the library-built form where the
library chose the mixture; `restarts=` only where it exists (`Model.fit`, `learn_mixture_structure`).
**Notes:** `_calling_verb` decides "caller-built" from the verb's signature rather than from whether
an estimator was supplied; the PPL fit lowers to `optimize()` and inherits it. The regression test
asserts the literal `restarts= or MixtureEstimator(..., init='dirichlet')` string on the optimize
route, pinning the nonexistent knob. Repairs concerned: R07-F07, A-02, R07-F06.

### Q08-F04 -- real -- `learn_mixture_structure` prints 1533 A-02 warning lines to stderr for one call on 401 rows under the default warning filters

**Surface:** `mixle.inference.structure.learn_mixture_structure` (library-built route of the A-02
note, R07-F07).

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-08/repairs/p_lms_default_filters.py > /tmp/q08f04.txt; grep -c 'UserWarning: this mixture fit left' /tmp/q08f04.txt; grep -c 'built this mixture itself' /tmp/q08f04.txt
```
**Observed:** 1533 stderr lines "UserWarning: this mixture fit left 1 of 2 component(s) with less
data than they have parameters ... learn_mixture_structure() built this mixture itself ..." for
`learn_mixture_structure(pairs, 2, restarts=1, seed=0, max_its=60)` on 400 N(0,1) pairs plus one
outlier (`repairs/out_lms_default_filters_venv-full.txt`, `PYTHONWARNINGS=default`). Under
`always` the call records 2206 notes with 6 distinct texts; a direct `optimize(rows,
MixtureEstimator)` shows 1 (`repairs/out_r07f07_knobs_venv-full.txt`).
**Expected:** One note per returned model (or per distinct text); the default once-per-location
filter should collapse repeats.
**Notes:** Every candidate/restart fit inside the search raises the note; 1533 shown for 6 distinct
texts means they are attributed to different locations -- not traced further. Surface belongs to
pass 07; reached while verifying R07-F07's library-built branch. Repairs concerned: R07-F07, A-02, A-03.

### Q08-F05 -- real -- `backend='mp'` changes the EM trajectory with the worker count: at the default `num_workers` the README block-5 two-Gaussian mixture needs 182 iterations and is returned unseparated and unconverged at `max_its=30`, while local and `num_workers=2/4/8` converge in 7-19

**Surface:** `optimize(backend='mp')` (README "Engines & scale" one-liner), base install;
`mixle/utils/parallel/multiprocessing.py`.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && PYT_CWD=$R/pass-08/readme/out $R/tools/pyt.py 900 $R/venv-base/bin/python $R/pass-08/readme/s10_mp_workers.py
```
**Observed** (`readme/out/s10_mp_workers_venv-base.txt`, `os.cpu_count()=18`): same data (300 N(0,1)
+ 300 N(6,1)), estimator and `rng=RandomState(0)`: local `max_its=30` -> converged=True, 8
iterations, objective -1249.4711, components (0.033, var 1.019) and (6.079, var 0.880); mp
`num_workers=2/4/8` -> converged=True in 19/7/12 iterations, same objective and components; mp
default workers `max_its=30` -> converged=False at 30, objective -1544.7422, components (2.83, var
10.05) and (3.32, var 10.00), weights 0.547/0.453, cap note; mp default workers `max_its=300` ->
converged only at iteration 182. Also `readme/out/s5b_mp_venv-base.txt` (`same as ref: False`). On
venv-full every mp attempt hit "parallel worker 0 timed out after 30.000s during setup" under load
average 58 (inconclusive there).
**Expected:** Same seed, data and estimator -> the same trajectory whatever the shard count, or at
least comparable iteration counts; README readers pin the seed, not the CPU count.
**Notes:** Not root-caused; two near-identical components that take ~180 iterations to separate is
what a symmetric initialization produces, suggesting the initial estimate is formed per shard when
600 rows are split 18 ways. The 30 s setup timeout is `MPWorkerPool(response_timeout=30.0)`
(`multiprocessing.py:150`) and is not reachable from `optimize()`.

### Q08-F06 -- real -- EM step acceptance uses an absolute 1e-12 tolerance, below one ulp of the objective, so rounding noise ends a run as a "rejected update": tutorial `latent_variable_models` cell 7 stops at 8 of 200 with `converged=False` on a 5.46e-12 wobble, and the README `TorchEngine(dtype='float32')` one-liner stops at 5 of 30 on a 1e-3 float32 wobble

**Surface:** `mixle/inference/estimation.py` `_em_loop` (`accepted = ... dll >= -1.0e-12 ...`) and
the fused loop; `tutorials/latent_variable_models.ipynb` code cell 7; README "Engines & scale"
`TorchEngine(device=..., dtype="float32")`.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-08/probes/p_lvm_cell7_rejected.py; $R/venv-full/bin/python - <<'EOF'
import warnings, numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.engines import TorchEngine
d = list(np.random.RandomState(0).normal(0, 1, 300)) + list(np.random.RandomState(1).normal(6, 1, 300))
est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
for kw in ({}, {"engine": TorchEngine(device="cpu", dtype="float32")}):
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        m = optimize(d, est, max_its=30, rng=np.random.RandomState(0), out=None, **kw)
    fp = m.fit_provenance(); print(kw, "converged=%s its=%s" % (fp.converged, fp.iterations), [str(w.message)[:150] for w in c])
EOF
```
**Observed:** Cell 7 (composite 3-component mixture, 800 rows, `max_its=200`): "optimize() stopped
at iteration 8 of max_its=200 on a rejected update: the proposal fell 5.46e-12 below the last
accepted objective while the last accepted step still gained 0.731 > delta=1e-09 ... More of the
same update would not help."; `fit_provenance`: converged=False, iterations=8,
final_objective=-10797.659647, `np.spacing(final_objective)=1.82e-12` (the "decrease" is 3 ulps);
continuing from the returned model with `prev_estimate=` moves the objective by exactly 0 and no
parameter changes (a fixed point); the Gaussian fields equal the stored notebook output to every
printed digit (`probes/out_lvm_cell7_rejected_venv-full.txt`). float32 TorchEngine on CPU:
"stopped at iteration 5 of max_its=30 on a rejected update: the proposal fell 0.000998 below the
last accepted objective while the last accepted step still gained 0.000623 > delta=1e-09 ...
converged=False", while the float64 run converges (`readme/out/s5_engines_venv-full.txt`).
**Expected:** Monotonicity judged at the objective's precision (relative to |ll| and the engine
dtype); a step within rounding noise is accepted or treated as converged; no note claims a genuine
descent or that more updates "would not help".
**Notes:** The stored corpus copy prints the same parameters with no warning, so the truncation
predates the note; the new R-round disclosure now reports a stop that is itself wrong. The fused
loop's post-loop fold uses the same absolute constant. `fit_provenance()` exposes no
`rejected_decrease` field although the note quotes one. Repair concerned: P02-F06.

### Q08-F07 -- real -- Migration guide says `how='map'` is refused when parameters are unpinned; `Normal(free, free).fit([1.0], how='map')` is not refused -- venv-full returns a 1e-8-variance point mass and venv-base raises a raw RuntimeError; `how='vi'`, which the guide says still fits, also raises RuntimeError on venv-base

**Surface:** `mixle.ppl RandomVariable.fit(how='map'|'vi')`; `docs/migrations/0.8.2.md`
"Probabilistic programs and surrogates", last bullet; CHANGELOG P10-F13.

**Reproduction:**
```
cd /tmp && for V in REVIEW_ROOT/venv-full/bin/python REVIEW_ROOT/venv-base/bin/python; do $V - <<'EOF'
import mixle; print(mixle.__path__[0])
from mixle.ppl import Normal, free
for how in ("map", "vi", "mcmc", "laplace"):
    try: print(how, "->", Normal(free, free).fit([1.0], how=how))
    except Exception as e: print(how, "->", type(e).__name__, str(e)[:150])
EOF
done
```
**Observed:** venv-full: map -> `GaussianDistribution(1.0, 1.0000000000000017e-08)`; vi ->
`GaussianDistribution(0.92, 1563624565.68)`; mcmc, laplace -> ValueError "RandomVariable has 2
parameter(s) with no prior (arg0, arg1) but was given 1 observation(s): nothing pins them ...".
venv-base: map -> RuntimeError "inference model construction or likelihood evaluation failed at
unconstrained coordinates [1.0, -374.6974625775336]"; vi -> RuntimeError "... [-7.6e+115, 365.9]";
mcmc/laplace refused as on full (`probes/out_migration_venv-full.txt` vs
`probes/out_migration_venv-base.txt`).
**Expected:** The named ValueError for `how='map'` on both installs, as the guide lists it among
the refused routes; identical behaviour on base and full for a numpy-only model.
**Notes:** Surface owned by pass 06/07; the migration-guide sentence is in scope here. On venv-full
the MAP route evidently resolves to closed-form EM (variance floored at 1e-8) and skips the guard;
on venv-base it goes numerical and fails before the guard. Repair concerned: P10-F13.

### Q08-F08 -- blocking -- R05-F07 claims every family that defines `quantile` refuses an out-of-domain `q`; `UniformDistribution` and `LaplaceDistribution` still return NaN for q = -0.5, 1.5 and NaN, unchanged from 0.8.1

**Surface:** `UniformDistribution.quantile`, `LaplaceDistribution.quantile`; CHANGELOG 0.8.2
R05-F07 ("Every family that defines `quantile` refuses an out-of-domain `q` the same way");
`docs/migrations/0.8.2.md` ("`quantile(q)` refuses a `q` outside `[0, 1]` on every family").

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 300 $R/venv-full/bin/python $R/pass-08/probes/p_quantile_uniform_laplace.py; $R/tools/pyt.py 300 <review-store>/candidate-081/venv/bin/python $R/pass-08/probes/p_quantile_uniform_laplace.py
```
**Observed:** 0.8.2: `UniformDistribution(0,1).quantile(-0.5)` -> nan, `(1.5)` -> nan, `(nan)` ->
nan; `LaplaceDistribution(0,1).quantile(-0.5|1.5|nan)` -> nan; `GaussianDistribution(0,1).quantile(1.5)`
-> ValueError "GaussianDistribution.quantile: q must be in [0, 1]." 0.8.1: Uniform and Laplace
identical (nan), Gaussian nan (`probes/out_quantile_uniform_laplace_venv-full.txt`, `_081.txt`;
also `probes/out_migration_venv-full.txt`, where the other eleven probed families raise).
**Expected:** ValueError naming the family, as the CHANGELOG and migration guide state for every
family.
**Notes:** Rated blocking because a 0.8.2 repair claim (R05-F07) is false on two documented
families reachable by an ordinary user; the consequence is the pre-repair silent NaN, so the fix is
small (route both through `validated_quantile_probability`). Surface owned by pass 01. Repair
concerned: R05-F07.

### Q08-F09 -- real -- docs quickstart "Create a Certified Artifact": `create(rows, calibrate=0.2, quantify_uq=True, seed=0)` on the page's own rows fails its requested UQ internally, records it only in `postconditions` with no warning, and prints `Guarantee.UNVERIFIED` (0) and `is_calibrated()` False that the page never explains

**Surface:** `mixle.inference.create` (`docs/quickstart.rst` lines 219-231);
`CreatedModel.guarantee` / `is_calibrated()` / `postconditions`.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 600 $R/venv-full/bin/python $R/pass-08/probes/p_quickstart_create.py
```
**Observed** (`probes/out_quickstart_create_venv-full.txt`): warnings `[]`; guarantee
`<Guarantee.UNVERIFIED: 0>`; `is_calibrated()` False; `is_certified()` False; postconditions:
calibration performed=True; uq requested=True, performed=False, error
"NotImplementedError: laplace_posterior cannot flatten a SequenceDistribution; add it to
_leaf_flatteners (the same per-family extend point as register_family), or use the model's bespoke
inference."; calibration report n=48, method 'log-density', "model has no scalar predictive CDF;
PIT calibration not applicable"; certificate blocks field[0] GLOBAL_UNIQUE, the mixture and its
components and field[2] UNVERIFIED. The page transcription prints the same `0` / `False` on both
venvs (`probes/out_quickstart_venv-*.txt`).
**Expected:** A requested post-condition that could not be performed is reported to the caller
(`create.py`'s own comment: "non-fatal, but never silent"), and the quickstart shows or explains
what its two print lines produce on its own data.
**Notes:** The page's rows carry a count sequence per record -- exactly the leaf the UQ lowering
cannot flatten -- so the documented call never delivers the UQ it asks for. Surface owned by pass 10.

### Q08-F10 -- docs -- README nested-HMM budget note: at the README's own `max_its=300` the flagship snippet is still capped and unconverged and the transition matrix still has both rows pointing at one state, on 3 of 3 seeds, identically on 0.8.1 and 0.8.2

**Surface:** README "Compose to any depth" block (`max_its=300`) and the paragraph "`max_its` is
the one knob that is not optional here ... the number to raise is `max_its`" (P08-F02 repair).

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 1500 $R/venv-full/bin/python $R/pass-08/readme/s4_hmm_nest.py 3; $R/tools/pyt.py 1500 <review-store>/candidate-081/venv/bin/python $R/pass-08/readme/s4_hmm_nest_sum.py 1
```
**Observed:** Data as in the P08-F02 ledger entry (60 sequences x 30 steps; state A = five clusters
at -20/-10/0/10/20, state B = N(12,1), switch 0.15). Default: converged=False,
T=[[0.88,0.12],[0.818,0.182]], neural sigma 3.66 -- as the prose says. `max_its=300`: seed 0
converged=False at 300, T=[[0.911,0.089],[0.834,0.166]], mixture means
[-20.72,-19.72,-10.08,-0.02,11.49], neural mu=19.92 sigma=1.04, cap note; seeds 1 and 2 likewise
unconverged with both rows pointing at state 0 (`readme/out/s4_hmm_nest_venv-full.txt`). The
published 0.8.1 gives digit-identical numbers for seed 0 (`readme/out/s4_hmm_nest_sum_081_seed0.txt`
vs `s4_hmm_nest_sum_venv-full_seed0.txt`).
**Expected:** The README's remedy produces what its prose implies (a converged two-state HMM near
[[0.85,0.15],[0.15,0.85]] with the neural leaf on the N(12,1) regime), or the prose says 300 may
still stop at the cap in another basin and what to do then.
**Notes:** Not a library change (0.8.1 identical), so docs only. Repair concerned: P08-F02.

### Q08-F11 -- docs -- README enumeration snippet's printed numbers (rank=6, cumulative_prob=0.114, "here: 6, not 5") only reproduce with float32 weights; as written, `from_pretrained` loads SmolLM2-135M in bfloat16 and prints rank=4, cumulative_probability=0.0915

**Surface:** README "Enumeration & ranking" block.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-08/readme/s6_enum.py; $R/tools/pyt.py 900 $R/venv-full/bin/python $R/pass-08/readme/s6_enum_f32.py
```
**Observed:** Verbatim: llm dtype torch.bfloat16; `top_k(3)` -> `[' located in the', ' the city of',
' the capital of']` (matches); `unrank(5)` -> ' situated in the'; `rank(unrank(5)[0])` ->
`DensityRankResult(cumulative_probability=0.0915, rank=4, exact=True)`. With
`dtype=torch.float32`: `unrank(5)` -> ' Paris, the'; rank=6, cumulative_probability=0.11396, matching
the README (`readme/out/s6_enum_venv-full.txt`, `s6_enum_f32_venv-full.txt`).
**Expected:** The snippet pins the dtype it was measured with, or the comment carries the bf16 numbers.
**Notes:** P08-F01 (bf16 logits rejected) is repaired; the documented numbers moved with the
precision. Repair concerned: P08-F01.

### Q08-F12 -- docs -- README "Tests" command `python -m pytest -m full -m ""` is described as "everything non-optional", but pytest keeps only the last `-m`, so it selects every test including optional and benchmark tiers

**Surface:** README "Tests" section; `mixle/tests/README.md` (whose full-gate spelling is
`-m "slow and not optional and not benchmark"`).

**Reproduction:**
```
cd REVIEW_ROOT/pass-08/tests/mdemo && V=REVIEW_ROOT/venv-full/bin/python; $V -m pytest -p no:randomly -p no:cacheprovider -m full -m "" --collect-only -q demo_test.py; $V -m pytest -p no:randomly -p no:cacheprovider -m full --collect-only -q demo_test.py; $V -m pytest -p no:randomly -p no:cacheprovider -m "not optional" --collect-only -q demo_test.py
```
**Observed:** On a 3-test file (one `full`, one `optional`, one unmarked): `-m full -m ""` collects
3/3 (identical to `-m ""`); `-m full` collects 1/3; `-m "not optional"` collects 2/3
(`tests/mdemo_out.txt`).
**Expected:** A command whose selection matches its description.
**Notes:** pytest registers `-m` with `action='store'`. The README review record did not check
this command.

### Q08-F13 -- docs -- Migration guide sentences not true of the wheel: `potentials=` is listed among `optimize()` arguments that raise TypeError by name (optimize has no such parameter), and "No name was added, renamed, or removed" while the guide itself lists added names (+79/-3 module-level names measured)

**Surface:** `docs/migrations/0.8.2.md` lines 5 and 33; `optimize` signature; module-level names of
the 779 importable mixle modules.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && $R/venv-full/bin/python -c "import inspect; from mixle.inference import optimize; print('potentials' in inspect.signature(optimize).parameters)"; $R/venv-base/bin/python $R/pass-08/probes/names_diff.py $R/pass-08/probes/names_081.json $R/pass-08/probes/names_082.json | tail -n 4
```
**Observed:** `'potentials' in optimize's parameters` -> False; `optimize(data, potentials=5)` ->
TypeError "optimize() got an unexpected keyword argument 'potentials'" (Python's generic message;
`probes/out_migration_venv-full.txt`). Names 0.8.1 vs 0.8.2: 779 modules each, none added or
removed, 79 module-level names added (e.g. `mixle.ppl.regression.predict_at`,
`mixle.stats.compute.kernel.HAS_NUMBA`, `validate_drift_thresholds`, and the
`surrogate_repairs`/`numerical_repairs` the guide itself announces), 3 removed (stdlib re-exports
`Real`, `math`, `Mapping`) (`probes/out_names_diff.txt`).
**Expected:** The optimize() bullet names the surface that takes `potentials=` (the PPL sampler
fit, CHANGELOG P04-F11); the "no name added" sentence is scoped to the documented public API.
**Notes:** The compatibility half holds: every 0.8.1 pickle/JSON/str model the cross-version probe
wrote loads and scores identically on 0.8.2 (`probes/xver/out_load_venv-*.txt`). Repair concerned:
P04-F11.

### Q08-F14 -- docs -- mixle-notebooks d99d296 stored outputs still show the pre-repair numbers for P08-F04, P08-F08 and P08-F14 that the wheel now prints correctly

**Surface:** corpus `tutorials/probabilistic_programming.ipynb` cells 3 and 8;
`tutorials/latent_variable_models.ipynb` cells 3/10/12/14/16 and `estimation_using_spark.ipynb`
cell 6 (the `print_iter` cells).

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && W=$R/pass-08 && $R/venv-full/bin/python $W/nb/nbdiff.py $R/corpus/notebooks/tutorials/probabilistic_programming.ipynb $W/corpus/notebooks/tutorials/probabilistic_programming.ipynb --show 4; $R/venv-full/bin/python $W/nb/nbdiff.py $R/corpus/notebooks/tutorials/latent_variable_models.ipynb $W/corpus/notebooks/tutorials/latent_variable_models.ipynb --show 3
```
**Observed:** probabilistic_programming cell 3: stored `HMC  acc 1.00  ESS 3000`, fresh `ESS 1000`
(1000 draws); cell 8: stored `mean=1.02 var=8.95`, fresh `mean=1.00 var=9.00` beside "-> Normal(1,
9), exact". latent_variable_models cells 3/10/12/14/16 and spark cell 6: stored outputs carry no
progress lines although the cells pass `print_iter=50/25/10`; fresh copies print them
(`nb/diff_probabilistic_programming.txt`, `nb/diff_latent_variable_models.txt`,
`nb/diff_estimation_using_spark.txt`).
**Expected:** Stored outputs that match the wheel the corpus is pinned to.
**Notes:** Stale in the direction opposite to 0.8.1's P08-F16: the library is right and the corpus
is behind. Repairs concerned: P08-F04, P08-F08, P08-F14.

### Q08-F15 -- minor -- An (n, 1) column into `PoissonEstimator` raises numpy's "shapes (50,1) and (50,) not aligned" instead of the named `numpy.ravel` message the CHANGELOG claims for all thirty univariate families

**Surface:** `PoissonEstimator` via `optimize` (CHANGELOG P01-F13/P06-F08; migration guide "Latent
models" bullet 2).

**Reproduction:**
```
cd /tmp && REVIEW_ROOT/venv-full/bin/python - <<'EOF'
import numpy as np
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, PoissonEstimator
for est, col in ((GaussianEstimator(), np.random.RandomState(0).normal(size=(50, 1))), (PoissonEstimator(), np.random.RandomState(0).poisson(3.0, size=(50, 1)))):
    try: print(type(est).__name__, "->", optimize(col, est, out=None))
    except Exception as e: print(type(est).__name__, "->", type(e).__name__, str(e)[:170])
EOF
```
**Observed:** Gaussian -> ValueError "GaussianDataEncoder expects a one-dimensional sequence of
scalar observations, but the data has shape (50, 1). Flatten it with numpy.ravel(data) ...";
Poisson -> ValueError "shapes (50,1) and (50,) not aligned: 1 (dim 1) != 50 (dim 0)"
(`probes/out_migration_venv-full.txt`, identical on venv-base).
**Expected:** The named message on Poisson as on Gaussian.
**Notes:** Only Gaussian and Poisson probed. Surface owned by pass 01. Repairs concerned: P01-F13,
P06-F08.

### Q08-F16 -- minor -- The README's five base-install examples print nine "stopped at the max_its cap" UserWarnings in a fresh process (gallery_structured_example 7, scaling_example 2), from deliberate `max_its=1/10/40` calls that do not pass `delta=None` as the note instructs

**Surface:** `examples/gallery_structured_example.py` lines 53, 60, 82, 89, 105, 156, 164;
`examples/scaling_example.py` line 37 (README "Examples"); P08-F05.

**Reproduction:**
```
cd /tmp && R=REVIEW_ROOT && mkdir -p $R/pass-08/examples/cwd && for ex in gallery_structured_example scaling_example; do PYT_CWD=$R/pass-08/examples/cwd $R/tools/pyt.py 1800 $R/venv-base/bin/python $R/source/examples/$ex.py > /tmp/q08_$ex.txt; echo "$ex: $(grep -c 'UserWarning: optimize() stopped' /tmp/q08_$ex.txt) cap notes, $(tail -n 1 /tmp/q08_$ex.txt)"; done
```
**Observed:** All five exit 0 (`examples/base5/summary.txt`). gallery_structured: 7 notes -- six
"stopped at the max_its cap (1|40) before the objective settled" and, at line 105, "stopped at
iteration 9 of max_its=10 on a rejected update ... an unconverged fit"; scaling: 2 notes at
`max_its=1` (`examples/base5/*.venv-base.out`).
**Expected:** Examples the README sends a new user to run clean: a fixed iteration count passes
`delta=None`, and the genuinely unconverged fit gets its budget.
**Notes:** The CHANGELOG says R07-F07 "closes the open half of P08-F05"; the examples' own capped
calls were that finding's other expectation. Repair concerned: P08-F05.

### Q08-F17 -- minor -- The P08-F17 regression tests (`TorchModuleShapeTest`) fail with ModuleNotFoundError instead of skipping on an install without torch

**Surface:** `mixle/tests/notebook_surface_repairs_test.py` `TorchModuleShapeTest` (2 tests);
`RegressionSurfaceTest` in the same file skips correctly without pandas.

**Reproduction:**
```
cd REVIEW_ROOT/pass-08/tests && REVIEW_ROOT/venv-base/bin/python -m pytest -p no:randomly -q -m "" -n 0 notebook_surface_repairs_test.py -k TorchModuleShapeTest
```
**Observed:** venv-base: 2 failed (`torch = __import__("torch")` -> ModuleNotFoundError), 20 passed,
4 skipped for the whole file; venv-full: 26 passed (`tests/pytest_results.txt`).
**Expected:** `@unittest.skipUnless(HAS_TORCH, ...)` like the file's pandas-gated class.
**Notes:** Tests are not shipped in the wheel, so this bites only a base-environment run of the
source suite. Repair concerned: P08-F17.

### Q08-F18 -- minor -- Fitted regression `predict(given)` raises a raw KeyError for a missing covariate, while `fit(given=)` names the missing field

**Surface:** `mixle.ppl RandomVariable.predict` on a `RegressionResult` fit (P08-F15 repair).

**Reproduction:**
```
cd /tmp && REVIEW_ROOT/venv-base/bin/python - <<'EOF'
import numpy as np
from mixle.ppl import Normal, Field, free
rng = np.random.RandomState(0); x, z = rng.normal(size=300), rng.normal(size=300)
y = (2.0 * x - 1.0 * z + 0.5 + rng.normal(0.0, 0.3, 300)).tolist()
m = Normal(free * Field("x") + free * Field("z") + free, free)
f = m.fit(y, given={"x": x, "z": z}); print(f)
for label, fn in (("fit(given missing z)", lambda: m.fit(y, given={"x": x})), ("predict(given missing z)", lambda: f.predict({"x": [1.0]})), ("predict(scalars)", lambda: f.predict({"x": 1.0, "z": 0.0}))):
    try: print(label, "->", fn())
    except Exception as e: print(label, "->", type(e).__name__, str(e)[:120])
EOF
```
**Observed:** fit(given missing z) -> ValueError "given is missing required field(s): ['z'].";
predict(given missing z) -> KeyError 'z'; predict(scalars) -> ValueError "given['x'] must be a
non-empty finite numeric one-dimensional sequence." (`repairs/out_f15_venv-full.txt`,
`out_f15_venv-base.txt`). The repair itself holds (fitted repr; DataFrame `given=`, including a
filtered non-0..n index, matches the dict fit; `predict` draws at the covariates: mean of 2000
draws at x=1, z=0 is 2.471 for truth 2.5).
**Expected:** `predict` validates `given` the way `fit` does and names the missing field.
**Notes:** `log_density(y, given=...)` remains unsupported (TypeError), as on 0.8.1; the CHANGELOG
does not claim it. Repair concerned: P08-F15.

## Attacks that did not break anything

- **P08-F04 (`print_iter`)** every spelling on venv-full, venv-base and 0.8.1
  (`repairs/out_f04_*.txt`): default and `print_iter=None` quiet; `print_iter=1` prints one line
  per iteration to stdout, `=2` every second, `=0` none, `np.int64(1)` accepted, `-1`/`True`/`False`/`1.0`
  refused by name; `out=` alone keeps the old cadence; `out=` plus `print_iter` is not duplicated to
  stdout; `best_of`, `mixle.Model(est).fit` and `Mix(...).fit` honour it; 0.8.1 printed nothing
  for `print_iter=1` (the finding as ledgered). `learn_mixture_structure` does not take it (TypeError).
- **P08-F08 (ESS)** `repairs/out_f08_*.txt`: hmc 1000 draws -> ESS 1000 (0.8.1: 3000); nuts 500 ->
  372.8/500 (0.8.1: 716.4); 2 and 4 chains of 500 -> 1000 and 2000 (0.8.1: 2699, 5397.9); random-walk
  and ensemble chains still report far fewer (153.5, 134.3); a synthetic exact-antithetic chain is
  clipped to n=2000 (0.8.1: 6602) -- by design per the CHANGELOG, noted as a choice, not a defect.
- **P08-F14 (affine moments)** `repairs/out_f14_*.txt`: five affine forms, nested to four levels,
  exact to the digit at both `RV` and `TransformDistribution` level; `exp()` refuses at the
  distribution level with a NotImplementedError naming the affine exception and `RV.mean()` falls
  back to a deterministic sample; 0.8.1 printed 1.0248/8.9512.
- **P08-F15 / P08-F17 / R07-F07 on `solve()`, `create()`, `learn_mixture_structure`**: see the
  README table and Q08-F18; the library-built wording is correct on those three verbs and the
  observation half of the note is identical on every route.
- **Migration guide, verified true** (`probes/out_migration_*.txt`, `probes/migration_claims.py`):
  Exponential/Poisson/Geometric/Uniform/LogSeries refusals name the family and support (and NaN is
  reported as missing data, not a support violation); a zero-weight negative row is still exempt;
  eleven of thirteen probed families refuse an out-of-domain `q` (the two exceptions are Q08-F08);
  `optimize` refuses empty data, zero-row `enc_data`, `delta=0`, wrong-typed `rng`/`prev_estimate`,
  `str`/`bytes`, masked arrays, `numpy.matrix`, 0-d arrays, and normalizes a mapping of columns, a
  structured array and a one-shot iterator (200 rows fitted); `Model.fit`/`evaluate`/`propose`
  share the refusals; mixtures and HMMs refuse a `SequenceDistribution` without a length model;
  `HMM.seq_posterior` returns (2, 12, 2) arrays on the base install and with `filtered=True`;
  the A-02 note carries `component_row_mass` and variance floors reach `fit_provenance().repairs`;
  `learn_bayesian_network` refuses empty/ragged input; `propose` refuses a one-row training split;
  `Service(keep=0|-1|2.0|True)` and `Registry.checkpointer(every=0|-2, resume='yes'|1)` refuse by
  name; `NumbaKernelFactory().build` raises `KernelCapabilityDeclinedError` on venv-base and
  `GeneratedNumbaKernelFactory` falls back to `GenericKernel`; an all-constant `.fit()` warns and
  returns `result=None`; the sampler/Laplace guard fires for mcmc and laplace and lets a
  prior-pinned or two-observation fit through; `BayesOptResult.numerical_repairs()` and
  `surrogate_repairs` exist (absent on 0.8.1). Every 0.8.1 pickle, `to_json` and `str` form of five
  fitted models plus two pickled PPL fits load and score identically on both 0.8.2 installs
  (`probes/xver/`).
- **README claims** (`readme/out/s8_*`, `s9_*`): `propose(data)`/`fit=True`, `explain_fit()`,
  every named `mixle.ppl` object, `.each`, constraints, `mixle.utils.parallel.planner`,
  `SymbolicEngine`, `EnumerationError`, `compare([a, b], data)` by AIC/BIC (misuse `compare(a, b)`
  is refused with a message naming the list form), `compare(..., by='waic'|'loo')` on point fits
  refuses "use plugin_log_likelihood(data), AIC, or BIC instead" and ranks Bayesian fits,
  `m.result.summary()` carries `r_hat`, `split_r_hat`, `ess_bulk`, `ess_tail`. Every local link
  target exists (`docs/backend-support.rst`, `release-checklists/0.8.0-cuda-receipt.json`,
  `CHANGELOG.md`, `LICENSE`, `mixle/tests/README.md`, the five `docs/*.rst` pages); CI runs
  ubuntu-latest and macos-14 on 3.11 and 3.12; `requires-python = ">=3.11,<3.13"`; 15,905 `def test_`
  functions in 1288 test files (the record's 16,849 collected was not re-collected); every extra
  named in the README exists in `pyproject.toml`.
- **Review records:** `readme-review.md` and `documentation-review.md` remain true of this tree for
  what they checked; the README `Tests` command (Q08-F12) and the enumeration numbers (Q08-F11)
  were outside what they verified.
- **Regression tests** on the wheel: `notebook_surface_repairs_test.py` 26 passed on venv-full;
  R07-F06/F07 classes of `adversarial_review_082_repairs_test.py` 6 passed on both venvs.
- **Backends:** `backend="bogus"` lists the 13 registered names; an unguarded script with
  `backend="mp"` on macOS has its spawned children print the bootstrapping RuntimeError while the
  parent completes (`readme/out/s5c_mp_unguarded_venv-base.txt`) -- the README does not mention the
  `__main__` guard; noted, not filed.
- **Neural/vision notebooks:** all five reproduce their stored numbers (cifar10 epoch accuracies and
  87.93% final; tiny_vision 220 scenes / 553 slots; adaptive_mesh L2 errors and angles) apart from
  wall time.

## What was not covered

- `backend="mp"` on venv-full: every attempt timed out in worker setup (30 s, not adjustable from
  `optimize()`) under a load average of 50-60; not retried on a quiet machine, so Q08-F05 is
  measured on venv-base only.
- `TorchEngine(device="cuda")` (no CUDA here) and the README's "device= changes the answer" Metal
  measurement.
- The "BEST iterate" claim of the going-down cap note for a torch module mutated in place: the probe
  (`repairs/p_gradleaf_best.py`) never reached that branch, and the peft example, which did, was not
  instrumented.
- `mixle.inference.diagnostics.ess` called directly with a `(chains, draws)` array was refused
  ("requires at least two finite draws"); its shape convention was not pursued.
- Migration-guide items whose probes were malformed and are therefore inconclusive: `Monitor(...)`
  threshold validation (constructor called without `estimator`/`reference`), the terminal-state HMM
  `seq_posterior` and the lookback HMM `terminal_states` refusal (models built without a path to
  termination / without per-state initials).
- Migration-guide items not probed at all: `LDAEstimator`'s alpha warning, the terminal-state HMM
  `latent_posterior`/`viterbi`/`mode`/`sample` changes, the symmetry-broken HMM initialization,
  per-prompt seeds in `mixle.task`, registry/checkpoint/deploy corrupt-file refusals, GP jitter
  escalation, and the refusal of a sampler that accepted no proposals.
- `(n, 1)` columns into families other than Gaussian and Poisson; `quantile` on families beyond
  the thirteen probed.
- Docs-site URLs (`gmboquet.github.io/mixle/v0.8.2/...`) were not fetched; only the corresponding
  local `docs/*.rst` files were checked to exist.
- The stored-vs-fresh comparison of `estimation_using_spark` beyond cell 6 and the Spark
  stage-progress stderr; the notebook took 1803 s under load (exit 0), no retry needed.

DONE 08 18 findings
