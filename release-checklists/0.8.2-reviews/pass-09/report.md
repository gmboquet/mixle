# Pass 09 -- example scripts 1-28 (auto_example ... hidden_association_example) and the repairs they exercise

**Recovery note.** The pass-09 reviewer ran the 28 examples on venv-full and 26 of them on venv-base, wrote 20 probe scripts (`probes/p01`-`p17`) with saved outputs, and was stopped by an API rate limit before writing this report. This report was written by a recovery reviewer from that preserved evidence (`RECOVERED_NOTES.md` is the original reviewer's command trail); probes `p17b`-`p22c` were added by the recovery reviewer to finish pending items and to confirm or reject findings (a 0.8.1 comparison where a regression question arose). No example was re-run.

- **Focus:** examples `auto_example`, `autoregressive_enumeration_example`, `calibrated_report_demo`, `capability_layer_example`, `copula_vine_example`, `cross_modal_fit_receipt`, `doe_example`, `engine_benchmark_example`, `enumeration_example`, `enumeration_showcase_example`, `extensibility_seams_example`, `flagship_kg_agent`, `flagship_physics_inverse`, `flagship_triage_app`, `frontier_ecosystem_demo`, `frontier_family_showcase`, `gallery_combinators_example`, `gallery_directional_example`, `gallery_graphs_example`, `gallery_multivariate_example`, `gallery_processes_example`, `gallery_rankings_example`, `gallery_structured_example`, `gallery_univariate_example`, `geoscience_inversion_report`, `heterogeneous_correctness_example`, `heterogeneous_representation_example`, `hidden_association_example`; repairs P09-F03, P09-F04, P09-F05, P09-F08, P09-F09, P09-F10, P09-F12, P09-F13, plus A-01/P09-F07/P09-F11, R07-F05, R07-F06, R02-F13, P10-F13 and P05-F17 as the examples exercise them.
- **Wheel:** `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da`
- **Commit:** `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013`
- **Work dir:** `REVIEW_ROOT/pass-09/` (probe scripts in `probes/`, every probe and example output in `out/`, example working directories in `cwd/full/` and `cwd/base/`)
- **Verification** (`cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, saved verbatim as `out/00_verify_env.txt`; the last block is the published 0.8.1 used only for before/after comparison; venv-nonumba was verified but nothing was run on it):

```
=== venv-full ===
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
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
=== venv-nonumba ===
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

## Corpus executed on the candidate

Notebooks: none assigned to this pass.

Examples (`REVIEW_ROOT/tools/pyt.py 1800 <venv>/bin/python REVIEW_ROOT/source/examples/<name>.py`, `PYT_CWD=cwd/<env>/<name>`, `MPLBACKEND=Agg`, three at a time on a shared 18-core machine, so wall times are inflated by contention and are not performance evidence; exit code and wall seconds from the `== exit` line of `out/ex_<env>_<name>.txt`):

| example | venv-full exit | venv-full wall s | venv-base exit | venv-base wall s |
|---|---|---|---|---|
| auto_example | 0 | 14.5 | 0 | 28.3 |
| autoregressive_enumeration_example | 0 | 13.6 | 0 | 23.9 |
| calibrated_report_demo | 0 | 20.5 | 1 | 29.8 |
| capability_layer_example | 0 | 25.7 | 0 | 27.4 |
| copula_vine_example | 0 | 94.5 | 0 | 141.4 |
| cross_modal_fit_receipt | 0 | 41.2 | 0 | 56.4 |
| doe_example | 0 | 129.8 | 1 | 28.2 |
| engine_benchmark_example | 0 | 67.7 | 0 | 34.3 |
| enumeration_example | 0 | 41.3 | 0 | 24.9 |
| enumeration_showcase_example | 0 | 38.9 | 0 | 29.5 |
| extensibility_seams_example | 0 | 47.7 | 0 | 26.1 |
| flagship_kg_agent | 0 | 66.1 | 0 | 13.4 |
| flagship_physics_inverse | 0 | 644.4 | 0 | 217.2 |
| flagship_triage_app | 0 | 107.5 | 0 | 17.3 |
| frontier_ecosystem_demo | 0 | 153.7 | 0 | 26.0 |
| frontier_family_showcase | 0 | 769.7 | not run (module-level `import torch`, out/p08_base_imports_base.txt) | - |
| gallery_combinators_example | 0 | 99.4 | 0 | 11.4 |
| gallery_directional_example | 0 | 100.6 | 0 | 12.2 |
| gallery_graphs_example | 0 | 93.2 | 0 | 14.3 |
| gallery_multivariate_example | 0 | 102.3 | 0 | 15.0 |
| gallery_processes_example | 0 | 159.8 | 0 | 29.1 |
| gallery_rankings_example | 0 | 120.5 | 0 | 19.4 |
| gallery_structured_example | 0 | 214.9 | 0 | 39.7 |
| gallery_univariate_example | 0 | 101.4 | 0 | 8.5 |
| geoscience_inversion_report | 0 | 234.5 | 1 | 21.7 |
| heterogeneous_correctness_example | 0 | 120.1 | 0 | 10.0 |
| heterogeneous_representation_example | 0 | 209.7 | not run (module-level `import torch`, out/p08_base_imports_base.txt) | - |
| hidden_association_example | 0 | 516.9 | 0 | 186.2 |

**Tally:** venv-full 28/28 exit 0. venv-base 23/26 exit 0 of the 26 run (the 3 failures are `calibrated_report_demo`, `doe_example`, `geoscience_inversion_report`, all on the missing optional `torch`; see Q09-F03); `frontier_family_showcase` and `heterogeneous_representation_example` were not run there because they import torch at module level (`out/p08_base_imports_base.txt`: the former with a clean `ImportError: ... requires torch: pip install 'mixle[torch]'`, the latter with a bare `ModuleNotFoundError`).

Fresh output versus stored claims:
- `gallery_univariate_example`, `gallery_structured_example`: raw stdout sha256 (`238593732ace...`, `691caecdd76b...`) match `release-checklists/0.8.2-repro-bundle.json` exactly, every `contains` string present, input hashes match (`out/p12_bundle_sha.txt`).
- `engine_benchmark_example`: the docstring's "the table compares equal work" contradicts the run -- GMM row 5/5/3 iterations, HMM row 5/5/4 (Q09-F06).
- `calibrated_report_demo`: the comment "at 0.05 it must abstain on the low-margin (faint) volumes" contradicts the gate it builds, which serves 145/180 faint volumes at the demo's seed and nothing at 7 of 10 other seeds (Q09-F04).
- `autoregressive_enumeration_example`: "Exact count / rank / unrank" contradicts the documented quantized `unrank` ordering (Q09-F16).
- `gallery_combinators_example`: prints its DiracLengthMixture fit under a class name that does not exist (Q09-F09).
- `geoscience_inversion_report`: now serves the calibrated report with passing SBC/coverage receipts, as the P09-F04 repair claims; the execution manifest's narrative still describes the old abstain path (Q09-F15).
- All other examples: prose and numbers consistent with the fresh run.

## Findings

Counts: 1 blocking, 4 real, 9 minor, 2 docs.

### Q09-F01 · blocking · 0.8.2 regression: a TreeHiddenMarkovModelDistribution fitted by optimize() can no longer be JSON-serialized -- dump_models/to_json refuse with 'does not match its constructor-owned schema' because the fit leaves a populated _p_level_cache; the same fit round-trips on 0.8.1

**Surface:** mixle.stats.TreeHiddenMarkovModelDistribution fitted through optimize(TreeHiddenMarkovEstimator) (any max_its, cold or warm start); mixle.stats.dump_models/load_models and to_json/from_json; examples/gallery_structured_example.py (its TreeHiddenMarkov section fits exactly this)

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python (0.8.2), cwd /tmp. The same script on candidate-081/venv (0.8.1) prints "JSON OK" on every line.
import warnings, numpy as np
from mixle.inference import optimize
from mixle.stats import (TreeHiddenMarkovModelDistribution, TreeHiddenMarkovEstimator, GaussianDistribution,
                         GaussianEstimator, PoissonDistribution, PoissonEstimator, dump_models, load_models)
warnings.simplefilter("ignore")
d = TreeHiddenMarkovModelDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5],
                                      [[0.8, 0.2], [0.2, 0.8]], len_dist=PoissonDistribution(0.8), terminal_level=3)
x = d.sampler(1).sample(150)
f = optimize(x, TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], len_estimator=PoissonEstimator()),
             max_its=8, rng=np.random.RandomState(1), out=None)
for label, m in (("constructed", d), ("fitted", f)):
    try:
        load_models(dump_models(m)); print(label, "JSON OK")
    except Exception as e:
        print(label, "JSON", type(e).__name__, str(e)[:160])
print("fitted _p_level_cache:", type(f._p_level_cache).__name__, "| constructed:", type(d._p_level_cache).__name__)
f._p_level_cache = None
load_models(dump_models(f)); print("fitted with _p_level_cache=None: JSON OK")
```

**Observed:** 0.8.2 (venv-full): the constructed instance round-trips; the fitted one raises SerializationError "dump_models produced JSON that load_models cannot read back (SerializationError: serialized state for 'mixle.stats.latent.tree_hidden_markov_model.TreeHiddenMarkovModelDistribution' does not match its constructor-owned schema; define __pysp_setstate__ for validated custom state). Refusing to return a write-only serialization ..." on max_its=1, max_its=8 and a warm start (prev_estimate=d); to_json/from_json raises the same; pickle round-trips; every nested component (topics[0], topics[1], len_dist) round-trips alone. The fitted object's _p_level_cache is a tuple (bytes, bytes, ndarray(13, 2)) where the constructed one's is None; setting it to None, or copying the constructed value, makes the identical fitted object round-trip, while clearing _fit_provenance does not. 0.8.1 (candidate-081): every one of those fits round-trips (its _p_level_cache is None after the fit). Outputs: out/p21_fitted_json_full.txt, out/p21_fitted_json_081.txt, out/p22b_treehmm_json_regression_full.txt, out/p22b_treehmm_json_regression_081.txt, out/p22c_treehmm_cache_full.txt.

**Expected:** A fitted TreeHMM persists through the documented safe route exactly as it did in 0.8.1: a derived-constant cache is either excluded from the serialized state (as the 0.8.2 P05-F03 repair did for .header) or declared in the constructor-owned schema.

**Notes:** Mechanism measured, not read from source: the fit populates a per-level probability memo (_p_level_cache) that the serializer writes and the schema check rejects. The cache is new in 0.8.2 (None after a 0.8.1 fit), so it arrived with this release's TreeHMM work -- the CHANGELOG's only TreeHMM entries are the A-01/P09-F07 symmetry-breaking start -- and whichever change added it did not extend the schema. Rated blocking as a regression that crashes the documented persistence route for a model the gallery itself fits. The refusal is loud (no write-only text escapes) and pickle still works, which is why it is not silent data loss. Same defect class as P05-F03 (an attribute the fit attaches makes the model unserializable).

**Repairs concerned:** A-01, P09-F07

### Q09-F02 · real · Seven gallery families are write-only for JSON once FITTED: estimate()/optimize() attach fit_metadata, which the constructor-owned schema rejects (VonMises, WrappedNormal, WrappedCauchy, ProjectedNormal, Watson, VonMisesFisher, IntegerBernoulliSet); KnowledgeGraphDistribution has no JSON form even when constructed -- all pre-existing in 0.8.1, none on the P05-F17 list

**Surface:** mixle.stats.dump_models/load_models and to_json/from_json on fitted VonMisesDistribution, WrappedNormalDistribution, WrappedCauchyDistribution, ProjectedNormalDistribution, WatsonDistribution (examples/gallery_directional_example.py fits all five), VonMisesFisherDistribution and IntegerBernoulliSetDistribution (examples/gallery_multivariate_example.py), KnowledgeGraphDistribution (examples/gallery_graphs_example.py)

**Reproduction:**

```
# any 0.8.2 venv (numpy only), cwd /tmp; identical on candidate-081/venv (0.8.1)
import numpy as np
from mixle.inference import estimate
from mixle.stats import (VonMisesDistribution, WrappedNormalDistribution, WrappedCauchyDistribution, ProjectedNormalDistribution,
                         WatsonDistribution, VonMisesFisherDistribution, IntegerBernoulliSetDistribution, KnowledgeGraphDistribution,
                         GaussianDistribution, dump_models, load_models)
for d in (VonMisesDistribution(0.7, 4.0), WrappedNormalDistribution(0.7, 0.8), WrappedCauchyDistribution(0.7, 0.6),
          ProjectedNormalDistribution(1.5, -0.5), WatsonDistribution([0.0, 0.0, 1.0], 4.0), VonMisesFisherDistribution([0.6, 0.8], 8.0),
          IntegerBernoulliSetDistribution(np.log([0.8, 0.5, 0.1, 0.3])), GaussianDistribution(0.0, 1.0)):
    f = estimate(d.sampler(seed=1).sample(300), d.estimator())
    try:
        load_models(dump_models(f)); print(type(d).__name__, "fitted: JSON OK")
    except Exception as e:
        print(type(d).__name__, "fitted: JSON refused; attrs the fit added:", sorted(set(vars(f)) - set(vars(d))), "|", str(e)[:110])
try:
    dump_models(KnowledgeGraphDistribution(np.random.RandomState(0).randn(6, 3), np.random.RandomState(1).randn(2, 3)))
except Exception as e:
    print("KnowledgeGraph (constructed):", str(e)[:220])
```

**Observed:** On 0.8.2 and identically on 0.8.1: each fitted directional/set model raises SerializationError "... serialized state for '<class>' does not match its constructor-owned schema; define __pysp_setstate__ for validated custom state ... Refusing to return a write-only serialization ..."; the only attribute the fitted object carries that a constructed one lacks is fit_metadata (plus _fit_provenance on the optimize route, which is harmless: a Gaussian fitted by optimize round-trips). The constructed instances of all seven round-trip. KnowledgeGraphDistribution refuses even when constructed: "requires a class-owned __pysp_setstate__ hook; constructor fields are absent: entity_embeddings, relation_embeddings". Pickle works for all eight; the Gaussian control round-trips on every route. Outputs: out/p07_gallery_roundtrip_full.txt, out/p20_writeonly_json_full.txt, out/p20_writeonly_json_081.txt, out/p21_fitted_json_full.txt, out/p21_fitted_json_081.txt.

**Expected:** The fits the galleries print persist through dump_models/to_json, the CHANGELOG's stated safe route (0.8.2 lists eight more families repaired under P05-F17/P09-F05/P10-F05), or the families are not listed as serializable.

**Notes:** Same defect class as P05-F03 (.header) and P05-F17 (six write-only families), on families neither ledger entry names; not a regression (0.8.1 identical). The refusal is loud, so no write-only text escapes; the cost is that the directional gallery's fits cannot be persisted except by pickle, and the error text's remedy (__pysp_setstate__) is the library's to apply, not the user's.

**Repairs concerned:** P05-F17, P09-F05, P05-F03

### Q09-F03 · real · Base install (numpy+scipy only): solve_structured spends every teacher call (361 for 360 records) and learn_inverse every simulator call (1500 of 1500) before dying on a bare "ModuleNotFoundError: No module named 'torch'"; learn_inverse(family='flow') likewise simulates 1500 times before rejecting a 1-D theta -- while doe.minimize and task.solve refuse up front, naming the extra

**Surface:** mixle.task.structured_out.solve_structured (numeric field -> solve_regression -> regress._fit_reg_mlp, regress.py:157); mixle.task.inverse.learn_inverse (_build_student -> inverse.py:78; the family='flow' dimensionality check at inverse.py:911); examples/calibrated_report_demo.py and examples/geoscience_inversion_report.py (both exit 1 on venv-base)

**Reproduction:**

```
# REVIEW_ROOT/venv-base/bin/python (numpy+scipy only), cwd /tmp
import importlib.util, sys, numpy as np
R = "<review-root>"
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{R}/source/examples/{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
crd, geo = load("calibrated_report_demo"), load("geoscience_inversion_report")
from mixle.task.structured_out import solve_structured
calls = {"n": 0}
def teacher(rec):
    calls["n"] += 1
    return crd.claim_teacher(rec)
train = crd.build_records(n_per_shape=120, seed=0)
try:
    solve_structured(teacher, train, tol={"brightness": 0.08}, alpha=0.1, seed=0, epochs=200)
except Exception as e:
    print("solve_structured:", type(e).__name__, e, "| teacher calls spent:", calls["n"], "of", len(train))
from mixle.task.inverse import learn_inverse
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
sims = {"n": 0}; forward_rng = np.random.RandomState(9 + 7919)
def forward_salt(theta_row):
    sims["n"] += 1
    clean = np.asarray(geo._amplitude(float(np.ravel(theta_row)[0]), geo.TRUE_FORMATION), dtype=float)
    return clean + geo.SENSOR_NOISE * forward_rng.randn(clean.shape[0])
y_obs = np.asarray(geo._amplitude(4.2, "salt"), dtype=float) + geo.SENSOR_NOISE * np.random.RandomState(123).randn(3)
for family in ("mdn", "flow"):
    sims["n"] = 0
    try:
        learn_inverse(forward_salt, GaussianDistribution(mu=4.518, sigma2=0.480 ** 2), family=family, n_sims=1500, n_sbc_replications=150,
                      coverage_levels=(0.5, 0.9), y_obs=y_obs, rounds=1, seed=9, m_steps=200, max_its=1)
    except Exception as e:
        print(f"learn_inverse(family={family!r}):", type(e).__name__, str(e)[:80], "| simulator calls spent:", sims["n"], "of 1500")
from mixle.doe import minimize
try:
    minimize(lambda p: float(p[0] ** 2), [(-1.0, 1.0)], n_init=3, n_iter=2, seed=0)
except Exception as e:
    print("doe.minimize:", type(e).__name__, "...", str(e)[-58:])
```

**Observed:** venv-base: calibrated_report_demo exits 1 with a raw traceback ending "ModuleNotFoundError: No module named 'torch'" (regress.py:157) after the teacher was called 361 times for 360 records; geoscience_inversion_report exits 1 the same way (inverse.py:78) after learn_inverse ran the simulator 1500 times; family='flow' with a 1-D theta also simulates 1500 times before "family='flow' requires theta ... >= 2-dimensional" (inverse.py:911). On the same install mixle.doe.minimize raises ImportError "... requires the optional torch dependency, which is not installed. Install it with pip install \"mixle[torch]\". Refusing before any objective evaluations are spent." and task.solve raises ImportError naming the torch-free student, both before any teacher call (0 of 200). Outputs: out/p14_solve_structured_base.txt, out/p19_geo_base_spend_base.txt, out/ex_base_calibrated_report_demo.txt, out/ex_base_geoscience_inversion_report.txt.

**Expected:** The torch requirement (and the flow dimensionality precondition) is checked before the metered teacher or simulator is spent, with the same install hint the sibling routes give; the manifest's own policy for these two examples is 'Execute with optional-dependency status recorded'.

**Notes:** Not a wrong result, so not blocking: the fix is the documented optional install. But the base environment is what pip install mixle gives, the brief's requirement there is 'fail cleanly', and the spend is real when the teacher is an LLM or the simulator a physics code. The categorical-only solve_structured path does refuse with the good ImportError (from task.distill) -- but also only after the 361 teacher calls. P09-F09 touched learn_inverse's optimize() call, not its import path.

**Repairs concerned:** none

### Q09-F04 · real · calibrated_report_demo's shape gate is a seed lottery: at 7 of 10 calibration seeds it certifies threshold=+inf and serves 0 of 120 volumes (receipt: target_attainable=True, accepted=0); at the demo's own seed it serves 145 of 180 FAINT volumes, although the source comment and the 0.8.1 CHANGELOG say faint volumes abstain

**Surface:** examples/calibrated_report_demo.py (build_shape_gate at alpha=0.05, calibration seed 1; the comment in main()); mixle.task.calibrated_generator.CalibratedGenerator.calibrate / serve / risk_receipt

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python, cwd /tmp
import importlib.util, sys
R = "<review-root>"
spec = importlib.util.spec_from_file_location("crd", R + "/source/examples/calibrated_report_demo.py")
d = importlib.util.module_from_spec(spec); sys.modules["crd"] = d; spec.loader.exec_module(d)
from mixle.task.calibrated_generator import ABSTAIN
probe = d.build_records(40, 2, ambiguous_fraction=0.5)          # the demo's own probe set
for cal_seed in range(10):                                        # the demo calibrates with cal_seed=1
    gate = d.build_shape_gate(d.build_records(120, cal_seed, ambiguous_fraction=0.5), alpha=0.05, seed=0)
    rr = gate.risk_receipt
    print(f"cal seed {cal_seed}: threshold={rr['threshold']} accepted={rr['accepted']} target_attainable={rr['target_attainable']}"
          f" served {sum(gate.serve(r) is not ABSTAIN for r in probe)}/120")
gate = d.build_shape_gate(d.build_records(120, 1, ambiguous_fraction=0.5), alpha=0.05, seed=0)
for amb, label in ((0.0, "clear"), (1.0, "faint")):
    recs = d.build_records(60, 3, ambiguous_fraction=amb)
    print(label, "volumes served:", sum(gate.serve(r) is not ABSTAIN for r in recs), "/", len(recs))
```

**Observed:** cal seeds 1, 2, 9: threshold 0.0140 / 0.0155 / 0.0095, accepted 165 / 165 / 170 of 180, served 113 / 109 / 118 of 120; cal seeds 0, 3, 4, 5, 6, 7, 8: threshold 'inf', accepted 0, served 0/120 -- every receipt carrying 'target_attainable': True and 'attainable_error_upper': 0.0445. At the demo's seed (1): clear volumes served 180/180, faint volumes served 145/180 (142 of the 145 agree with the teacher). The scorer's top-1 error on the 50/50 mix is 3.3%-6.1% depending on the seed. Outputs: out/p11_calgate_full.txt, out/p11b_calgate_seeds_full.txt.

**Expected:** Per the example's own comment ("at 0.05 it must abstain on the low-margin (faint) volumes to certify, which is the behaviour this demo is about") and the 0.8.1 CHANGELOG ("clear volumes are served while faint ones abstain"): a gate that reproducibly serves clear volumes and abstains on faint ones.

**Notes:** The library's certificate does what it says: with 180 certification rows and 181 Bonferroni-corrected thresholds, a zero-error accepted set of about 163 rows is needed for the Clopper-Pearson bound to clear alpha=0.05 (attainable_error_upper=0.0445 at 180 rows), so with ~4% scorer error nothing certifies unless every error falls below the top-163 score cut -- a coin flip the demo's seed happens to win. The defect is the example's parameterisation and narrative (P09-F02's 0.8.1 repair made the demo pass at one seed), plus a receipt that says target_attainable=True beside accepted=0 with no field separating 'attainable in principle' from 'attained'. gate.serve(seed=3) is refused as the docstring promises; empty and 1-row calibration sets are refused cleanly; an all-wrong oracle yields qhat=inf and abstains.

**Repairs concerned:** P09-F02

### Q09-F05 · real · The P10-F13 fewer-observations-than-parameters guard covers the sampler and laplace routes only: Normal(free, free).fit([1.0], how='vi') returns GaussianDistribution(-887.6, 1.18e12) silently -- the very 'wander to scales that overflow' the guard's message describes -- while how='mcmc' and how='laplace' refuse the same call

**Surface:** mixle.ppl RandomVariable.fit(how='vi') (also how='map'/'auto', which return a variance-floored point mass GaussianDistribution(1.0, 1e-08) for the same call); the guard mixle/ppl/inference.py _refuse_fewer_observations_than_parameters is reached only through _prepare_target (the shared sampler setup)

**Reproduction:**

```
# any 0.8.2 venv with torch (venv-full), cwd /tmp
import numpy as np
from mixle.ppl import Normal, free
for how in ("mcmc", "laplace", "vi", "map"):
    kw = {"draws": 50, "burn": 10, "rng": np.random.RandomState(0)} if how == "mcmc" else {}
    try:
        m = Normal(free, free).fit([1.0], how=how, **kw)
        print(how, "->", m.dist)
    except Exception as e:
        print(how, "-> refused:", type(e).__name__, str(e)[:110])
```

**Observed:** mcmc -> refused: ValueError "RandomVariable has 2 parameter(s) with no prior (arg0, arg1) but was given 1 observation(s): nothing pins them, and a fit will wander to scales that overflow double precision rather than converging. Give those parameters a prior, fit a simpler model, or supply at least 2 observations."; laplace -> the same refusal; vi -> GaussianDistribution(-887.6399276279334, 1182441064506.2285) with summary std 174396 and no warning; map and auto -> GaussianDistribution(1.0, 1e-08). Zero observations are refused on every route ("fit() received empty data"). Output: out/p05_physics_ess_full.txt.

**Expected:** Every route refuses the state the guard names, or the variational route at least warns; one spelling of the same fit must not return a wandering posterior that its twin refuses.

**Notes:** The CHANGELOG's claim is scoped to 'a sampler fit', so the repair itself is not false; the gap is that the unguarded twin produces the failure the guard was written for. The physics example's own shape (a prior-carrying k with one observation) passes on every route, as it should.

**Repairs concerned:** P10-F13

### Q09-F06 · minor · float32 torch-mps engine: the monotone acceptance gate treats a roundoff-level objective decrease (0.0232 on -42042.9) as a rejected update, so engine_benchmark's GMM fit stops at iteration 3 of 10 with converged=False and 'more of the same update would not help' while numpy/torch-cpu converge at 5 -- the docstring's 'the table compares equal work' is not what runs (HMM row: 4 vs 5 iterations)

**Surface:** mixle.inference.optimize(engine=TorchEngine(device='mps')) (float32 by construction); examples/engine_benchmark_example.py docstring and its GMM/HMM rows

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python on an Apple-silicon Mac (MPS), cwd /tmp
import importlib.util, warnings, numpy as np
R = "<review-root>"
spec = importlib.util.spec_from_file_location("eb", R + "/source/examples/engine_benchmark_example.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
from mixle.inference import optimize
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator
rng = np.random.RandomState(0); n = 20_000
data = np.concatenate([rng.normal(-4, 1, n // 2), rng.normal(4, 1, n // 2)]).tolist()
init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
for name, engine in eb._engines():                       # numpy, torch-cpu (float64), torch-mps (float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = optimize(data, est, max_its=10, engine=engine, prev_estimate=init, out=None)
    p = m.fit_provenance()
    print(name, "iterations", p.iterations, "converged", p.converged,
          [(round(float(c.mu), 4), round(float(c.sigma2), 4)) for c in m.components], [str(x.message)[:140] for x in w])
```

**Observed:** GMM max_its=10: numpy iterations=5 converged=True; torch-cpu 5 / True; torch-mps 3 / False with UserWarning "optimize() stopped at iteration 3 of max_its=10 on a rejected update: the proposal fell 0.0232 below the last accepted objective while the last accepted step still gained 27.3 > delta=1e-09. The returned model is the last accepted one, an unconverged fit, and its fit_provenance() reports converged=False. More of the same update would not help."; the mps components (-4.0181, 0.9764) / (4.0114, 0.9846) equal the converged numpy fit to four decimals. HMM max_its=5: numpy and torch-cpu 5 iterations, torch-mps 4 (rejected, fell 0.000477). With delta=None mps still stops at 3. The example's own stderr shows the same two rejections. Outputs: out/p18_engine_warnings_full.txt, out/p13_engine_iters_full.txt, out/ex_full_engine_benchmark_example.txt.

**Expected:** A rejection tolerance scaled to the engine's precision (the example's docstring says the HMM accumulator's mass check was scaled exactly for this reason), or a note that names float32 roundoff instead of advising restarts; and a benchmark row that reports the iteration count it timed.

**Notes:** The returned model is right; the receipt (converged=False) and the advice are what mislead, and the timing ratio understates the mps per-iteration cost by 40% (GMM) / 20% (HMM). The rejected-update note itself is the P02-F06/FU-03 mechanism working as designed on float64.

**Repairs concerned:** P02-F06

### Q09-F07 · minor · Fused EM at the max_its cap: the final step's gain is computed for the receipt but never tested against delta, so a fit whose last gain (3.86e-10) is below delta (1e-09) is reported converged=False, and the cap note reads 'before the objective settled (last objective gain 3.86e-10, delta=1e-09)'

**Surface:** mixle.inference.optimize (fused EM loop in mixle/inference/estimation.py: the post-loop 'exhausted' block sets trace.objective_gain = final_ll - prev_ll without re-testing convergence); engine_benchmark_example's HMM row on the numpy and torch-cpu engines

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python, cwd /tmp
import warnings, numpy as np
from mixle.inference import optimize
from mixle.engines import NUMPY_ENGINE
from mixle.stats import GaussianDistribution, GaussianEstimator, HiddenMarkovEstimator, HiddenMarkovModelDistribution
rng = np.random.RandomState(0)
seqs = [[float(rng.normal(-4 if rng.rand() < 0.5 else 4, 1.0)) for _ in range(20)] for _ in range(400)]
init = HiddenMarkovModelDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = optimize(seqs, HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()]), max_its=5, engine=NUMPY_ENGINE, prev_estimate=init, out=None)
p = m.fit_provenance()
print("iterations", p.iterations, "converged", p.converged, "objective_gain", p.objective_gain, "delta", p.delta)
print([str(x.message) for x in w])
```

**Observed:** iterations=5 converged=False objective_gain=3.8562575355172157e-10 delta=1e-09, with UserWarning "optimize() stopped at the max_its cap (5) before the objective settled (last objective gain 3.86e-10, delta=1e-09): the returned model is an unconverged fit, and its fit_provenance() reports converged=False. Raise max_its to fit to convergence, or pass delta=None to request a fixed iteration count without this note."; torch-cpu: gain 3.93e-10, the same note. Output: out/p18_engine_warnings_full.txt.

**Expected:** A receipt whose converged flag agrees with its own numbers -- a last gain below delta at the cap is convergence -- or a note that does not quote a gain below the delta it says was not met.

**Notes:** Mechanism read in the source: the fused loop judges convergence on the PREVIOUS step's gain (ll_model scores the step's input model), and when the loop is exhausted the final step's gain is folded into the receipt post hoc with no convergence test, so a fused fit needs max_its+1 iterations to notice a convergence that happened at its last step. The plain loop tests the accepted step's gain every iteration and does not have this.

**Repairs concerned:** none

### Q09-F08 · minor · Following the cap note's own remedy (delta=None) on a fit that converges before the cap produces a second warning that calls the converged fit 'not a converged fit': GMM max_its=10, delta=None stops at 5 with gain 1.4e-10 and advises a restart

**Surface:** mixle.inference.optimize(delta=None) -- the documented MXR-080-T3-02 rejected-step exit and its note; engine_benchmark_example's GMM fit

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python, cwd /tmp
import warnings, numpy as np
from mixle.inference import optimize
from mixle.engines import NUMPY_ENGINE
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator
rng = np.random.RandomState(0); n = 20_000
data = np.concatenate([rng.normal(-4, 1, n // 2), rng.normal(4, 1, n // 2)]).tolist()
init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
for delta in ({}, {"delta": None}):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = optimize(data, est, max_its=10, engine=NUMPY_ENGINE, prev_estimate=init, out=None, **delta)
    p = m.fit_provenance()
    print(delta or "default delta", "-> iterations", p.iterations, "converged", p.converged, "gain", p.objective_gain, [str(x.message) for x in w])
```

**Observed:** default delta: iterations=5 converged=True gain=1.38e-10, no warning. delta=None: iterations=5 converged=False gain=1.38e-10 with UserWarning "optimize() was called with delta=None (documented as \"a fixed iteration count\": run max_its=10 iterations) but only 5 of them ran: a proposed update was rejected (a non-improving or non-finite step), which still ends the loop even when delta=None. This is not a converged fit -- more of the SAME update would not help, though a different restart/initialization might -- and fit_provenance() reports iterations=5, max_iterations=10, converged=False; compare the two to see the shortfall."; torch-cpu: 6 of 10 ran, same text. Output: out/p18_engine_warnings_full.txt.

**Expected:** The delta=None note distinguishes a fixed point reached (a rejection after a ~1e-10 gain) from a mid-trajectory rejection, or stays silent in the former case; a remedy the library advises should not produce a contradictory diagnosis on the same fit.

**Notes:** The source docstring documents the delta=None early exit (MXR-080-T3-02) and the note is deliberate; what is wrong is its text for the converged case. Distinct from Q09-F07 (different loop, different bookkeeping).

**Repairs concerned:** none

### Q09-F09 · minor · DiracLengthMixtureDistribution.__str__ prints 'LengthDiracMixtureDistribution(...)', a class name that exists nowhere in the package; gallery_combinators prints it as the fit

**Surface:** mixle.stats.DiracLengthMixtureDistribution.__str__ (mixle/stats/latent/dirac_length.py:134); examples/gallery_combinators_example.py output line 'fit: LengthDiracMixtureDistribution(...)'

**Reproduction:**

```
# any 0.8.2 venv, cwd /tmp
import mixle.stats as st
from mixle.stats import DiracLengthMixtureDistribution, PoissonDistribution
d = DiracLengthMixtureDistribution(PoissonDistribution(6.0), p=0.7, v=0)      # the gallery's constructor call
print(type(d).__name__, "->", str(d))
print("mixle.stats exports LengthDiracMixtureDistribution:", hasattr(st, "LengthDiracMixtureDistribution"))
```

**Observed:** The gallery's fit line is "fit: LengthDiracMixtureDistribution(len_dist=PoissonDistribution(5.984901209041194, name=None, keys=None), p=0.710788674936453, v=0, name=None)" under the heading "# DiracLengthMixture" (out/ex_full_gallery_combinators_example.txt); grep -rn LengthDiracMixture over the source finds only that __str__ line, and mixle/stats/__init__.py exports DiracLengthMixtureDistribution only, so the printed name resolves to nothing.

**Expected:** str() names the class it prints, as every other family's does, so the printed fit can be looked up and re-constructed.

**Notes:** Prose-level defect in code, not in docs; the JSON/pickle round trip of the fitted DiracLengthMixture is fine (out/p07_gallery_roundtrip_full.txt).

**Repairs concerned:** none

### Q09-F10 · minor · The P09-F13 public replacements accept malformed input silently or fail inside: design_row() with too few parent values returns a wrong-length row; RandomVariable.fitted(None) returns RV(bound=None); a @register_fitter function returning a bare Distribution dies with AttributeError "'GaussianDistribution' object has no attribute '_cache'"

**Surface:** mixle.inference.bayesian_network _LinearGaussianFactor.design_row (returns self._row(values) unchecked); mixle.ppl.core.RandomVariable.fitted; mixle.ppl.core.register_fitter / RandomVariable.fit(how=<custom>)

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python, cwd /tmp
import numpy as np
from mixle.inference import learn_bayesian_network
from mixle.ppl import Normal, free
from mixle.ppl.core import RandomVariable, register_fitter
from mixle.stats import GaussianDistribution
rng = np.random.RandomState(0)
rows = [(("pro" if rng.rand() > 0.5 else "free"), tuple(rng.normal(0, 1, 3)), tuple(rng.normal(0, 1, 3)), 0.0) for _ in range(120)]
rows = [(a, b, c, 30 * b[0] + 20 * c[0] + rng.randn()) for a, b, c, _ in rows]
pf = next(f for f in learn_bayesian_network(rows, max_parents=2).factors if f.child == 3)
print("parents", pf.parents, "| len(coef)", len(pf.coef), "| design_row(all parents)", len(pf.design_row([rows[0][p] for p in pf.parents])),
      "| design_row(first parent only)", len(pf.design_row([rows[0][pf.parents[0]]])))
print("RandomVariable.fitted(None) ->", RandomVariable.fitted(None, name="x"))
@register_fitter("bare")
def _bare(rv, data, **_):
    return GaussianDistribution(0.0, 1.0)          # a Distribution, not RandomVariable.fitted(...)
try:
    Normal(free, free).fit([1.0, 2.0, 3.0], how="bare")
except Exception as e:
    print("bare-Distribution fitter:", type(e).__name__, e)
```

**Observed:** parents [1, 2], len(coef) 7; design_row(all parents) -> 7 entries; design_row(first parent only) -> 4 entries, no error (a wrong-width vector parent -> 9 entries, no error); RandomVariable.fitted(None, name='x') -> RV(bound=None); the bare-Distribution fitter -> AttributeError 'GaussianDistribution' object has no attribute '_cache' from inside fit. Output: out/p04_public_names_full.txt.

**Expected:** design_row refuses a value list whose length is not len(self.parents) (its docstring says the result pairs elementwise with self.coef); fitted() refuses dist=None; a fitter returning the wrong type gets a message naming RandomVariable.fitted, the helper P09-F13 added for exactly that.

**Notes:** The repair itself holds: design_row, RandomVariable.fitted and fit.result exist, are documented, and examples 1-28 no longer touch underscore names (the only remaining '._' hits are autoregressive_enumeration_example's own attribute). Re-registering a fitter name is refused ('silent replacement is forbidden') and an unknown how= lists the registered names.

**Repairs concerned:** P09-F13

### Q09-F11 · minor · A bare str is refused by optimize()/fit()/Model.fit() ('received a str, which iterates as its individual characters') but accepted by estimate(), initialize() and get_estimator(), which fit a categorical over the characters

**Surface:** mixle.inference.estimate, mixle.inference.initialize, mixle.utils.automatic.get_estimator (accept); mixle.inference.optimize, mixle.inference.fit, mixle.Model.fit (refuse)

**Reproduction:**

```
# any 0.8.2 venv, cwd /tmp
import numpy as np
from mixle.inference import estimate, initialize, optimize
from mixle.stats import CategoricalEstimator
from mixle.utils.automatic import get_estimator
print("estimate  :", estimate("hello", CategoricalEstimator()))
print("initialize:", initialize("hello", CategoricalEstimator(), np.random.RandomState(0)))
print("get_estimator:", type(get_estimator("hello")).__name__)
try:
    optimize("hello", CategoricalEstimator(), out=None)
except Exception as e:
    print("optimize  :", type(e).__name__, str(e)[:120])
```

**Observed:** estimate('hello', CategoricalEstimator()) -> CategoricalDistribution({'e': 0.2, 'h': 0.2, 'l': 0.4, 'o': 0.2}); initialize(...) -> CategoricalDistribution({'l': 1.0}); get_estimator('hello') -> CategoricalEstimator; estimate(b'hello', ...) -> a categorical over byte values; optimize / fit / Model.fit -> ValueError "... received a str, which iterates as its individual characters/bytes -- fitting it produces a categorical over those, which is almost never what was meant ...". Output: out/p16b_front_door_twins_base.txt.

**Expected:** The same refusal on every front door, or on none.

**Notes:** The str guard on optimize()/fit() is a 0.8.x repair whose id I did not locate (grep 'received a str' finds no CHANGELOG line). NaN and +/-inf through the automatic path are handled consistently (NaN as the documented missingness, +/-inf as a missingness indicator with a warning naming DatumNode.get_estimator) and were not filed.

**Repairs concerned:** none

### Q09-F12 · minor · mixle.doe.minimize on degenerate inputs: a NaN or inf objective dies with TypeError "unsupported operand type(s) for *: 'NoneType' and 'float'", an objective of 1e300 with 'amplitude must be finite', and seed=np.random.default_rng(0) with a numpy cast TypeError -- none names the input at fault

**Surface:** mixle.doe.bayesopt.minimize (objective values; seed spellings)

**Reproduction:**

```
# REVIEW_ROOT/venv-full/bin/python (torch), cwd /tmp
import numpy as np
from mixle.doe import minimize
bounds = [(-5.0, 5.0), (-5.0, 5.0)]
obj = lambda p: float((p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2)
for label, f, kw in (("nan objective", lambda p: float("nan"), dict(n_iter=5, seed=0)), ("inf objective", lambda p: float("inf"), dict(n_iter=5, seed=0)),
                     ("huge objective", lambda p: 1e300 * (p[0] ** 2 + 1), dict(n_iter=5, seed=0)),
                     ("seed=Generator", obj, dict(n_iter=3, seed=np.random.default_rng(0))), ("seed=RandomState", obj, dict(n_iter=3, seed=np.random.RandomState(0)))):
    try:
        res = minimize(f, bounds, n_init=5, **kw); print(label, "-> OK best_x", np.round(np.asarray(res.best_x, float), 3))
    except Exception as e:
        print(label, "->", type(e).__name__, str(e)[:110])
```

**Observed:** nan objective -> TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'; inf objective -> the same; huge objective -> ValueError: amplitude must be finite (after a numpy overflow RuntimeWarning); seed=Generator -> TypeError: Cannot cast scalar from dtype('O') to dtype('int64') according to the rule 'safe' (seed='0' -> the same with dtype('<U1')); seed=None / seed=RandomState / int seeds / bounds as ndarray / 1-D bounds / n_iter=0 work; lo==hi, lo>hi, n_init=0, n_iter=-1 and latin_hypercube n=0 / -1 / 2.5 / empty bounds are refused with clear messages. Output: out/p01b_doe_degenerate_full.txt.

**Expected:** A non-finite objective value is refused by name at the evaluation that produced it; a numpy Generator is accepted as a seed (initialize() and the samplers accept one since R02-F13) or refused by name.

**Notes:** P09-F03 itself holds: all 12 seeds of the example's call converge near (1, -2) and disclose their covariance ridging through BayesOptResult.numerical_repairs() (R07-F05) -- 2 to 7 repairs per seed of 0.2-4.0 beyond jitter=1e-6, 28 for a 40-iteration run (out/p01a_doe_seeds_full.txt).

**Repairs concerned:** P09-F03, R07-F05, R02-F13

### Q09-F13 · minor · Enumeration rough edges: AutoregressiveEnumerable.count(-inf) raises OverflowError 'cannot convert float infinity to integer' while count(+inf) is refused by name; enumerator().seek(10**12) on the showcase's infinite-support record did not return within 890 s (seek(10**8): 8 s) and has no budget or fail-fast

**Surface:** mixle.enumeration.AutoregressiveEnumerable.count; CompositeDistribution(IntegerCategorical, Poisson, Geometric).enumerator().seek (examples/enumeration_showcase_example.py's record)

**Reproduction:**

```
# REVIEW_ROOT/venv-base/bin/python, cwd /tmp
import sys, time
sys.path.insert(0, "<review-root>/source/examples")
import autoregressive_enumeration_example as ae
from mixle.enumeration import AutoregressiveEnumerable
from mixle.stats import CompositeDistribution, IntegerCategoricalDistribution, PoissonDistribution, GeometricDistribution
ar = AutoregressiveEnumerable(ae.make_synthetic_model(5, 3, seed=0), max_len=3, oversample=64)
for v in (float("inf"), float("-inf")):
    try:
        print(f"count({v}) ->", ar.count(v))
    except Exception as e:
        print(f"count({v}) ->", type(e).__name__, e)
e = CompositeDistribution((IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]), PoissonDistribution(4.0), GeometricDistribution(0.3))).enumerator()
for n in (10 ** 5, 10 ** 8):
    t0 = time.time(); r = e.seek(n); print(f"seek({n}) exact={r.exact} truncated={r.truncated} {time.time() - t0:.1f}s")
# e.seek(10 ** 12) produced no output in the 890 s left of a 900 s budget (out/p17_enumeration_base.txt) -- not included so this block terminates
```

**Observed:** count(+inf) -> ValueError "log_prob must be <= 0 (a probability > 1 is invalid), got inf"; count(-inf) -> OverflowError "cannot convert float infinity to integer" (also for a model with a zero-probability token). seek(1e4) 0.6 s exact, seek(1e5) 4.9 s exact, seek(1e6) 54.5 s exact, seek(1e7) 3.3 s inexact (bracket +/-4e3), seek(1e8) 8.2 s inexact, seek(1e12): no output in the 890 s remaining of a 900 s budget. Outputs: out/p17_enumeration_base.txt, out/p17b_enumeration_noseek_base.txt, out/p17c_seek_timing_base.txt.

**Expected:** count(-inf) answers (every sequence) or is refused like +inf; a seek beyond what the index can answer in bounded time returns the truncated=True bracket its result type already carries, or refuses.

**Notes:** Everything else attacked on these surfaces held: top_k / rank / nucleus_size agree with brute force (rank 10000 exact; nucleus [212, 232] vs brute 231; mixture rank(5)=17 exact), out-of-support and wrong-arity values are refused or ranked None, negative and non-integer k are refused. Whether the 1e12 seek is unbounded or merely slow was not characterised.

**Repairs concerned:** none

### Q09-F14 · minor · estimate(data, TreeHiddenMarkovEstimator(...)) with no prev_estimate dies inside the accumulator with AttributeError "'NoneType' object has no attribute 'dist_to_encoder'" instead of saying the family needs an initial estimate (same on 0.8.1)

**Surface:** mixle.inference.estimate with TreeHiddenMarkovEstimator (mixle/stats/latent/tree_hidden_markov_model.py update(), reached from mixle/stats/compute/sequence.py estimate)

**Reproduction:**

```
# any 0.8.2 venv, cwd /tmp
from mixle.inference import estimate
from mixle.stats import (TreeHiddenMarkovModelDistribution, TreeHiddenMarkovEstimator, GaussianDistribution, GaussianEstimator,
                         PoissonDistribution, PoissonEstimator)
d = TreeHiddenMarkovModelDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5],
                                      [[0.8, 0.2], [0.2, 0.8]], len_dist=PoissonDistribution(0.8), terminal_level=3)
try:
    estimate(d.sampler(1).sample(150), TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], len_estimator=PoissonEstimator()))
except Exception as e:
    print(type(e).__name__, e)
```

**Observed:** AttributeError: 'NoneType' object has no attribute 'dist_to_encoder' (tree_hidden_markov_model.py:1179 via sequence.py:856) on 0.8.2, and the same on 0.8.1. Outputs: out/p22b_treehmm_json_regression_full.txt, out/p22b_treehmm_json_regression_081.txt.

**Expected:** A message naming the precondition (initialize first / pass prev_estimate, or use optimize), as the front doors do elsewhere.

**Notes:** Pre-existing; found while chasing Q09-F01. The gallery uses optimize for this family, which works.

**Repairs concerned:** none

### Q09-F15 · docs · docs/example-execution-manifest.rst does not name torch as the base-install blocker for calibrated_report_demo, geoscience_inversion_report and heterogeneous_representation_example (it does for doe_example), and its 2026-07-21 geoscience note still describes the abstain path the P09-F04 repair removed

**Surface:** docs/example-execution-manifest.rst (lines 215-219 and the Inventory rows at 410-463)

**Reproduction:**

```
cd <review-root>/source
grep -n -A1 'calibrated_report_demo.py\|geoscience_inversion_report.py\|heterogeneous_representation_example.py\|doe_example.py``' docs/example-execution-manifest.rst | sed -n '1,40p'
sed -n '215,219p' docs/example-execution-manifest.rst
grep -n 'SBC p-value\|coverage@0.5\|calibrated report' ../pass-09/out/ex_full_geoscience_inversion_report.txt
awk -F'\t' '$2 != 0' ../pass-09/out/ex_base_summary.tsv
```

**Observed:** Manifest rows: calibrated_report_demo and heterogeneous_representation_example -> 'Execute with optional-dependency status recorded.'; geoscience_inversion_report -> 'Execute or mark blocked on scientific dependencies.'; doe_example -> 'Blocked on torch in a base install ...'. Lines 215-219: "Confirmed geoscience_inversion_report.py ... The script's calibration layer detected a poorly calibrated candidate and abstained." On venv-base all three exit 1 on "No module named 'torch'" (heterogeneous_representation_example at its own top-level import torch, per out/p08_base_imports_base.txt), and on venv-full the geoscience report now serves: "SBC p-value=0.109 (pass=True)", "coverage@0.5=0.453 (pass=True), coverage@0.9=0.893 (pass=True)", "calibrated report: \"depth_km is between 4.007 and 4.407\"", "true depth 4.2 km contained: True" (out/ex_full_geoscience_inversion_report.txt, out/ex_base_summary.tsv).

**Expected:** The manifest names torch for every example that cannot run without it, and its geoscience narrative matches the repaired script.

**Notes:** P09-F04 itself holds on the candidate (posterior mean 4.204 sd 0.097 km against the exact 4.205 / 0.099 the 0.8.1 reviewer computed); only the manifest prose is stale.

**Repairs concerned:** P09-F04

### Q09-F16 · docs · autoregressive_enumeration_example's docstring promises 'Exact count / rank / unrank', but unrank's ordering is quantized by design: at the example's own oversample=64, rank(unrank(i)) != i at 2-18 of 125 indices per seed, adjacent near-ties (0.007 nats apart) coming back swapped

**Surface:** examples/autoregressive_enumeration_example.py docstring (lines 1-8) versus mixle.enumeration.AutoregressiveEnumerable.unrank's docstring, which documents the fine-bucket quantization

**Reproduction:**

```
# REVIEW_ROOT/venv-base/bin/python, cwd /tmp
import sys, itertools
sys.path.insert(0, "<review-root>/source/examples")
import autoregressive_enumeration_example as ae
from mixle.enumeration import AutoregressiveEnumerable
for seed in range(5):
    nl = ae.make_synthetic_model(5, 3, seed=seed)
    ar = AutoregressiveEnumerable(nl, max_len=3, oversample=64)          # the example's own settings
    bad = [(i, ar.unrank(i)[0], ar.rank(ar.unrank(i)[0]).rank) for i in range(125) if ar.rank(ar.unrank(i)[0]).rank != i]
    print(f"seed {seed}: rank(unrank(i)) != i at {len(bad)} of 125 indices, e.g. {bad[:3]}")
nl = ae.make_synthetic_model(5, 3, seed=2)
lp = lambda s: sum(dict(nl(tuple(s[:i])))[t] for i, t in enumerate(s))
print("seed 2: unrank(99) =", AutoregressiveEnumerable(nl, max_len=3, oversample=64).unrank(99)[0], "| lp(0,2,4) =", round(lp((0, 2, 4)), 4), "lp(4,0,2) =", round(lp((4, 0, 2)), 4))
```

**Observed:** seed 2: unrank(99) -> (0, 2, 4) (log_p -10.2057) although (4, 0, 2) (log_p -10.1989) is strictly more probable and rank((0, 2, 4)).rank == 100; rank(unrank(i)) != i at 14, 18, 18, 2 and 9 of 125 indices for seeds 0-4; top_k, rank, threshold and count all agree with brute force. Outputs: out/p17d_rank_tie_base.txt, out/p17b_enumeration_noseek_base.txt.

**Expected:** Prose that says what the library documents: count / rank / top_k exact, unrank exact between fine buckets of width bin_width_bits/oversample and unspecified within one.

**Notes:** The example's printed round trip (index 42) happens to land outside a tie; the library docstring is accurate and even predicts 'rank(unrank(i)) can differ from i for near-tied neighbours'.

**Repairs concerned:** none

## Attacks that did not break anything

- **P09-F03 / R07-F05 (DOE surrogate):** all 12 seeds of the example's `minimize(obj, bounds, n_init=5, n_iter=15, seed=s)` converge near (1, -2) and every seed but 0 discloses 2-7 `covariance-ridged(...; beyond jitter=1e-06, to factor the GP kernel matrix)` repairs through `BayesOptResult.numerical_repairs()` (the same spelling on `surrogate_repairs`); a 40-iteration run reports 28. No warning, no LinAlgError (`out/p01a_doe_seeds_full.txt`). lo==hi, lo>hi, n_init=0, n_iter=-1, 1-D bounds, ndarray bounds, seed=None/RandomState, and `latin_hypercube` n=0/-1/2.5/empty are refused or handled by name (`out/p01b_doe_degenerate_full.txt`).
- **P09-F12 (RDPG clip disclosure):** the clip is disclosed in both directions (`edge-probability-clipped(2 of 56 ...; inner products span [0.08637, 1.074] ...)` for the 0.8.1 reproduction's positions, `4 of 6 ... [-1, 0.5]` for negative inner products), survives pickle and JSON, and is reported on the fit route too (complete graphs: 50 of 56; a one-graph fit: 8 of 56); the gallery's scaled positions have empty repairs and its printed fit-vs-truth gap 0.0125 reproduces; degenerate adjacencies (self loops, asymmetric, wrong size, non-binary) and a zero-graph fit are refused by name (`out/p03_rdpg_full.txt`).
- **P09-F05 (copula serialization):** the example's fitted `CopulaDistribution` with the R-vine core and with the Gaussian core, both cores alone, a Clayton core, and R-vines over Beta marginals in dims 2 and 4 all round-trip through `dump_models`/`load_models`, `to_json`/`from_json` and pickle with max |dLL| = 0 and identical same-seed samples; edges survive; 2-, 3- and 5-row fits work; NaN, inf and negative-Gamma rows are refused by name; a constant column fits with a floored Gamma (`out/p02_copula_roundtrip_full.txt`).
- **P09-F13 (public names):** `design_row()`, `RandomVariable.fitted()` and `fit.result` exist with docstrings that cite P09-F13; a custom `@register_fitter` built on `fitted()` fits, summarises and explains; no underscore names remain in examples 1-28 other than the autoregressive example's own attribute (`out/p04_public_names_full.txt`). The manifests are per-package name lists / stable-module signatures, so method-level absence there is by design.
- **P09-F08 (physics inverse):** 6000 retained draws, reported bulk ESS 1449 against an independent Geyer estimate of 1444 (lag-1 autocorrelation 0.60), one budget for the headline interval and the coverage replicates, headline 90% interval [1.326, 1.398] against the exact grid posterior [1.3255, 1.3975], coverage 11/12 consistent with the printed EXCLUDES line, identical draws on a same-seed rerun, ESS <= retained draws (`out/p05_physics_ess_full.txt`, `out/ex_full_flagship_physics_inverse.txt`).
- **P09-F09 / R07-F06 (warning attribution):** `calibrated_report_demo` emits 2 stderr lines (was 44), both attributed to its own `learn_structure(...)` line 274 and advising `max_its`, which `learn_structure` accepts; `frontier_family_showcase` and `geoscience_inversion_report` are warning-free; every other example's notes point at the example's own line (`out/ex_full_*.txt` stderr blocks). `gallery_structured`'s 7 budget-stop notes are disclosed by its docstring.
- **P09-F10 (frontier DONE line):** now "headline + 1 priced rung(s) of 2 attempted (rung_mid), each I1-quantized and priced; ... monotone family frontier: False"; the `earns_its_complexity()` assertion holds at zero escalation (its `tol` covers cascade cost == student cost) (`out/ex_full_frontier_family_showcase.txt`).
- **P09-F04 (geoscience):** posterior mean 4.204 sd 0.097 km, SBC p=0.109 pass, coverage 0.453/0.893 pass, report served and contains the truth (`out/ex_full_geoscience_inversion_report.txt`).
- **A-01 / P09-F07 / P09-F11 (HMM start):** the gallery's TreeHMM call recovers (-3.002, 1.022)/(2.949, 0.896), held-out -15.180 vs true -15.071, at rng seeds 1-8 and data seeds 1-5; the SegmentalHMM -16.359 vs -16.347 at rng seeds 1-6; a plain Gaussian HMM at seeds 1-6; categorical and quantized HMMs decline the start (quantized theta 0.9149 as the gallery prints); unbalanced states, constant data (variance floor disclosed in `numerical_repairs()` and provenance), and a 3-state fit on 2-cluster data all fit; n=10 and n=3 TreeHMM fits are refused with `ImpossibleEvidenceError` naming the rows (`out/p09_hmm_init_full.txt`).
- **Calibrated gate (P09-F02 neighbourhood):** `serve(seed=3)` on a certified generator is refused as the docstring says; empty and 1-row calibration sets are refused; an all-wrong oracle gives qhat=inf and abstains; the teacher matches the planted shape on 96.9% of clear and 57.2% of faint volumes; the scorer's top candidate matches the teacher on 100% of clear volumes (`out/p11_calgate_full.txt`).
- **auto_example:** the fitted composite round-trips (dump/load, pickle) with identical log-likelihoods; `optimize(data, est)` and bare `optimize(data)` work; 16 degenerate record shapes through `get_estimator` (one row, empty list, all-None field/row, mixed types, ragged tuples, empty bags, NaN/inf/bool/bytes/numpy scalars, list of strings, dict rows) each yield a fit or a named refusal (`out/p16_auto_example_base.txt`).
- **Gallery round trips:** 58 of 68 fitted gallery models round-trip through `dump_models`, `to_json` and pickle with identical log-densities; the 10 exceptions are the 9 in Q09-F01/Q09-F02 plus `SelectDistribution` over a lambda (unregistered callable, refused by name) (`out/p07_gallery_roundtrip_full.txt`). `estimate()` results carry `fit_provenance()` None while `optimize()` stamps one; `estimate`'s docstring makes no provenance promise, so this was not filed.
- **Enumeration:** `top_k`, `rank`, `seek`, `nucleus_size`, `threshold`, `count`, `cumulative` agree with brute force on the showcase record, a Poisson mixture and the autoregressive example's 125-sequence and terminating models; k<0, non-integer k, wrong arity, out-of-vocabulary tokens, `max_len=0`, `oversample=0`, unnormalised tables and a model with neither eos nor max_len are refused by name (`out/p17b_enumeration_noseek_base.txt`).
- **Base install:** 23 of the 26 examples run there finish with exit 0 on numpy+scipy alone.

## What was not covered

- `probes/p10_geo_seeds.py` (geoscience at other inversion/observation seeds and depths, and the docstring's 58%-coverage claim) was written but never run: each `learn_inverse` call takes minutes under this machine's contention. P09-F04 is verified only at the example's own seed.
- `probes/p15_hidden_assoc.py` (hidden_association at other seeds, shrunk data, JSON round trip of the 1000-row fit, degenerate rows) was written but never run; the example's fit alone took 186-517 s here. Only the short-budget JSON round trip in p07 covers this family.
- `sobol_indices` with degenerate n (n=0/1/2, a constant function): the DOE probe crashed on its own result formatting before reaching them.
- venv-nonumba was verified but no example or probe was executed on it.
- 0.8.1 comparisons were run only where a regression question arose (p20/p21/p22); the other findings are not characterised as regressions or otherwise.
- `distill()` called with a string-labelled teacher on the base install died with `TypeError: '>' not supported between 'str' and 'int'` after one teacher call (`out/p14_solve_structured_base.txt`); its contract (texts plus labels) was not checked, so this is recorded here, not filed.
- The regression tests guarding the P09 repairs were read by the original reviewer, whose trail records no conclusion; the recovery reviewer did not re-read them.
- Why the constructed `TreeHiddenMarkovModelDistribution`'s `_p_level_cache` is populated by a scoring call yet still round-trips was not investigated; only the fitted object's failure and its cause were measured (Q09-F01).

DONE 09 16
