# Adversarial review -- PASS 10 of 10 (mixle 0.8.2 release candidate)

- Focus: example scripts 29-57 in sorted order (`hierarchical_mixture_example.py` ... `win_demo_example.py`, 29 scripts), the repairs they exercise (P10-F01..F13, A-01..A-03), and the `mixle.ops.project` / dependency-screen / serialization surfaces behind them.
- Wheel: `mixle-0.8.2-py3-none-any.whl`, sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da`.
- Commit `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013` (read-only export at `REVIEW_ROOT/source`).
- Work dir: `REVIEW_ROOT/pass-10/` (all paths below are relative to it).

## Environment verification (`cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py`, verbatim)

`venv-full` (`env/verify_venv-full.txt`):

```
executable       <review-root>/venv-full/bin/python
mixle.__path__   <review-root>/venv-full/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-base` (`env/verify_venv-base.txt`):

```
executable       <review-root>/venv-base/bin/python
mixle.__path__   <review-root>/venv-base/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

`venv-nonumba` (`env/verify_venv-nonumba.txt`):

```
executable       <review-root>/venv-nonumba/bin/python
mixle.__path__   <review-root>/venv-nonumba/lib/python3.12/site-packages/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
```

0.8.1 comparison venv (`env/verify_081.txt`, read-only, used only for before/after):

```
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

Pinned reproduction closure (numpy 2.4.6 / scipy 1.17.1, Python 3.12.12: the read-only interpreter at `.ci-repro-colima/rehearsal-8534ddeb/venv-lock`, which has no mixle installed, with the byte-verified wheel unpacked at `wheel-unpacked/` on `PYTHONPATH`; used only for the repro-bundle digest check, `env/verify_pinned_closure.txt`):

```
executable       <review-store>/rehearsal-8534ddeb/venv-lock/bin/python
mixle.__path__   <review-root>/pass-10/wheel-unpacked/mixle
dist version     0.8.2
source_commit    866078be520b22188110780be957150dc6da964c
source_tree      7922a8c59283ecef6877104b9a7499afd52d9013
content_sha256   2b52a709e08f298971da77af1022b76b3312383d4a969ac2ab4d1faa4f445554
numpy 2.4.6 scipy 1.17.1 3.12.12
```

`venv-nonumba` was verified (above) but no run in this pass used it.

Recovery note: the original pass-10 reviewer executed the corpus, wrote the probes under `probes/` and their outputs, and left `findings_notes.md` (ten confirmed items plus a pending list) and this header before its session was cut off by an API rate limit. This report was assembled from that preserved evidence by a second reviewer session, which added only: the warning audit over the existing logs (`probes/p17_warning_audit.full.txt`), the 0.8.1-vs-0.8.2 comparison of `learn_mixture_structure`'s warnings (`probes/p22_*`), the A-02 note-vs-returned-model probe (`probes/p24_*`), the source-tree digest check (`probes/p23_*`), the semi-supervised datum probe (`probes/p25_*`), the pending `probes/p20_*`, and the venv-base execution of the 29 scripts (`runs/base/`). Nothing already logged was re-run. The original reviewer's command trail is in `RECOVERED_NOTES.md`.

## Corpus executed on the candidate

Notebooks: none are assigned to this pass.

Examples (29, `REVIEW_ROOT/source/examples/`, numbered by their position in the sorted list of 57). `venv-full` column: `runs/full/<script>.txt`, run through `runs/run_examples.py` (tools/pyt.py, three scripts at a time on a machine whose load average was 57-87 -- wall times are contention, not performance evidence). `venv-base` column: `runs/base/<script>.txt`, two at a time, later in the day under lighter load. "warning lines" = `UserWarning` lines in the venv-full stderr (`probes/p17_warning_audit.full.txt`).

| # | script | venv-full exit | wall s | warning lines | venv-base exit | wall s | venv-base note |
|---|---|---|---|---|---|---|---|
| 29 | hierarchical_mixture_example | 0 | 337.8 | 0 | 0 | 45.0 | runs (same numbers) |
| 30 | joint_mixture_example | 0 | 74.5 | 4 | 0 | 7.9 | runs (same numbers) |
| 31 | label_economics_demo | 0 | 75.2 | 0 | 0 | 7.9 | runs (same numbers) |
| 32 | latent_variable_models_example | 0 | 70.7 | 0 | 0 | 7.8 | runs (same numbers) |
| 33 | lookback_hmm_example | 0 | 145.3 | 0 | 0 | 91.5 | runs (1e-14 float noise in printed params) |
| 34 | mixture_reduction_benchmark | 0 | 421.4 | 3 | 0 | 24.5 | runs (only the speed ratio differs) |
| 35 | model_comparison_example | 0 | 543.3 | 0 | 0 | 41.6 | runs (same numbers) |
| 36 | multimodal_stage1_demo | 0 | 230.2 | 1 | 1 | 2.9 | raw ModuleNotFoundError (Q10-F12) |
| 37 | peft_lora_grad_leaf | 0 | 714.1 | 2 | 1 | 2.8 | refuses by name (missing extra; P10-F08) |
| 38 | ppl_example | 0 | 85.1 | 0 | 0 | 3.2 | runs (same numbers) |
| 39 | precedence_scheduling_example | 0 | 83.2 | 0 | 0 | 3.0 | runs (same numbers) |
| 40 | production_example | 0 | 152.2 | 0 | 0 | 7.4 | runs (same numbers) |
| 41 | project_neural_to_structured | 0 | 978.2 | 5 | 1 | 3.0 | refuses by name (missing extra) |
| 42 | quickstart_example | 0 | 188.9 | 0 | 0 | 5.8 | runs; only the unseeded `sampler().sample(3)` draws differ |
| 43 | reasoner_investigation_demo | 0 | 113.6 | 1 | 0 | 3.0 | runs (same numbers, same library-attributed warning) |
| 44 | scaling_example | 0 | 201.9 | 2 | 0 | 7.9 | runs (same numbers) |
| 45 | semi_supervised_mixture_example | 0 | 215.3 | 0 | 0 | 7.9 | runs (same final numbers; see note) |
| 46 | shared_embedding_example | 0 | 119.3 | 0 | 1 | 3.4 | refuses by name (missing extra) |
| 47 | skeptic_challenge_example | 0 | 304.4 | 6 | 1 | 4.4 | raw ModuleNotFoundError for sklearn (Q10-F12) |
| 48 | structure_learning_example | 0 | 272.0 | 65 | 0 | 17.2 | runs (same numbers; 65 warning lines too) |
| 49 | structured_hmm_example | 0 | 382.6 | 0 | 0 | 110.9 | runs (same numbers) |
| 50 | structured_leaves_example | 0 | 31.9 | 0 | 0 | 2.6 | runs (same numbers) |
| 51 | symbolic_export_example | 0 | 35.6 | 0 | 1 | 0.1 | raw ModuleNotFoundError for sympy (Q10-F12) |
| 52 | task_cascade_economics_example | 0 | 56.9 | 4 | 1 | 2.8 | refuses by name (missing extra) |
| 53 | task_distill_example | 0 | 139.3 | 0 | 1 | 3.4 | refuses by name (missing extra) |
| 54 | task_extraction_example | 0 | 120.7 | 0 | 1 | 3.0 | raw ModuleNotFoundError from inside mixle.task (Q10-F12) |
| 55 | task_llm_active_example | 0 | 54.0 | 0 | 1 | 3.0 | refuses by name (missing extra) |
| 56 | vlm_trust_receipts_demo | 0 | 29.3 | 0 | 1 | 3.7 | raw ModuleNotFoundError (Q10-F12) |
| 57 | win_demo_example | 0 | 21.3 | 1 | 1 | 3.5 | refuses by name (missing extra) |

Tally: venv-full 29/29 exit 0 (the background full execution in `REVIEW_ROOT/exec/examples.log`, cited not made, also has all 29 at exit 0); venv-base 17/29 exit 0, 7 named refusals, 5 raw `ModuleNotFoundError`s (Q10-F12). The stdout of the 17 scripts that ran on both environments is identical except for unseeded draws (quickstart), 1e-14 float noise in printed parameters and 1e-9 iteration gains (lookback, semi-supervised), the timing ratio (mixture_reduction, "~39x" vs "~33x"), and the `compiled-em: component-level fused full-tree execution` lines the numba path writes to the `out` stream the semi-supervised example requests with `print_iter` (estimation.py:2327).

Fresh output vs prose (the three lines below are also findings):
- `mixture_reduction_benchmark`: the `project()` docstring's `delta=None` promise ("exactly max_its iterations ... no unconverged-fit note") vs the three notes the run prints (22, 35, 42 of 50 iterations) -- Q10-F03.
- `lookback_hmm_example`: docstring "1000 EM iterations with delta=None" vs the script's `best_of(..., 4, 400, 1.0, None, ...)` -- Q10-F04.
- `skeptic_challenge_example`: act-3 banner "and it pays" vs the printed hybrid -2.108 < pure-neural -2.020 nats/row -- Q10-F10.
- Otherwise none: the printed numbers that can be recomputed independently were, and agree -- precedence closure +35.0 and schedule NPV 30.5353 by brute force, label_economics sign-test p 0.0078125, task_cascade exact paired p 0.0414 / 0.8318, task_llm_active p 0.125, symbolic_export scores at x=10/100 (`probes/p14_deterministic_recompute.base.txt` on venv-base, plus an inline `math.comb` check in this session's trail); `mixture_reduction_benchmark`'s two takeaway lines (4/5, monotone warm / non-monotone cold) match its own table; `latent_variable_models_example` prints the fitted alpha [2.58, 2.53] and TV distances [0.149, 0.119] with hedged prose (P10-F12); `model_comparison_example` prints B's ESS (worst bulk 370, acceptance 0.16; P10-F06); `structured_hmm_example` reports "best of 3 restarts by likelihood" (P10-F04); `lookback_hmm_example` prints its held-out gap, 0.091 nats/sequence (P10-F07); `peft_lora_grad_leaf` prints the Hub asset and revision (P10-F08).

## Findings

### Q10-F01 (real) -- `propose()`/`analyze_structure` dependency screen: an integer count field with <=128 distinct values is scored one code per value, so the BIC penalty at the quickstart's n=240 hides a planted 0.5-0.6-bit dependence that the same screen reports at 0.43-0.52 bits when the field is binned

Surface: `mixle.utils.automatic.analyze_structure` (pairwise screen: `profiling._encode_for_pairwise`, `_mi_from_encoded`) and `propose().explain()`'s dependency lines. Repair concerned: P10-F01.

Reproduction (`probes/p15_count_field_screen.py`, output `probes/p15_count_field_screen.full.txt`; siblings `probes/p09_p10f01_overdispersed.py`, `probes/p04_p10f01_dependency_screen.py`):

```python
import warnings, numpy as np
from mixle.utils.automatic import analyze_structure
import mixle as M
def planted_counts(seed, n=240):
    rng = np.random.RandomState(seed)
    c = rng.choice(["paid", "free"], size=n, p=[0.4, 0.6])
    k = np.where(c == "paid", rng.negative_binomial(1, 0.05, n), rng.negative_binomial(1, 0.5, n))  # means ~19 vs ~1
    f = np.where(c == "paid", rng.random(n) < 0.9, rng.random(n) < 0.05)
    return [(int(a), str(b), bool(d)) for a, b, d in zip(k, c, f)]
def hints(records, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = analyze_structure(records, **kw)
    return sorted((h.left, h.right, round(float(h.bic_gain_bits), 3), h.method) for h in p.pairwise_hints)
for seed in (0, 100, 200):
    rec = planted_counts(seed)
    print(seed, "default:", hints(rec))
    print(seed, "max_cardinality=20 (forces bins):", hints(rec, max_cardinality=20))
    print(seed, "n=2000 default:", hints(planted_counts(seed, n=2000)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = M.propose(planted_counts(0), fit=True, seed=0)
print([ln.strip() for ln in res.explain().splitlines() if "dependency" in ln.lower()])
print([(c["name"], round(c["heldout_mean_log_density"], 3)) for c in res.frontier if "heldout_mean_log_density" in c])
```

Observed: seed 0 (46 distinct count values, MI(count, category) = 0.566 bits raw): default `max_cardinality=128` -> hints `[((1,),(2,),0.584,'empirical_discrete/empirical_discrete')]` only -- the count<->category pair is not reported; `max_cardinality=20` -> `[((0,),(1,),0.432,'quantile_bins/..'), ((0,),(2,),0.243,..), ((1,),(2,),0.584,..)]`; n=2000 default -> the pair appears at 0.331 bits. Seed 100: default reports it at 0.01 bits, binned 0.524; seed 200: default absent, binned 0.422. `propose(rec, fit=True, seed=0).explain()` lists only `- dependency: $[1] <-> $[2] (0.6 bits for joint modeling)` while its own frontier prefers the structured candidate (held-out -3.301 vs recommended -3.923). Siblings: NB(1,0.05) vs NB(1,0.5) counts by category (40 distinct, MI 0.56 bits) -> no hint (p09); "overdispersed counts by category" -> 0.033 bits (p04); Poisson(30) vs Poisson(3) (31 distinct, MI 0.973) -> 0.479 bits, half swallowed (p09).

Expected: the screen's sensitivity should not fall with the number of distinct integer values at fixed n -- bin integer fields the way continuous and >128-distinct fields are binned (or use the Miller-Madow-adjusted MI the code already computes) so a 0.5-bit planted dependence on the quickstart's own n is reported; at minimum `explain()` should report the dependence the fitted frontier then exploits.

Notes: mechanism at `profiling.py:1747-1766`: `exactly_encodable = distinct <= max_cardinality` (default 128); an integer field not recommended gaussian/poisson (here negative_binomial / beta_binomial / integer_categorical) is encoded `empirical_discrete` with one code per value. `_mi_from_encoded` (1789-1791) subtracts `_bic_penalty_bits((kx-1)(ky-1), n) = 0.5*(k-1)*log2(n)/n` = 0.741 bits at k=46, n=240 (the probe prints this), larger than the 0.566-bit MI, so the gain is negative and the pair drops below the 0.05-bit threshold. The P10-F01 repair (bin what the discrete encoder cannot represent exactly) fixed the continuous quickstart column -- verified on the shipped planting at seeds 0/7/100/200/300, all three pairs reported at 0.36-0.75 bits (`probes/p04_p10f01_dependency_screen.full.txt`) -- but a count column is the sibling it does not reach. The fitted model is right (the frontier picks `structured`); the explanation of why is wrong.

### Q10-F02 (real) -- `release-checklists/0.8.2-repro-bundle.json` pins a production-provenance stdout digest (d4f80090...) that the 0.8.2 wheel does not produce in either arithmetic closure; the receipt's justification is contradicted by the published 0.8.1 wheel, which produces the OLD pin (aa6e4d98...) through the same volatile rules

Surface: `release-checklists/0.8.2-repro-bundle.json` (entry `production-provenance`, `expected.stdout_sha256`), `scripts/run_repro_entry.py` (`_normalize_output`), `release-checklists/0.8.2-receipts/reproducibility-bundle-and-independent-replay.json`, `examples/production_example.py` line 59. Release evidence; no P/A id.

Reproduction (`probes/p05_bundle_digests.py` on venv-full; the same with the byte-verified wheel unpacked on `PYTHONPATH` under the pinned closure numpy 2.4.6 / scipy 1.17.1, `env/verify_pinned_closure.txt`; the 0.8.1 venv through tools/pyt.py; digests recomputed from the saved stdouts):

```python
import hashlib, re, json
R = "<review-root>"
e = {x["id"]: x for x in json.load(open(R + "/source/release-checklists/0.8.2-repro-bundle.json"))["entries"]}["production-provenance"]
def digest(text):
    for v in e["expected"]["volatile"]:
        text = re.sub(v["pattern"], lambda _m: v["placeholder"], text)
    return hashlib.sha256(text.encode()).hexdigest()
for p in ("out/production-provenance.venv-full.stdout.txt", "out/production-provenance.pinned.stdout.txt"):
    print(p, digest(open(R + "/pass-10/" + p).read()))
s081 = open(R + "/pass-10/out/production-provenance.081.pyt.txt").read().split("\n== exit")[0].rstrip("\n") + "\n"
print("0.8.1 wheel", digest(s081))
print("bundle expects", e["expected"]["stdout_sha256"])
# regenerate a stdout: REVIEW_ROOT/tools/pyt.py 600 <venv>/bin/python REVIEW_ROOT/source/examples/production_example.py
```

Observed: 0.8.2 wheel on venv-full -> `777c247cf7d81276af43bdb9e49993b14ed7db5812188535853fb63c8e084320`; 0.8.2 wheel under the pinned closure -> the byte-identical stdout, `777c247c...`; published 0.8.1 wheel -> `aa6e4d987ca9245e44c0730d68103479d62e4ad2cce7c0a87559e572d03faa9b`, exactly the pin the 0.8.2 bundle replaced; bundle expects `d4f800900718c2d3729788103a63ec71b891b6cf40e32c33dd6f0df56e73565c` -> MISMATCH (`probes/p05_bundle_digests.full.txt`). The sibling entry `scaling-backend` matches its pin (`5fd8dd73...`). The 0.8.1 and 0.8.2 outputs differ only in the version token after the commit (`/ 0.8.1` vs `/ 0.8.2`), which the volatile rule `git / mixle  : (?:[0-9a-f]{7,40}|unknown) / ` does not cover; 600 version/commit/newline variants of the 0.8.2 output do not reach `d4f80090`. `scripts/build_repro_bundle.py:162-167` and the receipt state "release/0.8.1 and this tree both normalize to d4f80090..., and neither is aa6e4d98..., so the old value could not have been produced by a run through this normalization" -- the 0.8.1 wheel produces `aa6e4d98` through exactly that normalization. The receipt records PASS for all four entries at candidate `f3111688` with `PYTHONPATH=. <venv>/bin/python scripts/run_repro_entry.py` (source tree first on the path); the header's version is `mixle.__version__` (`mixle/inference/production/provenance.py:31-35`), so that route prints whatever tree/dist was loaded. With the 866078be source tree first on `sys.path` the shipped script prints `git / mixle  : unknown / 0.8.2` and the same digest `777c247c` as the wheel (`probes/p23_source_tree_digest.full.txt`), so the pinned value is not produced by this commit's source tree either.

Expected: the bundle's own acceptance text ("each passes exact output validation offline") must hold for the shipped wheel: pin a digest taken from a wheel run under the pinned closure (as the 0.8.1 receipt describes doing), or declare the version token volatile, and give a true reason in the receipt.

Notes: the two 0.8.2 closures agree with each other, so this is not arithmetic drift; the release-gate receipt "Reproducibility bundle and independent replay" cannot be reproduced from the artifact it attests. Evidence: `probes/p05_bundle_digests.full.txt`, `out/production-provenance.venv-full.stdout.txt`, `out/production-provenance.pinned.stdout.txt`, `out/production-provenance.081.pyt.txt`, `env/verify_pinned_closure.txt`, `probes/p23_source_tree_digest.full.txt`.

### Q10-F03 (docs) -- `mixle.ops.project` docstring: `delta=None` promises "exactly max_its iterations with no early stop and no unconverged-fit note"; the benchmark's own calls stop at 22, 35 and 42 of 50 iterations and print a note each time

Surface: `mixle.ops.project` (`mixle/ops.py:167-208`, the `delta` paragraph); `docs/migrations/0.8.2.md` lines 31-32 ("Pass `delta=None` for a fixed iteration count"); `examples/mixture_reduction_benchmark.py` lines 85 and 92. Repair concerned: P10-F03.

Reproduction: `PYT_CWD=<dir> REVIEW_ROOT/tools/pyt.py 1800 REVIEW_ROOT/venv-full/bin/python REVIEW_ROOT/source/examples/mixture_reduction_benchmark.py` (the benchmark's own calls are `project(teacher, target.estimator(), n_samples=4000, seed=0, max_its=50, delta=None)` and `project(teacher, cf, n_samples=4000, seed=0, max_its=50, init=cf, delta=None)`; `probes/p21_project_delta_none.py` reads `fit_provenance()` for M=2,4,6,8 but was not run).

Observed (`runs/full/mixture_reduction_benchmark.txt`): three UserWarnings of the form `project() was called with delta=None (documented as "a fixed iteration count": run max_its=50 iterations) but only 22 of them ran: a proposed update was rejected (a non-improving or non-finite step), which still ends the loop even when delta=None. This is not a converged fit ... fit_provenance() reports iterations=22, max_iterations=50, converged=False` -- 22 (cold, line 85), 35 and 42 (warm, line 92). The docstring: "``delta`` is the fitter's convergence tolerance, or ``None`` for exactly ``max_its`` iterations with no early stop and no unconverged-fit note -- what a budget-matched comparison between two projections wants".

Expected: either the loop honours `delta=None` (a budget-matched comparison is what the docstring says the mode is for) or the docstring and the migration guide say that a rejected update ends a `delta=None` run early and that a note is emitted.

Notes: the note quotes the docstring it contradicts. The benchmark's "cold" column ran less than half its budget on this run, which weakens the budget-matched comparison the P10-F03 repair added.

### Q10-F04 (docs) -- `lookback_hmm_example` docstring says "Runtime is ~20-25 s (1000 EM iterations with delta=None, i.e. no early stop)"; the script runs `best_of` with 4 restarts of at most 400 iterations each

Surface: `examples/lookback_hmm_example.py` line 27 (docstring) vs line 80 (`best_of(data, valid, est, 4, 400, 1.0, None, np.random.RandomState(1))`); `mixle.inference.best_of` positional signature (`estimation.py:2512`: `trials, max_its, init_p, delta`). Repair concerned: P10-F07.

Reproduction: `sed -n 27p REVIEW_ROOT/source/examples/lookback_hmm_example.py; sed -n 80p REVIEW_ROOT/source/examples/lookback_hmm_example.py; grep -n -A8 'def best_of' REVIEW_ROOT/source/mixle/inference/estimation.py`.

Observed: `best_of(..., 4, 400, 1.0, None, ...)` binds `trials=4, max_its=400, init_p=1.0, delta=None`: four restarts of up to 400 EM iterations each (at most 1600); no run of 1000 iterations exists in the script. The run prints `held-out mean log-density: fit -8.847 vs generating model -8.756 (gap 0.091 nats/sequence)` (the P10-F07 repair is present).

Expected: the docstring states the actual budget (4 restarts x 400 iterations, `delta=None`) or the code is changed to match it.

Notes: wall times are not evidence under contention (145.3 s in this pass, 12.6 s in the cited exec log). Evidence: `runs/full/lookback_hmm_example.txt`.

### Q10-F05 (minor) -- `learn_bayesian_network` attributes a perfect-separation GLM warning to `mixle/inference/bayesian_network.py:484` for an orientation the structure search scores and discards; the returned network has no binomial factor

Surface: `mixle.inference.learn_bayesian_network` (reached by `examples/reasoner_investigation_demo.py`); the binomial branch of the regression factor at `bayesian_network.py:484` (`glm(x, y01, family="binomial")`). Repairs concerned: P09-F09, P10-F09.

Reproduction (`probes/p18_reasoner_warning.py`; its trailing `AttributeError` is the probe's own -- `mixle` has no top-level `optimize`, the examples import it from `mixle.inference`):

```python
import warnings, numpy as np
from mixle.inference import learn_bayesian_network
def plan_spend(n, seed):
    r = np.random.RandomState(seed)
    return [(["free", "pro"][i % 2], float(20 + 80 * (i % 2) + 3 * r.randn())) for i in range(n)]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    net = learn_bayesian_network(plan_spend(500, 0), max_parents=1)
print(net.edges(), [type(f).__name__ for f in net.factors])
for x in w:
    print(x.filename, x.lineno, str(x.message)[:120])
```

Observed: returned `HeterogeneousBayesianNetwork` with edges `[(0, 1)]`, factors `[_MarginalFactor, _LinearGaussianFactor]` (spend | plan). One UserWarning from `.../site-packages/mixle/inference/bayesian_network.py:484`: "perfect separation detected between the binomial classes: some direction in the design separates them (500 of 500 fitted probabilities are within 1e-8 of 0 or 1), so at least one coefficient has no finite maximum-likelihood estimate ..." -- about the plan | spend binomial candidate that is not in the returned model. The shipped example prints the same line (`runs/full/reasoner_investigation_demo.txt`); it is the only library-attributed warning across the 29 venv-full runs (`probes/p17_warning_audit.full.txt`).

Expected: no warning for a candidate the search only scores, or one attributed to the caller's line naming the candidate orientation; a note should describe the model returned.

Notes: same class as P09-F09/P10-F09 ("the structure search no longer narrates the candidate models it only scores"): that repair walks the stack for the cap notes; this note is raised inside `glm()` with no such walk. Evidence: `probes/p18_reasoner_warning.full.txt`.

### Q10-F06 (minor) -- `learn_mixture_structure(rows, 2, restarts=4)` still prints 64 warning lines under the default filter on 0.8.2 (32 `dependency_gain scored its candidate fits at the max_its cap` + 32 cap notes) against 62 on 0.8.1; `structure_learning_example` prints 65 (58 in the 0.8.1 round) -- the attribution was repaired, the volume was not

Surface: `mixle.inference.learn_mixture_structure` and `dependency_gain` (`examples/structure_learning_example.py` line 76). Repairs concerned: P10-F09, P09-F09.

Reproduction (`probes/p22_lms_default_filter.py` through tools/pyt.py, i.e. `PYTHONWARNINGS=default`, on venv-full and on the 0.8.1 venv; then `grep -c 'Warning:'` on the captured output):

```python
import numpy as np
from mixle.inference import learn_mixture_structure
def _two_regime(seed, n=1600):
    r = np.random.RandomState(seed); out = []
    for _ in range(n):
        z = r.randint(0, 2); c = "hi" if r.rand() < 0.5 else "lo"; base = 5.0 if z == 0 else -5.0
        out.append((str(c), float(base + (3.0 if c == "hi" else -3.0) + r.randn())))
    return out
mot = learn_mixture_structure(_two_regime(3), 2, restarts=4)
```

Observed: 0.8.2 -- 64 lines, all attributed to the caller's line: 32 x "dependency_gain scored its candidate fits at the max_its cap (30) before they settled: the two log-likelihoods are budget-matched, so the comparison is still meaningful, but raise max_its if a borderline gain matters." and 32 x "learn_mixture_structure() stopped at the max_its cap (30) before the objective settled (last objective gain 0.744, delta=1e-06): ... Raise max_its to fit to convergence." (`probes/p22_lms_default_filter.full.txt`). 0.8.1 -- 62 lines, all attributed to `site-packages/mixle/inference/estimation.py:2117` ("optimize() stopped at the max_its cap ...") (`probes/p22_lms_default_filter.081.txt`). The shipped `structure_learning_example`: 65 lines on 0.8.2 (32 + 32 + one `fit()` note), none library-attributed (`runs/full/structure_learning_example.txt`, `probes/p19_notes_user_view.full.txt`); the 0.8.1 round's pass-10 report counted 58 for the same script.

Expected: per the P09-F09/P10-F09 entry the structure search "no longer narrates the candidate models it only scores"; on this verb every candidate scoring (32) and every restart's cap (32) still prints. One summary note per call, or a docstring sentence in the example, is what a reader can act on.

Notes: the attribution half of the repair is verified (0 library-attributed cap notes across the 29 runs). The example's docstring does not mention the warnings.

### Q10-F07 (real) -- the A-02 under-support note is emitted on every EM iteration and deduplicated by text, so one fit prints several contradictory "this mixture fit left N of K component(s)" lines; only the last one describes the returned model

Surface: `mixle/stats/latent/mixture.py:_disclosing_component_support` (lines 2071-2126), reached by every mixture fit: `optimize()`/`fit()`/`best_of`, `project()`, `DensityGate.fit` (`examples/task_cascade_economics_example.py` line 96), `solve()`. Repair concerned: A-02.

Reproduction (`probes/p24_a02_note_vs_returned_model.py`; also `probes/p20_a02_note_repeats.py`):

```python
import re, warnings, numpy as np
import mixle.stats as S
from mixle.inference import optimize
from mixle.stats.latent.mixture import _free_parameter_count
rng = np.random.RandomState(0)
rows = list(rng.normal(0.0, 1.0, 40)) + [40.0, 41.0] + list(rng.normal(20.0, 1.0, 3))
est = S.MixtureEstimator([S.GaussianEstimator()] * 4)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = optimize(rows, est, max_its=100, rng=np.random.RandomState(3), out=None)
notes = [str(x.message) for x in w if "less data than they have parameters" in str(x.message)]
print(len(notes), "emissions,", len(set(notes)), "distinct")
mass = np.asarray(m.component_row_mass, float); free = [_free_parameter_count(c) for c in m.components]
print(np.round(mass, 3).tolist(), free, [i for i in range(4) if mass[i] < free[i]])
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("default")
    optimize(rows, est, max_its=100, rng=np.random.RandomState(3), out=None)
print([re.search(r"left \d+ of \d+", str(x.message)).group(0) for x in w2 if "less data" in str(x.message)])
```

Observed (`probes/p24_a02_note_vs_returned_model.full.txt`): one `optimize()` call of 100 iterations emits the note 100 times (once per M-step), 2 distinct texts. Under the default filter the user sees two lines for the same fit: "this mixture fit left 4 of 4 component(s) with less data than they have parameters: component 0 ..., component 1 ..., component 2 ..., component 3 ..." (iteration 1) and "left 1 of 4 component(s): component 2" (iterations 2-100). Returned model: `component_row_mass` [25.927, 5.057, 1.224, 12.791], free parameters [2, 2, 2, 2] -> exactly 1 of 4 under-supported; the "4 of 4" line is false for the model returned. The shipped task_cascade example prints three such lines from one `DensityGate.fit` ("3 of 3"; "2 of 3: components 0, 1"; "2 of 3: components 1, 2"; `runs/full/task_cascade_economics_example.txt`); `probes/p01_project_guards.full.txt` case Q prints "7 of 8" and "8 of 8" for one `project()` call; `probes/p20_a02_note_repeats.full.txt` prints 2 lines for each of two fits.

Expected: one note per fit describing the returned model (emit from the EM driver after the last accepted iterate, or compare the final state), as the CHANGELOG describes: "A fitted mixture records how much data each component actually won ... and says so".

Notes: the source comment at `mixture.py:2096-2099` shows the design relies on the once-per-location warning filter and deliberately names components rather than masses so the text stays stable across iterations; it does not account for the starving set changing along the trajectory. A user reading warnings cannot tell which line is about the model they got, and under `warnings.simplefilter("always")` every iteration prints. The original reviewer's preliminary rating was `minor`; the p24 comparison with `component_row_mass` is what raises it to `real` (a disclosure reporting a number that does not apply to the returned fit).

### Q10-F08 (minor) -- `project()` argument validation: numpy-integer `n_samples`/`max_its` are refused as "not a positive integer"; `seed=RandomState/Generator/'3'` fail with a numpy cast TypeError; `init=` of the wrong component count fails as "mixture estimate and accumulator component counts must match"; `delta=` and a string `init=` are refused in `optimize()`'s name

Surface: `mixle.ops.project` keyword arguments `n_samples=`, `max_its=`, `seed=`, `init=`, `delta=` (the `init=`/`delta=` surface is new in 0.8.2). Repairs concerned: P10-F03, P10-F13.

Reproduction (`probes/p03_project_seeds.py`, outputs `.full.txt` for 0.8.2 and `.081.txt` for 0.8.1; `probes/p01_project_guards.py` for the `init=`/`delta=` cases):

```python
import numpy as np
from mixle.ops import project
from mixle.inference.project import reduce_mixture
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
def random_gmm(k, d, seed):
    rng = np.random.RandomState(seed); mus = rng.randn(k, d) * 4.0; covs = []
    for _ in range(k):
        a = rng.randn(d, d) * 0.5; covs.append(a @ a.T + np.diag(rng.uniform(0.3, 1.2, d)))
    return GaussianMixtureDistribution(mus, np.stack(covs), rng.dirichlet(np.ones(k) * 2.0))
teacher, tgt = random_gmm(12, 2, 1), random_gmm(2, 2, 7)
for label, kw in [("n_samples=np.int64(400)", dict(n_samples=np.int64(400))), ("max_its=np.int64(5)", dict(max_its=np.int64(5))),
                  ("seed=RandomState(3)", dict(seed=np.random.RandomState(3))), ("seed=default_rng(3)", dict(seed=np.random.default_rng(3))),
                  ("seed='3'", dict(seed="3")), ("init=4-comp for 8-comp target", dict(init=reduce_mixture(teacher, 4))),
                  ("init='cf4'", dict(init="cf4")), ("delta=0", dict(delta=0))]:
    args = dict(n_samples=400, seed=0, max_its=5); args.update(kw)
    target = random_gmm(8, 2, 7).estimator() if "8-comp" in label else tgt
    try:
        project(teacher, target, **args); print(label, "-> OK")
    except Exception as e:
        print(label, "->", type(e).__name__, str(e)[:150])
```

Observed on 0.8.2 (0.8.1 identical where marked): `n_samples=np.int64(400)` / `np.int32(400)` -> ValueError "n_samples must be a positive integer number of draws, got np.int64(400)" (0.8.1 same); `max_its=np.int64(5)` -> ValueError "max_its must be a positive integer, got np.int64(5)" (0.8.1 same); `seed=np.random.RandomState(3)` or `default_rng(3)` -> TypeError "Cannot cast scalar from dtype('O') to dtype('int64') according to the rule 'safe'"; `seed='3'` -> the same with `dtype('<U1')`; `init=` a 4-component model for an 8-component target -> ValueError "mixture estimate and accumulator component counts must match" (neither `init=` nor `project()` named, counts not given); `init='cf4'` -> TypeError "optimize(prev_estimate=...) takes a fitted distribution to start from, not str ..."; `delta=0` / `-1` / `'1e-6'` -> ValueError "optimize(): delta must be None (a fixed iteration count) or a finite positive number, got 0". `seed=np.int64(3)`, `seed=None` and `n_samples` equal to the component count all work; `n_samples=400.0` / `True` are refused by name.

Expected: accept `numbers.Integral` for `n_samples`/`max_its` (the values a user gets from numpy indexing) or say `int` in the message; refuse RandomState/Generator/str seeds by name ("seed= takes an int"); name `init=` and the two component counts in the mismatch; name `project()` (and `init=`) rather than `optimize(prev_estimate=...)` in the refusals of the new keywords.

Notes: the `n_samples`/`max_its` refusals pre-date 0.8.2; the `init=`/`delta=` messages are new surface.

### Q10-F09 (minor) -- `moment_project()`'s cap note drops both advice clauses because `_calling_verb()` reads `moment_project`'s signature (`**sampling_kw`), although the call passed `max_its` and `project()` accepts `max_its` and `delta`

Surface: `mixle.inference.moment_project` (`mixle/inference/project.py:255-276`) -> `mixle.inference.estimation._calling_verb` / `_knob_clause` (`estimation.py:907-960`, `1084-1096`); `examples/project_neural_to_structured.py` line 55. Repairs concerned: P10-F09, R07-F06.

Reproduction (`probes/p19_notes_user_view.py`, part (a)):

```python
import warnings, numpy as np
from mixle.inference import moment_project
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
rng = np.random.RandomState(1)
teacher = GaussianMixtureDistribution(rng.randn(6, 2) * 4, np.stack([np.eye(2)] * 6), np.ones(6) / 6)
target = GaussianMixtureDistribution(np.zeros((3, 2)), np.stack([np.eye(2)] * 3), np.ones(3) / 3)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    moment_project(teacher, target.estimator(), exact=False, n_samples=2000, seed=0, max_its=3)
for x in w:
    print(str(x.message)); print("Raise max_its" in str(x.message), "delta=None" in str(x.message))
```

Observed (`probes/p19_notes_user_view.full.txt`): "moment_project() stopped at the max_its cap (3) before the objective settled (last objective gain 117, delta=1e-06): the returned model is an unconverged fit, and its fit_provenance() reports converged=False." -- no "Raise max_its to fit to convergence" and no "pass delta=None ..." clause, although the call passed `max_its=3`. The shipped `project_neural_to_structured` run prints four such clause-less notes (line 55, `max_its=80`) while its `fit()` call on line 78 keeps the clauses (`runs/full/project_neural_to_structured.txt`).

Expected: the note names the knobs the caller controls on this route -- `max_its=` and `delta=` are forwarded through `**sampling_kw` to `project()`, which accepts both -- or the walk falls back to the innermost public verb that names them.

Notes: `_calling_verb()` takes the outermost public mixle frame and `inspect.signature(function).parameters`; for `moment_project` that set is `{teacher, target, exact, sampling_kw}`, so `_knob_clause` finds neither `max_its` nor `delta` and returns an empty remedy. Introduced by the R07-F06 repair of the P09-F09/P10-F09 notes.

### Q10-F10 (docs) -- `skeptic_challenge_example` act-3 banner "a torch flow EM-fit INSIDE a mixture, and it pays" contradicts its own printed numbers (hybrid -2.108 vs pure-neural -2.020 nats/row) and the docstring's "without embedding a winner in the source"

Surface: `examples/skeptic_challenge_example.py` line 223 (the banner) vs lines 276-281 (the measured comparison) and docstring lines 15-17.

Reproduction: `PYT_CWD=<dir> REVIEW_ROOT/tools/pyt.py 1800 REVIEW_ROOT/venv-full/bin/python REVIEW_ROOT/source/examples/skeptic_challenge_example.py`; `sed -n 15,17p REVIEW_ROOT/source/examples/skeptic_challenge_example.py; sed -n 223p REVIEW_ROOT/source/examples/skeptic_challenge_example.py`.

Observed (`runs/full/skeptic_challenge_example.txt`): the banner is printed unconditionally; the run then prints "all-neural (RealNVP coupling flow) held-out ll/row -2.020", "HYBRID mixture [flow, Gaussian], same EM held-out ll/row -2.108", "-> hybrid vs classical: +0.547 nats/row; hybrid vs pure-neural: -0.088 nats/row". The docstring: "The script prints all held-out likelihoods without embedding a winner in the source."

Expected: a banner that does not pre-announce the verdict (or one conditioned on the measured ordering), so the source embeds no winner as the docstring promises.

Notes: single run under contention (venv-full, 304 s); the ordering may change with the seed, which is exactly why the docstring promises no embedded winner.

### Q10-F11 (minor) -- `SemiSupervisedMixtureDistribution.log_density` on the distribution's own sampler output raises `TypeError: 'float' object is not iterable` -- a composite 2-tuple value is silently unpacked as (value, prior)

Surface: `mixle.stats.latent.semi_supervised_mixture.SemiSupervisedMixtureDistribution.log_density` (line 187; `_sum_prior_weights` at line 69) vs `SemiSupervisedMixtureSampler.sample` ("no prior labels are generated"); the configuration of `examples/semi_supervised_mixture_example.py`.

Reproduction (`probes/p25_semisup_sampler_datum.py`; first seen in `probes/p08_roundtrip_latent.full.txt`):

```python
from mixle.stats import CompositeDistribution, SequenceDistribution, CategoricalDistribution, GaussianDistribution
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureDistribution
mk = lambda p, mu: CompositeDistribution((SequenceDistribution(CategoricalDistribution(p), len_dist=CategoricalDistribution({10: 1.0})), GaussianDistribution(mu, 1.0)))
ssd = SemiSupervisedMixtureDistribution([mk({"a": .8, "b": .1, "c": .1}, 0.0), mk({"a": .1, "b": .8, "c": .1}, 1.0), mk({"a": .1, "b": .1, "c": .8}, 2.0)], [0.6, 0.3, 0.1])
raw = ssd.sampler(seed=1).sample(3)
print(ssd.log_density((raw[0], None)))   # the documented (value, prior-or-None) form
print(ssd.log_density(raw[0]))           # the sampler's own output
```

Observed (`probes/p25_semisup_sampler_datum.full.txt`): `ssd.log_density((raw[0], None))` -> -12.1542 and `ssd.log_density((raw[0], [(0, 1.0)]))` -> -11.6437; `ssd.log_density(raw[0])` -> `TypeError: 'float' object is not iterable`, raised at `semi_supervised_mixture.py:69` (`for idx, val in prior`) from `log_density` line 207: the (sequence, gaussian) composite sample was split into `x=sequence` and `prior=<the gaussian float>`. The fitted model's `dump_models`/`load_models`, `to_json`/`from_json` and pickle round trips in `probes/p08` used the `(value, None)` form and were exact.

Expected: either accept the sampler's own output (prior absent), or refuse a datum that is not a (value, prior-or-None) pair with a message naming that contract, instead of an internal TypeError that depends on what the component value happens to be.

Notes: the "a state the library itself produces is refused by its own surface" class. Because the datum contract is a 2-tuple and a Composite value is also a tuple, the ambiguity is silent for two-field composites: with a different second field the same call could mis-score rather than fail. (The probe's `seq_encode` calls fail with `AttributeError` because this family exposes `seq_log_density` through its encoder object, not a `seq_encode` method -- a probe error, not a finding.)

### Q10-F12 (minor) -- on the base install `mixle.task.distill_extractor` dies with a raw `ModuleNotFoundError` for torch (an unguarded `import torch` at `extract.py:307`) while `solve()`/`distill` refuse by naming the optional extra; four sibling examples crash on their own unguarded imports, one without declaring the dependency

Surface: `mixle.task.distill_extractor` (`mixle/task/extract.py:291-307`) via `examples/task_extraction_example.py` line 70; `examples/multimodal_stage1_demo.py:134` (`import torch`), `examples/vlm_trust_receipts_demo.py:73` (`import torch`), `examples/symbolic_export_example.py:34` (`import sympy`), `examples/skeptic_challenge_example.py:89` (`from sklearn.ensemble import ...`). Repair concerned: P10-F08.

Reproduction (venv-base = numpy + scipy only; outputs in `runs/base/<name>.txt`):

```sh
for n in task_extraction_example task_distill_example multimodal_stage1_demo vlm_trust_receipts_demo symbolic_export_example skeptic_challenge_example peft_lora_grad_leaf; do
  PYT_CWD=<dir> REVIEW_ROOT/tools/pyt.py 300 REVIEW_ROOT/venv-base/bin/python REVIEW_ROOT/source/examples/$n.py | tail -4
done
```

Observed: `task_extraction_example` -> traceback ending in `mixle/task/extract.py` line 307, `distill_extractor` -> `ModuleNotFoundError: No module named 'torch'`. By contrast `task_distill` / `task_cascade_economics` / `task_llm_active` / `win_demo` end with `ImportError: the default distilled student (a torch MLP) requires the optional torch dependency, which is not installed. Install it with pip install "mixle[torch]", or use the torch-free generative student (student="generative" in solve(), or distill_structured_from_labels / the mixle.task.generative_text distillers for text).` (`mixle/task/distill.py:687`); `peft_lora_grad_leaf` names torch, the pip command and the Hugging Face download (the P10-F08 repair, verified here); `project_neural_to_structured` says "this example trains a torch normalizing flow: pip install torch"; `shared_embedding` says "ImportError: build_causal_lm requires torch.". The four sibling examples die raw: `multimodal_stage1_demo` at its line 134 (its docstring does say `mixle[torch]`), `vlm_trust_receipts_demo` at line 73 (its 41-line docstring never names torch), `symbolic_export_example` at line 34 in 0.1 s (its docstring calls only Sage optional; sympy is not a mixle dependency), `skeptic_challenge_example` at line 89 for sklearn after 4 s of fitting.

Expected: `distill_extractor` raises the same named ImportError its sibling distillers raise (the guard at `distill.py:687` already has the wording); the examples either guard their imports with the P10-F08 wording or declare the extra in their docstring.

Notes: "the library must work and fail cleanly on the base install" holds for the 17 scripts that run there and for the `solve()`/`distill` verbs; `distill_extractor` is the one library surface in this half that does not.

## Attacks that did not break anything

- P10-F13 on the benchmark's own target family (`probes/p01_project_guards.full.txt` A-C): `project()` of the 8-component `GaussianMixtureEstimator` with 1 or 4 draws is refused by name ("project cannot fit a 8-component target from 1 draw(s) ... Raise n_samples, or project onto a smaller family"), as is the generic `MixtureEstimator`; the boundary `n_samples == components` passes (p01 P/Q, p03 P). `init=` really warm-starts: one EM iteration from the closed form gives KL 0.1208 vs the closed form's 0.1252 (p01 R).
- `project()` seed spellings (`probes/p03_project_seeds.{full,081}.txt`): `seed=np.int64(3)` and `seed=None` accepted on both versions; `n_samples=400.0` / `True` refused by name.
- P10-F01 on the shipped quickstart planting at seeds 0/7/100/200/300 (`probes/p04_p10f01_dependency_screen.full.txt`): all three pairs reported at 0.36-0.75 bits; controls (gaussian-recommended amount, `None` every 17th row, a constant field, heavy-tailed t2, 3-level numeric, integer counts by category) screened sensibly; `explain()` at seed 0 lists the three dependencies and the non-tree-edge note (also in `runs/full/quickstart_example.txt`).
- P10-F05 round trips (`probes/p08_roundtrip_latent.full.txt`): `dump_models`/`load_models` (idempotent), `to_json`/`from_json` and pickle of fitted and generating `HierarchicalMixture` and `JointMixture`, and of the fitted `SemiSupervisedMixture`: max |delta log-density| = 0 on held-out rows.
- P10-F04 at the shipped seed (`probes/p02_structured_hmm_sweep.full.txt`): low-rank section, seed 0, 60 sequences, 3 restarts -> per-restart errors 0.14/0.16/0.16, best-by-likelihood 0.16 from an init error of 1.97 (PASS). The sweep over further seeds timed out (see below).
- Repro bundle entry `scaling-backend`: digest matches its pin (`probes/p05_bundle_digests.full.txt`).
- Independent recomputation of printed numbers (`probes/p14_deterministic_recompute.base.txt` on venv-base, plus inline `math.comb` checks): precedence closure +35.0 and schedule NPV 30.5353 (brute force over assignments), label_economics sign test 0.0078125, task_cascade exact paired p 0.0414 / 0.8318, task_llm_active 0.125, symbolic_export scores at x=10/100 (Gaussian -5.3333/-65.3333, StudentT -0.4808/-0.0500) -- all match the runs.
- Warning audit over the 29 venv-full runs (`probes/p17_warning_audit.full.txt`): no cap note attributed to a library file (the P10-F09 attribution repair); the two downhill runs (`peft_lora_grad_leaf`, the skeptic's flow) print the "BEST iterate seen" wording the migration guide describes; the task-cluster counts fell from 144/129/66/41/28 (0.8.1 round) to 0/0/1/4/6 (`task_distill`, `task_llm_active`, `win_demo`, `task_cascade`, `skeptic`).
- venv-base vs venv-full stdout for the 17 scripts that run on both: identical except unseeded draws, 1e-14 float noise, the timing ratio, and the numba path's `compiled-em:` diagnostic lines (see the corpus table); the pure-numpy kernels reproduce the numba results.
- The production-provenance stdout is byte-identical between the installed numpy/scipy and the pinned closure (numpy 2.4.6 / scipy 1.17.1), and between the wheel and the 866078be source tree on `sys.path` (`probes/p23_source_tree_digest.full.txt`), so Q10-F02 is not arithmetic drift.

## What was not covered

- P10-F04 across seeds: `probes/p02_structured_hmm_sweep.py` finished only seed 0 before its 1700 s budget ran out under the shared load; seeds 1-6 and the 10 % data variant of the 0.8.1 report are untested.
- P10-F02: LDA at the sampling seeds that used to raise `LDAConvergenceError` (`probes/p06_p10f02_lda_seeds.py`, written, not run); the shipped example ran clean at its own seed only.
- P10-F13 at the example level (one-row joint / semi-supervised / model-comparison variants, the empty encoded batch, the empty evaluation split; `probes/p07_p10f13_examples_one_row.py`, not run) -- only the `project()` guard was verified.
- P10-F03 across teacher/projection seeds (`probes/p11_reduction_seeds.py`), P10-F06 chain health at other seeds (`probes/p12_modelcmp_seeds.py`), `structure_learning_example` at other seeds and the A-03 `tie_tolerance` (`probes/p13_structure_learning.py`), P10-F07/A-01 on the lookback example across seeds and single-restart 0.8.1-vs-0.8.2 (`probes/p10_lookback.py`), and `probes/p21_project_delta_none.py`: written by the original reviewer, not run (each is several minutes under the current load); the example runs themselves carry the numbers the findings cite.
- venv-nonumba: no runs.
- The docstring runtime claims (hierarchical "about twenty seconds", lookback "~20-25 s", multimodal "milliseconds", model_comparison "a couple of seconds", win_demo "~1 minute"): wall times under this load are not evidence; the cited exec log has 29.7 s, 12.6 s, 12.4 s, 19.2 s and 5.2 s.
- LDA / IBP / Gaussian-admixture serialization round trips (`probes/p08_roundtrip_latent.py` sections after the Q10-F11 crash) were not reached.
- The regression tests behind P10-F01, P10-F05, P10-F13, A-01/P10-F07 and A-02/A-03 were read by the original reviewer (see `RECOVERED_NOTES.md`) and not re-run here; the per-example claims in `manifests/` were not re-checked against the fresh outputs beyond what the findings name.

DONE 10 12
