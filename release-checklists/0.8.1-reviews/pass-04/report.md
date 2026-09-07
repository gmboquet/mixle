# Adversarial review — PASS 04 — mixle 0.8.1 release candidate

- **Pass:** 04 of 10 (independent)
- **Focus:** `mixle.ppl` — RandomVariable construction, `.fit` (`map`/`mcmc`/`hmc`/`nuts`/`ensemble`/`vi`/`laplace`/`auto`/`posterior`/`sample`), constraints (`< > <= >=`, `eq`, `ne`, `increasing`/`decreasing`/`convex`/`concave`/`lipschitz`, `& | ~`), `penalty=`, `potentials=`, `given()`/`constrain()`, `summary()`, `params`, `explain_fit`, and the rewritten derivative-free constrained MAP path (Powell penalty ramp + feasibility repair + Nelder-Mead polish; grouped path too).
- **Wheel:** `/Users/grantboquet/mixle/.ci-repro-colima/candidate-081/dist/mixle-0.8.1-py3-none-any.whl`
  sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d` (re-hashed locally, matches), built from tree `6040ea38` on `release/0.8.1`.
- **Version verification** (run from `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-04/`):
  `python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"` →
  `0.8.1 /Users/grantboquet/mixle/.ci-repro-colima/candidate-081/venv/lib/python3.12/site-packages/mixle`
- Interpreter: CPython 3.12.12; torch 2.14.0 present in the venv (torch-absent paths exercised by patching `mixle.ppl.autograd.torch_available` in-process; the venv was not modified).
- Work dir (all scripts, logs, executed notebooks): `/Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-04/`. Nothing outside it was modified.

## Corpus executed on the candidate

Command per notebook: `<venv>/bin/python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 <nb> --output <workdir>/executed-N.ipynb` (wrapped in a 1300 s Python `subprocess.run` timeout; macOS has no `timeout`). Wall times from `corpus_results.tsv`.

| # | Notebook | Exit | Wall |
|---|---|---|---|
| 1 | tutorials/probabilistic_programming.ipynb | 0 | 243 s |
| 2 | data_science/constrained_inference.ipynb | 0 | 11 s |
| 3 | data_science/ppl_end_to_end_case_study.ipynb | 0 | 8 s |
| 4 | data_science/mcmc_uncertainty_quantification.ipynb | 0 | 16 s |
| 5 | data_science/variational_inference_vs_mcmc.ipynb | 0 | 8 s |
| 6 | data_science/bayesian_workflow_predictive_checks.ipynb | 0 | 21 s |
| 7 | data_science/hierarchical_partial_pooling.ipynb | 0 | 6 s |
| 8 | data_science/exponential_families_and_conjugacy.ipynb | 0 | 5 s |
| 9 | architecture_studies/ppl_scaling_vs_pyro_stan.ipynb | 0 | 20 s |

No executed cell raised. Executed copies: `executed-1.ipynb` … `executed-9.ipynb`; logs `nb-N.log`.

Examples (every example importing `mixle.ppl`), run from the work dir:

| Example | Exit | Wall |
|---|---|---|
| examples/ppl_example.py | 0 | 3 s |
| examples/extensibility_seams_example.py | 0 | 2 s |
| examples/model_comparison_example.py | 0 | 4 s |
| examples/shared_embedding_example.py | 0 | 3 s |
| examples/flagship_physics_inverse.py | 0 | 5 s |

Output logs: `ex-<name>.log`. Printed values are consistent with the scripts' own stated truths (e.g. `ppl_example`: Poisson 3.479 vs 3.50, mixture means [-5.00, 5.02]).

## Attack scripts (all in the work dir)

| Script | What it does | Log |
|---|---|---|
| `attackA_isotonic.py` | constrained MAP vs hand PAV / convex-regression targets, d = 1, 2, 5, 10, 20; boundary-active and interior scenarios; `increasing`/`decreasing`/`convex` | `attackA.log`, `attackA_results.json` |
| `attackB_edges.py` | infeasible sets, wrong-variable constraints, `penalty` 0/neg/huge/inf/nan, `max_iter`=0/1/1.5/True, `tol`=0/-1/inf, hard+soft mixes, `ne`/`~`/`\|`, `how='auto'`+`explain_fit`, MCMC draws/burn/thin/chains extremes, determinism, params shapes, `summary()` unfitted, pickling | `attackB.log` |
| `attackC_surfaces.py` | `given()`, `constrain()`, `potentials`, scalar-prior boundary optimum (hand answer), `Mix` ordered means by MAP, `Beta` `eq` penalty, grouped (`.each()`) constrained MAP/MCMC, coupled mean+var free | `attackC.log` |
| `attackD_sweep.py` | dimension sweep d = 6…15 for `convex`/`concave`/`lipschitz`; `increasing` with random-permutation column means at d = 10, 20; traces `_repair_feasibility` | `attackD.log`, `attackD_results.json` |
| `attackE_misc.py` | `dump_models` on PPL fits, `how='posterior'`/`'sample'` + constraints vs `explain_fit`, samplers at a hard wall, torch-absent path via `torch_available` patch, `max_iter=1` silent success | stdout (re-run to reproduce) |
| `attackF_parallel_hmc.py` | `parallel=True` timing; HMC/NUTS/ensemble/MCMC at a hard wall with 2000 draws vs truncated-normal reference | stdout |
| `attackG_hmc.py`, `attackH_diag.py` | scope of the HMC stall; whether `posterior_summary` catches it | stdout |
| `repro_F01.py`, `repro_convex10.py`, `repro_decreasing5.py`, `repro_pickle.py` | minimal reproductions cited by the findings | stdout |

---

## Findings

### P04-F01 (blocking) — Constrained derivative-free MAP returns its *starting point* and reports "constrained optimum reached" whenever the penalty optimum is a hair infeasible on several tied constraints

**Surface:** `RandomVariable.fit(data, how='map'|'auto', constraints=[increasing(v) | decreasing(v) | convex(v) | concave(v) | ...])` → `mixle.ppl.inference.map_fit` → `_derivative_free_constrained` → `_repair_feasibility` (`inference.py` lines 1442–1475, 1477–1541).

**Reproduction:** `cd /Users/grantboquet/mixle/.ci-repro-colima/reviews-081/pass-04 && <venv>/bin/python repro_F01.py` (no monkeypatching; numpy/scipy + wheel only). Output on the candidate:

```
--- increasing d=10           (column means = RandomState(101).permutation(10)*0.3, sd 0.5, n=200/column, var fixed)
  fitted mean : [1.347]*10                       <- constant vector = grand mean = optimizer start u0
  hand target : [1.032, 1.032, 1.032, 1.168, 1.534, 1.534, 1.534, 1.534, 1.534, 1.534]   (PAV)
  max|fit-target| = 0.315   NLL(fit)-NLL(target) = 221.4 nats
--- convex d=10               (column means 0,2,1,3,4,...,9 — the tutorial's pattern extended to 10-D)
  fitted mean : [4.492]*9 + [4.497]
  hand target : [0.456, 1.095, 1.734, 2.787, 3.841, 4.894, 5.948, 7.002, 8.055, 9.109]  (convex regression, SLSQP)
  max|fit-target| = 4.612   NLL(fit)-NLL(target) = 34222.2 nats
```

Breadth (`attackA.log`, `attackD.log`; `nll_gap_nats` = objective loss vs the hand optimum, `returned_anchor` from tracing `_repair_feasibility`):

| constraint | d | scenario | result |
|---|---|---|---|
| `increasing` | 10 | random-perm means, seeds 100/101/102 | collapsed 3/3 (gaps 1417 / 216 / 15 nats), `returned_anchor: true` |
| `increasing` | 20 | random-perm means, seeds 101/102 | collapsed 2/2 (8708 / 3613 nats); seed 100 "ok" only because its PAV target *is* the constant vector |
| `convex` | 7, 8, 10 (4/4 seeds), 12, 15, 20 (3/3 scenarios) | one adjacent swap / reversed / interior | collapsed every time (10.7k → 111k nats); d = 20 *interior* (constraint should be inactive) returns constant 19.01 vs target 0.03…38.0, `max_abs_err` 18.98 |
| `concave` | 6, 7, 8, 9, 12, 15 | one adjacent swap | collapsed every time |
| `increasing`/`decreasing`/`convex` | 1, 2, 5; `increasing` 10/20 one-swap; `lipschitz` 6–15 | — | correct (max err ≤ 3e-3) |

**Mechanism (from `repro_convex10.py`, which traces every `scipy.optimize.minimize` call):** the Powell ramp itself converges correctly (d=10 convex: weight-1e9 iterate `[0.474, 1.109, 1.744, 2.795, …]`, within 0.02 of the target). Its result is O(1e-13) infeasible. `_repair_feasibility` first tries a step along the summed constraint normals (fails: with several tied first/second differences one step un-fixes neighbours), then its "last resort" walks the segment `u + t*(anchor - u)`, `t ∈ logspace(-9, 0, 46)`, toward the anchor `u0`. `u0` is the constant grand-mean vector, which lies **on** the boundary of every shape constraint (all differences exactly 0), so every point of the segment with `t < 1` is still infeasible and the loop returns `t = 1`, i.e. `u0` itself. `_derivative_free_constrained` then polishes with Nelder-Mead from `u0` (simplex vertices `u0 + 1e-3 e_k` are infeasible for shape constraints, so it cannot move) and returns `(u0, f, "powell-penalty+nelder-mead", True, "constrained optimum reached")`. `map_fit` sees `ok=True` and returns the fit with no warning; `explain_fit()` says route `map`, `.result` is `None`.

**Expected:** the constrained MAP (isotonic/convex projection of the column means for a fixed-variance diagonal Gaussian), or an exception. **Observed:** the untouched initial point, presented as a converged constrained optimum; up to 1e5 nats worse than the optimum; in the "interior" convex case the returned answer is 19 units from the unconstrained MLE that the constraint should not have touched.

**Notes:** This is the exact defect class the 0.8.1 CHANGELOG entry ("Constrained derivative-free MAP fits … no longer run Nelder-Mead into a hard feasibility wall … reported convergence at a nearly constant vector far from the constrained optimum") claims to have removed. The tutorial's 5-D case now matches PAV (verified, `executed-1.ipynb` cell 22: `[0.01 1.44 1.44 3.06 3.99]`), but the fix holds only where the normals-step repair happens to succeed. Any user applying the notebooks' own recipe (`DiagGaussian(d, mean=free(d)).fit(X, how='map', constraints=[increasing(v)])`) to ≥ 8–10 positions with more than one pooled block gets a silently wrong point estimate. Since the polish result is discarded when it does not improve, the code path has no check that the returned point is *anywhere near* the ramp's own final iterate — comparing `walled(u_repaired)` against the ramp objective would have caught every case above.

### P04-F02 (real) — The constrained MAP path never checks Powell's convergence status; `max_iter=1` returns a wrong answer as a success while the unconstrained path raises

**Surface:** `map_fit` derivative-free branch; `_derivative_free_constrained` ignores `penalized.success`/`nit` (inference.py 1503–1516).

**Reproduction:**
```python
import numpy as np; from mixle.ppl import DiagGaussian, free, increasing
r = np.random.RandomState(0); X = np.stack([r.normal(l, 0.5, 300) for l in [0,2,1,3,4]], axis=1).tolist()
v = free(5, name="v"); f = DiagGaussian(5, mean=v, var=np.full(5, .25)).fit(X, how="map", constraints=[increasing(v)], max_iter=1, rng=np.random.RandomState(0))
print(np.round(f.params["mean"], 3))       # [0.014 1.191 1.191 3.059 3.992]   (PAV: 1.44, 1.44) -- no error, no warning
from mixle.ppl import Normal
Normal(Normal(0,10,name="mu"), free).fit(list(r.normal(5,2,400)), how="map", max_iter=1)   # RuntimeError: MAP optimization failed: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
```
(also `attackB.log` "max_iter=1 (kw)" / "max_its=1 (fit arg)", `attackE` "constrained map max_iter=1".)

**Expected:** a budget-exhausted constrained fit raises (as the unconstrained path does) or at least reports non-convergence; **Observed:** wrong point returned with `success`. Together with F01 this means `ok=True` on the constrained path carries no information. The CHANGELOG's "Failure messages name `max_iter`" is moot when the failure is never detected.

### P04-F03 (real) — Constrained MAP whose optimum is a fully-tied corner (all coordinates pooled) stops 2–3 standard errors short of it

**Surface:** same path; the Nelder-Mead polish against the `1e18` wall cannot move a point whose every constraint is active.

**Reproduction:** `<venv>/bin/python repro_decreasing5.py` (d=5, `decreasing(v)` on increasing data: optimum = constant grand-mean vector). Seeds 0/1/2: fit 3.9829 / 4.0311 / 3.9813 vs target 3.9774 / 4.0194 / 3.9761 (errors 0.0055–0.0117; 0.06–0.27 nats). `attackA.log` d=10 interior `decreasing`: 9.0301 vs 9.0016 (err 0.0285, ≈ 3.3 nats; pooled-mean SE is 0.011, so 2.6 SE off). Trace shows the weight-1e9 Powell iterate already off by that amount and the polish returning its start.

**Expected:** with `tol=1e-8` the pooled mean to ~1e-6; **Observed:** 5e-3–3e-2 error growing with d. Minor in absolute terms but it is a systematic bias at exactly the boundary configuration the constraint exists for.

### P04-F04 (real) — `how='hmc'` with an *active* hard constraint returns a degenerate "posterior" (0 % acceptance, every draw equal to the projected start, std 0) with no error or warning; vector models raise instead

**Surface:** `mixle.ppl.inference.hmc_fit` (2123–2181) + `mixle.inference.mcmc.samplers.hamiltonian_monte_carlo`.

**Reproduction:** `<venv>/bin/python attackF_parallel_hmc.py` (second block) / `attackG_hmc.py`:
```
hmc       mean=6.2353 std=0.0000 min=6.2353 q2.5=6.2353 ess_bulk=nan acc=0.0 frac<6.05=0.000   (2000 draws, burn 500, constraints=[mu > 6])
nuts      mean=6.0113 std=0.0112 ... acc=0.899
ensemble  mean=6.0117 std=0.0116 ... acc=0.663
mcmc      mean=6.0099 std=0.0096 ... acc=0.136
reference truncated-normal mean=6.0091 std=0.0090
```
Same with `chains=2` and with torch absent. Inactive constraints (`mu > 0`, `mu < 100`), `penalty=` (soft), and `potentials` all sample normally (acc ≈ 0.99). Vector case: `DiagGaussian(5, mean=free(5)).fit(X, how='hmc', constraints=[increasing(v)])` raises `ValueError: grad_log_target returned non-finite values` from the numeric gradient at the projected (boundary) start.

`f.summary()` reports `{'mean': 6.2353, 'std': 0.0, 'q2.5': 6.2353, 'q97.5': 6.2353, ..., '_acceptance_rate': 0.0}` silently. `posterior_summary(f)` does mark it `diagnostic_status: 'unusable'`, so the release's gate catches it when the user goes through that surface.

**Expected:** either samples from the truncated posterior (as NUTS/ensemble/MCMC do) or a raised error; **Observed:** a fixed point dressed as a posterior. The docstring only says ensemble/MCMC "usually mixes better".

### P04-F05 (real) — Every posterior-bearing PPL fit is unpicklable; `dump_models` rejects `RandomVariable`; `explain_fit` docstring promises pickling

**Surface:** `RandomVariable` results of `how='conjugate'|'mcmc'|'hmc'|'nuts'|'ensemble'|'laplace'|'vi'|'vmp'`; `mixle.stats.dump_models`.

**Reproduction:** `<venv>/bin/python repro_pickle.py`:
```
map_constrained_diag: pickle OK   map_normal: OK   em_normal: OK   hierarchical: OK
conjugate: PICKLE FAILED AttributeError: Can't get local object '_conj_normal_mean.<locals>.<lambda>'
mcmc/hmc/nuts/laplace: PICKLE FAILED ... '_finalize.<locals>.predictive'
ensemble: ... '_finalize_chains.<locals>.predictive'   vi: ... 'vi_fit.<locals>.predictive'   vmp: ... 'vmp_fit.<locals>.predictive'
```
`dump_models(rv)` → `SerializationError: objects of type mixle.ppl.core.RandomVariable are not JSON serializable by mixle` for every fit; `dump_models(rv.dist)` works but keeps only the point distribution (posterior, chains, summary, explanation all dropped) — `attackE` block 1.

**Expected:** per `RandomVariable.explain_fit` docstring (core.py 2612): "The record travels with the model through pickling, so a reloaded artifact still explains how it was fit." **Observed:** only point-estimate fits survive `pickle`; no PPL-level serialization exists for a posterior. (These closures are also why `parallel='process'` is forced off with constraints.)

### P04-F06 (docs) — Two shipped notebooks carry stored outputs that show the pre-fix collapse (and one contradicts its own prose)

**Surface:** `notebooks/tutorials/probabilistic_programming.ipynb` cell 22; `notebooks/data_science/constrained_inference.ipynb` cell 13.

**Reproduction:** compare the shipped notebooks with `executed-1.ipynb` / `executed-2.ipynb`:
- Tutorial cell 22 stored: `increasing(v) : [2.   2.   2.   2.   2.11] (monotone: True )`; candidate: `[0.01 1.44 1.44 3.06 3.99]`. The code cell still carries the comment `# pinned: constrained 5-D Nelder-Mead MAP is sample-sensitive; this sample converges` and `max_iter=20000`.
- constrained_inference cell 13 stored: `RMSE to the true profile   unconstrained MLE: 0.100   increasing-constrained: 0.860` directly under a markdown section asserting the constraint "beats the unconstrained estimate"; candidate prints `0.100 / 0.100` (the MLE is already monotone for this seed, so the constrained fit equals it).

**Expected:** shipped outputs match the candidate; **Observed:** the shipped corpus documents the bug the CHANGELOG says is fixed and, in the second case, an 8.6× worse "regularized" fit presented as an improvement.

### P04-F07 (minor) — Non-finite penalty/potential values are accepted and surface as "Optimization terminated successfully … Raise max_iter"

**Surface:** `_soft_penalty` (accepts `penalty=inf`/`nan`: `w > 0` passes), `_potential_term`, and the walled Nelder-Mead fallback's error string (inference.py ~2513).

**Reproduction:** `attackB.log` "penalty=inf"/"penalty=nan" → `RuntimeError: MAP optimization failed (nelder-mead): Optimization terminated successfully.. Raise max_iter (currently 5000) or loosen tol.`; `attackC.log` "potential returning nan/+inf" → same message; `attackD.log` d=9 `convex` (repair fails entirely, walled NM starts infeasible) → same message. **Expected:** `penalty` validated finite; the failure message naming the actual cause (non-finite objective / infeasible start). **Observed:** contradictory message that recommends raising `max_iter`.

### P04-F08 (docs) — `potential()` silently accepts an RV that is not a model parameter and turns it into an extra latent, contradicting its docstring

**Surface:** `mixle.ppl.core.potential` docstring (887–911: "every referenced variable must be a parameter of the fitted model … as with constraints"); `inference._potential_latents` (790) adds foreign RVs as latents.

**Reproduction:** `attackC.log` "potential on foreign var": `Normal(mu, sg).fit(dd, how='map', potentials=potential(lambda z: 0.0, zz))` with `zz = Normal(0,1,name='zz')` not in the model → returns `{'mean': 4.94, 'sd': 1.98}`, no error (a constraint on the same `zz` raises). **Expected/Observed:** docstring says raise; code integrates `zz` with its prior. One of them is wrong; the behaviour is defensible but undocumented.

### P04-F09 (minor) — `explain_fit()` on a fitted model loses the auto-routing reason

**Reproduction:** `attackB.log` "auto with constraints -> explain_fit": pre-fit `{'route': 'map', 'reason': 'constraints/potentials need the numerical joint -> MAP (a point estimate)'}`; after `fit(how='auto', constraints=...)` the bound model reports `reason: "explicit how='map' (requested how='auto'; resolved to 'map' by fit)"`, `route_requested: 'auto'`. **Expected:** the stored record keeps *why* auto chose MAP; **Observed:** "explicit how='map'" for a fit the user never requested as map (the `route_requested` key does preserve the request, so this is wording, not a wrong route).

### P04-F10 (minor) — `summary()`/`params` on an unfitted vector-parameter model raises an unrelated numpy error

**Reproduction:** `attackB.log`: `DiagGaussian(5, mean=free(5, name='v'), var=np.full(5,.25)).summary()` → `ValueError: setting an array element with a sequence.` whereas `Normal(free, free).summary()` → `ValueError: Normal has unresolved free parameters; call .fit(data) first.` **Expected:** the same clear message.

### P04-F11 (minor) — Input-validation gaps around samplers/potentials

- `potentials='x'` → `AttributeError: 'str' object has no attribute 'vars'` (`attackC.log`).
- `how='mcmc', draws=1` (single chain) is accepted and returns `std 0.0`, `_acceptance_rate 0.0`, all diagnostics NaN, while `draws=1, chains=2` raises `split_rhat(): at least 4 draws per chain are required` (`attackB.log`) — the single-chain path has no floor.
- `rng=` must be a `numpy.random.RandomState`; an int seed or `default_rng` raises `TypeError` for every sampler, while `predict(rng=int|Generator)` accepts both (`attackB.log` "mcmc rng=int seed").

### P04-F12 (minor) — Grouped constrained MAP reports `optimizer.iterations = 0`

**Reproduction:** `attackC.log` "grouped map constrained m>3": `Normal(Normal(m, 3.0, name='th').each(), free).fit(groups, how='map', constraints=[m > 3])` → `'optimizer': {'algorithm': 'powell-penalty+nelder-mead', 'success': True, 'iterations': 0, 'message': 'constrained optimum reached', ...}` (inference.py ~2421: `grouped_nit = 0` hard-coded on the derivative-free branch). The value itself is right (`m = 3.000000000000088`, boundary). **Expected:** a real iteration/evaluation count or the key omitted.

---

## Attacks that did not break anything

- **Constrained MAP, low dimension / benign ties:** d = 1, 2, 5 for `increasing`/`decreasing`/`convex` (boundary and interior), d = 10/20 `increasing` with a single adjacent swap, `lipschitz(0.5)` d = 6…15: max error ≤ 3e-3, feasible, 0.02–62 s (`attackA.log`, `attackD.log`). The tutorial's 5-D case matches PAV on the candidate. Constrained fits are deterministic with and without `rng`.
- **Scalar-prior boundary optimum (hand answer):** `Normal(Normal(0,10,name='mu'), free).fit(dd, how='map', constraints=[mu > 6])` with x̄ = 4.94 → `mu = 6 + 2.8e-12`, `sd` within 1.7e-6 of `sqrt(mean((x-6)^2))`; `mu >= 6` and `how='auto'` identical; inactive `mu > 0` reproduces the unconstrained MAP to 2e-8 (`attackC.log`).
- **Coupled mean+var free with `increasing(mean)`:** matches a hand weighted-PAV/variance fixed-point iteration to 1e-3 in mean and 5e-7 nats (`attackC.log`).
- **Infeasible constraint sets** (`increasing(strict) & decreasing(strict)`, `(v[0] > 1) & (v[0] < 0)`, MCMC with `(mu > 100) & (mu < -100)`): raise `ValueError: could not find a parameter point satisfying the constraints` in ≤ 1.2 s; no spin. `given(x > 50)` / `constrain((a<b)&(b<a))` raise after 102,400 attempts (≈ 0.02 s); `prob()` receipts return 0 with a Wilson interval.
- **Wrong-variable constraints:** foreign handle / foreign scalar prior → clear `ValueError`. `penalty=0`/negative → `ValueError: penalty weight must be positive`. `penalty=1e300` and `1e-300` run. `max_iter=0/1.5/True`, `tol=-1/inf` → clear `ValueError`/`TypeError`. `tol=0` runs (13 s). `constraints=['bad']` → `TypeError`. Python chained comparison → the documented `TypeError`. `ne`/`~` with `penalty=` → clear `ValueError`; hard `ne`/`~increasing` and `increasing | decreasing` all fit (walled Nelder-Mead fallback).
- **Mixed hard + soft:** `[increasing(v), eq(v[0], 0)]` → soft optimum `v[0] = 0.008` matches the closed-form penalized value (0.0075) with default weight 1000; with `penalty=1e6` all constraints go soft and `v[0] = 0.000`. `Beta(a,b)` with `eq(a+b, 10)` gives a+b = 9.995 (penalty 500) / 9.998 (auto 1000), matching the notebooks.
- **`how` gating:** `em`/`vi`/`conjugate` with constraints or potentials raise the documented `ValueError`; `laplace` with hard constraints raises `NotImplementedError` (and `explain_fit(how='posterior', constraints=...)` says `laplace` for a fit that then raises — borderline, not filed since `posterior` is documented as the *ladder*, not a promise); `how='sample'` with constraints dispatches to `ensemble` and the bound model's `explain_fit` reports `route='ensemble', route_requested='sample'` truthfully; `explain_fit(how='bogus')` lists the exact `fit()` vocabulary.
- **Samplers at a hard wall (except HMC):** NUTS, ensemble, and RW-Metropolis with `constraints=[mu > 6]` all match the truncated-normal reference (mean 6.009–6.012, std 0.0096–0.0116, min draw ≥ 6.0000); `penalty=10` makes the wall soft and draws go below 6 as designed.
- **MCMC controls:** `draws=0`, `thin=0`, `chains=0` → clear `ValueError`s; `burn > draws` runs. Same `RandomState` → bit-identical summaries for mcmc/hmc/nuts/ensemble/vi/map (the `==` false results in `attackB.log` are only NaN R-hat ≠ NaN). `parallel=True` with constraints silently runs serial and gives the identical result to `parallel=False` (5.7 s process-pool spawn overhead without constraints).
- **Torch-absent path** (patching `mixle.ppl.autograd.torch_available`): unconstrained MAP emits the documented `RuntimeWarning` and matches the L-BFGS answer to 1e-8; constrained MAP, Laplace, HMC, VI all run with numeric gradients and agree with the torch path; `max_iter=1` raises with a message naming `max_iter`. (A `sys.modules['torch']=None` subprocess was invalid — it breaks scipy's array-API shim, not mixle.)
- **`given()`/`constrain()`:** truncation moments (E[x|x>0] = 0.797, var 0.364, log_prob −inf outside), nested/foreign/free/measure-zero misuse all raise clear errors; `constrain(a<b, b<c)` means (−0.85, 0.00, 0.86); `columns`, `prob()`, `log_prob` on/off region behave.
- **`params`/`summary` keys and shapes:** `{'mean': (d,), 'var': (d,)}` for DiagGaussian on EM/MAP/constrained/mean+var-free; `{'mean','sd'}` for Normal; `{'a','b'}` for Beta; posterior summaries carry `mean/std/q2.5/q97.5/mcse` plus R-hat/ESS as documented. Concrete `Normal(0,1).summary()` → params. Unfitted `Normal(free, free).summary()` → clear error.
- **Grouped path:** `.each()` plain auto → `HierarchicalPosterior`; constrained MAP (`m > 3` boundary → 3.0000; `increasing(th)` inactive → equals unconstrained L-BFGS to 1e-6); constrained MCMC runs.
- **`Mix` ordered means by MAP** (`mu0 < mu1` and reversed): correct component means either way (12 s vs 0.1 s, no hang).
- **Memory/hangs:** nothing exceeded 62 s or grew memory; every run stayed under its alarm.
