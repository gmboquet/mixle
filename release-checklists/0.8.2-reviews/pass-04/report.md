# Pass 04 (recovery write-up) -- the probabilistic-programming layer

**Focus:** RandomVariable, .fit on every route (map/mcmc/hmc/nuts/ensemble/vi/laplace/auto/conjugate), constraints, penalties, potentials, summary/explain_fit/predict, posteriors and their persistence, regressions/GLMs.

**Wheel:** `mixle-0.8.2-py3-none-any.whl` sha256 `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da`  
**Commit:** `866078be520b22188110780be957150dc6da964c`  **Tree:** `7922a8c59283ecef6877104b9a7499afd52d9013`  
**Work dir:** `REVIEW_ROOT/pass-04/` (probes in `probes/`, outputs in `out/` and `out/q/`, executed notebooks in `corpus/notebooks/`).

**Recovery note.** The original reviewer executed the corpus and wrote probes p1-p3b (outputs `out/p*.txt`) and was cut off before reporting. This write-up reconstructs its findings from those outputs (`RECOVERED_NOTES.md`) and adds the q-series probes (`probes/q*.py`, `out/q/`) for the repairs the trail had not reached. Nothing below is reported without a saved output.

## Environment verification

Command: `cd /tmp && <venv>/bin/python REVIEW_ROOT/tools/verify_env.py` (out/verify_env.txt):

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
executable       <review-store>/candidate-081/venv/bin/python
mixle.__path__   <review-store>/candidate-081/venv/lib/python3.12/site-packages/mixle
dist version     0.8.1
source_commit    c9c5fbbbbd63afbebcaa00d4bf4471d7464d8b02
source_tree      187468d13b768e166ed42f80420071ed23d3e790
content_sha256   717f80ef0b6cf32759a26ccf2696e049d8c3bf4c2c0fa61fdf35d90688e5eb59
```

## Corpus executed on the candidate

Environment: `venv-full` (numba, torch, pandas, pyspark, scikit-learn), 3 notebooks at a time, `tools/run_nb.sh` timeout 1800 s; examples through `tools/pyt.py 1800` sequentially (`run_all_nbs.py`, `run_examples.sh`).

| notebook | exit | wall |
|---|---|---|
| data_science/bayesian_model_comparison.ipynb | 0 | 11s |
| data_science/causal_inference_from_observational_data.ipynb | 0 | 21s |
| data_science/bayesian_inverse_problems.ipynb | 0 | 23s |
| data_science/cox_proportional_hazards.ipynb | 0 | 13s |
| data_science/constrained_inference.ipynb | 0 | 39s |
| data_science/hierarchical_partial_pooling.ipynb | 0 | 48s |
| data_science/newton_continuation_and_folds.ipynb | 0 | 74s |
| data_science/ppl_end_to_end_case_study.ipynb | 0 | 70s |
| data_science/mcmc_uncertainty_quantification.ipynb | 0 | 169s |
| data_science/quantile_regression.ipynb | 0 | 122s |
| data_science/regression_and_glms.ipynb | 0 | 119s |
| data_science/regularization_and_sparsity.ipynb | 0 | 169s |
| data_science/variational_inference_vs_mcmc.ipynb | 0 | 201s |
| tutorials/mcmc_sampling.ipynb | 0 | 246s |
| tutorials/probabilistic_programming.ipynb | 0 | 472s |
| data_science/constrained_and_multiobjective_optimization.ipynb | 0 | 1922s |

| example | exit | wall |
|---|---|---|
| ppl_example | 0 | 41.5s |
| flagship_physics_inverse | 0 | 545.3s |
| geoscience_inversion_report | 0 | 185.8s |
| project_neural_to_structured | 0 | 979.1s |
| symbolic_export_example | 0 | 36.1s |

**Fresh output vs stored output vs prose** (`probes/q0_nb_compare.py`, out/q/q0_nb_compare.txt; cell-by-cell text comparison, timings/addresses normalized): no fresh output contradicts a prose claim -- **none**. Stored outputs that the wheel no longer reproduces, all consistent with the prose: `mcmc_uncertainty_quantification` cell 5 `ESS(mu)=4764 / 1500 draws` -> `1500 / 1500` and `tutorials/probabilistic_programming` cell 3 `HMC ESS 3000` -> `1000` (the P08-F08 clip); `probabilistic_programming` cell 8 `3*N(0,1)+1 mean=1.02 var=8.95` -> `1.00 / 9.00` (P08-F14), cell 12 nuts acceptance 0.89 -> 0.95 and VI/ensemble point estimates within 0.07 of the stored ones, cell 1 unseeded `sample(3)` values; `variational_inference_vs_mcmc` VI sds 0.324/0.358 -> 0.301/0.302 against MCMC 0.358 (the prose claims mean-field is narrower and that mean-field ~ full-rank here; the fresh numbers support both better than the stored ones) and `~4x faster` -> `~3x` (timing). Every other differing cell is a stdout stream split into chunks with identical text. No fresh error cells. The stored corpus (d99d296) therefore still carries pre-0.8.2 numbers in two notebooks, but no prose quotes them.

## Findings

### Q04-F01 -- blocking -- A conjugate posterior restored from a pickle still answers predict() with the plug-in predictive; the R02-F04 refusal is wired only into Posterior, not ConjugatePosterior

- **Surface:** mixle.ppl.RandomVariable.predict on a fit whose .result is a mixle.ppl.inference.ConjugatePosterior (how='auto'/'conjugate'/'posterior' on Normal-Normal, Poisson-Gamma, Exponential-Gamma, Bernoulli-Beta) after pickle.loads(pickle.dumps(fit))
- **Reproduction:**

```python
import pickle, numpy as np
from mixle.ppl import Normal
RS = np.random.RandomState
rows = list(RS(0).normal(5, 2, 3))
mu = Normal(0, 10, name='mu')
f = Normal(mu, 2.0).fit(rows)                      # default route -> conjugate
live = np.asarray(f.predict(20000, rng=RS(1))).std()
g = pickle.loads(pickle.dumps(f))
rest = np.asarray(g.predict(20000, rng=RS(1))).std()   # no exception
post_sd = 1 / np.sqrt(3 / 4 + 1 / 100)
print(f.explain_fit()['route'], type(g.result).__name__, getattr(g.result, '_dropped_closures', None))
print('integrated predictive sd=%.4f plug-in sd=2.0 | live=%.4f restored=%.4f' % (np.sqrt(4 + post_sd**2), live, rest))
print(hasattr(g.result, '_refuse_dropped_closure'))
```

- **Observed:** venv-full (out/p2_followups.082full.txt block B; out/q/q2_pickle_conjugate.082full.txt): route=conjugate, result type ConjugatePosterior, _dropped_closures=('predictive',). Live predictive sd 2.3003 (integrated answer 2.3056); restored predictive sd 2.0280 (the plug-in sqrt(4)=2.0), returned silently with no warning. Same on every conjugate pair (out/p3_conjugate_mix_glm.082full.txt block A, 20000 draws, same rng): Poisson-Gamma live sd 2.0714 -> restored 1.8742; Exponential-Gamma live mean/sd 1.0144/1.3438 -> restored 0.8037/0.8022; Bernoulli-Beta 0.601 -> 0.603. hasattr(g.result, '_refuse_dropped_closure') is False. The Posterior-backed routes (mcmc/hmc/nuts/ensemble/laplace/vi) DO refuse ("this posterior was restored from a pickle, so predict() cannot integrate over its draws...", out/p1_ledger_replay.082full.txt block P04-F05). 0.8.1: the conjugate fit cannot be pickled at all (AttributeError on a local lambda), so the silent plug-in answer is new to 0.8.2.
- **Expected:** CHANGELOG R02-F04: "predict() on a posterior restored from a pickle refuses instead of answering with the plug-in predictive ... It now refuses by name". The conjugate route is the DEFAULT route for every conjugate model and is listed first in the P04-F05 pickling repair, so a restored conjugate fit must refuse predict() the way the sampler routes do (or integrate over its exact posterior, which it still holds: g.result.samples('mu') works after restore).
- **Notes:** Mechanism (source/mixle/ppl/core.py RandomVariable.predict): after `pred is None` it refuses only `if r is not None and hasattr(r, "_refuse_dropped_closure")`, then falls through to `self.sample(n, seed)` = the posterior-MEAN model. `Posterior` (inference.py:179) defines `_refuse_dropped_closure`; `ConjugatePosterior` (inference.py:3623) copies the `_CLOSURE_FIELDS`/`__getstate__`/`__setstate__` machinery and records `_dropped_closures=('predictive',)` but never defines the refusal method, so the guard is skipped. The regression test `RestoredPredictiveTest` (tests/adversarial_review_082_repairs_test.py:116) only iterates how in ("mcmc", "laplace"), so the conjugate gap is untested. Cross-process restore (pickle to file, load in a fresh interpreter) gives the same 2.028 (out/q/q2_pickle_conjugate.082full.txt). Severity: a wrong result (a predictive distribution narrower than the posterior predictive, with the parameter uncertainty silently dropped) on the default route of a documented surface, and a repair the CHANGELOG claims closed.
- **Repairs concerned:** R02-F04, P04-F05

### Q04-F02 -- real -- predict(<covariates>) on a fitted Poisson or Bernoulli GLM returns the mean response, not "one draw per row": non-integer counts, probabilities instead of 0/1, and rng is ignored, while a Normal regression does draw

- **Surface:** mixle.ppl.RandomVariable.predict(mapping|DataFrame) / mixle.ppl.regression.predict_at on fits of Poisson(free*Field('x')+free) and Bernoulli(free*Field('x')+free)
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Poisson, Bernoulli, Normal, Field, free
RS = np.random.RandomState
r = RS(1); n = 400; x = r.normal(size=n)
yp = list(r.poisson(np.exp(0.5 + 0.8 * x)).astype(float))
fp = Poisson(free * Field('x') + free).fit(yp, given={'x': list(x)})
print(fp.predict({'x': [0.0, 1.0, 2.0]}, rng=RS(0)), fp.predict({'x': [0.0, 1.0, 2.0]}, rng=RS(1)), fp.result.predict({'x': [0.0, 1.0, 2.0]}))
yb = list((r.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 1.5 * x)))).astype(float))
fb = Bernoulli(free * Field('x') + free).fit(yb, given={'x': list(x)})
print(fb.predict({'x': [-2.0, 0.0, 2.0]}, rng=RS(0)))
yn = list(2 * x + 0.5 + r.normal(scale=0.3, size=n))
fn = Normal(free * Field('x') + free, free).fit(yn, given={'x': list(x)})
print(fn.predict({'x': [0.0, 1.0, 2.0]}, rng=RS(0)), fn.predict({'x': [0.0, 1.0, 2.0]}, rng=RS(1)))
print([l.strip() for l in fp.predict.__doc__.splitlines() if 'draw per row' in l])
```

- **Observed:** venv-full (out/p3_conjugate_mix_glm.082full.txt block C; out/p3b_mix_weights.082full.txt; out/q/q7_misc.082full.txt): Poisson GLM predict at x=[0,1,2] -> [1.6455 3.7508 8.5495] for rng=RS(0) AND for rng=RS(1), identical to fp.result.predict (the fitted mean exp(0.498+0.824x)); integer-valued: False. Bernoulli GLM -> [0.0434 0.5777 0.9763] for both rngs (probabilities, not 0/1). The Normal regression DOES draw (rng=RS(0) -> 2.9666 at x=1,z=0 where the mean is 2.4612; rng=3 -> 2.9736). The docstring of the same method says "one draw per row comes back".
- **Expected:** Either one predictive draw per row for every family (Poisson counts, Bernoulli 0/1, i.e. sample the fitted conditional law), as the docstring and the P08-F15 repair text ("predict() takes the covariates it is predicting at") promise, or a documented mean-response contract with the rng argument refused/ignored explicitly and the same behaviour on every family.
- **Notes:** Mechanism: source/mixle/ppl/regression.py predict_at: `if result.link != "identity" or not np.isfinite(scale) or scale <= 0.0: return mean # a non-Gaussian family's mean response; there is no additive noise scale` -- the draw is only added for the identity link. Both RandomVariable.predict and predict_at carry the docstring "One predictive draw per row". New surface in 0.8.2 (0.8.1 refused covariates: "predictive sample count must be an exact positive integer"), so this is the P08-F15 repair shipping with a contract its own docstring contradicts on two of the three GLM families the regression notebook uses.
- **Repairs concerned:** P08-F15

### Q04-F03 -- real -- Mix(..., weights=free) crashes on its default route with an internal TypeError ('_Free' is not a real number) after explain_fit() promised route 'em'; weights=Dirichlet(...) crashes with numpy's 'setting an array element with a sequence'

- **Surface:** mixle.ppl.Mix(components, weights=free | Dirichlet(...)).fit(data) with how='auto' (default) or how='em'; mixle/ppl/_lowering.py _mix_est
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Mix, Normal, Dirichlet, free
RS = np.random.RandomState
dm = list(RS(1).normal(-3, 1, 200)) + list(RS(2).normal(3, 1, 200))
m = Mix([Normal(free, free), Normal(free, free)], weights=free)
print(m.explain_fit())                     # route 'em'
try: m.fit(dm)
except Exception as e: print(type(e).__name__, e)
try: Mix([Normal(free, free), Normal(free, free)], weights=Dirichlet([1.0, 1.0], name='w')).fit(dm)
except Exception as e: print(type(e).__name__, e)
print(Mix([Normal(free, free), Normal(free, free)], weights=free).fit(dm, how='map').params['weights'])   # works
```

- **Observed:** venv-full and 0.8.1 (out/p3b_mix_weights.082full.txt, out/p3b_mix_weights.081.txt, out/p3_conjugate_mix_glm.082full.txt block B): explain_fit() -> {'route': 'em', 'reason': 'all-free parameters, no priors -> maximum-likelihood EM'}; .fit(dm) and .fit(dm, how='em') -> `TypeError: float() argument must be a string or a real number, not '_Free'` raised from _lowering.py:607 `np.asarray(weights, float)` in _mix_est. weights=Dirichlet([1,1]) on the default route -> `ValueError: setting an array element with a sequence.` how='map'/'vi'/'laplace'/'mcmc' accept weights=free and fit (weights [0.5 0.5] on map).
- **Expected:** Either the constructor refuses `weights=free`/`weights=Dirichlet(...)` by name, or the EM route estimates the weights (free weights are the ordinary mixture-EM case) and explain_fit() names a route that actually runs. The library's own error text elsewhere (vmp.py:179 "supports only Mix([Normal(free, free), ...], weights=None|free)") treats weights=free as a valid spelling.
- **Notes:** Pre-existing in 0.8.1 (identical traceback), not in the 0.8.2 ledger (no P/R id mentions Mix weights). _mixture_weights() in the constructor accepts the handle; _mix_est() then coerces it with np.asarray(..., float). The Mix docstring documents only the weightless spelling, so the contract for a free weight vector is undocumented and the default route is the one that fails.
- **Repairs concerned:** none

### Q04-F04 -- docs -- The migration guide lists how='map' among the routes refused by the 'no prior and no data pin them' guard; map fits Normal(free, free) on one observation

- **Surface:** docs/migrations/0.8.2.md 'Probabilistic programs and surrogates' bullet on the sampler/Laplace refusal; mixle.ppl.inference.map_fit vs _refuse_fewer_observations_than_parameters
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free
for how in ('map', 'laplace', 'mcmc'):
    try:
        f = Normal(free, free).fit([5.0], how=how, rng=np.random.RandomState(0), **({'draws': 20, 'burn': 5} if how == 'mcmc' else {}))
        print(how, 'FITTED', f.params)
    except ValueError as e:
        print(how, 'refused:', str(e)[:90])
```

- **Observed:** venv-full (out/p1_ledger_replay.082full.txt block P10-F13; out/q/q7_misc.082full.txt): how='map' -> FITTED {'mean': 8.528, 'sd': 0.001} (the same on how='auto'); how='laplace', 'mcmc', 'nuts' -> refused "RandomVariable has 2 parameter(s) with no prior (arg0, arg1) but was given 1 observation(s)". The migration guide sentence: "A **sampler or Laplace** fit (`how='mcmc'|'nuts'|'hmc'|'ensemble'|'map'|'laplace'`) is refused when the model has parameters that NO prior and no data pin".
- **Expected:** The guide names exactly the routes the guard covers. Either drop 'map' from the list (and say map returns a degenerate point, sd at its floor 0.001, for such a model) or wire the guard into the non-grouped map path.
- **Notes:** Mechanism: _refuse_fewer_observations_than_parameters is called only from _prepare_target (inference.py:2614); map_fit calls _prepare_target only in its `_is_grouped(rv)` branch (inference.py:3104), so a scalar/vector MAP never reaches the guard. The guard's own docstring already says "Both were rejected here while `map` fitted the same models on the same data". Same behaviour on 0.8.1 (map fitted). Docs only: the wheel's behaviour is the documented "degenerate but determined" point estimate; the guide's route list is what is wrong.
- **Repairs concerned:** R02-F03, P10-F13

### Q04-F05 -- minor -- A fitted regression/GLM has no working log-likelihood surface, and the NotImplementedError that refuses it says the value 'is available from log_likelihood(data)' -- the very call that raised it

- **Surface:** mixle.ppl.RandomVariable.log_likelihood / aic / bic / plugin_log_likelihood / log_density on a fitted regression; RegressionResult.loglik
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Poisson, Field, free
RS = np.random.RandomState
r = RS(1); x = r.normal(size=400); yp = list(r.poisson(np.exp(0.5 + 0.8 * x)).astype(float))
fp = Poisson(free * Field('x') + free).fit(yp, given={'x': list(x)})
for call in (lambda: fp.log_likelihood(yp), lambda: fp.aic(yp), lambda: fp.plugin_log_likelihood(yp),
             lambda: fp.log_likelihood(yp, given={'x': list(x)}), lambda: fp.log_density(yp[:3], given={'x': list(x[:3])})):
    try: print(call())
    except Exception as e: print(type(e).__name__, str(e)[:230])
print(fp.result.loglik)
```

- **Observed:** venv-full (out/p3_conjugate_mix_glm.082full.txt block C, out/p3b_mix_weights.082full.txt, out/p2_followups.082full.txt block A, out/q/q7_misc.082full.txt): log_likelihood(y), aic(y), bic, plugin_log_likelihood(y) all raise `NotImplementedError: this fitted model is a conditional (regression / mixed-effects) fit ... its total marginal log-likelihood is available from `log_likelihood(data)` / `aic(data)` / `bic(data)`.`; log_likelihood(y, given=...) and log_density(y, given=...) raise `TypeError: ... got an unexpected keyword argument 'given'`; result.loglik is None. Same on 0.8.1.
- **Expected:** A message that names a call that works (a given=-aware log_likelihood/aic, or the result's own log-likelihood), and a stored log-likelihood on the RegressionResult; at minimum the message must not point back at itself.
- **Notes:** Mechanism: core.py lower() raises the message for a bound RV with _dist None (D-0190); log_likelihood/aic/plugin_log_likelihood all lower to a dist first, so every listed alternative hits the same raise. P08-F15 ("log_density(given=) unsupported") is ledgered as the fourth item of that finding and the CHANGELOG lists the other three items as repaired without mentioning it, so this is the unrepaired remainder plus the self-referential message.
- **Repairs concerned:** P08-F15

### Q04-F06 -- minor -- Categorical with a Dirichlet prior accepts only numeric labels on the conjugate route while Categorical(free) accepts strings; how='em' and how='map' die with internal messages

- **Surface:** mixle.ppl.Categorical(Dirichlet([...], name=...)).fit(labels, how='auto'|'conjugate'|'em'|'map')
- **Reproduction:**

```python
from mixle.ppl import Categorical, Dirichlet, free
for rows in (['a', 'b', 'a'], [0, 1, 0]):
    for how in ('auto', 'em', 'map'):
        try: print(rows, how, Categorical(Dirichlet([1.0, 1.0, 1.0], name='pi')).fit(rows, how=how).summary())
        except Exception as e: print(rows, how, type(e).__name__, str(e)[:110])
print(Categorical(free).fit(['a', 'b', 'a']).params)
```

- **Observed:** venv-full and 0.8.1 (out/p3b_mix_weights.082full.txt, out/p3b_mix_weights.081.txt): strings on auto/conjugate -> `TypeError: conjugate observations must be a numeric one-dimensional sequence`; how='em' (any labels) -> `NotImplementedError: latent/random parameters (a distribution in a slot) land in build slice 5.`; how='map' with int labels -> `TypeError: 'float' object is not iterable`, with strings -> `ValueError: could not convert string to float: 'a'`. Categorical(free).fit(['a','b','a']) -> CategoricalDistribution({'a': 0.667, 'b': 0.333}). Numeric labels on the conjugate route work (hyper alpha [3,2,1]).
- **Expected:** The same label vocabulary on both spellings of the model (a label -> index map for the Dirichlet-Categorical update), and refusals on em/map that name the route/prior combination rather than a build slice or a float iteration.
- **Notes:** Not ledgered. The conjugate path validates observations as numeric before mapping them onto the Dirichlet's support; the em/map lowerings do not special-case a Dirichlet in the pmf slot. Additional label edges (out-of-range, 1.5, -1) are in out/q/q7_misc.082full.txt.
- **Repairs concerned:** none

### Q04-F07 -- minor -- RandomVariable.fit on a one-shot iterator dies with `TypeError: float() argument ... not 'list_iterator'` on map/mcmc/laplace/vi/auto/nuts, while how='em' materializes it and 'conjugate' refuses by name

- **Surface:** mixle.ppl.RandomVariable.fit(iter(data), how=...) on a model with a prior
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free
RS = np.random.RandomState; dd = list(RS(0).normal(5, 2, 50))
mu = Normal(0, 10, name='mu')
for how, kw in (('mcmc', dict(draws=20, burn=5)), ('map', {}), ('laplace', {}), ('vi', {}), ('auto', {}), ('nuts', dict(draws=20, burn=5)), ('em', {}), ('conjugate', {})):
    model = Normal(mu, 2.0) if how == 'conjugate' else (Normal(free, free) if how == 'em' else Normal(mu, free))
    try: print(how, 'OK', model.fit(iter(dd), how=how, rng=RS(0), **kw).params)
    except Exception as e: print(how, type(e).__name__, str(e)[:100])
```

- **Observed:** venv-full (out/p2_followups.082full.txt block G): mcmc/map/laplace/vi/auto/nuts -> `TypeError: float() argument must be a string or a real number, not 'list_iterator'`; em -> fitted {'mean': 5.281, 'sd': 2.251}; conjugate -> `TypeError: conjugate observations must be a numeric one-dimensional sequence`. Tuple and pandas.Series spellings are in out/q/q7_misc.082full.txt.
- **Expected:** Either materialize the iterator once (as em, optimize()/fit()/propose() and R02-F10's front door do) or refuse it by name on every route; not numpy's coercion error. No route returns a wrong fit, so this is a validation gap only.
- **Notes:** Related to P03-F02 / P06-F01 / R02-F10 (one-shot iterators through optimize()/fit()/propose()/Model.fit), which do not cover the PPL front door.
- **Repairs concerned:** R02-F10

### Q04-F08 -- minor -- One flat scale parameter with a proper prior on the mean and ONE observation passes the 'nothing pins them' guard (equality allowed) and the routes return an improper posterior as numbers that disagree by eight orders of magnitude

- **Surface:** Normal(Normal(0,10,name='mu'), free).fit([5.0], how='mcmc'|'hmc'|'vi'|'map'|'laplace'|'nuts'|'ensemble')
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free
RS = np.random.RandomState
for how, kw in (('mcmc', dict(draws=200, burn=50)), ('hmc', dict(draws=200, burn=50)), ('vi', {}), ('map', {}), ('laplace', {}), ('nuts', dict(draws=200, burn=50)), ('ensemble', dict(draws=200, burn=50))):
    try:
        f = Normal(Normal(0, 10, name='mu'), free).fit([5.0], how=how, rng=RS(0), **kw); print(how, f.params)
    except Exception as e: print(how, type(e).__name__, str(e)[:120])
```

- **Observed:** venv-full (out/p2_followups.082full.txt block H): mcmc -> {'mean': -7.397, 'sd': 57483.432} (summary std of sd 179667); hmc -> sd 1.97e11; vi -> sd 4551; map -> sd 0.001; laplace -> RuntimeError "Laplace Hessian is not positive definite (rank=1/2, min_eigenvalue=0)"; nuts -> FloatingPointError "cannot differentiate a non-finite autograd log target."; ensemble -> RuntimeError "inference model construction or likelihood evaluation failed at unconstrained coordinates [-30.87, 566.65]". No warning on the three routes that return numbers. The all-flat sibling Normal(free, free) on one observation, which the migration guide says how='vi' "still fits": vi -> {'mean': 198.3, 'sd': 67226.4} silently (out/q/q7_misc.082full.txt); on venv-base the same call dies with an internal "failed at unconstrained coordinates [4.5e166, 388.07]" (out/q/p1_ledger_replay.082base.txt).
- **Expected:** One flat scale and one observation give an improper posterior (the likelihood is unbounded as sd -> 0 and the flat prior has infinite mass as sd -> inf); the guard's "equality is allowed" rule treats it as "degenerate but determined". Either count a flat scale parameter as needing two observations, or make the samplers' non-finite/blow-up exits name the cause (nuts and ensemble currently raise internal messages), so an ordinary user is not handed sd=2e11.
- **Notes:** This is the boundary of the P10-F13 / R02-F03 guard (documented in _refuse_fewer_observations_than_parameters' docstring as allowed), not a regression: 0.8.1 gave the same numbers on mcmc/vi/map. Rated minor because the model is unusual; the two internal messages (nuts, ensemble) are the actionable part.
- **Repairs concerned:** R02-F03, P10-F13

### Q04-F09 -- minor -- The same bad input gets a different refusal on different routes: rng=True vs rng='x' use two message templates, and an (n,1) column gets the numpy.ravel advice on em/auto but a bare 'autograd observations' message on map/mcmc/laplace

- **Surface:** RandomVariable.fit(rng=...) validation; RandomVariable.fit((n,1) array, how=...)
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free
mu = Normal(0, 10, name='mu'); dd = list(np.random.RandomState(0).normal(5, 2, 50))
for rng in ('x', True):
    try: Normal(mu, free).fit(dd, how='map', rng=rng)
    except TypeError as e: print(repr(rng), e)
col = np.asarray(dd).reshape(-1, 1)
for how in ('em', 'map'):
    try: Normal(free, free).fit(col, how=how)
    except ValueError as e: print(how, str(e)[:120])
```

- **Observed:** venv-full (out/p1_ledger_replay.082full.txt block P04-F11; out/p3_conjugate_mix_glm.082full.txt block E): rng='x' -> "rng must be a numpy.random.RandomState, a numpy.random.Generator, or an integer seed, got str"; rng=True -> "rng must be a numpy RandomState, a Generator, or an integer seed, got a bool". (n,1) on em/auto -> "GaussianDataEncoder expects a one-dimensional sequence of scalar observations, but the data has shape (50, 1). Flatten it with numpy.ravel(data) ..."; on map/mcmc/laplace -> "autograd observations must be a non-empty one-dimensional sequence."
- **Expected:** One template per refusal, and the same shape advice on every route.
- **Notes:** Cosmetic; recorded so the next pass does not re-probe it. Both refusals are correct in substance. Same family (out/q/q4_auto_reason_free.082full.txt): a vector handle in a scalar slot, `Normal(free(5, name='v'), 1.0).fit(dd)`, dies with numpy's `TypeError: only 0-dimensional arrays can be converted to Python scalars` while the DiagGaussian(3)/free(5) mismatch is refused by name ("mean parameter handle must declare a length-3 vector").
- **Repairs concerned:** P04-F11, R02-F12

### Q04-F10 -- minor -- An infeasible starting point behind a -inf potential or a hard constraint, a huge proposal scale, and an unconstrained MAP that hits max_its are refused with raw internal messages that name neither the potential/constraint nor the argument to change

- **Surface:** RandomVariable.fit(how='mcmc'|'hmc'|'nuts'|'ensemble', potentials=[potential(<returns -inf>)] | constraints=[...] | scale=1e9); how='map' with max_its=1; how='nuts' with a hard constraint on the base (no-torch) install
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free, potential
RS = np.random.RandomState; dd = list(RS(0).normal(5, 2, 400))
wall = lambda m: 0.0 if m > 7 else float('-inf')          # the MAP start (~4.93) is infeasible
for how in ('mcmc', 'hmc', 'nuts', 'ensemble'):
    for chains in (1, 2):
        mu = Normal(0, 10, name='mu')
        try: Normal(mu, free).fit(dd, how=how, potentials=[potential(wall, mu)], draws=200, burn=50, chains=chains, rng=RS(0))
        except Exception as e: print(how, chains, type(e).__name__, e)
mu = Normal(0, 10, name='mu')
try: Normal(mu, free).fit(dd, how='nuts', constraints=[mu > 50], draws=200, burn=50, chains=2, rng=RS(0))
except Exception as e: print('nuts chains=2 mu>50', type(e).__name__, e)
try: Normal(mu, free).fit(dd, how='mcmc', draws=200, burn=50, scale=1e9, rng=RS(0))
except Exception as e: print('mcmc scale=1e9', type(e).__name__, e)
try: Normal(mu, free).fit(dd, how='map', max_its=1)
except Exception as e: print('map max_its=1', type(e).__name__, e)
```

- **Observed:** venv-full and venv-nonumba, identical (out/q/q1_zero_acceptance.082full.txt, .082nonumba.txt): the same infeasible start is refused as `ValueError: initial state has non-finite log target: -inf.` (mcmc, hmc), `ValueError: initial state has non-finite log target.` (nuts chains=1), `ValueError: some initial walkers have non-finite log target.` (ensemble; also for the wall INSIDE the posterior mass, mu<5.05, which mcmc/hmc/nuts sample fine), `ValueError: gradient contains non-finite values at a finite target state.` (nuts chains=2, wall mu<5.05) and `FloatingPointError: cannot differentiate a non-finite autograd log target.` (nuts chains=2, hard mu>50). mcmc scale=1e9 -> `FloatingPointError: autograd log target produced NaN or positive infinity.` Unconstrained map max_its=1 -> `RuntimeError: MAP optimization failed: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` (the constrained path says "penalty ramp exhausted max_iter without converging. Raise max_iter (currently 1)", naming `max_iter` where the fit argument is `max_its`; both spellings are accepted, out/q/q8_followups.082full.txt). venv-base (out/q/p1_ledger_replay.082base.txt): nuts with the active hard constraint mu>6, which samples on venv-full (acceptance 0.87), ends in `ValueError: gradient contains non-finite values at a finite target state.` on the numeric-gradient path. A narrow but non-empty feasible band (out/q/q9_ensemble_hierarchical.082full.txt): constraints=[mu > 6, mu < 6.001] is refused on mcmc and map with "could not find a parameter point satisfying the constraints; check that the region is non-empty and consistent with the supports" (the region is not empty; mcmc samples [6, 6.01] and map refuses it too; both accept [6, 6.1]) -- the feasibility projection's failure is blamed on the user's region. All of these are identical on 0.8.1 (out/q/q1_zero_acceptance.081.txt).
- **Expected:** One refusal per cause, naming what the user passed (the potential/constraint, `scale=`, `max_its=`) and the way out (`penalty=`, a feasible start, `how='mcmc'`), the way the new zero-acceptance refusal does; not a message about autograd internals or an L-BFGS status string.
- **Notes:** Same defect class as R02-F17 (raw non-finite-gradient message under a hard constraint, ledgered minor, not repaired) extended to potentials, to the proposal scale and to the base install. Not a regression; the one-point refusal itself now fires on every route (see 'Attacks that did not break').
- **Repairs concerned:** R02-F02, R02-F17, P04-F07

### Q04-F11 -- blocking -- how='ensemble' at its default budget returns a wrong posterior on ordinary 5- and 18-parameter models -- means 4-8 posterior sds off, sds 3-9x too wide -- with no warning; a single chain reports mcse 0.001-0.004 for errors of 0.13, while mcmc/nuts/hmc defaults are right on the same models

- **Surface:** RandomVariable.fit(how='ensemble') with the default draws=1500, burn=500, walkers=2*(d+1), chains=1 -- mixle.ppl.inference.ensemble_fit / _ensemble_p0; DiagGaussian(5, mean=free(5)) and Bernoulli(Beta(a,b).each()) on 18 groups
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import DiagGaussian, free, Bernoulli, Beta
RS = np.random.RandomState
r = RS(0); X = np.stack([r.normal(l, 0.5, 300) for l in [0, 2, 1, 3, 4]], axis=1)   # posterior mean = column mean, sd = 0.5/sqrt(300) = 0.0289
for how in ('ensemble', 'mcmc', 'nuts', 'hmc'):
    v = free(5, name='v'); g = DiagGaussian(5, mean=v, var=np.full(5, .25)).fit(X.tolist(), how=how, rng=RS(0)); s = g.summary()
    means = np.array([s['v%d' % i]['mean'] for i in range(5)]); sds = np.array([s['v%d' % i]['std'] for i in range(5)])
    print(how, 'max|mean-exact|/sd=%.2f' % (np.abs(means - X.mean(0)) / 0.02887).max(), 'sd ratio', (sds / 0.02887).round(2), 'ess_bulk', s['v0']['ess_bulk'])
hs = [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]
games = [[1.0] * h + [0.0] * (45 - h) for h in hs]; a, b = 164.0, 453.9
ex = np.array([(a + h) / (a + b + 45) for h in hs]); ex_sd = np.sqrt(ex * (1 - ex) / (a + b + 46))
f = Bernoulli(Beta(a, b).each()).fit(games, how='ensemble', rng=RS(0)); s = f.summary()
err = np.abs(np.array([s['p[%d]' % i]['mean'] for i in range(18)]) - ex) / ex_sd
print('worst group %d: %.2f posterior sds off, mcse %.4f; groups >3 sd off: %d/18' % (err.argmax(), err.max(), s['p[%d]' % err.argmax()]['mcse'], (err > 3).sum()))
```

- **Observed:** venv-full (out/q/q9_ensemble_hierarchical.082full.txt, out/q/q8_followups.082full.txt, out/q/q11_ensemble_seeds_hier.*): DiagGaussian(5) at the default budget -- ensemble: max|mean-exact| = 3.73 posterior sds, sd ratio 3.19-9.14, acceptance 0.526, ess_bulk(v0) = 32; mcmc 0.15 sd (sd ratio 0.92-1.09), nuts 0.02 (0.96-1.01), hmc 0.05 (0.99-1.00). Grouped Bernoulli-Beta, 18 groups, default budget (38 walkers, 57,000 pooled draws): worst group 13 is 7.78 posterior sds off (mean 0.3953, exact 0.2625) with mcse 0.0040; 6/18 groups more than 3 sds off; ess_bulk(p0) = 278; chains=4: 9/18 groups off (q8: chains=2 reports r_hat 3.29 / split_r_hat 2.13, so the multi-chain diagnostic sees it; the single-chain default has none). draws=300/burn=100: 0.4083 vs 0.2625 (q6). walkers=100 at the default draws: worst 0.45 sd; draws=20000/burn=5000: 0.05 sd -- the sampler is correct, its default budget is not. The ensemble docstring: "the pooled posterior has draws*walkers near-independent samples" -- 57,000 pooled draws with ess_bulk 278. Seeds, 0.8.1 and venv-nonumba: out/q/q11_ensemble_seeds_hier.{082full,081,082nonumba}.txt (see notes).
- **Expected:** Either a default budget (walkers/burn) under which the ensemble converges on models of this size, or a refusal/warning when the ensemble has not contracted (the walkers' own spread is a within-run R-hat the sampler can compute; chains>=2 already reports r_hat 3.3), and a docstring that does not call the pooled draws near-independent. A posterior whose single-chain summary claims mcse 0.004 for an error of 0.13 is a wrong result on a documented route at default settings.
- **Notes:** Mechanism (inference.py ensemble_fit / _ensemble_p0): walkers default to max(2*(d+1), 8) -- 12 for d=5, 38 for d=18; the initial cloud is the data-informed point plus 0.1*_init_scale*sqrt(n) jitter for the first half of the walkers and INDEPENDENT PRIOR DRAWS for the second half (the STAT-RR22-12 overdispersed start, introduced before 0.8.2), and 500 burn sweeps of the stretch move at acceptance 0.3-0.5 do not contract it in 5 or 18 dimensions; the pooled draws are then reported through a single-chain summary whose ESS/mcse cannot see that the walkers disagree. The grouped Bernoulli-Beta model became reachable on this route only in 0.8.2 (P07-F04), so on that surface it is new; the DiagGaussian case is byte-identical on 0.8.1 and on venv-nonumba (out/q/q11_ensemble_seeds_hier.{081,082nonumba}.txt), so the defect itself is pre-existing. Seeds (q11, d=5, default budget): rng=RS(0)/RS(1)/RS(2) -> 3.73 / 3.60 / 1.82 posterior sds off with sd ratios up to 9.1 / 8.3 / 3.3 -- every seed is wrong; chains=2 reports r_hat 1.12 (above the 1.01 threshold the CHANGELOG's own status ladder uses) so a multi-chain run would be flagged; walkers=100, burn=5000 or draws=10000 each bring it within 0.1 sd. Not ledgered: R07-F01 (the ensemble collapsing to one point behind a constraint) is the opposite failure and is now refused.
- **Repairs concerned:** P07-F04, R02-F02

### Q04-F12 -- minor -- The default route of a grouped model (how='auto' -> hierarchical) crashes on rng= with `hierarchical_fit() got an unexpected keyword argument 'rng'`, and its posterior draws cannot be seeded at all

- **Surface:** RandomVariable.fit(groups, rng=...) on Bernoulli(Beta(a, b).each()) / any .each() model routed to mixle.ppl.inference.hierarchical_fit
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Bernoulli, Beta
games = [[1.0] * h + [0.0] * (45 - h) for h in [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]]
m = Bernoulli(Beta(164.0, 453.9).each())
print(m.explain_fit()['route'])
for kw in (dict(rng=np.random.RandomState(0)), dict(rng=0), dict(seed=0), {}):
    try: print(kw, 'fitted', type(m.fit(games, **kw).result).__name__)
    except Exception as e: print(kw, type(e).__name__, e)
```

- **Observed:** venv-full and 0.8.1 (out/q/q6_grouped_bernoulli.082full.txt, .081.txt; out/q/q8_followups.082full.txt): explain_fit route 'hierarchical'; fit(games, rng=RandomState(0)) -> `TypeError: hierarchical_fit() got an unexpected keyword argument 'rng'`; rng=0 and seed=0 likewise; fit(games) works. Every other route (mcmc/nuts/hmc/ensemble/laplace/map) takes rng= on the same model.
- **Expected:** rng= accepted (the route returns "group posterior draws" that a caller may want reproducible) or refused by name ("how='hierarchical' does not take rng"), not a Python signature error from an internal function.
- **Notes:** Mechanism: inference.py:4591 `def hierarchical_fit(rv, data, *, max_its=300, tol=1e-8)`; core.py fit() forwards **kw to the fitter unchanged. The R02-F12 repair made rng spellings uniform across laplace/vi/map/samplers but the hierarchical route was not in its list. Pre-existing in 0.8.1.
- **Repairs concerned:** R02-F12, P04-F11

### Q04-F13 -- minor -- pointwise_log_likelihood() on a conjugate (exact Bayesian) fit refuses with 'unavailable for this point-estimate artifact', live and restored

- **Surface:** RandomVariable.pointwise_log_likelihood on a fit whose .result is a ConjugatePosterior
- **Reproduction:**

```python
import numpy as np, pickle
from mixle.ppl import Normal
rows = list(np.random.RandomState(0).normal(5, 2, 3))
f = Normal(Normal(0, 10, name='mu'), 2.0).fit(rows)          # conjugate
for obj in (f, pickle.loads(pickle.dumps(f))):
    try: print(obj.pointwise_log_likelihood(rows))
    except Exception as e: print(type(e).__name__, e)
print(f.log_likelihood(rows))
```

- **Observed:** venv-full (out/p3_conjugate_mix_glm.082full.txt block A; out/q/q2_pickle_conjugate.082full.txt): live and restored -> `NotImplementedError: Bayesian pointwise log likelihood is unavailable for this point-estimate artifact; use plugin_log_likelihood(data), AIC, or BIC instead.` log_likelihood(rows) -> -3.8506 works.
- **Expected:** A per-observation log predictive density integrated over the exact posterior (the conjugate route holds it in closed form), or a refusal that says the conjugate route has not implemented it -- not a message calling an exact posterior a point-estimate artifact and steering to AIC/BIC.
- **Notes:** Message only; the mcmc/laplace routes answer pointwise_log_likelihood from their draws. Not ledgered.
- **Repairs concerned:** P04-F05

### Q04-F14 -- real -- The default route of a grouped model (how='auto' -> hierarchical) returns an iteration-capped, unconverged empirical-Bayes fit with no warning, and fit(max_its=..., delta=...) are silently swallowed so the cap cannot be raised through .fit()

- **Surface:** RandomVariable.fit on a .each() model routed to mixle.ppl.inference.hierarchical_fit (Beta-Bernoulli, Gamma-Poisson, Normal-Normal); RandomVariable.fit's max_its/delta forwarding
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Bernoulli, Beta
hs = [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]
games = [[1.0] * h + [0.0] * (45 - h) for h in hs]
for kw in ({}, dict(max_its=30000), dict(max_its=3000, delta=1e-12)):
    hy = Bernoulli(Beta(164.0, 453.9).each()).fit(games, **kw).summary()['hyper']       # no warning is emitted
    print(kw, hy['iterations'], hy['converged'], round(hy['a'], 6), round(hy['b'], 6))
from mixle.ppl import inference as _inf
hy = _inf.hierarchical_fit(Bernoulli(Beta(164.0, 453.9).each()), games, max_its=100000, tol=1e-8).summary()['hyper']
print('internal function, max_its=100000:', hy['iterations'], hy['converged'], round(hy['a'], 6), round(hy['b'], 6))
```

- **Observed:** venv-full (out/q/q9_ensemble_hierarchical.082full.txt, out/q/q10_hierarchical_families.082full.txt, out/q/q8_followups.082full.txt): the baseball-style model's default fit reports hyper {'a': 189.636535, 'b': 524.808084, 'iterations': 300, 'converged': False} -- 300 is hierarchical_fit's default max_its -- with no warning (warnings.catch_warnings recorded none); fit(games, max_its=3000), max_its=30000, delta=1e-4 and max_its=3000+delta=1e-12 all return byte-identical hyper with iterations=300, converged=False. Gamma-Poisson: max_its=1 still reports iterations=11 (q10). The internal function with a raised cap (out/q/q11_ensemble_seeds_hier.082full.txt) converges at iteration 5752 to a=204.397259, b=565.657531 (tol 1e-8) -- the returned 300-iteration hyper-parameters (189.64, 524.81) are 7% short of the fixed point (the population mean 0.265432 is unaffected; the concentration, i.e. the shrinkage, is what is wrong). Identical on 0.8.1 (out/q/q11_ensemble_seeds_hier.081.txt).
- **Expected:** The P07-F05 contract ("LocalLevel().fit() warns when it returns an iteration-capped, unconverged fit, the same way optimize() does") applied to the sibling hierarchical route: a warning naming max_its when the cap is hit, and fit(max_its=, delta=) reaching hierarchical_fit(max_its=, tol=) -- the documented front-door arguments -- instead of being consumed by fit()'s signature and dropped.
- **Notes:** Mechanism (core.py fit dispatch, ~line 3165): `if how in {"map", "laplace"}: kw.setdefault("max_iter", max_its) ...` -- max_its/delta are forwarded to those two routes only; every other fitter (hierarchical_fit(max_its=300, tol=1e-8), the samplers) never sees them, and because max_its/delta are named parameters of fit() they are silently absorbed rather than refused (q6: max_its=50 on how='mcmc' is a no-op). hierarchical_fit's moment-matching loop tests an absolute 1e-8 change on hyper-parameters of size 190/525 and the Beta-Bernoulli case needs more than 300 iterations to meet it; the receipt records converged=False but nothing surfaces it. Pre-existing in 0.8.1; the P07-F05 repair covered LocalLevel only.
- **Repairs concerned:** P07-F05, P07-F04

### Q04-F15 -- minor -- fit() arguments a route does not use are silently accepted: max_its/delta on every sampler and the hierarchical route, and rng of any type or draws= on the conjugate route; an unknown name is refused with the internal fitter's signature error

- **Surface:** RandomVariable.fit(**kw) dispatch; conjugate_fit; the sampler fitters
- **Reproduction:**

```python
import numpy as np
from mixle.ppl import Normal, free
dd = list(np.random.RandomState(0).normal(5, 2, 400))
m = Normal(Normal(0, 10, name='mu'), free)
print(m.fit(dd, how='mcmc', draws=12, burn=5, max_its=50, rng=0).summary()['mu']['mean'])       # max_its silently ignored
c = Normal(Normal(0, 10, name='mu'), 2.0)
for rng in ('x', 1.5, True, -1):
    print(rng, c.fit(dd, how='conjugate', rng=rng).explain_fit()['route'])                       # accepted, ignored
print(c.fit(dd, how='conjugate', draws=5).explain_fit()['route'])                                  # accepted, ignored
try: m.fit(dd, how='mcmc', draws=12, burn=5, seed=0)
except TypeError as e: print(e)
```

- **Observed:** venv-full and venv-nonumba, identical (out/q/q5b_draws_chains_rng.{082full,082nonumba}.txt; out/q/q6_grouped_bernoulli.082full.txt; out/q/q10_hierarchical_families.082full.txt): how='mcmc' with max_its=50 -> fitted, identical to the run without it; the conjugate route accepts rng='x', 1.5, True, -1, 2**64 and draws=5 without a word, where the nine other routes refuse 'x'/1.5/True by name and -1/2**64 with numpy's "Seed must be between 0 and 2**32 - 1"; hierarchical: max_its=1/5/3000 and delta=1e-3 change nothing (Q04-F14). Unknown names: `TypeError: mcmc_fit() got an unexpected keyword argument 'bogus'` / `'seed'`, `map_fit() ... 'bogus'`.
- **Expected:** An argument the route cannot honour is refused by name ("how='conjugate' takes no rng") or forwarded (max_its/delta to the routes that have a cap), the way draws/chains/burn are validated on every sampler; an unknown name is refused in terms of fit(), not of an internal function.
- **Notes:** Mechanism: fit() declares max_its/delta as its own parameters and forwards them only for how in {'map', 'laplace'} (core.py ~3165); conjugate_fit takes **kw-style extras without validating rng. Pre-existing. The consequence for the grouped default route is Q04-F14.
- **Repairs concerned:** P04-F11, R02-F12

## Attacks that did not break anything
- P04-F04/R02-F02 zero-acceptance refusal: hmc with an active hard constraint (mu > 6) refuses by name; mcmc/nuts/ensemble sample the truncated
  posterior (means 6.01-6.02); the vector `DiagGaussian + increasing(v)` case refuses on mcmc at draws=100 ("nothing this sampler proposed was feasible")
  and samples at draws=2000/burn=500 (acceptance 0.038, v1 = v2 = 1.46 on the boundary); hmc/nuts on that case still end in the R02-F17 raw
  non-finite-gradient messages (ledgered, not repaired, not new). Zero-acceptance at ordinary budgets: draws=50/100/200 burn=0 never refused over 8 seeds;
  draws=4 burn=100 refused 2/8 seeds with the "acceptance rate of 0.50 but every retained draw is the same point" cause. Route-by-route sweep
  (out/q/q1_zero_acceptance.{082full,082nonumba,081}.txt, identical on venv-full and venv-nonumba): every case that 0.8.1 returned as a silent
  std-0 point is refused by name on 0.8.2 -- hmc mu>6 (chains 1 and 2), hmc mu>50 chains=2, hmc step_size=1e3 ("Reduce step_size ..."), mcmc and
  ensemble on the vector hard constraint, ensemble mu>50 ("reported an acceptance rate of 0.89 but every retained draw is the same point", the R07-F01
  stretch-move case); no SILENT-POINT result anywhere on 0.8.2. The truncated targets that are sampled agree across routes (mu>6: 6.01-6.02 on
  mcmc/nuts/ensemble; mu>50: hmc 50.12/45.7, nuts 50.09/45.1 for (mu, sd)); the random walk at draws=200 from that infeasible start returns an
  unconverged chain (mcmc chains=1: mu 56.3, sd 3.8e4) -- a budget problem the diagnostics exist for, not a guard defect. penalty=10 turns every
  vector case into a sampled soft posterior on all four routes.
- P04-F07 on every route (out/q/q3_penalty_routes.082full.txt): penalty=0/-1/inf/nan/'x'/True/[1.0]/ndarray refused by name on map, laplace, mcmc, hmc,
  nuts and ensemble; np.float32 and int accepted; penalty= is validated even with no constraints; vi refuses constraints altogether by name. MAP failure
  messages: an infeasible constraint set -> "could not find a parameter point satisfying the constraints"; a -inf potential at the start -> "the objective
  is not finite at the starting point"; a constrained max_its=1 -> "penalty ramp exhausted max_iter without converging. Raise max_iter (currently 1)"
  (P04-F02's silent success is gone); laplace under a hard constraint refuses by name.
- P04-F05 pickling: every posterior route (conjugate, mcmc, hmc, nuts, ensemble, laplace, vi) pickles and its summary() is identical after restore;
  the six Posterior-backed routes refuse predict() by name after restore; pointwise_log_likelihood refuses by name; dump_models still raises
  SerializationError for a RandomVariable (not claimed). The conjugate predict() gap is Q04-F01.
- P04-F07 penalty=: inf/nan/0/-1/'x'/True refused by name on how='map' ("penalty must be a finite positive number, got ..."); np.float64(2.0) accepted; a
  potential returning nan/inf on map gives "the objective is not finite at the starting point ... raising max_iter cannot help". Other routes and spellings:
  out/q/q3_penalty_routes.082full.txt.
- P04-F09: explain_fit() after how='auto' keeps the router reason and adds route_requested='auto' (out/q/q4_* covers the conjugate/em/regression/map/Mix routes).
- P04-F10/R02-F06: unfitted DiagGaussian(5, mean=free(5, name='v')) -> summary()/params raise "has unresolved `free` parameters; call .fit(data) first";
  repr/str print `RV(DiagGaussian(5, free(5, name='v'), array([...])))`; repr(free(3)) -> `free(3)`.
- P04-F11/R02-F12 (out/q/q5b_draws_chains_rng.{082full,082nonumba}.txt, identical): on mcmc, hmc, nuts and ensemble alike draws=0/-1/3 ->
  "draws must be an integer >= 4" (ValueError), 4.0/'100'/None/2.5/True -> the same text as TypeError, np.int64 accepted; chains=0/-1 -> ">= 1",
  1.5/'2'/True refused; burn=-1 -> ">= 0", 'x'/None/2.5 refused, burn > draws accepted. rng=int/np.int64/Generator/RandomState/None accepted and
  'x'/1.5/True/np.True_ refused on mcmc, hmc, nuts, ensemble, laplace, vi, map, em and auto (-1 and 2**64 get numpy's "Seed must be between 0 and
  2**32 - 1"); rng=0 twice, rng=0 vs RandomState(0), and Generator(0) twice give identical mcmc draws. potentials='x' refused by name. The conjugate
  route's silent acceptance of any rng is Q04-F15. (The first q5 run timed out at 900 s on my own burn=10**6 case on hmc, out/q/q5_*.txt; q5b is the
  same probe without it.)
- P08-F08 with chains=2 (q5b): hmc draws=100 x 2 chains -> ess_bulk 200.0 = the pooled draw count (raw per-parameter ESS [200, 200]); nuts raw
  [94.9, 200] clipped for one parameter; mcmc 22.7/26.7 unclipped -- the clip is at the total pooled draws, as it should be.
- Unknown keyword names are refused (`mcmc_fit() got an unexpected keyword argument 'bogus'`), so a typo cannot pass silently; a fit()-named
  argument can (Q04-F15).
- P04-F12: the grouped constrained MAP receipt reports objective_evaluations=173 and no hard-coded iterations=0.
- P07-F03: Poisson(2.0)/NegativeBinomial(2.0,0.5)/StudentT(5,0,1).fit warn "had nothing to estimate"; StudentT(free,free,free) also warns that df is not fitted.
- P07-F04 (out/q/q6_grouped_bernoulli.{082full,082nonumba,081}.txt, identical on venv-full/nonumba; 0.8.1 raises NotImplementedError on every
  route): grouped Bernoulli-Beta against the exact per-group Beta(a+h, b+45-h) posterior (p[0] exact mean 0.2746, sd 0.0173): mcmc default budget
  max|mean-exact| 0.0076 (sd ratio 0.88-1.16, acceptance 0.222), 3000 draws 0.0058, chains=2 0.0046 with split_r_hat 1.06; nuts 0.0014; hmc 0.0007;
  laplace 0.0014 (sd ratio 0.96-1.02); map returns an IndexedPosterior. Degenerate groups on mcmc: all-0 and all-1 groups, a single-game group, int
  labels, an (18,45) ndarray and a single group all sample (list, ndarray and list-of-ndarray inputs give identical chains at the same budget and rng,
  out/q/q8_followups.082full.txt); an empty group, a value 2.0 and a NaN are refused by name. The ledger spelling max_its=50 is accepted and ignored on
  the mcmc route (Q04-F15). The ensemble route on this model is Q04-F11; the default (hierarchical) route's receipt is Q04-F14 and its rng= crash Q04-F12.
- Constraint bands (out/q/q9_ensemble_hierarchical.082full.txt): [6, 6.1] is sampled by mcmc (mean 6.054) and solved by map (6.000); [6, 6.01] is
  sampled by mcmc (mean 6.0048, sd 3.7e-4). The narrower bands are in Q04-F10.
- P04-F09 on every resolved route (out/q/q4_auto_reason_free.082full.txt): conjugate, em, map-with-prior, Mix->em, Poisson-Gamma and the grouped
  hierarchical route all keep the router's reason and add "(requested how='auto'; resolved to '<route>' by fit)" with route_requested='auto'; the
  regression route keeps its reason but records route_requested=None (cosmetic). free(0/-1/2.5/(2,3)/'a'/None/[3]/True) are refused by name
  ("free dimension must be an exact positive integer"); an unnamed free(5) fits and summarizes; an unfitted vector handle refuses sample/log_density/
  mean with the unresolved-free message and explain_fit() still answers.
- P08-F15 predict(given) on the Normal regression draws (different rng -> different values; result.predict gives the mean); the Poisson/Bernoulli case
  is Q04-F02.
- venv-nonumba: the ledger replay (out/q/p1_ledger_replay.082nonumba.txt) is byte-identical to venv-full apart from paths and timings. venv-base
  (out/q/p1_ledger_replay.082base.txt): MAP warns that it uses a derivative-free optimizer without torch; the degenerate one-observation
  Normal(free,free) cases fail differently (map: "inference likelihood returned invalid value nan"; vi: internal "failed at unconstrained coordinates
  [4.5e166, 388]") -- part of Q04-F08; nuts under a hard constraint is in Q04-F10.
- P08-F08: HMC ESS is clipped to the draw count (raw ESS [200, 200] for 200 draws; 0.8.1 reported 460); the tutorials now print "HMC ESS 1000" for
  1000 draws and "ESS(mu)=1500 / 1500 draws".
- P08-F14: 3*Normal(0,1)+1 reports mean 1.0 / var 9.0 exactly; TransformDistribution.mean() for ExpTransform raises NotImplementedError while
  RandomVariable.mean() falls back to Monte Carlo (1.6548), as the migration guide says.
- P08-F15: str(fitted regression) -> `RV(fitted RegressionResult: x=1.986, z=-0.9743, intercept=0.4753)`; given=pd.DataFrame and y as pd.Series fit;
  predict(DataFrame) works; ragged/missing/nan/str/2-d covariates refused by name (missing field is a bare KeyError 'z' on the RandomVariable path but
  "given is missing required field" on the regression path); a fitted regression pickles and predicts after restore.
- R02-F03 guard: proper-prior one-observation fits (Normal(mu~N(0,10), sd~Gamma(2,2)) on 1 obs; DiagGaussian(5) on 3 rows) fit on mcmc/laplace/map;
  Mix(weights=free) on 4 and 5 observations fits on mcmc (guard counts 4 flat parameters, not 5).
- Random-walk mcmc at draws=300/burn=100 over 30 seeds: posterior sd 0.080-0.093 against the exact 0.1, acceptance 0.44-0.51, no refusals.
- split_r_hat is NaN for a single chain on every sampler: documented in both the Posterior.summary docstring and diagnostics.split_rhat ("at least two
  chains"), so not a finding; ess_bulk/ess_tail are finite for one chain.
- Notebook prose: the fresh numbers that changed (ESS clips, VI sds 0.301/0.302 vs MCMC 0.358, affine moments 1.00/9.00, nuts acceptance 0.95) all
  support the notebooks' qualitative claims; no stored number is quoted in prose.

## What was not covered
- how='vmp' and how='sample' routes; SemiMix, Seq, LocalLevel/AR1 through the PPL front door (assigned to other passes); the Cox and quantile-regression
  notebooks' internals beyond execution and output comparison; potentials over non-parameter RVs (P04-F08, docs, not in the repair list).
- `dump_models`/`to_json` of fitted RandomVariables (still refused by SerializationError; not claimed repaired).
- Three repairs on the assignment list were not replayed for lack of budget after the sampler work: P07-F13 (MarkovChainEstimator pseudo_count
  docstring), P07-F15 (NumbaKernelFactory.build without numba) and P08-F17 (an (N,1) column of user-module scores); none is a mixle.ppl surface.
- The background exec (`REVIEW_ROOT/exec/`) had not logged any of this pass's notebooks when read (its examples.log agrees with the five example
  exits here: all 0), so the notebook table above rests on this pass's own execution only.
- The pyspark/torch-free base environment beyond the ledger replay (out/q/p1_ledger_replay.082base.txt): the q-probes were run on venv-full,
  venv-nonumba and 0.8.1 only.
- Performance: every wall time here was measured under 3 concurrent notebooks plus probes on a shared machine (project_neural_to_structured took 979 s
  here vs 27 s in the background exec) and is not evidence.
- The previous reviewer's transcript ended before any report was written; everything above is reconstructed from its saved probe outputs plus the
  q-series probes added in this recovery pass. The examples' README/manifests named in the brief do not exist in this export (source/examples has no
  README.md; source/manifests/ holds repository-level manifests), so example claims were read from each script's own printed prose.

DONE 04 15 findings
