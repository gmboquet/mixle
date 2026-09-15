# Independent reproductions of blocking findings (campaign lead, not the pass reviewers)

Every probe here was run through `tools/pyt.py` from `/tmp` with `PYTHONPATH` unset; the interpreter
printed `mixle.__path__[0]` first. Scripts live in `verification/`.

## Q04-F01 (blocking) -- CONFIRMED 2026-09-11 18:4xZ+8

Script `verification/q04f01.py` (the reviewer's reproduction, with the refusal wrapped in try/except).

| interpreter | result |
|---|---|
| `venv-full` (0.8.2 wheel e0c5087d, commit 866078be) | `restored predict: NO refusal, sd=2.0280`; route `conjugate`, result type `ConjugatePosterior`, `_dropped_closures=('predictive',)`, `hasattr(_refuse_dropped_closure)=False`; integrated predictive sd 2.3056, live predictive sd 2.3003 |
| `venv-base` (numpy+scipy only) | identical |
| `candidate-081/venv` (published 0.8.1) | `pickle failed: AttributeError Can't get local object '_conj_normal_mean.<locals>.<lambda>'` -- the fit could not be pickled at all, so the silent plug-in answer is new in 0.8.2 |

Mechanism confirmed in the source export: `mixle/ppl/core.py:2602` refuses only `if r is not None and
hasattr(r, "_refuse_dropped_closure")`; that method is defined on `Posterior` (`inference.py:233`)
and not on `ConjugatePosterior` (`inference.py:3623`); `RestoredPredictiveTest`
(`tests/adversarial_review_082_repairs_test.py:134`) iterates `how in ("mcmc", "laplace")` only.
Verdict: the R02-F04 CHANGELOG claim ("predict() on a posterior restored from a pickle refuses
instead of answering with the plug-in predictive") does not hold on the default (conjugate) route.

## Q01-F01 (blocking) -- CONFIRMED (`verification/q01-f01-print.py`, output `q01-f01-print.venv-full.out`)

On the candidate (`venv-full`): `HalfNormal(1).quantile(-0.1)` -> `-0.1257`; `BetaBinomial(10,2,3).quantile(1.1)` -> `10.0`
and `.quantile(nan)` -> `10.0`; `Uniform(-1,2).quantile(1.1)` -> `nan`; `Laplace(0,1).quantile(nan)` -> `nan`;
`StudentT(5).quantile(-0.1)` -> `nan`. Gaussian, Poisson and Exponential refuse with the shared
`ValueError: <Family>.quantile: q must be in [0, 1].` The CHANGELOG (R05-F07: "Every family that defines
`quantile` refuses an out-of-domain `q` the same way") and the migration guide ("refuses a `q` outside `[0, 1]`
on every family") are contradicted on at least these five families (the reviewer lists 16). Note: the
reviewer's reproduction block as written stops at its last line, which deliberately triggers the correct
refusal without catching it; the finding itself stands.

## Q01-F02 / Q02-F02 (blocking, found independently by passes 01 and 02) -- CONFIRMED (`verification/q01-f02.py`)

`optimize(counts, MixtureEstimator([PoissonEstimator(), GeometricEstimator()]))` on counts containing zeros is
refused from `seq_initialize` (`geometric.py:565` -> `refuse_unsupported_observations`): "GeometricDistribution
has support k in {1, 2, 3, ...}, but at least 4 observation(s) carrying weight are below one ...". The mixture
initialization handed weighted zero rows to the Geometric component, so the new P01-F07 guard refuses the whole
fit and blames the data. Contradicts P02-F03 ("mixture ... initialization consult each component's support
before handing it any responsibility"); pass 02 traced the mechanism to `supported_rows` being implemented by
the ten continuous families only.

## Q02-F01 (blocking) -- CONFIRMED (`verification/q02-f01.py`)

On a two-state terminal-state HMM with x = [-3, -3, -3, -3] (reviewer's model): the admissible last-row marginal
is `[0, 1]`; `latent_posterior().marginals()` returned `[1.0, 3.8e-09]`; `viterbi`/`mode` returned `[0 0 0 0]`
(a path ending in a non-terminal state, which the model scores as impossible); the length-1 sequence was assigned
state `[0]`; 200 of 200 FFBS samples ended in a non-terminal state. `seq_posterior` and `log_density` are
correct (the R05-F01 repair); the other readouts enforce only "no terminal state before the end".

## Q05-F01 (blocking) -- CONFIRMED (`verification/q05-f01.py`, output `q05-f01.venv-full.out`)

`optimize(df, max_its=2)` -> `CompositeDistribution` over the rows; `fit_with_provenance(df, None, max_its=2)`
-> `CategoricalDistribution({'k': 0.5, 'x': 0.5})` with `n_records=2` (the column names);
`fit_with_provenance('hello world hello', ...)` -> `n_records=17` (the characters; `optimize` refuses a bare
`str` by name); a mapping of columns -> the same categorical over the names. The reviewer's block then calls
`Service(comp).score(df)`, which raises `ContractError` as the comment predicts (uncaught in the block as
written). Contradicts R06-F03 ("one table gets one answer whichever verb reads it").

## Q04-F11 (blocking) -- CONFIRMED (`verification/q04-f11.py`, output `q04-f11.venv-full.out`, 38 s)

`DiagGaussian(5, mean=free(5), var=0.25)` on 300 rows, default budgets, `rng=RandomState(0)`:
`ensemble` max|mean-exact|/sd = 3.73, sd ratio [6.36 9.14 5.60 4.43 3.19], ess_bulk 31.5; `mcmc` 0.15 / [0.96..1.09];
`nuts` 0.02 / [0.96..1.01]; `hmc` 0.05 / [0.99..1.00]. Grouped `Bernoulli(Beta(164, 453.9).each())` on 18 groups,
`how='ensemble'` default budget: worst group 13 is 7.78 posterior sds off the exact conjugate mean with a reported
mcse of 0.0040; 6 of 18 groups are more than 3 sds off. No warning on the single-chain default. The reviewer reports
the same on 0.8.1 (pre-existing), and that `chains>=2` does expose it through R-hat.

## Q08-F08 (blocking) -- duplicate of Q01-F01, CONFIRMED by the same run

Pass 08 independently found `UniformDistribution.quantile` and `LaplaceDistribution.quantile` returning NaN for
q = -0.5, 1.5 and NaN on the candidate (unchanged from 0.8.1) against the R05-F07 claim. My `q01-f01-print.py`
run shows exactly that (`Uniform(-1,2).quantile(1.1) -> nan`, `Laplace(0,1).quantile(nan) -> nan`). One defect,
two passes.

## Execution fact on the campaign wheel (`exec/`)

Examples: 57 of 57 exit 0 (`exec/examples.log`, sequential, 2026-09-11 03:05-03:51Z). Notebooks: 131 of 131
exit 0 (`exec/status.log`, `xargs -P 2`, 10:06-12:26Z, under a peer-session load of 60-90), including
`applications/malware_certificate_embedding.ipynb` (4426 s), `data_science/model_based_embeddings.ipynb`
(2846 s), `applications/radar_tomography.ipynb` (1684 s) and `tutorials/estimation_using_spark.ipynb` (with
the openjdk@17 JDK on PATH). Wheel `e0c5087d…` (commit 866078be), `venv-full`.

## Q09-F01 (blocking) -- CONFIRMED (`verification/q09-f01.py`, output `q09-f01.out`)

Two-topic `TreeHiddenMarkovModelDistribution` fitted by `optimize(..., max_its=8)` on 150 sampled trees:
0.8.2 (`venv-full`): constructed instance `JSON OK`; fitted instance `SerializationError: dump_models produced
JSON that load_models cannot read back (... does not match its constructor-owned schema ...)`; the fitted
object's `_p_level_cache` is a `tuple` (constructed: `None`); setting it to `None` makes the same object round-trip.
0.8.1 (`candidate-081/venv`): the fitted instance round-trips and its `_p_level_cache` is `None` after the fit.
A regression introduced on the 0.8.2 line; pass 02 (Q02-F09) hit the same refusal.
