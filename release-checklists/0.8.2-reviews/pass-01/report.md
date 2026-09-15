# Pass 01 -- univariate laws and their estimators (continuous and discrete), enumeration

Recovery report. The original reviewer executed the corpus, wrote and ran the probes under `probes/`,
`nb/`, `ex/` and `tests/`, and was killed by an API rate limit before writing this file. Everything
below is drawn from that preserved evidence plus five short confirmation runs made for this report
(`probes/confirm_*.py`, `tests/run_pytest2.sh`); no notebook or example was re-executed.

## Header

- Wheel: `mixle-0.8.2-py3-none-any.whl`, sha256
  `e0c5087d1ce4463e91105b96675e38f724e0a86314471944c96d2491dd0895da` (`env/wheel_sha256.txt`).
- Commit `866078be520b22188110780be957150dc6da964c`, tree `7922a8c59283ecef6877104b9a7499afd52d9013`.
- Work dir: `REVIEW_ROOT/pass-01/` (all paths below are relative to it).
- numpy 2.5.3, scipy 1.18.1, Python 3.12.12 in every environment used (the same scipy in the 0.8.1 venv).

Verification command, `cd /tmp && <interpreter> REVIEW_ROOT/tools/verify_env.py`, verbatim output per
environment used (`env/verify_*.txt`):

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

Every probe prints `mixle.__path__[0]` on its first line (`probes/*.txt`); every one loaded the
site-packages tree of the interpreter named in its filename.

## Corpus executed on the candidate

Notebooks: `venv-full`, executed in place in the APFS clone `corpus/` via `tools/run_nb.sh`, three at a
time (`nb/run_all_nb.sh`, `nb/table.txt`, per-notebook `corpus/notebooks/<dir>/<name>.log`). Wall times
include contention from the 3-way parallel run and the background full-corpus execution.

| notebook | exit | wall |
| --- | --- | --- |
| data_science/multiple_testing_and_fdr | 0 | 38 s |
| data_science/ab_testing_and_power | 0 | 50 s |
| tutorials/bayesian_distributions | 0 | 64 s |
| data_science/extreme_value_theory | 0 | 33 s |
| tutorials/enumeration | 0 | 60 s |
| data_science/bootstrap_and_uncertainty | 0 | 117 s |
| data_science/exponential_families_and_conjugacy | 0 | 99 s |
| data_science/count_data_overdispersion | 0 | 107 s |
| tutorials/fitting_and_estimation | 0 | 175 s |
| data_science/information_theory_in_practice | 0 | 204 s |
| data_science/mle_and_sufficient_statistics | 0 | 163 s |
| data_science/heavy_tails_and_survival | 0 | 156 s |
| data_science/conjugate_and_nonparametric_bayes | 0 | 487 s |
| tutorials/distributions_and_combinators | 0 | 198 s |
| data_science/density_estimation_mixtures_vs_kde | 0 | 1150 s |

Examples: `venv-full`, `tools/pyt.py 1800`, each with `PYT_CWD=ex/cwd_<name>` (`ex/run_all_ex.sh`,
`ex/table.txt`, stdout in `ex/<name>.txt`; none wrote an artifact).

| script | exit | wall |
| --- | --- | --- |
| gallery_univariate_example | 0 | 53 s |
| enumeration_example | 0 | 37 s |
| enumeration_showcase_example | 0 | 55 s |
| autoregressive_enumeration_example | 0 | 74 s |
| frontier_family_showcase | 0 | 766 s |

Fresh vs stored outputs, cell by cell (`probes/nb_compare.py`; `nb/compare_part1.txt`,
`nb/compare_part2.txt`): no notebook's fresh output contradicts a stored number or a prose claim.
Eleven notebooks are identical after normalization. Four differ only in bookkeeping:
`count_data_overdispersion`, `conjugate_and_nonparametric_bayes`, `density_estimation_mixtures_vs_kde`
gain `Iteration N: ...` progress lines that 0.8.2's `print_iter` change (CHANGELOG: "An explicit
`print_iter=` prints without `out=`") now writes to stdout, with the same final objectives;
`fitting_and_estimation` additionally moves the "stopped at the max_its cap" warnings from
`estimation.py:2170` in the stored copy (which was executed on the 0.8.1 venv -- its warning paths
name `candidate-081`) to the calling cell, and relabels the `best_of()` ones. `distributions_and_combinators`
executes fully (exit 0) although nbconvert logs a notebook-format validation error (a markdown cell
carries an `id` under nbformat 4.4) -- a corpus-format nit, not a library defect.

Regression tests: the twelve assigned test files copied to `tests/run/` and run against the installed
wheel from there. The brief's recipe fails at collection because `mixle/tests/conftest.py` reads
`--num-shards`, which only the repo-root `conftest.py` defines (`tests/pytest.venv-*.txt`,
INTERNALERROR); with that file loaded as a plugin (`tests/run/rootconf.py`, `-p rootconf`;
`tests/run_pytest2.sh`) the suite passes: `venv-full` 319 passed, 563 subtests, 334 s
(`tests/pytest2.venv-full.txt`); `venv-base` 311 passed, 8 skipped (torch-GP surrogate tests), 559
subtests, 204 s (`tests/pytest2.venv-base.txt`).

## Findings

### Q01-F01 -- blocking -- `quantile(q)` still answers an out-of-domain `q` on 16 of the 31 families the CHANGELOG (R05-F07) and the migration guide say now refuse it; two of them answer with a number

**Surface:** `BetaBinomialDistribution`, `ExponentiallyModifiedGaussianDistribution`,
`GeneralizedExtremeValueDistribution`, `GeneralizedGaussianDistribution`, `HalfNormalDistribution`,
`InverseGaussianDistribution`, `LaplaceDistribution`, `LogisticDistribution`, `NakagamiDistribution`,
`ParetoDistribution`, `RayleighDistribution`, `RicianDistribution`, `SkellamDistribution`,
`SkewNormalDistribution`, `StudentTDistribution`, `UniformDistribution` `.quantile(q)`; CHANGELOG 0.8.2
R05-F07 paragraph; `docs/migrations/0.8.2.md` section "`quantile(q)` refuses a `q` outside `[0, 1]` on
every family"; `adversarial_review_082_repairs_test.py::QuantileDomainTest`.

**Reproduction** (`probes/quantile_domain.py`, `probes/confirm_quantile_tails.py`):

```python
import mixle.stats as S
S.HalfNormalDistribution(1.0).quantile(-0.1)                    # -0.12566134685507405 (negative, from a law on x >= 0)
S.BetaBinomialDistribution(10, 2.0, 3.0).quantile(1.1)          # 10.0
S.BetaBinomialDistribution(10, 2.0, 3.0).quantile(float('nan')) # 10.0
S.UniformDistribution(-1.0, 2.0).quantile(1.1)                  # nan
S.LaplaceDistribution(0.0, 1.0).quantile(float('nan'))          # nan
S.StudentTDistribution(5.0).quantile(-0.1)                      # nan
S.GaussianDistribution(0.0, 1.0).quantile(-0.1)                 # ValueError: GaussianDistribution.quantile: q must be in [0, 1].
```

**Observed** (`probes/quantile_domain.venv-full.txt`, `probes/confirm_quantile_tails.venv-full.txt`):
for `q` in {-0.1, 1.1, nan, inf}, 15 families raise `<Family>.quantile: q must be in [0, 1].`; 14 return
`nan`; `BetaBinomial` returns `0.0` at `q=-0.1` and `10.0` at `1.1`, `nan` and `inf`; `HalfNormal`
returns `-0.1257` at `q=-0.1` and `nan` at `1.1`/`nan`. The 16 behave identically on 0.8.1
(`probes/quantile_domain.c081.txt`): the repair reached 15 families. `Laplace` and `Uniform` are two of
the seven continuous families R05-F07 itself listed as its surface.

**Expected:** CHANGELOG: "Every family that defines `quantile` refuses an out-of-domain `q` the same
way, through one shared `validated_quantile_probability`"; migration guide: "on every family ... All
thirteen now raise `ValueError` naming the family."

**Notes:** `validated_quantile_probability` is called by eight continuous families (Beta, Exponential,
Gamma, Gaussian, GeneralizedPareto, InverseGamma, LogGaussian, Weibull) and six discrete ones
(Bernoulli, Binomial, Geometric, LogSeries, NegativeBinomial, Poisson); Gumbel has its own P01-F12
check; the other 16 never validate `q`. The regression test `QuantileDomainTest._families()` builds
each quantile-bearing class from a fixed tuple of nine kwargs shapes and silently skips
(`except Exception: continue`) every class whose constructor takes other names -- exactly these 16 --
and `test_the_survey_reaches_every_family_it_claims_to` only asserts `>= 13` families, so the test
passes without ever calling `quantile` on them. Repairs: R05-F07, P01-F12.

### Q01-F02 -- blocking -- a support-limited component still cannot be mixed unless it is one of eight families: five continuous and three count encoders refuse out-of-support rows outright, and for Poisson/Geometric/LogSeries the mixture hands those rows initial responsibility, so the new P01-F06/F07/F09 support guards refuse the fit and blame the data

**Surface:** `mixle.inference.optimize`/`fit` with a `MixtureEstimator` whose components include
`ParetoEstimator`, `GeneralizedParetoEstimator`, `NakagamiEstimator`, `RicianEstimator`,
`TweedieEstimator`, `NegativeBinomialEstimator`, `BetaBinomialEstimator`, `BernoulliEstimator` (encoder
refusal) or `PoissonEstimator`, `GeometricEstimator`, `LogSeriesEstimator` (guard refusal on the
initial responsibilities); `MixtureDistribution.seq_log_density` on such batches; those families'
`dist_to_encoder().seq_encode`.

**Reproduction** (`probes/confirm_mixture_guards.py`, `probes/encoder_refusals_in_mixtures.py`):

```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
rng = np.random.RandomState(0)
counts = np.concatenate([rng.poisson(2, 200), rng.geometric(0.3, 200)])   # 28 zeros; the Poisson component explains them
optimize(counts, S.MixtureEstimator([S.PoissonEstimator(), S.GeometricEstimator()]), max_its=20, rng=np.random.RandomState(1))
# ValueError: GeometricDistribution has support k in {1, 2, 3, ...}, but at least 4 observation(s) carrying weight are below one, ...
x = np.concatenate([rng.normal(-3, 1, 200), rng.pareto(3., 200) + 1.0])
optimize(x, S.MixtureEstimator([S.GaussianEstimator(), S.ParetoEstimator()]), max_its=20, rng=np.random.RandomState(1))
# TypeError: MixtureDistribution could not encode the data with all of its component encoders ...
S.ParetoDistribution(1.0, 3.0).dist_to_encoder().seq_encode([-1.0, 1.5])   # ValueError: ParetoDistribution requires observations x > 0.
S.ParetoDistribution(1.0, 3.0).log_density(-1.0)                           # -inf  (the scalar path scores it)
```

**Observed** (`probes/confirm_mixture_guards.venv-full.txt`,
`probes/encoder_refusals_in_mixtures.venv-full.txt`; identical on `venv-base` and `venv-nonumba`):
Gaussian+{Exponential, Gamma, Beta, HalfNormal, Rayleigh, InverseGaussian, LogGaussian, Weibull} encode,
batch-score and fit with out-of-support rows present. Gaussian+{Pareto, GeneralizedPareto, Nakagami,
Rician, Tweedie} and IntegerCategorical+{NegativeBinomial, BetaBinomial, Bernoulli} raise the encoder
`TypeError` from `seq_log_density` and `optimize`, while `MixtureDistribution.log_density` scores the
same rows (`-4.737...`). Poisson+Geometric (seeds 1, 7, 11), Poisson+LogSeries, NegBin+Geometric,
Skellam+Poisson and IntegerCategorical+Poisson raise the P01-F06/F07/F09 `ValueError` naming 4 to 14
"observation(s) carrying weight". On 0.8.1 (`probes/confirm_mixture_guards.c081.txt`) every one of
these -- the Exponential and Gamma controls included -- failed at encode time with the `TypeError`: the
P02-F03 repair reached 8 of the 16 support-limited families and none of the count families.

**Expected:** CHANGELOG P02-F03: "A support-limited component can be mixed. ... The encoders now admit
out-of-support observations and the vectorized scorers return the `-inf` the scalar path returns; ...
mixture, heterogeneous-mixture and hidden-Markov initialization consult each component's support before
handing it any responsibility." A Poisson+Geometric mixture on counts containing zeros, or a
Gaussian+Pareto mixture on data with negative values, is an ordinary two-population model.

**Notes:** the support consultation is wired for the ten continuous families that use
`refuse_unsupported_observations`; the three count families use the same helper for the refusal but
initialization does not consult their support, so the guard fires on the random initial
responsibilities -- a fail-closed guard rejecting a state the library itself produces (the P01-F06,
P01-F07, P01-F09 guards are the ones firing). The five continuous and three count encoders that still
refuse never received the P02-F03 encoder change. Overlaps pass 02's area; reported here because the
encoders and the guards belong to the univariate families. Repairs: P02-F03, P01-F06, P01-F07, P01-F09.

### Q01-F03 -- real -- the zero-weight exemption (R05-F05) holds only on the vectorized `seq_update`/`seq_initialize` routes; the scalar `update`/`initialize` routes still turn a zero-weight non-finite row into NaN statistics, and the public `initialize()` verb takes the scalar route, so it reproduces the pre-repair symptom -- and for LogSeries a silent wrong fit

**Surface:** `<Family>Accumulator.update`/`.initialize` for Exponential, Gaussian, Weibull, Beta (NaN
row), ExponentiallyModifiedGaussian, GeneralizedExtremeValue, GeneralizedGaussian, Gumbel, Logistic,
SkewNormal, StudentT, Poisson, Geometric, LogSeries; `mixle.inference.initialize(data, estimator, rng, p)`;
`LogisticAccumulator.seq_update` (zero-weight +-inf row).

**Reproduction** (`probes/zero_weight.py`, `probes/zw_trace.py`, `probes/confirm_initialize_zero_weight.py`):

```python
import numpy as np
from mixle.stats import ExponentialEstimator, ExponentialDistribution, LogSeriesEstimator
from mixle.inference import initialize, estimate
est = ExponentialEstimator(); rows = [1., 2., 1., 3., 1., 2., 1., 3.]
enc = ExponentialDistribution(1.0).dist_to_encoder().seq_encode([*rows, float('inf')]); w = np.array([1.] * 8 + [0.])
a = est.accumulator_factory().make(); a.seq_update(enc, w, None); print(a.value())   # (8.0, 14.0)  repaired route
b = est.accumulator_factory().make()
for x, ww in zip([*rows, float('inf')], w): b.update(x, ww, None)
print(b.value())                                                                     # (8.0, nan)   unrepaired route
initialize([*rows, float('inf')], est, np.random.RandomState(1), 0.5)   # ValueError: ExponentialDistribution requires beta > 0.
print(estimate([1, 2, 1, 3, 1, 2, 1, 3], LogSeriesEstimator()))        # LogSeriesDistribution(0.6434789567977548, ...)
m = initialize([1, 2, 1, 3, 1, 2, 1, 3, float('nan')], LogSeriesEstimator(), np.random.RandomState(1), 0.5)
print(m, m.numerical_repairs())                                        # LogSeriesDistribution(1.0000000000000002e-12, ...) ()
```

**Observed** (`probes/zero_weight.venv-full.txt`, `probes/zw_trace.venv-full.txt`,
`probes/confirm_initialize_zero_weight.venv-full.txt`): scalar `update` with a zero-weight inf/-inf/nan
row leaves NaN in a moment on the 14 families listed (Exponential `(8.0, nan)`; Gaussian `(nan, nan, ..)`;
Weibull `(nan, 30.0, 8.0)`; Poisson `(4.0, nan)`; ...) while `seq_update` and `seq_initialize` return the
clean statistic on every one of the ten repaired families. `initialize(..., p=0.5)` with one non-finite
row: on the seeds where the row draws weight 0 the call ends in "ExponentialDistribution requires
beta > 0." / "PoissonDistribution requires lam > 0." / "GeometricDistribution requires p in (0, 1]."
(2 of 6 seeds each), and LogSeries returns `p = 1e-12` (clean fit 0.643) with `numerical_repairs() == ()`
on 2 of 6 seeds. Logistic's `seq_update` also yields NaN for a zero-weight +-inf row. On 0.8.1
(`probes/zero_weight.c081.txt`, `probes/confirm_initialize_zero_weight.c081.txt`) the scalar route gave
the same NaNs, `initialize` failed on all 6 seeds for Exponential/Poisson and crashed for LogSeries
("cannot convert float NaN to integer"): 0.8.2 repaired the vectorized route and left the scalar one.

**Expected:** CHANGELOG R05-F05: "A zero-weight row contributes exactly zero to a sufficient statistic,
on all ten continuous families that exempt one"; migration guide: "a zero-weight row is still exempt";
the `initialize()` docstring: "Seq_initialize() ... should produce the same initialized model for the
same data sets." The scalar routes should mask the row exactly as `weighted_statistic_sum` does, and a
zero-weight row must never change a fitted parameter -- least of all silently with an empty
`numerical_repairs()`.

**Notes:** `mixle/stats/compute/sequence.py:645 initialize()` calls `accumulator.initialize(x, w, rng)`
row by row (line 695), which for these families is the unmasked scalar arithmetic `x * w`
(`inf * 0.0 = nan`); `weighted_statistic_sum` is applied inside `seq_update` only. The regression test
`ZeroWeightExemptionTest` exercises `seq_update` alone (line 47 of the copied test), so it cannot see
this. Logistic is not one of the ten (it does not use `refuse_unsupported_observations`) but its encoder
admits +-inf, so the same NaN appears on its vectorized route too. Repairs: R05-F05, P02-F03,
P01-F05, P01-F06, P01-F07, P01-F09.

### Q01-F04 -- real -- the R05-F09 diagnosis fires only when every observation is unscorable; the finding's own reproduction (one out-of-support row in a two-Exponential mixture) still ends in the bare internal message it reported

**Surface:** `mixle.inference.optimize`/`fit` with a `MixtureEstimator` of support-limited components; the
`ValueError` raised from `mixle/inference/estimation.py` ("fused EM did not produce a finite objective
from its non-finite initial model.").

**Reproduction** (`probes/r05f09_variants.py`, `probes/ledger_replay.py`; the R05-F09 batch verbatim):

```python
import numpy as np, mixle.stats as S
from mixle.inference import optimize
rng = np.random.RandomState(0)
x = np.concatenate([rng.exponential(1., 150), rng.exponential(5., 150), [-0.5]])
optimize(x, S.MixtureEstimator([S.ExponentialEstimator(), S.ExponentialEstimator()]), max_its=10, rng=np.random.RandomState(1))
# ValueError: fused EM did not produce a finite objective from its non-finite initial model.
optimize(-np.abs(x), S.MixtureEstimator([S.ExponentialEstimator(), S.ExponentialEstimator()]), max_its=10, rng=np.random.RandomState(1))
# ValueError: ... Every one of the 301 observation(s) scores -inf under it: none of them is in the support of any component. ...
```

**Observed** (`probes/r05f09_variants.venv-full.txt`, `probes/ledger_replay.venv-full.txt`): one negative
row (last or first), half the rows negative, 2xBeta with one 1.5, 2xGamma with one -1.0 -> the bare
message; `fit()` -> "EM did not produce a finite objective from its non-finite initial model."; only the
batch in which every row is out of support gets the diagnosis. A one-component `ExponentialEstimator` on
the same batch names the family and support, and so do `Composite(Gaussian, Exponential)` and an HMM
with Exponential emissions holding the same row.

**Expected:** the CHANGELOG lists R05-F09 among the re-review findings repaired afterwards and says a
fit whose objective never becomes finite "names the cause when it is knowable". On the finding's own
batch the cause is knowable (`MixtureDistribution.posterior(-0.5)` is `[0, 0]`; exactly one row scores
`-inf` under every component): the message should name that row, or at least say that some rows lie
outside every component's support.

**Notes:** `mixle/inference/estimation.py:198` builds the diagnosis only when the number of `-inf` rows
equals the batch size. The CHANGELOG paragraph re-scopes the finding to "a mixture whose components all
refuse every observation", which is not the case R05-F09 reported; its rated harm (an opaque message,
minor) is unchanged, and the repair claim does not cover the reported case. Repairs: R05-F09, P02-F03.

### Q01-F05 -- real -- scalar `log_density` disagrees with `seq_log_density` on large or non-finite observations: `GeneralizedGaussianDistribution.log_density` raises `OverflowError` where the vectorized path returns `-inf` (the P01-F04 defect on the sibling family), `InverseGaussianDistribution.log_density(1e308)` returns NaN, and `LogGaussianDistribution.log_density(+-inf)` raises where `seq_log_density` returns `-inf`

**Surface:** `GeneralizedGaussianDistribution.log_density`, `InverseGaussianDistribution.log_density`,
`LogGaussianDistribution.log_density` versus the same families' `seq_log_density`.

**Reproduction** (`probes/scalar_vs_vector.py`, `probes/encoder_refusals_in_mixtures.py` sections D/E):

```python
import numpy as np
from mixle.stats import GeneralizedGaussianDistribution as GG, InverseGaussianDistribution as IG, LogGaussianDistribution as LG, GeneralizedGaussianEstimator
from mixle.inference import estimate
d = GG(0.0, 1.0, 2.0); e = d.dist_to_encoder()
print(np.asarray(d.seq_log_density(e.seq_encode([1e300]))))   # [-inf]
d.log_density(1e300)                                           # OverflowError: (34, 'Result too large')
GG(0.0, 1.0, 50.0).log_density(2e6)                            # OverflowError  (beta = 50 is GeneralizedGaussianEstimator's beta_bounds ceiling)
f = estimate(np.random.RandomState(0).normal(0, 1, 500), GeneralizedGaussianEstimator()); f.log_density(1e200)   # OverflowError on a fitted law
g = IG(1.0, 2.0); print(g.log_density(1e308), np.asarray(g.seq_log_density(g.dist_to_encoder().seq_encode([1e308]))))   # nan [-inf]
print(g.log_density(1e300), np.asarray(g.seq_log_density(g.dist_to_encoder().seq_encode([1e300]))))                   # -inf [-1e+300]
h = LG(0.0, 1.0); print(np.asarray(h.seq_log_density(h.dist_to_encoder().seq_encode([float('inf')]))))   # [-inf]
h.log_density(float('inf'))   # UnscorableObservation: LogGaussianDistribution rejects infinite observations.
```

**Observed:** as in the comments (`probes/scalar_vs_vector.venv-full.txt`,
`probes/encoder_refusals_in_mixtures.venv-full.txt`, `probes/ledger_replay.venv-full.txt`; identical on
`venv-base`). GeneralizedGaussian and InverseGaussian behave the same on 0.8.1
(`probes/scalar_vs_vector.c081.txt`); LogGaussian's vectorized path raised on 0.8.1 ("requires support
x in (0,inf)") and now returns `-inf` while the scalar path still raises, so the two routes were made
to disagree by the 0.8.2 change.

**Expected:** `-inf` from every route for an observation of vanishing density, as P01-F04 established
for `WeibullDistribution` and R05-F06 for the vectorized paths; the scalar and vectorized routes of one
family agree on every input.

**Notes:** GeneralizedGaussian's scalar path evaluates `(|x - mu| / alpha) ** beta` in Python floats
(`OverflowError` past roughly `alpha * 1e(308/beta)`), the vectorized path in numpy (overflow -> inf ->
`-inf`). Any scalar scoring loop over a fitted GeneralizedGaussian crashes on a large value instead of
scoring it `-inf` -- the shape P01-F04 repaired for Weibull. Repairs: P01-F04, R05-F06.

### Q01-F06 -- minor -- the engine route `backend_seq_log_density` returns NaN (and `+inf` for LogSeries at 0) for out-of-support rows the encoders now admit, where `log_density` and `seq_log_density` return `-inf`

**Surface:** `backend_seq_log_density(encoded, NUMPY_ENGINE)` on `BetaDistribution` (x < 0, x > 1, +-inf),
`GammaDistribution` (+inf), `InverseGammaDistribution` (x <= 0, -inf), `LogGaussianDistribution` (x <= 0,
-inf), `LogSeriesDistribution` (x = -1 -> nan, x = 0 -> +inf), `SkewNormalDistribution` (1e300),
`WeibullDistribution` (+inf); reachable through the `mixle.inference.jit` scorer.

**Reproduction** (`probes/scalar_vs_vector.py`):

```python
import numpy as np
from mixle.stats import BetaDistribution, LogSeriesDistribution
from mixle.engines.numpy_engine import NUMPY_ENGINE
d = BetaDistribution(2.0, 3.0); e = d.dist_to_encoder().seq_encode([0.5, -1.0, 2.5])
print(np.asarray(d.seq_log_density(e)))                        # [ 0.47  -inf  -inf]
print(np.asarray(d.backend_seq_log_density(e, NUMPY_ENGINE)))  # [ 0.47   nan   nan]
l = LogSeriesDistribution(0.5); e = l.dist_to_encoder().seq_encode([1, 0, -1])
print(np.asarray(l.backend_seq_log_density(e, NUMPY_ENGINE)))  # [-0.3266  inf  nan]
```

**Observed:** `probes/scalar_vs_vector.venv-full.txt` (identical in `venv-base`), the rows flagged
`backend!=scalar` for Beta, Gamma, InverseGamma, LogGaussian, LogSeries, SkewNormal and Weibull. On 0.8.1
the encoders refused these rows so the route was unreachable; the P02-F03 encoder change admits them and
only the `seq_log_density` scorers were given the `-inf` rule.

**Expected:** `-inf`, the same as the other two routes (R05-F06's rule for the vectorized path).

**Notes:** no library caller besides `mixle/inference/jit.py` (`backend_seq_log_density` from
`mixle.stats.compute.backend`), so `optimize`/`estimate` never see it; the `+inf` for LogSeries at
`x = 0` is nevertheless a wrong number from a public capability method (`capability.py` lists
`backend_seq_log_density()`). Repairs: P02-F03, R05-F06.

### Q01-F07 -- minor -- `entropy()` crashes at extreme but constructible parameters (P01-F10's shape): NegativeBinomial with `p <= 1e-160` and `Poisson(1e300)`

**Surface:** `NegativeBinomialDistribution.entropy`, `PoissonDistribution.entropy`.

**Reproduction** (`probes/entropy_accuracy.py`):

```python
from mixle.stats import NegativeBinomialDistribution as NB, PoissonDistribution
NB(1.0, 1e-150).entropy()             # 346.387763949  (matches the closed form)
NB(1.0, 1e-160).entropy()             # OverflowError: cannot convert float infinity to integer
NB(3.0, 1e-300).entropy()             # ZeroDivisionError: float division by zero  (mean() is a finite 3e300)
PoissonDistribution(1e300).entropy()  # OverflowError: (34, 'Result too large')
```

**Observed:** `probes/entropy_accuracy.venv-full.txt`: every other case checked agrees with an exact
float64 block sum, the `r = 1` closed form or the Gamma-limit reference to <= 4.5e-7 (most to 1e-9 or
better), across the Poisson asymptotic switch at `lam = 1e3`, the negative binomial Gaussian-limit and
blocked+tail branches, and `r = 1` down to `p = 1e-150`; the constructors accept the crashing parameters.
Not compared with 0.8.1, whose entropy at these supports is the P01-F02 memory bomb.

**Expected:** the limit or a finite number, as P01-F10 established for `mean`/`variance`.

**Notes:** the summation cap is computed as an integer from `mean + k*sd`, which is `inf` at these
parameters. Repairs: P01-F02, P01-F10.

### Q01-F08 -- minor -- `GeneralizedGaussianDistribution.quantile` returns -+inf for `q < 5.6e-17` or `q > 1 - 1.1e-16` and loses digits near those limits, because it inverts through `2|q - 1/2|`

**Surface:** `GeneralizedGaussianDistribution.quantile`.

**Reproduction** (`probes/confirm_quantile_tails.py`):

```python
from mixle.stats import GeneralizedGaussianDistribution, GaussianDistribution
g = GeneralizedGaussianDistribution(0.0, 1.0, 2.0)               # a Normal(0, sigma = 1/sqrt(2))
print(g.quantile(1e-16), g.quantile(1e-17), g.quantile(1e-20))   # -5.805018683193453 -inf -inf
print(GaussianDistribution(0.0, 0.5).quantile(1e-17))            # -6.006018786764246
```

**Observed:** `probes/confirm_quantile_tails.venv-full.txt` (scipy `norm.ppf(scale=1/sqrt(2))`: -6.006 at
1e-17, -5.814 at 1e-16, -15.04 at 1e-100). Same on 0.8.1 (`probes/quantile_domain.c081.txt`, `q = 1e-300`).

**Expected:** a finite quantile, as the Gaussian route returns for the same `q`.

**Notes:** `generalized_gaussian.py:272-274`: `2 * abs(q - 0.5)` rounds to 1.0 below `q ~ 5.6e-17` and
`gammaincinv(1/beta, 1.0)` is `inf`. `StudentTDistribution(5).quantile(1e-300) = +inf` was also observed
but matches `scipy.stats.t.ppf` exactly (scipy's own limitation), so it is not reported. Repairs: none
(P01-F12 neighbourhood).

### Q01-F09 -- minor -- boolean observations are refused by `PoissonEstimator` with a message that misdescribes them ("negative, fractional, NaN, or infinite"); 0.8.1 accepted them as 0/1 counts

**Surface:** `PoissonEstimator` through `mixle.inference.estimate` (and `GeometricEstimator`/
`LogSeriesEstimator` via the shared `is_whole_number`; only Poisson was reproduced).

**Reproduction** (`probes/ledger_replay.py`, R05-F02 block):

```python
import numpy as np
from mixle.stats import PoissonEstimator
from mixle.inference import estimate
estimate([True, False, True], PoissonEstimator())
# ValueError: PoissonDistribution has support x in {0, 1, 2, ...}, but at least 1 observation(s) carrying weight are negative, fractional, NaN, or infinite ...
estimate(np.array([True, False, True]), PoissonEstimator())   # same
```

**Observed:** `probes/ledger_replay.venv-full.txt`, identical on `venv-base`; 0.8.1 returned
`PoissonDistribution(0.6666666666666666)` (`probes/ledger_replay.c081.txt`).

**Expected:** either accept bool as the integer subtype numpy treats it as (0.8.1's behaviour, and what
the R05-F02 repair says it does for "an integer typed as `int` or `numpy.integer`"), or refuse with a
message naming the actual cause (boolean dtype); the migration guide lists no such refusal.

**Notes:** a behaviour change relative to 0.8.1 that neither the CHANGELOG nor the migration guide
mentions; the message names four causes, none of which applies. Repairs: R05-F02, P01-F06.

## Attacks that did not break anything

All on `venv-full` and `venv-base` unless noted (`probes/ledger_replay.*.txt` unless noted); the base
venv's output is identical to the full venv's for every probe run on both, and `venv-nonumba` matched
on the one probe run there (`probes/encoder_refusals_in_mixtures.venv-nonumba.txt`).

- P01-F02: `Poisson(lam).entropy()` and `NegativeBinomial(lam, 0.5).entropy()` at `lam` = 1e8, 1e12, 1e15
  finish in < 0.01 s with a traced peak under 1.1 kB; accuracy against exact block sums, the `r = 1`
  closed form and the Gamma-limit reference is <= 4.5e-7 everywhere it was checked, including both sides
  of the `lam = 1e3` switch and the `skew <= 1e-2` branch (`probes/entropy_accuracy.venv-full.txt`).
- P01-F03: `estimate([2, 2, 2], Weibull)` discloses `shape-unresolvable(zero sample variance -> 1000)`;
  `[1e-300] * 5` adds `mean-floored(1e-300 -> 1e-12)` (0.8.1: `()`).
- P01-F04: `Weibull(1000, 1).log_density(10.0)` and `Weibull(1000, 1e-12).log_density(1.0)` return `-inf`
  (0.8.1: `OverflowError`); scalar, vectorized and backend routes agree except at `+inf` (Q01-F06).
- P01-F05..F09: every ledger reproduction (negative, NaN, +-inf, fractional, all-invalid, zero rows) is
  refused with the family-and-support message on both 0.8.2 venvs where 0.8.1 fitted the remaining rows
  or raised a bare conversion/parameter error; the floors are disclosed (`poisson-rate-floored(0 -> 1e-12)`,
  `geometric-p-clamped(1 -> 0.999999999999)`, `logseries-p-floored(sample mean 1 -> p=1e-12)`,
  `width-floored(0 -> 1e-08)`). The refusals hold through `optimize`, `Composite`, an HMM and a mixture
  with a Gaussian sibling (`probes/r05f09_variants.venv-full.txt`).
- P01-F10: `Weibull(1e-3, 1).mean/variance` -> `inf`, `Beta(1e-300, 1e-300).variance` -> 0.25,
  `LogSeries(1e-300).mean/variance` -> 1.0 / 5e-301, `SkewNormal(0, 1, 1e300).entropy` -> 0.7258 (0.8.1:
  `OverflowError`/`ZeroDivisionError`/NaN).
- P01-F11 and the Poisson-quantile scipy note: `Poisson(lam).quantile(0.5)` returns exactly `lam` at
  `lam` = 1e10, 1e12, 1e15 with `cdf(lam) = 0.5000000084 > 0.5`, on scipy 1.18.1 (0.8.1: NaN at 1e12 and
  1e15); `quantile(0.99)` finite on all three.
- P01-F12: `q = 0` and `q = 1` (as `int`, `float`, `np.float32`, 0-d array) return the support bounds on
  all 31 quantile-bearing families (Gumbel +-inf, discrete families their minimum/`inf`, `GEV(0,1,0.1)`
  its finite lower endpoint -10, Uniform its bounds); a 1-D array or list `q` raises `TypeError` on all
  31 -- the scalar-only contract, unchanged from 0.8.1; a string `"0.5"` is accepted by the 15 guarded
  families (`float()` coercion) and raises a numpy `UFuncTypeError` on the others
  (`probes/quantile_domain.venv-full.txt`).
- P01-F13 / P06-F08: an `(n, 1)` column into `estimate`/`optimize` with a Gaussian estimator is refused
  by the message naming the estimator, the row shape and `numpy.ravel` (0.8.1: numpy's `TypeError`/
  shape error); `optimize(col, max_its=2)` with no estimator still returns a `SequenceDistribution`.
- R05-F02: integral `float32`, `float16` and `np.float32`-list values are accepted by Poisson, Geometric
  and LogSeries.
- R05-F05 on its own route: `ExponentialAccumulator.seq_update` with weights `[1, 1, 0]` and an `inf`
  third row returns `(2.0, 3.0)` (0.8.1: `(2.0, nan)`); `optimize([1, 2, 3, inf], Exponential)` gives the
  support message (0.8.1: "requires beta > 0"); the regression test's ten families all pass on
  `seq_update`; `seq_initialize` masks too (`probes/confirm_initialize_zero_weight.venv-full.txt`).
- R05-F06: `Weibull(2, 1).log_density(inf)` and `seq_log_density([0.5, inf])` -> `-inf` (0.8.1: NaN).
- R05-F07 on the 15 guarded families: Poisson, Gaussian, LogSeries (now naming itself) and Bernoulli
  (no longer answering a support point) refuse `-0.1`, `1.5` and NaN identically.
- R05-F09 on the all-rows-unscorable batch: the diagnosis names the count and what to check.
- Migration guide section 1 ("Univariate estimators refuse observations outside their support"): true
  for the five estimators on the `estimate` and `optimize` routes; its "a zero-weight row is still
  exempt" is true on the vectorized route only (Q01-F03).
- Scalar vs vectorized scoring at edge observations on every family (`probes/scalar_vs_vector.*.txt`):
  the remaining disagreements are the ones 0.8.1 already had -- the count and positive-support families'
  scalar `log_density` scores NaN (and out-of-support values) as `-inf` while the vectorized path refuses
  them (Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial, LogSeries, Skellam, BetaBinomial,
  IntegerUniformSpike, HalfNormal, Rayleigh, Beta, Gamma, InverseGamma, InverseGaussian, Tweedie;
  `Exponential.log_density(nan)` raises `UnscorableObservation` where the encoder raises `ValueError`).
  These are refusal-style differences, not wrong numbers; the ones that produce a crash or a NaN are in
  Q01-F05. `Skellam(2, 1)` at `x = 1e6`: backend `-1.21e7`, scalar/vectorized `-inf` -- the backend is
  the more accurate one.
- Zero-weight rows on the vectorized route for every family (`probes/zero_weight.venv-full.txt`): only
  Logistic (+-inf) differs from the clean statistic (Q01-F03); encoder refusals of the row itself
  (Bernoulli, Binomial, BetaBinomial, EMG, Gaussian, ...) are the pre-existing "must be finite exact
  integers"/"requires support" checks and are the same on 0.8.1.
- Enumeration: `tutorials/enumeration.ipynb` and the three enumeration examples run to completion with
  outputs that state their own consistency (`rank((0, 0, 2)) -> 4`, `cumulative` agreeing with
  `rank()`'s own cumulative probability, `exact=True`; `rank(5) -> 17`, method `exact-head`), and the
  six enumeration test files pass in both venvs (`tests/pytest2.*.txt`).
- The whole assigned corpus (15 notebooks, 5 examples) exits 0 with no numeric change against the
  stored outputs.

## What was not covered

- Serialization round trips (`to_json`/`from_json`, `dump_models`/`load_models`, pickle) of fitted
  univariate objects: the original reviewer never reached them and they were not started here.
- Direct attacks on the enumeration surfaces (`enumerator()` on empty or degenerate supports, huge `k`,
  ties in `rank`/`cumulative`): only the notebook, the three examples and the six test files were run.
- `venv-nonumba` for every probe but one (base and full were identical on all of them, which suggests
  no numba kernel is involved in these paths, but it was not checked).
- The 0.8.1 side of the entropy extremes (Q01-F07): 0.8.1's entropy at those supports is the P01-F02
  memory bomb, so it was deliberately not run.
- The trail's last pending item -- which accumulators expose `supported_rows` and how mixture
  initialization consults it -- was not finished; Q01-F02 is established by probe, the exact code path
  is not cited.
- Weights (negative, huge), `rng` spellings, `keys`/`name` round trips; heterogeneous mixtures and HMMs
  with the count families of Q01-F02 (only `MixtureEstimator` was probed there).
- The regression tests were run only after loading the repo-root `conftest.py` as a plugin; whether the
  brief's copy-and-run recipe is meant to work without it was not pursued.

DONE 01 9 findings
