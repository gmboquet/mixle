# Adversarial review — PASS 01 of 10

- **Focus area:** univariate distributions and their estimators (continuous and discrete)
- **Candidate wheel:** `mixle-0.8.1-py3-none-any.whl`, sha256 `3190de824b710780422d898b38cbe154f708747bf668d1c359bf58720fdc333d`
- **Git tree:** `6040ea38` (branch `release/0.8.1`)
- **Version verification** (run from the pass work dir, outside any checkout):

```
$ <review-root>/candidate-081/venv/bin/python -c "import mixle, importlib.metadata as m; print(m.version('mixle'), mixle.__path__[0])"
0.8.1 <review-root>/candidate-081/venv/lib/python3.12/site-packages/mixle
```

- **Work dir:** `<review-root>/reviews-081/pass-01/` (all scratch scripts and outputs referenced below live there; nothing outside it was modified).
- Note: this Mac has no `timeout`/`gtimeout` binary; a subprocess wrapper `pyt.py <secs> <cmd...>` in the work dir was used for every bounded run.

## Corpus executed on the candidate

Notebooks (`python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200`, executed copies in `nb_out/`):

| notebook | exit | wall |
|---|---|---|
| data_science/mle_and_sufficient_statistics.ipynb | 0 | 6 s |
| data_science/heavy_tails_and_survival.ipynb | 0 | 5 s |
| tutorials/distributions_and_combinators.ipynb | 0 | 8 s |
| tutorials/fitting_and_estimation.ipynb | 0 | 6 s |
| tutorials/bayesian_distributions.ipynb | 0 | 6 s |
| data_science/conjugate_and_nonparametric_bayes.ipynb | 0 | 16 s |
| data_science/density_estimation_mixtures_vs_kde.ipynb | 0 | 90 s |

Examples (`<venv>/bin/python <example>` from the work dir, logs in `ex_out/`):

| example | exit | wall |
|---|---|---|
| gallery_univariate_example.py | 0 | 10 s |
| quickstart_example.py | 0 | 18 s |
| frontier_family_showcase.py | 0 | 155 s |
| model_comparison_example.py | 0 | 23 s |
| heterogeneous_correctness_example.py | 0 | 14 s |

All ran clean. The printed notebook outputs in my area (Gaussian MLE = hand MLE, Wald vs bootstrap CIs, censored-Weibull recovery, Pareto max/median, memorylessness check, conjugate MAPs) were checked against their claims and none is contradicted by the candidate. One example output is misleading (F14).

## Findings

### P01-F01 (blocking) — `LogSeriesDistribution.cdf` returns negative and non-monotone values past the 0.8.1 term cap; `quantile` inherits the error

This is the code the 0.8.1 changelog headlines ("The log-series CDF, quantile, and entropy are finite computations in bounded memory ... block-summed with saturation detection, a closed-form tail bound"). Bounded memory and time are real (every call I timed at p up to `nextafter(1,0)` finished in < 1 s with flat RSS). The *values* are wrong once the scan hits `_SERIES_CAP = 50_000_000` terms.

Mechanism (`mixle/stats/univariate/discrete/logseries.py`): past the cap, `cdf` returns `min(1.0, 1.0 - self._tail_upper_bound(k))` where `_tail_upper_bound(k) = p^(k+1) / ((k+1)(1-p)) / norm`. That is a valid upper bound on the tail but its slack is a factor of roughly `1/(k(1-p))`, so for `k < 1/(1-p)` — the entire bulk of the distribution when p is within 1e-8 of 1 — it exceeds 1 and the CDF goes negative. There is no lower clamp. The docstring claims the bound "is exact to double rounding by then"; it is off by 6.8x at the cap for 1−p = 1e-9 and by 1823x for 1−p = 1e-12.

Reproduction (verified against an exact chunked summation of the pmf, `logseries_exact.py`, and against the log-series asymptotic `1 − E₁(k(1−p))/(−ln(1−p))`, which agreed with the exact sum to 1e-8 wherever both were computed):

```python
from mixle.stats import LogSeriesDistribution
d = LogSeriesDistribution(1 - 1e-9)
print(d.cdf(5*10**7))   # 0.880912  (exact 0.880912)
print(d.cdf(6*10**7))   # 0.242587  (exact 0.889240)  <- CDF decreased
print(d.cdf(10**8))     # 0.563371  (exact 0.912035)
print(d.quantile(0.9))  # 342580000.0 ; exact cdf at that k is 0.961, true q(0.9) ~ 7.07e7
d = LogSeriesDistribution(1 - 1e-12)
print(d.cdf(10**8))     # -360.8835917992805   (exact 0.687553)
print(d.cdf(10**9))     # -35.155805           (exact 0.770854)
d = LogSeriesDistribution(1 - 1e-10)
print(d.cdf(10**8))     # -3.299731            (exact 0.824635)
```

Extent (`ls_monotone.py`, 29-point log grid k ∈ [1e6, 1e13]): worst |cdf − truth| is 5.4e-6 at 1−p=1e-7, 2.8e-2 at 1e-8, 0.70 at 1e-9, 7.5 at 1e-10, 643 at 1e-12; non-monotone steps at every 1−p ≤ 1e-8. Everything below k = 5e7 is exact to 1e-16. `entropy()` is *not* affected in value (it matched Monte Carlo at every p tested, |z| ≤ 1.02, because the capped quantile only sets an over-generous summation limit).

Why blocking: a CDF below zero is a hard correctness defect, it is in the exact regime (p → 1, large k) that the 0.8.1 change was written for and describes in its docstring as exact, and it ships under the release's own headline entry. Fix direction: the tail needs the incomplete-gamma / `E₁` form (`E₁(k(1−p))/(−ln(1−p))` is accurate to < 1e-5 for k ≥ 1e3 in this regime) or the summation must continue, not a geometric bound; and `cdf` should at minimum be clamped to `[0, 1]` and monotone.

### P01-F02 (real) — `PoissonDistribution.entropy()` and `NegativeBinomialDistribution.entropy()` allocate one array up to the upper quantile (unbounded memory), the pattern 0.8.1 fixed for LogSeries only

`poisson.py:354-357` builds `np.arange(kmax+1)` with `kmax = lam + 40 sqrt(lam) + 40`; `negative_binomial.py:270-273` builds `np.arange(quantile(1-1e-16)+50)`. Peak allocation measured with `tracemalloc` (`poisson_big.py`) scales at ~32 bytes per unit of λ (or r):

| λ | Poisson entropy peak | NegBin(r=λ, 0.5) entropy peak |
|---|---|---|
| 1e5 | 3.6 MB | 3.3 MB |
| 1e6 | 33 MB | 32 MB |
| 1e7 | 324 MB | 321 MB |
| 1e8 | **3.21 GB** (1.4 s) | **3.20 GB** (1.9 s) |
| 1e12 | did not finish in 60 s (thrashing) | did not finish in 60 s |
| 1e15 | `MemoryError: Unable to allocate 7.11 PiB for an array with shape (1000001264911106,)` | MemoryError |

The returned values are correct where it completes (λ=1e8 gives 10.629 = ½ln(2πeλ)). The 0.8.1 changelog describes this exact failure mode ("allocated one array up to the quantile (226 GiB ...)") as the cause of every CI runner kill and fixes it for LogSeries; it remains in two sibling families. λ ≥ 1e9 will OOM a 16 GB machine.

```python
import tracemalloc; from mixle.stats import PoissonDistribution
tracemalloc.start(); PoissonDistribution(1e8).entropy(); print(tracemalloc.get_traced_memory()[1]/1e9, "GB")  # ~3.2
PoissonDistribution(1e15).entropy()   # MemoryError (7.11 PiB)
```

### P01-F03 (real) — `WeibullEstimator` clamps shape to `max_shape=1000` and scale to `min_scale` without disclosing it in `numerical_repairs()`

The 0.8.0/0.8.1 changelog repeatedly asserts a disclosure contract ("matching every other family's `numerical_repairs()` contract"; Gamma's shape ceiling, GPD's shape clamp, InverseGaussian's clamp and Bernoulli's boundary clamp were all fixed to disclose). Weibull's moment estimator (`weibull.py:_shape_from_moments`, lines 44-52 and 422-426) returns `max_shape` for zero-variance data and floors scale, and records nothing:

```python
from mixle.stats import WeibullEstimator; from mixle.inference import estimate
f = estimate([2.0, 2.0, 2.0], WeibullEstimator()); print(f, f.numerical_repairs())
# WeibullDistribution(1000.0, 2.001153119489987, ...) ()
f = estimate([1e-300]*5, WeibullEstimator()); print(f, f.numerical_repairs())
# WeibullDistribution(1000.0, 1.0005765597449936e-12, ...) ()   <- both clamps bound, silent
```

Compare `GammaEstimator` on the same input: `shape-ceiling-clamped(2.5e+14 -> 1e+12)` is reported.

### P01-F04 (real) — `WeibullDistribution.log_density` raises `OverflowError` where `seq_log_density` and scipy return `-inf`

Scalar and vectorized paths disagree on legitimately constructible parameters, and the scalar path escapes as a Python exception from `math.exp`/`pow`:

```python
from mixle.stats import WeibullDistribution
d = WeibullDistribution(1000.0, 1.0)
d.log_density(10.0)                                     # OverflowError: (34, 'Result too large')
d.seq_log_density(d.dist_to_encoder().seq_encode([10.0]))  # array([-inf])   (scipy: -inf)
WeibullDistribution(1000.0, 1e-12).log_density(1.0)     # OverflowError  (this is the F03 fit)
```

`d.log_density(2.0)` returns −1.07e301 correctly; only past float range does it raise.

### P01-F05 (real) — `ExponentialEstimator` silently ignores negative, NaN and −inf observations

No error, no `numerical_repairs()` entry; the fit is identical to the fit without the bad points, and all-negative data returns the default `beta=1.0`:

```python
from mixle.stats import ExponentialEstimator; from mixle.inference import estimate
estimate([1.0,2.0,3.0], ExponentialEstimator())                 # ExponentialDistribution(2.0)
estimate([1.0,2.0,3.0,-100.0], ExponentialEstimator())          # ExponentialDistribution(2.0)  <- -100 dropped
estimate([1.0,2.0,3.0,float('nan')], ExponentialEstimator())    # ExponentialDistribution(2.0)
estimate([1.0,2.0,3.0,float('-inf')], ExponentialEstimator())   # ExponentialDistribution(2.0)
estimate([-1.0,-2.0,-0.5], ExponentialEstimator())              # ExponentialDistribution(1.0)  <- default
estimate([1.0,2.0,3.0,float('inf')], ExponentialEstimator())    # ValueError (only +inf is caught)
```

Every other continuous family (Gamma, Weibull, Rayleigh, Pareto, HalfNormal, InverseGamma, ...) raises a family-named `ValueError` on the same inputs.

### P01-F06 (real) — `PoissonEstimator` averages negative counts into λ and accepts −inf; the λ floor is undisclosed

```python
from mixle.stats import PoissonEstimator; from mixle.inference import estimate
estimate([1,2,3,-5], PoissonEstimator())                 # PoissonDistribution(0.25)   <- (1+2+3-5)/4
estimate([1.0,2.0,3.0,float('-inf')], PoissonEstimator())  # PoissonDistribution(1e-12)
estimate([-1,-2], PoissonEstimator())                    # PoissonDistribution(1e-12)
estimate([0,0,0], PoissonEstimator())                    # PoissonDistribution(1e-12), numerical_repairs() == ()
estimate([1,2,float('nan')], PoissonEstimator())         # ValueError: PoissonDistribution requires lam > 0.
```

NaN and +inf are rejected only incidentally by the *parameter* validator (the message names `lam`, not the data). `BernoulliEstimator` discloses its identical floor as `bernoulli-p-clamped(0 -> 1e-12)`; Poisson does not.

### P01-F07 (real) — `GeometricEstimator` silently ignores NaN, −inf, negative and zero observations; accepts +inf; undisclosed p clamp

```python
from mixle.stats import GeometricEstimator; from mixle.inference import estimate
estimate([1,2,3], GeometricEstimator())                    # GeometricDistribution(0.5)
estimate([1.0,2.0,3.0,float('nan')], GeometricEstimator())   # GeometricDistribution(0.5)   <- NaN dropped
estimate([1.0,2.0,3.0,float('-inf')], GeometricEstimator())  # GeometricDistribution(0.5)
estimate([1,2,3,-7], GeometricEstimator())                 # GeometricDistribution(0.5)
estimate([1.0,2.0,3.0,float('inf')], GeometricEstimator())   # GeometricDistribution(1e-12)  <- no error
estimate([0,0,0], GeometricEstimator())                    # GeometricDistribution(0.5)   (support is k>=1: log_density(0) = -inf)
estimate([1,1,1], GeometricEstimator())                    # GeometricDistribution(0.999999999999), numerical_repairs() == ()
```

### P01-F08 (minor) — `UniformEstimator` silently drops NaN observations; width floor undisclosed

```python
estimate([0.0,1.0,float('nan')], UniformEstimator())   # UniformDistribution(0.0, 1.0)
estimate([5.0,5.0,5.0], UniformEstimator())            # UniformDistribution(4.999999995, 5.000000005), numerical_repairs() == ()
```

### P01-F09 (minor) — `LogSeriesEstimator` surfaces raw Python conversion errors on NaN/inf; p floor undisclosed

```python
estimate([1.0,2.0,float('nan')], LogSeriesEstimator())   # ValueError: cannot convert float NaN to integer
estimate([1.0,2.0,float('inf')], LogSeriesEstimator())   # OverflowError: cannot convert float infinity to integer
estimate([1,1,1], LogSeriesEstimator())                  # LogSeriesDistribution(1e-12), numerical_repairs() == ()
```

Every other family raises a `ValueError` naming the family and its support.

### P01-F10 (minor) — `mean()`/`variance()`/`entropy()` raise or return NaN at extreme but constructible parameters

```python
WeibullDistribution(1e-3, 1.0).mean()            # OverflowError: math range error   (scipy: inf)
BetaDistribution(1e-300, 1e-300).variance()      # ZeroDivisionError                  (scipy: nan)
LogSeriesDistribution(1e-300).mean()             # ZeroDivisionError                  (scipy: 1.0; true limit 1)
LogSeriesDistribution(1e-300).variance()         # ZeroDivisionError                  (true limit 0)
SkewNormalDistribution(0,1,1e300).entropy()      # nan   (0.72579 at shape 1e10; scipy 0.72579)
```

### P01-F11 (minor) — `PoissonDistribution.quantile` silently returns `nan` for λ ≳ 1e12

Delegates to `scipy.stats.poisson.ppf`, which returns nan there; `cdf` at the same λ works. `quantile(0.5)` is `1e10` at λ=1e10 and `nan` at λ=1e12, 1e14, 1e15, with no error, while `quantile(0.99)` still returns a number. (scipy parity, but a silent nan from a method whose sibling works.)

### P01-F12 (minor) — inconsistent `quantile` boundary conventions

`GumbelDistribution.quantile(0.0)` / `quantile(1.0)` raise `ValueError`; every other continuous family returns `-inf`/`inf` or its support bound. `PoissonDistribution`, `BinomialDistribution`, `NegativeBinomialDistribution` return `quantile(0.0) == -1.0`, outside the support (scipy convention), whereas `GeometricDistribution`, `LogSeriesDistribution`, `BernoulliDistribution` return their support minimum.

### P01-F13 (minor) — an `(n, 1)` column array into any univariate estimator fails with an opaque `TypeError`

`estimate(np.random.rand(20, 1), GaussianEstimator())` → `TypeError: only 0-dimensional arrays can be converted to Python scalars`, identically for all 30 families tested. The error names neither the family nor the expected shape. A flat array, list, `pd.Series`, `float32` array, `int64` array and a list of numpy scalars all work.

### P01-F14 (docs) — `gallery_univariate_example.py` presents a biased Binomial fit as parameter recovery with no remark

The example's docstring says it prints "the true vs. recovered parameters"; its Binomial row prints

```
Binomial
  true: BinomialDistribution(p=0.4, n=10, ...)
  fit : BinomialDistribution(p=0.444022, n=9, min_val=0, ...)
```

with `BinomialEstimator(min_val=0)` and no `max_val`: n is taken as the sample maximum (P(X=10)=1e-4, so no 10 among 5000 draws), and p = mean/9. Every other row recovers within ~2%; this one is 11% off in p and wrong in n, and the output offers no explanation. Either pass `max_val=10` or annotate the row. The example's output is pinned byte-for-byte by the reproduction bundle, so the pinned output carries the wrong fit.

## Attacks that did not break anything

- **Sample → fit recovery** for 32 family instances at n=20000 (`consistency.py`): every estimator recovered its generating parameters within sampling error, `numerical_repairs()` empty on all.
- **`log_density` vs scipy** on 200 own samples per family (Gaussian, LogGaussian, Gamma, Beta, Weibull, Rayleigh, Laplace, Logistic, StudentT, Pareto, Uniform, Gumbel, GPD ξ=±, GEV, HalfNormal, InverseGamma, InverseGaussian, Nakagami, Rician, SkewNormal, GeneralizedGaussian, EMG, Poisson, Binomial, Bernoulli, LogSeries p=0.5/0.99, BetaBinomial): max abs diff ≤ 2.2e-15.
- **`seq_log_density` vs `log_density`** on every family: ≤ 8.9e-16.
- **`cdf(quantile(q)) == q`** for q ∈ {1e-6 … 1−1e-6} on every family with both methods; `quantile` vs `scipy.ppf` and `cdf` vs `scipy.cdf`: ≤ 6.3e-14. (Round-trips *also* pass in the F01 regime, because the wrong CDF is self-consistent with the wrong quantile — the defect is only visible against an exact sum.)
- **Entropy vs Monte Carlo** on every family with `entropy()`: within 4 SE. LogSeries entropy at p ∈ {0.9, 0.99, 1−1e-6, 1−1e-9, 1−1e-12}: |z| ≤ 1.02 (note scipy's own `logser(0.99).entropy()` = 2.62 is scipy's truncation; mixle's 3.661 matches Monte Carlo 3.663).
- **LogSeries p→1 time/memory** (the 0.8.1 headline claim): entropy, quantile at q ∈ {0.5, 1−1e-9, 1−1e-16, 1.0}, cdf at 1e6 and 1e18, log_density(1e12), mean, variance, 10000-sample draw, at p ∈ {1−1e-6, 1−1e-9, 1−1e-12, 1−1e-15, nextafter(1,0)}: every call < 0.9 s, RSS flat at the import baseline. Bounded, as claimed — only the values fail (F01).
- **Weight-before-square overflow / anchored-location claims**: Gaussian, LogGaussian, Logistic, StudentT, Gumbel, GPD, Poisson, Exponential, Uniform, Weibull fits on data at 1e15 and 1e15+offset and at 1e-300: all finite, variance correct (Gaussian at 1e15 offset: 1e-8 floor disclosed; Weibull at 1e15: shape 1.6487 recovered).
- **GPD shape clamp disclosure (0.8.1 claim)**: `GeneralizedParetoEstimator` on data at 1e15+offset reports `shape-clamped(moment estimate -2.07e+29 -> xi_min=-10)`; on zero-variance data reports `variance-floored(...)`. Verified.
- **Empty data, single observation, all-identical data** on all 30 estimators: no crash; Gaussian/LogGaussian/Logistic/StudentT/Gumbel/GPD/InverseGaussian/Gamma/Bernoulli disclose their floors via `numerical_repairs()` (F03/F06/F07/F08/F09 are the families that do not).
- **NaN / ±inf / negative / zero observations** on all 30 estimators: 25 families raise a family-named `ValueError` (the exceptions are F05–F09).
- **Float observations into Binomial, NegativeBinomial, Bernoulli, BetaBinomial**: rejected with "must be finite exact integers". (Poisson, Geometric and LogSeries accept non-integer floats and fit the mean; their `log_density` of a non-integer is −inf — noted, not filed.)
- **Input containers**: list, `np.ndarray` float64/float32/int64, list of `np.float64`, `pd.Series`, list of bools — all accepted by every family.
- **Constructor validation**: sigma2/scale/shape/rate ≤ 0 or NaN, Beta a=0, Poisson λ≤0, NegBin p∉(0,1), Binomial p>1 or n<0, Bernoulli/LogSeries p∉(0,1), Uniform low≥high, Nakagami m<½, EMG/InverseGaussian λ=0 — all rejected with clear `ValueError`s. Integer and `np.float32` constructor arguments accepted and behave as floats.
- **Extreme parameters** (Gaussian σ²=1e-300 and μ=σ²=1e300, Gamma k=1e-300/1e15, Beta a=b=1e-300/1e15, NegBin p=1−1e-12 and r=1e-300, Binomial n=0/p=0/p=1/n=1e9, GPD ξ ∈ {−10, −1, 0, 1e-300, 0.5, 1}, StudentT df ∈ {1e-300, 1, 2, 1e300}, Pareto α ∈ {1, 2}, InvGamma α ∈ {0.5, 1.5}, GEV ξ ∈ {−1, 0, 1}, GenGaussian β ∈ {0.25, 50}, Rician ν ∈ {0, 1e3}, Geometric p=1): log_density/cdf/quantile/mean/variance/sampler all finite or correctly ±inf, matching scipy where scipy has the family. StudentT df ≤ 0.01 produces non-finite samples at exactly scipy's rate. GPD/GEV at |ξ| = 1e-300 return cdf(1.0)=1.0 — identical to scipy (both wrong; not filed).
- **GPD/GEV tiny-shape branch**: |ξ| ∈ [1e-9, 1e-8] uses the ξ=0 branch and is off from scipy by ≤ 6e-9 in log_density; below 1e-9 and above 1e-7 exact. Not filed.
- **NegativeBinomial cdf/quantile/log_density/sampler at r ∈ {1e10, 1e12, 1e15}**: all < 0.01 s and agree with scipy (only `entropy` is F02).
- **Geometric entropy/quantile/cdf at 1/p up to 1e6**: closed form, instant, matches scipy.
- **Notebook printed claims** in the seven executed notebooks (MLE = hand MLE, CI agreement, censored-Weibull recovery, Pareto tail statements, memorylessness, conjugate MAP values): none contradicted.
- **KDE / mixture / anchored-track / PPL / production claims** in the changelog were outside this pass's area and were not tested.
