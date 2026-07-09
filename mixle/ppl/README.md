# mixle.ppl — probabilistic programming for Mixle distributions

`mixle.ppl` is a compact probabilistic-programming surface over Mixle's distribution and
sufficient-statistic engine. It uses EM and variational Bayes where the model structure supports them,
exact conjugate updates where available, and MCMC when full posterior sampling is required.

```python
from mixle.ppl import Normal, free
m = Normal(free, free).fit(data)     # fit by EM
m.sample(100); m.log_prob(x)         # query
```

The core rule is simple: a model is ordinary Mixle construction where a parameter slot may
hold a **value** (fixed), the token **`free`** (estimate it), or **another distribution**
(make it random). The `mixle.stats` distributions are untouched; this is a thin, optional dialect over the existing distribution and
estimator contracts.

## Install / import

```python
from mixle.ppl import (
    Normal, Poisson, Gamma, Exponential, Bernoulli, Geometric, Beta, Categorical,
    StudentT, LogNormal, NegativeBinomial, Dirichlet,   # heavy-tailed / positive / count / simplex
    MVN, DiagGaussian,                            # multivariate (vector data)
    Mix, Seq, Markov, LDA,                         # mixture / sequence / HMM / topic model
    Field, Graph, compare, free,                  # regression / VMP graph / model selection
)
```

## Maximum likelihood (EM), in one line

```python
Normal(free, free).fit(data)          # estimate mean & sd
Poisson(free).fit(counts)
```

`fit` runs mixle's EM. It threads the **parallel / distributed** backends straight
through — nothing else to change:

```python
Normal(free, free).fit(data, backend="mp", num_workers=8)     # multiprocess EM
# backend="mpi" / "dask" also supported (see mixle.utils.estimation.optimize)
```

## Regression & GLMs (covariates with `Field`)

A linear predictor in a parameter slot makes a regression; the **outer family sets the link**:

```python
from mixle.ppl import Normal, Bernoulli, Poisson, Field, free

Normal(free*Field("x") + free, free)    # linear regression  (identity link)
Bernoulli(free*Field("x") + free)       # logistic regression (logit link)
Poisson(free*Field("x") + free)         # Poisson regression  (log link)

# heteroskedastic: a linear predictor in the *scale* slot (log link) -> location-scale regression
Normal(free*Field("x") + free, free*Field("x") + free)   # mean AND log-sd vary with x
LogNormal(free, free*Field("z") + free)                  # positive, right-skewed; dispersion ~ z
```

Coefficients are `free` (MLE) or Normal priors (Bayesian / ridge — MAP). Fit with the
response positionally and covariates via `given=`. Homoskedastic GLMs fit by IRLS (Fisher
scoring); a scale-slot linear predictor switches to a location-scale fit (separate mean and
log-scale coefficients, `result.coefficients` / `result.scale_coefficients`,
`result.predict(...)` returns `loc`/`scale`).

```python
m = Bernoulli(free*Field("x") + free*Field("z") + free).fit(y, given={"x": xs, "z": zs})
m.params                                # {'x': {...}, 'z': {...}, 'intercept': {...}}
m.result.predict({"x": [0], "z": [0]})  # probability (through the link)

a, b = Normal(0, 10), Normal(0, 10)     # Bayesian linear regression
m = Normal(a*Field("x") + b, free).fit(y, given={"x": xs})
m.posterior(a)                          # slope posterior draws
m.result.predict({"x": [0, 1, 2]})      # predicted means
```

Mixed-effects: a `Group("g")` term adds a per-group random intercept (lme4's `(1|g)`),
fit by a linear-mixed-model EM:

```python
m = Normal(free*Field("x") + free + Group("subject"), free).fit(
        y, given={"x": xs, "subject": ids})
m.result.coefficients   # fixed effects;  m.result.tau / .sigma — variance components
m.result.group_effects  # per-group random intercepts (BLUPs)
```

## Mixtures With Automatic EM Initialization

```python
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)
m.posterior(data)        # responsibilities (the E-step, exposed)
```

Mixture components are auto-initialized with **k-means++** seeding, so well-separated
clusters usually separate without manual initialization.

Other structured models, one line each:

```python
Markov(Normal(free, free), states=2).fit(sequences)   # 2-state Gaussian HMM
Seq(Normal(free, free)).fit(list_of_sequences)        # iid sequence model
LDA(num_topics=10, vocab_size=5000).fit(docs)         # topic model; docs are (word_id, count) bags
LocalLevel().fit(timeseries)                          # state space: trend smoothing (Kalman/RTS)
AR1().fit(timeseries)                                 # AR(1) + noise; estimates phi, forecasts
```

## Bayesian inference

A *prior* is just a distribution in the slot — no special syntax.

### Exact conjugate posteriors

When the prior is conjugate to the likelihood, `fit` returns the **closed-form**
posterior — exact, no iteration, no sampling:

```python
mu = Normal(0, 10, name="mu")
m  = Normal(mu, 2.0).fit(data)        # known sd -> conjugate Normal-Normal (auto)
m.dist.mu                             # posterior mean
m.result.summary()                    # {'mu': {'mean':..., 'posterior':'Normal', 'hyper':{...}}}
m.posterior("mu")                     # exact posterior draws

Poisson(Gamma(2, 1, name="rate")).fit(counts)     # Poisson-Gamma
Bernoulli(Beta(1, 1, name="p")).fit(flips)        # Bernoulli-Beta
Exponential(Gamma(2, 1, name="rate")).fit(waits)  # Exponential-Gamma
```

### MAP

```python
Normal(Normal(0, 10), free).fit(data, how="map")   # maximize the joint with priors
```

### MCMC / HMC (when you want the full posterior)

```python
mu = Normal(0, 10, name="mu")
post = Normal(mu, free).fit(data, how="mcmc", draws=2000, burn=1000)   # adaptive RW
post = Normal(mu, free).fit(data, how="hmc",  draws=1000, burn=500)    # Hamiltonian MC
post.dist.mu                          # posterior mean
post.result.acceptance_rate           # RW targets ~0.44; HMC ~0.8-1.0
post.posterior("mu")                  # posterior draws (by name, handle, or index)
post.result.summary()                 # mean/std/95% CI per parameter
post.result.raw.effective_sample_size()
```

RW Metropolis uses an adaptive, data-informed scale (high throughput); HMC is
mass-preconditioned to the posterior scale and achieves near-perfect mixing
(ESS ≈ draws). See [BENCHMARKS.md](BENCHMARKS.md).

## Algebra & conditioning

```python
3 * Normal(0, 1) + 1          # affine -> N(1, 9)        (exact)
Normal(0, 1).exp()            # lognormal               (exact, Jacobian)
Normal(0, 1) + Normal(5, 2)   # convolution -> N(5, √5) (exact for conjugate-stable
Poisson(2) + Poisson(3)       #   pairs: Normal, Poisson, Gamma; KDE fallback otherwise)
Normal(5, 1) - Normal(2, 1)   # difference -> N(3, √2)

x = Normal(0, 1)
q = x.given(x > 0)            # condition on an event (truncation)
q.sample(1000)               # rejection sampling
q.log_prob(1.0)              # renormalized density
q.prob_of_event()            # P(x > 0)
```

## Constraints & inequalities among random variables

Comparisons build a `Constraint` — `x > 0` (RV vs constant), `a < b` (RV vs RV), or
`2*a - b >= 1` (linear/transformed expressions on either side) — combined with `&` `|` `~`.
The *same* constraint object works two ways.

**Generative — `constrain(...)`** conditions several RVs on a relation among them:

```python
a, b, c = Normal(0, 1, name="a"), Normal(0, 1, name="b"), Normal(0, 1, name="c")
ordered = constrain(a < b, b < c)     # the ordered triple (combine with & under the hood)
ordered.sample(1000)                  # (1000, 3) array, columns in ordered.columns
ordered.mean()                        # per-variable means
ordered.prob()                        # P(a < b < c)  ->  ~1/6
ordered.log_prob(x)                   # renormalized joint density on the region

tails = constrain((x < -1) | (x > 1)) # boolean combinators; ~c negates
```

(Python's chained `a < b < c` is intentionally rejected — use `(a < b) & (b < c)`.)

**Differential / shape constraints** act on a vector RV (a discretized function) via finite
differences — monotonicity (first difference), curvature (second difference), bounded variation:

```python
v = MVN(20, name="v")
constrain(increasing(v))     # constrain(decreasing(v)) / monotone(v)
constrain(convex(v))         # constrain(concave(v))     -- second difference
constrain(lipschitz(v, 0.5)) # |v[i+1]-v[i]| <= 0.5      -- bounded variation
```

Each carries a continuous residual, so it also feeds the soft-penalty inference path.

**Inference — `fit(..., constraints=...)`** restricts the feasible parameter region. The
constrained variables are the model's (named) priors; `map`/`mcmc`/`ensemble` honor a hard
truncation of the posterior, e.g. an identifiability ordering:

```python
a = Normal(2, 5, name="alpha");  b = Normal(5, 5, name="beta")
Beta(a, b).fit(data, constraints=a < b)            # auto -> constrained MAP
Beta(a, b).fit(data, how="ensemble", constraints=a < b)   # full posterior on the region
```

## Vector / matrix parameters

Structural parameters of a combinator are inferable, each reparameterized to its natural
manifold (no explicit constraint or Jacobian needed — the transform is exact):

```python
# mixture weights / HMM transition matrix (simplex via the Gamma representation)
Mix([...], free).fit(data, how="ensemble")
Markov([Normal(m0,1), Normal(m1,1)], transitions=free, initial=free).fit(seqs, how="ensemble")

# MVN mean vector + full covariance (covariance = L Lᵀ, SPD by construction)
MVN(d, mean=free, cov=free).fit(X, how="ensemble")
DiagGaussian(d, mean=free, var=free).fit(X, how="map")      # diagonal variances (positive)

# ordered mean vector (increasing by construction — identifiability, no rejection)
MVN(d, mean=ordered, cov=free).fit(X, how="ensemble")
```

Manifolds covered: real / positive / unit vectors, the **simplex** (and rows of a stochastic
matrix), **SPD covariance** (Cholesky), and **ordered** vectors.

## Bayesian mixture via VBEM (discrete latents)

`Mix(...).fit(how="vmp")` runs variational Bayes for a Gaussian mixture — per-datapoint
categorical responsibilities, Normal-prior means, Gamma-prior precisions, Dirichlet weights:

```python
m = Mix([Normal(free, free)] * 3).fit(data, how="vmp")
m.result.components          # [{'mean':.., 'sd':..}, ...]   (Bayesian component posteriors)
m.result.weights            # E[pi] (Dirichlet posterior mean)
m.result.responsibilities   # (N, K) soft assignments
```

## Variational message passing (`how="vmp"`)

A real conjugate-exponential VMP engine ([vmp.py](vmp.py)): exponential-family variational
nodes exchange natural-parameter messages under coordinate ascent, with a monotonically
increasing ELBO. `how="vmp"` **auto-builds the graph** from a nested Gaussian model — it
handles models the closed-form registry can't, including a Gaussian with **both unknown
mean and unknown precision** (mean-field `q(μ)q(τ)`) and **mean hyperpriors of any depth**:

```python
m = Normal(Normal(0, 10), Gamma(1, 1)).fit(data, how="vmp")          # unknown mean + precision
m = Normal(Normal(Normal(0, 100), 5.0), Gamma(1, 1)).fit(data, how="vmp")  # + mean hyperprior
m.params                 # {'mean': 4.95, 'sd': 1.96}
m.result.q_mu            # {'mean': 4.95, 'sd': 0.031}   (variational factor over μ)
m.result.q_tau           # {'shape':..., 'rate':..., 'mean': 0.26}   (factor over precision)
m.result.elbo_trace      # monotonically non-decreasing
```

Nodes (`GaussianNode`, `GammaNode`) and factors are reusable; new conjugate models are new
factors, not new engines — this is the general core the conjugate registry and the
hierarchical pairs are special cases of.

### Arbitrary DAGs + shared variables — `Graph`

`Graph` builds a conjugate-Gaussian factor graph from `observe(model, data)` calls. Nodes
are keyed by **object identity**, so the *same* `RandomVariable` handle used in multiple
positions becomes **one node that combines messages from every factor** (parameter tying /
shared latents). Priors that are themselves handles become parent nodes — hierarchies of
any depth.

```python
mu = Normal(0, 10)                              # one shared latent
fit = (Graph()
       .observe(Normal(mu, 1.0), data_a)        # factor A uses mu
       .observe(Normal(mu, 1.0), data_b)        # factor B uses the SAME mu
       .fit())
fit.posterior(mu)        # evidence from A and B combined; ELBO monotone

# deep hierarchy: grand-mean -> group means -> data, any depth
grand = Normal(0, 100)
g = Graph()
for gi in groups:
    mu_i = Normal(grand, 3.0)                   # each group mean shares `grand` as parent
    g.observe(Normal(mu_i, 1.0), gi)
g.fit().posterior(grand)

# discrete: Dirichlet-Categorical (shared simplex pools counts across factors)
pi = Dirichlet([1, 1, 1])
Graph().observe(Categorical(pi), labels_a).observe(Categorical(pi), labels_b).fit().posterior(pi)
```

The graph node library covers the **Gaussian-Gamma-Dirichlet** conjugate-exponential
families: Gaussian means, Gamma precisions, and Dirichlet simplexes.

## Hierarchical / random effects — `.each()`

Mark a prior per-group with `.each()`. That single change turns a global parameter into a
random effect; the data is a list of groups.

```python
# mu_i ~ Normal(m, tau) per group ;  y_ij ~ Normal(mu_i, sigma)
fit = Normal(Normal(0, 100).each(), free).fit(grouped_data)
fit.result.hyper          # {'m':..., 'tau':..., 'sigma':...}  (population)
fit.result.group_means    # per-group shrinkage posteriors

# the conjugate hierarchical core spans pairs, not just Normal:
Poisson(Gamma(1, 1).each()).fit(count_groups)       # Gamma-Poisson random rates
Bernoulli(Beta(1, 1).each()).fit(binary_groups)     # Beta-Bernoulli random probs
```

Inference is a conjugate E-step (exact per-group posteriors) + an empirical-Bayes population
M-step, dispatched by conjugate pair — the first slice of a general variational
message-passing core.

## `how=` routing

| `how` | engine | when |
|---|---|---|
| `"em"` | EM / MLE (`optimize`, parallel backends) | plain `free` models |
| `"conjugate"` | closed-form posterior | conjugate prior + known other params |
| `"hierarchical"` | conjugate VB/EM random effects | `.each()` group priors |
| `"map"` | maximize joint (scipy) | priors, point estimate |
| `"vi"` | ADVI — `family='meanfield'|'fullrank'`, tilted Renyi `alpha=`, `batch_size=` (SGVB) | non-conjugate priors, scalable approximate posterior |
| `"vmp"` | variational message passing (closed-form, ELBO) | conjugate-exponential models (e.g. Gaussian mean+precision) |
| `"mcmc"` | adaptive Metropolis (`mixle.utils.mcmc`) | full posterior, fast throughput |
| `"hmc"` | Hamiltonian MC, preconditioned (fixed step) | full posterior |
| `"nuts"` | No-U-Turn Sampler (auto-tuned HMC, dual-averaging) | correlated / higher-dim posteriors |
| `"ensemble"` | affine-invariant ensemble (Goodman & Weare) | low/medium-dim, highest ESS/sec |
| `"sample"` | **auto-picks the sampler** (ensemble low-dim, NUTS higher-dim) | just want the posterior |
| `"auto"` (default) | hierarchical → conjugate(/mixture) → map (if priors) → em | — |

You rarely need to name a sampler: `how="sample"` chooses one. Constraints also just work —
`fit(constraints=...)` auto-uses rejection for inequalities and a soft penalty for equalities /
ODE residuals (no `penalty=` needed), and you only add `name=` to a prior if you want to read it
back by name (constraints match by identity).

`map`/`mcmc`/`hmc`/`ensemble` work on **composite** models too (mixtures, sequences): the leaf
`free`/prior parameters are collected across the tree and a concrete model is rebuilt per
evaluation. Mixtures need an identifiability constraint (ordered component means) to break
label-switching — e.g. `Mix([Normal(m0, 1), Normal(m1, 1)]).fit(data, how="ensemble",
constraints=m0 < m1)`.

## The result object

Whatever the method, the fitted RandomVariable answers the same verbs:

```python
m.params          # fitted params in the SAME parameterization you built with: {'mean':.., 'sd':..}
                  #   composites recurse: {'components': [{'mean':..,'sd':..}, ...], 'weights': [..]}
m.components      # composite sub-models as RandomVariables (query each: c.params, c.sample, ...)
m.dist            # the underlying mixle distribution (full original API — escape hatch)
m.sample(n)       # draw
m.mean(); m.var() # moments (Monte-Carlo; works for any RV)
m.log_prob(x)     # density (scalar or vectorized)
m.posterior(x)    # latent-state posterior (data) OR parameter posterior (name/handle)
m.predict(n)      # posterior-predictive draws (Bayesian) or plug-in predictive (point fit)
m.waic(data)      # WAIC: Bayesian predictive accuracy {waic, elpd_waic, p_waic, se, ...}
m.loo(data)       # PSIS-LOO cross-validation {loo, elpd_loo, p_loo, se, khat_max, ...}
m.result          # inference metadata: posterior draws, summary, diagnostics
```

Model comparison spans plug-in and predictive criteria:

```python
compare([m1, m2], data, by="waic")   # 'aic' | 'bic' | 'loglik' | 'waic' | 'loo'
#   waic/loo integrate over posterior uncertainty (the modern Stan/ArviZ criteria);
#   rows sort best-first and report d_elpd (elpd difference from the best model).
```

## Design guarantees

- **One immutable wrapper type** (`RandomVariable`); families are data, not subclasses.
- **All routing in one `lower()` function**; methods only dispatch.
- **Off the hot path**: lowering is cached; scoring/estimation runs the vectorized
  `seq_log_density` / `seq_update` engine underneath.
- **Optional**: every concrete distribution still works directly; `.dist` is always there.

These guarantees keep the PPL surface inspectable: model declarations lower to
ordinary Mixle objects, and unsupported combinations should fail explicitly.

## Performance & execution stack

The PPL is a thin lowering layer — it inherits mixle's full execution stack rather
than reimplementing scoring. Nothing in the hot path is a Python per-element loop.

- **NumPy vectorization** — `log_prob`/`fit` run the vectorized `seq_log_density` /
  `seq_update` kernels (a 200k-point Gaussian EM fits in <20 ms).
- **Numba** — inherited through the distribution kernels mixle already JIT-compiles.
- **Torch engine** — `fit(..., engine=TorchEngine())` runs the E-step/scoring on the torch
  ComputeEngine (GPU-capable); verified to match the NumPy result.
- **Parallel / distributed EM** — `fit(..., backend="mp"|"mpi"|"dask", num_workers=…,
  comm=…, client=…)` threads straight through to `optimize`; pass an **RDD as `enc_data`**
  for the Spark/RDD path. `precision="float32"|"auto"` is also forwarded.

These apply to the **EM/MLE path** — i.e. all the standard models (scalar families,
mixtures, HMMs, sequences, LDA, MVN) via `.fit()` / `fit(how="em")`. The Bayesian/VB paths
(`conjugate`, `hierarchical`, `vmp`, `vi`, `mcmc`, `hmc`, regression, state-space) are
vectorized NumPy and single-machine: conjugate is one O(N) pass; hierarchical/VMP are
vectorized over groups; MCMC/HMC/VI score each step through the vectorized `seq_log_density`.
The distributed path is centered on the EM/MLE workflows listed above; use the single-machine
Bayesian routes when their posterior semantics are the primary requirement.

## Status

Implemented & tested: EM (parallel backends), mixtures (k-means++ init) + responsibilities,
HMM (`Markov`) and `Seq`, conjugate exact Bayes (Normal-Normal, Poisson-Gamma,
Bernoulli-Beta, Exponential-Gamma), hierarchical random-effects across conjugate pairs
(`.each()`: Normal-Normal, Gamma-Poisson, Beta-Bernoulli), MAP, **mean-field VI/ADVI**
(non-conjugate), parameter MCMC (adaptive RW) and **HMC** (preconditioned),
posterior-predictive (`predict`), posterior/diagnostics, RV algebra (`x.exp()`, affine),
11 scalar families. Benchmarked (see [BENCHMARKS.md](BENCHMARKS.md)): 17–37× faster than
the torch/Pyro-SVI approach for EM, ~600× for conjugate vs MCMC.

Also: a conjugate-exponential **VMP engine** (`how="vmp"` and `Graph`) with message-passing
nodes + monotone ELBO — arbitrary conjugate-Gaussian DAGs, deep hierarchies, and **shared
variable instances** (one handle reused across factors → one node combining all messages).

Also: **regression & GLMs** (`Field`: linear/OLS, Bayesian, logistic, Poisson) and
**mixed-effects** models (`Group` random intercepts, LMM EM), **mixtures & HMMs with any
emission family** (Gaussian / Poisson / Categorical / …), **multivariate Gaussian**
(`MVN`, `DiagGaussian`), **LDA** topic models, **Dirichlet-Categorical** VMP nodes,
**RV+RV convolution** (`x + y`), **event conditioning** (`.given`), **Bayesian mixture via
VBEM**, **moments** (`mean`/`var`), and **model comparison** (`log_likelihood`, `aic`/`bic`,
plus Bayesian predictive **WAIC** and **PSIS-LOO** via `m.waic`/`m.loo`/`compare(by="waic"|"loo")`).
12 scalar families + multivariate + 6 structured model types.

Future: LDA in-graph as VMP factors, exact (FFT) convolution for non-conjugate continuous
sums, analytic gradients for faster HMC.
