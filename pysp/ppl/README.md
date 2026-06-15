# pysp.ppl — probabilistic programming, the pysparkplug way

An elegant, fast PPL surface over pysparkplug's distribution + sufficient-statistic
engine. **EM / variational-Bayes at the core, exact where structure allows, MCMC when
you want it** — and a one-line modeling surface.

```python
from pysp.ppl import Normal, free
m = Normal(free, free).fit(data)     # fit by EM
m.sample(100); m.log_prob(x)         # query
```

There is one rule: a model is plain pysparkplug construction where a parameter slot may
hold a **value** (fixed), the token **`free`** (estimate it), or **another distribution**
(make it random). That's the whole language. The 86 `pysp.stats` distributions are
untouched; this is a thin, optional dialect. Design: [../../notes/ppl-syntax-spec.md](../../notes/ppl-syntax-spec.md).

## Install / import

```python
from pysp.ppl import (
    Normal, Poisson, Gamma, Exponential, Bernoulli, Geometric, Beta, Categorical,
    StudentT, LogNormal, NegativeBinomial,        # heavy-tailed / positive / count
    Mix, Seq, Markov,                             # mixture / sequence / HMM
    free,
)
```

## Maximum likelihood (EM), in one line

```python
Normal(free, free).fit(data)          # estimate mean & sd
Poisson(free).fit(counts)
```

`fit` runs pysparkplug's EM. It threads the **parallel / distributed** backends straight
through — nothing else to change:

```python
Normal(free, free).fit(data, backend="mp", num_workers=8)     # multiprocess EM
# backend="mpi" / "dask" also supported (see pysp.utils.estimation.optimize)
```

## Regression (covariates with `Field`)

A linear predictor in the mean slot makes it a regression. Coefficients can be `free` (OLS)
or Normal priors (Bayesian / ridge — closed-form Gaussian posterior); `sigma` constant or
`free`. Pass the response positionally and covariates via `given=`.

```python
from pysp.ppl import Normal, Field, free

# OLS, multiple covariates
m = Normal(free*Field("x") + free*Field("z") + free, free).fit(y, given={"x": xs, "z": zs})
m.params                  # {'x': {...}, 'z': {...}, 'intercept': {...}};  m.result.sigma

# Bayesian: coefficient posteriors + prediction
a, b = Normal(0, 10), Normal(0, 10)
m = Normal(a*Field("x") + b, free).fit(y, given={"x": xs})
m.posterior(a)                          # posterior draws for the slope
m.result.predict({"x": [0, 1, 2]})      # predict at new covariates
```

## Mixtures — and EM "just works"

```python
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)
m.posterior(data)        # responsibilities (the E-step, exposed)
```

Mixture components are auto-initialized with **k-means++** seeding, so well-separated
clusters separate reliably — no manual init, no babysitting restarts.

## Bayesian inference

A *prior* is just a distribution in the slot — no special syntax.

### Exact, instant: conjugate posteriors (the VB ideal)

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
| `"vi"` | mean-field ADVI (reparameterized ELBO) | non-conjugate priors, fast approximate posterior |
| `"vmp"` | variational message passing (closed-form, ELBO) | conjugate-exponential models (e.g. Gaussian mean+precision) |
| `"mcmc"` | adaptive Metropolis (`pysp.utils.mcmc`) | full posterior, fast throughput |
| `"hmc"` | Hamiltonian MC, preconditioned | full posterior, best mixing |
| `"auto"` (default) | hierarchical → conjugate → map (if priors) → em | — |

## The result object

Whatever the method, the fitted RandomVariable answers the same verbs:

```python
m.params          # fitted params in the SAME parameterization you built with: {'mean':.., 'sd':..}
                  #   composites recurse: {'components': [{'mean':..,'sd':..}, ...], 'weights': [..]}
m.components      # composite sub-models as RandomVariables (query each: c.params, c.sample, ...)
m.dist            # the underlying pysp distribution (full original API — escape hatch)
m.sample(n)       # draw
m.log_prob(x)     # density (scalar or vectorized)
m.posterior(x)    # latent-state posterior (data) OR parameter posterior (name/handle)
m.predict(n)      # posterior-predictive draws (Bayesian) or plug-in predictive (point fit)
m.result          # inference metadata: posterior draws, summary, diagnostics
```

## Design guarantees

- **One immutable wrapper type** (`RandomVariable`); families are data, not subclasses.
- **All routing in one `lower()` function**; methods only dispatch.
- **Off the hot path**: lowering is cached; scoring/estimation runs the vectorized
  `seq_log_density` / `seq_update` engine underneath.
- **Optional**: every concrete distribution still works directly; `.dist` is always there.

See [../../notes/ppl-syntax-spec.md](../../notes/ppl-syntax-spec.md) for the full charter
and invariants.

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

Also: **linear regression** (`Field`, OLS + Bayesian), **Dirichlet-Categorical** VMP nodes,
**RV+RV convolution** (`x + y`), **event conditioning** (`.given`), and a **Bayesian mixture
via VBEM** (discrete per-datapoint latents).

Future: LDA / topic models in-graph, exact numerical (FFT) convolution for non-conjugate
continuous sums, analytic gradients for faster HMC.
```
