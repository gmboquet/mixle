<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/mixle_logo.png" alt="mixle" width="480"/>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2700%2B-brightgreen)

**mixle is a deep model architecture framework.** Neural networks — layers of ReLU units trained by
gradient descent — are one narrow, special-case model class. mixle builds, fits, and scores the
broader space: mixtures, hidden Markov models, probabilistic circuits (sum-product networks), graph
grammars, Bayesian hierarchies, dynamic relational models, and neural regressors, all composable
with each other under a single contract.

A single observation can be a tuple of a category, a real, a count sequence, a vector, a set, or a
tree. mixle models the whole record jointly and **chooses the inference algorithm from the model's
own structure** — closed-form conjugate, EM, MAP, variational, MCMC — locally on NumPy/Numba or
distributed across Spark, MPI, or multiple processes.

Compute is a first-class concern. The precision layer spans **fp1 through fp1024** — sub-byte packed
codes, float32/64, double-double extended precision, MPFR arbitrary precision — with automatic
data-aware allocation and a compiled transcendental-free forward path in the logarithmic number
system (LNS).

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Core concepts](#core-concepts) ·
[Compute & precision](#compute--precision) ·
[Distribution catalog](#distribution-catalog) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Frequentist & Bayesian](#frequentist--bayesian) · [Distributed & scale-out](#distributed--scale-out) ·
[Enumeration & ranking](#enumeration--ranking) · [Beyond fitting](#beyond-fitting) ·
[Examples](#examples) · [Tests](#tests) · [Maintainers & contributors](#maintainers--contributors) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). On PyPI as `mixle`; the import name is `mixle`.

```sh
pip install mixle          # base (numpy, scipy, mpmath): every distribution + local EM
pip install "mixle[all]"   # acceleration, scale-out, and connectors
```

The base install fits every model locally. Acceleration and scale-out are opt-in extras:

| Extra | Adds |
| --- | --- |
| `numba` | JIT-compiled hot paths (falls back to pure NumPy when absent) |
| `torch` | GPU / autograd engine |
| `spark` · `dask` · `mpi` | distributed estimation backends |
| `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `data` | data-source connectors |
| `gmpy2` | GMP-FFT big-integer multiply for count-DP ranking |
| `umap` | model-based UMAP embeddings |
| `sympy` · `sage` | symbolic / closed-form export |
| `grammar` | graph-grammar models (networkx) |

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

Each record here is a `(category, real, variable-length count sequence)`. Fit a two-component mixture
straight from a list of records:

```python
from mixle.stats import *
from mixle.inference import optimize

data = [
    ('a', -0.4, [5, 7]),       ('b', 4.9, [11, 9]),
    ('a',  0.2, [6, 5, 4]),    ('b', 5.3, [10, 12, 11]),
    ('a', -1.1, [4, 6]),       ('b', 4.5, [9, 10]),
    ('a',  0.7, [5, 5]),       ('b', 5.1, [12, 8]),
    ('a', -0.2, [7, 6, 5]),    ('b', 4.7, [9, 11]),
]

# The estimator mirrors the distribution's structure exactly.
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100)
model.sampler(seed=0).sample(3)   # draw new records from the fitted model
```

You don't have to spell the estimator out. `optimize` (and `fit`) also accept a **prototype
distribution** — its matching estimator is taken automatically — or just the data, from which an
estimator is inferred:

```python
proto = MixtureDistribution([GaussianDistribution(-1, 1), GaussianDistribution(1, 1)], [0.5, 0.5])
optimize(reals, proto)    # build the model's shape once, fit it directly
optimize(reals)           # or let mixle infer the estimator from the data
```

The same model in the shorter [`mixle.ppl`](#probabilistic-programming-mixleppl) dialect is a few lines.

## Core concepts

Each family is five cooperating pieces:

| Piece             | Role                                                                       |
| ----------------- | ------------------------------------------------------------------------- |
| `...Distribution` | parameters + `log_density(x)` and vectorized `seq_log_density(encoded)`    |
| `...Sampler`      | draw observations — `dist.sampler(seed).sample(size)`                      |
| `...Estimator`    | declares the model to fit; closed-form M-step in `estimate()`             |
| `...Accumulator`  | sufficient statistics for the E-step, mergeable across data partitions     |
| `...DataEncoder`  | packs raw Python records into arrays for the fast path                     |

`optimize(data, est)` (in `mixle.inference`) runs EM to convergence — vectorized locally, or
distributed via `backend=`. It also accepts a distribution **prototype** (`optimize(data, proto)`) or
nothing but the data (`optimize(data)`, which infers the estimator). Related entry points:

- `best_of` — multi-restart EM
- `StreamingEstimator` — online EM
- `fit_mle` / `fit_map` — autograd fitting with typed priors
- `mixle.utils.automatic.get_estimator(data)` — infer an estimator from raw data

Families live in `mixle.stats`; operations on them are grouped by concern:

- `mixle.inference` — fit: MLE / EM / MAP / conjugate / NUTS / VI / Fisher
- `mixle.enumeration` — rank / top-k / unranking
- `mixle.ops` — quantize / condition / marginalize / project
- `mixle.describe(x)` — report what any object supports

Drawing is a method, not a concern: `dist.sampler(seed).sample(n)`.

## Compute & precision

The compute layer is explicit and composable. Distributions own the likelihood math; **engines**
supply array ops, device, and numerical format — so the same model runs at different precision
levels or on different hardware without touching the model code.

### Precision spectrum: fp1 through fp1024

| Layer | What it does |
| --- | --- |
| **Sub-byte packing** (`mixle.engines.bitpacked`) | 1-bit binary and 2-bit ternary GEMM via packed `uint64` popcount |
| **Float32 / float64** | default paths; fused numba kernels on contiguous buffers |
| **Double-double extended** (`mixle.engines.extended`) | EFT via TwoSum / TwoProd / Veltkamp split; `dd_dot` with compiled FMA kernel (2.95× vs numpy) |
| **Codebook / VQ** (`mixle.engines.formats`) | 1-D k-means quantization with bit-packed codes; compress / decompress round-trips exactly |
| **MPFR arbitrary** | mpmath backend for arbitrary-mantissa computation |
| **Interval + affine error tracing** (`mixle.engines.error_tracing`) | outward-rounding interval arithmetic; `sum_error_bound`; auto-routes to double-double when float64 is not accurate enough |

Automatic allocation: `optimize(data, model, precision='minimal')` inspects data magnitude,
model fusibility, and leaf families, then picks the narrowest format that is provably safe:

```python
optimize(data, est, precision='minimal')    # float32 fused when safe, float64 otherwise
optimize(data, est, precision='auto')       # accumulate in float64, compute in float32
optimize(data, est, engine=TorchEngine(device="cuda", dtype="float32"))
```

### Logarithmic number system: transcendental-free integer forward passes

Probabilistic models naturally operate in log-space. mixle exploits this: quantize each
log-probability as an integer `k = round(log_p / step)` and all arithmetic becomes integer:

- **Products → integer ADD** (no `exp` or `log`)
- **Log-sum-exp → integer `max + LUT[|Δ|]`** (Gaussian-log lookup, no transcendentals)
- **All activations → `table[code]`** (sigmoid, tanh, GELU, SiLU, softplus — every nonlinearity
  is a LUT gather when the operand is quantized)

The compiled Cython kernel (`mixle/engines/_lns_kernel.pyx`) runs a pairwise tree fold.
On a 4096×50 000 language-model cross-entropy benchmark: **14.4× faster than float64 / 8.4×
faster than numpy LNS**.

```python
from mixle.engines.lns import LogNumberSystem
from mixle.engines.lns_nn import cross_entropy, SumProductCircuit

lns = LogNumberSystem(step=0.005)

# LM head / attention normalizer in integer log-space (~14x vs fp64)
ce_loss = cross_entropy(logits, targets, lns)

# Entire probabilistic circuit forward in integer log-space (product=add, sum=logadd)
circuit = SumProductCircuit(nodes)
log_probs = circuit.evaluate_lns(lns, leaf_log_values)
```

### Probabilistic circuits (sum-product networks)

`ProbabilisticCircuitDistribution` is a DAG of product nodes (log-space ADD), sum nodes
(LNS logadd), and typed leaves — with decomposability and smoothness enforced at construction.
Leaves can be any mixle distribution, so a circuit describes a structured generative model
over heterogeneous records. The whole forward pass runs in integer log-space when `lns_step=`
is set; EM uses circuit-flow soft counts.

```python
from mixle.stats.latent.probabilistic_circuit import leaf, prod, summ, ProbabilisticCircuitDistribution as PC

root = summ([
    prod([leaf(0, GaussianDistribution(-3, 1)), leaf(1, GaussianDistribution(-3, 1))]),
    prod([leaf(0, GaussianDistribution( 4, 1)), leaf(1, GaussianDistribution( 4, 1))]),
], [0.7, 0.3])

pc = PC(root, num_vars=2, lns_step=0.005)   # whole DAG scored in integer log-space
fit = optimize(data, pc.estimator(), prev_estimate=pc, max_its=40)
```

### Quantized nonlinearities

`mixle.engines.qlut` replaces any scalar nonlinearity with a nearest-code LUT and
linear-tail extrapolation for unbounded activations (1.5–8× vs real transcendental):

```python
from mixle.engines.qlut import quantized_activation, step_for_tolerance

gelu = quantized_activation('gelu', step=step_for_tolerance(1e-4))
out  = gelu(x)   # table[code] — no transcendental
```

## Distribution catalog

About 90 families in `mixle.stats`. The distinguishing feature: the **combinators model a whole
heterogeneous record as one distribution**. One observation under each:

| Model | One observation |
| --- | --- |
| `GaussianDistribution` / `PoissonDistribution` / `CategoricalDistribution` | `-0.31` / `7` / `'b'` |
| `MultivariateGaussianDistribution` | `[1.2, -0.4, 0.8]` |
| `CompositeDistribution((Cat, Gaussian, Poisson))` | `('a', -0.31, 7)` |
| `RecordDistribution({...})` | `{'country': 'US', 'age': 41, 'spend': 12.5}` |
| `SequenceDistribution(Poisson)` | `[5, 4, 6]` (variable length) |
| `OptionalDistribution(Gaussian)` | `-0.31` or `None` |
| `MixtureDistribution([...])` / `HiddenMarkovModelDistribution` | a component's shape, with the cluster / state latent |
| `ProbabilisticCircuitDistribution(root)` | a DAG of product / sum / leaf nodes, scored in integer log-space |

- **Univariate:** Gaussian, Student-t/Cauchy, Logistic, LogGaussian, Laplace, Uniform, Exponential,
  Gamma, Inverse Gamma/Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh, Pareto, Poisson,
  Bernoulli, Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet, categorical;
  multivariate/diagonal Gaussian, von Mises–Fisher, multivariate Student-t.
- **Combinators:** Composite (tuples), Record (named fields), Sequence, Optional (missing data),
  Transform, Conditional, Weighted.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), probabilistic
  circuits (sum-product networks), PCFGs, Markov chains, hidden associations, IBP, Pitman-Yor
  processes, Bernoulli sets.
- **Permutations & graphs:** Mallows / Plackett-Luce, matchings, spanning trees, random graphs
  (Erdős–Rényi, stochastic block, random dot-product), Spearman ranking, and graph grammars over
  networks (vertex-replacement / NLC and hyperedge-replacement) — `log_density` is the marginal
  likelihood, computed by parsing the graph back to the start symbol.
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet, SymmetricDirichlet)
  and variational Dirichlet-process / hierarchical-DP mixtures.

Estimator knobs (every family): `pseudo_count` (regularization) · `prior=` (conjugate; `None` is MLE) ·
`keys` (tie statistics across parts). One stem per family — `<Stem>Distribution`, `<Stem>Estimator`,
`<Stem>Sampler`, and so on.

## Probabilistic programming (`mixle.ppl`)

A concise dialect over the same distributions. **One rule:** any parameter slot is a value, the token
`free` (estimate it), or another distribution (a prior).

```python
from mixle.ppl import Normal, Mix, Markov, Field, free

Normal(0.0, 1.0)              # fixed parameters
Normal(free, free)            # estimate the mean and standard deviation
Normal(Normal(0, 10), 1.0)    # a prior on the mean (hierarchical)

data = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1]                          # reals from two clusters
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)
m.posterior(data)                                                 # per-point responsibilities

seqs = [[0.1, 5.1, 4.9], [4.8, 5.0], [0.0, 0.2]]                  # variable-length real sequences
Markov(Normal(free, free), states=2).fit(seqs)                    # 2-state Gaussian HMM

#   y[i] ~ Normal(b0 + b1*x[i] + b2*z[i], sd)   — a linear model
Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Mix([Normal(a, 1), Normal(b, 1)]).fit(data, constraints=a < b)    # ordered means break label-switching
```

- **`how=`** selects the route: `auto` reads the model's *structure* and picks the algorithm **family**
  — `conjugate | em | map | laplace | vi | vmp | mcmc | hmc | nuts | ensemble` — crossing the
  closed-form ↔ EM ↔ MAP ↔ hierarchical ↔ state-space boundary that other "auto" knobs stay inside.
- **See the choice before you fit:** `m.explain_fit()` (or `mixle.describe(m)`) reports the route `auto`
  will take, *why*, and its honest caveats; `how='laplace'` adds a cheap Gaussian posterior where
  `how='map'` gives only a point.
- **Constraints** among named variables are plain comparisons (combine with `& | ~`) and shape both
  inference and sampling.
- **Closed form:** for conjugate / exponential-family / mixture models, `.fit(...)` returns the exact
  posterior.
- **Constructors:** `Mix · Seq · Markov · LDA · MVN · DiagGaussian · LocalLevel · AR1 · Graph`;
  `compare([m1, m2], data)` ranks fitted models.

A slot is not limited to a single value/`free`/prior — it can be an **expression over latents**, and
latents can be coupled, indexed, or grouped. All of the below fit through the same `how=` routes:

```python
from mixle.ppl import Normal, Poisson, Field, Group, free, potential

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Normal(a + b, 1.0).fit(data)                       # deterministic expressions over latents
Normal(0.0, a.exp()).fit(data)                     #   …and transforms of them
Normal(a, 1.0).fit(data, potentials=potential(lambda av, bv: -0.5 * (av - bv) ** 2, a, b))  # custom log-factors

Normal(Normal(0, 5).each(), free).fit(groups)                # random effects: one list per group
Normal(Normal(0, 5).each(by="school"), free).fit(y, given={"school": labels})  #   …or a flat array + index
Poisson(free * Field("x") + Group("g")).fit(counts, given={"x": x, "g": g})    # non-Normal GLMM (PQL)

theta = free(8)                                              # a latent vector, indexed by data
Normal(theta[Field("g")], free).fit(y, given={"g": labels})  #   y[i] ~ Normal(theta[g[i]], sd)

Categorical(free).fit(labels)                                # the category set is inferred from the data
```

- **Custom factors:** `potential(fn, *vars)` adds an arbitrary `fn(*values)` log-term to the joint
  (the equivalent of Stan's `target +=`), and may introduce auxiliary latents.
- **Hierarchies & GLMMs:** `.each()` / `.each(by=...)` are random effects; `Group(...)` is the same in a
  regression predictor, for a Normal, Poisson, or Bernoulli response.
- **Diagnostics:** a multi-chain fit (`how="nuts", chains=4`) folds per-parameter R̂ and ESS straight
  into `m.result.summary()`; `waic` / `loo` / `compare` rank fitted models.

The dialect is thin — the `mixle.stats` classes underneath are untouched.

## Frequentist & Bayesian

The prior is the only switch — no prior is MLE; a conjugate `prior=` makes the same machinery Bayesian:

```python
from mixle.inference.priors import NormalGammaPrior

GaussianEstimator()                          # MLE
GaussianEstimator(prior=NormalGammaPrior())  # closed-form conjugate posterior — same optimize() call
```

- `optimize` / `fit` pick the objective from the model — likelihood, MAP, or variational ELBO.
- `BayesianStreamingEstimator` carries a posterior across batches; `mixle.stats.bayes` adds
  (hierarchical) Dirichlet-process mixtures.
- Gradient MAP with typed priors: `mixle.inference.gradient_fit.fit_map`
  (`NormalGammaPrior` / `DirichletPrior` / `MixturePrior`).
- **Honest densities:** `supports(x, ExactDensity)` / `describe(x)` flag when a model's `log_density`
  is a variational bound (e.g. LDA's per-document ELBO) rather than the exact `log p(x)`.

## Distributed & scale-out

All distributed backends share the same EM contract: workers ship fixed-size sufficient-statistic
payloads; the root folds with `combine()` and a tree-reduce. Scale-out is a `backend=` argument,
not a rewrite.

```python
optimize(rdd,  est, backend="spark")   # Spark RDD.treeReduce — no driver collect
optimize(data, est, backend="mpi")     # MPI comm.reduce, object-mode tree fold
optimize(data, est, backend="mp")      # multiprocessing.ProcessPoolExecutor
optimize(data, est, backend="dask")    # also: ray · lightning
```

- **Spark** (`mixle.inference.spark_executor`): shards data to an RDD, maps the E-step per shard,
  and `treeReduce(combine, depth)` on the root. Verified on PySpark 4.1 + Java 17.
- **MPI** (`mixle.inference.mpi_executor`): each rank E-steps its shard, then
  `comm.reduce(local, op=combine, root=0)`. Verified under `mpirun -n 3` (Open MPI 5 + mpi4py 4).
- **Multi-process** (`mixle.inference.heterogeneous_executor`): `ProcessPoolExecutor`; pickles only
  the shard + estimator, not the full dataset.
- The **planner** (`mixle.utils.parallel.planner`) turns a hardware budget into a memory-aware
  placement you compute once and reuse.
- **Symbolic export:** `SymbolicEngine` emits a model's closed-form log-density as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded:

```python
e = dist.enumerator()
e.top_k(5)        # the 5 most probable (value, log_prob)
e.top_p(0.95)     # smallest set covering 95% of the mass (the nucleus)
e.rank(value)     # how many values are strictly more probable than `value`
e.seek(10_000)    # the ~10,000th most probable value, by structural count-DP
```

- **Decomposable families** (Composite / Record / Sequence / MarkovChain): rank ↔ value is an exact
  count-DP at any depth (`count_dp_rank`, `count_dp_seek`); budget-bounded quantized indexes
  (`count_budget_index`) seek the most-probable region of an infinite support (the `gmpy2` extra uses
  GMP's FFT multiply for the big-integer convolution).
- **Non-decomposable families** (mixtures, HMMs): exact marginal rank is provably hard, so they return
  the Viterbi bound or a certified Monte-Carlo estimate (`density_rank`, with a standard error) — never
  a silent approximation.
- **Continuous families** realize the same operations through `cdf(x)` / `quantile(q)`.

## Beyond fitting

- **Inference** (`mixle.inference`): `mcmc` (MH / HMC / NUTS / VMP), `em` (hard, annealed, ECM,
  Monte-Carlo, variational, online, restart), `fisher` (geometry views), and the `Posterior` algebra —
  `posterior(model, data, over="latent"|"params"|"predictive")` returns one object you `sample` /
  `mean` / `interval`. An engine-agnostic facade runs NUTS/ADVI on any differentiable target with
  parallel chains (R̂ + pooled ESS).
- **Design & analysis of experiments** (`mixle.doe`): space-filling designs, GP Bayesian optimization,
  and the analysis half — Sobol/Morris sensitivity, uncertainty propagation, Kennedy-O'Hagan calibration.
- **Embeddings** (`mixle.utils.hvis`): model-based t-SNE / UMAP over per-record posteriors.
- **Supervised & non-iid models** (`mixle.models`): GP regression, neural regressors, random forests
  (a conditional `p(y | x)` leaf), random graphs, grammars, knowledge graphs.
- **MLOps** (`mixle.inference.production`): reproducible model artifacts (`fit_with_provenance` → a
  `Header` with config, data hash, model-hash lineage, convergence, timing, resources, env), drift
  detection + a `Monitor` (retrain-and-swap), and a versioned `Registry` + `Service` (scoring +
  activity logging). A container / Kubernetes serving layer lives in the separate
  [mixle-deploy](https://github.com/gmboquet/mixle-deploy) package.

## Examples

Self-contained scripts in [examples/](https://github.com/gmboquet/mixle/tree/main/examples):

```sh
cd examples
python gallery_univariate_example.py    # tour the scalar families
python gallery_structured_example.py    # mixtures / HMMs / LDA / probabilistic circuits
python ppl_example.py                   # the equation-style mixle.ppl surface
python production_example.py            # provenance, registry, serving, drift, checkpoints
python scaling_example.py               # the same fit distributed (local / mp / mpi / spark)
```

**Distributed backends:** `local` and `mp` run out of the box; `mpi` and Spark need a launcher.
Spark also needs a JVM (Java 17/21):

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PYSPARK_PYTHON=/path/to/venv/bin/python PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
```

**Compiled kernels** (optional; pure-numpy fallback always available):

```sh
python -c "from mixle.engines.build_kernels import compile_lns_kernel; compile_lns_kernel()"
# _lns_kernel.pyx: 14x LNS cross-entropy    _dd_kernels.pyx: 2.95x double-double dot
```

## Tests

```sh
python -m pytest                                       # fast gate (parallel), ~25 s
python -m pytest -m "not optional and not benchmark"   # full suite incl. slow tests
```

`base_dist_test.py` exercises each family end to end: sampler repeatability, `str`/`eval` round-trips,
vectorized-vs-scalar density agreement, EM convergence. See
[`mixle/tests/README.md`](https://github.com/gmboquet/mixle/blob/main/mixle/tests/README.md).

## Maintainers & contributors

Maintained by **Grant Boquet** ([@gmboquet](https://github.com/gmboquet) ·
grant.boquet@gmail.com).

mixle began life as **pysparkplug**, developed at Lawrence Livermore National Laboratory; thanks to the
LLNL contributors who built the original library and to everyone in the
[git history](https://github.com/gmboquet/mixle/graphs/contributors). Contributions, issues, and
discussion are welcome — open a PR or an issue.

## License

MIT — see [LICENSE](https://github.com/gmboquet/mixle/blob/main/LICENSE).

© 2014–2025, developed at Lawrence Livermore National Laboratory (LLNL-CODE-844837).
