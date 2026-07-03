<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/mixle_logo.png" alt="mixle" width="480"/>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2700%2B-brightgreen)

# mixle — automatic inference for composable models of heterogeneous data

Real datasets rarely fit one estimator class. A record might pair a category with a real measurement and a
variable-length count sequence; the interesting structure is often a latent cluster or regime over the whole
record. `mixle` models data like that directly: **the distribution is the unit of composition.** Scalar
families combine into tuples and records, records become mixture components, and mixtures become the
emissions of an HMM — all through one contract.

Because the estimator mirrors the model, fitting is a single call and the result is an inspectable
object, not a bespoke pipeline:

```python
from mixle.stats import *
from mixle.inference import optimize

# each record is (category, real value, variable-length count sequence), with a latent cluster
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100)
model.log_density(('a', 0.1, [5, 6]))     # score a record
model.sampler(seed=0).sample(3)           # draw new ones
```

The same `optimize` call fits everything — EM for latent models, maximum likelihood otherwise, closed-form
conjugate posteriors when a prior is supplied — and runs unchanged from a laptop to a Spark cluster by
switching a backend argument.

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Core concepts](#core-concepts) ·
[Distributions](#distributions) · [Frequentist & Bayesian](#frequentist--bayesian) ·
[Probabilistic programming](#probabilistic-programming) · [Scaling](#scaling-out) ·
[Enumeration & ranking](#enumeration--ranking) · [Ecosystem](#ecosystem) · [Examples](#examples) ·
[Tests](#tests) · [Maintainers](#maintainers) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). On PyPI as `mixle`.

```sh
pip install mixle          # base (numpy, scipy, mpmath): every distribution + local EM
pip install "mixle[all]"   # acceleration, scale-out, and connectors
```

The base install fits every distribution locally. Acceleration and scale-out are opt-in extras:

| Extra | Adds |
| --- | --- |
| `numba` | JIT-compiled hot paths (falls back to pure NumPy when absent) |
| `torch` | GPU / autograd engine |
| `spark` · `dask` · `mpi` | distributed estimation backends |
| `pandas` · `arrow` · `sql` · `mongo` · `hadoop` | data-source connectors |
| `sympy` · `sage` | symbolic / closed-form export |
| `umap` · `gmpy2` · `grammar` | model-based embeddings · big-integer ranking · graph-grammar models |

## Quickstart

The estimator mirrors the distribution's structure, so the shape you write is the shape you fit:

```python
from mixle.stats import *
from mixle.inference import optimize

data = [
    ('a', -0.4, [5, 7]),       ('b', 4.9, [11, 9]),
    ('a',  0.2, [6, 5, 4]),    ('b', 5.3, [10, 12, 11]),
    ('a', -1.1, [4, 6]),       ('b', 4.5, [9, 10]),
    ('a',  0.7, [5, 5]),       ('b', 5.1, [12, 8]),
]

est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100)
```

The fitted object is a distribution — score records one at a time or vectorized over a batch:

```python
model.log_density(('a', 0.1, [5, 6]))
model.seq_log_density(model.dist_to_encoder().seq_encode(data))
```

You need not spell the estimator out. `optimize` also accepts a **prototype distribution**, or infers a
first estimator straight from the data:

```python
from mixle.utils.automatic import get_estimator

model = optimize(reals, MixtureDistribution([GaussianDistribution(-1, 1),
                                             GaussianDistribution(1, 1)], [0.5, 0.5]))
model = optimize(data, get_estimator(data), max_its=100)   # estimator inferred from raw rows
```

## Core concepts

Every family is five cooperating pieces, so a fitted model exposes its likelihood, its sufficient
statistics, and its generative process rather than hiding them:

| Piece             | Role                                                                       |
| ----------------- | -------------------------------------------------------------------------- |
| `...Distribution` | parameters + `log_density(x)` and vectorized `seq_log_density(encoded)`     |
| `...Sampler`      | draw observations — `dist.sampler(seed).sample(size)`                       |
| `...Estimator`    | declares the model to fit; closed-form M-step in `estimate()`              |
| `...Accumulator`  | sufficient statistics for the E-step, mergeable across data partitions      |
| `...DataEncoder`  | packs raw Python records into arrays for the vectorized path                |

`optimize(data, est)` fits to convergence and dispatches the algorithm from the model's structure — EM for
latent models, maximum likelihood otherwise. Related entry points share the same contract:

- `best_of` — multi-restart EM
- `StreamingEstimator` — online EM across batches
- `fit_mle` / `fit_map` — autograd fitting with typed priors
- `mixle.utils.automatic.get_estimator(data)` — infer an estimator from raw data

Families live in `mixle.stats`; operations on a fitted model are grouped by concern — `mixle.inference`
(fitting), `mixle.enumeration` (rank / top-k), `mixle.ops` (quantize / condition / marginalize), and
`mixle.describe(x)` to report what any object supports.

## Distributions

`mixle.stats` provides a broad set of scalar, multivariate, structured, latent, and Bayesian families. The
distinguishing feature is that **combinators model a whole heterogeneous record as one distribution**:

| Model | One observation |
| --- | --- |
| `GaussianDistribution` / `PoissonDistribution` / `CategoricalDistribution` | `-0.31` / `7` / `'b'` |
| `MultivariateGaussianDistribution` | `[1.2, -0.4, 0.8]` |
| `CompositeDistribution((Cat, Gaussian, Poisson))` | `('a', -0.31, 7)` |
| `RecordDistribution({...})` | `{'country': 'US', 'age': 41, 'spend': 12.5}` |
| `SequenceDistribution(Poisson)` | `[5, 4, 6]` (variable length) |
| `OptionalDistribution(Gaussian)` | `-0.31` or `None` |
| `MixtureDistribution([...])` / `HiddenMarkovModelDistribution` | a component's shape, with the cluster / state latent |

- **Univariate:** Gaussian, Student-t/Cauchy, Logistic, LogGaussian, Laplace, Uniform, Exponential, Gamma,
  Inverse Gamma/Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh, Pareto, Poisson, Bernoulli,
  Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet, categorical; plus
  multivariate/diagonal Gaussian, von Mises–Fisher, and multivariate Student-t.
- **Combinators:** Composite (tuples), Record (named fields), Sequence, Optional (missing data), Transform,
  Conditional, Weighted.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA, PLSI,
  probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains, Indian
  buffet and Pitman-Yor processes.
- **Permutations & graphs:** Mallows / Plackett-Luce, matchings, spanning trees, random graphs
  (Erdős–Rényi, stochastic block, random dot-product), and graph grammars whose `log_density` is the exact
  marginal likelihood, computed by parsing a graph back to the start symbol.
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet) and variational
  Dirichlet-process / hierarchical-DP mixtures.

Every estimator shares the same knobs — `pseudo_count` (regularization), `prior=` (conjugate; `None` is
MLE), and `keys` (tie statistics across parts) — and one naming stem per family
(`<Stem>Distribution`, `<Stem>Estimator`, `<Stem>Sampler`, …).

## Frequentist & Bayesian

The prior is the only switch: no prior is maximum likelihood; a conjugate `prior=` makes the same machinery
Bayesian, with a closed-form posterior from the same `optimize` call.

```python
from mixle.inference.priors import NormalGammaPrior

GaussianEstimator()                          # MLE
GaussianEstimator(prior=NormalGammaPrior())  # closed-form conjugate posterior
```

`optimize` picks the objective from the model — likelihood, MAP, or a variational bound. When a model's
`log_density` is a bound rather than the exact `log p(x)` (as with LDA's per-document ELBO),
`supports(x, ExactDensity)` and `mixle.describe(x)` say so plainly.

## Probabilistic programming

`mixle.ppl` is a concise equation-style dialect over the same distributions. **One rule:** any parameter
slot is a value, the token `free` (estimate it), or another distribution (a prior).

```python
from mixle.ppl import Normal, Mix, Markov, Field, free

Normal(free, free).fit(data)                                  # estimate mean and standard deviation
Mix([Normal(free, free), Normal(free, free)]).fit(data)       # two-cluster Gaussian mixture
Markov(Normal(free, free), states=2).fit(seqs)                # two-state Gaussian HMM

# y[i] ~ Normal(b0 + b1*x[i] + b2*z[i], sd)
Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})
```

Slots can be expressions over named latents, and latents can be coupled, grouped for random effects, or
indexed by the data. `how=` selects the inference route (`conjugate | em | map | laplace | vi | mcmc |
nuts | …`); `m.explain_fit()` reports which route `auto` chose and why; multi-chain fits fold R̂ and ESS
into `m.result.summary()`, and `waic` / `loo` / `compare` rank fitted models. The dialect is thin — the
`mixle.stats` classes underneath are untouched.

## Scaling out

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array ops,
device, and precision. Scale-out is a backend argument, not a rewrite:

```python
from mixle.engines import TorchEngine

optimize(data, est, engine=TorchEngine(device="cuda", dtype="float32"))   # GPU
optimize(rdd,  est, backend="spark")                                      # also: mp · dask · mpi · ray
```

The same EM contract runs unchanged on NumPy, Numba, Torch, or a symbolic backend; new frameworks register
a factory rather than editing a dispatch table. A `SymbolicEngine` can emit a model's closed-form
log-density as LaTeX, SymPy, or Sage.

## Enumeration & ranking

Discrete and structured models enumerate their support in descending-probability order and answer exact
**rank / cumulative-probability** queries — even over enormous or unbounded supports:

```python
e = dist.enumerator()
e.top_k(5)        # the 5 most probable (value, log_prob)
e.top_p(0.95)     # smallest set covering 95% of the mass (the nucleus)
e.seek(10_000)    # the ~10,000th most probable value, by structural count-DP
```

For decomposable families (Composite / Record / Sequence / Markov chain), rank ↔ value is an exact
count-DP at any depth. For families where exact marginal rank is provably hard (mixtures, HMMs), the query
returns a Viterbi bound or a certified Monte-Carlo estimate with a standard error — never a silent
approximation.

## Ecosystem

The distribution contract is the spine; the surrounding namespaces reuse it for applied workflows:

- **`mixle.task`** — distill a frontier LLM, hosted endpoint, or slow rule into a small local model with
  conformal answer sets, density gates, and cascades.
- **`mixle.reason`** — LLM-answer uncertainty: semantic entropy, claim reliability, and cross-modal
  evidence fusion.
- **`mixle.doe`** — design of experiments: space-filling designs, Bayesian optimization, and sensitivity
  analysis.
- **`mixle.evolve`** — measure–propose–verify–promote loops with held-out gates and anti-regression
  ledgers.
- **`mixle.represent`** / **`mixle.models`** — shared vector representations across modalities, and neural
  likelihood leaves that drop into the same estimator tree.

Companion projects build on the core:

- **[mixle-notebooks](https://github.com/gmboquet/mixle-notebooks)** — tutorials, data-science recipes,
  applied case studies, and architecture/scaling studies as runnable notebooks.
- **[mixle-mlops](https://github.com/gmboquet/mixle-mlops)** — an OpenAI-compatible gateway that hosts
  fitted mixle models alongside open and hosted LLMs, with fine-tuning, registries, and serving.
- **[mixle-pde](https://github.com/gmboquet/mixle-pde)** — a differentiable PDE / physics stack
  (`Differential`, `make_ops`, `laplacian`, `NavierStokes2D`) for scientific inverse problems.

## Examples

Self-contained scripts in [examples/](https://github.com/gmboquet/mixle/tree/main/examples):

```sh
cd examples
python gallery_univariate_example.py    # tour the scalar families (also gallery_{multivariate,combinators,structured})
python ppl_example.py                   # the equation-style mixle.ppl surface
python scaling_example.py               # the same fit distributed by backend= (local / mp / mpi / spark)
python structure_learning_example.py    # dependency proposals before modeling
python production_example.py            # provenance, registry, serving, drift, checkpoints
```

**Distributed backends** (`scaling_example.py`): `local` and `mp` run out of the box; `mpi` and Spark need
a launcher, and Spark needs a JVM (Java 17/21) with workers on the driver's Python:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PYSPARK_PYTHON=/path/to/venv/bin/python PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
```

## Tests

```sh
python -m pytest                                       # fast gate (parallel), ~25 s
python -m pytest -m "not optional and not benchmark"   # full suite incl. slow tests
```

`base_dist_test.py` exercises each family end to end: sampler repeatability, `str`/`eval` round-trips,
vectorized-vs-scalar density agreement, and EM convergence.

## Maintainers

Maintained by **Grant Boquet** ([@gmboquet](https://github.com/gmboquet) · grant.boquet@gmail.com).

mixle began as **pysparkplug**, developed at Lawrence Livermore National Laboratory. Thanks to the LLNL
contributors who built the original library and to everyone in the
[git history](https://github.com/gmboquet/mixle/graphs/contributors). Contributions, issues, and discussion
are welcome.

## License

MIT — see [LICENSE](https://github.com/gmboquet/mixle/blob/main/LICENSE).

© 2014–2025, developed at Lawrence Livermore National Laboratory (LLNL-CODE-844837).
