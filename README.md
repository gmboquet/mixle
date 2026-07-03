<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/mixle_logo.png" alt="mixle" width="480"/>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2700%2B-brightgreen)

**Automatic inference for composable models of heterogeneous data.** Every model in mixle is a
*distribution* with the same five-piece contract, so a neural language model, a classical density, and a
latent structure (a mixture, an HMM) snap into one object you fit with a single call — and the inference
*follows from the structure you built*: conjugate, EM, MAP, variational, or MCMC, chosen for you. The same
fit runs locally on NumPy / Numba / GPU or scales out across Spark, Dask, Ray, or MPI by a `backend=` argument.

The unit of composition is the distribution: leaves (a Transformer LM, a Gaussian, a Poisson, …) combine
into tuples, tuples become mixture components, mixtures become HMM emissions, to any depth. A model and the
estimator that fits it have the same shape — so **what you can express, you can fit**.

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Core concepts](#core-concepts) ·
[Distribution catalog](#distribution-catalog) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Frequentist & Bayesian](#frequentist--bayesian) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Beyond fitting](#beyond-fitting) ·
[Companion projects](#companion-projects) · [Examples](#examples) · [Tests](#tests) ·
[Maintainers & contributors](#maintainers--contributors) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). On PyPI as `mixle`; the import name is `mixle`.

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
| `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `data` | data-source connectors |
| `gmpy2` | GMP-FFT big-integer multiply for count-DP ranking |
| `umap` | model-based UMAP embeddings |
| `sympy` · `sage` | symbolic / closed-form export |
| `grammar` | graph-grammar models (networkx) |

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

**Compose neural and classical models — and share parameters across them.** Here one learned word
embedding is tied across a plain language model and a topic mixture of language models, trained jointly in
a single fit, so every topic and the LM read and write the *same* word vectors:

```python
import numpy as np
from mixle.models import CategoricalEmbedding, TransformerLMEstimator
from mixle.stats import MixtureEstimator
from mixle.inference import optimize

V, d, B, K = 60, 24, 8, 3                        # vocab, embedding dim, context window, number of topics
emb = CategoricalEmbedding(V, d, name="word")    # ONE learned word embedding, declared once

# the SAME word vectors feed a plain language model AND a K-topic mixture of language models
lm     = TransformerLMEstimator(V, d_model=d, n_layer=2, block=B, embedding=emb)
topics = MixtureEstimator([TransformerLMEstimator(V, d_model=d, n_layer=2, block=B, embedding=emb)
                           for _ in range(K)])

# a document is (context window, next word); fitting the topic mixture trains the shared embedding jointly
rng   = np.random.RandomState(0)
docs  = [(list(rng.randint(0, V, size=B)), int(rng.randint(0, V))) for _ in range(240)]
model = optimize(docs, topics, max_its=5)

model.posterior(docs[0])                                    # soft topic assignment for a document
{id(lm.module.tok.weight)} | {id(c.module.tok.weight) for c in model.components}   # one shared tensor
```

The same one-`optimize` fit handles an ordinary heterogeneous record just as well — a web session,
`(device, minutes on site, [clicks per page])`: a category, a real, and a variable-length count sequence,
with a latent user segment over the whole record:

```python
from mixle.stats import *
from mixle.inference import optimize

# one record per web session: (device, minutes on site, [clicks on each page visited])
data = [
    ('ios', 2.3, [4, 1]),      ('web', 11.5, [9, 12, 7]),
    ('ios', 1.1, [2]),         ('web',  9.8, [8, 10]),
    ('ios', 3.0, [5, 3, 2]),   ('web', 12.1, [11, 9, 13]),
    ('ios', 0.8, [1, 2]),      ('web', 10.4, [7, 8]),
]

# the estimator mirrors the model's shape — two latent user segments over the whole record
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),                                             # device
    GaussianEstimator(),                                               # minutes on site
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),  # clicks per page
))] * 2)

model = optimize(data, est, max_its=100)
model.posterior(('web', 10.0, [9, 8]))   # soft segment assignment for a new session
model.sampler(seed=0).sample(3)          # synthesize brand-new sessions
```

You don't have to spell the estimator out. `optimize` (and `fit`) also accept a **prototype
distribution** — its matching estimator is taken automatically — or just the data, from which an
estimator is inferred:

```python
reals = [-2.1, -1.8, -2.0, 1.9, 2.3, 2.1]     # two clusters

proto = MixtureDistribution([GaussianDistribution(-1, 1), GaussianDistribution(1, 1)], [0.5, 0.5])
optimize(reals, proto)    # fit the shape you drew — no estimator to spell out
optimize(reals)           # or hand over just the data and let mixle infer the estimator
```

The same model in the shorter [`mixle.ppl`](#probabilistic-programming-mixleppl) dialect is a few lines.

**The whole lifecycle is one object.** `mixle.propose(data)` fits every proposer the library has on a
train split, ranks them on held-out data, and returns the winner — then the verbs chain:

```python
data = ...    # your records — any mix of types

m = mixle.propose(data, fit=True)   # fit every proposer on a split, rank on held-out, keep the winner
m.evaluate(...); m.sample(5); m.posterior(...); m.explain()
m.deploy("artifacts/m")             # durable artifact; mixle.Model.load() restores it
```

**Replace a function with a model.** `solve()` closes the loop: the code currently doing the job labels
the dataset, a small student trains, and the deployable answers locally only when a conformally
calibrated, in-distribution decision is safe — otherwise it calls the original code:

```python
from mixle.task import solve

route = ...     # the function doing the job today — a rule, an API call, an LLM
tickets = ...   # a list of representative inputs

sol = solve(route, tickets, propose="auto", synthesize=200)   # label with route(); train; conformally calibrate
sol(tickets[0])      # drop-in: answers locally when SURE, else falls back to route()
sol.improve()        # fold escalations back in; promote only if it verifies better
sol.save("artifacts/router")
```

The student defaults to a compact hashed-feature classifier; `solve(..., student="generative")` swaps in a
generative distribution instead — interpretable and torch-free.

## Core concepts

Each family is five cooperating pieces:

| Piece             | Role                                                                       |
| ----------------- | ------------------------------------------------------------------------- |
| `...Distribution` | parameters + `log_density(x)` and vectorized `seq_log_density(encoded)`    |
| `...Sampler`      | draw observations — `dist.sampler(seed).sample(size)`                      |
| `...Estimator`    | declares the model to fit; closed-form M-step in `estimate()`             |
| `...Accumulator`  | sufficient statistics for the E-step, mergeable across data partitions     |
| `...DataEncoder`  | packs raw Python records into arrays for the fast path                     |

`optimize(data, est)` (in `mixle.inference`) fits the model to convergence — EM for latent models
(mixtures, HMMs), maximum likelihood otherwise — vectorized locally, or distributed via `backend=`. It also accepts a distribution **prototype** (`optimize(data, proto)`) or
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

- **Univariate:** Gaussian, Student-t/Cauchy, Logistic, LogGaussian, Laplace, Uniform, Exponential,
  Gamma, Inverse Gamma/Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh, Pareto, Poisson,
  Bernoulli, Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet, categorical;
  multivariate/diagonal Gaussian, von Mises–Fisher, multivariate Student-t.
- **Combinators:** Composite (tuples), Record (named fields), Sequence, Optional (missing data),
  Transform, Conditional, Weighted.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains,
  hidden associations, IBP, Pitman-Yor processes, Bernoulli sets.
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

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Mix([Normal(a, 1), Normal(b, 1)]).fit(data, constraints=a < b)    # ordered means break label-switching

# y[i] ~ Normal(b0 + b1*x[i] + b2*z[i], sd) — Bayesian linear regression over columns you supply
Normal(free * Field("x") + free * Field("z") + free, free).fit(..., given={"x": ..., "z": ...})
```

- **`how=`** selects the route: `auto` reads the model's *structure* and picks the algorithm **family**
  — `conjugate | em | map | laplace | vi | vmp | mcmc | hmc | nuts | ensemble`.
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
from mixle.ppl import Normal, Poisson, Categorical, Field, Group, free, potential

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Normal(a + b, 1.0).fit(...)                        # deterministic expressions over latents
Normal(0.0, a.exp()).fit(...)                      #   …and transforms of them
Normal(a, 1.0).fit(..., potentials=potential(lambda av, bv: -0.5 * (av - bv) ** 2, a, b))  # custom log-factors

Normal(Normal(0, 5).each(), free).fit(...)                         # random effects: one list per group
Normal(Normal(0, 5).each(by="school"), free).fit(..., given={"school": ...})  #   …or a flat array + index
Poisson(free * Field("x") + Group("g")).fit(..., given={"x": ..., "g": ...})  # non-Normal GLMM (PQL)

theta = free(8)                                             # a latent vector, indexed by data
Normal(theta[Field("g")], free).fit(..., given={"g": ...})  #   y[i] ~ Normal(theta[g[i]], sd)

Categorical(free).fit(...)                                  # the category set is inferred from the data
```

- **Custom factors:** `potential(fn, *vars)` adds an arbitrary `fn(*values)` log-term to the joint, and
  may introduce auxiliary latents.
- **Hierarchies & GLMMs:** `.each()` / `.each(by=...)` are random effects; `Group(...)` is the same in a
  regression predictor, for a Normal, Poisson, or Bernoulli response.
- **Diagnostics:** a multi-chain fit (`how="nuts", chains=4`) folds per-parameter R̂ and ESS straight
  into `m.result.summary()`; `waic` / `loo` / `compare` rank fitted models.

When the density itself should be neural, the same dialect exposes flow / VAE / autoregressive
constructors that fit with `.fit()` and compose into mixtures like any distribution — no training loop in
user code:

```python
from mixle.ppl import Flow, MDN

Flow(2).fit(...)                      # p(x): a normalizing flow (also MAF, VAE, DiscreteAR)
MDN(1, 1).fit(..., given={"x": ...})  # p(y | x): a mixture density network (also CondFlow, CondDiscreteAR)
```

The dialect is thin — the `mixle.stats` classes underneath are untouched.

## Frequentist & Bayesian

The prior is the only switch — no prior is MLE; a conjugate `prior=` makes the same machinery Bayesian:

```python
from mixle.stats import GaussianEstimator
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

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
ops, device, and precision — so **scale-out is a backend argument, not a rewrite**:

```python
from mixle.engines import TorchEngine

optimize(..., engine=TorchEngine(device="cuda", dtype="float32"))   # GPU: the same fit, one extra argument
optimize(..., precision="auto")                                     # mixed precision; stats still accumulate in float64
optimize(..., backend="spark")                                      # distributed: mp · dask · mpi · ray · lightning
```

- The same EM contract runs unchanged on NumPy, Numba, Torch, or a symbolic backend.
- New frameworks register a factory (`register_encoded_data_backend`) — no dispatch to edit.
- The planner (`mixle.utils.parallel.planner`) turns a hardware budget into a memory-aware placement
  (chunking, device assignment, Torch sharding) you compute once and reuse.
- The `SymbolicEngine` runs a density through SymPy, so a model can emit its closed-form log-density
  as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded:

```python
from mixle.stats import CategoricalDistribution, SequenceDistribution, PoissonDistribution

# a toy language model: skewed letters in Poisson-length "words" — an unbounded support
lm = SequenceDistribution(CategoricalDistribution({"a": .5, "b": .3, "c": .2}),
                          len_dist=PoissonDistribution(3.0))

e = lm.enumerator()
e.top_k(5)                     # the 5 most probable words, in order — the rest are never touched
e.seek(1_000).value            # jump straight to the 1,000th-most-probable word, by structural count-DP
e.rank(["b", "a", "c"]).rank   # how many words are strictly more probable than "bac"
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
- **Neural & language leaves** (`mixle.models`): a causal-Transformer LM (`LM` / `StreamingTransformer`),
  neural experts (`NeuralGaussian`, `NeuralCategorical`), and preference-tuned (`DPOModel`) leaves — each a
  distribution that composes into mixtures / composites / HMM emissions and trains by EM (the E-step
  weights it; its M-step is gradient descent on the net). GPU/distributed pretraining via `LM.fit`.
- **Supervised & non-iid models** (`mixle.models`): GP regression, neural regressors, random forests
  (a conditional `p(y | x)` leaf), random graphs, grammars, knowledge graphs.
- **MLOps** (`mixle.inference.production`): reproducible model artifacts (`fit_with_provenance` → a
  `Header` with config, data hash, model-hash lineage, convergence, timing, resources, env), drift
  detection + a `Monitor` (retrain-and-swap), and a versioned `Registry` + `Service` (scoring +
  activity logging). A full OpenAI-compatible serving gateway lives in the separate
  [mixle-mlops](https://github.com/gmboquet/mixle-mlops) project.

## Companion projects

The core library stands alone; three sibling projects build on it:

- **[mixle-notebooks](https://github.com/gmboquet/mixle-notebooks)** — runnable tutorials, data-science
  recipes, applied case studies, and architecture/scaling studies.
- **[mixle-mlops](https://github.com/gmboquet/mixle-mlops)** — an OpenAI-compatible gateway that serves
  fitted mixle models alongside open and hosted LLMs, with fine-tuning, registries, and monitoring.
- **[mixle-pde](https://github.com/gmboquet/mixle-pde)** — a differentiable PDE / physics stack
  (`Differential`, `make_ops`, `laplacian`, `NavierStokes2D`) for scientific inverse problems.

## Examples

Self-contained scripts in [examples/](https://github.com/gmboquet/mixle/tree/main/examples)
— each samples from a known model, refits, and recovers it (no downloads):

```sh
cd examples
python gallery_univariate_example.py    # tour the scalar families (also gallery_{multivariate,combinators,…})
python gallery_structured_example.py    # mixtures / HMMs / LDA / latent-variable models
python ppl_example.py                   # the equation-style mixle.ppl surface
python production_example.py            # provenance, registry, serving, drift, checkpoints
python scaling_example.py               # the same fit distributed by backend= (local / mp / mpi / spark)
```

**Distributed backends** (see `scaling_example.py`): `local` and `mp` run out of the box; `mpi` and Spark
need a launcher. Spark also needs a JVM (Java 17/21) with workers on the driver's Python:

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
vectorized-vs-scalar density agreement, EM convergence. See
[`mixle/tests/README.md`](https://github.com/gmboquet/mixle/blob/main/mixle/tests/README.md).

## Maintainers & contributors

Maintained by **Grant Boquet** ([@gmboquet](https://github.com/gmboquet) ·
grant.boquet@gmail.com).

Contributions, issues, and discussion are welcome — open a PR or an issue.

## License

MIT — see [LICENSE](https://github.com/gmboquet/mixle/blob/main/LICENSE).

mixle began life as **pysparkplug**, developed at Lawrence Livermore National Laboratory 2014–2025 (LLNL-CODE-844837).
