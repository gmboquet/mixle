<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/mixle_logo.png" alt="mixle" width="480"/>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2700%2B-brightgreen)
[![docs](https://img.shields.io/badge/docs-gmboquet.github.io%2Fmixle-blue)](https://gmboquet.github.io/mixle/)

**Automatic inference for composable models of heterogeneous data.** Every model in mixle is a
*distribution* with the same five-piece contract, so a neural language model, a classical density, and a
latent structure (a mixture, an HMM) snap into one object you fit with a single call — and the inference
*follows from the structure you built*: conjugate, EM, MAP, variational, or MCMC, chosen for you. The same
fit runs locally on NumPy / Numba / GPU or scales out across Spark, Dask, Ray, or MPI by a `backend=` argument.

The unit of composition is the distribution: leaves (a Transformer LM, a Gaussian, a Poisson, …) combine
into tuples, tuples become mixture components, mixtures become HMM emissions, to any depth. A model and the
estimator that fits it have the same shape — so **what you can express, you can fit**.

📖 **Full documentation:** [gmboquet.github.io/mixle](https://gmboquet.github.io/mixle/) — guides, the
distribution catalog, and the API reference.

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Package highlights](#package-highlights) · [Companion projects](#companion-projects) · [Examples](#examples) ·
[Tests](#tests) · [Maintainers & contributors](#maintainers--contributors) · [License](#license)

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

**Both worlds, blended — frontier quality at a fraction of the cost.** Distill a slow, expensive teacher
(a frontier LLM, a human, a rule) into a tiny local model, then serve a *cascade*: a **neural** student
answers when a **classical** conformal gate says it is confident, and only the hard cases escalate to the
teacher.

```python
from mixle.task import distill, CalibratedTaskModel, Cascade, CostModel

def teacher(texts):
    ...   # a slow, expensive "frontier" model — an LLM, a human, a rule; ground truth, but $ per call

# text for the task (e.g. spam vs ham): `train` to distill on, `cal` to calibrate, `stream` to serve
train, cal, stream = ..., ..., ...

# distill the teacher into a tiny local model (a ~33K-parameter MLP over hashed n-grams, ~130 KB),
# calibrate WHEN to trust it (conformal), and serve a cascade — local when confident, escalate the rest
student = distill(teacher, train, n=4, dim=512, hidden=[64], epochs=250, task="spam vs ham")
gated   = CalibratedTaskModel(student, alpha=0.1).calibrate(cal, teacher(cal))
cascade = Cascade(gated, teacher, cost=CostModel(c_local=0.0, c_frontier=0.01))

cascade.serve(stream)   # frontier-quality answers, ~92% from the 33K-parameter local model
cascade.report()        # -> ~8% escalated; ~$2.76 saved / 300 requests vs frontier-only (on spam-vs-ham)
```

The tiny model handles the easy majority and defers the hard cases, so the blend matches the teacher while
running the large model on a fraction of requests. The same pattern distills tool-callers, extractors, and
structured classifiers (`mixle.task`).

**Compose arbitrarily deep — and tie parameters across the structure.** A segmental HMM whose every state
emits a *composite* segment (a two-mode mixture plus a phrase scored by a PCFG), with the mixture's first
mode **coupled across states by `keys=`**. One `optimize` call fits the whole tree by EM:

```python
from mixle.stats import *
from mixle.inference import optimize

# a PCFG over a 2-terminal "phrase": S -> A B | B A, with Gaussian terminals
def pcfg(a, b):
    return HeterogeneousPCFGDistribution(binary_rules={"S": [("A", "B", .5), ("B", "A", .5)]},
        terminal_rules={"A": [(GaussianDistribution(a, .5), 1.)], "B": [(GaussianDistribution(b, .5), 1.)]}, start="S")

# each state emits a SEGMENT = Composite(a 2-mode "tone" mixture, a PCFG phrase)
def emit(tone, a, b):
    return CompositeDistribution((MixtureDistribution(
        [GaussianDistribution(tone, .5), GaussianDistribution(tone + 3, .5)], [.6, .4]), pcfg(a, b)))

truth = SegmentalHiddenMarkovModelDistribution([emit(-2, -1, 1), emit(2, 3, 5)], [.5, .5],
    [[.8, .2], [.3, .7]], len_dist=CategoricalDistribution({3: 1.0}))
data = truth.sampler(0).sample(200)     # each obs: a length-3 sequence of (tone, [phrase]) segments

# fit by EM; keys="tone" ties the mixture's first mode across BOTH states (a shared parameter, one gradient)
def emest():
    return CompositeEstimator((MixtureEstimator([GaussianEstimator(keys="tone"), GaussianEstimator()]),
        HeterogeneousPCFGEstimator(binary_rules={"S": [("A", "B", .5), ("B", "A", .5)]},
            terminal_rules={"A": [(GaussianEstimator(), 1.)], "B": [(GaussianEstimator(), 1.)]}, start="S")))
fit = optimize(data, SegmentalHiddenMarkovEstimator(
    [emest(), emest()], len_estimator=CategoricalDistribution({3: 1.0}).estimator()), max_its=15)

fit.log_density(data[0])   # score a structured observation under the whole composed model
```

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

## Package Highlights

**The family contract.** Every family is five cooperating pieces — a `...Distribution` (`log_density` and
vectorized `seq_log_density`), a `...Sampler`, a `...Estimator` (closed-form M-step), a mergeable
`...Accumulator` (sufficient statistics), and a `...DataEncoder`. `optimize(data, est)` fits to convergence
— EM for latent models, maximum likelihood otherwise — and also accepts a distribution **prototype**
(`optimize(data, proto)`) or nothing but the data (`optimize(data)`, which infers the estimator). Related
entry points: `best_of` (multi-restart EM), `StreamingEstimator` (online EM), `fit_mle` / `fit_map`
(autograd fitting with typed priors), and `mixle.utils.automatic.get_estimator(data)`.

**~90 distributions, and combinators that model a whole heterogeneous record as one distribution.** Scalar
leaves (Gaussian, Student-t/Cauchy, Gamma, Beta, Weibull, Poisson, Bernoulli, Geometric, Binomial,
Categorical, von Mises, Dirichlet, …), plus multivariate/diagonal Gaussian, von Mises–Fisher, and
multivariate Student-t; combinators (Composite tuples, Record named fields, Sequence, Optional missing
data, Transform, Conditional, Weighted); latent structure (mixtures — plain / heterogeneous / hierarchical
/ joint / semi-supervised, LDA, PLSI, probabilistic PCA, HMMs — standard / segmental / lookback / tree /
quantized, PCFGs, Markov chains, IBP, Pitman-Yor); permutations and graphs (Mallows / Plackett-Luce,
matchings, spanning trees, random graphs, Spearman ranking, graph grammars whose `log_density` is the exact
marginal likelihood); and Bayesian families (conjugate priors + variational Dirichlet-process /
hierarchical-DP mixtures). Every estimator shares the same knobs — `pseudo_count`, `prior=`, `keys` — and
one naming stem per family (`<Stem>Distribution`, `<Stem>Estimator`, `<Stem>Sampler`).

**Frequentist or Bayesian — the prior is the only switch.** No prior is maximum likelihood; a conjugate
`prior=` makes the same `optimize` call return a closed-form posterior. `optimize` / `fit` pick the
objective from the model (likelihood, MAP, or a variational bound), and `supports(x, ExactDensity)` /
`mixle.describe(x)` flag when a `log_density` is a bound (e.g. LDA's per-document ELBO) rather than exact.

**Beyond fitting:**

- **Inference** (`mixle.inference`): MCMC (MH / HMC / NUTS / VMP), EM variants (hard, annealed, ECM,
  Monte-Carlo, variational, online, restart), Fisher / geometry views, and the `Posterior` algebra over
  latents / params / predictive; an engine-agnostic facade runs NUTS/ADVI on any differentiable target
  with parallel chains (R̂ + pooled ESS).
- **Design & analysis of experiments** (`mixle.doe`): space-filling designs, GP Bayesian optimization,
  Sobol/Morris sensitivity, uncertainty propagation, and Kennedy-O'Hagan calibration.
- **Task distillation** (`mixle.task`): distill teachers (LLMs, humans, rules) into small local models with
  conformal calibration, density gates, cascades, and routers; tool-caller / extractor / structured
  classifiers included.
- **Neural & language leaves** (`mixle.models`): a causal-Transformer LM (`LM` / `StreamingTransformer`),
  neural experts (`NeuralGaussian`, `NeuralCategorical`), and preference-tuned (`DPOModel`) leaves — each a
  distribution that composes into mixtures / composites / HMM emissions and trains by EM. Plus GP
  regression, neural regressors, random forests, random graphs, grammars, knowledge graphs.
- **Representations & embeddings** (`mixle.represent`, `mixle.utils.hvis`): one shared vector space across
  text / image / signal / structure with learned cross-modal tokenization; model-based t-SNE / UMAP over
  per-record posteriors.
- **MLOps** (`mixle.inference.production`): reproducible artifacts (`fit_with_provenance` — config, data
  hash, model-hash lineage, convergence, timing, env), drift detection + a `Monitor`, and a versioned
  `Registry` + scoring `Service`. A full OpenAI-compatible serving gateway lives in the separate
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

