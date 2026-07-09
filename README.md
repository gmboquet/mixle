<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gmboquet/mixle/main/assets/mixle_logo_dark.png"/>
    <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/assets/mixle_logo.png" alt="mixle" width="480"/>
  </picture>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2700%2B-brightgreen)
[![docs](https://img.shields.io/badge/docs-gmboquet.github.io%2Fmixle-blue)](https://gmboquet.github.io/mixle/)

**mixle is a Python library for specifying, training, deploying, and maintaining models of heterogeneous
data.** Behind a one-line API — hand it raw data and it selects and fits a model; hand it a structure and it
fits that — is a complete probabilistic-modeling stack: around 90 distributions, mixtures and hidden Markov
models, automatic model selection, Bayesian inference from EM to NUTS, design-of-experiments optimization,
and calibrated, monitored deployment. It stays efficient and easy to distribute: the same model runs on any
engine — NumPy, Numba, GPU — and scales across any backend — Spark, Dask, Ray, MPI — by changing a single
argument, with no rewrite. It is a serious statistics library at its core, with one idea that ties it
together: a classical distribution, a neural network, and a latent-variable model are the same kind of
object, so they compose freely and one `optimize(...)` call fits the whole thing.

Lab-grade AI, without the lab. Three things people reach for it for:

- **Less code.** No training loops, no batching or convergence boilerplate, no glue: point `optimize` at
  your data or your PyTorch module and it does the heavy lifting.
- **Lower cost.** Distill a slow, expensive model — a frontier LLM, an API, a rule — into a tiny local one
  that answers the easy cases itself and escalates only the hard ones, and reports what it saved.
- **Honest uncertainty.** It is calibrated to know when it is unsure and defer rather than guess, so it is
  safe to put in front of users.

mixle handles what you actually have — numbers, text, categories, mixed and missing values, directional and
angular data, rankings, graphs — all modeled the same way.

Fitting follows from the structure, not a flag: closed-form where a part has a closed form, gradient descent
where it is a neural network, EM where there are latent variables, all inside one loop. The deeper machinery
is there when you want it, but you rarely need it to get started.

**Full documentation:** [gmboquet.github.io/mixle](https://gmboquet.github.io/mixle/) — guides, the model
catalog, and the API reference.

Release-branch notes live in [CHANGELOG.md](CHANGELOG.md) and
[`docs/release-notes.rst`](docs/release-notes.rst).

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Package highlights](#package-highlights) · [Companion projects](#companion-projects) · [Examples](#examples) ·
[Tests](#tests) · [Maintainers & contributors](#maintainers--contributors) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). On PyPI as `mixle`; the import name is `mixle`. CI tests Linux
x86_64; macOS (incl. Apple Silicon) is the primary day-to-day dev platform and works in practice, but
isn't CI-gated. Windows is untested and unclaimed.

```sh
pip install mixle          # base (numpy, scipy, mpmath): every family, fit locally
pip install "mixle[all]"   # acceleration, scale-out, and connectors
```

The base install fits every distribution locally. Acceleration and scale-out are opt-in extras:

| Extra                                                    | Adds                                                          |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| `numba`                                                  | JIT-compiled hot paths (falls back to pure NumPy when absent) |
| `torch`                                                  | GPU / autograd engine                                         |
| `spark` · `dask` · `ray` · `lightning` · `mpi`           | distributed estimation backends                               |
| `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `data` · `arrays` | data-source connectors                              |
| `gmpy2`                                                  | GMP-FFT big-integer multiply for count-DP ranking             |
| `umap`                                                   | model-based UMAP embeddings                                   |
| `sympy` · `sage`                                         | symbolic / closed-form export                                 |
| `grammar`                                                | graph-grammar models (networkx)                               |

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

**Hand it data, get a model back.** No estimator, no configuration: mixle infers the model and fits it.

```python
from mixle.inference import optimize

records = [...]                  # your rows — any mix of numbers, text, categories, missing values
model = optimize(records)        # mixle works out the model and fits it

model.log_density(records[0])    # score an observation
model.sampler().sample(5)        # draw new ones
```

**Distill a slow, expensive model into a cheap one that knows when to defer.** Point `solve` at the
function doing the job today — an LLM, an API, a rule — and it trains a small local model that answers the
easy cases itself and escalates only the hard ones.

```python
from mixle.task import solve

# teacher = the function doing the job now; inputs = representative examples
assistant = solve(teacher, inputs)   # `teacher` labels once; a small local model learns from it

assistant(x)            # answers locally when confident, calls `teacher` only when it is not
assistant.report()      # how often it matched the teacher, and how much it deferred
assistant.save("assistant/")
```

You pay the expensive model on only a fraction of requests. The same pattern distills classifiers,
extractors, and tool-callers, with conformal calibration, cascades, and a cost model underneath
(`mixle.task`).

**A PyTorch module fits in one line — the training code you did not write.** Any module exposing
`log_density(x)` fits with one call: no loop, no batching, no eval or convergence boilerplate.

```python
from mixle.inference import optimize

model = optimize(x, my_module)   # your nn.Module — trained
model.module                     # the raw module back, nothing trapped
```

Freeze submodules, swap the optimizer, or distribute the fit with `backend=`; parity with a hand-written
training loop is checked by a test, not claimed here.

**Compose to any depth; one call fits the whole thing.** You hand it estimators, not fitted
distributions — you don't know the parameters yet, and that's the point. Real observations arrive in
several channels at once; a composite models each channel with its own family — here a two-cluster
Gaussian mixture on one, the neural density from above on the other — and a single `optimize` learns
them together.

```python
from mixle.stats import GaussianEstimator, MixtureEstimator, CompositeEstimator
from mixle.models import GradEstimator

# each record is a pair (x0, x1); nothing below fixes a parameter
model = optimize(pairs, CompositeEstimator([
    MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),  # channel 0: a two-cluster mixture
    GradEstimator(my_module),                                      # channel 1: a neural density
]))
```

One call, each part fit by the right M-step: EM for the mixture, gradient descent for the neural leaf.
Every node is an estimator, so this whole tree is itself just a node — drop it into a mixture, or make it
the emission of a hidden Markov model, and the call at the top never changes.

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
ops, device, and precision — so **scale-out is a backend argument, not a rewrite**:

```python
from mixle.engines import TorchEngine

optimize(..., engine=TorchEngine(device="cuda", dtype="float32"))  # GPU: one arg
optimize(..., precision="auto")   # mixed precision; stats accumulate in float64
optimize(..., backend="spark")    # distributed: mp · dask · mpi · ray · lightning
```

- The same fit runs unchanged on NumPy, Numba, Torch, or a symbolic backend.
- New frameworks register a factory (`register_encoded_data_backend`) — no dispatch to edit.
- The planner (`mixle.utils.parallel.planner`) turns a hardware budget into a memory-aware placement
  (chunking, device assignment, Torch sharding) you compute once and reuse.
- The `SymbolicEngine` runs a density through SymPy, so a model can emit its closed-form log-density
  as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded.
This works on a real neural LM and on a model you just fit:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mixle.enumeration import AutoregressiveEnumerable

name      = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(name)
llm       = AutoModelForCausalLM.from_pretrained(name).eval()
prompt    = tokenizer("The capital of France is", return_tensors="pt").input_ids

@torch.no_grad()
def next_logprobs(continuation):   # tokens so far -> [(token_id, log_prob), ...]
    ids = (torch.cat([prompt, torch.tensor([continuation], dtype=torch.long)], 1)
           if continuation else prompt)
    return list(enumerate(torch.log_softmax(llm(ids).logits[0, -1], -1).tolist()))

# branch_cap tames the 49K-token vocab
continuations = AutoregressiveEnumerable(next_logprobs, max_len=3, branch_cap=8)

continuations.top_k(3)      # -> [' located in the', ' the city of', ' the capital of']
continuations.unrank(100)   # 100th-most-probable, no generation -> ' in the country'

answer = continuations.unrank(5)[0]   # the ' Paris, the' continuation
continuations.rank(answer)  # inverse -> rank=6, cumulative_prob=0.114 (exact)
```

The same operations work on a fitted latent model. Here an HMM learns *when to stop* from an absorbing
terminal state, and its EOL-terminated support is enumerated in descending probability:

```python
from mixle.inference import optimize
from mixle.stats import HiddenMarkovEstimator, CategoricalEstimator

# your sequences, each ending in an EOL token
sequences = [["team", "meet", "buy", "<EOL>"],
             ["now", "now", "<EOL>"],
             ["meet", "meet", "<EOL>"],
             ...]

# fit a 3-state HMM by EM; state 2 is terminal, so the model learns WHEN to
# stop — its emission converges to "<EOL>" and the length becomes a learned
# stopping time (no separate len_dist)
model = optimize(sequences,
    HiddenMarkovEstimator([CategoricalEstimator()] * 3, terminal_states={2}))

emitted = model.enumerator()
# most probable EOL-terminated sequences:
emitted.top_k(3)          # -> [('buy <EOL>', -2.09), ('meet <EOL>', -2.12), ...]
emitted.from_index(3, 6)  # stream ranks 3..5 without materializing 0..2
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

data = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1]   # reals from two clusters
seqs = [[0.1, 5.1, 4.9], [4.8, 5.0], [0.0, 0.2]]   # variable-length sequences
Normal(free, free).fit(data)               # estimate mean + standard deviation
Normal(Normal(0, 10), 1.0).fit(data)       # a prior on the mean (hierarchical)
Mix([Normal(free, free), Normal(free, free)]).fit(data)   # two-cluster mixture
Markov(Normal(free, free), states=2).fit(seqs)   # a 2-state Gaussian HMM

# a slot can be an expression over named latents or data columns:
Normal(free * Field("x") + free * Field("z") + free, free).fit(
    ..., given={"x": ..., "z": ...})   # a regression
```

- **`how=`** picks the inference route from the model's structure (`conjugate | em | map | laplace | vi |
  mcmc | nuts | …`); `m.explain_fit()` reports the choice and why.
- **Hierarchies & GLMMs:** `.each(by=...)` and `Group(...)` are random effects; `potential(fn, *vars)` adds
  a custom log-factor; constraints (`a < b`) shape inference and sampling.
- **Neural densities:** `Flow`, `MDN`, `VAE` fit with `.fit()` and compose into mixtures like any distribution.
- **Diagnostics:** multi-chain fits fold R̂ / ESS into `m.result.summary()`; `waic` / `loo` / `compare` rank
  fitted models.

## Package highlights

- **Just pass data** — `optimize(data)` and `mixle.propose(data)` work out a model for you: they pick a
  family per field, notice when fields depend on each other, and (new in 0.6.3) fit a copula or vine when
  continuous columns are correlated — heavy joint tails included.
- **A torch module, fit for you** (`mixle.models`) — any module with a forward pass and an objective fits
  in one call (`optimize(x, module)`); freeze submodules, swap the optimizer, and get the raw module back —
  nothing is trapped. Parity with a hand-written loop is pinned by a test.
- **Distill big models into small ones, and route by cost** (`mixle.task`) — turn a slow, expensive teacher
  (an LLM, a rule, a human) into a tiny local model; a **conformal gate** decides when the local answer is
  safe, a **cascade / router** escalates only the hard cases to the teacher, and a **cost model** reports
  the dollars saved and the break-even volume so the trade-off is a number, not a guess. Soft-label
  distillation, density-gated and cost-aware routing thresholds, harvest-and-re-distill loops, and multi-tier
  routing with bandits / RL for the decision policy are all included.
- **Build by nesting** — mixtures, sequences, records, HMMs (segmental / lookback / tree / quantized), LDA,
  PCFGs, and more compose to any depth, with parameters tied across the structure by `keys=`; one call fits
  the whole tree.
- **A large catalog of building blocks** — around 90 families (continuous, discrete, directional,
  multivariate), copulas and vines for dependence, permutations and graphs (Mallows, matchings, spanning
  trees, grammars), and neural leaves: a Transformer LM, energy models, and constrained networks
  (physics-informed, monotonic, input-convex, conservation-preserving, permutation-invariant).
- **Scale with a flag** — the same fit runs local (NumPy / Numba / GPU) or distributed
  (Spark / Dask / Ray / MPI) via `backend=`; the Transformer LM trains sharded (FSDP2) under torchrun.
- **When you want the full picture** — exact enumeration and rank / quantile queries over discrete and
  structured models, MCMC and variational inference for posteriors (a `prior=` is the only switch from
  point estimate to posterior), design of experiments with Bayesian optimization, cross-modal
  representations, and reproducible artifacts with a serving gateway
  ([mixle-mlops](https://github.com/gmboquet/mixle-mlops)).

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
python gallery_univariate_example.py   # scalar families (+ multivariate, …)
python gallery_structured_example.py   # mixtures / HMMs / LDA / latent models
python ppl_example.py                  # the equation-style mixle.ppl surface
python production_example.py           # provenance, registry, serving, drift
python scaling_example.py              # same fit by backend= (mp / mpi / spark)
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
