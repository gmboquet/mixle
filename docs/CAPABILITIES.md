# What pysparkplug can do

The front door. pysparkplug is a probability/statistics library built on one idea: **dispatch on what
an object *supports*, not what class it is.** This page is the map — the families, the things you can
do to them, and how to ask.

For the *interface* design (the ABCs/Protocols behind all this) see [`ABSTRACTIONS.md`](ABSTRACTIONS.md).

---

## The one call that answers everything

```python
>>> import pysp
>>> pysp.describe(GaussianDistribution(0.0, 1.0))
GaussianDistribution — distribution.
  can:       score · sample · estimate · ConjugateUpdatable · ExponentialFamily · SupportsBackendScoring
  engines:   numpy, torch
  inference: closed-form conjugate Bayes, or numerical (MAP/Laplace/MCMC/VI)
  cannot:    Enumerable · RankableByIndex · Conditionable · Marginalizable · LatentStructured · ...

>>> pysp.describe(CategoricalDistribution({"a": .5, "b": .3, "c": .2}))
CategoricalDistribution — discrete distribution (finite support).
  can:       score · sample · estimate · ConjugateUpdatable · Enumerable · ExponentialFamily · RankableByIndex · ...

>>> pysp.describe(MixtureDistribution(...))
MixtureDistribution — latent-variable model.
  can:       score · sample · estimate · LatentStructured · PosteriorPredictive · ...
  inference: numerical (MAP/Laplace/MCMC/VI) — no closed-form conjugate prior
```

`pysp.describe(x)` works on any object. The query surface underneath it:

```python
pysp.supports(x, Cap)        # does x have capability Cap?
pysp.capabilities(x)         # the set of capabilities x has
pysp.require(x, Cap, "op")   # raise a clear error early if it doesn't
pysp.what_supports(Cap, xs)  # which of xs have Cap
pysp.catalog()               # the full capability vocabulary, as data
```

---

## The library at a glance

**Distributions** (`pysp.stats`) — score / sample / estimate over many supports:

| Support kind | Families |
|---|---|
| discrete finite | Categorical, Bernoulli, Binomial, IntegerCategorical |
| discrete countable | Poisson, Geometric, NegativeBinomial, Logseries, Skellam |
| continuous | Gaussian, Gamma, Beta, Exponential, StudentT, Weibull, EVT (GEV/GPD/Gumbel), … |
| vector / matrix / SPD | MultivariateGaussian, Dirichlet, MatrixNormal, Wishart, InverseWishart |
| directional | VonMises, VonMisesFisher, Watson |
| combinatorial | Mallows, PlackettLuce, ChowLiuTree, BernoulliSet, graphs (ER/SBM/RDPG/KG) |
| temporal point process | Hawkes, PowerLawHawkes, MultivariateHawkes, InhomogeneousPoisson, BirthDeath |

**Composition** — leaves combine into bigger models:
- **Combinators** (`pysp.stats.combinator`): Sequence, Composite, Conditional, Mixture, Truncated/Censored/Survival/Hurdle/ZeroInflated, Transform, ExponentialTilt — capabilities compose (a `Composite` of finite leaves is itself `FiniteSupport`/`RankableByIndex`).
- **Latent-state models** (`pysp.stats.latent`): finite latent (Mixture/LDA/PCA), sequential latent (HMM family — the hidden finite-state automaton), Markov chains (observed FSM), PCFG (grammar).
- **Bayesian / nonparametric** (`pysp.stats.bayes`): conjugate posteriors, Dirichlet/Pitman-Yor process mixtures.

**Subsystems** built on the same contract:
- **Enumeration** (`pysp.utils.enumeration`, `quantization`) — descending-probability iteration + count-budget unranking by index.
- **Relations** (`pysp.relations`) — optimisation-as-distribution over constrained combinatorial spaces.
- **PPL** (`pysp.ppl`) — `RandomVariable` + `fit(how=…)`: MAP / Laplace / MCMC / HMC / NUTS / VI / VMP / conjugate, plus GP/GMRF fields and the PDE-inverse stack.
- **DOE** (`pysp.doe`) — LHS/Sobol designs, Bayesian optimisation (EI/PI/UCB), D/A/I-optimal designs.
- **UQ** (`pysp.uq`) — Monte-Carlo / unscented propagation, Sobol/Morris sensitivity, calibration.
- **Engines** (`pysp.engines`) — the same computation on numpy / torch / symbolic.

---

## Inference: closed-form vs numerical

Every distribution can be fit by maximum likelihood (`dist.estimator()`). For *Bayesian* inference the
capability tells you which path you get:

- `supports(dist, ConjugateUpdatable)` → **closed-form** conjugate posterior (`conjugate_posterior(dist, data)`): exact samples, marginal likelihood, posterior predictive. (~20 families.)
- otherwise → **numerical** approximate inference via the PPL fitters: `fit(how="map"|"laplace"|"mcmc"|"hmc"|"nuts"|"vi"|"vmp")`. (The symbolic engine is a *representation* layer — it does not derive posteriors.)

This is one example of the recurring pattern — capabilities form **tiers** with graceful fallback:

```
ConjugateUpdatable → closed-form     |  RankableByIndex → Enumerable → sample
        ↓ else (numerical fit)       |  engine-resident → host
```

---

## Operations — the verbs (`pysp.ops`)

The third axis: what you can do *to* a distribution, and how each operation changes its capabilities.
Every verb has a **capability signature**.

```python
from pysp import ops, describe
q = ops.quantize(GaussianDistribution(0, 1), bits=8)   # continuous -> discrete
describe(q)   #  CategoricalDistribution — discrete (finite support). can: … Enumerable · RankableByIndex …
```

| Operation | input requires | output gains |
|---|---|---|
| `ops.quantize(dist, bits)` | any Distribution | FiniteSupport · Enumerable · RankableByIndex |
| `ops.truncate(dist, allowed=/forbidden=)` | Distribution | Distribution (Enumerable preserved) |
| `ops.condition(dist, observed)` | Conditionable | Distribution |
| `ops.marginalize(dist, keep)` | Marginalizable | Distribution |
| `ops.mixture(dists, w)` | Distributions | LatentStructured |
| `ops.transform(dist, f)` | Distribution + invertible f | Distribution (Jacobian-corrected) |
| `ops.tilt(dist, theta)` | ExponentialFamily | Distribution |

Operations are **capability-gated**: `ops.condition(gaussian, …)` raises a clear `CapabilityError`
because a Gaussian isn't `Conditionable`, while `ops.condition(mvn, …)` works.

## Enumeration is a concern, not a property (`pysp.enumeration`)

Enumeration is shared by distributions **and** relations **and** quantized objects — anything that can
walk its support in descending probability. `pysp.enumeration` is the one home for that concern: the
contract (`DistributionEnumerator`), the capability lens (`Enumerable`/`RankableByIndex`), the k-best
algorithms, and the count-budget unranking. A `Categorical` and an `Assignment` relation both satisfy
`pysp.supports(x, Enumerable)` through the same `enumerator()`. (This is the exemplar of the
concern-oriented layout proposed in [`ARCHITECTURE.md`](ARCHITECTURE.md).)

---

## The capability vocabulary

The full catalog (also `pysp.catalog()` at runtime — this table is rendered from it, so it never drifts):

| Capability | What it means | Kind | Backed by | Home |
|---|---|---|---|---|
| `Distribution` | score · sample · estimate | core contract | `log_density/sampler/estimator (ABC)` | `pysp.stats.compute.pdist` |
| `Sampler` | draw observations | core contract | `DistributionSampler.sample (ABC)` | `pysp.stats.compute.pdist` |
| `Estimator` | fit parameters from data (M-step) | core contract | `ParameterEstimator.estimate (ABC)` | `pysp.stats.compute.pdist` |
| `Enumerator` | k-best descending-probability iteration | core contract | `DistributionEnumerator (ABC)` | `pysp.stats.compute.pdist` |
| `Enumerable` | iterate the support in descending probability | distribution facet | `enumerator()` | `pysp.capability` |
| `FiniteSupport` | a finite number of support points | distribution facet | `support_size()` | `pysp.capability` |
| `RankableByIndex` | random access / unrank the support by integer rank | distribution facet | `count_budget_index()` | `pysp.capability` |
| `ExponentialFamily` | canonical exp-family form; generated numpy/torch kernels | distribution facet | `compute_declaration().exponential_family` | `pysp.capability` |
| `ConjugateUpdatable` | closed-form conjugate Bayesian posterior | distribution facet | `conjugate_posterior registry` | `pysp.capability` |
| `Conditionable` | condition on a subset of coordinates | distribution facet | `condition()` | `pysp.capability` |
| `Marginalizable` | marginalise to a subset of coordinates | distribution facet | `marginal()` | `pysp.capability` |
| `LatentStructured` | expose q(z\|x), the latent posterior + posterior-predictive | distribution facet | `latent_posterior()` | `pysp.capability` |
| `PosteriorPredictive` | sample/score new data from inferred latent state | distribution facet | `posterior_predictive()` | `pysp.capability` |
| `TemporalPointProcess` | conditional intensity λ(t) + compensator | distribution facet | `intensity()/expected_count()` | `pysp.capability` |
| `SetValued` | distribution over sets with forced membership | distribution facet | `required/num_required` | `pysp.capability` |
| `Neutral` | the identity / no-op element of a combinator | distribution facet | `isinstance Null*` | `pysp.capability` |
| `SupportsBackendScoring` | score a batch directly on the active engine | distribution facet | `backend_seq_log_density()` | `pysp.capability` |
| `EngineResidentEStep` | run the E-step on the engine without leaving it | object contract | `seq_update_engine()` | `pysp.capability` |
| `Transform` | invertible change of variables with a Jacobian | object contract | `forward/inverse/log_abs_det` | `pysp.capability` |
| `SupportsStackedBackend` | score a homogeneous component stack on the engine | object contract | `backend_stacked_*` | `pysp.capability` |
| `EncodedFold` | fold the E-step over distributed/streaming data | object contract | `pysp_seq_* methods` | `pysp.planner` |
| `EMStrategy` | an EM-step strategy | object contract | `step() -> EMStepResult` | `pysp.utils.em` |
| `Relation` | optimisation-as-distribution over a constrained space | subsystem role | `Relation (ABC): enumerate/solve/top/sample` | `pysp.relations` |
| `ComputeEngine` | a numpy/torch/symbolic backend (REQUIRED_OPS) | subsystem role | `ComputeEngine (ABC)` | `pysp.engines.base` |
| `DecomposableSemiring` | a semiring for structural count/enumeration DP | subsystem role | `DecomposableSemiring (ABC)` | `pysp.utils.quantization` |
| `DynamicsOperator` | a PDE/ODE evolution operator | subsystem role | `DynamicsOperator (ABC)` | `pysp.ppl.dynamics` |
| `ForwardOperator` | the PDE forward solve + adjoint namespace | subsystem role | `ForwardOperator/ForwardModel protocols` | `pysp.ppl._operator` |
| `Surrogate` | a fit/predict surrogate for Bayesian optimisation | subsystem role | `Surrogate protocol` | `pysp.doe._contracts` |
| `Acquisition` | a BO acquisition function (EI/PI/UCB) | subsystem role | `Acquisition protocol + register_acquisition` | `pysp.doe._contracts` |

Every capability is **structurally typed** — a third-party or app-layer distribution that implements
the method automatically gains the capability (and works with every capability-based algorithm) with
no subclassing or registration. That is the whole point: *what it supports, not what class it is.*
